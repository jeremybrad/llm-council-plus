"""Claims API router.

Thin API layer that delegates to backend/claims.py for business logic.
Routes validate requests, call CRUD functions, and return serialized responses.

Endpoints:
- GET /api/claims - List claims with filters and pagination
- GET /api/claims/export - Export claims bundle (backup/migration)
- POST /api/claims/import - Import claims from export bundle
- GET /api/claims/{claim_id} - Get single claim
- POST /api/claims - Create candidate claim
- PATCH /api/claims/{claim_id} - Update status/temporal fields
- POST /api/claims/{claim_id}/evidence - Add evidence
- POST /api/claims/{claim_id}/validate - Single-claim SADB validation
- POST /api/claims/{claim_id}/adjudicate - Panel adjudication
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from ..claims import (
    add_claim,
    get_claim,
    query_claims,
    update_claim,
    add_evidence,
    get_all_claims,
    import_claim,
    Evidence as EvidenceDataclass,
    Claim as ClaimDataclass,
    StatusTransitionError,
    TRANSITION_RULES,
)
from ..adjudicator import adjudicate_claim, AdjudicationResult
from ..evidence import search_evidence, get_sadb_status
from ..scorer import classify_support


router = APIRouter(prefix="/api/claims", tags=["claims"])


# =============================================================================
# Pydantic Request Models
# =============================================================================


class CreateClaimRequest(BaseModel):
    """Request body for creating a new claim."""

    claim_text: str = Field(..., min_length=1, max_length=2000)
    claim_type: str = Field(..., pattern="^(biographical|preference|belief|event|relationship|project)$")
    as_of: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class UpdateClaimRequest(BaseModel):
    """Request body for updating a claim."""

    status: Optional[str] = Field(None, pattern="^(candidate|accepted|disputed|deprecated)$")
    as_of: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    note: Optional[str] = Field(None, max_length=500, description="Reason for update, stored in review_history")
    force: bool = Field(False, description="Force status transition, bypassing validation rules (admin override)")


class AddEvidenceRequest(BaseModel):
    """Request body for adding evidence to a claim."""

    source_type: str = Field(..., pattern="^(repo_doc|transcript|note|chat_summary|j1_fact)$")
    source_id: str = Field(..., min_length=1, max_length=500)
    quote: str = Field(..., min_length=1, max_length=2000)
    support: str = Field(..., pattern="^(supports|contradicts|neutral)$")
    weight: float = Field(..., ge=0.0, le=1.0)
    retrieval_query: str = Field(default="manual", max_length=500)
    retrieved_at: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    source_hash: Optional[str] = None


class AdjudicateRequest(BaseModel):
    """Request body for panel adjudication."""

    models: Optional[List[str]] = Field(
        None,
        description="Model IDs for panel (e.g., ['ollama:llama3.2', 'ollama:mistral']). Uses council_models if not provided.",
    )
    panel_size: int = Field(
        default=3,
        ge=1,
        le=7,
        description="Number of panel members (1-7)",
    )
    mode: str = Field(
        default="strict",
        pattern="^(strict|lenient)$",
        description="strict requires higher consensus for status update",
    )
    update_status: bool = Field(
        default=False,
        description="If true, update claim status based on strong consensus",
    )


# =============================================================================
# Pydantic Response Models
# =============================================================================


class ScoreBreakdownResponse(BaseModel):
    """Score breakdown for explainability."""

    base_score: float
    supporting_bonus: float
    independence_bonus: float
    contradiction_penalty: float
    final_score: float
    evidence_ids_used: List[str]
    calculation_timestamp: str
    cap_applied: float
    cap_reason: str


class EvidenceResponse(BaseModel):
    """Evidence response with optional quote truncation."""

    evidence_id: str
    source_type: str
    source_id: str
    quote: str
    support: str
    weight: float
    retrieved_at: str
    retrieval_query: str
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    source_hash: Optional[str] = None

    @classmethod
    def from_dataclass(
        cls,
        evidence: EvidenceDataclass,
        quote_max_len: int = 280,
    ) -> "EvidenceResponse":
        """Create from Evidence dataclass with optional quote truncation."""
        quote = evidence.quote
        if len(quote) > quote_max_len:
            quote = quote[:quote_max_len] + "..."

        return cls(
            evidence_id=evidence.evidence_id,
            source_type=evidence.source_type,
            source_id=evidence.source_id,
            quote=quote,
            support=evidence.support,
            weight=evidence.weight,
            retrieved_at=evidence.retrieved_at,
            retrieval_query=evidence.retrieval_query,
            span_start=evidence.span_start,
            span_end=evidence.span_end,
            source_hash=evidence.source_hash,
        )


class ClaimResponse(BaseModel):
    """Claim response with optional evidence and history."""

    claim_id: str
    claim_text: str
    claim_type: str
    confidence: float
    status: str
    evidence_count: int
    as_of: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    score_breakdown: Optional[ScoreBreakdownResponse] = None
    created_at: str
    last_reviewed_at: Optional[str] = None

    # Optional fields (controlled by include_* params)
    evidence: Optional[List[EvidenceResponse]] = None
    review_history: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dataclass(
        cls,
        claim: ClaimDataclass,
        include_evidence: bool = False,
        include_history: bool = False,
        quote_max_len: int = 280,
    ) -> "ClaimResponse":
        """Create from Claim dataclass with optional includes."""
        # Build score breakdown if present
        score_breakdown = None
        if claim.score_breakdown:
            score_breakdown = ScoreBreakdownResponse(
                base_score=claim.score_breakdown.base_score,
                supporting_bonus=claim.score_breakdown.supporting_bonus,
                independence_bonus=claim.score_breakdown.independence_bonus,
                contradiction_penalty=claim.score_breakdown.contradiction_penalty,
                final_score=claim.score_breakdown.final_score,
                evidence_ids_used=claim.score_breakdown.evidence_ids_used,
                calculation_timestamp=claim.score_breakdown.calculation_timestamp,
                cap_applied=claim.score_breakdown.cap_applied,
                cap_reason=claim.score_breakdown.cap_reason,
            )

        # Build evidence list if requested
        evidence = None
        if include_evidence:
            evidence = [
                EvidenceResponse.from_dataclass(e, quote_max_len=quote_max_len)
                for e in claim.evidence
            ]

        # Build history if requested
        review_history = None
        if include_history:
            review_history = claim.review_history

        return cls(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            claim_type=claim.claim_type,
            confidence=claim.confidence,
            status=claim.status,
            evidence_count=len(claim.evidence),
            as_of=claim.as_of,
            valid_from=claim.valid_from,
            valid_until=claim.valid_until,
            score_breakdown=score_breakdown,
            created_at=claim.created_at,
            last_reviewed_at=claim.last_reviewed_at,
            evidence=evidence,
            review_history=review_history,
        )


class ClaimsListResponse(BaseModel):
    """Paginated list of claims."""

    items: List[ClaimResponse]
    total: int
    limit: int
    offset: int


class ValidateResponse(BaseModel):
    """Response from single-claim validation."""

    claim_id: str
    sadb_available: bool
    evidence_added: int
    old_confidence: float
    new_confidence: float
    claim: Optional[ClaimResponse] = None


class PanelVerdictResponse(BaseModel):
    """Single panel member's verdict."""

    model: str
    verdict: str
    confidence: float
    reasoning: str
    cited_evidence: List[str]
    concerns: List[str]
    error: Optional[str] = None


class AdjudicateResponse(BaseModel):
    """Response from panel adjudication."""

    claim_id: str
    panel_size: int
    mode: str
    panel_verdicts: List[PanelVerdictResponse]
    consensus_verdict: str
    consensus_confidence: float
    timestamp: str
    status_updated: bool
    new_status: Optional[str] = None
    claim: Optional[ClaimResponse] = None


class ExportMetadata(BaseModel):
    """Metadata for export bundle."""

    exported_at: str
    format: str
    total_claims: int
    total_evidence: int
    filters_applied: Dict[str, Any]
    transition_rules: Dict[str, Any]


class ExportClaimData(BaseModel):
    """Full claim data for export (no truncation)."""

    claim_id: str
    claim_text: str
    claim_type: str
    confidence: float
    status: str
    as_of: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    created_at: str
    last_reviewed_at: Optional[str] = None
    score_breakdown: Optional[ScoreBreakdownResponse] = None
    evidence: List[EvidenceResponse]
    review_history: List[Dict[str, Any]]

    @classmethod
    def from_dataclass(cls, claim: ClaimDataclass) -> "ExportClaimData":
        """Create from Claim dataclass with full data (no truncation)."""
        score_breakdown = None
        if claim.score_breakdown:
            score_breakdown = ScoreBreakdownResponse(
                base_score=claim.score_breakdown.base_score,
                supporting_bonus=claim.score_breakdown.supporting_bonus,
                independence_bonus=claim.score_breakdown.independence_bonus,
                contradiction_penalty=claim.score_breakdown.contradiction_penalty,
                final_score=claim.score_breakdown.final_score,
                evidence_ids_used=claim.score_breakdown.evidence_ids_used,
                calculation_timestamp=claim.score_breakdown.calculation_timestamp,
                cap_applied=claim.score_breakdown.cap_applied,
                cap_reason=claim.score_breakdown.cap_reason,
            )

        # Full quotes, no truncation
        evidence = [
            EvidenceResponse.from_dataclass(e, quote_max_len=10000)
            for e in claim.evidence
        ]

        return cls(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            claim_type=claim.claim_type,
            confidence=claim.confidence,
            status=claim.status,
            as_of=claim.as_of,
            valid_from=claim.valid_from,
            valid_until=claim.valid_until,
            created_at=claim.created_at,
            last_reviewed_at=claim.last_reviewed_at,
            score_breakdown=score_breakdown,
            evidence=evidence,
            review_history=claim.review_history,
        )


class ExportBundleResponse(BaseModel):
    """Complete export bundle with metadata and claims."""

    metadata: ExportMetadata
    claims: List[ExportClaimData]


class ImportClaimData(BaseModel):
    """Claim data for import (matches export format)."""

    claim_id: str
    claim_text: str
    claim_type: str = Field(..., pattern="^(biographical|preference|belief|event|relationship|project)$")
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    status: str = Field(default="candidate", pattern="^(candidate|accepted|disputed|deprecated)$")
    as_of: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    created_at: Optional[str] = None
    last_reviewed_at: Optional[str] = None
    score_breakdown: Optional[Dict[str, Any]] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    review_history: List[Dict[str, Any]] = Field(default_factory=list)


class ImportBundleRequest(BaseModel):
    """Request body for importing claims bundle."""

    claims: List[ImportClaimData]
    on_duplicate: str = Field(
        default="skip",
        pattern="^(skip|overwrite|error)$",
        description="How to handle existing claim_ids: skip, overwrite, or error",
    )


class ImportResultItem(BaseModel):
    """Result for a single imported claim."""

    claim_id: str
    status: str  # "created", "skipped", "overwritten", "error"
    error: Optional[str] = None


class ImportBundleResponse(BaseModel):
    """Response from import operation."""

    imported_at: str
    total_in_bundle: int
    created: int
    skipped: int
    overwritten: int
    errors: int
    results: List[ImportResultItem]


# =============================================================================
# Endpoints
# =============================================================================


@router.get("", response_model=ClaimsListResponse)
async def list_claims(
    status: Optional[str] = Query(None, pattern="^(candidate|accepted|disputed|deprecated)$"),
    claim_type: Optional[str] = Query(None, pattern="^(biographical|preference|belief|event|relationship|project)$"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    valid_at: Optional[str] = Query(None, description="ISO timestamp for temporal filtering"),
    q: Optional[str] = Query(None, max_length=200, description="Substring search on claim_text"),
    sort: Optional[str] = Query(
        None,
        pattern="^(created_at|last_reviewed_at|confidence|claim_text|status|evidence_count)$",
        description="Field to sort by",
    ),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_evidence: bool = Query(False),
    include_history: bool = Query(False),
    quote_max_len: int = Query(280, ge=50, le=2000),
):
    """List claims with optional filters, sorting, and pagination.

    Query parameters:
    - status: Filter by claim status
    - claim_type: Filter by claim type
    - min_confidence: Filter by minimum confidence score
    - valid_at: Filter by temporal validity at given ISO timestamp
    - q: Substring search on claim_text
    - sort: Field to sort by (created_at, last_reviewed_at, confidence, claim_text, status, evidence_count)
    - order: Sort order (asc, desc) - default desc
    - limit: Max results (default 50, max 200)
    - offset: Skip first N results
    - include_evidence: Include evidence array (default false)
    - include_history: Include review_history array (default false)
    - quote_max_len: Truncate quotes in evidence (default 280)
    """
    # Query claims using store function
    claims = query_claims(
        status=status,
        claim_type=claim_type,
        min_confidence=min_confidence,
        valid_at=valid_at,
    )

    # Apply substring search if provided
    if q:
        q_lower = q.lower()
        claims = [c for c in claims if q_lower in c.claim_text.lower()]

    # Apply sorting if requested
    if sort:
        reverse = order == "desc"

        def get_sort_key(claim):
            if sort == "evidence_count":
                return len(claim.evidence)
            value = getattr(claim, sort, None)
            # Handle None values - sort them last
            if value is None:
                return "" if isinstance(value, str) else 0
            return value

        claims = sorted(claims, key=get_sort_key, reverse=reverse)

    # Get total before pagination
    total = len(claims)

    # Apply pagination
    claims = claims[offset : offset + limit]

    # Convert to response models
    items = [
        ClaimResponse.from_dataclass(
            c,
            include_evidence=include_evidence,
            include_history=include_history,
            quote_max_len=quote_max_len,
        )
        for c in claims
    ]

    return ClaimsListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/export", response_model=ExportBundleResponse)
async def export_claims(
    status: Optional[str] = Query(None, pattern="^(candidate|accepted|disputed|deprecated)$"),
    claim_type: Optional[str] = Query(None, pattern="^(biographical|preference|belief|event|relationship|project)$"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    valid_at: Optional[str] = Query(None, description="ISO timestamp for temporal filtering"),
    q: Optional[str] = Query(None, max_length=200, description="Substring search on claim_text"),
):
    """Export claims bundle with full evidence and history.

    Returns a complete export bundle suitable for backup, migration, or analysis.
    Unlike the list endpoint, this includes:
    - Full quotes (no truncation)
    - All evidence for each claim
    - Complete review history
    - Export metadata with filters applied

    Query parameters (all optional):
    - status: Filter by claim status
    - claim_type: Filter by claim type
    - min_confidence: Filter by minimum confidence score
    - valid_at: Filter by temporal validity
    - q: Substring search on claim_text
    """
    # Query claims using store function
    claims = query_claims(
        status=status,
        claim_type=claim_type,
        min_confidence=min_confidence,
        valid_at=valid_at,
    )

    # Apply substring search if provided
    if q:
        q_lower = q.lower()
        claims = [c for c in claims if q_lower in c.claim_text.lower()]

    # Convert to export format (full data, no truncation)
    export_claims = [ExportClaimData.from_dataclass(c) for c in claims]

    # Count total evidence
    total_evidence = sum(len(c.evidence) for c in export_claims)

    # Build filters applied dict
    filters_applied = {}
    if status:
        filters_applied["status"] = status
    if claim_type:
        filters_applied["claim_type"] = claim_type
    if min_confidence is not None:
        filters_applied["min_confidence"] = min_confidence
    if valid_at:
        filters_applied["valid_at"] = valid_at
    if q:
        filters_applied["q"] = q

    metadata = ExportMetadata(
        exported_at=datetime.now(timezone.utc).isoformat(),
        format="json",
        total_claims=len(export_claims),
        total_evidence=total_evidence,
        filters_applied=filters_applied,
        transition_rules=TRANSITION_RULES,
    )

    return ExportBundleResponse(
        metadata=metadata,
        claims=export_claims,
    )


@router.post("/import", response_model=ImportBundleResponse)
async def import_claims(request: ImportBundleRequest):
    """Import claims from an export bundle.

    Restores claims from a previously exported bundle. Each claim preserves:
    - Original claim_id
    - All evidence with IDs
    - Complete review history (with import event appended)
    - Score breakdown
    - Temporal validity fields

    The `on_duplicate` parameter controls behavior when a claim_id already exists:
    - `skip` (default): Keep existing claim, skip import
    - `overwrite`: Replace existing claim with imported data
    - `error`: Return error for that claim (other claims still processed)

    Returns a summary of the import operation with per-claim results.
    """
    results = []
    created = 0
    skipped = 0
    overwritten = 0
    errors = 0

    for claim_data in request.claims:
        try:
            # Convert Pydantic model to dict
            claim_dict = claim_data.model_dump()

            claim, status = import_claim(claim_dict, on_duplicate=request.on_duplicate)

            results.append(ImportResultItem(
                claim_id=claim_data.claim_id,
                status=status,
                error=None,
            ))

            if status == "created":
                created += 1
            elif status == "skipped":
                skipped += 1
            elif status == "overwritten":
                overwritten += 1

        except ValueError as e:
            # on_duplicate="error" case
            results.append(ImportResultItem(
                claim_id=claim_data.claim_id,
                status="error",
                error=str(e),
            ))
            errors += 1

        except Exception as e:
            results.append(ImportResultItem(
                claim_id=claim_data.claim_id,
                status="error",
                error=f"Unexpected error: {str(e)}",
            ))
            errors += 1

    return ImportBundleResponse(
        imported_at=datetime.now(timezone.utc).isoformat(),
        total_in_bundle=len(request.claims),
        created=created,
        skipped=skipped,
        overwritten=overwritten,
        errors=errors,
        results=results,
    )


@router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim_by_id(
    claim_id: str,
    include_evidence: bool = Query(False),
    include_history: bool = Query(False),
    quote_max_len: int = Query(280, ge=50, le=2000),
):
    """Get a single claim by ID.

    Query parameters:
    - include_evidence: Include evidence array (default false)
    - include_history: Include review_history array (default false)
    - quote_max_len: Truncate quotes in evidence (default 280)
    """
    claim = get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    return ClaimResponse.from_dataclass(
        claim,
        include_evidence=include_evidence,
        include_history=include_history,
        quote_max_len=quote_max_len,
    )


@router.post("", response_model=ClaimResponse, status_code=201)
async def create_claim(request: CreateClaimRequest):
    """Create a new candidate claim.

    The claim starts with status='candidate' and confidence=0.3 (no evidence).
    """
    claim = add_claim(
        claim_text=request.claim_text,
        claim_type=request.claim_type,
        as_of=request.as_of,
    )

    # Apply optional temporal fields if provided
    if request.valid_from or request.valid_until:
        kwargs = {}
        if request.valid_from:
            kwargs["valid_from"] = request.valid_from
        if request.valid_until:
            kwargs["valid_until"] = request.valid_until
        claim = update_claim(claim.claim_id, **kwargs)

    return ClaimResponse.from_dataclass(claim, include_evidence=True, include_history=True)


class StatusTransitionErrorResponse(BaseModel):
    """Response for blocked status transitions."""

    detail: str
    current_status: str
    target_status: str
    reason: str
    transition_rules: Dict[str, Any]


@router.patch("/{claim_id}", response_model=ClaimResponse, responses={
    409: {"model": StatusTransitionErrorResponse, "description": "Status transition blocked by rules"}
})
async def update_claim_by_id(claim_id: str, request: UpdateClaimRequest):
    """Update a claim's status or temporal fields.

    Only the following fields can be updated:
    - status
    - as_of, valid_from, valid_until
    - note (stored in review_history)
    - force (bypass status transition rules)

    Status transitions are validated against rules:
    - candidate → accepted: requires min confidence (0.7) and supporting evidence
    - candidate → disputed: requires contradicting evidence
    - accepted → disputed: requires strong contradiction
    - * → deprecated: always allowed
    - disputed → candidate: always allowed (rework)
    - disputed → accepted: higher confidence bar (0.8)

    Use force=true to bypass validation (admin override).
    """
    existing = get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    # Build kwargs for update
    kwargs = {}
    if request.status is not None:
        kwargs["status"] = request.status
    if request.as_of is not None:
        kwargs["as_of"] = request.as_of
    if request.valid_from is not None:
        kwargs["valid_from"] = request.valid_from
    if request.valid_until is not None:
        kwargs["valid_until"] = request.valid_until

    # Build review history event if note provided
    review_history_event = None
    if request.note:
        review_history_event = {
            "event": "api_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": request.note,
            "fields_updated": list(kwargs.keys()),
            "forced": request.force,
        }

    if not kwargs and not review_history_event:
        # Nothing to update
        return ClaimResponse.from_dataclass(existing, include_evidence=True, include_history=True)

    try:
        claim = update_claim(
            claim_id,
            review_history_event=review_history_event,
            force=request.force,
            **kwargs,
        )
    except StatusTransitionError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(e),
                "current_status": e.current_status,
                "target_status": e.target_status,
                "reason": e.reason,
                "transition_rules": TRANSITION_RULES,
            },
        )

    return ClaimResponse.from_dataclass(claim, include_evidence=True, include_history=True)


@router.post("/{claim_id}/evidence", response_model=ClaimResponse)
async def add_evidence_to_claim(claim_id: str, request: AddEvidenceRequest):
    """Add evidence to a claim.

    Recalculates confidence and appends to audit history.
    Returns the updated claim with evidence and history included.
    """
    existing = get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    # Create Evidence dataclass
    evidence = EvidenceDataclass(
        evidence_id=str(uuid.uuid4()),
        source_type=request.source_type,
        source_id=request.source_id,
        quote=request.quote,
        support=request.support,
        weight=request.weight,
        retrieved_at=request.retrieved_at or datetime.now(timezone.utc).isoformat(),
        retrieval_query=request.retrieval_query,
        span_start=request.span_start,
        span_end=request.span_end,
        source_hash=request.source_hash,
    )

    claim = add_evidence(claim_id, evidence)

    return ClaimResponse.from_dataclass(claim, include_evidence=True, include_history=True)


@router.post("/{claim_id}/validate", response_model=ValidateResponse)
async def validate_claim(claim_id: str):
    """Validate a single claim against SADB evidence.

    This is a quick single-claim check, not the full Night Shift job.
    If SADB is unavailable, returns gracefully without error.
    """
    existing = get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    old_confidence = existing.confidence

    # Check SADB availability
    sadb_status = get_sadb_status()
    if not sadb_status["available"]:
        return ValidateResponse(
            claim_id=claim_id,
            sadb_available=False,
            evidence_added=0,
            old_confidence=old_confidence,
            new_confidence=old_confidence,
            claim=ClaimResponse.from_dataclass(existing, include_evidence=True),
        )

    # Search for evidence
    new_evidence = await search_evidence(existing.claim_text, top_k=3)

    # Classify support for each piece
    for ev in new_evidence:
        ev.support = classify_support(existing.claim_text, ev.quote)

    # Add evidence (fingerprint dedupe happens in add_evidence)
    evidence_added = 0
    for ev in new_evidence:
        before_count = len(get_claim(claim_id).evidence)
        add_evidence(claim_id, ev)
        after_count = len(get_claim(claim_id).evidence)
        if after_count > before_count:
            evidence_added += 1

    # Get updated claim
    updated = get_claim(claim_id)

    return ValidateResponse(
        claim_id=claim_id,
        sadb_available=True,
        evidence_added=evidence_added,
        old_confidence=old_confidence,
        new_confidence=updated.confidence,
        claim=ClaimResponse.from_dataclass(updated, include_evidence=True),
    )


@router.post("/{claim_id}/adjudicate", response_model=AdjudicateResponse)
async def adjudicate_claim_endpoint(claim_id: str, request: AdjudicateRequest):
    """Run panel adjudication on a claim.

    A panel of LLMs evaluates the claim against its evidence and returns
    a consensus verdict. Each panel member must cite only existing evidence IDs.

    Verdicts:
    - accept: Evidence strongly supports the claim
    - dispute: Evidence contradicts the claim
    - insufficient: Not enough evidence to determine
    - abstain: Unable to evaluate

    If update_status=true and consensus is strong enough:
    - strict mode: requires ≥70% confidence
    - lenient mode: requires ≥50% confidence
    """
    existing = get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    try:
        result = await adjudicate_claim(
            claim_id=claim_id,
            models=request.models,
            panel_size=request.panel_size,
            mode=request.mode,
            update_status=request.update_status,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Get updated claim for response
    updated_claim = get_claim(claim_id)

    return AdjudicateResponse(
        claim_id=result.claim_id,
        panel_size=result.panel_size,
        mode=result.mode,
        panel_verdicts=[
            PanelVerdictResponse(
                model=v.model,
                verdict=v.verdict,
                confidence=v.confidence,
                reasoning=v.reasoning,
                cited_evidence=v.cited_evidence,
                concerns=v.concerns,
                error=v.error,
            )
            for v in result.panel_verdicts
        ],
        consensus_verdict=result.consensus_verdict,
        consensus_confidence=result.consensus_confidence,
        timestamp=result.timestamp,
        status_updated=result.status_updated,
        new_status=result.new_status,
        claim=ClaimResponse.from_dataclass(updated_claim, include_evidence=True, include_history=True),
    )
