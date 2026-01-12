"""Claims API router.

Thin API layer that delegates to backend/claims.py for business logic.
Routes validate requests, call CRUD functions, and return serialized responses.

Endpoints:
- GET /api/claims - List claims with filters and pagination
- GET /api/claims/{claim_id} - Get single claim
- POST /api/claims - Create candidate claim
- PATCH /api/claims/{claim_id} - Update status/temporal fields
- POST /api/claims/{claim_id}/evidence - Add evidence
- POST /api/claims/{claim_id}/validate - Single-claim SADB validation
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
    Evidence as EvidenceDataclass,
    Claim as ClaimDataclass,
)
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
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_evidence: bool = Query(False),
    include_history: bool = Query(False),
    quote_max_len: int = Query(280, ge=50, le=2000),
):
    """List claims with optional filters and pagination.

    Query parameters:
    - status: Filter by claim status
    - claim_type: Filter by claim type
    - min_confidence: Filter by minimum confidence score
    - valid_at: Filter by temporal validity at given ISO timestamp
    - q: Substring search on claim_text
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


@router.patch("/{claim_id}", response_model=ClaimResponse)
async def update_claim_by_id(claim_id: str, request: UpdateClaimRequest):
    """Update a claim's status or temporal fields.

    Only the following fields can be updated:
    - status
    - as_of, valid_from, valid_until
    - note (stored in review_history)
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
        }

    if not kwargs and not review_history_event:
        # Nothing to update
        return ClaimResponse.from_dataclass(existing, include_evidence=True, include_history=True)

    claim = update_claim(
        claim_id,
        review_history_event=review_history_event,
        **kwargs,
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
