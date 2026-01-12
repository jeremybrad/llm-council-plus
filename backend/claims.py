"""Claims store with evidence-backed confidence scoring.

This module provides a persistent claims store for storing facts about Jeremy
with linked evidence and confidence scoring. Key features:

- Atomic writes with cross-platform file locking
- Evidence provenance tracking (source, span, query, hash)
- Fingerprint-based deduplication (same source+span+quote = duplicate)
- Independence-based scoring (same source = no independence bonus)
- Full audit trail in review_history
- Temporal validity for time-sensitive claims
"""

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import portalocker

from .scorer import ScoreBreakdown, calculate_confidence_with_breakdown


# Storage paths (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"
CLAIMS_DIR = DATA_DIR / "claims"
CLAIMS_FILE = CLAIMS_DIR / "claims.json"
LOCK_FILE = CLAIMS_DIR / "claims.json.lock"
HISTORY_DIR = CLAIMS_DIR / "history"


@dataclass
class Evidence:
    """A piece of evidence supporting or contradicting a claim.

    Attributes:
        evidence_id: Unique identifier (UUID)
        source_type: Type of source (repo_doc, transcript, note, chat_summary, j1_fact)
        source_id: Identifier within the source (path, conversation_id, fact_id)
        quote: The relevant text snippet (max 500 chars recommended)
        support: Classification (supports, contradicts, neutral)
        weight: Evidence strength 0.0-1.0

        Provenance fields:
        retrieved_at: ISO timestamp when evidence was retrieved
        retrieval_query: The query used to find this evidence
        span_start: Character offset in source (optional)
        span_end: Character offset end (optional)
        source_hash: SHA256 prefix of source content for drift detection (optional)
    """
    evidence_id: str
    source_type: str
    source_id: str
    quote: str
    support: str  # "supports", "contradicts", "neutral"
    weight: float
    retrieved_at: str
    retrieval_query: str
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    source_hash: Optional[str] = None

    @property
    def independence_key(self) -> str:
        """Key for scoring independence (NOT storage dedupe).

        Two pieces of evidence from the same source have the same independence_key,
        meaning they don't contribute to independence bonus in scoring.
        """
        return f"{self.source_type}:{self.source_id}"

    @property
    def fingerprint(self) -> str:
        """Unique fingerprint for storage deduplication.

        Uses source + span + quote hash. Two evidence pieces with the same
        fingerprint are considered duplicates and won't be stored twice.
        This allows multiple different quotes from the same source.
        """
        span = f"{self.span_start or ''}:{self.span_end or ''}"
        qh = hashlib.sha256(self.quote.encode("utf-8")).hexdigest()[:16]
        return f"{self.source_type}:{self.source_id}:{span}:{qh}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Claim:
    """A claim about Jeremy with evidence and confidence scoring.

    Attributes:
        claim_id: Stable UUID identifier
        claim_text: The claim statement
        claim_type: Category (biographical, preference, belief, event, relationship, project)
        confidence: Current confidence score 0.0-1.0
        status: Lifecycle status (candidate, accepted, disputed, deprecated)
        evidence: List of attached evidence pieces

        Temporal validity:
        as_of: When this claim was known to be true (for time-sensitive claims)
        valid_from: Start of validity period
        valid_until: End of validity period (None = still valid)

        Explainability:
        score_breakdown: Detailed breakdown of how confidence was calculated

        Audit trail:
        created_at: When the claim was first created
        last_reviewed_at: Last time evidence was added or claim reviewed
        review_history: Full audit log of all changes
    """
    claim_id: str
    claim_text: str
    claim_type: str
    confidence: float
    status: str
    evidence: List[Evidence]
    as_of: Optional[str]
    valid_from: Optional[str]
    valid_until: Optional[str]
    score_breakdown: Optional[ScoreBreakdown]
    created_at: str
    last_reviewed_at: Optional[str]
    review_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "confidence": self.confidence,
            "status": self.status,
            "evidence": [e.to_dict() for e in self.evidence],
            "as_of": self.as_of,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "score_breakdown": self.score_breakdown.to_dict() if self.score_breakdown else None,
            "created_at": self.created_at,
            "last_reviewed_at": self.last_reviewed_at,
            "review_history": self.review_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create from dictionary."""
        evidence = [Evidence.from_dict(e) for e in data.get("evidence", [])]
        score_breakdown = None
        if data.get("score_breakdown"):
            score_breakdown = ScoreBreakdown.from_dict(data["score_breakdown"])

        return cls(
            claim_id=data["claim_id"],
            claim_text=data["claim_text"],
            claim_type=data["claim_type"],
            confidence=data["confidence"],
            status=data["status"],
            evidence=evidence,
            as_of=data.get("as_of"),
            valid_from=data.get("valid_from"),
            valid_until=data.get("valid_until"),
            score_breakdown=score_breakdown,
            created_at=data["created_at"],
            last_reviewed_at=data.get("last_reviewed_at"),
            review_history=data.get("review_history", []),
        )


def _ensure_dirs() -> None:
    """Ensure all required directories exist."""
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _save_claims(claims: Dict[str, Claim]) -> None:
    """Atomic write with cross-platform file locking for crash safety.

    Uses portalocker for Windows/Mac/Linux compatibility.
    Writes to temp file, fsyncs, then atomically renames.
    """
    _ensure_dirs()

    # Acquire exclusive lock
    with portalocker.Lock(str(LOCK_FILE), timeout=10, mode='w'):
        # Write to temp file in same directory (ensures same filesystem)
        fd, tmp_path = tempfile.mkstemp(dir=CLAIMS_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(
                    {k: v.to_dict() for k, v in claims.items()},
                    f,
                    indent=2
                )
                f.flush()
                os.fsync(f.fileno())
            # Atomic rename
            os.rename(tmp_path, CLAIMS_FILE)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


def _load_claims() -> Dict[str, Claim]:
    """Load claims with shared lock."""
    _ensure_dirs()

    if not CLAIMS_FILE.exists():
        return {}

    with portalocker.Lock(str(LOCK_FILE), timeout=10, mode='r'):
        with open(CLAIMS_FILE) as f:
            data = json.load(f)
        return {k: Claim.from_dict(v) for k, v in data.items()}


def _log_to_history(event: Dict[str, Any]) -> None:
    """Append an event to today's history log."""
    _ensure_dirs()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history_file = HISTORY_DIR / f"{today}.jsonl"

    with open(history_file, 'a') as f:
        f.write(json.dumps(event) + "\n")


def _is_valid_at(claim: Claim, timestamp: str) -> bool:
    """Check if a claim was valid at a given timestamp.

    Args:
        claim: The claim to check
        timestamp: ISO format timestamp to check validity at

    Returns:
        True if claim was valid at that time
    """
    if claim.valid_from and timestamp < claim.valid_from:
        return False
    if claim.valid_until and timestamp > claim.valid_until:
        return False
    return True


# =============================================================================
# Public API
# =============================================================================

def add_claim(
    claim_text: str,
    claim_type: str,
    as_of: Optional[str] = None,
) -> Claim:
    """Create a new candidate claim with stable UUID.

    Args:
        claim_text: The claim statement
        claim_type: Category (biographical, preference, belief, event, relationship, project)
        as_of: When this claim was known to be true (optional, defaults to now)

    Returns:
        The created Claim object
    """
    now = datetime.now(timezone.utc).isoformat()
    as_of_ts = as_of or now

    claim = Claim(
        claim_id=str(uuid.uuid4()),
        claim_text=claim_text,
        claim_type=claim_type,
        confidence=0.3,  # No evidence = low confidence
        status="candidate",
        evidence=[],
        as_of=as_of_ts,
        valid_from=as_of_ts,
        valid_until=None,
        score_breakdown=None,
        created_at=now,
        last_reviewed_at=None,
        review_history=[{
            "event": "claim_created",
            "timestamp": now,
            "claim_type": claim_type,
        }],
    )

    claims = _load_claims()
    claims[claim.claim_id] = claim
    _save_claims(claims)

    _log_to_history({
        "event": "claim_created",
        "timestamp": now,
        "claim_id": claim.claim_id,
        "claim_text": claim_text[:100],
        "claim_type": claim_type,
    })

    return claim


def get_claim(claim_id: str) -> Optional[Claim]:
    """Retrieve a claim by ID.

    Args:
        claim_id: The claim's UUID

    Returns:
        The Claim object or None if not found
    """
    claims = _load_claims()
    return claims.get(claim_id)


def query_claims(
    status: Optional[str] = None,
    claim_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    valid_at: Optional[str] = None,
) -> List[Claim]:
    """Query claims with optional filters.

    Args:
        status: Filter by status (candidate, accepted, disputed, deprecated)
        claim_type: Filter by type (biographical, preference, etc.)
        min_confidence: Filter by minimum confidence score
        valid_at: Filter by temporal validity at given ISO timestamp

    Returns:
        List of matching claims
    """
    claims = _load_claims()
    results = list(claims.values())

    if status:
        results = [c for c in results if c.status == status]
    if claim_type:
        results = [c for c in results if c.claim_type == claim_type]
    if min_confidence is not None:
        results = [c for c in results if c.confidence >= min_confidence]
    if valid_at:
        results = [c for c in results if _is_valid_at(c, valid_at)]

    return results


def add_evidence(claim_id: str, evidence: Evidence) -> Claim:
    """Add evidence to a claim and recalculate confidence.

    Deduplicates by fingerprint (not independence_key). This allows
    multiple different quotes from the same source to be stored,
    but prevents exact duplicates.

    Args:
        claim_id: The claim to add evidence to
        evidence: The Evidence object to attach

    Returns:
        The updated Claim object

    Raises:
        ValueError: If claim not found
    """
    claims = _load_claims()
    claim = claims.get(claim_id)
    if not claim:
        raise ValueError(f"Claim {claim_id} not found")

    now = datetime.now(timezone.utc).isoformat()
    old_confidence = claim.confidence

    # Deduplicate by fingerprint (NOT independence_key)
    # This allows multiple pieces from same source if they're different quotes/spans
    existing_fingerprints = {e.fingerprint for e in claim.evidence}
    if evidence.fingerprint not in existing_fingerprints:
        claim.evidence.append(evidence)

        # Log to review_history
        claim.review_history.append({
            "event": "evidence_added",
            "timestamp": now,
            "evidence_id": evidence.evidence_id,
            "source_type": evidence.source_type,
            "support": evidence.support,
            "weight": evidence.weight,
        })

    # Recalculate confidence with breakdown
    claim.confidence, claim.score_breakdown = calculate_confidence_with_breakdown(claim.evidence)
    claim.last_reviewed_at = now

    # Log confidence change if significant
    if abs(claim.confidence - old_confidence) > 0.01:
        claim.review_history.append({
            "event": "confidence_updated",
            "timestamp": now,
            "old_confidence": old_confidence,
            "new_confidence": claim.confidence,
            "score_breakdown": claim.score_breakdown.to_dict() if claim.score_breakdown else None,
        })

    claims[claim_id] = claim
    _save_claims(claims)

    _log_to_history({
        "event": "evidence_added",
        "timestamp": now,
        "claim_id": claim_id,
        "evidence_id": evidence.evidence_id,
        "support": evidence.support,
        "old_confidence": old_confidence,
        "new_confidence": claim.confidence,
    })

    return claim


def update_claim(
    claim_id: str,
    review_history_event: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Claim:
    """Update claim fields and optionally append to review_history.

    Args:
        claim_id: The claim to update
        review_history_event: Optional event dict to append to history
        **kwargs: Fields to update (status, confidence, evidence, etc.)

    Returns:
        The updated Claim object

    Raises:
        ValueError: If claim not found
    """
    claims = _load_claims()
    claim = claims.get(claim_id)
    if not claim:
        raise ValueError(f"Claim {claim_id} not found")

    now = datetime.now(timezone.utc).isoformat()
    old_status = claim.status

    # Update fields
    for key, value in kwargs.items():
        if hasattr(claim, key):
            setattr(claim, key, value)

    # Log status change if it happened
    if 'status' in kwargs and kwargs['status'] != old_status:
        claim.review_history.append({
            "event": "status_changed",
            "timestamp": now,
            "old_status": old_status,
            "new_status": kwargs['status'],
        })
        _log_to_history({
            "event": "status_changed",
            "timestamp": now,
            "claim_id": claim_id,
            "old_status": old_status,
            "new_status": kwargs['status'],
        })

    # Append custom review history event if provided
    if review_history_event:
        claim.review_history.append(review_history_event)

    claim.last_reviewed_at = now
    claims[claim_id] = claim
    _save_claims(claims)

    return claim


def archive_claim(claim_id: str) -> Claim:
    """Archive a claim by setting status to deprecated.

    Args:
        claim_id: The claim to archive

    Returns:
        The updated Claim object
    """
    return update_claim(claim_id, status="deprecated")


def get_claims_for_review(
    n: int = 5,
    band: Tuple[float, float] = (0.60, 0.98),
) -> List[Claim]:
    """Select claims in confidence band for validation.

    Prioritizes claims that have never been reviewed or were reviewed
    longest ago.

    Args:
        n: Maximum number of claims to return
        band: (min_confidence, max_confidence) tuple

    Returns:
        List of claims ready for review, oldest-reviewed first
    """
    # Get claims in candidate or accepted status
    candidates = query_claims(status="candidate")
    accepted = query_claims(status="accepted")
    all_claims = candidates + accepted

    # Filter to confidence band
    in_band = [c for c in all_claims if band[0] <= c.confidence <= band[1]]

    # Sort by last_reviewed_at (None = never reviewed = highest priority)
    # Empty string sorts before any date, so never-reviewed claims come first
    in_band.sort(key=lambda c: c.last_reviewed_at or "")

    return in_band[:n]


def get_all_claims() -> List[Claim]:
    """Get all claims in the store.

    Returns:
        List of all claims
    """
    claims = _load_claims()
    return list(claims.values())


def delete_claim(claim_id: str) -> bool:
    """Permanently delete a claim (use archive_claim instead when possible).

    Args:
        claim_id: The claim to delete

    Returns:
        True if claim was deleted, False if not found
    """
    claims = _load_claims()
    if claim_id not in claims:
        return False

    del claims[claim_id]
    _save_claims(claims)

    _log_to_history({
        "event": "claim_deleted",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "claim_id": claim_id,
    })

    return True


def clear_all_claims() -> int:
    """Delete all claims (for testing only).

    Returns:
        Number of claims deleted
    """
    claims = _load_claims()
    count = len(claims)
    _save_claims({})
    return count
