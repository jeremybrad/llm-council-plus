"""Confidence scorer for claims with full explainability.

This module provides v1 heuristic confidence scoring for claims based on evidence.
Every score change includes a breakdown showing exactly how the score was calculated.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .claims import Evidence


@dataclass
class ScoreBreakdown:
    """Explainability for confidence scores.

    Every confidence calculation produces a breakdown showing:
    - How each component contributed to the score
    - Which evidence was used in the calculation
    - What cap was applied and why
    """

    base_score: float
    supporting_bonus: float
    independence_bonus: float
    contradiction_penalty: float
    final_score: float
    evidence_ids_used: list[str]
    calculation_timestamp: str
    cap_applied: float
    cap_reason: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreBreakdown":
        """Create from dictionary."""
        return cls(**data)


# Confidence caps
CAP_STRONG = 0.99  # Only with 2+ strong independent sources
CAP_WEAK = 0.98  # Default cap when insufficient independence


def calculate_confidence_with_breakdown(
    evidence_list: list["Evidence"],
) -> tuple[float, ScoreBreakdown]:
    """
    Calculate confidence score with full explainability breakdown.

    v1 heuristic rules:
    - Base: 0.5 for any claim with at least one supporting evidence
    - +0.1 per additional supporting evidence (cap at 0.3 bonus)
    - +0.1 for independent sources (different independence_key)
    - -0.2 per contradicting evidence
    - Cap at 0.98 unless 2+ strong independent sources (then 0.99)

    Args:
        evidence_list: List of Evidence objects attached to a claim

    Returns:
        Tuple of (confidence_score, ScoreBreakdown)
    """
    now = datetime.now(timezone.utc).isoformat()
    evidence_ids_used = []

    # No evidence case
    if not evidence_list:
        breakdown = ScoreBreakdown(
            base_score=0.3,
            supporting_bonus=0.0,
            independence_bonus=0.0,
            contradiction_penalty=0.0,
            final_score=0.3,
            evidence_ids_used=[],
            calculation_timestamp=now,
            cap_applied=CAP_WEAK,
            cap_reason="no_evidence",
        )
        return 0.3, breakdown

    # Categorize evidence
    supporting = [e for e in evidence_list if e.support == "supports"]
    contradicting = [e for e in evidence_list if e.support == "contradicts"]

    # Track which evidence contributed
    for e in supporting + contradicting:
        evidence_ids_used.append(e.evidence_id)

    # No supporting evidence case
    if not supporting:
        penalty = len(contradicting) * 0.2
        final = max(0.1, 0.3 - penalty)
        breakdown = ScoreBreakdown(
            base_score=0.3,
            supporting_bonus=0.0,
            independence_bonus=0.0,
            contradiction_penalty=round(penalty, 2),
            final_score=round(final, 2),
            evidence_ids_used=evidence_ids_used,
            calculation_timestamp=now,
            cap_applied=CAP_WEAK,
            cap_reason="no_supporting_evidence",
        )
        return round(final, 2), breakdown

    # Base score for having supporting evidence
    base_score = 0.5

    # Supporting bonus: +0.1 per additional supporting evidence, cap at 0.3
    supporting_bonus = min(0.3, (len(supporting) - 1) * 0.1)

    # Independence bonus: +0.1 if multiple unique sources
    independence_keys = set(e.independence_key for e in supporting)
    independence_bonus = 0.1 if len(independence_keys) > 1 else 0.0

    # Contradiction penalty: -0.2 per contradicting piece
    contradiction_penalty = len(contradicting) * 0.2

    # Calculate raw score
    raw_score = base_score + supporting_bonus + independence_bonus - contradiction_penalty

    # Determine confidence cap based on independent sources
    # Only allow 0.99 when there are 2+ strong independent sources (weight > 0.7)
    strong_sources = [e for e in supporting if e.weight > 0.7]
    strong_keys = {e.independence_key for e in strong_sources}

    if len(strong_keys) >= 2:
        cap = CAP_STRONG
        cap_reason = "strong_independent_sources"
    else:
        cap = CAP_WEAK
        cap_reason = "insufficient_independent_sources"

    # Apply floor and cap
    final_score = max(0.1, min(cap, raw_score))

    breakdown = ScoreBreakdown(
        base_score=base_score,
        supporting_bonus=round(supporting_bonus, 2),
        independence_bonus=round(independence_bonus, 2),
        contradiction_penalty=round(contradiction_penalty, 2),
        final_score=round(final_score, 2),
        evidence_ids_used=evidence_ids_used,
        calculation_timestamp=now,
        cap_applied=cap,
        cap_reason=cap_reason,
    )

    return round(final_score, 2), breakdown


def classify_support(claim_text: str, evidence_quote: str) -> str:
    """
    Classify whether evidence supports, contradicts, or is neutral to a claim.

    v1 implementation uses simple heuristics:
    - Negation polarity detection
    - Keyword overlap analysis

    Args:
        claim_text: The claim being evaluated
        evidence_quote: The evidence text to classify

    Returns:
        "supports", "contradicts", or "neutral"
    """
    claim_lower = claim_text.lower()
    quote_lower = evidence_quote.lower()

    # Negation patterns
    negations = [
        "not",
        "never",
        "no longer",
        "doesn't",
        "isn't",
        "wasn't",
        "don't",
        "didn't",
        "won't",
        "wouldn't",
        "can't",
        "couldn't",
        "shouldn't",
        "haven't",
        "hasn't",
        "hadn't",
    ]

    claim_negated = any(neg in claim_lower for neg in negations)
    quote_negated = any(neg in quote_lower for neg in negations)

    # If opposite polarity, likely contradicts
    if claim_negated != quote_negated:
        return "contradicts"

    # Calculate keyword overlap
    # Remove common stop words for better signal
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "about",
        "against",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
    }

    claim_words = set(claim_lower.split()) - stop_words
    quote_words = set(quote_lower.split()) - stop_words

    if not claim_words:
        return "neutral"

    overlap = len(claim_words & quote_words) / len(claim_words)

    if overlap > 0.3:
        return "supports"

    return "neutral"
