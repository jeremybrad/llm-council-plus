"""Tests for the claims store and confidence scoring.

Tests cover:
- Claim CRUD operations
- Evidence management with fingerprint deduplication
- Confidence scoring with explainability
- Independence key only affects scoring (not storage)
- Temporal validity filtering
- Review history audit trail
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


# Patch the data directories before importing claims module
@pytest.fixture
def temp_claims_dir(tmp_path):
    """Create temporary claims directory and patch the module to use it."""
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir()
    history_dir = claims_dir / "history"
    history_dir.mkdir()

    with patch.multiple(
        "backend.claims",
        CLAIMS_DIR=claims_dir,
        CLAIMS_FILE=claims_dir / "claims.json",
        LOCK_FILE=claims_dir / "claims.json.lock",
        HISTORY_DIR=history_dir,
    ):
        # Also need to patch DATA_DIR since _ensure_dirs might use it
        with patch("backend.claims.DATA_DIR", tmp_path):
            yield claims_dir


@pytest.fixture
def claims_module(temp_claims_dir):
    """Import claims module with patched directories."""
    from backend import claims
    from backend.claims import (
        add_claim,
        get_claim,
        add_evidence,
        query_claims,
        update_claim,
        get_claims_for_review,
        clear_all_claims,
        Evidence,
        Claim,
    )

    # Clear any existing claims
    clear_all_claims()

    return {
        "add_claim": add_claim,
        "get_claim": get_claim,
        "add_evidence": add_evidence,
        "query_claims": query_claims,
        "update_claim": update_claim,
        "get_claims_for_review": get_claims_for_review,
        "clear_all_claims": clear_all_claims,
        "Evidence": Evidence,
        "Claim": Claim,
    }


class TestClaimCreation:
    """Tests for claim creation."""

    def test_add_claim_generates_stable_uuid(self, claims_module):
        """New claims should have valid UUIDs."""
        add_claim = claims_module["add_claim"]

        claim = add_claim("Jeremy uses Python", "preference")

        assert claim.claim_id is not None
        assert len(claim.claim_id) == 36  # UUID format: 8-4-4-4-12
        assert "-" in claim.claim_id

    def test_add_claim_sets_defaults(self, claims_module):
        """New claims should have correct default values."""
        add_claim = claims_module["add_claim"]

        claim = add_claim("Test claim", "biographical")

        assert claim.status == "candidate"
        assert claim.confidence == 0.3  # No evidence default
        assert claim.evidence == []
        assert claim.claim_type == "biographical"
        assert claim.created_at is not None
        assert claim.review_history is not None
        assert len(claim.review_history) == 1  # claim_created event

    def test_add_claim_with_as_of(self, claims_module):
        """Claims can specify when they were known to be true."""
        add_claim = claims_module["add_claim"]

        as_of = "2025-01-01T00:00:00+00:00"
        claim = add_claim("Jeremy preferred Vim in 2025", "preference", as_of=as_of)

        assert claim.as_of == as_of
        assert claim.valid_from == as_of

    def test_claim_persists_across_loads(self, claims_module):
        """Claims should persist and be retrievable."""
        add_claim = claims_module["add_claim"]
        get_claim = claims_module["get_claim"]

        claim = add_claim("Persistent claim", "biographical")
        retrieved = get_claim(claim.claim_id)

        assert retrieved is not None
        assert retrieved.claim_id == claim.claim_id
        assert retrieved.claim_text == "Persistent claim"


class TestEvidenceManagement:
    """Tests for evidence attachment and deduplication."""

    def test_add_evidence_updates_confidence(self, claims_module):
        """Adding evidence should update confidence score."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Jeremy uses Python", "preference")
        initial_confidence = claim.confidence

        evidence = Evidence(
            evidence_id="ev_001",
            source_type="transcript",
            source_id="conv123",
            quote="I really like Python for scripting",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="Jeremy Python preference",
        )

        updated = add_evidence(claim.claim_id, evidence)

        assert updated.confidence > initial_confidence
        assert len(updated.evidence) == 1

    def test_add_evidence_includes_breakdown(self, claims_module):
        """Evidence addition should include score breakdown."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Jeremy uses Python", "preference")

        evidence = Evidence(
            evidence_id="ev_001",
            source_type="transcript",
            source_id="conv123",
            quote="I really like Python",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="Jeremy Python",
        )

        updated = add_evidence(claim.claim_id, evidence)

        assert updated.score_breakdown is not None
        assert "ev_001" in updated.score_breakdown.evidence_ids_used
        assert updated.score_breakdown.base_score == 0.5
        assert updated.score_breakdown.calculation_timestamp is not None

    def test_fingerprint_dedup_allows_different_quotes(self, claims_module):
        """Different quotes from same source should both be stored."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test dedup", "biographical")

        ev1 = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="First quote about topic",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test query",
        )

        ev2 = Evidence(
            evidence_id="ev2",
            source_type="transcript",
            source_id="doc1",  # Same source
            quote="Different quote from same doc",  # Different quote
            support="supports",
            weight=0.7,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test query",
        )

        add_evidence(claim.claim_id, ev1)
        updated = add_evidence(claim.claim_id, ev2)

        assert len(updated.evidence) == 2  # Both stored (different fingerprints)

    def test_fingerprint_dedup_blocks_duplicates(self, claims_module):
        """Exact same quote+source+span should be deduplicated."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test dedup", "biographical")

        ev1 = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="Same exact quote",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test query",
            span_start=100,
            span_end=200,
        )

        ev2 = Evidence(
            evidence_id="ev2",  # Different ID
            source_type="transcript",
            source_id="doc1",  # Same source
            quote="Same exact quote",  # Same quote
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test query",
            span_start=100,  # Same span
            span_end=200,
        )

        add_evidence(claim.claim_id, ev1)
        updated = add_evidence(claim.claim_id, ev2)

        assert len(updated.evidence) == 1  # Deduplicated

    def test_independence_key_only_affects_scoring(self, claims_module):
        """Multiple pieces from same source get stored but don't boost independence bonus."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test independence", "biographical")

        ev1 = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="Quote 1 from doc1",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        ev2 = Evidence(
            evidence_id="ev2",
            source_type="transcript",
            source_id="doc1",  # Same source = same independence_key
            quote="Quote 2 different from doc1",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        add_evidence(claim.claim_id, ev1)
        updated = add_evidence(claim.claim_id, ev2)

        assert len(updated.evidence) == 2  # Both stored
        assert updated.score_breakdown.independence_bonus == 0.0  # No bonus


class TestConfidenceScoring:
    """Tests for confidence scoring rules."""

    def test_confidence_cap_at_098_default(self, claims_module):
        """Confidence should cap at 0.98 without strong independent sources."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test cap", "biographical")

        # Add many supporting pieces from same source
        for i in range(10):
            ev = Evidence(
                evidence_id=f"ev{i}",
                source_type="transcript",
                source_id="doc1",  # All same source
                quote=f"Supporting quote {i}",
                support="supports",
                weight=0.9,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                retrieval_query="test",
            )
            add_evidence(claim.claim_id, ev)

        from backend.claims import get_claim
        final = get_claim(claim.claim_id)

        assert final.confidence <= 0.98
        assert final.score_breakdown.cap_applied == 0.98
        assert final.score_breakdown.cap_reason == "insufficient_independent_sources"

    def test_confidence_cap_099_with_strong_independent(self, claims_module):
        """Confidence can reach 0.99 with 2+ strong independent sources."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test high cap", "biographical")

        # Add strong evidence from two independent sources
        ev1 = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="Strong supporting evidence from source 1",
            support="supports",
            weight=0.9,  # Strong (> 0.7)
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        ev2 = Evidence(
            evidence_id="ev2",
            source_type="repo_doc",  # Different source_type = different independence_key
            source_id="readme.md",
            quote="Strong supporting evidence from source 2",
            support="supports",
            weight=0.85,  # Strong (> 0.7)
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        # Add a third piece of supporting evidence
        ev3 = Evidence(
            evidence_id="ev3",
            source_type="note",
            source_id="note1",
            quote="Third supporting quote",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        add_evidence(claim.claim_id, ev1)
        add_evidence(claim.claim_id, ev2)
        add_evidence(claim.claim_id, ev3)

        from backend.claims import get_claim
        final = get_claim(claim.claim_id)

        assert final.score_breakdown.cap_applied == 0.99
        assert final.score_breakdown.cap_reason == "strong_independent_sources"

    def test_contradiction_reduces_confidence(self, claims_module):
        """Contradicting evidence should reduce confidence."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test contradiction", "preference")

        # Add supporting evidence
        ev1 = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="Supporting quote",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        add_evidence(claim.claim_id, ev1)

        from backend.claims import get_claim
        after_support = get_claim(claim.claim_id)
        confidence_before = after_support.confidence

        # Add contradicting evidence
        ev2 = Evidence(
            evidence_id="ev2",
            source_type="transcript",
            source_id="doc2",
            quote="Contradicting quote",
            support="contradicts",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        add_evidence(claim.claim_id, ev2)
        final = get_claim(claim.claim_id)

        assert final.confidence < confidence_before
        assert final.score_breakdown.contradiction_penalty == 0.2


class TestQueryAndFilter:
    """Tests for claim querying and filtering."""

    def test_query_by_status(self, claims_module):
        """Should filter claims by status."""
        add_claim = claims_module["add_claim"]
        query_claims = claims_module["query_claims"]
        update_claim = claims_module["update_claim"]

        claim1 = add_claim("Candidate claim", "biographical")
        claim2 = add_claim("Accepted claim", "biographical")
        update_claim(claim2.claim_id, status="accepted", force=True)

        candidates = query_claims(status="candidate")
        accepted = query_claims(status="accepted")

        assert len(candidates) == 1
        assert candidates[0].claim_id == claim1.claim_id
        assert len(accepted) == 1
        assert accepted[0].claim_id == claim2.claim_id

    def test_query_by_type(self, claims_module):
        """Should filter claims by type."""
        add_claim = claims_module["add_claim"]
        query_claims = claims_module["query_claims"]

        add_claim("Bio claim", "biographical")
        add_claim("Pref claim", "preference")

        bio_claims = query_claims(claim_type="biographical")
        pref_claims = query_claims(claim_type="preference")

        assert len(bio_claims) == 1
        assert len(pref_claims) == 1

    def test_query_by_min_confidence(self, claims_module):
        """Should filter claims by minimum confidence."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        query_claims = claims_module["query_claims"]
        Evidence = claims_module["Evidence"]

        claim1 = add_claim("Low confidence", "biographical")  # 0.3 default
        claim2 = add_claim("High confidence", "biographical")

        # Add evidence to raise confidence
        ev = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="Supporting",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )
        add_evidence(claim2.claim_id, ev)

        low = query_claims(min_confidence=0.4)
        all_claims = query_claims()

        assert len(low) == 1  # Only claim2 above 0.4
        assert len(all_claims) == 2

    def test_temporal_validity_filtering(self, claims_module):
        """Should filter claims by temporal validity."""
        add_claim = claims_module["add_claim"]
        query_claims = claims_module["query_claims"]
        update_claim = claims_module["update_claim"]

        # Claim valid in 2025 only
        claim = add_claim("Valid in 2025", "preference", as_of="2025-01-01T00:00:00+00:00")
        update_claim(claim.claim_id, valid_until="2025-12-31T23:59:59+00:00")

        # Query at different times
        during = query_claims(valid_at="2025-06-15T00:00:00+00:00")
        after = query_claims(valid_at="2026-06-15T00:00:00+00:00")

        assert len(during) == 1
        assert len(after) == 0

    def test_get_claims_for_review(self, claims_module):
        """Should return claims in confidence band, oldest first."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        get_claims_for_review = claims_module["get_claims_for_review"]
        Evidence = claims_module["Evidence"]

        # Create claims at different confidence levels
        low = add_claim("Low conf", "biographical")  # 0.3 - below band
        mid = add_claim("Mid conf", "biographical")  # Will be 0.7

        # Raise mid claim confidence
        for i in range(3):
            ev = Evidence(
                evidence_id=f"ev{i}",
                source_type="transcript",
                source_id=f"doc{i}",
                quote=f"Quote {i}",
                support="supports",
                weight=0.8,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                retrieval_query="test",
            )
            add_evidence(mid.claim_id, ev)

        # Get claims in default band (0.60-0.98)
        for_review = get_claims_for_review(n=5)

        assert len(for_review) == 1
        assert for_review[0].claim_id == mid.claim_id


class TestReviewHistory:
    """Tests for audit trail in review_history."""

    def test_claim_created_event(self, claims_module):
        """Claim creation should log event."""
        add_claim = claims_module["add_claim"]

        claim = add_claim("Test claim", "biographical")

        assert len(claim.review_history) == 1
        assert claim.review_history[0]["event"] == "claim_created"
        assert "timestamp" in claim.review_history[0]

    def test_evidence_added_event(self, claims_module):
        """Evidence addition should log event."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test claim", "biographical")

        ev = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="Quote",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        updated = add_evidence(claim.claim_id, ev)

        # Find evidence_added event
        ev_added = [e for e in updated.review_history if e["event"] == "evidence_added"]
        assert len(ev_added) == 1
        assert ev_added[0]["evidence_id"] == "ev1"

    def test_status_change_event(self, claims_module):
        """Status changes should log event."""
        add_claim = claims_module["add_claim"]
        update_claim = claims_module["update_claim"]

        claim = add_claim("Test claim", "biographical")
        updated = update_claim(claim.claim_id, status="accepted", force=True)

        # Find status_changed event
        status_changed = [e for e in updated.review_history if e["event"] == "status_changed"]
        assert len(status_changed) == 1
        assert status_changed[0]["old_status"] == "candidate"
        assert status_changed[0]["new_status"] == "accepted"
        assert status_changed[0]["forced"] is True

    def test_confidence_updated_event(self, claims_module):
        """Significant confidence changes should log event with breakdown."""
        add_claim = claims_module["add_claim"]
        add_evidence = claims_module["add_evidence"]
        Evidence = claims_module["Evidence"]

        claim = add_claim("Test claim", "biographical")

        ev = Evidence(
            evidence_id="ev1",
            source_type="transcript",
            source_id="doc1",
            quote="Supporting quote",
            support="supports",
            weight=0.8,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test",
        )

        updated = add_evidence(claim.claim_id, ev)

        # Find confidence_updated event
        conf_updated = [e for e in updated.review_history if e["event"] == "confidence_updated"]
        assert len(conf_updated) == 1
        assert conf_updated[0]["old_confidence"] == 0.3
        assert conf_updated[0]["new_confidence"] == updated.confidence
        assert conf_updated[0]["score_breakdown"] is not None
