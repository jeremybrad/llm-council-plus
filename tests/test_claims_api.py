"""Tests for Claims API endpoints.

Uses FastAPI TestClient for HTTP-level testing.
Tests request validation, response shaping, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from backend.main import app
from backend.claims import Claim, Evidence, clear_all_claims, add_claim, add_evidence
from backend.scorer import ScoreBreakdown


@pytest.fixture(autouse=True)
def clean_claims():
    """Clear all claims before each test."""
    clear_all_claims()
    yield
    clear_all_claims()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_claim():
    """Create a sample claim for testing."""
    return add_claim(
        claim_text="Jeremy prefers Python over JavaScript",
        claim_type="preference",
    )


@pytest.fixture
def sample_claim_with_evidence(sample_claim):
    """Create a claim with evidence attached."""
    evidence = Evidence(
        evidence_id="ev_test_001",
        source_type="transcript",
        source_id="conv_123",
        quote="I really enjoy writing Python code",
        support="supports",
        weight=0.8,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        retrieval_query="Jeremy Python preference",
        span_start=100,
        span_end=200,
        source_hash="abc123",
    )
    add_evidence(sample_claim.claim_id, evidence)
    return sample_claim.claim_id


# =============================================================================
# GET /api/claims - List Claims
# =============================================================================


class TestListClaims:
    """Tests for GET /api/claims endpoint."""

    def test_list_claims_empty(self, client):
        """Empty store returns empty list."""
        response = client.get("/api/claims")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_list_claims_with_data(self, client, sample_claim):
        """Returns claims in store."""
        response = client.get("/api/claims")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["claim_id"] == sample_claim.claim_id

    def test_list_claims_filter_by_status(self, client, sample_claim):
        """Filter by status."""
        response = client.get("/api/claims?status=candidate")
        assert response.status_code == 200
        assert response.json()["total"] == 1

        response = client.get("/api/claims?status=accepted")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_list_claims_filter_by_type(self, client, sample_claim):
        """Filter by claim type."""
        response = client.get("/api/claims?claim_type=preference")
        assert response.status_code == 200
        assert response.json()["total"] == 1

        response = client.get("/api/claims?claim_type=biographical")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_list_claims_filter_by_min_confidence(self, client, sample_claim_with_evidence):
        """Filter by minimum confidence."""
        response = client.get("/api/claims?min_confidence=0.4")
        assert response.status_code == 200
        assert response.json()["total"] == 1

        response = client.get("/api/claims?min_confidence=0.9")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_list_claims_substring_search(self, client, sample_claim):
        """Substring search on claim_text."""
        response = client.get("/api/claims?q=Python")
        assert response.status_code == 200
        assert response.json()["total"] == 1

        response = client.get("/api/claims?q=TypeScript")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_list_claims_pagination(self, client):
        """Pagination with limit and offset."""
        # Create 5 claims
        for i in range(5):
            add_claim(f"Claim {i}", "biographical")

        response = client.get("/api/claims?limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

        response = client.get("/api/claims?limit=2&offset=2")
        data = response.json()
        assert len(data["items"]) == 2
        assert data["offset"] == 2

    def test_list_claims_include_evidence(self, client, sample_claim_with_evidence):
        """Include evidence array when requested."""
        response = client.get("/api/claims?include_evidence=true")
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["evidence"] is not None
        assert len(data["items"][0]["evidence"]) == 1

        response = client.get("/api/claims?include_evidence=false")
        data = response.json()
        assert data["items"][0]["evidence"] is None

    def test_list_claims_include_history(self, client, sample_claim_with_evidence):
        """Include review_history when requested."""
        response = client.get("/api/claims?include_history=true")
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["review_history"] is not None

        response = client.get("/api/claims?include_history=false")
        data = response.json()
        assert data["items"][0]["review_history"] is None

    def test_list_claims_quote_truncation(self, client, sample_claim_with_evidence):
        """Quote truncation respects quote_max_len."""
        # Default truncation
        response = client.get("/api/claims?include_evidence=true")
        data = response.json()
        # Quote is short enough not to be truncated
        assert "..." not in data["items"][0]["evidence"][0]["quote"]

    def test_list_claims_invalid_status(self, client):
        """Invalid status returns 422."""
        response = client.get("/api/claims?status=invalid")
        assert response.status_code == 422

    def test_list_claims_invalid_claim_type(self, client):
        """Invalid claim_type returns 422."""
        response = client.get("/api/claims?claim_type=invalid")
        assert response.status_code == 422


# =============================================================================
# GET /api/claims/{claim_id} - Get Single Claim
# =============================================================================


class TestGetClaim:
    """Tests for GET /api/claims/{claim_id} endpoint."""

    def test_get_claim_success(self, client, sample_claim):
        """Get existing claim."""
        response = client.get(f"/api/claims/{sample_claim.claim_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["claim_id"] == sample_claim.claim_id
        assert data["claim_text"] == sample_claim.claim_text

    def test_get_claim_not_found(self, client):
        """Non-existent claim returns 404."""
        response = client.get("/api/claims/nonexistent-id")
        assert response.status_code == 404

    def test_get_claim_with_evidence(self, client, sample_claim_with_evidence):
        """Include evidence when requested."""
        response = client.get(f"/api/claims/{sample_claim_with_evidence}?include_evidence=true")
        assert response.status_code == 200
        data = response.json()
        assert data["evidence"] is not None
        assert len(data["evidence"]) == 1

    def test_get_claim_with_history(self, client, sample_claim_with_evidence):
        """Include history when requested."""
        response = client.get(f"/api/claims/{sample_claim_with_evidence}?include_history=true")
        assert response.status_code == 200
        data = response.json()
        assert data["review_history"] is not None


# =============================================================================
# POST /api/claims - Create Claim
# =============================================================================


class TestCreateClaim:
    """Tests for POST /api/claims endpoint."""

    def test_create_claim_minimal(self, client):
        """Create claim with required fields only."""
        response = client.post(
            "/api/claims",
            json={
                "claim_text": "New test claim",
                "claim_type": "biographical",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["claim_text"] == "New test claim"
        assert data["claim_type"] == "biographical"
        assert data["status"] == "candidate"
        assert data["confidence"] == 0.3

    def test_create_claim_with_temporal_fields(self, client):
        """Create claim with temporal validity."""
        response = client.post(
            "/api/claims",
            json={
                "claim_text": "Jeremy worked at Company X",
                "claim_type": "event",
                "as_of": "2024-01-01T00:00:00Z",
                "valid_from": "2020-01-01T00:00:00Z",
                "valid_until": "2023-12-31T00:00:00Z",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["valid_from"] == "2020-01-01T00:00:00Z"
        assert data["valid_until"] == "2023-12-31T00:00:00Z"

    def test_create_claim_returns_evidence_and_history(self, client):
        """Response includes evidence and history arrays."""
        response = client.post(
            "/api/claims",
            json={
                "claim_text": "Test claim",
                "claim_type": "biographical",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["evidence"] == []
        assert data["review_history"] is not None

    def test_create_claim_empty_text(self, client):
        """Empty claim_text returns 422."""
        response = client.post(
            "/api/claims",
            json={
                "claim_text": "",
                "claim_type": "biographical",
            },
        )
        assert response.status_code == 422

    def test_create_claim_invalid_type(self, client):
        """Invalid claim_type returns 422."""
        response = client.post(
            "/api/claims",
            json={
                "claim_text": "Test claim",
                "claim_type": "invalid_type",
            },
        )
        assert response.status_code == 422


# =============================================================================
# PATCH /api/claims/{claim_id} - Update Claim
# =============================================================================


class TestUpdateClaim:
    """Tests for PATCH /api/claims/{claim_id} endpoint."""

    def test_update_claim_status(self, client, sample_claim):
        """Update claim status."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"

    def test_update_claim_temporal_fields(self, client, sample_claim):
        """Update temporal fields."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={
                "valid_from": "2020-01-01T00:00:00Z",
                "valid_until": "2025-01-01T00:00:00Z",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid_from"] == "2020-01-01T00:00:00Z"
        assert data["valid_until"] == "2025-01-01T00:00:00Z"

    def test_update_claim_with_note(self, client, sample_claim):
        """Update with note adds to review_history."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={
                "status": "disputed",
                "note": "Found contradicting evidence",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disputed"
        # Note should be in review_history
        assert any("Found contradicting evidence" in str(h) for h in data["review_history"])

    def test_update_claim_not_found(self, client):
        """Non-existent claim returns 404."""
        response = client.patch(
            "/api/claims/nonexistent-id",
            json={"status": "accepted"},
        )
        assert response.status_code == 404

    def test_update_claim_invalid_status(self, client, sample_claim):
        """Invalid status returns 422."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "invalid_status"},
        )
        assert response.status_code == 422

    def test_update_claim_no_changes(self, client, sample_claim):
        """Empty update returns current claim."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["claim_id"] == sample_claim.claim_id


# =============================================================================
# POST /api/claims/{claim_id}/evidence - Add Evidence
# =============================================================================


class TestAddEvidence:
    """Tests for POST /api/claims/{claim_id}/evidence endpoint."""

    def test_add_evidence_success(self, client, sample_claim):
        """Add evidence to claim."""
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_456",
                "quote": "I love using Python for data science",
                "support": "supports",
                "weight": 0.9,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["evidence_count"] == 1
        assert data["confidence"] > 0.3  # Should increase with supporting evidence

    def test_add_evidence_updates_confidence(self, client, sample_claim):
        """Adding evidence recalculates confidence."""
        initial_confidence = sample_claim.confidence

        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_789",
                "quote": "Python is my favorite language",
                "support": "supports",
                "weight": 0.8,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] > initial_confidence

    def test_add_evidence_includes_response_fields(self, client, sample_claim):
        """Response includes evidence and history."""
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "note",
                "source_id": "note_001",
                "quote": "Test quote",
                "support": "neutral",
                "weight": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["evidence"] is not None
        assert data["review_history"] is not None

    def test_add_evidence_not_found(self, client):
        """Non-existent claim returns 404."""
        response = client.post(
            "/api/claims/nonexistent-id/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_001",
                "quote": "Test",
                "support": "supports",
                "weight": 0.5,
            },
        )
        assert response.status_code == 404

    def test_add_evidence_invalid_source_type(self, client, sample_claim):
        """Invalid source_type returns 422."""
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "invalid_type",
                "source_id": "test",
                "quote": "Test",
                "support": "supports",
                "weight": 0.5,
            },
        )
        assert response.status_code == 422

    def test_add_evidence_invalid_support(self, client, sample_claim):
        """Invalid support returns 422."""
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "test",
                "quote": "Test",
                "support": "invalid",
                "weight": 0.5,
            },
        )
        assert response.status_code == 422

    def test_add_evidence_weight_out_of_range(self, client, sample_claim):
        """Weight outside 0-1 returns 422."""
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "test",
                "quote": "Test",
                "support": "supports",
                "weight": 1.5,  # Out of range
            },
        )
        assert response.status_code == 422


# =============================================================================
# POST /api/claims/{claim_id}/validate - Single-Claim Validation
# =============================================================================


class TestValidateClaim:
    """Tests for POST /api/claims/{claim_id}/validate endpoint."""

    def test_validate_claim_not_found(self, client):
        """Non-existent claim returns 404."""
        response = client.post("/api/claims/nonexistent-id/validate")
        assert response.status_code == 404

    def test_validate_claim_sadb_unavailable(self, client, sample_claim):
        """Returns gracefully when SADB unavailable."""
        with patch("backend.api.claims.get_sadb_status") as mock_status:
            mock_status.return_value = {"available": False}

            response = client.post(f"/api/claims/{sample_claim.claim_id}/validate")
            assert response.status_code == 200
            data = response.json()
            assert data["sadb_available"] is False
            assert data["evidence_added"] == 0
            assert data["old_confidence"] == data["new_confidence"]

    def test_validate_claim_with_sadb(self, client, sample_claim):
        """Validation adds evidence when SADB available."""
        mock_evidence = Evidence(
            evidence_id="ev_sadb_001",
            source_type="transcript",
            source_id="conv_sadb_123",
            quote="I prefer Python",
            support="neutral",
            weight=0.7,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            retrieval_query="test query",
        )

        with patch("backend.api.claims.get_sadb_status") as mock_status, \
             patch("backend.api.claims.search_evidence") as mock_search, \
             patch("backend.api.claims.classify_support") as mock_classify:

            mock_status.return_value = {"available": True}
            mock_search.return_value = [mock_evidence]
            mock_classify.return_value = "supports"

            response = client.post(f"/api/claims/{sample_claim.claim_id}/validate")
            assert response.status_code == 200
            data = response.json()
            assert data["sadb_available"] is True
            assert data["evidence_added"] == 1
            assert data["claim"] is not None


# =============================================================================
# Response Shaping Tests
# =============================================================================


class TestResponseShaping:
    """Tests for response shaping controls."""

    def test_evidence_quote_truncation(self, client, sample_claim):
        """Long quotes are truncated."""
        # Add evidence with long quote
        long_quote = "A" * 500  # Longer than default 280
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "test",
                "quote": long_quote,
                "support": "supports",
                "weight": 0.5,
            },
        )
        assert response.status_code == 200

        # Get with default truncation
        response = client.get(
            f"/api/claims/{sample_claim.claim_id}?include_evidence=true"
        )
        data = response.json()
        quote = data["evidence"][0]["quote"]
        assert len(quote) <= 283  # 280 + "..."
        assert quote.endswith("...")

    def test_custom_quote_truncation_length(self, client, sample_claim):
        """Custom quote_max_len is respected."""
        long_quote = "A" * 500
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "test",
                "quote": long_quote,
                "support": "supports",
                "weight": 0.5,
            },
        )

        # Get with custom truncation
        response = client.get(
            f"/api/claims/{sample_claim.claim_id}?include_evidence=true&quote_max_len=100"
        )
        data = response.json()
        quote = data["evidence"][0]["quote"]
        assert len(quote) <= 103  # 100 + "..."

    def test_score_breakdown_included(self, client, sample_claim_with_evidence):
        """Score breakdown is included in response."""
        response = client.get(f"/api/claims/{sample_claim_with_evidence}")
        assert response.status_code == 200
        data = response.json()
        assert data["score_breakdown"] is not None
        assert "base_score" in data["score_breakdown"]
        assert "final_score" in data["score_breakdown"]
        assert "cap_applied" in data["score_breakdown"]
