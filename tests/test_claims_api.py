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

    def test_list_claims_sort_by_confidence_desc(self, client):
        """Sort by confidence descending (default order)."""
        # Create claims with different confidence via evidence
        claim1 = add_claim("Low confidence claim", "preference")
        claim2 = add_claim("High confidence claim", "preference")

        # Add evidence to claim2 to increase confidence
        add_evidence(
            claim2.claim_id,
            Evidence(
                evidence_id="ev_sort_001",
                source_type="transcript",
                source_id="conv_sort",
                quote="Test quote for sorting",
                support="supports",
                weight=0.9,
                retrieved_at="2026-01-01T00:00:00Z",
                retrieval_query="test",
            ),
        )

        response = client.get("/api/claims?sort=confidence&order=desc")
        assert response.status_code == 200
        data = response.json()
        # Higher confidence should come first
        assert data["items"][0]["claim_id"] == claim2.claim_id

    def test_list_claims_sort_by_confidence_asc(self, client):
        """Sort by confidence ascending."""
        claim1 = add_claim("Low confidence claim", "preference")
        claim2 = add_claim("High confidence claim", "preference")

        add_evidence(
            claim2.claim_id,
            Evidence(
                evidence_id="ev_sort_002",
                source_type="transcript",
                source_id="conv_sort",
                quote="Test quote",
                support="supports",
                weight=0.9,
                retrieved_at="2026-01-01T00:00:00Z",
                retrieval_query="test",
            ),
        )

        response = client.get("/api/claims?sort=confidence&order=asc")
        assert response.status_code == 200
        data = response.json()
        # Lower confidence should come first
        assert data["items"][0]["claim_id"] == claim1.claim_id

    def test_list_claims_sort_by_created_at(self, client):
        """Sort by created_at."""
        import time
        claim1 = add_claim("First claim", "preference")
        time.sleep(0.01)  # Ensure different timestamps
        claim2 = add_claim("Second claim", "preference")

        response = client.get("/api/claims?sort=created_at&order=desc")
        assert response.status_code == 200
        data = response.json()
        # Most recent should come first
        assert data["items"][0]["claim_id"] == claim2.claim_id

        response = client.get("/api/claims?sort=created_at&order=asc")
        data = response.json()
        # Oldest should come first
        assert data["items"][0]["claim_id"] == claim1.claim_id

    def test_list_claims_sort_by_claim_text(self, client):
        """Sort by claim_text alphabetically."""
        claim_z = add_claim("Zebra claim", "preference")
        claim_a = add_claim("Apple claim", "preference")

        response = client.get("/api/claims?sort=claim_text&order=asc")
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["claim_text"] == "Apple claim"

        response = client.get("/api/claims?sort=claim_text&order=desc")
        data = response.json()
        assert data["items"][0]["claim_text"] == "Zebra claim"

    def test_list_claims_sort_invalid_field(self, client):
        """Invalid sort field returns 422."""
        response = client.get("/api/claims?sort=invalid_field")
        assert response.status_code == 422

    def test_list_claims_sort_invalid_order(self, client):
        """Invalid order returns 422."""
        response = client.get("/api/claims?sort=confidence&order=invalid")
        assert response.status_code == 422

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
        """Update claim status (using force to bypass validation)."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted", "force": True},
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
                "force": True,  # Bypass transition rules for this test
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


# =============================================================================
# POST /api/claims/{claim_id}/adjudicate - Panel Adjudication
# =============================================================================


class TestAdjudicateClaim:
    """Tests for POST /api/claims/{claim_id}/adjudicate endpoint."""

    def test_adjudicate_claim_not_found(self, client):
        """Non-existent claim returns 404."""
        response = client.post(
            "/api/claims/nonexistent-id/adjudicate",
            json={"models": ["mock:model"], "panel_size": 1},
        )
        assert response.status_code == 404

    def test_adjudicate_claim_success(self, client, sample_claim_with_evidence):
        """Successful adjudication returns panel verdicts."""
        import json as json_module

        mock_response = {
            "content": json_module.dumps({
                "verdict": "accept",
                "confidence": 0.85,
                "reasoning": "Evidence supports the claim",
                "cited_evidence": ["ev_test_001"],
                "concerns": [],
            })
        }

        with patch("backend.adjudicator.query_model") as mock_query:
            # Make it async
            import asyncio
            async def async_return(*args, **kwargs):
                return mock_response
            mock_query.side_effect = async_return

            response = client.post(
                f"/api/claims/{sample_claim_with_evidence}/adjudicate",
                json={"models": ["mock:model1", "mock:model2"], "panel_size": 2},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["claim_id"] == sample_claim_with_evidence
            assert data["panel_size"] == 2
            assert len(data["panel_verdicts"]) == 2
            assert data["consensus_verdict"] == "accept"
            assert "claim" in data

    def test_adjudicate_claim_invalid_mode(self, client, sample_claim):
        """Invalid mode returns 422."""
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/adjudicate",
            json={"mode": "invalid_mode"},
        )
        assert response.status_code == 422

    def test_adjudicate_claim_panel_size_validation(self, client, sample_claim):
        """Panel size outside range returns 422."""
        response = client.post(
            f"/api/claims/{sample_claim.claim_id}/adjudicate",
            json={"panel_size": 10},  # Max is 7
        )
        assert response.status_code == 422

    def test_adjudicate_claim_no_models_configured(self, client, sample_claim):
        """Returns 422 when no models available."""
        with patch("backend.adjudicator.get_settings") as mock_settings:
            mock_settings.return_value = {"council_models": []}

            response = client.post(
                f"/api/claims/{sample_claim.claim_id}/adjudicate",
                json={},  # No models provided, and settings has none
            )
            assert response.status_code == 422
            assert "No models" in response.json()["detail"]

    def test_adjudicate_claim_with_status_update(self, client, sample_claim_with_evidence):
        """Adjudication can update claim status."""
        import json as json_module

        mock_response = {
            "content": json_module.dumps({
                "verdict": "accept",
                "confidence": 0.9,
                "reasoning": "Strong evidence",
                "cited_evidence": ["ev_test_001"],
                "concerns": [],
            })
        }

        with patch("backend.adjudicator.query_model") as mock_query:
            import asyncio
            async def async_return(*args, **kwargs):
                return mock_response
            mock_query.side_effect = async_return

            response = client.post(
                f"/api/claims/{sample_claim_with_evidence}/adjudicate",
                json={
                    "models": ["mock:model1", "mock:model2"],
                    "panel_size": 2,
                    "update_status": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status_updated"] is True
            assert data["new_status"] == "accepted"
            assert data["claim"]["status"] == "accepted"

    def test_adjudicate_records_in_history(self, client, sample_claim_with_evidence):
        """Adjudication is recorded in claim history."""
        import json as json_module

        mock_response = {
            "content": json_module.dumps({
                "verdict": "insufficient",
                "confidence": 0.4,
                "reasoning": "Not enough evidence",
                "cited_evidence": [],
                "concerns": ["Need more sources"],
            })
        }

        with patch("backend.adjudicator.query_model") as mock_query:
            import asyncio
            async def async_return(*args, **kwargs):
                return mock_response
            mock_query.side_effect = async_return

            response = client.post(
                f"/api/claims/{sample_claim_with_evidence}/adjudicate",
                json={"models": ["mock:model"], "panel_size": 1},
            )

            assert response.status_code == 200
            data = response.json()

            # Check history contains adjudication event
            history = data["claim"]["review_history"]
            adj_events = [h for h in history if h.get("event") == "panel_adjudication"]
            assert len(adj_events) == 1
            assert adj_events[0]["consensus_verdict"] == "insufficient"


# =============================================================================
# Status Transition Rules Tests
# =============================================================================


class TestStatusTransitionRules:
    """Tests for status transition validation rules."""

    def test_candidate_to_accepted_blocked_no_evidence(self, client, sample_claim):
        """Cannot accept claim without supporting evidence."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted"},
        )
        assert response.status_code == 409
        data = response.json()
        assert "detail" in data
        assert data["detail"]["current_status"] == "candidate"
        assert data["detail"]["target_status"] == "accepted"
        # Should mention either confidence or evidence requirement
        reason = data["detail"]["reason"].lower()
        assert "confidence" in reason or "evidence" in reason

    def test_candidate_to_accepted_blocked_low_confidence(self, client, sample_claim):
        """Cannot accept claim with insufficient confidence."""
        # Add weak evidence
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_weak",
                "quote": "Maybe Python is okay",
                "support": "supports",
                "weight": 0.3,  # Low weight
            },
        )

        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted"},
        )
        assert response.status_code == 409
        data = response.json()
        assert "confidence" in data["detail"]["reason"].lower()

    def test_candidate_to_accepted_succeeds_with_strong_evidence(self, client, sample_claim):
        """Can accept claim with sufficient evidence."""
        # Add strong supporting evidence from two independent sources
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_001",
                "quote": "I really love Python programming",
                "support": "supports",
                "weight": 0.9,
            },
        )
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "note",
                "source_id": "note_001",
                "quote": "Python is my favorite language",
                "support": "supports",
                "weight": 0.8,
            },
        )

        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

    def test_candidate_to_disputed_blocked_no_contradiction(self, client, sample_claim):
        """Cannot dispute claim without contradicting evidence."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "disputed"},
        )
        assert response.status_code == 409
        data = response.json()
        assert "contradict" in data["detail"]["reason"].lower()

    def test_candidate_to_disputed_succeeds_with_contradiction(self, client, sample_claim):
        """Can dispute claim with contradicting evidence."""
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_contra",
                "quote": "Python is actually not my thing",
                "support": "contradicts",
                "weight": 0.7,
            },
        )

        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "disputed"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "disputed"

    def test_to_deprecated_always_allowed(self, client, sample_claim):
        """Can always deprecate a claim (soft delete)."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "deprecated"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "deprecated"

    def test_disputed_to_candidate_always_allowed(self, client, sample_claim):
        """Can always return disputed claim to candidate for rework."""
        # First dispute the claim
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_contra",
                "quote": "Contradicting evidence",
                "support": "contradicts",
                "weight": 0.7,
            },
        )
        client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "disputed"},
        )

        # Now return to candidate
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "candidate"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "candidate"

    def test_disputed_to_accepted_requires_higher_confidence(self, client, sample_claim):
        """Re-accepting after dispute requires higher confidence (0.8 vs 0.7)."""
        # Add evidence and dispute
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_001",
                "quote": "Supporting but moderate evidence",
                "support": "supports",
                "weight": 0.8,
            },
        )
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_contra",
                "quote": "Some contradiction",
                "support": "contradicts",
                "weight": 0.5,
            },
        )
        # Force dispute to test re-acceptance
        client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "disputed", "force": True},
        )

        # Try to re-accept - should fail without enough confidence
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted"},
        )
        assert response.status_code == 409
        data = response.json()
        assert "0.8" in data["detail"]["reason"]  # Higher bar mentioned

    def test_force_bypasses_validation(self, client, sample_claim):
        """force=True bypasses transition rules."""
        # Try to accept without evidence - should fail normally
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted"},
        )
        assert response.status_code == 409

        # Now with force=True
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted", "force": True},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

    def test_force_recorded_in_history(self, client, sample_claim):
        """Forced transitions are recorded in review history."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={
                "status": "accepted",
                "force": True,
                "note": "Admin override for testing",
            },
        )
        assert response.status_code == 200
        history = response.json()["review_history"]

        # Should have status_changed event with forced=True
        status_events = [h for h in history if h.get("event") == "status_changed"]
        assert len(status_events) == 1
        assert status_events[0]["forced"] is True

    def test_transition_rules_in_error_response(self, client, sample_claim):
        """409 response includes transition rules for transparency."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted"},
        )
        assert response.status_code == 409
        data = response.json()
        assert "transition_rules" in data["detail"]
        assert "accept_min_confidence" in data["detail"]["transition_rules"]

    def test_accepted_to_disputed_requires_strong_contradiction(self, client, sample_claim):
        """Disputing an accepted claim requires strong contradiction."""
        # First accept the claim with force
        client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted", "force": True},
        )

        # Try to dispute without strong contradiction
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "disputed"},
        )
        assert response.status_code == 409
        assert "contradiction" in response.json()["detail"]["reason"].lower()

    def test_accepted_to_disputed_succeeds_with_strong_contradiction(self, client, sample_claim):
        """Can dispute accepted claim with strong contradiction."""
        # Accept with force
        client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "accepted", "force": True},
        )

        # Add strong contradiction
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_strong_contra",
                "quote": "Actually I hate Python now",
                "support": "contradicts",
                "weight": 0.8,
            },
        )

        # Now dispute should work
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "disputed"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "disputed"

    def test_same_status_is_noop(self, client, sample_claim):
        """Setting same status is allowed (no-op)."""
        response = client.patch(
            f"/api/claims/{sample_claim.claim_id}",
            json={"status": "candidate"},  # Already candidate
        )
        assert response.status_code == 200
        assert response.json()["status"] == "candidate"


# =============================================================================
# GET /api/claims/export - Export Claims Bundle
# =============================================================================


class TestExportClaims:
    """Tests for GET /api/claims/export endpoint."""

    def test_export_empty(self, client):
        """Export with no claims returns empty bundle."""
        response = client.get("/api/claims/export")
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["total_claims"] == 0
        assert data["metadata"]["total_evidence"] == 0
        assert data["metadata"]["format"] == "json"
        assert data["claims"] == []

    def test_export_with_claims(self, client, sample_claim_with_evidence):
        """Export includes claims with full evidence."""
        response = client.get("/api/claims/export")
        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["total_claims"] == 1
        assert data["metadata"]["total_evidence"] == 1
        assert len(data["claims"]) == 1

        # Check claim has full data
        claim = data["claims"][0]
        assert claim["claim_id"] == sample_claim_with_evidence
        assert claim["evidence"] is not None
        assert len(claim["evidence"]) == 1
        assert claim["review_history"] is not None

    def test_export_no_quote_truncation(self, client, sample_claim):
        """Export includes full quotes without truncation."""
        # Add evidence with very long quote
        long_quote = "A" * 1000
        client.post(
            f"/api/claims/{sample_claim.claim_id}/evidence",
            json={
                "source_type": "transcript",
                "source_id": "conv_long",
                "quote": long_quote,
                "support": "supports",
                "weight": 0.7,
            },
        )

        response = client.get("/api/claims/export")
        assert response.status_code == 200
        data = response.json()

        # Quote should NOT be truncated
        quote = data["claims"][0]["evidence"][0]["quote"]
        assert len(quote) == 1000
        assert "..." not in quote

    def test_export_filter_by_status(self, client):
        """Export respects status filter."""
        # Create claims with different statuses
        claim1 = add_claim("Candidate claim", "preference")
        claim2 = add_claim("Accepted claim", "preference")
        from backend.claims import update_claim
        update_claim(claim2.claim_id, status="accepted", force=True)

        response = client.get("/api/claims/export?status=candidate")
        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["total_claims"] == 1
        assert data["metadata"]["filters_applied"]["status"] == "candidate"
        assert data["claims"][0]["status"] == "candidate"

    def test_export_filter_by_type(self, client, sample_claim):
        """Export respects claim_type filter."""
        # sample_claim is type "preference"
        add_claim("Biographical claim", "biographical")

        response = client.get("/api/claims/export?claim_type=preference")
        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["total_claims"] == 1
        assert data["metadata"]["filters_applied"]["claim_type"] == "preference"
        assert data["claims"][0]["claim_type"] == "preference"

    def test_export_filter_by_min_confidence(self, client, sample_claim_with_evidence):
        """Export respects min_confidence filter."""
        add_claim("Low confidence claim", "preference")  # Default 0.3

        response = client.get("/api/claims/export?min_confidence=0.4")
        assert response.status_code == 200
        data = response.json()

        # Only the claim with evidence (higher confidence) should be included
        assert data["metadata"]["total_claims"] == 1
        assert data["metadata"]["filters_applied"]["min_confidence"] == 0.4

    def test_export_filter_by_substring(self, client, sample_claim):
        """Export respects substring search."""
        add_claim("JavaScript is great", "preference")

        response = client.get("/api/claims/export?q=Python")
        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["total_claims"] == 1
        assert data["metadata"]["filters_applied"]["q"] == "Python"
        assert "Python" in data["claims"][0]["claim_text"]

    def test_export_includes_transition_rules(self, client, sample_claim):
        """Export metadata includes transition rules."""
        response = client.get("/api/claims/export")
        assert response.status_code == 200
        data = response.json()

        rules = data["metadata"]["transition_rules"]
        assert "accept_min_confidence" in rules
        assert "accept_min_supporting" in rules
        assert "dispute_min_contradiction_weight" in rules

    def test_export_includes_score_breakdown(self, client, sample_claim_with_evidence):
        """Export includes score breakdown for claims with evidence."""
        response = client.get("/api/claims/export")
        assert response.status_code == 200
        data = response.json()

        claim = data["claims"][0]
        assert claim["score_breakdown"] is not None
        assert "base_score" in claim["score_breakdown"]
        assert "final_score" in claim["score_breakdown"]

    def test_export_multiple_claims(self, client):
        """Export handles multiple claims correctly."""
        # Create 5 claims
        for i in range(5):
            claim = add_claim(f"Claim {i}", "biographical")
            # Add evidence to some
            if i % 2 == 0:
                client.post(
                    f"/api/claims/{claim.claim_id}/evidence",
                    json={
                        "source_type": "note",
                        "source_id": f"note_{i}",
                        "quote": f"Evidence for claim {i}",
                        "support": "supports",
                        "weight": 0.6,
                    },
                )

        response = client.get("/api/claims/export")
        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["total_claims"] == 5
        assert data["metadata"]["total_evidence"] == 3  # Claims 0, 2, 4 have evidence
        assert len(data["claims"]) == 5

    def test_export_timestamp_format(self, client, sample_claim):
        """Export timestamp is valid ISO format."""
        response = client.get("/api/claims/export")
        assert response.status_code == 200
        data = response.json()

        # Should be parseable as ISO timestamp
        from datetime import datetime
        exported_at = data["metadata"]["exported_at"]
        datetime.fromisoformat(exported_at.replace("Z", "+00:00"))
