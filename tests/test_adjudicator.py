"""Tests for panel adjudication."""

import pytest
from unittest.mock import AsyncMock, patch
import json

from backend.adjudicator import (
    adjudicate_claim,
    _parse_verdict_json,
    _validate_verdict,
    _compute_consensus,
    PanelVerdict,
)
from backend.claims import add_claim, add_evidence, Evidence, clear_all_claims


@pytest.fixture(autouse=True)
def clear_claims():
    """Clear claims store before each test."""
    clear_all_claims()
    yield
    clear_all_claims()


class TestParseVerdictJson:
    """Tests for JSON parsing."""

    def test_parse_direct_json(self):
        response = '{"verdict": "accept", "confidence": 0.8, "reasoning": "test", "cited_evidence": []}'
        result = _parse_verdict_json(response)
        assert result is not None
        assert result["verdict"] == "accept"

    def test_parse_json_in_code_block(self):
        response = """Here is my verdict:
```json
{"verdict": "dispute", "confidence": 0.6, "reasoning": "contradicted", "cited_evidence": ["ev_1"]}
```
"""
        result = _parse_verdict_json(response)
        assert result is not None
        assert result["verdict"] == "dispute"

    def test_parse_json_with_prefix(self):
        response = 'Based on evidence: {"verdict": "insufficient", "confidence": 0.3, "reasoning": "not enough", "cited_evidence": []}'
        result = _parse_verdict_json(response)
        assert result is not None
        assert result["verdict"] == "insufficient"

    def test_parse_invalid_json(self):
        response = "This is not JSON at all"
        result = _parse_verdict_json(response)
        assert result is None


class TestValidateVerdict:
    """Tests for verdict schema validation."""

    def test_valid_verdict(self):
        parsed = {
            "verdict": "accept",
            "confidence": 0.8,
            "reasoning": "Strong evidence",
            "cited_evidence": ["ev_1", "ev_2"],
        }
        error = _validate_verdict(parsed, ["ev_1", "ev_2", "ev_3"])
        assert error is None

    def test_missing_field(self):
        parsed = {"verdict": "accept", "confidence": 0.8}
        error = _validate_verdict(parsed, [])
        assert error is not None
        assert "Missing required field" in error

    def test_invalid_verdict_value(self):
        parsed = {
            "verdict": "maybe",
            "confidence": 0.5,
            "reasoning": "unsure",
            "cited_evidence": [],
        }
        error = _validate_verdict(parsed, [])
        assert error is not None
        assert "Invalid verdict" in error

    def test_confidence_out_of_range(self):
        parsed = {
            "verdict": "accept",
            "confidence": 1.5,
            "reasoning": "too confident",
            "cited_evidence": [],
        }
        error = _validate_verdict(parsed, [])
        assert error is not None
        assert "out of range" in error

    def test_invalid_evidence_id(self):
        parsed = {
            "verdict": "accept",
            "confidence": 0.8,
            "reasoning": "cited wrong",
            "cited_evidence": ["ev_999"],
        }
        error = _validate_verdict(parsed, ["ev_1", "ev_2"])
        assert error is not None
        assert "Invalid evidence IDs" in error


class TestComputeConsensus:
    """Tests for consensus computation."""

    def test_unanimous_accept(self):
        verdicts = [
            PanelVerdict("m1", "accept", 0.9, "yes", [], [], None, None),
            PanelVerdict("m2", "accept", 0.8, "yes", [], [], None, None),
            PanelVerdict("m3", "accept", 0.85, "yes", [], [], None, None),
        ]
        verdict, confidence = _compute_consensus(verdicts)
        assert verdict == "accept"
        assert confidence > 0.8

    def test_majority_dispute(self):
        verdicts = [
            PanelVerdict("m1", "dispute", 0.7, "no", [], [], None, None),
            PanelVerdict("m2", "dispute", 0.8, "no", [], [], None, None),
            PanelVerdict("m3", "accept", 0.6, "yes", [], [], None, None),
        ]
        verdict, confidence = _compute_consensus(verdicts)
        assert verdict == "dispute"

    def test_no_majority(self):
        verdicts = [
            PanelVerdict("m1", "accept", 0.7, "yes", [], [], None, None),
            PanelVerdict("m2", "dispute", 0.7, "no", [], [], None, None),
            PanelVerdict("m3", "insufficient", 0.5, "maybe", [], [], None, None),
        ]
        verdict, confidence = _compute_consensus(verdicts)
        assert verdict == "insufficient"

    def test_all_abstain(self):
        verdicts = [
            PanelVerdict("m1", "abstain", 0.0, "", [], [], "error", None),
            PanelVerdict("m2", "abstain", 0.0, "", [], [], "error", None),
        ]
        verdict, confidence = _compute_consensus(verdicts)
        assert verdict == "insufficient"
        assert confidence == 0.0


class TestAdjudicateClaimIntegration:
    """Integration tests for adjudicate_claim."""

    @pytest.fixture
    def claim_with_evidence(self):
        """Create a claim with evidence."""
        claim = add_claim(
            claim_text="Jeremy prefers Python over JavaScript",
            claim_type="preference",
        )
        add_evidence(
            claim.claim_id,
            Evidence(
                evidence_id="ev_001",
                source_type="transcript",
                source_id="conv_1",
                quote="I really enjoy Python for data work",
                support="supports",
                weight=0.8,
                retrieved_at="2026-01-01T00:00:00Z",
                retrieval_query="python preference",
            ),
        )
        add_evidence(
            claim.claim_id,
            Evidence(
                evidence_id="ev_002",
                source_type="transcript",
                source_id="conv_2",
                quote="Python is my go-to language",
                support="supports",
                weight=0.9,
                retrieved_at="2026-01-01T00:00:00Z",
                retrieval_query="python preference",
            ),
        )
        return claim

    @pytest.mark.asyncio
    async def test_adjudicate_with_mocked_models(self, claim_with_evidence):
        """Test full adjudication flow with mocked model responses."""

        mock_response = {
            "content": json.dumps({
                "verdict": "accept",
                "confidence": 0.85,
                "reasoning": "Both evidence pieces support the claim",
                "cited_evidence": ["ev_001", "ev_002"],
                "concerns": [],
            })
        }

        with patch("backend.adjudicator.query_model", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_response

            result = await adjudicate_claim(
                claim_id=claim_with_evidence.claim_id,
                models=["mock:model1", "mock:model2", "mock:model3"],
                panel_size=3,
                mode="strict",
                update_status=False,
            )

            assert result.claim_id == claim_with_evidence.claim_id
            assert result.panel_size == 3
            assert result.consensus_verdict == "accept"
            assert len(result.panel_verdicts) == 3
            assert all(v.verdict == "accept" for v in result.panel_verdicts)

    @pytest.mark.asyncio
    async def test_adjudicate_updates_status(self, claim_with_evidence):
        """Test that adjudication updates status when enabled."""

        mock_response = {
            "content": json.dumps({
                "verdict": "accept",
                "confidence": 0.9,
                "reasoning": "Strong support",
                "cited_evidence": ["ev_001"],
                "concerns": [],
            })
        }

        with patch("backend.adjudicator.query_model", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_response

            result = await adjudicate_claim(
                claim_id=claim_with_evidence.claim_id,
                models=["mock:model1", "mock:model2"],
                panel_size=2,
                mode="strict",
                update_status=True,
            )

            assert result.status_updated is True
            assert result.new_status == "accepted"

    @pytest.mark.asyncio
    async def test_adjudicate_handles_model_error(self, claim_with_evidence):
        """Test handling of model errors."""

        with patch("backend.adjudicator.query_model", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {"error": True, "error_message": "Model unavailable"}

            result = await adjudicate_claim(
                claim_id=claim_with_evidence.claim_id,
                models=["mock:broken"],
                panel_size=1,
            )

            assert len(result.panel_verdicts) == 1
            assert result.panel_verdicts[0].error is not None
            assert result.consensus_verdict == "insufficient"

    @pytest.mark.asyncio
    async def test_adjudicate_retries_on_bad_json(self, claim_with_evidence):
        """Test that adjudication retries on malformed JSON."""

        call_count = 0

        async def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"content": "Not valid JSON"}
            else:
                return {
                    "content": json.dumps({
                        "verdict": "accept",
                        "confidence": 0.7,
                        "reasoning": "Fixed",
                        "cited_evidence": [],
                        "concerns": [],
                    })
                }

        with patch("backend.adjudicator.query_model", side_effect=mock_query):
            result = await adjudicate_claim(
                claim_id=claim_with_evidence.claim_id,
                models=["mock:retry_model"],
                panel_size=1,
            )

            # Should have retried and succeeded
            assert result.panel_verdicts[0].verdict == "accept"
            assert result.panel_verdicts[0].error is None

    @pytest.mark.asyncio
    async def test_adjudicate_claim_not_found(self):
        """Test error on non-existent claim."""

        with pytest.raises(ValueError, match="not found"):
            await adjudicate_claim(
                claim_id="nonexistent",
                models=["mock:model"],
            )

    @pytest.mark.asyncio
    async def test_adjudicate_records_history(self, claim_with_evidence):
        """Test that adjudication is recorded in review_history."""

        mock_response = {
            "content": json.dumps({
                "verdict": "insufficient",
                "confidence": 0.4,
                "reasoning": "Need more data",
                "cited_evidence": [],
                "concerns": ["Limited evidence"],
            })
        }

        with patch("backend.adjudicator.query_model", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_response

            await adjudicate_claim(
                claim_id=claim_with_evidence.claim_id,
                models=["mock:model"],
                panel_size=1,
            )

            # Check history was updated
            from backend.claims import get_claim
            updated = get_claim(claim_with_evidence.claim_id)

            # Find adjudication event in history
            adj_events = [e for e in updated.review_history if e.get("event") == "panel_adjudication"]
            assert len(adj_events) == 1
            assert adj_events[0]["consensus_verdict"] == "insufficient"
