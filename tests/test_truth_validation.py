"""Tests for the truth validation Night Shift job.

Tests cover:
- Preflight information generation
- Idempotent execution
- Checkpoint-based resumability
- Smart disputed logic
- Report generation
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from backend.jobs.truth_validation import TruthValidationJob, ValidationCheckpoint
from backend.jobs.base import JobStatus
from backend.preflight import BudgetConfig


@pytest.fixture
def temp_job_dirs(tmp_path):
    """Create temporary directories for job execution."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    claims_dir = data_dir / "claims"
    claims_dir.mkdir()
    history_dir = claims_dir / "history"
    history_dir.mkdir()

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    return {
        "data_dir": data_dir,
        "claims_dir": claims_dir,
        "reports_dir": reports_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def mock_claims_module(temp_job_dirs):
    """Mock the claims module functions."""
    claims_dir = temp_job_dirs["claims_dir"]

    # Patch claims module paths
    with patch.multiple(
        "backend.claims",
        CLAIMS_DIR=claims_dir,
        CLAIMS_FILE=claims_dir / "claims.json",
        LOCK_FILE=claims_dir / "claims.json.lock",
        HISTORY_DIR=claims_dir / "history",
        DATA_DIR=temp_job_dirs["data_dir"],
    ):
        yield


@pytest.fixture
def mock_evidence_search():
    """Mock evidence search to return deterministic results."""

    async def mock_search(query: str, top_k: int = 5):
        """Return mock evidence based on query."""
        from backend.claims import Evidence

        return [
            Evidence(
                evidence_id=f"mock_ev_{i}",
                source_type="transcript",
                source_id=f"conv_{i}",
                quote=f"Mock quote {i} for: {query[:50]}",
                support="neutral",  # Will be classified by scorer
                weight=0.7,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                retrieval_query=query,
            )
            for i in range(min(top_k, 2))
        ]

    return mock_search


@pytest.fixture
def mock_sadb_status():
    """Mock SADB status check."""
    return {"available": True, "sadb_path": "/mock/path", "sadb_path_exists": True}


class TestValidationCheckpoint:
    """Tests for ValidationCheckpoint dataclass."""

    def test_checkpoint_serialization(self):
        """Checkpoint should serialize to/from dict."""
        checkpoint = ValidationCheckpoint(
            job_run_id="val_20260111_120000",
            claims_to_process=["claim1", "claim2", "claim3"],
            claims_processed=["claim1"],
            started_at="2026-01-11T12:00:00+00:00",
            last_updated="2026-01-11T12:05:00+00:00",
        )

        as_dict = checkpoint.to_dict()
        restored = ValidationCheckpoint.from_dict(as_dict)

        assert restored.job_run_id == checkpoint.job_run_id
        assert restored.claims_to_process == checkpoint.claims_to_process
        assert restored.claims_processed == checkpoint.claims_processed

    def test_checkpoint_tracks_progress(self):
        """Checkpoint should correctly track remaining work."""
        checkpoint = ValidationCheckpoint(
            job_run_id="val_test",
            claims_to_process=["a", "b", "c", "d", "e"],
            claims_processed=["a", "b"],
            started_at="2026-01-11T00:00:00+00:00",
            last_updated="2026-01-11T00:00:00+00:00",
        )

        remaining = [c for c in checkpoint.claims_to_process if c not in checkpoint.claims_processed]
        assert remaining == ["c", "d", "e"]


class TestJobPreflight:
    """Tests for preflight information."""

    def test_preflight_returns_info(
        self, temp_job_dirs, mock_claims_module, mock_sadb_status
    ):
        """Preflight should return valid PreflightInfo."""
        with patch("backend.jobs.truth_validation.get_sadb_status", return_value=mock_sadb_status):
            with patch("backend.jobs.truth_validation.get_claims_for_review", return_value=[]):
                job = TruthValidationJob(
                    job_name="truth_validation",
                    data_dir=temp_job_dirs["data_dir"],
                    reports_dir=temp_job_dirs["reports_dir"],
                )

                info = job.get_preflight_info()

                assert info.job_name == "truth_validation"
                assert "claims_to_review" in info.input_counts
                assert "sadb_available" in info.input_counts
                assert len(info.output_paths) == 2  # MD and JSON

    def test_preflight_detects_sadb_unavailable(
        self, temp_job_dirs, mock_claims_module
    ):
        """Preflight should warn when SADB is unavailable."""
        sadb_unavailable = {
            "available": False,
            "sadb_path": "/mock/path",
            "error": "SADB module could not be imported",
        }

        with patch("backend.jobs.truth_validation.get_sadb_status", return_value=sadb_unavailable):
            with patch("backend.jobs.truth_validation.get_claims_for_review", return_value=[]):
                job = TruthValidationJob(
                    job_name="truth_validation",
                    data_dir=temp_job_dirs["data_dir"],
                    reports_dir=temp_job_dirs["reports_dir"],
                )

                info = job.get_preflight_info()

                assert info.input_counts["sadb_available"] == 0
                assert any("SADB unavailable" in w for w in info.warnings)

    def test_preflight_detects_resume(self, temp_job_dirs, mock_claims_module, mock_sadb_status):
        """Preflight should detect existing checkpoint for resume."""
        # Create a checkpoint file
        checkpoint_path = temp_job_dirs["data_dir"] / "truth_validation_checkpoint.json"
        checkpoint = ValidationCheckpoint(
            job_run_id="val_previous",
            claims_to_process=["c1", "c2", "c3"],
            claims_processed=["c1"],
            started_at="2026-01-10T00:00:00+00:00",
            last_updated="2026-01-10T00:05:00+00:00",
        )
        checkpoint_path.write_text(json.dumps(checkpoint.to_dict()))

        with patch("backend.jobs.truth_validation.get_sadb_status", return_value=mock_sadb_status):
            job = TruthValidationJob(
                job_name="truth_validation",
                data_dir=temp_job_dirs["data_dir"],
                reports_dir=temp_job_dirs["reports_dir"],
            )

            info = job.get_preflight_info()

            assert any("Resuming from checkpoint" in w for w in info.warnings)
            assert info.input_counts["claims_to_review"] == 2  # 3 - 1 processed


class TestJobExecution:
    """Tests for job execution."""

    @pytest.mark.asyncio
    async def test_execute_with_no_claims(
        self, temp_job_dirs, mock_claims_module, mock_sadb_status
    ):
        """Job should complete successfully with no claims to review."""
        with patch("backend.jobs.truth_validation.get_sadb_status", return_value=mock_sadb_status):
            with patch("backend.jobs.truth_validation.get_claims_for_review", return_value=[]):
                job = TruthValidationJob(
                    job_name="truth_validation",
                    data_dir=temp_job_dirs["data_dir"],
                    reports_dir=temp_job_dirs["reports_dir"],
                )

                result = await job.execute()

                assert result.status == JobStatus.COMPLETED
                assert result.tasks_completed == 0
                assert "0 promoted" in result.summary

    @pytest.mark.asyncio
    async def test_execute_creates_reports(
        self, temp_job_dirs, mock_claims_module, mock_sadb_status
    ):
        """Job should create both MD and JSON reports."""
        with patch("backend.jobs.truth_validation.get_sadb_status", return_value=mock_sadb_status):
            with patch("backend.jobs.truth_validation.get_claims_for_review", return_value=[]):
                job = TruthValidationJob(
                    job_name="truth_validation",
                    data_dir=temp_job_dirs["data_dir"],
                    reports_dir=temp_job_dirs["reports_dir"],
                )

                result = await job.execute()

                assert len(result.output_paths) == 2
                for path in result.output_paths:
                    assert Path(path).exists()

                # Check MD report
                md_path = [p for p in result.output_paths if p.endswith(".md")][0]
                md_content = Path(md_path).read_text()
                assert "Truth Validation Report" in md_content

                # Check JSON report
                json_path = [p for p in result.output_paths if p.endswith(".json")][0]
                json_content = json.loads(Path(json_path).read_text())
                assert "summary" in json_content
                assert "promoted" in json_content

    @pytest.mark.asyncio
    async def test_execute_clears_checkpoint_on_success(
        self, temp_job_dirs, mock_claims_module, mock_sadb_status
    ):
        """Job should clear checkpoint file after successful completion."""
        checkpoint_path = temp_job_dirs["data_dir"] / "truth_validation_checkpoint.json"

        with patch("backend.jobs.truth_validation.get_sadb_status", return_value=mock_sadb_status):
            with patch("backend.jobs.truth_validation.get_claims_for_review", return_value=[]):
                job = TruthValidationJob(
                    job_name="truth_validation",
                    data_dir=temp_job_dirs["data_dir"],
                    reports_dir=temp_job_dirs["reports_dir"],
                )

                result = await job.execute()

                assert result.status == JobStatus.COMPLETED
                assert not checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_execute_respects_budget(
        self, temp_job_dirs, mock_claims_module, mock_sadb_status, mock_evidence_search
    ):
        """Job should stop when budget is exceeded."""
        # Create mock claims
        from backend.claims import Claim

        mock_claims = [
            MagicMock(
                claim_id=f"claim_{i}",
                claim_text=f"Test claim {i}",
                confidence=0.7,
                evidence=[],
            )
            for i in range(10)
        ]

        budget_config = BudgetConfig(max_tasks_per_run=2)  # Only allow 2 tasks

        with patch("backend.jobs.truth_validation.get_sadb_status", return_value=mock_sadb_status):
            with patch("backend.jobs.truth_validation.get_claims_for_review", return_value=mock_claims):
                with patch("backend.jobs.truth_validation.search_evidence", mock_evidence_search):
                    with patch("backend.jobs.truth_validation.get_claim") as mock_get:
                        mock_get.return_value = mock_claims[0]
                        with patch("backend.jobs.truth_validation.add_evidence"):
                            job = TruthValidationJob(
                                job_name="truth_validation",
                                data_dir=temp_job_dirs["data_dir"],
                                reports_dir=temp_job_dirs["reports_dir"],
                                budget_config=budget_config,
                            )

                            result = await job.execute()

                            # Should stop at budget limit
                            assert result.status == JobStatus.BUDGET_EXCEEDED
                            assert result.tasks_completed <= 2


class TestCheckpointManagement:
    """Tests for checkpoint save/load/clear."""

    def test_save_and_load_checkpoint(self, temp_job_dirs, mock_claims_module):
        """Checkpoint should round-trip correctly."""
        job = TruthValidationJob(
            job_name="truth_validation",
            data_dir=temp_job_dirs["data_dir"],
            reports_dir=temp_job_dirs["reports_dir"],
        )

        checkpoint = ValidationCheckpoint(
            job_run_id="test_run",
            claims_to_process=["a", "b", "c"],
            claims_processed=["a"],
            started_at="2026-01-11T00:00:00+00:00",
            last_updated="2026-01-11T00:00:00+00:00",
        )

        job._save_checkpoint(checkpoint)
        loaded = job._load_checkpoint()

        assert loaded is not None
        assert loaded.job_run_id == checkpoint.job_run_id
        assert loaded.claims_to_process == checkpoint.claims_to_process
        assert loaded.claims_processed == checkpoint.claims_processed

    def test_clear_checkpoint(self, temp_job_dirs, mock_claims_module):
        """Clear should remove checkpoint file."""
        job = TruthValidationJob(
            job_name="truth_validation",
            data_dir=temp_job_dirs["data_dir"],
            reports_dir=temp_job_dirs["reports_dir"],
        )

        checkpoint = ValidationCheckpoint(
            job_run_id="test_run",
            claims_to_process=["a"],
            claims_processed=[],
            started_at="2026-01-11T00:00:00+00:00",
            last_updated="2026-01-11T00:00:00+00:00",
        )

        job._save_checkpoint(checkpoint)
        assert job._get_checkpoint_path().exists()

        job._clear_checkpoint()
        assert not job._get_checkpoint_path().exists()

    def test_load_returns_none_without_checkpoint(self, temp_job_dirs, mock_claims_module):
        """Load should return None when no checkpoint exists."""
        job = TruthValidationJob(
            job_name="truth_validation",
            data_dir=temp_job_dirs["data_dir"],
            reports_dir=temp_job_dirs["reports_dir"],
        )

        loaded = job._load_checkpoint()
        assert loaded is None


class TestReportGeneration:
    """Tests for report generation."""

    def test_markdown_report_format(self, temp_job_dirs, mock_claims_module):
        """Markdown report should have correct structure."""
        job = TruthValidationJob(
            job_name="truth_validation",
            data_dir=temp_job_dirs["data_dir"],
            reports_dir=temp_job_dirs["reports_dir"],
        )

        # Set up checkpoint for report
        job.checkpoint = ValidationCheckpoint(
            job_run_id="test_run",
            claims_to_process=["a", "b", "c"],
            claims_processed=["a", "b", "c"],
            started_at="2026-01-11T00:00:00+00:00",
            last_updated="2026-01-11T00:00:00+00:00",
        )

        promoted = [{"claim_id": "claim_1", "old": 0.65, "new": 0.80}]
        disputed = [{"claim_id": "claim_2", "reason": "strong_independent_contradiction"}]
        unchanged = ["claim_3"]

        report = job._generate_markdown_report(promoted, disputed, unchanged)

        assert "# Truth Validation Report" in report
        assert "## Summary" in report
        assert "## Promoted Claims" in report
        assert "## Disputed Claims" in report
        assert "claim_1" in report
        assert "claim_2" in report
        assert "0.65" in report and "0.80" in report

    def test_json_report_structure(self, temp_job_dirs, mock_claims_module):
        """JSON report should have correct structure."""
        job = TruthValidationJob(
            job_name="truth_validation",
            data_dir=temp_job_dirs["data_dir"],
            reports_dir=temp_job_dirs["reports_dir"],
        )

        job.checkpoint = ValidationCheckpoint(
            job_run_id="test_run",
            claims_to_process=["a"],
            claims_processed=["a"],
            started_at="2026-01-11T00:00:00+00:00",
            last_updated="2026-01-11T00:00:00+00:00",
        )

        promoted = [{"claim_id": "c1", "old": 0.6, "new": 0.8}]
        disputed = []
        unchanged = []

        report = job._generate_json_report(promoted, disputed, unchanged)

        assert report["job_name"] == "truth_validation"
        assert report["job_run_id"] == "test_run"
        assert report["summary"]["promoted"] == 1
        assert report["summary"]["disputed"] == 0
        assert report["promoted"] == promoted
        assert "budget" in report
