"""Tests for Night Shift runner and jobs."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.preflight import (
    PreflightInfo,
    BudgetConfig,
    BudgetTracker,
    verify_repo_root,
)
from backend.job_registry import JOB_REGISTRY, register_job, get_available_jobs, get_job_class
from backend.jobs.base import BaseJob, JobResult, JobStatus


class TestPreflightInfo:
    """Test PreflightInfo class."""

    def test_estimated_runtime_seconds(self):
        """Test runtime display for seconds."""
        preflight = PreflightInfo(
            job_name="test",
            estimated_runtime_seconds=45
        )
        assert preflight.estimated_runtime_display() == "45s"

    def test_estimated_runtime_minutes(self):
        """Test runtime display for minutes."""
        preflight = PreflightInfo(
            job_name="test",
            estimated_runtime_seconds=125
        )
        assert preflight.estimated_runtime_display() == "2m 5s"

    def test_estimated_runtime_hours(self):
        """Test runtime display for hours."""
        preflight = PreflightInfo(
            job_name="test",
            estimated_runtime_seconds=3725
        )
        assert preflight.estimated_runtime_display() == "1h 2m"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        preflight = PreflightInfo(
            job_name="test_job",
            input_counts={"files": 10},
            estimated_runtime_seconds=60,
            target_device="cpu",
            output_paths=["output.txt"],
            budget_limits={"max_tasks": 100},
            warnings=["Warning 1"],
        )
        result = preflight.to_dict()

        assert result["job_name"] == "test_job"
        assert result["input_counts"] == {"files": 10}
        assert result["estimated_runtime_seconds"] == 60
        assert result["estimated_runtime_display"] == "1m 0s"
        assert result["target_device"] == "cpu"
        assert result["output_paths"] == ["output.txt"]
        assert result["warnings"] == ["Warning 1"]


class TestBudgetConfig:
    """Test BudgetConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BudgetConfig()
        assert config.max_tasks_per_run == 100
        assert config.max_tokens_total == 1_000_000
        assert config.max_spend_usd is None
        assert config.timeout_seconds == 3600

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "max_tasks_per_run": 50,
            "max_tokens_total": 500_000,
            "max_spend_usd": 10.0,
            "timeout_seconds": 1800,
        }
        config = BudgetConfig.from_dict(data)

        assert config.max_tasks_per_run == 50
        assert config.max_tokens_total == 500_000
        assert config.max_spend_usd == 10.0
        assert config.timeout_seconds == 1800

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BudgetConfig(
            max_tasks_per_run=25,
            max_tokens_total=250_000,
        )
        result = config.to_dict()

        assert result["max_tasks_per_run"] == 25
        assert result["max_tokens_total"] == 250_000


class TestBudgetTracker:
    """Test BudgetTracker class."""

    def test_record_task(self):
        """Test recording task completion."""
        config = BudgetConfig(max_tasks_per_run=10)
        tracker = BudgetTracker(config=config)

        tracker.record_task(tokens=100, cost_usd=0.01)
        tracker.record_task(tokens=200, cost_usd=0.02)

        assert tracker.tasks_completed == 2
        assert tracker.tokens_used == 300
        assert tracker.spend_usd == 0.03

    def test_is_within_budget_tasks(self):
        """Test budget check for max tasks."""
        config = BudgetConfig(max_tasks_per_run=2)
        tracker = BudgetTracker(config=config)

        tracker.record_task()
        is_ok, reason = tracker.is_within_budget()
        assert is_ok is True

        tracker.record_task()
        is_ok, reason = tracker.is_within_budget()
        assert is_ok is False
        assert "max_tasks_per_run" in reason

    def test_is_within_budget_tokens(self):
        """Test budget check for max tokens."""
        config = BudgetConfig(max_tokens_total=500)
        tracker = BudgetTracker(config=config)

        tracker.record_task(tokens=400)
        is_ok, reason = tracker.is_within_budget()
        assert is_ok is True

        tracker.record_task(tokens=200)
        is_ok, reason = tracker.is_within_budget()
        assert is_ok is False
        assert "max_tokens_total" in reason

    def test_is_within_budget_spend(self):
        """Test budget check for max spend."""
        config = BudgetConfig(max_spend_usd=1.0)
        tracker = BudgetTracker(config=config)

        tracker.record_task(cost_usd=0.50)
        is_ok, reason = tracker.is_within_budget()
        assert is_ok is True

        tracker.record_task(cost_usd=0.60)
        is_ok, reason = tracker.is_within_budget()
        assert is_ok is False
        assert "max_spend_usd" in reason

    def test_summary(self):
        """Test budget summary."""
        config = BudgetConfig(max_tasks_per_run=10, max_tokens_total=1000)
        tracker = BudgetTracker(config=config)
        tracker.start()
        tracker.record_task(tokens=100)

        summary = tracker.summary()
        assert summary["tasks_completed"] == 1
        assert summary["tasks_limit"] == 10
        assert summary["tokens_used"] == 100
        assert summary["tokens_limit"] == 1000


class TestVerifyRepoRoot:
    """Test repo root verification."""

    def test_valid_repo_root(self):
        """Test with valid repo root."""
        # Get actual repo root
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        actual_root = result.stdout.strip()

        is_valid, pwd, expected, actual = verify_repo_root(actual_root)
        assert is_valid is True
        assert actual == actual_root

    def test_invalid_repo_root(self):
        """Test with mismatched repo root."""
        is_valid, pwd, expected, actual = verify_repo_root("/nonexistent/path")
        assert is_valid is False
        assert expected == "/nonexistent/path"


class TestJobRegistry:
    """Test job registry functions."""

    def test_register_job(self):
        """Test job registration via decorator."""
        # Clear any existing test registrations
        test_job_name = "test_registry_job"
        if test_job_name in JOB_REGISTRY:
            del JOB_REGISTRY[test_job_name]

        @register_job(test_job_name)
        class TestJob(BaseJob):
            def get_preflight_info(self):
                return PreflightInfo(job_name=self.job_name)

            async def execute(self):
                return JobResult(
                    job_name=self.job_name,
                    status=JobStatus.COMPLETED,
                    started_at=datetime.now()
                )

        assert test_job_name in get_available_jobs()
        assert get_job_class(test_job_name) == TestJob

        # Cleanup
        del JOB_REGISTRY[test_job_name]

    def test_repo_docs_refresh_registered(self):
        """Test that repo_docs_refresh job is registered."""
        assert "repo_docs_refresh" in get_available_jobs()


class TestJobResult:
    """Test JobResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        started = datetime(2024, 1, 15, 10, 0, 0)
        completed = datetime(2024, 1, 15, 10, 5, 0)

        result = JobResult(
            job_name="test_job",
            status=JobStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
            tasks_completed=10,
            tasks_failed=2,
            output_paths=["output.txt"],
            summary="Test summary",
        )
        data = result.to_dict()

        assert data["job_name"] == "test_job"
        assert data["status"] == "completed"
        assert data["tasks_completed"] == 10
        assert data["tasks_failed"] == 2
        assert data["duration_seconds"] == 300.0

    def test_mark_completed(self):
        """Test marking job as completed."""
        result = JobResult(
            job_name="test",
            status=JobStatus.RUNNING,
            started_at=datetime.now()
        )
        result.mark_completed()

        assert result.status == JobStatus.COMPLETED
        assert result.completed_at is not None

    def test_mark_failed(self):
        """Test marking job as failed."""
        result = JobResult(
            job_name="test",
            status=JobStatus.RUNNING,
            started_at=datetime.now()
        )
        result.mark_failed("Something went wrong")

        assert result.status == JobStatus.FAILED
        assert "Something went wrong" in result.errors

    def test_mark_budget_exceeded(self):
        """Test marking job as budget exceeded."""
        result = JobResult(
            job_name="test",
            status=JobStatus.RUNNING,
            started_at=datetime.now()
        )
        result.mark_budget_exceeded("max_tasks reached")

        assert result.status == JobStatus.BUDGET_EXCEEDED
        assert any("max_tasks" in e for e in result.errors)


class TestRepoDocsRefreshJob:
    """Test RepoDocsRefreshJob."""

    def test_preflight_info(self, tmp_path):
        """Test preflight info generation."""
        from backend.jobs.repo_docs_refresh import RepoDocsRefreshJob

        # Create test structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        reports_dir = tmp_path / "reports"

        # Create a test markdown file
        (tmp_path / "README.md").write_text("# Test\n\nTest content")
        (tmp_path / "docs.md").write_text("# Docs\n\nMore content")

        job = RepoDocsRefreshJob(
            job_name="repo_docs_refresh",
            data_dir=data_dir,
            reports_dir=reports_dir,
        )

        preflight = job.get_preflight_info()

        assert preflight.job_name == "repo_docs_refresh"
        assert preflight.input_counts["documentation_files"] == 2

    @pytest.mark.asyncio
    async def test_execute(self, tmp_path):
        """Test job execution."""
        from backend.jobs.repo_docs_refresh import RepoDocsRefreshJob

        # Create test structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        reports_dir = tmp_path / "reports"

        # Create test files
        (tmp_path / "README.md").write_text("# Test\n\nTest content with a TODO: fix this")
        (tmp_path / "no_heading.txt").write_text("This file has no heading but has words")

        job = RepoDocsRefreshJob(
            job_name="repo_docs_refresh",
            data_dir=data_dir,
            reports_dir=reports_dir,
        )

        result = await job.execute()

        assert result.status == JobStatus.COMPLETED
        assert result.tasks_completed == 2
        assert len(result.output_paths) == 2
        assert any("docs_health.md" in p for p in result.output_paths)
        assert any("docs_health.json" in p for p in result.output_paths)

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, tmp_path):
        """Test that job respects budget limits."""
        from backend.jobs.repo_docs_refresh import RepoDocsRefreshJob

        # Create test structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        reports_dir = tmp_path / "reports"

        # Create multiple test files
        for i in range(10):
            (tmp_path / f"file{i}.md").write_text(f"# File {i}\n\nContent")

        # Set very low task limit
        budget = BudgetConfig(max_tasks_per_run=3)

        job = RepoDocsRefreshJob(
            job_name="repo_docs_refresh",
            data_dir=data_dir,
            reports_dir=reports_dir,
            budget_config=budget,
        )

        result = await job.execute()

        assert result.status == JobStatus.BUDGET_EXCEEDED
        assert result.tasks_completed == 3  # Stopped at budget limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
