"""Preflight module for Night Shift jobs.

Provides pre-execution checks and estimates for jobs that might be long-running
or resource-intensive. Every job that might exceed 5 minutes or touch heavy
compute should use preflight checks.

Usage:
    preflight = PreflightInfo(
        job_name="repo_docs_refresh",
        input_counts={"documents": 42, "total_tokens_estimate": 50000},
        estimated_runtime_seconds=300,
        target_device="cpu",
        output_paths=["reports/2024-01-15_morning.md"]
    )
    preflight.display()
"""

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PreflightInfo:
    """Pre-execution information for a Night Shift job."""

    job_name: str
    input_counts: dict[str, int] = field(default_factory=dict)
    estimated_runtime_seconds: int = 0
    target_device: str = "cpu"
    output_paths: list[str] = field(default_factory=list)
    budget_limits: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def estimated_runtime_display(self) -> str:
        """Format estimated runtime for display."""
        seconds = self.estimated_runtime_seconds
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def display(self) -> None:
        """Print preflight information to stdout."""
        print("\n" + "=" * 60)
        print(f"PREFLIGHT CHECK: {self.job_name}")
        print("=" * 60)

        print(f"\nTimestamp: {datetime.now().isoformat()}")
        print(f"Target Device: {self.target_device}")
        print(f"Estimated Runtime: {self.estimated_runtime_display()}")

        if self.input_counts:
            print("\nInput Counts:")
            for key, value in self.input_counts.items():
                print(f"  - {key}: {value:,}")

        if self.budget_limits:
            print("\nBudget Limits:")
            for key, value in self.budget_limits.items():
                print(f"  - {key}: {value}")

        if self.output_paths:
            print("\nOutput Paths:")
            for path in self.output_paths:
                print(f"  - {path}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        print("\n" + "-" * 60)
        print("To execute this job, add the --go flag")
        print("-" * 60 + "\n")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_name": self.job_name,
            "input_counts": self.input_counts,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "estimated_runtime_display": self.estimated_runtime_display(),
            "target_device": self.target_device,
            "output_paths": self.output_paths,
            "budget_limits": self.budget_limits,
            "warnings": self.warnings,
            "timestamp": datetime.now().isoformat(),
        }


def verify_repo_root(expected_root: str) -> tuple[bool, str, str, str]:
    """Verify we're in the expected repository root.

    Args:
        expected_root: Expected absolute path to repo root

    Returns:
        Tuple of (is_valid, current_pwd, expected_root, actual_root)
    """
    current_pwd = str(Path.cwd())

    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return (False, current_pwd, expected_root, "<not a git repo>")

        actual_root = result.stdout.strip()

        # Normalize paths for comparison
        expected_normalized = str(Path(expected_root).resolve())
        actual_normalized = str(Path(actual_root).resolve())

        is_valid = expected_normalized == actual_normalized
        return (is_valid, current_pwd, expected_root, actual_root)

    except subprocess.TimeoutExpired:
        return (False, current_pwd, expected_root, "<git command timeout>")
    except FileNotFoundError:
        return (False, current_pwd, expected_root, "<git not found>")


def print_repo_mismatch(current_pwd: str, expected: str, actual: str) -> None:
    """Print repo root mismatch error."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("STOP: Repository root mismatch detected", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"\nCurrent working directory: {current_pwd}", file=sys.stderr)
    print(f"Expected repo root: {expected}", file=sys.stderr)
    print(f"Actual repo root: {actual}", file=sys.stderr)
    print("\nThis safety check prevents running Night Shift in the wrong repo.", file=sys.stderr)
    print("Verify you're in the correct directory and try again.", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)


@dataclass
class BudgetConfig:
    """Budget configuration for job execution."""

    max_tasks_per_run: int = 100
    max_tokens_total: int = 1_000_000
    max_spend_usd: float | None = None
    timeout_seconds: int = 3600  # 1 hour default

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetConfig":
        """Create from dictionary."""
        return cls(
            max_tasks_per_run=data.get("max_tasks_per_run", 100),
            max_tokens_total=data.get("max_tokens_total", 1_000_000),
            max_spend_usd=data.get("max_spend_usd"),
            timeout_seconds=data.get("timeout_seconds", 3600),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_tasks_per_run": self.max_tasks_per_run,
            "max_tokens_total": self.max_tokens_total,
            "max_spend_usd": self.max_spend_usd,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class BudgetTracker:
    """Track budget consumption during job execution."""

    config: BudgetConfig
    tasks_completed: int = 0
    tokens_used: int = 0
    spend_usd: float = 0.0
    start_time: datetime | None = None

    def start(self) -> None:
        """Mark job start time."""
        self.start_time = datetime.now()

    def record_task(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record completion of a task."""
        self.tasks_completed += 1
        self.tokens_used += tokens
        self.spend_usd += cost_usd

    def is_within_budget(self) -> tuple[bool, str | None]:
        """Check if still within budget limits.

        Returns:
            Tuple of (is_within_budget, reason_if_exceeded)
        """
        if self.tasks_completed >= self.config.max_tasks_per_run:
            return (False, f"max_tasks_per_run ({self.config.max_tasks_per_run}) reached")

        if self.tokens_used >= self.config.max_tokens_total:
            return (False, f"max_tokens_total ({self.config.max_tokens_total:,}) reached")

        if self.config.max_spend_usd and self.spend_usd >= self.config.max_spend_usd:
            return (False, f"max_spend_usd (${self.config.max_spend_usd:.2f}) reached")

        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed >= self.config.timeout_seconds:
                return (False, f"timeout_seconds ({self.config.timeout_seconds}s) reached")

        return (True, None)

    def summary(self) -> dict[str, Any]:
        """Get budget consumption summary."""
        elapsed = 0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            "tasks_completed": self.tasks_completed,
            "tasks_limit": self.config.max_tasks_per_run,
            "tokens_used": self.tokens_used,
            "tokens_limit": self.config.max_tokens_total,
            "spend_usd": self.spend_usd,
            "spend_limit": self.config.max_spend_usd,
            "elapsed_seconds": int(elapsed),
            "timeout_seconds": self.config.timeout_seconds,
        }
