"""Base job class for Night Shift runner.

All jobs must inherit from BaseJob and implement:
- get_preflight_info(): Return PreflightInfo for the job
- execute(): Run the actual job and return JobResult
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..preflight import PreflightInfo, BudgetConfig, BudgetTracker


class JobStatus(Enum):
    """Status of a job execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class JobResult:
    """Result of a job execution."""

    job_name: str
    status: JobStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    output_paths: List[str] = field(default_factory=list)
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    budget_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_name": self.job_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            ),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "output_paths": self.output_paths,
            "summary": self.summary,
            "details": self.details,
            "errors": self.errors,
            "budget_summary": self.budget_summary,
        }

    def mark_completed(self) -> None:
        """Mark job as completed."""
        self.completed_at = datetime.now()
        self.status = JobStatus.COMPLETED

    def mark_failed(self, error: str) -> None:
        """Mark job as failed."""
        self.completed_at = datetime.now()
        self.status = JobStatus.FAILED
        self.errors.append(error)

    def mark_aborted(self, reason: str) -> None:
        """Mark job as aborted."""
        self.completed_at = datetime.now()
        self.status = JobStatus.ABORTED
        self.errors.append(f"Aborted: {reason}")

    def mark_budget_exceeded(self, reason: str) -> None:
        """Mark job as stopped due to budget limit."""
        self.completed_at = datetime.now()
        self.status = JobStatus.BUDGET_EXCEEDED
        self.errors.append(f"Budget exceeded: {reason}")


class BaseJob(ABC):
    """Base class for Night Shift jobs.

    Subclasses must implement:
    - get_preflight_info(): Analyze inputs and return preflight information
    - execute(): Run the job and return results
    """

    def __init__(
        self,
        job_name: str,
        data_dir: Path,
        reports_dir: Path,
        budget_config: Optional[BudgetConfig] = None,
        mode: str = "cpu",
    ):
        """Initialize the job.

        Args:
            job_name: Name of this job type
            data_dir: Path to data directory
            reports_dir: Path to reports output directory
            budget_config: Budget limits for this execution
            mode: Execution mode (cpu, gpu, skip)
        """
        self.job_name = job_name
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.budget_config = budget_config or BudgetConfig()
        self.budget_tracker = BudgetTracker(config=self.budget_config)
        self.mode = mode

        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_preflight_info(self) -> PreflightInfo:
        """Analyze inputs and return preflight information.

        This method should:
        - Count inputs to be processed
        - Estimate runtime
        - Identify output paths
        - Add any relevant warnings

        Returns:
            PreflightInfo with job details
        """
        pass

    @abstractmethod
    async def execute(self) -> JobResult:
        """Run the job and return results.

        This method should:
        - Process inputs respecting budget limits
        - Track progress via budget_tracker
        - Generate output files
        - Return comprehensive results

        Returns:
            JobResult with execution details
        """
        pass

    def check_budget(self) -> tuple[bool, Optional[str]]:
        """Check if job is still within budget.

        Returns:
            Tuple of (is_within_budget, reason_if_exceeded)
        """
        return self.budget_tracker.is_within_budget()

    def record_task(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record completion of a task.

        Args:
            tokens: Number of tokens used
            cost_usd: Cost in USD (if tracking cloud API spend)
        """
        self.budget_tracker.record_task(tokens=tokens, cost_usd=cost_usd)

    def get_report_path(self, suffix: str = "") -> Path:
        """Get path for today's report file.

        Args:
            suffix: Optional suffix before extension (e.g., "_morning")

        Returns:
            Path to report file
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.reports_dir / f"{date_str}{suffix}.md"

    def get_report_json_path(self, suffix: str = "") -> Path:
        """Get path for today's JSON report file.

        Args:
            suffix: Optional suffix before extension (e.g., "_morning")

        Returns:
            Path to JSON report file
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.reports_dir / f"{date_str}{suffix}.json"
