"""Night Shift Runner - Scheduled job execution with safety gates.

This module provides a CLI for running overnight batch jobs with:
- Mandatory repo root verification (prevents running in wrong directory)
- Preflight mode by default (displays what would happen without --go)
- Budget enforcement (max tasks, tokens, spend, timeout)
- Morning report generation

Usage:
    # Preflight only (default) - shows what would run
    python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/repo

    # Actually execute
    python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/repo --go

    # With options
    python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/repo --go --mode=cpu

Exit codes:
    0 - Success
    1 - Error during execution
    2 - Preflight only (no --go) or safety gate failed
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .preflight import (
    PreflightInfo,
    BudgetConfig,
    verify_repo_root,
    print_repo_mismatch,
)
from .jobs.base import BaseJob, JobResult, JobStatus
from .job_registry import get_available_jobs, get_job_class


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for Night Shift CLI."""
    parser = argparse.ArgumentParser(
        prog="nightshift",
        description="Night Shift Runner - Scheduled job execution with safety gates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preflight check (default)
    python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/repo

    # Execute with --go flag
    python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/repo --go

    # List available jobs
    python -m backend.nightshift list

Exit codes:
    0 - Success
    1 - Error during execution
    2 - Preflight only or safety gate failed
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a job")
    run_parser.add_argument(
        "--job",
        required=True,
        help="Job type to run (use 'list' command to see available jobs)"
    )
    run_parser.add_argument(
        "--expected-repo-root",
        required=True,
        help="Expected absolute path to repository root (safety check)"
    )
    run_parser.add_argument(
        "--go",
        action="store_true",
        help="Actually execute the job (without this, only preflight is shown)"
    )
    run_parser.add_argument(
        "--mode",
        choices=["cpu", "gpu", "skip"],
        default="cpu",
        help="Execution mode (default: cpu)"
    )
    run_parser.add_argument(
        "--max-tasks",
        type=int,
        default=100,
        help="Maximum tasks to process (default: 100)"
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1_000_000,
        help="Maximum tokens to use (default: 1,000,000)"
    )
    run_parser.add_argument(
        "--max-spend",
        type=float,
        default=None,
        help="Maximum USD to spend on cloud APIs (optional)"
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600 = 1 hour)"
    )

    # List command
    subparsers.add_parser("list", help="List available jobs")

    return parser


def run_preflight(job: BaseJob) -> int:
    """Run preflight check and display information.

    Returns:
        Exit code (2 for preflight-only)
    """
    preflight_info = job.get_preflight_info()
    preflight_info.display()
    return 2


async def run_job(job: BaseJob) -> int:
    """Execute a job and return exit code.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print(f"\n{'=' * 60}")
    print(f"EXECUTING: {job.job_name}")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"{'=' * 60}\n")

    try:
        job.budget_tracker.start()
        result = await job.execute()

        # Print result summary
        print(f"\n{'=' * 60}")
        print(f"JOB COMPLETED: {job.job_name}")
        print(f"{'=' * 60}")
        print(f"Status: {result.status.value}")
        print(f"Tasks completed: {result.tasks_completed}")
        print(f"Tasks failed: {result.tasks_failed}")

        if result.completed_at and result.started_at:
            duration = (result.completed_at - result.started_at).total_seconds()
            print(f"Duration: {duration:.1f}s")

        if result.output_paths:
            print(f"\nOutput files:")
            for path in result.output_paths:
                print(f"  - {path}")

        if result.summary:
            print(f"\nSummary: {result.summary}")

        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        print(f"{'=' * 60}\n")

        # Determine exit code based on status
        if result.status == JobStatus.COMPLETED:
            return 0
        elif result.status == JobStatus.BUDGET_EXCEEDED:
            print("Job stopped due to budget limits (this is expected behavior)")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"JOB FAILED: {job.job_name}", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for Night Shift CLI.

    Returns:
        Exit code
    """
    # Import jobs to register them
    from .jobs import repo_docs_refresh  # noqa: F401
    from .jobs import truth_validation  # noqa: F401

    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 2

    if args.command == "list":
        print("\nAvailable jobs:")
        jobs = get_available_jobs()
        if jobs:
            for job_name in jobs:
                print(f"  - {job_name}")
        else:
            print("  (no jobs registered)")
        print()
        return 0

    if args.command == "run":
        # Safety gate 1: Verify repo root
        is_valid, pwd, expected, actual = verify_repo_root(args.expected_repo_root)
        if not is_valid:
            print_repo_mismatch(pwd, expected, actual)
            return 2

        # Check if job exists
        job_class = get_job_class(args.job)
        if not job_class:
            print(f"\nError: Unknown job '{args.job}'", file=sys.stderr)
            print(f"Available jobs: {', '.join(get_available_jobs()) or '(none)'}", file=sys.stderr)
            return 1

        # Create budget config
        budget_config = BudgetConfig(
            max_tasks_per_run=args.max_tasks,
            max_tokens_total=args.max_tokens,
            max_spend_usd=args.max_spend,
            timeout_seconds=args.timeout,
        )

        # Set up paths relative to repo root
        repo_root = Path(args.expected_repo_root)
        data_dir = repo_root / "data"
        reports_dir = repo_root / "reports"

        # Create job instance
        job = job_class(
            job_name=args.job,
            data_dir=data_dir,
            reports_dir=reports_dir,
            budget_config=budget_config,
            mode=args.mode,
        )

        # Safety gate 2: Preflight mode by default
        if not args.go:
            return run_preflight(job)

        # Execute the job
        return asyncio.run(run_job(job))

    return 0


if __name__ == "__main__":
    sys.exit(main())
