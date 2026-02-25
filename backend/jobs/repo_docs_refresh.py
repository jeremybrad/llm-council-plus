"""Repo Docs Refresh job - Documentation health check and refresh.

This job scans the repository for documentation files and:
- Checks for broken internal links
- Identifies TODO/FIXME markers
- Validates markdown structure
- Generates a documentation health report

Usage via Night Shift:
    python -m backend.nightshift run --job=repo_docs_refresh --expected-repo-root=/path/to/repo --go
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..job_registry import register_job
from ..preflight import PreflightInfo
from .base import BaseJob, JobResult, JobStatus


@dataclass
class DocIssue:
    """An issue found in a documentation file."""

    file_path: str
    line_number: int
    issue_type: str
    message: str
    severity: str = "info"  # info, warning, error

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "issue_type": self.issue_type,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class DocFileResult:
    """Result of analyzing a single documentation file."""

    file_path: str
    word_count: int = 0
    line_count: int = 0
    heading_count: int = 0
    link_count: int = 0
    code_block_count: int = 0
    issues: list[DocIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "heading_count": self.heading_count,
            "link_count": self.link_count,
            "code_block_count": self.code_block_count,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@register_job("repo_docs_refresh")
class RepoDocsRefreshJob(BaseJob):
    """Job to refresh and validate repository documentation."""

    # Patterns to identify issues
    TODO_PATTERN = re.compile(r"\b(TODO|FIXME|XXX|HACK|BUG)\b[:\s]*(.*)", re.IGNORECASE)
    INTERNAL_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")

    # File patterns to scan
    DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".txt"}
    EXCLUDED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doc_files: list[Path] = []
        self.results: list[DocFileResult] = []

    def _find_doc_files(self) -> list[Path]:
        """Find all documentation files in the repository."""
        doc_files = []
        repo_root = self.data_dir.parent  # data_dir is repo/data, so parent is repo

        for ext in self.DOC_EXTENSIONS:
            for file_path in repo_root.rglob(f"*{ext}"):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in self.EXCLUDED_DIRS):
                    continue
                doc_files.append(file_path)

        return sorted(doc_files)

    def _analyze_file(self, file_path: Path) -> DocFileResult:
        """Analyze a single documentation file."""
        result = DocFileResult(file_path=str(file_path.relative_to(self.data_dir.parent)))

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            result.line_count = len(lines)
            result.word_count = len(content.split())

            # Count headings
            headings = self.HEADING_PATTERN.findall(content)
            result.heading_count = len(headings)

            # Count links
            links = self.INTERNAL_LINK_PATTERN.findall(content)
            result.link_count = len(links)

            # Count code blocks
            code_blocks = self.CODE_BLOCK_PATTERN.findall(content)
            result.code_block_count = len(code_blocks)

            # Check for TODOs
            for line_num, line in enumerate(lines, 1):
                match = self.TODO_PATTERN.search(line)
                if match:
                    result.issues.append(
                        DocIssue(
                            file_path=result.file_path,
                            line_number=line_num,
                            issue_type="todo",
                            message=f"{match.group(1)}: {match.group(2).strip()[:100]}",
                            severity="info",
                        )
                    )

            # Check for broken internal links
            for _link_text, link_url in links:
                if link_url.startswith(("http://", "https://", "mailto:", "#")):
                    continue  # Skip external links and anchors

                # Check if internal file exists
                linked_path = file_path.parent / link_url.split("#")[0]
                if not linked_path.exists():
                    result.issues.append(
                        DocIssue(
                            file_path=result.file_path,
                            line_number=0,  # Can't easily determine line number
                            issue_type="broken_link",
                            message=f"Broken internal link: {link_url}",
                            severity="warning",
                        )
                    )

            # Check for empty file
            if result.word_count == 0:
                result.issues.append(
                    DocIssue(
                        file_path=result.file_path,
                        line_number=1,
                        issue_type="empty_file",
                        message="File is empty",
                        severity="warning",
                    )
                )

            # Check for missing heading
            if result.heading_count == 0 and result.word_count > 10:
                result.issues.append(
                    DocIssue(
                        file_path=result.file_path,
                        line_number=1,
                        issue_type="no_heading",
                        message="File has no headings",
                        severity="info",
                    )
                )

        except Exception as e:
            result.issues.append(
                DocIssue(
                    file_path=result.file_path,
                    line_number=0,
                    issue_type="read_error",
                    message=f"Could not read file: {e}",
                    severity="error",
                )
            )

        return result

    def get_preflight_info(self) -> PreflightInfo:
        """Get preflight information for the job."""
        self.doc_files = self._find_doc_files()

        # Estimate runtime (roughly 0.1s per file)
        estimated_runtime = max(1, len(self.doc_files) // 10)

        output_paths = [
            str(self.get_report_path("_docs_health")),
            str(self.get_report_json_path("_docs_health")),
        ]

        warnings = []
        if len(self.doc_files) == 0:
            warnings.append("No documentation files found in repository")
        if len(self.doc_files) > 1000:
            warnings.append(f"Large number of doc files ({len(self.doc_files)}), may take longer")

        return PreflightInfo(
            job_name=self.job_name,
            input_counts={
                "documentation_files": len(self.doc_files),
            },
            estimated_runtime_seconds=estimated_runtime,
            target_device=self.mode,
            output_paths=output_paths,
            budget_limits=self.budget_config.to_dict(),
            warnings=warnings,
        )

    async def execute(self) -> JobResult:
        """Execute the documentation refresh job."""
        started_at = datetime.now()
        result = JobResult(
            job_name=self.job_name,
            status=JobStatus.RUNNING,
            started_at=started_at,
        )

        # Find doc files if not already done in preflight
        if not self.doc_files:
            self.doc_files = self._find_doc_files()

        self.results = []
        total_issues = 0
        issues_by_severity = {"error": 0, "warning": 0, "info": 0}
        issues_by_type: dict[str, int] = {}

        for file_path in self.doc_files:
            # Check budget
            is_ok, reason = self.check_budget()
            if not is_ok:
                result.mark_budget_exceeded(reason)
                break

            # Analyze file
            file_result = self._analyze_file(file_path)
            self.results.append(file_result)

            # Track issues
            for issue in file_result.issues:
                total_issues += 1
                issues_by_severity[issue.severity] += 1
                issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1

            # Record task completion
            self.record_task()
            result.tasks_completed += 1

        # Generate reports
        report_md_path = self.get_report_path("_docs_health")
        report_json_path = self.get_report_json_path("_docs_health")

        # Generate markdown report
        report_md = self._generate_markdown_report(issues_by_severity, issues_by_type)
        report_md_path.write_text(report_md)
        result.output_paths.append(str(report_md_path))

        # Generate JSON report
        report_json = {
            "generated_at": datetime.now().isoformat(),
            "job_name": self.job_name,
            "summary": {
                "files_analyzed": len(self.results),
                "total_issues": total_issues,
                "issues_by_severity": issues_by_severity,
                "issues_by_type": issues_by_type,
            },
            "files": [r.to_dict() for r in self.results],
            "budget": self.budget_tracker.summary(),
        }
        report_json_path.write_text(json.dumps(report_json, indent=2))
        result.output_paths.append(str(report_json_path))

        # Set result summary
        result.summary = (
            f"Analyzed {len(self.results)} files, "
            f"found {total_issues} issues "
            f"({issues_by_severity['error']} errors, "
            f"{issues_by_severity['warning']} warnings, "
            f"{issues_by_severity['info']} info)"
        )

        result.details = {
            "files_analyzed": len(self.results),
            "total_issues": total_issues,
            "issues_by_severity": issues_by_severity,
            "issues_by_type": issues_by_type,
        }

        result.budget_summary = self.budget_tracker.summary()

        if result.status == JobStatus.RUNNING:
            result.mark_completed()

        return result

    def _generate_markdown_report(self, issues_by_severity: dict[str, int], issues_by_type: dict[str, int]) -> str:
        """Generate markdown documentation health report."""
        lines = [
            "# Documentation Health Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Files analyzed**: {len(self.results)}",
            f"- **Total issues**: {sum(issues_by_severity.values())}",
            f"  - Errors: {issues_by_severity['error']}",
            f"  - Warnings: {issues_by_severity['warning']}",
            f"  - Info: {issues_by_severity['info']}",
            "",
        ]

        if issues_by_type:
            lines.extend(
                [
                    "## Issues by Type",
                    "",
                ]
            )
            for issue_type, count in sorted(issues_by_type.items(), key=lambda x: -x[1]):
                lines.append(f"- **{issue_type}**: {count}")
            lines.append("")

        # List files with issues
        files_with_issues = [r for r in self.results if r.issues]
        if files_with_issues:
            lines.extend(
                [
                    "## Files with Issues",
                    "",
                ]
            )
            for file_result in files_with_issues:
                lines.append(f"### {file_result.file_path}")
                lines.append("")
                for issue in file_result.issues:
                    severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•")
                    line_info = f" (line {issue.line_number})" if issue.line_number else ""
                    lines.append(f"- {severity_icon} **{issue.issue_type}**{line_info}: {issue.message}")
                lines.append("")

        # Statistics section
        total_words = sum(r.word_count for r in self.results)
        total_lines = sum(r.line_count for r in self.results)
        lines.extend(
            [
                "## Statistics",
                "",
                f"- Total words: {total_words:,}",
                f"- Total lines: {total_lines:,}",
                f"- Total headings: {sum(r.heading_count for r in self.results)}",
                f"- Total links: {sum(r.link_count for r in self.results)}",
                f"- Total code blocks: {sum(r.code_block_count for r in self.results)}",
                "",
            ]
        )

        return "\n".join(lines)
