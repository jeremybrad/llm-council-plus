"""Truth Validation job - validate claims against SADB evidence.

This Night Shift job reviews claims in the confidence band (0.60-0.98),
searches for new evidence from SADB, and updates confidence scores.

Key features:
- Idempotent: Can be run multiple times safely
- Resumable: Tracks progress in checkpoint file
- Smart disputed logic: Only marks disputed on strong independent contradictions
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..preflight import PreflightInfo
from ..job_registry import register_job
from .base import BaseJob, JobResult, JobStatus
from ..claims import (
    get_claims_for_review,
    get_claim,
    update_claim,
    add_evidence,
    Claim,
)
from ..evidence import search_evidence, get_sadb_status
from ..scorer import classify_support, calculate_confidence_with_breakdown


@dataclass
class ValidationCheckpoint:
    """Checkpoint for resumable validation.

    Stored between runs to allow resumption after interruption.
    """
    job_run_id: str
    claims_to_process: List[str]  # claim_ids
    claims_processed: List[str]
    started_at: str
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationCheckpoint":
        """Create from dictionary."""
        return cls(**data)


@register_job("truth_validation")
class TruthValidationJob(BaseJob):
    """Night Shift job to validate and score claims against SADB evidence.

    This job:
    1. Selects claims in the 0.60-0.98 confidence band for review
    2. Searches SADB for new evidence related to each claim
    3. Classifies evidence as supporting/contradicting/neutral
    4. Updates confidence scores with full explainability
    5. Marks claims as disputed only on strong contradictions

    The job is idempotent and resumable via checkpoint files.
    """

    CHECKPOINT_FILE = "truth_validation_checkpoint.json"
    CONFIDENCE_BAND = (0.60, 0.98)
    CLAIMS_PER_RUN = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint: Optional[ValidationCheckpoint] = None

    def _get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        return self.data_dir / self.CHECKPOINT_FILE

    def _load_checkpoint(self) -> Optional[ValidationCheckpoint]:
        """Load existing checkpoint if resuming."""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)
                return ValidationCheckpoint.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_checkpoint(self, checkpoint: ValidationCheckpoint) -> None:
        """Save checkpoint for resumability."""
        checkpoint_path = self._get_checkpoint_path()
        checkpoint.last_updated = datetime.now(timezone.utc).isoformat()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def _clear_checkpoint(self) -> None:
        """Remove checkpoint after successful completion."""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def get_preflight_info(self) -> PreflightInfo:
        """Get preflight information with full receipts.

        Returns:
            PreflightInfo with claims to review, SADB status, and warnings
        """
        warnings = []

        # Check for existing checkpoint (resumption)
        existing = self._load_checkpoint()
        if existing:
            remaining = len(existing.claims_to_process) - len(existing.claims_processed)
            warnings.append(
                f"Resuming from checkpoint: {len(existing.claims_processed)} already processed"
            )
            claims_count = remaining
        else:
            claims = get_claims_for_review(n=self.CLAIMS_PER_RUN, band=self.CONFIDENCE_BAND)
            claims_count = len(claims)

        # Check SADB availability
        sadb_status = get_sadb_status()
        if not sadb_status["available"]:
            warnings.append("SADB unavailable - evidence retrieval will be skipped")
            if sadb_status.get("error"):
                warnings.append(f"SADB error: {sadb_status['error']}")

        # Output paths
        output_paths = [
            str(self.get_report_path("_truth_validation")),
            str(self.get_report_json_path("_truth_validation")),
        ]

        return PreflightInfo(
            job_name=self.job_name,
            input_counts={
                "claims_to_review": claims_count,
                "sadb_available": 1 if sadb_status["available"] else 0,
            },
            estimated_runtime_seconds=claims_count * 30,  # ~30s per claim
            target_device=self.mode,
            output_paths=output_paths,
            budget_limits=self.budget_config.to_dict(),
            warnings=warnings,
        )

    async def execute(self) -> JobResult:
        """Execute validation job (idempotent, resumable).

        Returns:
            JobResult with promoted/disputed/unchanged claims
        """
        started_at = datetime.now(timezone.utc)
        self.budget_tracker.start()

        result = JobResult(
            job_name=self.job_name,
            status=JobStatus.RUNNING,
            started_at=started_at,
        )

        # Check for checkpoint (resume or fresh start)
        self.checkpoint = self._load_checkpoint()
        if self.checkpoint:
            claims_to_process = [
                cid for cid in self.checkpoint.claims_to_process
                if cid not in self.checkpoint.claims_processed
            ]
        else:
            claims = get_claims_for_review(n=self.CLAIMS_PER_RUN, band=self.CONFIDENCE_BAND)
            claims_to_process = [c.claim_id for c in claims]
            self.checkpoint = ValidationCheckpoint(
                job_run_id=f"val_{started_at.strftime('%Y%m%d_%H%M%S')}",
                claims_to_process=claims_to_process,
                claims_processed=[],
                started_at=started_at.isoformat(),
                last_updated=started_at.isoformat(),
            )
            self._save_checkpoint(self.checkpoint)

        promoted: List[Dict[str, Any]] = []
        disputed: List[Dict[str, Any]] = []
        unchanged: List[str] = []

        for claim_id in claims_to_process:
            # Budget check
            is_ok, reason = self.check_budget()
            if not is_ok:
                result.mark_budget_exceeded(reason)
                self._save_checkpoint(self.checkpoint)  # Save progress
                break

            claim = get_claim(claim_id)
            if not claim:
                continue

            try:
                outcome = await self._process_claim(claim)

                if outcome["type"] == "promoted":
                    promoted.append({
                        "claim_id": claim_id,
                        "old": outcome["old_confidence"],
                        "new": outcome["new_confidence"],
                    })
                elif outcome["type"] == "disputed":
                    disputed.append({
                        "claim_id": claim_id,
                        "reason": outcome["reason"],
                    })
                else:
                    unchanged.append(claim_id)

            except Exception as e:
                result.errors.append(f"Error processing claim {claim_id}: {str(e)}")
                result.tasks_failed += 1

            # Update checkpoint
            self.checkpoint.claims_processed.append(claim_id)
            self._save_checkpoint(self.checkpoint)

            self.record_task()
            result.tasks_completed += 1

        # Generate reports
        report_md = self._generate_markdown_report(promoted, disputed, unchanged)
        report_json = self._generate_json_report(promoted, disputed, unchanged)

        md_path = self.get_report_path("_truth_validation")
        json_path = self.get_report_json_path("_truth_validation")

        md_path.write_text(report_md)
        json_path.write_text(json.dumps(report_json, indent=2))

        result.output_paths = [str(md_path), str(json_path)]
        result.summary = (
            f"Reviewed {result.tasks_completed} claims: "
            f"{len(promoted)} promoted, {len(disputed)} disputed, {len(unchanged)} unchanged"
        )
        result.details = {
            "promoted": promoted,
            "disputed": disputed,
            "unchanged": unchanged,
            "job_run_id": self.checkpoint.job_run_id if self.checkpoint else None,
        }
        result.budget_summary = self.budget_tracker.summary()

        if result.status == JobStatus.RUNNING:
            result.mark_completed()
            self._clear_checkpoint()  # Clear on success

        return result

    async def _process_claim(self, claim: Claim) -> Dict[str, Any]:
        """Process a single claim: search evidence, update confidence.

        Args:
            claim: The claim to process

        Returns:
            Dict with outcome type and details
        """
        old_confidence = claim.confidence

        # Search for new evidence
        new_evidence = await search_evidence(claim.claim_text, top_k=3)

        # Classify support for each piece (v1 heuristic)
        for ev in new_evidence:
            ev.support = classify_support(claim.claim_text, ev.quote)

        # Add new evidence to claim (fingerprint-based dedupe happens in add_evidence)
        for ev in new_evidence:
            add_evidence(claim.claim_id, ev)

        # Reload claim to get updated state
        claim = get_claim(claim.claim_id)
        new_confidence = claim.confidence

        # Determine outcome with smart disputed threshold
        contradictions = [e for e in new_evidence if e.support == "contradicts"]
        supporting_keys = {e.independence_key for e in claim.evidence if e.support == "supports"}

        # Only mark disputed if:
        # 1. Contradicting evidence is strong (weight >= 0.7) AND independent, OR
        # 2. Confidence dropped below 0.6
        strong_independent_contradiction = any(
            c.weight >= 0.7 and c.independence_key not in supporting_keys
            for c in contradictions
        )
        confidence_collapsed = new_confidence < 0.6 and old_confidence >= 0.6

        if new_confidence > old_confidence + 0.1:
            return {
                "type": "promoted",
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
            }
        elif strong_independent_contradiction or confidence_collapsed:
            reason = (
                "strong_independent_contradiction"
                if strong_independent_contradiction
                else "confidence_collapsed"
            )
            update_claim(
                claim.claim_id,
                status="disputed",
                review_history_event={
                    "event": "marked_disputed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": reason,
                    "old_confidence": old_confidence,
                    "new_confidence": new_confidence,
                },
            )
            return {
                "type": "disputed",
                "reason": reason,
            }
        else:
            # Log contradiction detected but don't change status
            if contradictions:
                update_claim(
                    claim.claim_id,
                    review_history_event={
                        "event": "contradiction_detected",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "evidence_ids": [c.evidence_id for c in contradictions],
                        "note": "contradiction_logged_not_disputed",
                    },
                )
            return {"type": "unchanged"}

    def _generate_markdown_report(
        self,
        promoted: List[Dict],
        disputed: List[Dict],
        unchanged: List[str],
    ) -> str:
        """Generate markdown report.

        Args:
            promoted: List of promoted claims with old/new confidence
            disputed: List of disputed claims with reasons
            unchanged: List of unchanged claim IDs

        Returns:
            Markdown formatted report
        """
        lines = [
            "# Truth Validation Report",
            "",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Job Run ID**: {self.checkpoint.job_run_id if self.checkpoint else 'N/A'}",
            "",
            "## Summary",
            "",
            f"- **Claims reviewed**: {len(promoted) + len(disputed) + len(unchanged)}",
            f"- **Promoted** (confidence increased): {len(promoted)}",
            f"- **Disputed** (contradictions found): {len(disputed)}",
            f"- **Unchanged**: {len(unchanged)}",
            "",
        ]

        if promoted:
            lines.extend(["## Promoted Claims", ""])
            for p in promoted:
                lines.append(f"- `{p['claim_id'][:8]}...`: {p['old']:.2f} → {p['new']:.2f}")
            lines.append("")

        if disputed:
            lines.extend(["## Disputed Claims", ""])
            for d in disputed:
                lines.append(f"- `{d['claim_id'][:8]}...`: {d['reason']}")
            lines.append("")

        if unchanged:
            lines.extend(["## Unchanged Claims", ""])
            for u in unchanged:
                lines.append(f"- `{u[:8]}...`")
            lines.append("")

        # Budget summary
        budget = self.budget_tracker.summary()
        lines.extend([
            "## Budget Summary",
            "",
            f"- Tasks completed: {budget['tasks_completed']}/{budget['tasks_limit']}",
            f"- Elapsed time: {budget['elapsed_seconds']}s / {budget['timeout_seconds']}s",
            "",
        ])

        return "\n".join(lines)

    def _generate_json_report(
        self,
        promoted: List[Dict],
        disputed: List[Dict],
        unchanged: List[str],
    ) -> Dict[str, Any]:
        """Generate JSON report.

        Args:
            promoted: List of promoted claims with old/new confidence
            disputed: List of disputed claims with reasons
            unchanged: List of unchanged claim IDs

        Returns:
            JSON-serializable report dict
        """
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job_name": self.job_name,
            "job_run_id": self.checkpoint.job_run_id if self.checkpoint else None,
            "summary": {
                "total_reviewed": len(promoted) + len(disputed) + len(unchanged),
                "promoted": len(promoted),
                "disputed": len(disputed),
                "unchanged": len(unchanged),
            },
            "promoted": promoted,
            "disputed": disputed,
            "unchanged": unchanged,
            "budget": self.budget_tracker.summary(),
        }
