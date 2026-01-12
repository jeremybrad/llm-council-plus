"""Panel adjudication for claims.

This module provides multi-model adjudication where a panel of LLMs
evaluates a claim against its evidence and returns a verdict.

Key design principles:
- Evidence-ID-only citations (no inventing sources)
- Strict JSON schema output with 1 retry on failure
- Results written to review_history (never overwrites evidence)
- Verdicts are advisory - status updates are optional
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .claims import Claim, get_claim, update_claim
from .council import query_model
from .settings import get_settings


# JSON schema for panel response
VERDICT_SCHEMA = {
    "verdict": "accept | dispute | insufficient | abstain",
    "confidence": "0.0-1.0",
    "reasoning": "string explaining verdict",
    "cited_evidence": ["list of evidence_ids that support this verdict"],
    "concerns": ["optional list of issues or gaps"],
}

ADJUDICATION_PROMPT = """You are an impartial adjudicator evaluating a factual claim against available evidence.

## Claim
{claim_text}

## Claim Type
{claim_type}

## Available Evidence
{evidence_list}

## Your Task
Evaluate whether the evidence supports, contradicts, or is insufficient to verify this claim.

CRITICAL RULES:
1. You may ONLY cite evidence by its evidence_id (e.g., "ev_abc123")
2. You may NOT invent, assume, or reference any evidence not listed above
3. If evidence is insufficient, say so - do not speculate
4. Consider both supporting AND contradicting evidence

## Required Output Format
Respond with ONLY a JSON object matching this exact schema:
```json
{{
  "verdict": "accept|dispute|insufficient|abstain",
  "confidence": 0.0-1.0,
  "reasoning": "Your explanation here",
  "cited_evidence": ["ev_id1", "ev_id2"],
  "concerns": ["optional concern 1", "optional concern 2"]
}}
```

Verdict meanings:
- "accept": Evidence strongly supports the claim
- "dispute": Evidence contradicts or undermines the claim
- "insufficient": Not enough evidence to determine either way
- "abstain": Unable to evaluate (e.g., claim is ambiguous)

Respond with ONLY the JSON object, no other text."""


@dataclass
class PanelVerdict:
    """Single panel member's verdict."""
    model: str
    verdict: str  # accept, dispute, insufficient, abstain
    confidence: float
    reasoning: str
    cited_evidence: List[str]
    concerns: List[str]
    error: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class AdjudicationResult:
    """Complete adjudication result from panel."""
    claim_id: str
    panel_size: int
    mode: str
    panel_verdicts: List[PanelVerdict]
    consensus_verdict: str
    consensus_confidence: float
    timestamp: str
    status_updated: bool = False
    new_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "claim_id": self.claim_id,
            "panel_size": self.panel_size,
            "mode": self.mode,
            "panel_verdicts": [asdict(v) for v in self.panel_verdicts],
            "consensus_verdict": self.consensus_verdict,
            "consensus_confidence": self.consensus_confidence,
            "timestamp": self.timestamp,
            "status_updated": self.status_updated,
            "new_status": self.new_status,
        }


def _format_evidence_list(claim: Claim) -> str:
    """Format evidence for the prompt."""
    if not claim.evidence:
        return "(No evidence available)"

    lines = []
    for ev in claim.evidence:
        support_label = {
            "supports": "[SUPPORTS]",
            "contradicts": "[CONTRADICTS]",
            "neutral": "[NEUTRAL]",
        }.get(ev.support, "[UNKNOWN]")

        lines.append(
            f"- **{ev.evidence_id}** {support_label} (weight: {ev.weight:.2f})\n"
            f"  Source: {ev.source_type}/{ev.source_id}\n"
            f"  Quote: \"{ev.quote[:300]}{'...' if len(ev.quote) > 300 else ''}\""
        )

    return "\n\n".join(lines)


def _parse_verdict_json(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model response with fallback extraction."""
    # Try direct parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object
    brace_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_verdict(parsed: Dict[str, Any], valid_evidence_ids: List[str]) -> Optional[str]:
    """Validate parsed verdict against schema. Returns error message or None."""
    # Check required fields
    required = ["verdict", "confidence", "reasoning", "cited_evidence"]
    for field in required:
        if field not in parsed:
            return f"Missing required field: {field}"

    # Validate verdict value
    valid_verdicts = ["accept", "dispute", "insufficient", "abstain"]
    if parsed["verdict"] not in valid_verdicts:
        return f"Invalid verdict '{parsed['verdict']}', must be one of {valid_verdicts}"

    # Validate confidence range
    try:
        conf = float(parsed["confidence"])
        if not 0.0 <= conf <= 1.0:
            return f"Confidence {conf} out of range [0.0, 1.0]"
    except (ValueError, TypeError):
        return f"Invalid confidence value: {parsed['confidence']}"

    # Validate cited evidence IDs exist
    cited = parsed.get("cited_evidence", [])
    if not isinstance(cited, list):
        return "cited_evidence must be a list"

    invalid_ids = [eid for eid in cited if eid not in valid_evidence_ids]
    if invalid_ids:
        return f"Invalid evidence IDs cited: {invalid_ids}"

    return None


async def _query_panel_member(
    model: str,
    claim: Claim,
    evidence_list: str,
    valid_evidence_ids: List[str],
    temperature: float = 0.3,
) -> PanelVerdict:
    """Query a single panel member with retry on schema failure."""

    prompt = ADJUDICATION_PROMPT.format(
        claim_text=claim.claim_text,
        claim_type=claim.claim_type,
        evidence_list=evidence_list,
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(2):  # One retry
        try:
            result = await query_model(model, messages, temperature=temperature)

            if result.get("error"):
                return PanelVerdict(
                    model=model,
                    verdict="abstain",
                    confidence=0.0,
                    reasoning="Model error",
                    cited_evidence=[],
                    concerns=[],
                    error=result.get("error_message", "Unknown error"),
                    raw_response=None,
                )

            content = result.get("content", "")
            parsed = _parse_verdict_json(content)

            if not parsed:
                if attempt == 0:
                    # Retry with clarification
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Please respond with ONLY a valid JSON object matching the schema. No other text.",
                    })
                    continue
                return PanelVerdict(
                    model=model,
                    verdict="abstain",
                    confidence=0.0,
                    reasoning="Failed to parse JSON response",
                    cited_evidence=[],
                    concerns=[],
                    error="JSON parse failed after retry",
                    raw_response=content[:500],
                )

            # Validate schema
            validation_error = _validate_verdict(parsed, valid_evidence_ids)
            if validation_error:
                if attempt == 0:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"Schema error: {validation_error}. Please fix and respond with valid JSON only.",
                    })
                    continue
                return PanelVerdict(
                    model=model,
                    verdict="abstain",
                    confidence=0.0,
                    reasoning=f"Schema validation failed: {validation_error}",
                    cited_evidence=[],
                    concerns=[],
                    error=validation_error,
                    raw_response=content[:500],
                )

            # Success
            return PanelVerdict(
                model=model,
                verdict=parsed["verdict"],
                confidence=float(parsed["confidence"]),
                reasoning=parsed["reasoning"],
                cited_evidence=parsed.get("cited_evidence", []),
                concerns=parsed.get("concerns", []),
                error=None,
                raw_response=None,
            )

        except Exception as e:
            return PanelVerdict(
                model=model,
                verdict="abstain",
                confidence=0.0,
                reasoning="Exception during query",
                cited_evidence=[],
                concerns=[],
                error=str(e),
                raw_response=None,
            )

    # Should not reach here, but safety fallback
    return PanelVerdict(
        model=model,
        verdict="abstain",
        confidence=0.0,
        reasoning="Max retries exceeded",
        cited_evidence=[],
        concerns=[],
        error="Max retries exceeded",
        raw_response=None,
    )


def _compute_consensus(verdicts: List[PanelVerdict]) -> tuple[str, float]:
    """Compute consensus verdict from panel.

    Returns (verdict, confidence) where:
    - verdict is the majority opinion (or "insufficient" if no majority)
    - confidence is weighted average of agreeing members
    """
    # Count non-abstain verdicts
    valid_verdicts = [v for v in verdicts if v.verdict != "abstain" and not v.error]

    if not valid_verdicts:
        return "insufficient", 0.0

    # Count by verdict type
    counts: Dict[str, List[PanelVerdict]] = {}
    for v in valid_verdicts:
        counts.setdefault(v.verdict, []).append(v)

    # Find majority
    max_count = max(len(vlist) for vlist in counts.values())
    majority_threshold = len(valid_verdicts) / 2

    if max_count <= majority_threshold:
        # No clear majority
        return "insufficient", 0.3

    # Get majority verdict(s)
    majority_verdicts = [k for k, v in counts.items() if len(v) == max_count]

    if len(majority_verdicts) > 1:
        # Tie - return insufficient
        return "insufficient", 0.3

    winner = majority_verdicts[0]
    winner_members = counts[winner]

    # Calculate confidence as average of agreeing members
    avg_confidence = sum(v.confidence for v in winner_members) / len(winner_members)

    # Scale by agreement ratio
    agreement_ratio = len(winner_members) / len(valid_verdicts)
    final_confidence = avg_confidence * agreement_ratio

    return winner, round(final_confidence, 3)


async def adjudicate_claim(
    claim_id: str,
    models: Optional[List[str]] = None,
    panel_size: int = 3,
    mode: str = "strict",
    update_status: bool = False,
) -> AdjudicationResult:
    """Run panel adjudication on a claim.

    Args:
        claim_id: ID of claim to adjudicate
        models: List of model IDs to use (defaults to council_models from settings)
        panel_size: Number of panel members (default 3)
        mode: "strict" or "lenient" - affects status update thresholds
        update_status: If True, update claim status based on consensus

    Returns:
        AdjudicationResult with panel verdicts and consensus
    """
    claim = get_claim(claim_id)
    if not claim:
        raise ValueError(f"Claim {claim_id} not found")

    # Get models from settings if not provided
    if not models:
        settings = get_settings()
        models = settings.get("council_models", [])
        if not models:
            raise ValueError("No models configured for adjudication")

    # Limit to panel_size
    panel_models = models[:panel_size]

    # Build evidence context
    evidence_list = _format_evidence_list(claim)
    valid_evidence_ids = [ev.evidence_id for ev in claim.evidence]

    # Query panel in parallel
    tasks = [
        _query_panel_member(model, claim, evidence_list, valid_evidence_ids)
        for model in panel_models
    ]
    verdicts = await asyncio.gather(*tasks)

    # Compute consensus
    consensus_verdict, consensus_confidence = _compute_consensus(list(verdicts))

    timestamp = datetime.now(timezone.utc).isoformat()

    result = AdjudicationResult(
        claim_id=claim_id,
        panel_size=len(panel_models),
        mode=mode,
        panel_verdicts=list(verdicts),
        consensus_verdict=consensus_verdict,
        consensus_confidence=consensus_confidence,
        timestamp=timestamp,
        status_updated=False,
        new_status=None,
    )

    # Optionally update status
    if update_status and consensus_verdict in ("accept", "dispute"):
        threshold = 0.7 if mode == "strict" else 0.5

        if consensus_confidence >= threshold:
            new_status = "accepted" if consensus_verdict == "accept" else "disputed"

            # Write adjudication event to history
            history_event = {
                "event": "panel_adjudication",
                "timestamp": timestamp,
                "consensus_verdict": consensus_verdict,
                "consensus_confidence": consensus_confidence,
                "panel_size": len(panel_models),
                "models": panel_models,
                "status_updated_to": new_status,
            }

            # Panel consensus is sufficient justification for status change - use force
            update_claim(claim_id, status=new_status, review_history_event=history_event, force=True)
            result.status_updated = True
            result.new_status = new_status
    else:
        # Still record adjudication in history without status change
        history_event = {
            "event": "panel_adjudication",
            "timestamp": timestamp,
            "consensus_verdict": consensus_verdict,
            "consensus_confidence": consensus_confidence,
            "panel_size": len(panel_models),
            "models": panel_models,
            "status_updated_to": None,
        }
        update_claim(claim_id, review_history_event=history_event)

    return result
