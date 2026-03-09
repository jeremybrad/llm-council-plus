"""Turn capture tap for roundtable runs.

Captures completed roundtable runs as normalized turn records to
`_SADB_Data/hot/turns/<session_id>.jsonl` for downstream hot facts extraction.

PRD: PRD-06f1b_roundtable_turn_capture.md
"""

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_sadb_root(override: Path | None = None) -> Path:
    """Resolve the SADB data root directory.

    Resolution order:
    1. CLI/valve override (if provided)
    2. SADB_DATA_DIR environment variable
    3. Walk up to find SyncedProjects, then sibling _SADB_Data

    Args:
        override: Optional explicit path to use

    Returns:
        Path to SADB data root

    Raises:
        RuntimeError: If SADB root cannot be resolved
    """
    # 1. CLI/valve override
    if override:
        return override

    # 2. SADB_DATA_DIR environment variable
    env_root = os.environ.get("SADB_DATA_DIR")
    if env_root:
        return Path(env_root)

    # 3. Walk up to find SyncedProjects, then sibling _SADB_Data
    current = Path(__file__).resolve()
    while current.parent != current:
        if current.name == "SyncedProjects":
            return current / "_SADB_Data"
        current = current.parent

    raise RuntimeError("Could not resolve SADB root: SyncedProjects not found in path")


def get_session_id() -> str:
    """Get session ID from bbot context or generate a new one.

    Checks ~/.bbot/context/current.json for session_id field ONLY.
    If not found or null, generates roundtable-{uuid}.

    Returns:
        Session ID string
    """
    bbot_context_path = Path.home() / ".bbot" / "context" / "current.json"

    if bbot_context_path.exists():
        try:
            with open(bbot_context_path, encoding="utf-8") as f:
                context = json.load(f)
                # Only use session_id, NOT capsule_id
                session_id = context.get("session_id")
                if session_id:
                    return session_id
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to generate

    # Generate fallback UUID
    return f"roundtable-{uuid.uuid4()}"


def sanitize_for_turn(content: str) -> dict[str, Any]:
    """Sanitize content using C003 leakscan.

    Wraps C003 leakscan_jsonl.py functions to provide turn-specific
    sanitization semantics.

    Args:
        content: Raw content to sanitize

    Returns:
        Dictionary with:
        - content: str (possibly stubbed/redacted)
        - sanitized: bool (any changes made)
        - blocked: bool (secrets found -> entire content blocked)
        - leakscan_hits: {"secrets": int, "pii": int}
    """
    if not content:
        return {
            "content": content,
            "sanitized": False,
            "blocked": False,
            "leakscan_hits": {"secrets": 0, "pii": 0},
        }

    try:
        # Try to import C003 leakscan
        c003_path = Path(__file__).resolve().parents[4] / "C003_sadb_canonical" / "40_src" / "s1_normalize"
        if str(c003_path) not in sys.path:
            sys.path.insert(0, str(c003_path))

        from leakscan_jsonl import PII_PATTERNS, SECRET_PATTERNS, redact_text, scan_text

        # Scan for hits
        hits = scan_text(content)

        # Count secrets vs PII
        secret_count = sum(hits.get(name, 0) for name in SECRET_PATTERNS.keys())
        pii_count = sum(hits.get(name, 0) for name in PII_PATTERNS.keys())

        # If secrets found, block entire content
        if secret_count > 0:
            return {
                "content": "[BLOCKED_SECRET]",
                "sanitized": True,
                "blocked": True,
                "leakscan_hits": {"secrets": secret_count, "pii": pii_count},
            }

        # If PII found, redact inline
        if pii_count > 0:
            redacted = redact_text(content)
            return {
                "content": redacted,
                "sanitized": True,
                "blocked": False,
                "leakscan_hits": {"secrets": 0, "pii": pii_count},
            }

        # Clean content
        return {
            "content": content,
            "sanitized": False,
            "blocked": False,
            "leakscan_hits": {"secrets": 0, "pii": 0},
        }

    except ImportError:
        # Leakscan unavailable - block all content for safety
        return {
            "content": "[LEAKSCAN_UNAVAILABLE]",
            "sanitized": True,
            "blocked": True,
            "leakscan_hits": {"secrets": 0, "pii": 0},
        }
    except Exception as e:
        # Any other error - block for safety
        print(f"Turn capture leakscan error: {e}")
        return {
            "content": "[LEAKSCAN_UNAVAILABLE]",
            "sanitized": True,
            "blocked": True,
            "leakscan_hits": {"secrets": 0, "pii": 0},
        }


def transform_run_to_turns(run_data: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    """Transform a roundtable run into normalized turn records.

    Args:
        run_data: Complete run data from roundtable execution
        session_id: Session ID for all turns

    Returns:
        List of normalized turn records
    """
    turns = []
    run_id = run_data.get("run_id", "")
    conversation_id = run_data.get("conversation_id", "")
    question = run_data.get("question", "")

    # 1. User turn (the original question)
    user_sanitized = sanitize_for_turn(question)
    turns.append(
        {
            "turn_id": str(uuid.uuid4()),
            "session_id": session_id,
            "timestamp": run_data.get("created_at", datetime.now(timezone.utc).isoformat()),
            "source_system": "roundtable",
            "speaker_role": "user",
            "speaker_id": "user",
            "model": None,
            "content": user_sanitized["content"],
            "sanitized": user_sanitized["sanitized"],
            "blocked": user_sanitized["blocked"],
            "leakscan_hits": user_sanitized["leakscan_hits"],
            "meta": {
                "round_number": None,
                "round_name": None,
                "duration_ms": None,
                "run_id": run_id,
                "conversation_id": conversation_id,
                "error": None,
                "agent_role": None,
            },
        }
    )

    # 2. Agent turns from each round
    for round_result in run_data.get("rounds", []):
        round_number = round_result.get("round_number")
        round_name = round_result.get("round_name")

        for response in round_result.get("responses", []):
            agent_label = response.get("agent_label", "")
            role = response.get("role", "")
            model = response.get("model", "")
            content = response.get("content", "")
            error = response.get("error")
            duration_ms = response.get("duration_ms")

            content_sanitized = sanitize_for_turn(content)

            turns.append(
                {
                    "turn_id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_system": "roundtable",
                    "speaker_role": "assistant",
                    "speaker_id": f"agent:{agent_label}",
                    "model": model,
                    "content": content_sanitized["content"],
                    "sanitized": content_sanitized["sanitized"],
                    "blocked": content_sanitized["blocked"],
                    "leakscan_hits": content_sanitized["leakscan_hits"],
                    "meta": {
                        "round_number": round_number,
                        "round_name": round_name,
                        "duration_ms": duration_ms,
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "error": error,
                        "agent_role": role,
                    },
                }
            )

    # 3. Moderator turn
    moderator_summary = run_data.get("moderator_summary")
    if moderator_summary:
        mod_content = moderator_summary.get("content", "")
        mod_model = moderator_summary.get("model", "")
        mod_error = moderator_summary.get("error")

        mod_sanitized = sanitize_for_turn(mod_content)

        turns.append(
            {
                "turn_id": str(uuid.uuid4()),
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_system": "roundtable",
                "speaker_role": "assistant",
                "speaker_id": "moderator",
                "model": mod_model,
                "content": mod_sanitized["content"],
                "sanitized": mod_sanitized["sanitized"],
                "blocked": mod_sanitized["blocked"],
                "leakscan_hits": mod_sanitized["leakscan_hits"],
                "meta": {
                    "round_number": None,
                    "round_name": "moderator",
                    "duration_ms": None,
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "error": mod_error if mod_error else None,
                    "agent_role": "Moderator",
                },
            }
        )

    # 4. Chair turn
    chair_final = run_data.get("chair_final")
    if chair_final:
        chair_content = chair_final.get("content", "")
        chair_model = chair_final.get("model", "")
        chair_error = chair_final.get("error")

        chair_sanitized = sanitize_for_turn(chair_content)

        turns.append(
            {
                "turn_id": str(uuid.uuid4()),
                "session_id": session_id,
                "timestamp": run_data.get("completed_at", datetime.now(timezone.utc).isoformat()),
                "source_system": "roundtable",
                "speaker_role": "assistant",
                "speaker_id": "chair",
                "model": chair_model,
                "content": chair_sanitized["content"],
                "sanitized": chair_sanitized["sanitized"],
                "blocked": chair_sanitized["blocked"],
                "leakscan_hits": chair_sanitized["leakscan_hits"],
                "meta": {
                    "round_number": None,
                    "round_name": "chair",
                    "duration_ms": None,
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "error": chair_error if chair_error else None,
                    "agent_role": "Chair",
                },
            }
        )

    return turns


def append_turns(session_id: str, turns: list[dict[str, Any]], sadb_root: Path) -> Path:
    """Append turns to the session's JSONL file with atomic write.

    Args:
        session_id: Session ID for filename
        turns: List of turn records to append
        sadb_root: SADB data root directory

    Returns:
        Path to the output file
    """
    # Ensure directory exists
    turns_dir = sadb_root / "hot" / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    output_path = turns_dir / f"{session_id}.jsonl"

    # Read existing content if file exists
    existing_lines = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing_lines = f.readlines()

    # Write to temp file first (atomic write pattern)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        dir=turns_dir,
        delete=False,
    ) as tmp:
        # Write existing content
        for line in existing_lines:
            tmp.write(line)

        # Append new turns
        for turn in turns:
            tmp.write(json.dumps(turn, ensure_ascii=False) + "\n")

        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    # Atomic rename
    os.replace(tmp_path, output_path)

    return output_path


def capture_run(
    run_data: dict[str, Any],
    session_id: str | None = None,
    sadb_root: Path | None = None,
) -> Path | None:
    """Capture a completed roundtable run to JSONL turns file.

    Main entry point for turn capture. Transforms run data to normalized
    turns, sanitizes content, and appends to session file.

    Args:
        run_data: Complete run data from roundtable execution
        session_id: Optional session ID override (otherwise resolved)
        sadb_root: Optional SADB root override (otherwise resolved)

    Returns:
        Path to output file, or None if capture failed
    """
    try:
        # Resolve session ID
        if session_id is None:
            session_id = get_session_id()

        # Resolve SADB root
        resolved_root = resolve_sadb_root(sadb_root)

        # Transform run to turns
        turns = transform_run_to_turns(run_data, session_id)

        if not turns:
            return None

        # Append to file
        output_path = append_turns(session_id, turns, resolved_root)

        print(f"Turn capture: wrote {len(turns)} turns to {output_path}")
        return output_path

    except Exception as e:
        print(f"Turn capture error: {e}")
        return None
