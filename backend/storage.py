"""JSON-based storage for conversations and roundtable runs."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DATA_DIR

# Runs directory for roundtable mode - sibling to conversations
RUNS_DIR = Path(DATA_DIR).parent / "runs"


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def ensure_runs_dir(conversation_id: str = None):
    """Ensure the runs directory exists.

    Args:
        conversation_id: If provided, creates conversation-specific subdirectory
    """
    if conversation_id:
        RUNS_DIR.joinpath(conversation_id).mkdir(parents=True, exist_ok=True)
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def create_conversation(conversation_id: str) -> dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        New conversation dict
    """
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": "New Conversation",
        "messages": [],
    }

    # Save to file
    path = get_conversation_path(conversation_id)
    with open(path, "w") as f:
        json.dump(conversation, f, indent=2)

    return conversation


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)


def save_conversation(conversation: dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    ensure_data_dir()

    path = get_conversation_path(conversation["id"])
    with open(path, "w") as f:
        json.dump(conversation, f, indent=2)


def list_conversations() -> list[dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            path = os.path.join(DATA_DIR, filename)
            with open(path) as f:
                data = json.load(f)
                # Return metadata only
                conversations.append(
                    {
                        "id": data["id"],
                        "created_at": data["created_at"],
                        "title": data.get("title", "New Conversation"),
                        "message_count": len(data["messages"]),
                    }
                )

    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)

    return conversations


def add_user_message(conversation_id: str, content: str):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({"role": "user", "content": content})

    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: list[dict[str, Any]],
    stage2: list[dict[str, Any]] | None = None,
    stage3: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Add an assistant message to a conversation.

    Supports partial execution modes where stage2 and/or stage3 may be None.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses (always present)
        stage2: List of model rankings (None if execution_mode was 'chat_only')
        stage3: Final synthesized response (None if execution_mode was not 'full')
        metadata: Optional metadata including execution_mode, label_to_model, etc.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message = {
        "role": "assistant",
        "stage1": stage1,
    }

    # Only include stage2 and stage3 if they were executed
    if stage2 is not None:
        message["stage2"] = stage2
    if stage3 is not None:
        message["stage3"] = stage3

    if metadata:
        message["metadata"] = metadata

    conversation["messages"].append(message)

    save_conversation(conversation)


def add_error_message(conversation_id: str, error_text: str):
    """
    Add an error message to a conversation to record a failed turn.

    Args:
        conversation_id: Conversation identifier
        error_text: The error description
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message = {"role": "assistant", "content": None, "error": error_text, "stage1": [], "stage2": [], "stage3": None}

    conversation["messages"].append(message)
    save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["title"] = title
    save_conversation(conversation)


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        True if deleted, False if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return False

    os.remove(path)
    return True


# =============================================================================
# Roundtable Run Storage
# =============================================================================


def get_run_path(conversation_id: str, run_id: str) -> Path:
    """Get the file path for a run.

    Args:
        conversation_id: Parent conversation ID
        run_id: Run identifier

    Returns:
        Path to the run file
    """
    return RUNS_DIR / conversation_id / f"{run_id}.json"


def save_run(run_data: dict[str, Any]) -> None:
    """Save a roundtable run to storage.

    Uses atomic write (write to temp, fsync, rename) to prevent corruption.

    Args:
        run_data: Run data dict with 'run_id' and 'conversation_id' keys
    """
    conversation_id = run_data["conversation_id"]
    run_id = run_data["run_id"]

    ensure_runs_dir(conversation_id)

    run_path = get_run_path(conversation_id, run_id)
    run_dir = run_path.parent

    # Atomic write: write to temp file, fsync, then rename
    fd, tmp_path = tempfile.mkstemp(dir=run_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(run_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.rename(tmp_path, run_path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def get_run(conversation_id: str, run_id: str) -> dict[str, Any] | None:
    """Load a run from storage.

    Args:
        conversation_id: Parent conversation ID
        run_id: Run identifier

    Returns:
        Run data dict or None if not found
    """
    run_path = get_run_path(conversation_id, run_id)

    if not run_path.exists():
        return None

    with open(run_path) as f:
        return json.load(f)


def list_runs(conversation_id: str) -> list[dict[str, Any]]:
    """List all runs for a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        List of run metadata dicts (run_id, status, created_at)
    """
    runs_path = RUNS_DIR / conversation_id

    if not runs_path.exists():
        return []

    runs = []
    for filename in runs_path.iterdir():
        if filename.suffix == ".json":
            try:
                with open(filename) as f:
                    data = json.load(f)
                    runs.append(
                        {
                            "run_id": data.get("run_id"),
                            "status": data.get("status", "unknown"),
                            "created_at": data.get("created_at"),
                            "completed_at": data.get("completed_at"),
                        }
                    )
            except Exception:
                continue

    # Sort by creation time, newest first
    runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return runs


def add_roundtable_message(
    conversation_id: str, run_id: str, chair_final: dict[str, Any], metadata: dict[str, Any] | None = None
):
    """Add a roundtable assistant message to a conversation.

    This creates a slim message in the conversation that references the run file.
    The run file contains full round transcripts.

    Args:
        conversation_id: Conversation identifier
        run_id: The roundtable run ID
        chair_final: The chair's final synthesis
        metadata: Optional metadata including council_members, num_rounds, etc.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message = {
        "role": "assistant",
        "execution_mode": "roundtable",
        "run_id": run_id,
        "chair_final": chair_final,
    }

    if metadata:
        message["metadata"] = metadata

    conversation["messages"].append(message)
    save_conversation(conversation)
