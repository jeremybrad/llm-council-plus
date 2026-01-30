"""Session management for interactive modes.

Sessions store ledger + history + receipts server-side,
so clients only send session_id + user_message per turn.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModeSession:
    """A mode session with accumulated state."""

    session_id: str
    mode_id: str
    created_at: str
    updated_at: str
    status: str  # active | completed | aborted
    turn_count: int
    max_turns: int
    model: str | None

    # Accumulated state
    ledger: dict[str, Any]
    messages: list[dict[str, str]]  # role/content pairs
    turn_receipts: list[dict[str, Any]]  # per-turn metadata

    # Stop state
    stop_recommended: bool = False
    stop_criteria_met: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "session_id": self.session_id,
            "mode_id": self.mode_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "model": self.model,
            "ledger": self.ledger,
            "messages": self.messages,
            "turn_receipts": self.turn_receipts,
            "stop_recommended": self.stop_recommended,
            "stop_criteria_met": self.stop_criteria_met,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModeSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            mode_id=data["mode_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data["status"],
            turn_count=data["turn_count"],
            max_turns=data["max_turns"],
            model=data.get("model"),
            ledger=data.get("ledger", {}),
            messages=data.get("messages", []),
            turn_receipts=data.get("turn_receipts", []),
            stop_recommended=data.get("stop_recommended", False),
            stop_criteria_met=data.get("stop_criteria_met", []),
        )


class SessionStore:
    """In-memory session store with optional file persistence."""

    def __init__(self, persist_dir: Path | None = None):
        self._sessions: dict[str, ModeSession] = {}
        self.persist_dir = persist_dir

        # Load existing sessions if persist_dir exists
        if persist_dir and persist_dir.exists():
            self._load_sessions()

    def _load_sessions(self) -> None:
        """Load sessions from persist_dir."""
        if not self.persist_dir:
            return

        for file in self.persist_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                session = ModeSession.from_dict(data)
                self._sessions[session.session_id] = session
            except Exception as e:
                print(f"Warning: Could not load session {file}: {e}")

    def _persist_session(self, session: ModeSession) -> None:
        """Persist session to file if persist_dir is set."""
        if not self.persist_dir:
            return

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.persist_dir / f"{session.session_id}.json"

        # Atomic write
        tmp_path = file_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(session.to_dict(), indent=2))
        tmp_path.replace(file_path)

    def create_session(
        self, mode_id: str, initial_inquiry: str | None = None, max_turns: int = 12, model: str | None = None
    ) -> ModeSession:
        """Create a new mode session."""
        now = datetime.utcnow().isoformat() + "Z"

        session = ModeSession(
            session_id=str(uuid.uuid4()),
            mode_id=mode_id,
            created_at=now,
            updated_at=now,
            status="active",
            turn_count=0,
            max_turns=max_turns,
            model=model,
            ledger={
                "inquiry": initial_inquiry or "",
                "thesis": "",
                "definitions": [],
                "commitments": [],
                "assumptions": [],
                "counterexamples": [],
                "contradictions": [],
                "open_questions": [],
            },
            messages=[],
            turn_receipts=[],
        )

        self._sessions[session.session_id] = session
        self._persist_session(session)
        return session

    def get_session(self, session_id: str) -> ModeSession | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def update_session(
        self,
        session_id: str,
        ledger: dict[str, Any] | None = None,
        messages: list[dict[str, str]] | None = None,
        turn_receipt: dict[str, Any] | None = None,
        status: str | None = None,
        stop_recommended: bool | None = None,
        stop_criteria_met: list[str] | None = None,
    ) -> ModeSession | None:
        """Update session state."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        session.updated_at = datetime.utcnow().isoformat() + "Z"

        if ledger is not None:
            session.ledger = ledger

        if messages is not None:
            session.messages = messages

        if turn_receipt is not None:
            session.turn_receipts.append(turn_receipt)
            session.turn_count = len(session.turn_receipts)

        if status is not None:
            session.status = status

        if stop_recommended is not None:
            session.stop_recommended = stop_recommended

        if stop_criteria_met is not None:
            session.stop_criteria_met = stop_criteria_met

        self._persist_session(session)
        return session

    def complete_session(self, session_id: str) -> ModeSession | None:
        """Mark session as completed."""
        return self.update_session(session_id, status="completed")

    def abort_session(self, session_id: str) -> ModeSession | None:
        """Mark session as aborted."""
        return self.update_session(session_id, status="aborted")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]

        if self.persist_dir:
            file_path = self.persist_dir / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()

        return True

    def list_sessions(self, mode_id: str | None = None, status: str | None = None) -> list[ModeSession]:
        """List sessions, optionally filtered."""
        sessions = list(self._sessions.values())

        if mode_id:
            sessions = [s for s in sessions if s.mode_id == mode_id]

        if status:
            sessions = [s for s in sessions if s.status == status]

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
