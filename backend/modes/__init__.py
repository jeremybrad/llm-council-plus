"""Mode Engine - swap-in thinking protocols for structured reasoning.

Mode = Protocol + Roles + Ledger + Rubric + Stop Conditions

Each mode is a folder with:
- mode.yaml: manifest defining protocol, roles, inputs, outputs
- prompts/: template files for system, turn, summary prompts
- schemas/: JSON schemas for output validation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, AsyncIterator
from pathlib import Path
import copy
import uuid

from .registry import load_mode, list_modes, ModeDefinition
from .sessions import SessionStore, ModeSession
from .json_recovery import parse_json, extract_best_effort_question, recover_socrates_turn


@dataclass
class ModeRunResult:
    """Result of a mode run or turn."""
    run_id: str
    mode_id: str
    session_id: Optional[str]
    status: str  # completed | in_progress | aborted | error
    output: Dict[str, Any]
    receipts: Dict[str, Any] = field(default_factory=dict)


class ModeRunner:
    """Orchestrates mode execution with session management."""

    def __init__(self, modes_dir: Optional[Path] = None):
        self.modes_dir = modes_dir or Path(__file__).parent
        self.session_store = SessionStore()
        self._mode_cache: Dict[str, ModeDefinition] = {}

    async def get_mode(self, mode_id: str) -> ModeDefinition:
        """Load mode definition (cached)."""
        if mode_id not in self._mode_cache:
            self._mode_cache[mode_id] = load_mode(mode_id, self.modes_dir)
        return self._mode_cache[mode_id]

    async def get_all_modes(self) -> List[ModeDefinition]:
        """List all available modes."""
        return list_modes(self.modes_dir)

    def render_prompt(
        self,
        mode_id: str,
        template_name: str,
        variables: Dict[str, Any]
    ) -> str:
        """Render a prompt template with variables."""
        mode = load_mode(mode_id, self.modes_dir)
        template_path = self.modes_dir / mode_id / "prompts" / f"{template_name}.txt"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        template = template_path.read_text()

        # Simple variable substitution
        for key, value in variables.items():
            if isinstance(value, (dict, list)):
                import json
                value = json.dumps(value, indent=2)
            template = template.replace(f"{{{key}}}", str(value))

        return template

    async def create_session(
        self,
        mode_id: str,
        initial_inquiry: Optional[str] = None,
        model: Optional[str] = None
    ) -> ModeSession:
        """Create a new mode session."""
        mode = await self.get_mode(mode_id)
        session = self.session_store.create_session(
            mode_id=mode_id,
            initial_inquiry=initial_inquiry,
            max_turns=mode.protocol.get("max_turns_default", 12),
            model=model
        )
        return session

    async def get_session(self, session_id: str) -> Optional[ModeSession]:
        """Retrieve an existing session."""
        return self.session_store.get_session(session_id)

    def merge_ledger(self, current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ledger_update into current ledger.

        Uses ID-based tracking with status fields (active/superseded/retracted).
        Events are append-only; current state is derived.
        """
        merged = copy.deepcopy(current)

        # List fields with ID-based items
        list_fields = [
            "definitions", "commitments", "assumptions",
            "counterexamples", "contradictions", "open_questions"
        ]

        for field_name in list_fields:
            if field_name not in update:
                continue

            merged.setdefault(field_name, [])
            existing_ids = {item.get("id") for item in merged[field_name] if item.get("id")}

            for item in update[field_name]:
                # Ensure every item has an ID
                if not item.get("id"):
                    item["id"] = str(uuid.uuid4())[:8]

                # Ensure status field exists
                if "status" not in item:
                    item["status"] = "active"

                # Check for supersession
                if item.get("supersedes_id"):
                    for existing in merged[field_name]:
                        if existing.get("id") == item["supersedes_id"]:
                            existing["status"] = "superseded"
                            existing["superseded_by_id"] = item["id"]

                # Add if new ID
                if item["id"] not in existing_ids:
                    merged[field_name].append(item)
                else:
                    # Update existing item (for status changes like retraction)
                    for i, existing in enumerate(merged[field_name]):
                        if existing.get("id") == item["id"]:
                            merged[field_name][i] = {**existing, **item}
                            break

        # Scalar fields overwrite (inquiry, thesis)
        for scalar_field in ["inquiry", "thesis"]:
            if update.get(scalar_field):
                merged[scalar_field] = update[scalar_field]

        return merged

    def get_active_ledger_view(self, ledger: Dict[str, Any]) -> Dict[str, Any]:
        """Derive current state view from event-sourced ledger.

        Filters to only active items (not superseded or retracted).
        """
        view = {
            "inquiry": ledger.get("inquiry", ""),
            "thesis": ledger.get("thesis", "")
        }

        list_fields = [
            "definitions", "commitments", "assumptions",
            "counterexamples", "contradictions", "open_questions"
        ]

        for field_name in list_fields:
            items = ledger.get(field_name, [])
            view[field_name] = [
                item for item in items
                if item.get("status", "active") == "active"
            ]

        return view


# Module-level instance for convenience
_runner: Optional[ModeRunner] = None

def get_mode_runner() -> ModeRunner:
    """Get or create the global ModeRunner instance."""
    global _runner
    if _runner is None:
        _runner = ModeRunner()
    return _runner
