"""Socrates Mode runner - handles turn execution and session management."""

import json
from pathlib import Path
from typing import Any

from .json_recovery import parse_json, recover_socrates_turn
from .registry import load_mode
from .sessions import ModeSession


class SocratesRunner:
    """Handles Socrates mode execution."""

    def __init__(self, modes_dir: Path):
        self.modes_dir = modes_dir
        self.mode = load_mode("socrates", modes_dir)
        self._load_prompts()
        self._load_schema()

    def _load_prompts(self) -> None:
        """Load prompt templates."""
        prompts_dir = self.modes_dir / "socrates" / "prompts"
        self.system_prompt = (prompts_dir / "system.txt").read_text()
        self.turn_template = (prompts_dir / "turn.txt").read_text()
        self.stop_template = (prompts_dir / "stop_summary.txt").read_text()

    def _load_schema(self) -> None:
        """Load output schema for validation."""
        schema_path = self.modes_dir / "socrates" / "schemas" / "output.schema.json"
        self.output_schema = json.loads(schema_path.read_text())

    def format_ledger_for_prompt(self, ledger: dict[str, Any], active_only: bool = True) -> str:
        """Format ledger for inclusion in prompts."""
        if active_only:
            # Filter to active items only
            ledger = self._get_active_view(ledger)

        lines = []

        if ledger.get("inquiry"):
            lines.append(f"Inquiry: {ledger['inquiry']}")
        if ledger.get("thesis"):
            lines.append(f"Thesis: {ledger['thesis']}")

        if ledger.get("definitions"):
            lines.append("\nDefinitions:")
            for d in ledger["definitions"]:
                conf = f" (confidence: {d.get('confidence', '?')})" if d.get("confidence") else ""
                lines.append(f"  - [{d.get('id', '?')}] {d['term']}: {d['definition']}{conf}")

        if ledger.get("commitments"):
            lines.append("\nCommitments:")
            for c in ledger["commitments"]:
                lines.append(f"  - [{c.get('id', '?')}] {c['text']} (source: {c.get('source', '?')})")

        if ledger.get("assumptions"):
            lines.append("\nAssumptions:")
            for a in ledger["assumptions"]:
                lines.append(f"  - [{a.get('id', '?')}] {a['text']}")

        if ledger.get("counterexamples"):
            lines.append("\nCounterexamples:")
            for ce in ledger["counterexamples"]:
                lines.append(f"  - [{ce.get('id', '?')}] {ce['text']}")

        if ledger.get("contradictions"):
            lines.append("\nContradictions:")
            for con in ledger["contradictions"]:
                refs = ", ".join(con.get("item_ids", []))
                lines.append(f"  - [{con.get('id', '?')}] {con['text']} (between: {refs})")

        if ledger.get("open_questions"):
            lines.append("\nOpen Questions:")
            for oq in ledger["open_questions"]:
                lines.append(f"  - [{oq.get('id', '?')}] {oq['text']}")

        return "\n".join(lines) if lines else "(empty)"

    def _get_active_view(self, ledger: dict[str, Any]) -> dict[str, Any]:
        """Filter ledger to active items only."""
        view = {"inquiry": ledger.get("inquiry", ""), "thesis": ledger.get("thesis", "")}

        for field in [
            "definitions",
            "commitments",
            "assumptions",
            "counterexamples",
            "contradictions",
            "open_questions",
        ]:
            items = ledger.get(field, [])
            view[field] = [item for item in items if item.get("status", "active") == "active"]

        return view

    def format_conversation_history(self, messages: list[dict[str, str]]) -> str:
        """Format conversation history for prompts."""
        if not messages:
            return "(no prior messages)"

        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]  # Truncate long messages
            if role == "user":
                lines.append(f"INTERLOCUTOR: {content}")
            elif role == "assistant":
                # Try to extract just the question from Socrates responses
                parsed, _ = parse_json(content)
                if parsed and parsed.get("next_question"):
                    lines.append(f"SOCRATES: {parsed['next_question']}")
                else:
                    lines.append(f"SOCRATES: {content[:200]}...")

        return "\n\n".join(lines)

    def build_turn_prompt(self, session: ModeSession, user_message: str) -> str:
        """Build the prompt for a turn."""
        variables = {
            "TURN_NUMBER": session.turn_count + 1,
            "MAX_TURNS": session.max_turns,
            "INQUIRY": session.ledger.get("inquiry", "(not yet defined)"),
            "THESIS": session.ledger.get("thesis", "(not yet stated)"),
            "LEDGER_ACTIVE_VIEW": self.format_ledger_for_prompt(session.ledger, active_only=True),
            "CONVERSATION_HISTORY": self.format_conversation_history(session.messages),
            "USER_MESSAGE": user_message,
        }

        prompt = self.turn_template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt

    def build_summary_prompt(self, session: ModeSession) -> str:
        """Build the prompt for the stop summary."""
        variables = {
            "TURN_COUNT": session.turn_count,
            "INQUIRY": session.ledger.get("inquiry", "(not defined)"),
            "LEDGER_FULL": self.format_ledger_for_prompt(session.ledger, active_only=False),
            "CONVERSATION_HISTORY": self.format_conversation_history(session.messages),
        }

        prompt = self.stop_template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt

    def build_messages_for_llm(self, session: ModeSession, user_message: str) -> list[dict[str, str]]:
        """Build the messages array for the LLM call."""
        turn_prompt = self.build_turn_prompt(session, user_message)

        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": turn_prompt}]

    def process_turn_response(self, raw_response: str, session: ModeSession) -> dict[str, Any]:
        """Process LLM response with 3-pass recovery.

        Returns:
            Dict with:
            - success: bool
            - output: parsed or best-effort output
            - error: error message if any
            - retried: bool if retry was needed
        """
        output, success, error = recover_socrates_turn(raw_response)

        return {"success": success, "output": output, "error": error, "parse_error": not success}

    def should_recommend_stop(self, output: dict[str, Any], session: ModeSession) -> bool:
        """Check if we should recommend stopping."""
        # Check explicit stop recommendation from Socrates
        stop_check = output.get("stop_check", {})
        if stop_check.get("done"):
            return True

        # Check max turns
        if session.turn_count >= session.max_turns:
            return True

        return False

    def get_stop_criteria_met(self, output: dict[str, Any], session: ModeSession) -> list[str]:
        """Get list of stop criteria that are met."""
        criteria = []

        stop_check = output.get("stop_check", {})
        criteria.extend(stop_check.get("criteria", []))

        if session.turn_count >= session.max_turns:
            criteria.append("Max turns reached")

        return criteria
