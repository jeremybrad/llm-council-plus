"""Tests for the Mode Engine and Socrates Mode."""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from backend.modes import ModeRunner, get_mode_runner
from backend.modes.registry import load_mode, list_modes, ModeDefinition
from backend.modes.sessions import SessionStore, ModeSession
from backend.modes.json_recovery import (
    extract_json_block,
    repair_json,
    parse_json,
    extract_best_effort_question,
    recover_socrates_turn
)
from backend.modes.socrates_runner import SocratesRunner


# ========================================
# JSON Recovery Tests
# ========================================

class TestJSONRecovery:
    """Test the 3-pass JSON recovery system."""

    def test_extract_json_block_simple(self):
        """Extract JSON from text with leading/trailing content."""
        text = 'Here is my response: {"question": "What?"} Thanks!'
        result = extract_json_block(text)
        assert result == '{"question": "What?"}'

    def test_extract_json_block_markdown(self):
        """Extract JSON from markdown code block."""
        text = '''Here's the JSON:
```json
{"question": "Why?"}
```
Done.'''
        result = extract_json_block(text)
        assert result == '{"question": "Why?"}'

    def test_extract_json_block_nested(self):
        """Extract nested JSON correctly."""
        text = '{"outer": {"inner": {"deep": true}}}'
        result = extract_json_block(text)
        assert result == '{"outer": {"inner": {"deep": true}}}'

    def test_repair_json_trailing_commas(self):
        """Repair trailing commas before ] or }."""
        text = '{"items": [1, 2, 3,], "name": "test",}'
        result = repair_json(text)
        assert '3,]' not in result
        assert '"test",}' not in result

    def test_repair_json_smart_quotes(self):
        """Repair smart quotes to straight quotes."""
        text = '{"question": "What is this?"}'
        result = repair_json(text)
        assert '\u201c' not in result  # left double curly quote
        assert '\u201d' not in result  # right double curly quote

    def test_parse_json_valid(self):
        """Parse valid JSON directly."""
        text = '{"next_question": "What is X?", "question_type": "definition"}'
        result, error = parse_json(text)
        assert error is None
        assert result["next_question"] == "What is X?"

    def test_parse_json_with_recovery(self):
        """Parse JSON that needs recovery."""
        text = 'Response: {"next_question": "Why?", "items": [1, 2,],}'
        result, error = parse_json(text)
        assert error is None
        assert result["next_question"] == "Why?"

    def test_parse_json_failure(self):
        """Handle unparseable JSON."""
        text = 'This is not JSON at all'
        result, error = parse_json(text)
        assert result is None
        assert error is not None

    def test_extract_best_effort_question(self):
        """Extract question even from malformed response."""
        # From next_question key
        text = '{"next_question": "What do you mean by evidence?"'
        question = extract_best_effort_question(text)
        assert question == "What do you mean by evidence?"

        # From line ending in ?
        text = "I think we should ask: What is the core issue here?"
        question = extract_best_effort_question(text)
        assert "What is the core issue here?" in question

    def test_recover_socrates_turn_success(self):
        """Recover valid Socrates turn."""
        text = json.dumps({
            "next_question": "What do you mean by X?",
            "question_type": "definition",
            "why_this_question": ["Term used ambiguously"],
            "ledger_update": {"thesis": "X is important"},
            "stop_check": {"done": False, "criteria": []}
        })
        output, success, error = recover_socrates_turn(text)
        assert success
        assert error is None
        assert output["next_question"] == "What do you mean by X?"

    def test_recover_socrates_turn_best_effort(self):
        """Recover from malformed Socrates turn."""
        text = 'Let me ask: {"next_question": "What is Y?", broken json here'
        output, success, error = recover_socrates_turn(text)
        assert not success
        assert output["next_question"] == "What is Y?"
        assert output.get("_parse_error") is True


# ========================================
# Mode Registry Tests
# ========================================

class TestModeRegistry:
    """Test mode loading and listing."""

    @pytest.fixture
    def modes_dir(self):
        """Get the actual modes directory."""
        return Path(__file__).parent.parent / "backend" / "modes"

    def test_load_socrates_mode(self, modes_dir):
        """Load the Socrates mode from manifest."""
        mode = load_mode("socrates", modes_dir)
        assert mode.id == "socrates"
        assert mode.display_name == "Socrates Mode"
        assert mode.kind == "interactive"
        assert mode.category == "logic"
        assert "definition" in mode.protocol.get("question_types", []) or len(mode.stop_criteria) > 0

    def test_list_modes(self, modes_dir):
        """List all available modes."""
        modes = list_modes(modes_dir)
        assert len(modes) >= 1
        mode_ids = [m.id for m in modes]
        assert "socrates" in mode_ids

    def test_load_nonexistent_mode(self, modes_dir):
        """Raise error for nonexistent mode."""
        with pytest.raises(FileNotFoundError):
            load_mode("nonexistent", modes_dir)


# ========================================
# Session Store Tests
# ========================================

class TestSessionStore:
    """Test session management."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a session store with temp persist dir."""
        return SessionStore(persist_dir=tmp_path / "sessions")

    def test_create_session(self, store):
        """Create a new session."""
        session = store.create_session(
            mode_id="socrates",
            initial_inquiry="What is justice?",
            max_turns=10
        )
        assert session.session_id is not None
        assert session.mode_id == "socrates"
        assert session.ledger["inquiry"] == "What is justice?"
        assert session.max_turns == 10
        assert session.status == "active"

    def test_get_session(self, store):
        """Retrieve a session by ID."""
        created = store.create_session(mode_id="socrates")
        retrieved = store.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_update_session(self, store):
        """Update session state."""
        session = store.create_session(mode_id="socrates")

        # Update ledger
        new_ledger = {
            "inquiry": "What is courage?",
            "thesis": "Courage is facing fear",
            "definitions": [],
            "commitments": [],
            "assumptions": [],
            "counterexamples": [],
            "contradictions": [],
            "open_questions": []
        }
        store.update_session(session.session_id, ledger=new_ledger)

        updated = store.get_session(session.session_id)
        assert updated.ledger["thesis"] == "Courage is facing fear"

    def test_complete_session(self, store):
        """Mark session as completed."""
        session = store.create_session(mode_id="socrates")
        store.complete_session(session.session_id)

        completed = store.get_session(session.session_id)
        assert completed.status == "completed"

    def test_session_persistence(self, tmp_path):
        """Sessions persist to disk and reload."""
        persist_dir = tmp_path / "persist"

        # Create session in first store
        store1 = SessionStore(persist_dir=persist_dir)
        session = store1.create_session(
            mode_id="socrates",
            initial_inquiry="Test persistence"
        )
        session_id = session.session_id

        # Create new store pointing to same dir
        store2 = SessionStore(persist_dir=persist_dir)
        loaded = store2.get_session(session_id)

        assert loaded is not None
        assert loaded.ledger["inquiry"] == "Test persistence"


# ========================================
# ModeRunner Tests
# ========================================

class TestModeRunner:
    """Test the ModeRunner class."""

    @pytest.fixture
    def runner(self):
        """Create a ModeRunner."""
        return ModeRunner()

    @pytest.mark.asyncio
    async def test_get_mode(self, runner):
        """Get a mode definition."""
        mode = await runner.get_mode("socrates")
        assert mode.id == "socrates"

    @pytest.mark.asyncio
    async def test_get_all_modes(self, runner):
        """List all modes."""
        modes = await runner.get_all_modes()
        assert len(modes) >= 1

    @pytest.mark.asyncio
    async def test_create_session(self, runner):
        """Create a session via runner."""
        session = await runner.create_session(
            mode_id="socrates",
            initial_inquiry="What is truth?"
        )
        assert session.mode_id == "socrates"

    def test_merge_ledger_new_items(self, runner):
        """Merge new ledger items."""
        current = {
            "inquiry": "Original",
            "thesis": "",
            "definitions": [],
            "commitments": []
        }
        update = {
            "thesis": "New thesis",
            "definitions": [
                {"id": "def_001", "term": "X", "definition": "Something", "status": "active"}
            ]
        }
        merged = runner.merge_ledger(current, update)

        assert merged["thesis"] == "New thesis"
        assert len(merged["definitions"]) == 1
        assert merged["definitions"][0]["id"] == "def_001"

    def test_merge_ledger_supersession(self, runner):
        """Merge with supersession marking."""
        current = {
            "definitions": [
                {"id": "def_001", "term": "X", "definition": "Old def", "status": "active"}
            ]
        }
        update = {
            "definitions": [
                {"id": "def_002", "term": "X", "definition": "New def", "status": "active", "supersedes_id": "def_001"}
            ]
        }
        merged = runner.merge_ledger(current, update)

        # Old definition should be marked superseded
        old_def = next(d for d in merged["definitions"] if d["id"] == "def_001")
        assert old_def["status"] == "superseded"
        assert old_def["superseded_by_id"] == "def_002"

        # New definition should be active
        new_def = next(d for d in merged["definitions"] if d["id"] == "def_002")
        assert new_def["status"] == "active"

    def test_get_active_ledger_view(self, runner):
        """Get only active items from ledger."""
        ledger = {
            "inquiry": "Test",
            "thesis": "Thesis",
            "definitions": [
                {"id": "def_001", "text": "Old", "status": "superseded"},
                {"id": "def_002", "text": "Current", "status": "active"}
            ],
            "commitments": [
                {"id": "com_001", "text": "Retracted", "status": "retracted"},
                {"id": "com_002", "text": "Active", "status": "active"}
            ]
        }
        view = runner.get_active_ledger_view(ledger)

        assert len(view["definitions"]) == 1
        assert view["definitions"][0]["id"] == "def_002"
        assert len(view["commitments"]) == 1
        assert view["commitments"][0]["id"] == "com_002"


# ========================================
# SocratesRunner Tests
# ========================================

class TestSocratesRunner:
    """Test Socrates-specific functionality."""

    @pytest.fixture
    def socrates_runner(self):
        """Create a SocratesRunner."""
        modes_dir = Path(__file__).parent.parent / "backend" / "modes"
        return SocratesRunner(modes_dir)

    @pytest.fixture
    def mock_session(self):
        """Create a mock session for testing."""
        return ModeSession(
            session_id="test-session",
            mode_id="socrates",
            created_at="2026-01-11T00:00:00Z",
            updated_at="2026-01-11T00:00:00Z",
            status="active",
            turn_count=2,
            max_turns=12,
            model="ollama:llama3.2",
            ledger={
                "inquiry": "What is justice?",
                "thesis": "Justice involves fairness",
                "definitions": [
                    {"id": "def_001", "term": "justice", "definition": "giving each their due", "status": "active", "turn": 1}
                ],
                "commitments": [
                    {"id": "com_001", "text": "Fairness matters", "source": "human", "status": "active", "turn": 1}
                ],
                "assumptions": [],
                "counterexamples": [],
                "contradictions": [],
                "open_questions": []
            },
            messages=[
                {"role": "user", "content": "I think justice is about fairness"},
                {"role": "assistant", "content": '{"next_question": "What do you mean by fairness?"}'}
            ],
            turn_receipts=[]
        )

    def test_format_ledger_for_prompt(self, socrates_runner, mock_session):
        """Format ledger for prompt inclusion."""
        formatted = socrates_runner.format_ledger_for_prompt(mock_session.ledger)
        assert "What is justice?" in formatted
        assert "justice" in formatted
        assert "def_001" in formatted

    def test_build_turn_prompt(self, socrates_runner, mock_session):
        """Build the turn prompt."""
        prompt = socrates_runner.build_turn_prompt(mock_session, "Fairness means treating everyone equally")
        assert "Turn 3 of 12" in prompt
        assert "Fairness means treating everyone equally" in prompt
        assert "What is justice?" in prompt

    def test_build_messages_for_llm(self, socrates_runner, mock_session):
        """Build messages array for LLM call."""
        messages = socrates_runner.build_messages_for_llm(mock_session, "Test message")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Socrates" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_should_recommend_stop_max_turns(self, socrates_runner, mock_session):
        """Recommend stop when max turns reached."""
        mock_session.turn_count = 12
        mock_session.max_turns = 12
        output = {"stop_check": {"done": False, "criteria": []}}
        assert socrates_runner.should_recommend_stop(output, mock_session) is True

    def test_should_recommend_stop_explicit(self, socrates_runner, mock_session):
        """Recommend stop when Socrates recommends."""
        output = {"stop_check": {"done": True, "criteria": ["Clear definition agreed"]}}
        assert socrates_runner.should_recommend_stop(output, mock_session) is True


# ========================================
# Integration Tests (with mocked LLM)
# ========================================

class TestSocratesModeIntegration:
    """Integration tests with mocked LLM responses."""

    @pytest.fixture
    def mock_llm_response(self):
        """Generate mock Socrates responses."""
        def make_response(turn: int):
            return json.dumps({
                "next_question": f"Question for turn {turn}?",
                "question_type": "definition" if turn == 1 else "assumption",
                "why_this_question": [f"Reason {turn}"],
                "ledger_update": {
                    "thesis": f"Thesis after turn {turn}",
                    "definitions": [
                        {"id": f"def_{turn:03d}", "term": f"term_{turn}", "definition": f"Definition {turn}", "status": "active", "turn": turn}
                    ] if turn == 1 else [],
                    "commitments": [
                        {"id": f"com_{turn:03d}", "text": f"Commitment {turn}", "source": "human", "status": "active", "turn": turn}
                    ]
                },
                "stop_check": {
                    "done": turn >= 3,
                    "criteria": ["Max test turns reached"] if turn >= 3 else []
                }
            })
        return make_response

    @pytest.mark.asyncio
    async def test_multi_turn_session(self, mock_llm_response, tmp_path):
        """Run a multi-turn Socrates session with mocked LLM."""
        from backend.modes import ModeRunner
        from backend.modes.socrates_runner import SocratesRunner

        runner = ModeRunner()
        runner.session_store.persist_dir = tmp_path / "sessions"

        # Create session
        session = await runner.create_session(
            mode_id="socrates",
            initial_inquiry="What is knowledge?"
        )

        socrates_runner = SocratesRunner(runner.modes_dir)

        # Simulate 3 turns
        for turn in range(1, 4):
            user_message = f"User response {turn}"
            session.messages.append({"role": "user", "content": user_message})

            # Mock LLM response
            raw_response = mock_llm_response(turn)
            result = socrates_runner.process_turn_response(raw_response, session)

            assert result["success"]

            # Merge ledger
            if result["output"].get("ledger_update"):
                session.ledger = runner.merge_ledger(session.ledger, result["output"]["ledger_update"])

            session.messages.append({"role": "assistant", "content": raw_response})

        # Verify accumulated state
        assert session.ledger["thesis"] == "Thesis after turn 3"
        assert len(session.ledger["commitments"]) == 3
        assert len(session.messages) == 6  # 3 user + 3 assistant

        # Verify active view only shows active items
        active = runner.get_active_ledger_view(session.ledger)
        assert len(active["commitments"]) == 3


# ========================================
# Glossary Tests
# ========================================

class TestGlossary:
    """Test the fallacies glossary."""

    @pytest.fixture
    def glossary_path(self):
        """Get glossary path."""
        return Path(__file__).parent.parent / "backend" / "resources" / "fallacies_glossary.json"

    def test_glossary_exists(self, glossary_path):
        """Glossary file exists."""
        assert glossary_path.exists()

    def test_glossary_valid_json(self, glossary_path):
        """Glossary is valid JSON."""
        data = json.loads(glossary_path.read_text())
        assert isinstance(data, list)
        assert len(data) >= 10  # At least 10 fallacies

    def test_glossary_entry_structure(self, glossary_path):
        """Each glossary entry has required fields."""
        data = json.loads(glossary_path.read_text())
        required_fields = ["id", "name", "definition", "why_it_weakens", "how_to_fix", "example"]

        for entry in data:
            for field in required_fields:
                assert field in entry, f"Missing field '{field}' in entry: {entry.get('id', 'unknown')}"

    def test_glossary_ids_unique(self, glossary_path):
        """Glossary IDs are unique."""
        data = json.loads(glossary_path.read_text())
        ids = [entry["id"] for entry in data]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in glossary"
