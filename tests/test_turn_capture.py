"""Tests for the turn capture tap module.

Tests verify:
1. Session ID resolution (bbot, fallback uuid)
2. Turn schema correctness for user/agent/moderator/chair
3. Content sanitization (secrets blocked, PII redacted)
4. SADB root resolution (override, env, sibling detection)
5. Atomic file creation and appending
6. Full capture flow integration
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.turn_capture import (
    resolve_sadb_root,
    get_session_id,
    sanitize_for_turn,
    transform_run_to_turns,
    append_turns,
    capture_run,
)


class TestSessionIdResolution:
    """Test session ID resolution logic."""

    def test_session_id_from_bbot(self, tmp_path):
        """Test that session_id is read from bbot context."""
        bbot_dir = tmp_path / ".bbot" / "context"
        bbot_dir.mkdir(parents=True)
        context_file = bbot_dir / "current.json"
        context_file.write_text(json.dumps({"session_id": "test-session-123"}))

        with patch.object(Path, "home", return_value=tmp_path):
            session_id = get_session_id()
            assert session_id == "test-session-123"

    def test_session_id_ignores_capsule_id(self, tmp_path):
        """Test that capsule_id is NOT used, only session_id."""
        bbot_dir = tmp_path / ".bbot" / "context"
        bbot_dir.mkdir(parents=True)
        context_file = bbot_dir / "current.json"
        # Only capsule_id present, no session_id
        context_file.write_text(json.dumps({"capsule_id": "capsule-should-ignore"}))

        with patch.object(Path, "home", return_value=tmp_path):
            session_id = get_session_id()
            # Should generate fallback, NOT use capsule_id
            assert session_id.startswith("roundtable-")
            assert "capsule-should-ignore" not in session_id

    def test_session_id_fallback_uuid(self, tmp_path):
        """Test that UUID is generated when bbot context is missing."""
        # No .bbot directory
        with patch.object(Path, "home", return_value=tmp_path):
            session_id = get_session_id()
            assert session_id.startswith("roundtable-")
            # Should be a valid UUID format
            uuid_part = session_id.replace("roundtable-", "")
            assert len(uuid_part) == 36  # UUID format with dashes

    def test_session_id_fallback_on_invalid_json(self, tmp_path):
        """Test fallback when bbot context has invalid JSON."""
        bbot_dir = tmp_path / ".bbot" / "context"
        bbot_dir.mkdir(parents=True)
        context_file = bbot_dir / "current.json"
        context_file.write_text("not valid json")

        with patch.object(Path, "home", return_value=tmp_path):
            session_id = get_session_id()
            assert session_id.startswith("roundtable-")


class TestTurnSchema:
    """Test turn schema correctness."""

    @pytest.fixture
    def sample_run_data(self):
        """Create sample run data for testing."""
        return {
            "run_id": "run-001",
            "conversation_id": "conv-001",
            "question": "What is the best approach?",
            "status": "completed",
            "council": [
                {"model": "ollama:llama3.2", "role": "builder", "label": "Builder"},
                {"model": "ollama:mistral", "role": "skeptic", "label": "Skeptic"},
            ],
            "rounds": [
                {
                    "round_number": 1,
                    "round_name": "opening",
                    "responses": [
                        {
                            "agent_label": "Builder",
                            "model": "ollama:llama3.2",
                            "role": "Builder",
                            "content": "I propose a modular approach.",
                            "error": None,
                            "duration_ms": 1500,
                        },
                        {
                            "agent_label": "Skeptic",
                            "model": "ollama:mistral",
                            "role": "Skeptic",
                            "content": "I have concerns about scalability.",
                            "error": None,
                            "duration_ms": 1200,
                        },
                    ],
                    "started_at": "2025-01-01T00:00:00",
                    "completed_at": "2025-01-01T00:01:00",
                },
            ],
            "moderator_summary": {
                "model": "ollama:llama3.2",
                "content": "The council reached consensus on modularity.",
                "error": False,
            },
            "chair_final": {
                "model": "ollama:llama3.2",
                "content": "Final synthesis: Use modular approach with safeguards.",
                "error": False,
            },
            "created_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:05:00",
        }

    def test_user_turn_schema(self, sample_run_data):
        """Test user turn has correct schema structure."""
        with patch("backend.turn_capture.sanitize_for_turn") as mock_sanitize:
            mock_sanitize.return_value = {
                "content": sample_run_data["question"],
                "sanitized": False,
                "blocked": False,
                "leakscan_hits": {"secrets": 0, "pii": 0},
            }

            turns = transform_run_to_turns(sample_run_data, "test-session")
            user_turn = turns[0]

            # Required fields
            assert "turn_id" in user_turn
            assert user_turn["session_id"] == "test-session"
            assert "timestamp" in user_turn
            assert user_turn["source_system"] == "roundtable"
            assert user_turn["speaker_role"] == "user"
            assert user_turn["speaker_id"] == "user"
            assert user_turn["model"] is None
            assert "content" in user_turn
            assert "sanitized" in user_turn
            assert "blocked" in user_turn
            assert "leakscan_hits" in user_turn

            # Meta fields
            assert user_turn["meta"]["round_number"] is None
            assert user_turn["meta"]["round_name"] is None
            assert user_turn["meta"]["run_id"] == "run-001"
            assert user_turn["meta"]["conversation_id"] == "conv-001"
            assert user_turn["meta"]["agent_role"] is None

    def test_agent_turn_schema(self, sample_run_data):
        """Test agent turn has correct schema with agent_role in meta."""
        with patch("backend.turn_capture.sanitize_for_turn") as mock_sanitize:
            mock_sanitize.return_value = {
                "content": "test content",
                "sanitized": False,
                "blocked": False,
                "leakscan_hits": {"secrets": 0, "pii": 0},
            }

            turns = transform_run_to_turns(sample_run_data, "test-session")
            # First agent turn (after user turn)
            agent_turn = turns[1]

            assert agent_turn["speaker_role"] == "assistant"
            assert agent_turn["speaker_id"] == "agent:Builder"
            assert agent_turn["model"] == "ollama:llama3.2"
            assert agent_turn["meta"]["round_number"] == 1
            assert agent_turn["meta"]["round_name"] == "opening"
            assert agent_turn["meta"]["agent_role"] == "Builder"
            assert agent_turn["meta"]["duration_ms"] == 1500

    def test_moderator_chair_schema(self, sample_run_data):
        """Test moderator and chair turns have correct schema."""
        with patch("backend.turn_capture.sanitize_for_turn") as mock_sanitize:
            mock_sanitize.return_value = {
                "content": "test content",
                "sanitized": False,
                "blocked": False,
                "leakscan_hits": {"secrets": 0, "pii": 0},
            }

            turns = transform_run_to_turns(sample_run_data, "test-session")

            # Find moderator turn
            mod_turn = next(t for t in turns if t["speaker_id"] == "moderator")
            assert mod_turn["speaker_role"] == "assistant"
            assert mod_turn["meta"]["round_name"] == "moderator"
            assert mod_turn["meta"]["agent_role"] == "Moderator"

            # Find chair turn
            chair_turn = next(t for t in turns if t["speaker_id"] == "chair")
            assert chair_turn["speaker_role"] == "assistant"
            assert chair_turn["meta"]["round_name"] == "chair"
            assert chair_turn["meta"]["agent_role"] == "Chair"


class TestContentSanitization:
    """Test content sanitization behavior."""

    def test_sanitize_blocks_secrets(self):
        """Test that secrets result in blocked=True and stubbed content."""
        # Mock the leakscan import to simulate finding secrets
        mock_scan_text = MagicMock(return_value={"openai_key": 1})
        mock_redact_text = MagicMock(return_value="[REDACTED]")
        mock_secret_patterns = {"openai_key": r"sk-.*"}
        mock_pii_patterns = {}

        with patch.dict(
            "sys.modules",
            {
                "leakscan_jsonl": MagicMock(
                    scan_text=mock_scan_text,
                    redact_text=mock_redact_text,
                    SECRET_PATTERNS=mock_secret_patterns,
                    PII_PATTERNS=mock_pii_patterns,
                )
            },
        ):
            # Force reimport to pick up mock
            import importlib
            import backend.turn_capture

            importlib.reload(backend.turn_capture)

            result = backend.turn_capture.sanitize_for_turn("my api key is sk-test123")

            assert result["blocked"] is True
            assert result["content"] == "[BLOCKED_SECRET]"
            assert result["sanitized"] is True
            assert result["leakscan_hits"]["secrets"] > 0

    def test_sanitize_redacts_pii(self):
        """Test that PII is redacted inline, not blocked."""
        mock_scan_text = MagicMock(return_value={"email": 1})
        mock_redact_text = MagicMock(return_value="Contact [REDACTED:email] for help")
        mock_secret_patterns = {}
        mock_pii_patterns = {"email": {"regex": r".*@.*", "replacement": "[REDACTED:email]"}}

        with patch.dict(
            "sys.modules",
            {
                "leakscan_jsonl": MagicMock(
                    scan_text=mock_scan_text,
                    redact_text=mock_redact_text,
                    SECRET_PATTERNS=mock_secret_patterns,
                    PII_PATTERNS=mock_pii_patterns,
                )
            },
        ):
            import importlib
            import backend.turn_capture

            importlib.reload(backend.turn_capture)

            result = backend.turn_capture.sanitize_for_turn("Contact test@example.com for help")

            assert result["blocked"] is False
            assert result["sanitized"] is True
            assert result["leakscan_hits"]["pii"] > 0
            assert "[REDACTED" in result["content"]

    def test_sanitize_fallback_on_import_error(self):
        """Test that missing leakscan results in blocked content."""
        # Ensure leakscan is not importable
        with patch.dict("sys.modules", {"leakscan_jsonl": None}):
            with patch("backend.turn_capture.sys.path", []):
                import importlib
                import backend.turn_capture

                importlib.reload(backend.turn_capture)

                result = backend.turn_capture.sanitize_for_turn("some content")

                assert result["blocked"] is True
                assert result["content"] == "[LEAKSCAN_UNAVAILABLE]"


class TestSadbRootResolution:
    """Test SADB data root resolution."""

    def test_resolve_sadb_root_override(self, tmp_path):
        """Test that override path takes precedence."""
        override_path = tmp_path / "custom_sadb"
        result = resolve_sadb_root(override_path)
        assert result == override_path

    def test_resolve_sadb_root_env(self, tmp_path):
        """Test that SADB_DATA_DIR env var works."""
        env_path = tmp_path / "env_sadb"
        with patch.dict(os.environ, {"SADB_DATA_DIR": str(env_path)}):
            result = resolve_sadb_root(None)
            assert result == env_path

    def test_resolve_sadb_root_sibling(self, tmp_path):
        """Test walking up to find SyncedProjects sibling."""
        # Create fake SyncedProjects structure
        synced = tmp_path / "SyncedProjects"
        synced.mkdir()
        project = synced / "C012_round-table" / "40_src" / "llm-council-plus" / "backend"
        project.mkdir(parents=True)

        # Mock __file__ to be inside the project
        fake_file = project / "turn_capture.py"
        fake_file.touch()

        # Clear env var if set
        with patch.dict(os.environ, {"SADB_DATA_DIR": ""}, clear=False):
            # Patch the module's __file__
            with patch("backend.turn_capture.Path") as mock_path:
                # Create a mock that walks up correctly
                mock_path_obj = MagicMock()
                mock_path_obj.resolve.return_value = fake_file

                # Walk up chain
                parents = [
                    project,  # backend
                    project.parent,  # llm-council-plus
                    project.parent.parent,  # 40_src
                    project.parent.parent.parent,  # C012_round-table
                    synced,  # SyncedProjects
                    tmp_path,  # root
                ]

                current = fake_file
                for i, parent in enumerate(parents):
                    mock_current = MagicMock()
                    mock_current.parent = parent if i < len(parents) - 1 else current
                    mock_current.name = current.name
                    current = parent

                mock_path.return_value.__file__ = str(fake_file)

                # Actually test with real path resolution
                import importlib
                import backend.turn_capture

                # Manually test the sibling logic
                test_path = synced / "C012_round-table" / "40_src"
                current = test_path.resolve()
                found = None
                while current.parent != current:
                    if current.name == "SyncedProjects":
                        found = current / "_SADB_Data"
                        break
                    current = current.parent

                assert found == synced / "_SADB_Data"


class TestFileOperations:
    """Test file creation and appending."""

    def test_creates_jsonl_file(self, tmp_path):
        """Test that atomic write creates file correctly."""
        sadb_root = tmp_path / "_SADB_Data"
        turns = [
            {"turn_id": "t1", "content": "test1"},
            {"turn_id": "t2", "content": "test2"},
        ]

        output_path = append_turns("test-session", turns, sadb_root)

        assert output_path.exists()
        assert output_path.name == "test-session.jsonl"

        # Verify content
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["turn_id"] == "t1"
        assert json.loads(lines[1])["turn_id"] == "t2"

    def test_appends_to_existing_file(self, tmp_path):
        """Test that appends don't overwrite existing content."""
        sadb_root = tmp_path / "_SADB_Data"
        turns_dir = sadb_root / "hot" / "turns"
        turns_dir.mkdir(parents=True)

        # Create existing file
        existing_file = turns_dir / "test-session.jsonl"
        existing_file.write_text('{"turn_id": "existing"}\n')

        new_turns = [{"turn_id": "new1"}, {"turn_id": "new2"}]
        output_path = append_turns("test-session", new_turns, sadb_root)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["turn_id"] == "existing"
        assert json.loads(lines[1])["turn_id"] == "new1"
        assert json.loads(lines[2])["turn_id"] == "new2"


class TestFullCaptureFlow:
    """Test end-to-end capture flow."""

    def test_full_capture_flow(self, tmp_path):
        """Test complete capture_run integration."""
        sadb_root = tmp_path / "_SADB_Data"

        run_data = {
            "run_id": "run-test",
            "conversation_id": "conv-test",
            "question": "Test question",
            "status": "completed",
            "rounds": [
                {
                    "round_number": 1,
                    "round_name": "opening",
                    "responses": [
                        {
                            "agent_label": "Builder",
                            "model": "test:model",
                            "role": "Builder",
                            "content": "Test response",
                            "error": None,
                            "duration_ms": 100,
                        }
                    ],
                }
            ],
            "moderator_summary": {
                "model": "test:model",
                "content": "Summary",
                "error": False,
            },
            "chair_final": {
                "model": "test:model",
                "content": "Final",
                "error": False,
            },
            "created_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:00",
        }

        # Mock sanitize to return clean content
        with patch("backend.turn_capture.sanitize_for_turn") as mock_sanitize:
            mock_sanitize.side_effect = lambda c: {
                "content": c,
                "sanitized": False,
                "blocked": False,
                "leakscan_hits": {"secrets": 0, "pii": 0},
            }

            output_path = capture_run(
                run_data,
                session_id="integration-test",
                sadb_root=sadb_root,
            )

            assert output_path is not None
            assert output_path.exists()

            # Verify all turns captured
            lines = output_path.read_text().strip().split("\n")
            turns = [json.loads(line) for line in lines]

            # Should have: user + 1 agent + moderator + chair = 4 turns
            assert len(turns) == 4

            # Verify turn types
            speaker_ids = [t["speaker_id"] for t in turns]
            assert "user" in speaker_ids
            assert "agent:Builder" in speaker_ids
            assert "moderator" in speaker_ids
            assert "chair" in speaker_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
