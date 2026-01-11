"""Integration tests for roundtable mode with mocked provider.

These tests verify:
1. 3 rounds are executed and persisted correctly
2. Moderator and chair outputs exist
3. Template variables are substituted (no {QUESTION} literals in output)
4. Abort sets status correctly
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.roundtable import (
    run_roundtable,
    get_default_council,
    AgentConfig,
    RoundtableRun,
    load_template,
    format_round1_prompt,
)
from backend.storage import save_run, get_run, ensure_runs_dir


class TestRoundtableExecution:
    """Test the core roundtable execution flow."""

    @pytest.mark.asyncio
    async def test_full_roundtable_3_rounds(self, mock_query_model, mock_settings, tmp_path):
        """Test that a full roundtable executes 3 rounds with all events."""
        # Setup: Create agents
        agents = [
            AgentConfig(model="mock:builder", role="builder", label="Builder"),
            AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
            AgentConfig(model="mock:contrarian", role="contrarian", label="Contrarian"),
        ]

        conversation_id = "test-conv-001"
        question = "How should we design a REST API for a todo app?"

        # Patch query_model and settings
        with patch("backend.roundtable.query_model", mock_query_model), \
             patch("backend.roundtable.get_settings", return_value=mock_settings):

            events = []
            run_data = None

            async for event in run_roundtable(
                conversation_id=conversation_id,
                question=question,
                agents=agents,
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                context="This is for a personal productivity tool.",
                constraints="Must be RESTful and use JSON.",
                num_rounds=3,
                max_parallel=3,  # Allow all parallel for fast testing
            ):
                events.append(event)
                if event["type"] == "chair_complete":
                    run_data = event.get("run")

            # Verify event sequence
            event_types = [e["type"] for e in events]

            # Should have: roundtable_init, then 3 rounds (each with start, progress*, complete),
            # then moderator_start, moderator_complete, chair_start, chair_complete
            assert "roundtable_init" in event_types
            assert event_types.count("round_start") == 3, "Should have 3 round_start events"
            assert event_types.count("round_complete") == 3, "Should have 3 round_complete events"
            assert "moderator_start" in event_types
            assert "moderator_complete" in event_types
            assert "chair_start" in event_types
            assert "chair_complete" in event_types

            # Verify round data
            assert run_data is not None, "Run data should be returned"
            assert len(run_data["rounds"]) == 3, "Should have 3 rounds persisted"

            # Check round names
            round_names = [r["round_name"] for r in run_data["rounds"]]
            assert round_names == ["opening", "critique", "revision"]

            # Check each round has responses from all agents
            for round_result in run_data["rounds"]:
                assert len(round_result["responses"]) == 3, f"Round {round_result['round_number']} should have 3 responses"

    @pytest.mark.asyncio
    async def test_moderator_and_chair_outputs_exist(self, mock_query_model, mock_settings):
        """Test that moderator and chair produce non-empty outputs."""
        agents = [
            AgentConfig(model="mock:builder", role="builder", label="Builder"),
            AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
        ]

        with patch("backend.roundtable.query_model", mock_query_model), \
             patch("backend.roundtable.get_settings", return_value=mock_settings):

            run_data = None
            async for event in run_roundtable(
                conversation_id="test-conv-002",
                question="Test question",
                agents=agents,
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                num_rounds=3,
            ):
                if event["type"] == "chair_complete":
                    run_data = event.get("run")

            # Verify moderator output
            assert run_data["moderator_summary"] is not None
            assert run_data["moderator_summary"].get("content"), "Moderator content should not be empty"
            assert not run_data["moderator_summary"].get("error"), "Moderator should not have error"

            # Verify chair output
            assert run_data["chair_final"] is not None
            assert run_data["chair_final"].get("content"), "Chair content should not be empty"
            assert not run_data["chair_final"].get("error"), "Chair should not have error"

            # Verify content makes sense
            assert "Summary" in run_data["moderator_summary"]["content"] or "consensus" in run_data["moderator_summary"]["content"].lower()
            assert "Final" in run_data["chair_final"]["content"] or "synthesis" in run_data["chair_final"]["content"].lower()


class TestTemplateSubstitution:
    """Test that template variables are properly substituted."""

    def test_round1_template_substitution(self):
        """Test that Round 1 template substitutes all variables."""
        prompt = format_round1_prompt(
            agent_label="TestAgent",
            question="What is the meaning of life?",
            context="In the context of philosophy.",
            constraints="Be concise."
        )

        # Should NOT contain template placeholders
        assert "{AGENT_LABEL}" not in prompt
        assert "{QUESTION}" not in prompt
        assert "{CONTEXT}" not in prompt
        assert "{CONSTRAINTS}" not in prompt

        # Should contain substituted values
        assert "TestAgent" in prompt
        assert "What is the meaning of life?" in prompt
        assert "philosophy" in prompt
        assert "concise" in prompt

    @pytest.mark.asyncio
    async def test_no_template_literals_in_responses(self, mock_query_model, mock_settings):
        """Test that no {VARIABLE} patterns appear in final output."""
        agents = [
            AgentConfig(model="mock:builder", role="builder", label="Builder"),
        ]

        with patch("backend.roundtable.query_model", mock_query_model), \
             patch("backend.roundtable.get_settings", return_value=mock_settings):

            run_data = None
            async for event in run_roundtable(
                conversation_id="test-conv-003",
                question="Design a system",
                agents=agents,
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                num_rounds=2,
            ):
                if event["type"] == "chair_complete":
                    run_data = event.get("run")

            # Check all round responses for template literals
            for round_result in run_data["rounds"]:
                for response in round_result["responses"]:
                    content = response.get("content", "")
                    assert "{QUESTION}" not in content, f"Found {{QUESTION}} literal in {response['agent_label']}"
                    assert "{CONTEXT}" not in content, f"Found {{CONTEXT}} literal in {response['agent_label']}"
                    assert "{AGENT_LABEL}" not in content, f"Found {{AGENT_LABEL}} literal in {response['agent_label']}"

            # Check moderator and chair
            mod_content = run_data["moderator_summary"].get("content", "")
            chair_content = run_data["chair_final"].get("content", "")

            assert "{QUESTION}" not in mod_content, "Found {QUESTION} in moderator output"
            assert "{QUESTION}" not in chair_content, "Found {QUESTION} in chair output"

    @pytest.mark.asyncio
    async def test_query_model_receives_substituted_prompts(self, mock_query_model, mock_settings):
        """Test that query_model is called with properly substituted prompts."""
        agents = [
            AgentConfig(model="mock:builder", role="builder", label="Builder"),
        ]

        with patch("backend.roundtable.query_model", mock_query_model), \
             patch("backend.roundtable.get_settings", return_value=mock_settings):

            async for event in run_roundtable(
                conversation_id="test-conv-004",
                question="Build a feature",
                agents=agents,
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                num_rounds=1,
            ):
                pass  # Consume all events

            # Check call log for template literals in user prompts
            for call in mock_query_model.call_log:
                for msg in call["messages"]:
                    if msg["role"] == "user":
                        content = msg["content"]
                        # These should NOT appear - they should be substituted
                        assert "{QUESTION}" not in content, f"Found {{QUESTION}} in call to {call['model']}"
                        assert "{AGENT_LABEL}" not in content, f"Found {{AGENT_LABEL}} in call to {call['model']}"


class TestAbortBehavior:
    """Test abort/cancellation behavior."""

    @pytest.mark.asyncio
    async def test_abort_sets_status(self, mock_query_model, mock_settings):
        """Test that aborting a roundtable sets status to 'aborted'."""
        agents = [
            AgentConfig(model="mock:builder", role="builder", label="Builder"),
            AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
        ]

        # Create a mock that will be cancelled mid-execution
        call_count = 0
        original_side_effect = mock_query_model.side_effect

        async def slow_mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:  # Cancel after a couple of calls
                raise asyncio.CancelledError("Simulated abort")
            return await original_side_effect(*args, **kwargs)

        mock_query_model.side_effect = slow_mock

        with patch("backend.roundtable.query_model", mock_query_model), \
             patch("backend.roundtable.get_settings", return_value=mock_settings):

            events = []
            try:
                async for event in run_roundtable(
                    conversation_id="test-conv-abort",
                    question="Test abort",
                    agents=agents,
                    moderator_model="mock:moderator",
                    chair_model="mock:chair",
                    num_rounds=3,
                    max_parallel=1,  # Sequential to control timing
                ):
                    events.append(event)
            except asyncio.CancelledError:
                pass  # Expected

            # Should have aborted event
            event_types = [e["type"] for e in events]
            assert "roundtable_aborted" in event_types, "Should have roundtable_aborted event"

            # Get the aborted event
            aborted_event = next(e for e in events if e["type"] == "roundtable_aborted")
            run_data = aborted_event.get("run")

            assert run_data["status"] == "aborted", "Status should be 'aborted'"
            assert run_data["completed_at"] is not None, "Should have completed_at timestamp"


class TestCouncilConfiguration:
    """Test council/agent configuration."""

    def test_get_default_council_assigns_roles(self):
        """Test that get_default_council assigns roles correctly."""
        models = ["mock:m1", "mock:m2", "mock:m3"]
        council = get_default_council(models)

        assert len(council) == 3
        assert council[0].role == "builder"
        assert council[0].label == "Builder"
        assert council[1].role == "skeptic"
        assert council[1].label == "Skeptic"
        assert council[2].role == "contrarian"
        assert council[2].label == "Contrarian"

    def test_get_default_council_cycles_roles(self):
        """Test that roles cycle for councils larger than 6 agents."""
        models = [f"mock:m{i}" for i in range(8)]
        council = get_default_council(models)

        assert len(council) == 8
        # Roles should cycle
        assert council[6].role == "builder"  # Cycles back
        assert council[7].role == "skeptic"


class TestRoundtableStorage:
    """Test roundtable run storage."""

    def test_save_and_load_run(self, tmp_path):
        """Test saving and loading a run file."""
        # Patch RUNS_DIR to use temp directory
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        with patch("backend.storage.RUNS_DIR", runs_dir):
            run_data = {
                "run_id": "test-run-001",
                "conversation_id": "test-conv-001",
                "question": "Test question",
                "status": "completed",
                "council": [
                    {"model": "mock:m1", "role": "builder", "label": "Builder"}
                ],
                "rounds": [
                    {
                        "round_number": 1,
                        "round_name": "opening",
                        "responses": [
                            {"agent_label": "Builder", "model": "mock:m1", "role": "Builder", "content": "Test"}
                        ],
                        "started_at": "2024-01-01T00:00:00",
                        "completed_at": "2024-01-01T00:01:00",
                    }
                ],
                "moderator_summary": {"model": "mock:mod", "content": "Summary", "error": False},
                "chair_final": {"model": "mock:chair", "content": "Final", "error": False},
                "created_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:02:00",
            }

            save_run(run_data)

            # Verify file exists
            run_path = runs_dir / "test-conv-001" / "test-run-001.json"
            assert run_path.exists(), "Run file should exist"

            # Load and verify
            loaded = get_run("test-conv-001", "test-run-001")
            assert loaded is not None
            assert loaded["run_id"] == "test-run-001"
            assert loaded["status"] == "completed"
            assert len(loaded["rounds"]) == 1


class TestTemplateLoading:
    """Test prompt template loading."""

    def test_load_global_system_template(self):
        """Test loading global system template."""
        template = load_template("global_system")
        assert template, "Global system template should load"
        assert len(template) > 100, "Template should have substantial content"

    def test_load_role_templates(self):
        """Test loading all role templates."""
        roles = ["builder", "skeptic", "contrarian", "historian", "pragmatist", "stylist"]

        for role in roles:
            template = load_template(f"roles/{role}")
            assert template, f"Role template for {role} should load"
            assert role.lower() in template.lower() or len(template) > 50, f"Template should mention {role} or have content"

    def test_load_round_templates(self):
        """Test loading round templates."""
        rounds = ["r1_opening", "r2_critique", "r3_revision"]

        for round_name in rounds:
            template = load_template(f"rounds/{round_name}")
            assert template, f"Round template {round_name} should load"

    def test_load_moderator_template(self):
        """Test loading moderator template."""
        template = load_template("moderator")
        assert template, "Moderator template should load"
        assert "{QUESTION}" in template, "Moderator template should have {QUESTION} placeholder"

    def test_load_chair_template(self):
        """Test loading chair template."""
        template = load_template("chair")
        assert template, "Chair template should load"
        assert "{QUESTION}" in template, "Chair template should have {QUESTION} placeholder"


class TestEventStructure:
    """Test SSE event structure matches frontend expectations."""

    @pytest.mark.asyncio
    async def test_roundtable_init_event_structure(self, mock_query_model, mock_settings):
        """Test roundtable_init event has expected structure."""
        agents = [
            AgentConfig(model="mock:builder", role="builder", label="Builder"),
        ]

        with patch("backend.roundtable.query_model", mock_query_model), \
             patch("backend.roundtable.get_settings", return_value=mock_settings):

            init_event = None
            async for event in run_roundtable(
                conversation_id="test-conv-events",
                question="Test",
                agents=agents,
                moderator_model="mock:mod",
                chair_model="mock:chair",
                num_rounds=1,
            ):
                if event["type"] == "roundtable_init":
                    init_event = event
                    break

            assert init_event is not None
            assert "run_id" in init_event
            assert "total_rounds" in init_event
            assert "council_members" in init_event
            assert isinstance(init_event["council_members"], list)

    @pytest.mark.asyncio
    async def test_round_progress_event_structure(self, mock_query_model, mock_settings):
        """Test round_progress event has expected structure."""
        agents = [
            AgentConfig(model="mock:builder", role="builder", label="Builder"),
        ]

        with patch("backend.roundtable.query_model", mock_query_model), \
             patch("backend.roundtable.get_settings", return_value=mock_settings):

            progress_event = None
            async for event in run_roundtable(
                conversation_id="test-conv-events2",
                question="Test",
                agents=agents,
                moderator_model="mock:mod",
                chair_model="mock:chair",
                num_rounds=1,
            ):
                if event["type"] == "round_progress":
                    progress_event = event
                    break

            assert progress_event is not None
            assert "round_number" in progress_event
            assert "count" in progress_event
            assert "total" in progress_event
            assert "response" in progress_event

            # Response should have expected fields
            response = progress_event["response"]
            assert "agent_label" in response
            assert "model" in response
            assert "role" in response
            assert "content" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
