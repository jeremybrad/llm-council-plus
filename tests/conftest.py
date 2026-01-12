"""Pytest fixtures for roundtable tests."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

import pytest


# Mock response generator for deterministic testing
def make_mock_response(agent_label: str, role: str, round_num: int) -> str:
    """Generate a deterministic mock response for testing.

    Args:
        agent_label: Agent label (e.g., "Builder")
        role: Agent role key (e.g., "builder")
        round_num: Round number (1, 2, or 3)

    Returns:
        Deterministic response string
    """
    responses = {
        1: {
            "builder": f"[{agent_label}] Round 1 opening: I propose we build a solution using modular components. Key considerations include scalability and maintainability.",
            "skeptic": f"[{agent_label}] Round 1 opening: I have concerns about the proposed approach. What about edge cases and failure modes?",
            "contrarian": f"[{agent_label}] Round 1 opening: I disagree with the conventional wisdom here. Let me propose an alternative perspective.",
        },
        2: {
            "builder": f"[{agent_label}] Round 2 critique: The skeptic raises valid concerns. We should add error handling. My question: What specific edge cases worry you most?",
            "skeptic": f"[{agent_label}] Round 2 critique: The builder's modular approach is good, but needs more concrete implementation details. Question: How will modules communicate?",
            "contrarian": f"[{agent_label}] Round 2 critique: Both perspectives miss the fundamental issue. Question: Have we validated the problem statement?",
        },
        3: {
            "builder": f"[{agent_label}] Round 3 revision: Based on feedback, I've revised my proposal to include: 1) Error handling per skeptic, 2) Clear module interfaces, 3) Problem validation.",
            "skeptic": f"[{agent_label}] Round 3 revision: After reviewing critiques, I acknowledge the builder's approach is sound. Adding monitoring would address my remaining concerns.",
            "contrarian": f"[{agent_label}] Round 3 revision: I maintain my alternative view but concede the mainstream approach has merit when combined with proper validation.",
        },
    }

    return responses.get(round_num, {}).get(role, f"[{agent_label}] Round {round_num} response for {role}")


def make_moderator_response() -> str:
    """Generate deterministic moderator summary."""
    return """## Moderator Summary

### Points of Consensus
- Modular architecture is appropriate
- Error handling is necessary
- Problem validation is important

### Remaining Disagreements
- Contrarian maintains alternative perspective
- Skeptic wants more monitoring

### Recommended Next Steps
1. Finalize module interfaces
2. Add comprehensive error handling
3. Implement monitoring

### Minority Report
The Contrarian perspective deserves consideration for edge cases."""


def make_chair_response() -> str:
    """Generate deterministic chair final synthesis."""
    return """# Final Synthesis

## Executive Summary
The council has reached consensus on a modular approach with robust error handling.

## Key Decisions
1. **Architecture**: Modular components with clear interfaces
2. **Reliability**: Comprehensive error handling and monitoring
3. **Validation**: Problem statement validated before implementation

## Implementation Roadmap
1. Define module interfaces
2. Implement core functionality
3. Add error handling
4. Deploy monitoring

## Minority Views Addressed
The Contrarian's alternative perspective has been incorporated through the validation step.

---
*Synthesized by the Chair*"""


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    conversations_dir = data_dir / "conversations"
    conversations_dir.mkdir()

    runs_dir = data_dir / "runs"
    runs_dir.mkdir()

    return data_dir


@pytest.fixture
def mock_query_model():
    """Create a mock for query_model that returns deterministic responses.

    The mock tracks which calls have been made and returns appropriate
    responses based on the prompt content.
    """
    call_log = []

    async def mock_impl(model: str, messages: List[Dict[str, str]], timeout: float = 120.0, temperature: float = 0.7) -> Dict[str, Any]:
        """Mock implementation of query_model."""
        call_log.append({
            "model": model,
            "messages": messages,
            "timeout": timeout,
            "temperature": temperature,
        })

        # Analyze the prompt to determine what kind of response to generate
        user_content = ""
        system_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "system":
                system_content = msg["content"]

        # Determine response type based on prompt patterns
        content = ""

        # Check for moderator prompt
        if "synthesize the council's deliberation" in user_content.lower() or "moderator" in system_content.lower():
            content = make_moderator_response()

        # Check for chair prompt
        elif "chair" in system_content.lower() or "final answer" in user_content.lower():
            content = make_chair_response()

        # Round 1 (opening) - contains question and constraints
        elif "opening statement" in user_content.lower() or "round 1" in user_content.lower():
            # Extract agent label and role from prompt
            agent_label = "Agent"
            role = "builder"

            if "Builder" in system_content:
                agent_label = "Builder"
                role = "builder"
            elif "Skeptic" in system_content:
                agent_label = "Skeptic"
                role = "skeptic"
            elif "Contrarian" in system_content:
                agent_label = "Contrarian"
                role = "contrarian"
            elif "Historian" in system_content:
                agent_label = "Historian"
                role = "historian"
            elif "Pragmatist" in system_content:
                agent_label = "Pragmatist"
                role = "pragmatist"
            elif "Stylist" in system_content:
                agent_label = "Stylist"
                role = "stylist"

            content = make_mock_response(agent_label, role, 1)

        # Round 2 (critique) - references other messages
        elif "critique" in user_content.lower() or "round 2" in user_content.lower():
            agent_label = "Agent"
            role = "builder"

            if "Builder" in system_content:
                agent_label = "Builder"
                role = "builder"
            elif "Skeptic" in system_content:
                agent_label = "Skeptic"
                role = "skeptic"
            elif "Contrarian" in system_content:
                agent_label = "Contrarian"
                role = "contrarian"

            content = make_mock_response(agent_label, role, 2)

        # Round 3 (revision) - references your round 1
        elif "revis" in user_content.lower() or "round 3" in user_content.lower():
            agent_label = "Agent"
            role = "builder"

            if "Builder" in system_content:
                agent_label = "Builder"
                role = "builder"
            elif "Skeptic" in system_content:
                agent_label = "Skeptic"
                role = "skeptic"
            elif "Contrarian" in system_content:
                agent_label = "Contrarian"
                role = "contrarian"

            content = make_mock_response(agent_label, role, 3)

        else:
            # Generic response
            content = f"[Mock response for {model}]"

        return {
            "content": content,
            "model": model,
            "error": False,
        }

    mock = AsyncMock(side_effect=mock_impl)
    mock.call_log = call_log

    return mock


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    class MockSettings:
        council_temperature = 0.5
        stage2_temperature = 0.3
        chairman_temperature = 0.4
        roundtable_debug_prompts = False  # Debug off by default in tests
        # OpenAI-compatible endpoint settings
        council_models = []  # Empty by default, tests can override
        chairman_model = "mock:chairman"
        roundtable_num_rounds = 3
        roundtable_max_parallel = 2

    return MockSettings()
