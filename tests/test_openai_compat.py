"""Tests for OpenAI-compatible API endpoint.

These tests verify:
1. Request/response format matches OpenAI spec
2. Streaming responses work correctly
3. Model variants (roundtable:fast, etc.) are parsed
4. Configuration options work
5. Error handling is correct
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.openai_compat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    extract_user_question,
    extract_context,
    parse_model_config,
    generate_openai_stream,
    generate_non_streaming_response,
    get_available_models,
)


class TestRequestParsing:
    """Test parsing of OpenAI-format requests."""

    def test_extract_user_question_single_message(self):
        """Test extracting question from a single user message."""
        messages = [
            ChatMessage(role="user", content="What is the meaning of life?")
        ]
        question = extract_user_question(messages)
        assert question == "What is the meaning of life?"

    def test_extract_user_question_with_system(self):
        """Test that system messages don't affect question extraction."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Explain quantum computing"),
        ]
        question = extract_user_question(messages)
        assert question == "Explain quantum computing"

    def test_extract_user_question_conversation(self):
        """Test extracting last user message from conversation."""
        messages = [
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Follow-up question"),
        ]
        question = extract_user_question(messages)
        assert question == "Follow-up question"

    def test_extract_user_question_empty(self):
        """Test handling of empty messages."""
        messages = []
        question = extract_user_question(messages)
        assert question == ""

    def test_extract_user_question_no_user(self):
        """Test handling when no user message exists."""
        messages = [
            ChatMessage(role="system", content="System only"),
        ]
        question = extract_user_question(messages)
        assert question == ""

    def test_extract_context_with_system(self):
        """Test extracting context including system prompt."""
        messages = [
            ChatMessage(role="system", content="You are an expert."),
            ChatMessage(role="user", content="Question"),
        ]
        context = extract_context(messages)
        assert "[System Instructions]" in context
        assert "You are an expert." in context

    def test_extract_context_conversation_history(self):
        """Test extracting conversation history as context."""
        messages = [
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Second question"),
        ]
        context = extract_context(messages)
        assert "[Previous User Message]" in context
        assert "First question" in context
        assert "[Previous Assistant Response]" in context
        assert "First answer" in context
        # Second question is not in context (it's the question)
        assert "Second question" not in context


class TestModelParsing:
    """Test model string parsing for configuration."""

    def test_parse_model_default(self):
        """Test parsing default roundtable model."""
        config = parse_model_config("roundtable")
        assert config == {}

    def test_parse_model_with_rounds(self):
        """Test parsing model with round count."""
        config = parse_model_config("roundtable:3")
        assert config["num_rounds"] == 3

        config = parse_model_config("roundtable:1")
        assert config["num_rounds"] == 1

    def test_parse_model_fast(self):
        """Test parsing fast mode."""
        config = parse_model_config("roundtable:fast")
        assert config["num_rounds"] == 1

    def test_parse_model_thorough(self):
        """Test parsing thorough mode."""
        config = parse_model_config("roundtable:thorough")
        assert config["num_rounds"] == 3

    def test_parse_model_deep(self):
        """Test parsing deep mode."""
        config = parse_model_config("roundtable:deep")
        assert config["num_rounds"] == 5


class TestModelsEndpoint:
    """Test the models listing endpoint."""

    def test_get_available_models(self):
        """Test that models list includes expected variants."""
        models = get_available_models()

        assert len(models) >= 3
        model_ids = [m["id"] for m in models]

        assert "roundtable" in model_ids
        assert "roundtable:fast" in model_ids
        assert "roundtable:thorough" in model_ids

    def test_models_have_correct_structure(self):
        """Test that models have OpenAI-compatible structure."""
        models = get_available_models()

        for model in models:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "created" in model
            assert "owned_by" in model


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_default_values(self):
        """Test default values are set correctly."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="test")]
        )
        assert request.model == "roundtable"
        assert request.stream is True
        assert request.council_models is None
        assert request.num_rounds is None

    def test_custom_values(self):
        """Test custom values are preserved."""
        request = ChatCompletionRequest(
            model="roundtable:fast",
            messages=[ChatMessage(role="user", content="test")],
            stream=False,
            council_models=["model1", "model2"],
            num_rounds=2,
        )
        assert request.model == "roundtable:fast"
        assert request.stream is False
        assert request.council_models == ["model1", "model2"]
        assert request.num_rounds == 2


class TestStreamingResponse:
    """Test streaming response generation."""

    @pytest.mark.asyncio
    async def test_streaming_yields_chunks(self, mock_query_model, mock_settings):
        """Test that streaming yields properly formatted chunks."""
        mock_settings.council_models = ["mock:model1", "mock:model2"]
        mock_settings.chairman_model = "mock:chair"

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test question")],
            council_models=["mock:model1", "mock:model2"],
        )

        with patch("backend.openai_compat.run_roundtable") as mock_roundtable, \
             patch("backend.openai_compat.get_settings", return_value=mock_settings):

            # Mock roundtable to yield some events
            async def mock_events(*args, **kwargs):
                yield {"type": "roundtable_init", "total_rounds": 3, "council_members": []}
                yield {"type": "round_start", "round_number": 1, "round_name": "opening"}
                yield {"type": "chair_complete", "chair_final": {"content": "Final answer"}}

            mock_roundtable.return_value = mock_events()

            chunks = []
            async for chunk in generate_openai_stream(request):
                chunks.append(chunk)

            # Should have multiple chunks
            assert len(chunks) > 0

            # All chunks should start with "data: "
            for chunk in chunks:
                assert chunk.startswith("data: ")

            # Last chunk should be [DONE]
            assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_chunk_format(self, mock_query_model, mock_settings):
        """Test that streaming chunks have correct OpenAI format."""
        mock_settings.council_models = ["mock:model1", "mock:model2"]
        mock_settings.chairman_model = "mock:chair"

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test")],
            council_models=["mock:model1", "mock:model2"],
        )

        with patch("backend.openai_compat.run_roundtable") as mock_roundtable, \
             patch("backend.openai_compat.get_settings", return_value=mock_settings):

            async def mock_events(*args, **kwargs):
                yield {"type": "roundtable_init", "total_rounds": 1, "council_members": []}
                yield {"type": "chair_complete", "chair_final": {"content": "Done"}}

            mock_roundtable.return_value = mock_events()

            chunks = []
            async for chunk in generate_openai_stream(request):
                if chunk != "data: [DONE]\n\n" and not chunk.startswith("data: "):
                    continue
                if chunk == "data: [DONE]\n\n":
                    continue
                # Parse the JSON from the chunk
                json_str = chunk[6:].strip()  # Remove "data: " prefix
                if json_str:
                    parsed = json.loads(json_str)
                    chunks.append(parsed)

            # Verify chunk structure
            for chunk in chunks:
                assert "id" in chunk
                assert chunk["id"].startswith("chatcmpl-")
                assert "object" in chunk
                assert chunk["object"] == "chat.completion.chunk"
                assert "created" in chunk
                assert "model" in chunk
                assert "choices" in chunk
                assert len(chunk["choices"]) > 0


class TestNonStreamingResponse:
    """Test non-streaming response generation."""

    @pytest.mark.asyncio
    async def test_non_streaming_returns_response(self, mock_query_model, mock_settings):
        """Test that non-streaming returns complete response."""
        mock_settings.council_models = ["mock:model1", "mock:model2"]
        mock_settings.chairman_model = "mock:chair"

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test question")],
            stream=False,
            council_models=["mock:model1", "mock:model2"],
        )

        with patch("backend.openai_compat.run_roundtable") as mock_roundtable, \
             patch("backend.openai_compat.get_settings", return_value=mock_settings):

            async def mock_events(*args, **kwargs):
                yield {"type": "roundtable_init"}
                yield {"type": "chair_complete", "chair_final": {"content": "The final answer is 42."}}

            mock_roundtable.return_value = mock_events()

            response = await generate_non_streaming_response(request)

            assert isinstance(response, ChatCompletionResponse)
            assert response.id.startswith("chatcmpl-")
            assert response.object == "chat.completion"
            assert len(response.choices) == 1
            assert response.choices[0].message.role == "assistant"
            assert response.choices[0].message.content == "The final answer is 42."
            assert response.choices[0].finish_reason == "stop"


class TestErrorHandling:
    """Test error handling in OpenAI compat layer."""

    @pytest.mark.asyncio
    async def test_missing_user_message_error(self, mock_settings):
        """Test error when no user message provided."""
        mock_settings.council_models = ["mock:model1", "mock:model2"]

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="system", content="System only")],
        )

        with patch("backend.openai_compat.get_settings", return_value=mock_settings):
            chunks = []
            async for chunk in generate_openai_stream(request):
                chunks.append(chunk)

            # Should have error message
            found_error = False
            for chunk in chunks:
                if "Error: No user message" in chunk:
                    found_error = True
                    break
            assert found_error, "Should report missing user message error"

    @pytest.mark.asyncio
    async def test_missing_council_error(self, mock_settings):
        """Test error when council not configured."""
        mock_settings.council_models = []  # Empty council

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test")],
        )

        with patch("backend.openai_compat.get_settings", return_value=mock_settings):
            chunks = []
            async for chunk in generate_openai_stream(request):
                chunks.append(chunk)

            # Should have error about council
            found_error = False
            for chunk in chunks:
                if "at least 2 council models" in chunk.lower():
                    found_error = True
                    break
            assert found_error, "Should report missing council error"


class TestConfigurationInheritance:
    """Test that configuration is inherited correctly."""

    @pytest.mark.asyncio
    async def test_request_params_override_settings(self, mock_query_model, mock_settings):
        """Test that request parameters override settings."""
        mock_settings.council_models = ["settings:model1", "settings:model2"]
        mock_settings.roundtable_num_rounds = 3
        mock_settings.chairman_model = "settings:chair"

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test")],
            council_models=["request:model1", "request:model2"],
            num_rounds=1,
            chair_model="request:chair",
        )

        captured_kwargs = {}

        with patch("backend.openai_compat.run_roundtable") as mock_roundtable, \
             patch("backend.openai_compat.get_settings", return_value=mock_settings):

            async def capture_and_yield(*args, **kwargs):
                captured_kwargs.update(kwargs)
                yield {"type": "chair_complete", "chair_final": {"content": "Done"}}

            # Use side_effect to capture arguments when mock is called
            mock_roundtable.side_effect = capture_and_yield

            async for _ in generate_openai_stream(request):
                pass

            # Should use request params, not settings
            assert captured_kwargs.get("num_rounds") == 1
            assert captured_kwargs.get("chair_model") == "request:chair"

    @pytest.mark.asyncio
    async def test_model_string_override(self, mock_query_model, mock_settings):
        """Test that model string (roundtable:fast) overrides default rounds."""
        mock_settings.council_models = ["mock:model1", "mock:model2"]
        mock_settings.roundtable_num_rounds = 3
        mock_settings.chairman_model = "mock:chair"

        request = ChatCompletionRequest(
            model="roundtable:fast",
            messages=[ChatMessage(role="user", content="Test")],
        )

        captured_kwargs = {}

        with patch("backend.openai_compat.run_roundtable") as mock_roundtable, \
             patch("backend.openai_compat.get_settings", return_value=mock_settings):

            async def capture_and_yield(*args, **kwargs):
                captured_kwargs.update(kwargs)
                yield {"type": "chair_complete", "chair_final": {"content": "Done"}}

            # Use side_effect to capture arguments when mock is called
            mock_roundtable.side_effect = capture_and_yield

            async for _ in generate_openai_stream(request):
                pass

            # Model string should override to 1 round
            assert captured_kwargs.get("num_rounds") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
