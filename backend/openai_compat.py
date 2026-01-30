"""OpenAI-compatible API endpoint for Roundtable Mode.

This module provides an OpenAI-compatible `/v1/chat/completions` endpoint
that allows OpenWebUI and other OpenAI-compatible clients to use Roundtable.

The endpoint streams progress updates during deliberation and returns
the final chair synthesis as the assistant response.
"""

import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel

from .roundtable import get_default_council, run_roundtable
from .settings import get_settings

# ============================================================================
# OpenAI-Compatible Request/Response Models
# ============================================================================


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str  # "system", "user", "assistant"
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "roundtable"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = True

    # Roundtable-specific options (passed via extra_body or model string)
    # These override settings defaults
    council_models: list[str] | None = None
    num_rounds: int | None = None
    chair_model: str | None = None
    moderator_model: str | None = None
    max_parallel: int | None = None
    timeout_seconds: float | None = None


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible choice in completion response."""

    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    """OpenAI-compatible usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


# ============================================================================
# Request Processing
# ============================================================================


def extract_user_question(messages: list[ChatMessage]) -> str:
    """Extract the user's question from messages.

    Takes the last user message as the question for the roundtable.
    System messages and conversation history are preserved as context.
    """
    # Find the last user message
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


def extract_context(messages: list[ChatMessage]) -> str:
    """Extract system context and conversation history.

    Returns formatted context for the roundtable including:
    - System prompts
    - Previous conversation turns
    """
    context_parts = []

    for msg in messages:
        if msg.role == "system":
            context_parts.append(f"[System Instructions]\n{msg.content}")
        elif msg.role == "assistant":
            context_parts.append(f"[Previous Assistant Response]\n{msg.content}")
        # Skip user messages as the last one is the question

    # Exclude the last user message from context (it's the question)
    user_messages = [m for m in messages if m.role == "user"]
    if len(user_messages) > 1:
        for msg in user_messages[:-1]:
            context_parts.append(f"[Previous User Message]\n{msg.content}")

    return "\n\n---\n\n".join(context_parts) if context_parts else ""


def parse_model_config(model: str) -> dict[str, Any]:
    """Parse model string for embedded configuration.

    Supports formats like:
    - "roundtable" - Use defaults
    - "roundtable:3" - 3 rounds
    - "roundtable:fast" - 1 round, quick mode
    - "roundtable:thorough" - 3 rounds with larger council

    Returns dict of configuration overrides.
    """
    config = {}

    if ":" not in model:
        return config

    parts = model.split(":")
    modifier = parts[1].lower() if len(parts) > 1 else ""

    if modifier.isdigit():
        config["num_rounds"] = int(modifier)
    elif modifier == "fast":
        config["num_rounds"] = 1
    elif modifier == "thorough":
        config["num_rounds"] = 3
    elif modifier == "deep":
        config["num_rounds"] = 5

    return config


# ============================================================================
# Streaming Response Generator
# ============================================================================


async def generate_openai_stream(request: ChatCompletionRequest, http_request: Any = None) -> AsyncGenerator[str, None]:
    """Generate OpenAI-compatible streaming response from Roundtable.

    Streams progress updates as content deltas, then the final synthesis.

    Args:
        request: The chat completion request
        http_request: Optional FastAPI Request for disconnect detection

    Yields:
        SSE-formatted chunks in OpenAI streaming format
    """
    settings = get_settings()

    # Generate unique IDs
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Extract question and context from messages
    question = extract_user_question(request.messages)
    context = extract_context(request.messages)

    if not question:
        # No user message found - return error
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0, delta={"content": "Error: No user message found in request."}, finish_reason="stop"
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Parse model string for config overrides
    model_config = parse_model_config(request.model)

    # Build council configuration
    # Priority: request params > model string config > settings
    council_models = request.council_models or settings.council_models
    if not council_models or len(council_models) < 2:
        # Fallback to defaults if council not configured
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    delta={
                        "content": "Error: Roundtable requires at least 2 council models. Please configure models in settings."
                    },
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    agents = get_default_council(council_models)

    num_rounds = request.num_rounds or model_config.get("num_rounds") or settings.roundtable_num_rounds
    moderator_model = request.moderator_model or settings.chairman_model
    chair_model = request.chair_model or settings.chairman_model
    max_parallel = request.max_parallel or settings.roundtable_max_parallel

    # Generate a conversation ID for this request
    conversation_id = f"openai-compat-{uuid.uuid4().hex[:12]}"

    # Stream progress updates
    progress_buffer = []
    final_content = ""

    try:
        async for event in run_roundtable(
            conversation_id=conversation_id,
            question=question,
            agents=agents,
            moderator_model=moderator_model,
            chair_model=chair_model,
            context=context,
            num_rounds=num_rounds,
            max_parallel=max_parallel,
            request=http_request,
        ):
            event_type = event.get("type")

            # Stream progress as content deltas
            if event_type == "roundtable_init":
                total_rounds = event.get("total_rounds", num_rounds)
                council_size = len(event.get("council_members", []))
                progress_msg = f"[Roundtable Starting: {council_size} agents, {total_rounds} rounds]\n\n"

                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"role": "assistant", "content": progress_msg})],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "round_start":
                round_num = event.get("round_number", 0)
                round_name = event.get("round_name", "")
                progress_msg = f"**Round {round_num}: {round_name.title()}**\n"

                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg})],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "round_progress":
                response = event.get("response", {})
                agent_label = response.get("agent_label", "Agent")
                progress_msg = f"  - {agent_label} responded\n"

                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg})],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "round_complete":
                progress_msg = "\n"
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg})],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "moderator_start":
                progress_msg = "**Moderator Synthesis...**\n"
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg})],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "moderator_complete":
                progress_msg = "  - Complete\n\n"
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg})],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "chair_start":
                progress_msg = "**Chair Final Synthesis...**\n\n---\n\n"
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg})],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "chair_complete":
                # Stream the actual final content
                chair_final = event.get("chair_final", {})
                final_content = chair_final.get("content", "")

                if final_content:
                    # Stream in reasonable chunks for responsiveness
                    chunk_size = 50
                    for i in range(0, len(final_content), chunk_size):
                        text_chunk = final_content[i : i + chunk_size]
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[ChatCompletionChoice(index=0, delta={"content": text_chunk})],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

            elif event_type == "roundtable_aborted":
                progress_msg = "\n\n[Roundtable was aborted]"
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg}, finish_reason="stop")],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

        # Send finish chunk
        finish_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChoice(index=0, delta={}, finish_reason="stop")],
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_msg = f"\n\n[Error: {str(e)}]"
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChoice(index=0, delta={"content": error_msg}, finish_reason="stop")],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


async def generate_non_streaming_response(
    request: ChatCompletionRequest, http_request: Any = None
) -> ChatCompletionResponse:
    """Generate non-streaming response from Roundtable.

    Collects all events and returns a single response with the final synthesis.
    """
    settings = get_settings()

    # Generate unique IDs
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Extract question and context
    question = extract_user_question(request.messages)
    context = extract_context(request.messages)

    if not question:
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Error: No user message found in request."),
                    finish_reason="stop",
                )
            ],
        )

    # Build council configuration
    model_config = parse_model_config(request.model)
    council_models = request.council_models or settings.council_models

    if not council_models or len(council_models) < 2:
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Error: Roundtable requires at least 2 council models. Please configure models in settings.",
                    ),
                    finish_reason="stop",
                )
            ],
        )

    agents = get_default_council(council_models)
    num_rounds = request.num_rounds or model_config.get("num_rounds") or settings.roundtable_num_rounds
    moderator_model = request.moderator_model or settings.chairman_model
    chair_model = request.chair_model or settings.chairman_model
    max_parallel = request.max_parallel or settings.roundtable_max_parallel

    conversation_id = f"openai-compat-{uuid.uuid4().hex[:12]}"

    # Collect final content
    final_content = ""

    try:
        async for event in run_roundtable(
            conversation_id=conversation_id,
            question=question,
            agents=agents,
            moderator_model=moderator_model,
            chair_model=chair_model,
            context=context,
            num_rounds=num_rounds,
            max_parallel=max_parallel,
            request=http_request,
        ):
            if event.get("type") == "chair_complete":
                chair_final = event.get("chair_final", {})
                final_content = chair_final.get("content", "")

    except Exception as e:
        final_content = f"Error during roundtable: {str(e)}"

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final_content or "No response generated."),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=0,  # We don't track tokens yet
            completion_tokens=0,
            total_tokens=0,
        ),
    )


# ============================================================================
# Model List (for /v1/models endpoint)
# ============================================================================


def get_available_models() -> list[dict[str, Any]]:
    """Return list of available 'models' for OpenAI-compatible clients.

    Roundtable appears as a single model with variants.
    """
    return [
        {
            "id": "roundtable",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-council-plus",
            "permission": [],
            "root": "roundtable",
            "parent": None,
        },
        {
            "id": "roundtable:fast",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-council-plus",
            "permission": [],
            "root": "roundtable",
            "parent": "roundtable",
        },
        {
            "id": "roundtable:thorough",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-council-plus",
            "permission": [],
            "root": "roundtable",
            "parent": "roundtable",
        },
    ]
