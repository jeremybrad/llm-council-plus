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

from .council import preflight_roundtable_models
from .roundtable import get_default_council, run_roundtable
from .settings import get_settings


class RoundtableFailedError(Exception):
    """The council produced no answer; /v1 must not present this as a successful completion."""

    def __init__(
        self,
        message: str,
        outcome: dict[str, Any] | None = None,
        *,
        status_code: int = 502,
        error_type: str = "roundtable_failed",
    ):
        super().__init__(message)
        self.message = message
        self.outcome = outcome or {}
        self.status_code = status_code
        self.error_type = error_type


def _outcome(run: dict[str, Any] | None) -> dict[str, Any]:
    run = run or {}
    return {
        "status": run.get("status"),
        "failure_reason": run.get("failure_reason"),
        "failed_seats": run.get("failed_seats", []),
    }


NO_USER_MESSAGE = "no user message found in request"
TOO_FEW_SEATS = "roundtable requires at least 2 council models; configure models in settings"


def roundtable_route_problems(request: "ChatCompletionRequest", settings: Any = None) -> list[str]:
    """Fail-closed request and route preflight for /v1, using the same resolution as the generators.

    The route calls this before any response starts, so every refusal is an HTTP 400 rather
    than a 200 whose content says "Error" (WOR-1016).
    """
    settings = settings or get_settings()
    if not extract_user_question(request.messages):
        return [NO_USER_MESSAGE]
    council_models = request.council_models or settings.council_models
    if not council_models or len(council_models) < 2:
        return [TOO_FEW_SEATS]
    moderator_model = request.moderator_model or settings.chairman_model
    chair_model = request.chair_model or settings.chairman_model
    return preflight_roundtable_models(council_models, moderator_model, chair_model, settings)


def _refused(problems: list[str]) -> "RoundtableFailedError":
    return RoundtableFailedError(
        "Roundtable refused before any model call: " + "; ".join(problems),
        {"status": "refused", "failed_seats": []},
        status_code=400,
        error_type="roundtable_refused",
    )


def _chair_text(chair_final: dict[str, Any] | None) -> str:
    content = (chair_final or {}).get("content") or ""
    return content if content.strip() else ""


def _failure_text_safe(e: BaseException) -> str:
    return str(e).strip() or type(e).__name__


def _degraded_notice(outcome: dict[str, Any]) -> str:
    return f"[Degraded council: {outcome.get('failure_reason') or 'some council calls failed'}]\n\n"


def _failed_notice(outcome: dict[str, Any]) -> str:
    return (
        f"[Roundtable failed: {outcome.get('failure_reason') or 'no council answer'}. No council answer was produced.]"
    )


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

    num_rounds = request.num_rounds or model_config.get("num_rounds") or settings.roundtable_num_rounds
    moderator_model = request.moderator_model or settings.chairman_model
    chair_model = request.chair_model or settings.chairman_model
    max_parallel = request.max_parallel or settings.roundtable_max_parallel
    timeout_seconds = (
        request.timeout_seconds if request.timeout_seconds is not None else settings.roundtable_timeout_seconds
    )

    # The /v1 route refuses these with HTTP 400 before streaming starts; this in-stream
    # notice only serves direct callers of the generator, which cannot change the status.
    route_problems = preflight_roundtable_models(council_models, moderator_model, chair_model, settings)
    if route_problems:
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    delta={"content": "Error: roundtable refused before any model call: " + "; ".join(route_problems)},
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    agents = get_default_council(council_models)

    # Generate a conversation ID for this request
    conversation_id = f"openai-compat-{uuid.uuid4().hex[:12]}"

    # Stream progress updates
    final_content = ""
    answered = False

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
            timeout_seconds=timeout_seconds,
            request=http_request,
        ):
            event_type = event.get("type")

            # Stream progress as content deltas
            if event_type == "roundtable_budget_exceeded":
                progress_msg = (
                    f"ERROR: predicted {event.get('predicted_calls')} model calls exceed "
                    f"roundtable_max_calls_per_run={event.get('max_calls_per_run')}. "
                    "Quota units are unknown, not zero."
                )
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            delta={"content": progress_msg},
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

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
                if response.get("error"):
                    progress_msg = f"  - {agent_label} failed: {response.get('error')}\n"
                else:
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
                summary = event.get("moderator_summary") or {}
                if summary.get("error"):
                    progress_msg = f"  - Failed: {summary.get('error_message') or 'no summary'}\n\n"
                else:
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
                # Stream the actual final content, labelled with the run's real outcome
                outcome = _outcome(event.get("run"))
                final_content = _chair_text(event.get("chair_final"))
                if outcome["status"] == "failed" or not final_content:
                    final_content = _failed_notice(outcome)
                elif outcome["status"] == "degraded":
                    final_content = _degraded_notice(outcome) + final_content
                answered = True

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

            elif event_type == "roundtable_failed":
                progress_msg = "\n" + _failed_notice(_outcome(event.get("run")))
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, delta={"content": progress_msg}, finish_reason="stop")],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

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

        if not answered:
            # The engine ended without a chair answer or failure terminal: never a silent success.
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        delta={
                            "content": "\n"
                            + _failed_notice({"failure_reason": "the roundtable ended without an answer"})
                        },
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

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
        error_msg = f"\n\n[Error: {_failure_text_safe(e)}]"
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
        raise _refused([NO_USER_MESSAGE])

    # Build council configuration
    model_config = parse_model_config(request.model)
    council_models = request.council_models or settings.council_models

    if not council_models or len(council_models) < 2:
        raise _refused([TOO_FEW_SEATS])

    num_rounds = request.num_rounds or model_config.get("num_rounds") or settings.roundtable_num_rounds
    moderator_model = request.moderator_model or settings.chairman_model
    chair_model = request.chair_model or settings.chairman_model
    max_parallel = request.max_parallel or settings.roundtable_max_parallel
    timeout_seconds = (
        request.timeout_seconds if request.timeout_seconds is not None else settings.roundtable_timeout_seconds
    )

    route_problems = preflight_roundtable_models(council_models, moderator_model, chair_model, settings)
    if route_problems:
        raise _refused(route_problems)

    agents = get_default_council(council_models)

    conversation_id = f"openai-compat-{uuid.uuid4().hex[:12]}"

    # Collect final content
    final_content = ""
    failed_outcome: dict[str, Any] | None = None
    raise_after: RoundtableFailedError | None = None

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
            timeout_seconds=timeout_seconds,
            request=http_request,
        ):
            if event.get("type") == "roundtable_budget_exceeded":
                budget_message = (
                    f"Error: predicted {event.get('predicted_calls')} model calls exceed "
                    f"roundtable_max_calls_per_run={event.get('max_calls_per_run')}. "
                    "Quota units are unknown, not zero."
                )
                budget_outcome = _outcome(event.get("run"))
                budget_outcome["status"] = "budget_exceeded"
                budget_outcome["failure_reason"] = budget_outcome["failure_reason"] or budget_message
                raise_after = RoundtableFailedError(
                    budget_message,
                    budget_outcome,
                    status_code=429,
                    error_type="roundtable_budget_exceeded",
                )
            elif event.get("type") == "roundtable_aborted":
                raise_after = RoundtableFailedError(
                    "[Roundtable was aborted]", _outcome(event.get("run")), error_type="roundtable_aborted"
                )
            elif event.get("type") == "chair_complete":
                outcome = _outcome(event.get("run"))
                final_content = _chair_text(event.get("chair_final"))
                if outcome["status"] == "failed" or not final_content:
                    failed_outcome = outcome
                elif outcome["status"] == "degraded":
                    final_content = _degraded_notice(outcome) + final_content
            elif event.get("type") == "roundtable_failed":
                failed_outcome = _outcome(event.get("run"))

    except Exception as e:
        raise_after = RoundtableFailedError(
            f"Error during roundtable: {_failure_text_safe(e)}",
            {"status": "error", "failed_seats": []},
            status_code=500,
            error_type="roundtable_error",
        )

    if raise_after is not None:
        # Refused, budget-rejected, aborted and errored runs produced no council answer.
        raise raise_after
    if failed_outcome is not None:
        # A completely failed council is an error, not a successful completion.
        raise RoundtableFailedError(_failed_notice(failed_outcome), failed_outcome)
    if not final_content:
        # The engine ended without a chair answer or a failure terminal.
        incomplete = {"status": "incomplete", "failure_reason": "the roundtable ended without an answer"}
        raise RoundtableFailedError(
            _failed_notice(incomplete), {**incomplete, "failed_seats": []}, error_type="roundtable_incomplete"
        )

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final_content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            # OpenAI schema requires integers. These zeros are compatibility
            # placeholders, not measured subscription quota (unknown, not zero).
            prompt_tokens=0,
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
