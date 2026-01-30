"""FastAPI backend for LLM Council."""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import storage
from .api import claims_router
from .council import (
    PROVIDERS,
    calculate_aggregate_rankings,
    generate_conversation_title,
    generate_search_query,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
)
from .modes import ModeRunner
from .modes.socrates_runner import SocratesRunner
from .openai_compat import (
    ChatCompletionRequest,
    generate_non_streaming_response,
    generate_openai_stream,
)
from .openai_compat import get_available_models as get_roundtable_models
from .roundtable import get_default_council, run_roundtable
from .search import SearchProvider, perform_web_search
from .settings import (
    AVAILABLE_MODELS,
    DEFAULT_CHAIRMAN_MODEL,
    DEFAULT_COUNCIL_MODELS,
    get_settings,
    update_settings,
)

__version__ = "0.2.0"

app = FastAPI(
    title="LLM Council Plus API",
    description="Multi-LLM roundtable orchestration with role-based agents, claims validation, and evidence-backed truth verification.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health check and version info"},
        {"name": "claims", "description": "Claims API v1 for evidence-backed truth verification"},
        {"name": "claims-legacy", "description": "Deprecated: Use /api/v1/claims instead"},
        {"name": "council", "description": "Council deliberation endpoints"},
        {"name": "conversations", "description": "Conversation management"},
        {"name": "settings", "description": "Application settings"},
    ],
)

# Initialize mode runner with persist directory
MODE_SESSIONS_DIR = Path(__file__).parent.parent / "data" / "mode_sessions"
_mode_runner: ModeRunner | None = None


def get_runner() -> ModeRunner:
    """Get or create the ModeRunner instance."""
    global _mode_runner
    if _mode_runner is None:
        _mode_runner = ModeRunner()
        _mode_runner.session_store.persist_dir = MODE_SESSIONS_DIR
    return _mode_runner


# Enable CORS for local development and network access
# Allow requests from any hostname on ports 517x (Vite), 3000 (CRA), 8080 (common dev)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://.*:(517[0-9]|3000|8080)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
# Primary versioned API (v1)
app.include_router(claims_router, prefix="/api/v1/claims")
# Legacy unversioned API (deprecated, for backward compatibility)
app.include_router(claims_router, prefix="/api/claims", deprecated=True)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""

    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""

    content: str
    web_search: bool = False
    execution_mode: str = "full"  # 'chat_only', 'chat_ranking', 'full'


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""

    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""

    id: str
    created_at: str
    title: str
    messages: list[dict[str, Any]]


@app.get("/", tags=["health"])
async def root():
    """Root endpoint with basic status."""
    return {"status": "ok", "service": "LLM Council Plus"}


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint with version and status info for monitoring."""
    return {
        "status": "healthy",
        "service": "LLM Council Plus API",
        "version": __version__,
        "api_versions": {
            "current": "v1",
            "supported": ["v1"],
            "deprecated": ["v0 (/api/claims)"],
        },
    }


@app.get("/api/conversations", response_model=list[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    deleted = storage.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, body: SendMessageRequest, request: Request):
    """Send a message and stream the 3-stage council process."""
    # Validate execution_mode
    valid_modes = ["chat_only", "chat_ranking", "full", "roundtable"]
    if body.execution_mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid execution_mode. Must be one of: {valid_modes}")

    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Initialize variables for metadata
            stage1_results = []
            stage2_results = []
            stage3_result = None
            label_to_model = {}
            aggregate_rankings = {}

            # Add user message
            storage.add_user_message(conversation_id, body.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(body.content))

            # Perform web search if requested
            search_context = ""
            search_query = ""
            if body.web_search:
                # Check for disconnect before starting search
                if await request.is_disconnected():
                    print("Client disconnected before web search")
                    raise asyncio.CancelledError("Client disconnected")

                settings = get_settings()
                provider = SearchProvider(settings.search_provider)

                # Set API keys if configured
                if settings.tavily_api_key and provider == SearchProvider.TAVILY:
                    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
                if settings.brave_api_key and provider == SearchProvider.BRAVE:
                    os.environ["BRAVE_API_KEY"] = settings.brave_api_key

                yield f"data: {json.dumps({'type': 'search_start', 'data': {'provider': provider.value}})}\n\n"

                # Check for disconnect before generating search query
                if await request.is_disconnected():
                    print("Client disconnected during search setup")
                    raise asyncio.CancelledError("Client disconnected")

                # Generate search query (passthrough - no AI model needed)
                search_query = generate_search_query(body.content)

                # Check for disconnect before performing search
                if await request.is_disconnected():
                    print("Client disconnected before search execution")
                    raise asyncio.CancelledError("Client disconnected")

                # Run search (now fully async for Tavily/Brave, threaded only for DuckDuckGo)
                search_result = await perform_web_search(
                    search_query, 5, provider, settings.full_content_results, settings.search_keyword_extraction
                )
                search_context = search_result["results"]
                extracted_query = search_result["extracted_query"]
                yield f"data: {json.dumps({'type': 'search_complete', 'data': {'search_query': search_query, 'extracted_query': extracted_query, 'search_context': search_context, 'provider': provider.value}})}\n\n"
                await asyncio.sleep(0.05)

            # ========================================
            # ROUNDTABLE MODE: Multi-round deliberation
            # ========================================
            if body.execution_mode == "roundtable":
                settings = get_settings()

                # Build council from configured models
                council_models = settings.council_models
                if not council_models or len(council_models) < 2:
                    error_msg = "Roundtable requires at least 2 council models. Please configure models in settings."
                    storage.add_error_message(conversation_id, error_msg)
                    yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                    return

                agents = get_default_council(council_models)
                moderator_model = settings.chairman_model  # Reuse chairman as moderator
                chair_model = settings.chairman_model
                num_rounds = settings.roundtable_num_rounds
                max_parallel = settings.roundtable_max_parallel

                yield f"data: {json.dumps({'type': 'roundtable_start', 'data': {'num_rounds': num_rounds, 'council_size': len(agents)}})}\n\n"
                await asyncio.sleep(0.05)

                # Stream roundtable execution
                run_data = None
                aborted = False
                try:
                    async for event in run_roundtable(
                        conversation_id=conversation_id,
                        question=body.content,
                        agents=agents,
                        moderator_model=moderator_model,
                        chair_model=chair_model,
                        num_rounds=num_rounds,
                        context=search_context,
                        max_parallel=max_parallel,
                        request=request,
                    ):
                        event_type = event.get("type")

                        # Transform backend events to frontend format
                        if event_type == "roundtable_init":
                            # Already sent roundtable_start above, skip this
                            continue

                        elif event_type == "round_start":
                            yield f"data: {json.dumps({'type': 'round_start', 'data': {'round_number': event.get('round_number'), 'round_name': event.get('round_name')}})}\n\n"

                        elif event_type == "round_progress":
                            # Transform to agent_response format for frontend
                            yield f"data: {json.dumps({'type': 'agent_response', 'data': {'round_number': event.get('round_number'), 'response': event.get('response')}})}\n\n"

                        elif event_type == "round_complete":
                            yield f"data: {json.dumps({'type': 'round_complete', 'data': {'round_number': event.get('round_number')}})}\n\n"

                        elif event_type == "moderator_start":
                            # Can notify frontend that moderator is running
                            yield f"data: {json.dumps({'type': 'moderator_start'})}\n\n"

                        elif event_type == "moderator_complete":
                            yield f"data: {json.dumps({'type': 'moderator_complete', 'data': event.get('moderator_summary')})}\n\n"

                        elif event_type == "chair_start":
                            yield f"data: {json.dumps({'type': 'chair_start'})}\n\n"

                        elif event_type == "chair_complete":
                            # Store run data for saving
                            run_data = event.get("run")
                            yield f"data: {json.dumps({'type': 'chair_complete', 'data': event.get('chair_final')})}\n\n"
                            # Send roundtable_complete
                            yield f"data: {json.dumps({'type': 'roundtable_complete'})}\n\n"

                        elif event_type == "roundtable_aborted":
                            run_data = event.get("run")
                            aborted = True
                            yield f"data: {json.dumps({'type': 'roundtable_error', 'message': 'Roundtable was aborted'})}\n\n"

                        else:
                            # Forward unknown events as-is
                            yield f"data: {json.dumps(event)}\n\n"

                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    # Abort handled - run_data already captured from roundtable_aborted event
                    aborted = True

                # Handle title generation for first message
                if title_task:
                    try:
                        title = await title_task
                        storage.update_conversation_title(conversation_id, title)
                        yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"
                    except Exception as e:
                        print(f"Error waiting for title task: {e}")

                # Save roundtable run data and message to conversation
                # Note: run_data is captured on both completion (chair_complete) and abort (roundtable_aborted)
                if run_data:
                    if aborted:
                        print(
                            f"Saving aborted roundtable run: {run_data.get('run_id')} (status: {run_data.get('status')})"
                        )
                    # Persist full run data to data/runs/{conversation_id}/{run_id}.json
                    storage.save_run(run_data)

                    # Add slim reference to conversation
                    storage.add_roundtable_message(
                        conversation_id=conversation_id,
                        run_id=run_data["run_id"],
                        chair_final=run_data.get("chair_final", {}),
                        metadata={
                            "execution_mode": "roundtable",
                            "num_rounds": num_rounds,
                            "council_members": [a.model for a in agents],
                            "moderator_model": moderator_model,
                            "chair_model": chair_model,
                            "search_context": search_context if search_context else None,
                            "search_query": search_query if search_query else None,
                        },
                    )

                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                return  # Exit early - roundtable flow is complete

            # ========================================
            # STANDARD MODE: 3-stage deliberation
            # ========================================

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            await asyncio.sleep(0.05)

            total_models = 0

            async for item in stage1_collect_responses(body.content, search_context, request):
                if isinstance(item, int):
                    total_models = item
                    print(f"DEBUG: Sending stage1_init with total={total_models}")
                    yield f"data: {json.dumps({'type': 'stage1_init', 'total': total_models})}\n\n"
                    continue

                stage1_results.append(item)
                yield f"data: {json.dumps({'type': 'stage1_progress', 'data': item, 'count': len(stage1_results), 'total': total_models})}\n\n"
                await asyncio.sleep(0.01)

            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"
            await asyncio.sleep(0.05)

            # Check if any models responded successfully in Stage 1
            if not any(r for r in stage1_results if not r.get("error")):
                error_msg = "All models failed to respond in Stage 1, likely due to rate limits or API errors. Please try again or adjust your model selection."
                storage.add_error_message(conversation_id, error_msg)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return  # Stop further processing

            # Stage 2: Only if mode is 'chat_ranking' or 'full'
            if body.execution_mode in ["chat_ranking", "full"]:
                yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
                await asyncio.sleep(0.05)

                # Iterate over the async generator
                async for item in stage2_collect_rankings(body.content, stage1_results, search_context, request):
                    # First item is the label mapping
                    if isinstance(item, dict) and not item.get("model"):
                        label_to_model = item
                        # Send init event with total count
                        yield f"data: {json.dumps({'type': 'stage2_init', 'total': len(label_to_model)})}\n\n"
                        continue

                    # Subsequent items are results
                    stage2_results.append(item)

                    # Send progress update
                    print(f"Stage 2 Progress: {len(stage2_results)}/{len(label_to_model)} - {item['model']}")
                    yield f"data: {json.dumps({'type': 'stage2_progress', 'data': item, 'count': len(stage2_results), 'total': len(label_to_model)})}\n\n"
                    await asyncio.sleep(0.01)

                aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
                yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings, 'search_query': search_query, 'search_context': search_context}})}\n\n"
                await asyncio.sleep(0.05)

            # Stage 3: Only if mode is 'full'
            if body.execution_mode == "full":
                yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
                await asyncio.sleep(0.05)

                # Check for disconnect before starting Stage 3
                if await request.is_disconnected():
                    print("Client disconnected before Stage 3")
                    raise asyncio.CancelledError("Client disconnected")

                stage3_result = await stage3_synthesize_final(
                    body.content, stage1_results, stage2_results, search_context
                )
                yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                try:
                    title = await title_task
                    storage.update_conversation_title(conversation_id, title)
                    yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"
                except Exception as e:
                    print(f"Error waiting for title task: {e}")

            # Save complete assistant message with metadata
            metadata = {
                "execution_mode": body.execution_mode,  # Save mode for historical context
            }

            # Only include stage2/stage3 metadata if they were executed
            if body.execution_mode in ["chat_ranking", "full"]:
                metadata["label_to_model"] = label_to_model
                metadata["aggregate_rankings"] = aggregate_rankings

            if search_context:
                metadata["search_context"] = search_context
            if search_query:
                metadata["search_query"] = search_query

            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results if body.execution_mode in ["chat_ranking", "full"] else None,
                stage3_result if body.execution_mode == "full" else None,
                metadata,
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except asyncio.CancelledError:
            print(f"Stream cancelled for conversation {conversation_id}")
            # Even if cancelled, try to save the title if it's ready or nearly ready
            if title_task:
                try:
                    # Give it a small grace period to finish if it's close
                    title = await asyncio.wait_for(title_task, timeout=2.0)
                    storage.update_conversation_title(conversation_id, title)
                    print(f"Saved title despite cancellation: {title}")
                except Exception as e:
                    print(f"Could not save title during cancellation: {e}")
            raise
        except Exception as e:
            print(f"Stream error: {e}")
            # Save error to conversation history
            storage.add_error_message(conversation_id, f"Error: {str(e)}")
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


class UpdateSettingsRequest(BaseModel):
    """Request to update settings."""

    search_provider: str | None = None
    search_keyword_extraction: str | None = None
    ollama_base_url: str | None = None
    full_content_results: int | None = None

    # Custom OpenAI-compatible endpoint
    custom_endpoint_name: str | None = None
    custom_endpoint_url: str | None = None
    custom_endpoint_api_key: str | None = None

    # API Keys
    tavily_api_key: str | None = None
    brave_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    mistral_api_key: str | None = None
    deepseek_api_key: str | None = None
    groq_api_key: str | None = None

    # Enabled Providers
    enabled_providers: dict[str, bool] | None = None
    direct_provider_toggles: dict[str, bool] | None = None

    # Council Configuration (unified)
    council_models: list[str] | None = None
    chairman_model: str | None = None

    # Remote/Local filters
    council_member_filters: dict[int, str] | None = None
    chairman_filter: str | None = None
    search_query_filter: str | None = None

    # Temperature Settings
    council_temperature: float | None = None
    chairman_temperature: float | None = None
    stage2_temperature: float | None = None

    # Execution Mode
    execution_mode: str | None = None

    # System Prompts
    stage1_prompt: str | None = None
    stage2_prompt: str | None = None
    stage3_prompt: str | None = None


class TestTavilyRequest(BaseModel):
    """Request to test Tavily API key."""

    api_key: str | None = None


@app.get("/api/settings")
async def get_app_settings():
    """Get current application settings."""
    settings = get_settings()
    return {
        "search_provider": settings.search_provider,
        "search_keyword_extraction": settings.search_keyword_extraction,
        "ollama_base_url": settings.ollama_base_url,
        "full_content_results": settings.full_content_results,
        # Custom Endpoint
        "custom_endpoint_name": settings.custom_endpoint_name,
        "custom_endpoint_url": settings.custom_endpoint_url,
        # Don't send the API key to frontend for security
        # API Key Status
        "tavily_api_key_set": bool(settings.tavily_api_key),
        "brave_api_key_set": bool(settings.brave_api_key),
        "openrouter_api_key_set": bool(settings.openrouter_api_key),
        "openai_api_key_set": bool(settings.openai_api_key),
        "anthropic_api_key_set": bool(settings.anthropic_api_key),
        "google_api_key_set": bool(settings.google_api_key),
        "mistral_api_key_set": bool(settings.mistral_api_key),
        "deepseek_api_key_set": bool(settings.deepseek_api_key),
        "groq_api_key_set": bool(settings.groq_api_key),
        "custom_endpoint_api_key_set": bool(settings.custom_endpoint_api_key),
        # Enabled Providers
        "enabled_providers": settings.enabled_providers,
        "direct_provider_toggles": settings.direct_provider_toggles,
        # Council Configuration (unified)
        "council_models": settings.council_models,
        "chairman_model": settings.chairman_model,
        # Remote/Local filters
        "council_member_filters": settings.council_member_filters,
        "council_member_filters": settings.council_member_filters,
        "chairman_filter": settings.chairman_filter,
        "search_query_filter": settings.search_query_filter,
        # Temperature Settings
        "council_temperature": settings.council_temperature,
        "chairman_temperature": settings.chairman_temperature,
        "stage2_temperature": settings.stage2_temperature,
        # Prompts
        "stage1_prompt": settings.stage1_prompt,
        "stage2_prompt": settings.stage2_prompt,
        "stage3_prompt": settings.stage3_prompt,
    }


@app.get("/api/settings/defaults")
async def get_default_settings():
    """Get default model settings."""
    from .prompts import STAGE1_PROMPT_DEFAULT, STAGE2_PROMPT_DEFAULT, STAGE3_PROMPT_DEFAULT
    from .settings import DEFAULT_ENABLED_PROVIDERS

    return {
        "council_models": DEFAULT_COUNCIL_MODELS,
        "chairman_model": DEFAULT_CHAIRMAN_MODEL,
        "enabled_providers": DEFAULT_ENABLED_PROVIDERS,
        "stage1_prompt": STAGE1_PROMPT_DEFAULT,
        "stage2_prompt": STAGE2_PROMPT_DEFAULT,
        "stage3_prompt": STAGE3_PROMPT_DEFAULT,
    }


@app.put("/api/settings")
async def update_app_settings(request: UpdateSettingsRequest):
    """Update application settings."""
    updates = {}

    if request.search_provider is not None:
        # Validate provider
        try:
            provider = SearchProvider(request.search_provider)
            updates["search_provider"] = provider
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid search provider. Must be one of: {[p.value for p in SearchProvider]}"
            )

    if request.search_keyword_extraction is not None:
        if request.search_keyword_extraction not in ["direct", "yake"]:
            raise HTTPException(status_code=400, detail="Invalid keyword extraction mode. Must be 'direct' or 'yake'")
        updates["search_keyword_extraction"] = request.search_keyword_extraction

    if request.ollama_base_url is not None:
        updates["ollama_base_url"] = request.ollama_base_url

    # Custom endpoint
    if request.custom_endpoint_name is not None:
        updates["custom_endpoint_name"] = request.custom_endpoint_name
    if request.custom_endpoint_url is not None:
        updates["custom_endpoint_url"] = request.custom_endpoint_url
    if request.custom_endpoint_api_key is not None:
        updates["custom_endpoint_api_key"] = request.custom_endpoint_api_key

    if request.full_content_results is not None:
        # Validate range
        if request.full_content_results < 0 or request.full_content_results > 10:
            raise HTTPException(status_code=400, detail="full_content_results must be between 0 and 10")
        updates["full_content_results"] = request.full_content_results

    # Prompt updates
    if request.stage1_prompt is not None:
        updates["stage1_prompt"] = request.stage1_prompt
    if request.stage2_prompt is not None:
        updates["stage2_prompt"] = request.stage2_prompt
    if request.stage3_prompt is not None:
        updates["stage3_prompt"] = request.stage3_prompt

    if request.tavily_api_key is not None:
        updates["tavily_api_key"] = request.tavily_api_key
        # Also set in environment for immediate use
        if request.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = request.tavily_api_key

    if request.brave_api_key is not None:
        updates["brave_api_key"] = request.brave_api_key
        # Also set in environment for immediate use
        if request.brave_api_key:
            os.environ["BRAVE_API_KEY"] = request.brave_api_key

    if request.openrouter_api_key is not None:
        updates["openrouter_api_key"] = request.openrouter_api_key

    # Direct Provider Keys
    if request.openai_api_key is not None:
        updates["openai_api_key"] = request.openai_api_key
    if request.anthropic_api_key is not None:
        updates["anthropic_api_key"] = request.anthropic_api_key
    if request.google_api_key is not None:
        updates["google_api_key"] = request.google_api_key
    if request.mistral_api_key is not None:
        updates["mistral_api_key"] = request.mistral_api_key
    if request.deepseek_api_key is not None:
        updates["deepseek_api_key"] = request.deepseek_api_key
    if request.groq_api_key is not None:
        updates["groq_api_key"] = request.groq_api_key

    # Enabled Providers
    if request.enabled_providers is not None:
        updates["enabled_providers"] = request.enabled_providers

    if request.direct_provider_toggles is not None:
        updates["direct_provider_toggles"] = request.direct_provider_toggles

    # Council Configuration (unified)
    if request.council_models is not None:
        # Validate that at least two models are selected
        if len(request.council_models) < 2:
            raise HTTPException(status_code=400, detail="At least two council models must be selected")
        if len(request.council_models) > 8:
            raise HTTPException(status_code=400, detail="Maximum of 8 council models allowed")
        updates["council_models"] = request.council_models

    if request.chairman_model is not None:
        updates["chairman_model"] = request.chairman_model

    # Remote/Local filters
    if request.council_member_filters is not None:
        updates["council_member_filters"] = request.council_member_filters
    if request.chairman_filter is not None:
        updates["chairman_filter"] = request.chairman_filter
    if request.search_query_filter is not None:
        updates["search_query_filter"] = request.search_query_filter

    # Temperature Settings
    if request.council_temperature is not None:
        updates["council_temperature"] = request.council_temperature
    if request.chairman_temperature is not None:
        updates["chairman_temperature"] = request.chairman_temperature
    if request.stage2_temperature is not None:
        updates["stage2_temperature"] = request.stage2_temperature

    # Prompts   # Execution Mode
    if request.execution_mode is not None:
        valid_modes = ["chat_only", "chat_ranking", "full"]
        if request.execution_mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid execution_mode. Must be one of: {valid_modes}")
        updates["execution_mode"] = request.execution_mode

    if updates:
        settings = update_settings(**updates)
    else:
        settings = get_settings()

    return {
        "search_provider": settings.search_provider,
        "search_keyword_extraction": settings.search_keyword_extraction,
        "ollama_base_url": settings.ollama_base_url,
        "full_content_results": settings.full_content_results,
        # Custom Endpoint
        "custom_endpoint_name": settings.custom_endpoint_name,
        "custom_endpoint_url": settings.custom_endpoint_url,
        # API Key Status
        "tavily_api_key_set": bool(settings.tavily_api_key),
        "brave_api_key_set": bool(settings.brave_api_key),
        "openrouter_api_key_set": bool(settings.openrouter_api_key),
        "openai_api_key_set": bool(settings.openai_api_key),
        "anthropic_api_key_set": bool(settings.anthropic_api_key),
        "google_api_key_set": bool(settings.google_api_key),
        "mistral_api_key_set": bool(settings.mistral_api_key),
        "deepseek_api_key_set": bool(settings.deepseek_api_key),
        "groq_api_key_set": bool(settings.groq_api_key),
        "custom_endpoint_api_key_set": bool(settings.custom_endpoint_api_key),
        # Enabled Providers
        "enabled_providers": settings.enabled_providers,
        "direct_provider_toggles": settings.direct_provider_toggles,
        # Council Configuration (unified)
        "council_models": settings.council_models,
        "chairman_model": settings.chairman_model,
        # Remote/Local filters
        "council_member_filters": settings.council_member_filters,
        "chairman_filter": settings.chairman_filter,
        # Prompts
        "stage1_prompt": settings.stage1_prompt,
        "stage2_prompt": settings.stage2_prompt,
        "stage3_prompt": settings.stage3_prompt,
    }


@app.get("/api/models")
async def get_models():
    """Get available models for council selection."""
    from .openrouter import fetch_models

    # Try dynamic fetch first
    dynamic_models = await fetch_models()
    if dynamic_models:
        return {"models": dynamic_models}

    # Fallback to static list
    return {"models": AVAILABLE_MODELS}


@app.get("/api/models/direct")
async def get_direct_models():
    """Get available models from all configured direct providers."""
    all_models = []

    # Iterate over all providers
    for provider_id, provider in PROVIDERS.items():
        # Skip OpenRouter and Ollama as they are handled separately
        if provider_id in ["openrouter", "ollama", "hybrid"]:
            continue

        try:
            # Fetch models from provider
            models = await provider.get_models()
            all_models.extend(models)
        except Exception as e:
            print(f"Error fetching models for {provider_id}: {e}")

    return all_models


@app.post("/api/settings/test-tavily")
async def test_tavily_api(request: TestTavilyRequest):
    """Test Tavily API key with a simple search."""
    import httpx

    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": request.api_key or settings.tavily_api_key,
                    "query": "test",
                    "max_results": 1,
                    "search_depth": "basic",
                },
            )

            if response.status_code == 200:
                return {"success": True, "message": "API key is valid"}
            elif response.status_code == 401:
                return {"success": False, "message": "Invalid API key"}
            else:
                return {"success": False, "message": f"API error: {response.status_code}"}

    except httpx.TimeoutException:
        return {"success": False, "message": "Request timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


class TestBraveRequest(BaseModel):
    """Request to test Brave API key."""

    api_key: str | None = None


@app.post("/api/settings/test-brave")
async def test_brave_api(request: TestBraveRequest):
    """Test Brave API key with a simple search."""
    import httpx

    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": "test", "count": 1},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": request.api_key or settings.brave_api_key,
                },
            )

            if response.status_code == 200:
                return {"success": True, "message": "API key is valid"}
            elif response.status_code == 401 or response.status_code == 403:
                return {"success": False, "message": "Invalid API key"}
            else:
                return {"success": False, "message": f"API error: {response.status_code}"}

    except httpx.TimeoutException:
        return {"success": False, "message": "Request timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


class TestOpenRouterRequest(BaseModel):
    """Request to test OpenRouter API key."""

    api_key: str | None = None


class TestProviderRequest(BaseModel):
    """Request to test a specific provider's API key."""

    provider_id: str
    api_key: str


@app.post("/api/settings/test-provider")
async def test_provider_api(request: TestProviderRequest):
    """Test an API key for a specific provider."""
    from .council import PROVIDERS
    from .settings import get_settings

    if request.provider_id not in PROVIDERS:
        raise HTTPException(status_code=400, detail="Invalid provider ID")

    api_key = request.api_key
    if not api_key:
        # Try to get from settings
        settings = get_settings()
        # Map provider_id to setting key (e.g. 'openai' -> 'openai_api_key')
        setting_key = f"{request.provider_id}_api_key"
        if hasattr(settings, setting_key):
            api_key = getattr(settings, setting_key)

    if not api_key:
        return {"success": False, "message": "No API key provided or configured"}

    provider = PROVIDERS[request.provider_id]
    return await provider.validate_key(api_key)


class TestOllamaRequest(BaseModel):
    """Request to test Ollama connection."""

    base_url: str


@app.get("/api/ollama/tags")
async def get_ollama_tags(base_url: str | None = None):
    """Fetch available models from Ollama."""
    import httpx

    from .config import get_ollama_base_url

    if not base_url:
        base_url = get_ollama_base_url()

    if base_url.endswith("/"):
        base_url = base_url[:-1]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/api/tags")

            if response.status_code != 200:
                return {"models": [], "error": f"Ollama API error: {response.status_code}"}

            data = response.json()
            models = []
            for model in data.get("models", []):
                models.append(
                    {
                        "id": model.get("name"),
                        "name": model.get("name"),
                        # Ollama doesn't return context length in tags
                        "context_length": None,
                        "is_free": True,
                        "modified_at": model.get("modified_at"),
                    }
                )

            # Sort by modified_at (newest first), fallback to name
            models.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
            return {"models": models}

    except httpx.ConnectError:
        return {"models": [], "error": "Could not connect to Ollama. Is it running?"}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.post("/api/settings/test-ollama")
async def test_ollama_connection(request: TestOllamaRequest):
    """Test connection to Ollama instance."""
    import httpx

    base_url = request.base_url
    if base_url.endswith("/"):
        base_url = base_url[:-1]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/api/tags")

            if response.status_code == 200:
                return {"success": True, "message": "Successfully connected to Ollama"}
            else:
                return {"success": False, "message": f"Ollama API error: {response.status_code}"}

    except httpx.ConnectError:
        return {"success": False, "message": "Could not connect to Ollama. Is it running at this URL?"}
    except Exception as e:
        return {"success": False, "message": str(e)}


class TestCustomEndpointRequest(BaseModel):
    """Request to test custom OpenAI-compatible endpoint."""

    name: str
    url: str
    api_key: str | None = None


@app.post("/api/settings/test-custom-endpoint")
async def test_custom_endpoint(request: TestCustomEndpointRequest):
    """Test connection to a custom OpenAI-compatible endpoint."""
    from .providers.custom_openai import CustomOpenAIProvider

    provider = CustomOpenAIProvider()
    return await provider.validate_connection(request.url, request.api_key or "")


@app.get("/api/custom-endpoint/models")
async def get_custom_endpoint_models():
    """Fetch available models from the custom endpoint."""
    from .providers.custom_openai import CustomOpenAIProvider
    from .settings import get_settings

    settings = get_settings()
    if not settings.custom_endpoint_url:
        return {"models": [], "error": "No custom endpoint configured"}

    provider = CustomOpenAIProvider()
    models = await provider.get_models()
    return {"models": models}


@app.get("/api/models")
async def get_openrouter_models():
    """Fetch available models from OpenRouter API."""
    import httpx

    from .config import get_openrouter_api_key

    api_key = get_openrouter_api_key()
    if not api_key:
        return {"models": [], "error": "No OpenRouter API key configured"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )

            if response.status_code != 200:
                return {"models": [], "error": f"API error: {response.status_code}"}

            data = response.json()
            models = []

            # Comprehensive exclusion list for non-text/chat models
            excluded_terms = [
                "embed",
                "audio",
                "whisper",
                "tts",
                "dall-e",
                "realtime",
                "vision-only",
                "voxtral",
                "speech",
                "transcribe",
                "sora",
            ]

            for model in data.get("data", []):
                mid = model.get("id", "").lower()
                name_lower = model.get("name", "").lower()

                if any(term in mid for term in excluded_terms) or any(term in name_lower for term in excluded_terms):
                    continue

                # Extract pricing - free models have 0 cost
                pricing = model.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "0") or "0")
                completion_price = float(pricing.get("completion", "0") or "0")
                is_free = prompt_price == 0 and completion_price == 0

                models.append(
                    {
                        "id": f"openrouter:{model.get('id')}",
                        "name": f"{model.get('name', model.get('id'))} [OpenRouter]",
                        "provider": "OpenRouter",
                        "context_length": model.get("context_length"),
                        "is_free": is_free,
                    }
                )

            # Sort by name
            models.sort(key=lambda x: x["name"].lower())
            return {"models": models}

    except httpx.TimeoutException:
        return {"models": [], "error": "Request timed out"}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.post("/api/settings/test-openrouter")
async def test_openrouter_api(request: TestOpenRouterRequest):
    """Test OpenRouter API key with a simple request."""
    import httpx

    from .config import get_openrouter_api_key

    # Use provided key or fall back to saved key
    api_key = request.api_key if request.api_key else get_openrouter_api_key()

    if not api_key:
        return {"success": False, "message": "No API key provided or configured"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )

            if response.status_code == 200:
                return {"success": True, "message": "API key is valid"}
            elif response.status_code == 401:
                return {"success": False, "message": "Invalid API key"}
            else:
                return {"success": False, "message": f"API error: {response.status_code}"}

    except httpx.TimeoutException:
        return {"success": False, "message": "Request timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ========================================
# MODE ENGINE API ENDPOINTS
# ========================================


class CreateModeSessionRequest(BaseModel):
    """Request to create a new mode session."""

    mode_id: str
    initial_inquiry: str | None = None
    model: str | None = None
    max_turns: int | None = None


class ModeTurnRequest(BaseModel):
    """Request to run a mode turn."""

    session_id: str
    user_message: str


class ModeSummaryRequest(BaseModel):
    """Request to generate a mode summary."""

    session_id: str


@app.get("/api/modes")
async def list_modes():
    """List all available modes."""
    runner = get_runner()
    modes = await runner.get_all_modes()
    return {"modes": [m.to_dict() for m in modes]}


@app.get("/api/modes/{mode_id}")
async def get_mode(mode_id: str):
    """Get details for a specific mode."""
    runner = get_runner()
    try:
        mode = await runner.get_mode(mode_id)
        return mode.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Mode not found: {mode_id}")


@app.post("/api/modes/sessions")
async def create_mode_session(request: CreateModeSessionRequest):
    """Create a new mode session."""
    runner = get_runner()

    # Verify mode exists
    try:
        mode = await runner.get_mode(request.mode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Mode not found: {request.mode_id}")

    # Use mode's default max_turns if not specified
    max_turns = request.max_turns or mode.protocol.get("max_turns_default", 12)

    session = await runner.create_session(
        mode_id=request.mode_id,
        initial_inquiry=request.initial_inquiry,
        model=request.model,
    )
    session.max_turns = max_turns

    return session.to_dict()


@app.get("/api/modes/sessions/{session_id}")
async def get_mode_session(session_id: str):
    """Get the current state of a mode session."""
    runner = get_runner()
    session = await runner.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Include active ledger view
    active_view = runner.get_active_ledger_view(session.ledger)

    result = session.to_dict()
    result["ledger_active_view"] = active_view
    return result


@app.post("/api/modes/{mode_id}/turn")
async def run_mode_turn(mode_id: str, request: ModeTurnRequest):
    """Run a turn in a mode session.

    Generic endpoint that works with any interactive mode.
    """
    from .council import query_model

    runner = get_runner()
    session = await runner.get_session(request.session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.mode_id != mode_id:
        raise HTTPException(status_code=400, detail=f"Session is for mode '{session.mode_id}', not '{mode_id}'")

    if session.status != "active":
        raise HTTPException(status_code=400, detail=f"Session is not active (status: {session.status})")

    # Get mode-specific runner
    if mode_id == "socrates":
        socrates_runner = SocratesRunner(runner.modes_dir)

        # Build messages for LLM
        messages = socrates_runner.build_messages_for_llm(session, request.user_message)

        # Add user message to session history
        session.messages.append({"role": "user", "content": request.user_message})

        # Get model to use
        model = session.model
        if not model:
            settings = get_settings()
            model = settings.chairman_model  # Use chairman model as default for Socrates

        # Query the model
        start_time = asyncio.get_event_loop().time()
        try:
            response = await query_model(model=model, messages=messages, timeout=120.0, temperature=0.7)
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model query failed: {str(e)}")

        if response.get("error"):
            raise HTTPException(status_code=500, detail=f"Model error: {response.get('content', 'Unknown error')}")

        raw_response = response.get("content", "")

        # Process the response with 3-pass JSON recovery
        result = socrates_runner.process_turn_response(raw_response, session)
        output = result["output"]

        # Add assistant response to session history
        session.messages.append({"role": "assistant", "content": raw_response})

        # Merge ledger update
        if output.get("ledger_update"):
            session.ledger = runner.merge_ledger(session.ledger, output["ledger_update"])

        # Check stop conditions
        stop_recommended = socrates_runner.should_recommend_stop(output, session)
        criteria_met = socrates_runner.get_stop_criteria_met(output, session)

        # Create turn receipt
        turn_receipt = {
            "turn": session.turn_count + 1,
            "model": model,
            "duration_ms": duration_ms,
            "parse_success": result["success"],
            "parse_error": result.get("error"),
            "stop_recommended": stop_recommended,
            "criteria_met": criteria_met,
        }

        # Update session
        updated_session = runner.session_store.update_session(
            session_id=session.session_id,
            ledger=session.ledger,
            messages=session.messages,
            turn_receipt=turn_receipt,
            stop_recommended=stop_recommended,
            stop_criteria_met=criteria_met,
        )

        # Build response - use updated_session for correct turn_count
        return {
            "success": result["success"],
            "turn": updated_session.turn_count,
            "output": {
                "next_question": output.get("next_question"),
                "question_type": output.get("question_type"),
                "question_type_detail": output.get("question_type_detail"),
                "why_this_question": output.get("why_this_question"),
                "stop_check": output.get("stop_check"),
            },
            "ledger_active_view": runner.get_active_ledger_view(updated_session.ledger),
            "stop_recommended": stop_recommended,
            "criteria_met": criteria_met,
            "parse_error": result.get("error"),
            "session_status": updated_session.status,
            "turns_remaining": updated_session.max_turns - updated_session.turn_count,
            "session": updated_session.to_dict(),
        }
    else:
        raise HTTPException(status_code=400, detail=f"Mode '{mode_id}' does not support interactive turns")


@app.post("/api/modes/{mode_id}/stop")
async def stop_mode_session(mode_id: str, request: ModeSummaryRequest):
    """Stop a mode session and generate summary."""
    from .council import query_model

    runner = get_runner()
    session = await runner.get_session(request.session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.mode_id != mode_id:
        raise HTTPException(status_code=400, detail=f"Session is for mode '{session.mode_id}', not '{mode_id}'")

    if mode_id == "socrates":
        socrates_runner = SocratesRunner(runner.modes_dir)

        # Build summary prompt
        summary_prompt = socrates_runner.build_summary_prompt(session)

        # Get model
        model = session.model
        if not model:
            settings = get_settings()
            model = settings.chairman_model

        # Query for summary
        messages = [
            {"role": "system", "content": socrates_runner.system_prompt},
            {"role": "user", "content": summary_prompt},
        ]

        try:
            response = await query_model(model=model, messages=messages, timeout=120.0, temperature=0.5)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

        raw_summary = response.get("content", "")

        # Parse summary (also uses JSON recovery)
        from .modes.json_recovery import parse_json

        parsed_summary, error = parse_json(raw_summary)

        # Mark session as completed
        runner.session_store.complete_session(session.session_id)

        return {
            "session_id": session.session_id,
            "status": "completed",
            "turn_count": session.turn_count,
            "summary": parsed_summary or {"raw": raw_summary},
            "parse_error": error,
            "final_ledger": session.ledger,
            "final_ledger_active_view": runner.get_active_ledger_view(session.ledger),
        }
    else:
        raise HTTPException(status_code=400, detail=f"Mode '{mode_id}' does not support stop summary")


@app.delete("/api/modes/sessions/{session_id}")
async def delete_mode_session(session_id: str):
    """Delete a mode session."""
    runner = get_runner()
    deleted = runner.session_store.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted"}


@app.get("/api/glossary")
async def get_glossary():
    """Get the fallacies glossary."""
    glossary_path = Path(__file__).parent / "resources" / "fallacies_glossary.json"

    if not glossary_path.exists():
        raise HTTPException(status_code=404, detail="Glossary not found")

    return json.loads(glossary_path.read_text())


@app.get("/api/glossary/{fallacy_id}")
async def get_fallacy(fallacy_id: str):
    """Get a specific fallacy from the glossary."""
    glossary_path = Path(__file__).parent / "resources" / "fallacies_glossary.json"

    if not glossary_path.exists():
        raise HTTPException(status_code=404, detail="Glossary not found")

    glossary = json.loads(glossary_path.read_text())

    for fallacy in glossary:
        if fallacy.get("id") == fallacy_id:
            return fallacy

    raise HTTPException(status_code=404, detail=f"Fallacy not found: {fallacy_id}")


# ========================================
# OPENAI-COMPATIBLE API ENDPOINTS
# ========================================
# These endpoints allow OpenWebUI and other OpenAI-compatible
# clients to use Roundtable Mode as a selectable "model".


@app.get("/v1/models")
async def list_openai_models():
    """List available models (OpenAI-compatible).

    Returns Roundtable variants as available models.
    """
    return {"object": "list", "data": get_roundtable_models()}


@app.post("/v1/chat/completions")
async def openai_chat_completions(body: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions endpoint.

    Routes to Roundtable Mode for collaborative multi-model deliberation.
    Supports streaming and non-streaming responses.

    Configuration options (can be passed in request body):
    - council_models: List of model IDs for the council
    - num_rounds: Number of deliberation rounds (1-5)
    - chair_model: Model for final synthesis
    - moderator_model: Model for moderator synthesis
    - max_parallel: Max concurrent model queries

    Model variants:
    - "roundtable" - Default settings
    - "roundtable:fast" - 1 round, quick mode
    - "roundtable:thorough" - 3 rounds
    - "roundtable:deep" - 5 rounds
    - "roundtable:N" - N rounds (e.g., "roundtable:2")
    """
    if body.stream:
        return StreamingResponse(
            generate_openai_stream(body, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        response = await generate_non_streaming_response(body, request)
        return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
