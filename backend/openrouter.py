"""OpenRouter API client for making LLM requests."""

from typing import Any

import httpx

from .config import get_openrouter_api_key  # noqa: F401  (kept importable for callers/tests)

METERED_REFUSAL = (
    "route refused: provider 'openrouter' is a metered API route; refused under the $0 "
    "metered-model budget (an enabled flag or stored key is not spending authority)"
)


async def query_model(
    model: str, messages: list[dict[str, str]], timeout: float = 120.0, temperature: float = 0.7
) -> dict[str, Any] | None:
    """Refuse OpenRouter inference unconditionally (WOR-362 $0 metered-model budget).

    The upstream HTTP implementation was removed so no code path in this module can send
    an inference request; fetch_models (a model-list GET) is unchanged.
    """
    # Lowest-layer guard (WOR-362): OpenRouter inference is metered. No caller, flag or
    # stored key may reach the network from here; query_models_parallel inherits this.
    return {"error": True, "error_message": METERED_REFUSAL}


async def query_models_parallel(
    models: list[str], messages: list[dict[str, str]], temperature: float = 0.7
) -> dict[str, dict[str, Any] | None]:
    """
    Query multiple models in parallel with batching for rate limit protection.

    For 6+ models, processes in batches of 3 to avoid hitting rate limits.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # For 6+ models, use batching to avoid rate limits
    BATCH_SIZE = 3
    if len(models) >= 6:
        print(f"Batching {len(models)} models into groups of {BATCH_SIZE} to avoid rate limits")
        results = {}

        # Process in batches
        for i in range(0, len(models), BATCH_SIZE):
            batch = models[i : i + BATCH_SIZE]
            print(f"Processing batch {i // BATCH_SIZE + 1}: {batch}")

            # Create tasks for this batch
            tasks = [query_model(model, messages, temperature=temperature) for model in batch]

            # Wait for batch to complete
            batch_responses = await asyncio.gather(*tasks)

            # Map batch results
            for model, response in zip(batch, batch_responses, strict=False):
                results[model] = response

            # Small delay between batches (except for last batch)
            if i + BATCH_SIZE < len(models):
                await asyncio.sleep(0.5)

        return results

    # For 5 or fewer models, send all at once (original behavior)
    tasks = [query_model(model, messages, temperature=temperature) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return dict(zip(models, responses, strict=False))


async def fetch_models() -> list[dict[str, Any]]:
    """
    Fetch available models from OpenRouter API.
    Returns a list of model definitions compatible with the frontend.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://openrouter.ai/api/v1/models")
            if response.status_code != 200:
                print(f"Failed to fetch OpenRouter models: {response.status_code}")
                return []

            data = response.json()
            models = []
            for item in data.get("data", []):
                # Determine if free based on pricing
                pricing = item.get("pricing", {})
                is_free = float(pricing.get("prompt", 0)) == 0 and float(pricing.get("completion", 0)) == 0

                # Extract provider from ID (e.g. "google/gemini" -> "Google")
                model_id = item.get("id", "")
                provider = "OpenRouter"  # Default
                if "/" in model_id:
                    provider_slug = model_id.split("/")[0]
                    # Capitalize nicely
                    if provider_slug == "openai":
                        provider = "OpenAI"
                    elif provider_slug == "anthropic":
                        provider = "Anthropic"
                    elif provider_slug == "google":
                        provider = "Google"
                    elif provider_slug == "meta-llama":
                        provider = "Meta"
                    elif provider_slug == "mistralai":
                        provider = "Mistral"
                    elif provider_slug == "deepseek":
                        provider = "DeepSeek"
                    else:
                        provider = provider_slug.title()

                models.append(
                    {
                        "id": model_id,
                        "name": item.get("name"),
                        "provider": provider,
                        "source": "openrouter",  # Explicitly set for frontend grouping
                        "context_length": item.get("context_length"),
                        "is_free": is_free,
                        "pricing": pricing,
                    }
                )

            return models
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")
        return []
