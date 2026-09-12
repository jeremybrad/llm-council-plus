"""WOR-402: predicted call budget, attempted-call ledger, subscription parallelism."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.roundtable import (
    DEFAULT_MAX_CALLS_PER_RUN,
    QUOTA_UNITS_UNKNOWN,
    AgentConfig,
    predict_roundtable_calls,
    run_roundtable,
    uses_subscription_seat,
)


def test_predict_default_council():
    assert predict_roundtable_calls(4, 3) == 14
    assert predict_roundtable_calls(4, 3) == DEFAULT_MAX_CALLS_PER_RUN
    assert predict_roundtable_calls(4, 1) == 6  # roundtable:fast


def test_quota_units_are_unknown_not_zero():
    assert QUOTA_UNITS_UNKNOWN == "unknown"
    assert QUOTA_UNITS_UNKNOWN != 0
    assert QUOTA_UNITS_UNKNOWN != "0"


def test_subscription_seat_detection():
    assert uses_subscription_seat(["agentcli:claude", "openrouter:x"])
    assert uses_subscription_seat(["agentcli:grok"])
    assert not uses_subscription_seat(["mock:builder", "openai:gpt-4o"])


@pytest.mark.asyncio
async def test_budget_fail_fast_does_not_call_models(mock_query_model, mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 5  # 3 agents * 3 rounds + 2 = 11
    agents = [
        AgentConfig(model="mock:builder", role="builder", label="Builder"),
        AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
        AgentConfig(model="mock:contrarian", role="contrarian", label="Contrarian"),
    ]
    with (
        patch("backend.roundtable.query_model", mock_query_model),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        events = []
        async for event in run_roundtable(
            conversation_id="budget-1",
            question="q",
            agents=agents,
            moderator_model="mock:moderator",
            chair_model="mock:chair",
            num_rounds=3,
        ):
            events.append(event)

    types = [e["type"] for e in events]
    assert "roundtable_init" in types
    assert "roundtable_budget_exceeded" in types
    assert "round_start" not in types
    assert mock_query_model.await_count == 0
    exceeded = next(e for e in events if e["type"] == "roundtable_budget_exceeded")
    assert exceeded["predicted_calls"] == 11
    assert exceeded["max_calls_per_run"] == 5
    assert exceeded["quota_units"] == "unknown"
    assert exceeded["run"]["status"] == "budget_exceeded"
    assert (tmp_path / "sadb").exists() is False or not any((tmp_path / "sadb").rglob("*.jsonl"))


@pytest.mark.asyncio
async def test_attempted_calls_include_failures(mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    agents = [
        AgentConfig(model="mock:builder", role="builder", label="Builder"),
        AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
    ]

    async def flaky(model, messages, timeout=120.0, temperature=0.7):
        if model == "mock:skeptic":
            return {"error": True, "error_message": "transient"}
        return {"content": f"ok from {model}", "error": False}

    with (
        patch("backend.roundtable.query_model", AsyncMock(side_effect=flaky)),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        run = None
        async for event in run_roundtable(
            conversation_id="budget-2",
            question="q",
            agents=agents,
            moderator_model="mock:moderator",
            chair_model="mock:chair",
            num_rounds=1,
        ):
            if event["type"] == "chair_complete":
                run = event["run"]

    assert run is not None
    accounting = run["call_accounting"]
    assert accounting["predicted_calls"] == 4  # 2 agents * 1 round + 2
    assert accounting["attempted_calls"] == 4
    assert accounting["failed_calls"] == 1
    assert accounting["quota_units"] == "unknown"
    assert any(item["failed"] for item in accounting["attempts"] if item["model"] == "mock:skeptic")


@pytest.mark.asyncio
async def test_cancellation_records_partial_attempts(mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    agents = [
        AgentConfig(model="mock:builder", role="builder", label="Builder"),
        AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
    ]
    started = asyncio.Event()

    async def slow(model, messages, timeout=120.0, temperature=0.7):
        started.set()
        await asyncio.sleep(60)
        return {"content": "late", "error": False}

    with (
        patch("backend.roundtable.query_model", AsyncMock(side_effect=slow)),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        events = []

        async def consume():
            async for event in run_roundtable(
                conversation_id="budget-3",
                question="q",
                agents=agents,
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                num_rounds=1,
                max_parallel=1,
            ):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    aborted = next(e for e in events if e["type"] == "roundtable_aborted")
    accounting = aborted["run"]["call_accounting"]
    assert accounting["attempted_calls"] >= 1
    assert accounting["attempted_calls"] < accounting["predicted_calls"]
    assert accounting["failed_calls"] >= 1
    assert accounting["quota_units"] == "unknown"


@pytest.mark.asyncio
async def test_subscription_parallelism_capped(mock_query_model, mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    mock_settings.roundtable_subscription_max_parallel = 2
    agents = [
        AgentConfig(model="agentcli:claude", role="builder", label="Builder"),
        AgentConfig(model="agentcli:grok", role="skeptic", label="Skeptic"),
        AgentConfig(model="agentcli:codex", role="contrarian", label="Contrarian"),
    ]
    with (
        patch("backend.roundtable.query_model", mock_query_model),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        init = None
        async for event in run_roundtable(
            conversation_id="budget-4",
            question="q",
            agents=agents,
            moderator_model="agentcli:claude",
            chair_model="agentcli:claude",
            num_rounds=1,
            max_parallel=8,
        ):
            if event["type"] == "roundtable_init":
                init = event
                break
    assert init is not None
    assert init["max_parallel"] == 2


@pytest.mark.asyncio
async def test_subscription_parallelism_bounds_inflight(mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    mock_settings.roundtable_subscription_max_parallel = 2
    agents = [
        AgentConfig(model="agentcli:claude", role="builder", label="Builder"),
        AgentConfig(model="agentcli:grok", role="skeptic", label="Skeptic"),
        AgentConfig(model="agentcli:codex", role="contrarian", label="Contrarian"),
    ]
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def gated(model, messages, timeout=120.0, temperature=0.7):
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return {"content": f"ok {model}", "error": False}

    with (
        patch("backend.roundtable.query_model", gated),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        async for _event in run_roundtable(
            conversation_id="budget-5",
            question="q",
            agents=agents,
            moderator_model="agentcli:claude",
            chair_model="agentcli:claude",
            num_rounds=1,
            max_parallel=8,
        ):
            pass
    assert peak == 2
    assert current == 0


@pytest.mark.asyncio
async def test_subscription_parallelism_zero_serializes(mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    mock_settings.roundtable_subscription_max_parallel = 0
    agents = [
        AgentConfig(model="agentcli:claude", role="builder", label="Builder"),
        AgentConfig(model="agentcli:grok", role="skeptic", label="Skeptic"),
        AgentConfig(model="agentcli:codex", role="contrarian", label="Contrarian"),
    ]
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def gated(model, messages, timeout=120.0, temperature=0.7):
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return {"content": f"ok {model}", "error": False}

    with (
        patch("backend.roundtable.query_model", gated),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        async for _event in run_roundtable(
            conversation_id="budget-6",
            question="q",
            agents=agents,
            moderator_model="agentcli:claude",
            chair_model="agentcli:claude",
            num_rounds=1,
            max_parallel=8,
        ):
            pass
    assert peak == 1
    assert current == 0


@pytest.mark.asyncio
async def test_per_run_max_parallel_zero_serializes(mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    mock_settings.roundtable_subscription_max_parallel = 2
    agents = [
        AgentConfig(model="agentcli:claude", role="builder", label="Builder"),
        AgentConfig(model="agentcli:grok", role="skeptic", label="Skeptic"),
        AgentConfig(model="agentcli:codex", role="contrarian", label="Contrarian"),
    ]
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def gated(model, messages, timeout=120.0, temperature=0.7):
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return {"content": f"ok {model}", "error": False}

    with (
        patch("backend.roundtable.query_model", gated),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        async for _event in run_roundtable(
            conversation_id="budget-8",
            question="q",
            agents=agents,
            moderator_model="agentcli:claude",
            chair_model="agentcli:claude",
            num_rounds=1,
            max_parallel=0,
        ):
            pass
    assert peak == 1
    assert current == 0


@pytest.mark.asyncio
async def test_per_run_max_parallel_can_only_lower_ceiling(mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    mock_settings.roundtable_subscription_max_parallel = 2
    agents = [
        AgentConfig(model="agentcli:claude", role="builder", label="Builder"),
        AgentConfig(model="agentcli:grok", role="skeptic", label="Skeptic"),
        AgentConfig(model="agentcli:codex", role="contrarian", label="Contrarian"),
    ]
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def gated(model, messages, timeout=120.0, temperature=0.7):
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return {"content": f"ok {model}", "error": False}

    with (
        patch("backend.roundtable.query_model", gated),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        async for _event in run_roundtable(
            conversation_id="budget-9",
            question="q",
            agents=agents,
            moderator_model="agentcli:claude",
            chair_model="agentcli:claude",
            num_rounds=1,
            max_parallel=1,
        ):
            pass
    assert peak == 1
    assert current == 0


@pytest.mark.asyncio
async def test_peer_failure_does_not_mark_siblings_cancelled(mock_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 20
    agents = [
        AgentConfig(model="mock:builder", role="builder", label="Builder"),
        AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
    ]

    async def mixed(model, messages, timeout=120.0, temperature=0.7):
        if model == "mock:skeptic":
            return {"content": "", "error": True, "error_message": "peer boom"}
        return {"content": f"ok {model}", "error": False}

    with (
        patch("backend.roundtable.query_model", mixed),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        events = []
        async for event in run_roundtable(
            conversation_id="budget-7",
            question="q",
            agents=agents,
            moderator_model="mock:moderator",
            chair_model="mock:chair",
            num_rounds=1,
            max_parallel=2,
        ):
            events.append(event)
    complete = next(e for e in events if e["type"] == "chair_complete")
    accounting = complete["run"]["call_accounting"]
    failed_models = {item["model"] for item in accounting["attempts"] if item["failed"]}
    ok_models = {item["model"] for item in accounting["attempts"] if not item["failed"]}
    assert "mock:skeptic" in failed_models
    assert "mock:builder" in ok_models
    assert accounting["quota_units"] == "unknown"
