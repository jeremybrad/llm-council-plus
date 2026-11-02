"""WOR-1016 / WOR-362: honest run outcome and fail-closed model routing.

Every test here is synthetic. Provider dispatch, storage and SADB capture are
stubbed, and conftest's session isolation blocks network access and points all
data paths, settings and SADB_DATA_DIR at a disposable root.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend import council, main, openai_compat, settings, storage, turn_capture
from backend.providers.openrouter import OpenRouterProvider
from backend.roundtable import (
    CAPTURABLE_RUN_STATUSES,
    MAX_EXECUTED_ROUNDS,
    AgentConfig,
    run_roundtable,
)
from tests.conftest import ISOLATED_SADB_ROOT, ISOLATION_ROOT, NetworkBlockedError

REAL_SADB_ROOT = (Path.home() / "SyncedProjects" / "_SADB_Data").resolve()


# ---------------------------------------------------------------------------
# Isolation: verify the effective paths, not just the environment variable.
# ---------------------------------------------------------------------------


def test_effective_paths_are_disposable():
    sadb_root = turn_capture.resolve_sadb_root().resolve()
    assert sadb_root == ISOLATED_SADB_ROOT
    assert not str(sadb_root).startswith(str(REAL_SADB_ROOT))
    for path in (
        Path(storage.DATA_DIR).resolve(),
        Path(storage.RUNS_DIR).resolve(),
        Path(settings.SETTINGS_FILE).resolve(),
        storage.get_run_path("c", "r").resolve(),
    ):
        assert ISOLATION_ROOT in path.parents, path


def test_network_is_blocked():
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("127.0.0.1", 9))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()


def test_capture_writes_only_under_isolated_root():
    run = _run_dict("completed")
    out = turn_capture.capture_run(run, session_id="isolation-check")
    assert out is not None
    assert ISOLATION_ROOT in Path(out).resolve().parents
    assert not str(Path(out).resolve()).startswith(str(REAL_SADB_ROOT))
    meta = [json.loads(line)["meta"] for line in Path(out).read_text().splitlines()]
    assert {m["run_status"] for m in meta} == {"completed"}


# ---------------------------------------------------------------------------
# Fail-closed routing (Jeremy's WOR-362 ruling, 2026-11-01).
# ---------------------------------------------------------------------------


def _settings(**enabled):
    base = {"openrouter": False, "ollama": False, "groq": False, "direct": False, "custom": False, "agentcli": False}
    base.update(enabled)
    return SimpleNamespace(enabled_providers=base, direct_provider_toggles={"openai": False})


def test_default_settings_do_not_enable_openrouter():
    assert settings.DEFAULT_ENABLED_PROVIDERS["openrouter"] is False
    assert settings.Settings().enabled_providers["openrouter"] is False


@pytest.mark.parametrize(
    ("model", "fragment"),
    [
        ("", "blank"),
        ("   ", "blank"),
        (None, "blank"),
        ("gpt-4o", "no provider prefix"),
        ("anthropic/claude-sonnet-4", "no provider prefix"),
        ("antigravity:gemini", "unknown provider prefix"),
        ("mock:builder", "unknown provider prefix"),
        ("openrouter:openai/gpt-4o", "metered"),
        ("openai:gpt-4o", "metered"),
    ],
)
def test_refused_routes(model, fragment):
    reason = council.check_model_route(model, _settings(ollama=True))
    assert reason and fragment in reason


def test_enabled_route_is_allowed():
    assert council.check_model_route("ollama:llama3.2", _settings(ollama=True)) is None


def test_disabled_non_metered_provider_is_refused():
    assert "disabled" in council.check_model_route("ollama:llama3.2", _settings())
    assert "disabled" in council.check_model_route("agentcli:claude", _settings(ollama=True))


def test_get_provider_for_model_never_falls_back():
    for model in ("", "gpt-4o", "antigravity:x"):
        with pytest.raises(council.ModelRouteError):
            council.get_provider_for_model(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["", "gpt-4o", "antigravity:x", "openrouter:openai/gpt-4o", "ollama:x"])
async def test_query_model_refuses_without_contacting_any_provider(monkeypatch, model):
    spies = {name: AsyncMock(return_value={"content": "should not happen"}) for name in council.PROVIDERS}
    for name, provider in council.PROVIDERS.items():
        monkeypatch.setattr(provider, "query", spies[name])
    # A stored key and a default-enabled flag must not authorize this route.
    monkeypatch.setattr(council, "get_settings", lambda: _settings())
    result = await council.query_model(model, [{"role": "user", "content": "q"}])
    assert result["error"] is True
    assert result["error_message"].startswith("route refused:")
    for spy in spies.values():
        spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_openrouter_provider_refuses_when_disabled(monkeypatch):
    inner = AsyncMock(return_value={"content": "should not happen"})
    monkeypatch.setattr("backend.providers.openrouter.openrouter.query_model", inner)
    monkeypatch.setattr(
        "backend.providers.openrouter.get_settings",
        lambda: SimpleNamespace(enabled_providers={"openrouter": False}, openrouter_api_key="stored-key"),
    )
    result = await OpenRouterProvider().query("openrouter:x", [{"role": "user", "content": "q"}])
    assert result["error"] is True and "metered" in result["error_message"]
    inner.assert_not_awaited()


def test_preflight_rejects_default_blank_configuration():
    defaults = settings.Settings()
    problems = council.preflight_roundtable_models(
        defaults.council_models, defaults.chairman_model, defaults.chairman_model, defaults
    )
    joined = "; ".join(problems)
    assert "council seat 1: model id is blank" in joined
    assert "council seat 2: model id is blank" in joined
    assert "moderator: model id is blank" in joined
    assert "chair: model id is blank" in joined


def test_preflight_accepts_enabled_configuration():
    s = _settings(ollama=True)
    assert council.preflight_roundtable_models(["ollama:a", "ollama:b"], "ollama:c", "ollama:c", s) == []


async def _collect_main_stream(monkeypatch, settings_obj, fake_roundtable):
    monkeypatch.setattr(main, "run_roundtable", fake_roundtable)
    monkeypatch.setattr(main, "get_settings", lambda: settings_obj)
    monkeypatch.setattr(main.storage, "get_conversation", lambda _: {"messages": [{"role": "user"}]})
    mocks = {name: Mock() for name in ["add_user_message", "save_run", "add_roundtable_message", "add_error_message"]}
    for name, mock in mocks.items():
        monkeypatch.setattr(main.storage, name, mock)
    capture = Mock()
    monkeypatch.setattr(turn_capture, "capture_run", capture)
    response = await main.send_message_stream(
        "synthetic",
        main.SendMessageRequest(content="synthetic only", execution_mode="roundtable", web_search=False),
        SimpleNamespace(),
    )
    events = []
    async for chunk in response.body_iterator:
        events.append(json.loads(chunk.removeprefix("data: ").strip()))
    return events, mocks, capture


@pytest.mark.asyncio
async def test_ui_stream_refuses_blank_configuration_before_dispatch(monkeypatch):
    roundtable = Mock(side_effect=AssertionError("run_roundtable must not start"))
    blank = SimpleNamespace(
        council_models=["", "ollama:x"],
        chairman_model="",
        roundtable_num_rounds=1,
        roundtable_max_parallel=2,
        enabled_providers={"ollama": True},
        direct_provider_toggles={},
    )
    events, mocks, capture = await _collect_main_stream(monkeypatch, blank, roundtable)
    assert events[-1]["type"] == "error"
    assert "refused before any model call" in events[-1]["message"]
    assert "council seat 1: model id is blank" in events[-1]["message"]
    roundtable.assert_not_called()
    mocks["save_run"].assert_not_called()
    capture.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_v1_refuses_unknown_route_before_dispatch(monkeypatch, stream):
    roundtable = Mock(side_effect=AssertionError("run_roundtable must not start"))
    monkeypatch.setattr(openai_compat, "run_roundtable", roundtable)
    monkeypatch.setattr(
        openai_compat,
        "get_settings",
        lambda: SimpleNamespace(
            council_models=["ollama:a", "antigravity:b"],
            chairman_model="ollama:a",
            roundtable_num_rounds=1,
            roundtable_max_parallel=2,
            roundtable_timeout_seconds=5.0,
            enabled_providers={"ollama": True},
            direct_provider_toggles={},
        ),
    )
    request = openai_compat.ChatCompletionRequest(
        model="roundtable:fast", messages=[openai_compat.ChatMessage(role="user", content="q")], stream=stream
    )
    if stream:
        text = "".join([chunk async for chunk in openai_compat.generate_openai_stream(request)])
    else:
        text = (await openai_compat.generate_non_streaming_response(request)).choices[0].message.content
    assert "refused before any model call" in text
    assert "unknown provider prefix 'antigravity'" in text
    roundtable.assert_not_called()


# ---------------------------------------------------------------------------
# Honest run outcome (WOR-1016 acceptance criteria).
# ---------------------------------------------------------------------------


def _agents(n=3):
    roles = ["builder", "skeptic", "contrarian", "historian"]
    return [AgentConfig(model=f"mock:{roles[i]}", role=roles[i], label=roles[i].title()) for i in range(n)]


def _rt_settings(budget=14):
    return SimpleNamespace(
        council_temperature=0.5,
        stage2_temperature=0.3,
        chairman_temperature=0.4,
        roundtable_debug_prompts=False,
        roundtable_timeout_seconds=5.0,
        roundtable_max_calls_per_run=budget,
        roundtable_subscription_max_parallel=2,
    )


def _scripted(seat_fn, moderator=None, chair=None):
    """Build a query_model stub: seats via seat_fn(model, round_hint); mod/chair fixed results."""
    calls = []

    async def fake(model, messages, timeout=120.0, temperature=0.5):
        calls.append(model)
        if model == "mock:moderator":
            return moderator or {"content": "moderator summary", "error": False}
        if model == "mock:chair":
            return chair or {"content": "chair answer", "error": False}
        return seat_fn(model)

    return fake, calls


async def _run(fake, agents, num_rounds=3, budget=14):
    with (
        patch("backend.roundtable.query_model", fake),
        patch("backend.roundtable.get_settings", return_value=_rt_settings(budget)),
    ):
        return [
            e
            async for e in run_roundtable(
                conversation_id="wor1016",
                question="q",
                agents=agents,
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                num_rounds=num_rounds,
            )
        ]


@pytest.mark.asyncio
async def test_all_seats_fail_stops_after_round_one():
    fake, calls = _scripted(lambda m: {"error": True, "error_message": "synthetic outage"})
    events = await _run(fake, _agents(3))
    terminal = events[-1]
    assert terminal["type"] == "roundtable_failed"
    run = terminal["run"]
    assert run["status"] == "failed"
    assert run["failure_reason"] == "no council seat succeeded in round 1"
    assert run["failed_seats"] == ["Builder", "Skeptic", "Contrarian"]
    assert run["call_accounting"]["attempted_calls"] == 3
    assert run["call_accounting"]["failed_calls"] == 3
    assert "mock:moderator" not in calls and "mock:chair" not in calls
    assert len(run["rounds"]) == 1
    assert not any(e["type"] in {"moderator_start", "chair_start"} for e in events)


@pytest.mark.asyncio
async def test_one_failed_seat_is_degraded():
    fake, _ = _scripted(
        lambda m: {"error": True, "error_message": "skeptic down"} if m == "mock:skeptic" else {"content": "ok"}
    )
    events = await _run(fake, _agents(3))
    run = events[-1]["run"]
    assert events[-1]["type"] == "chair_complete"
    assert run["status"] == "degraded"
    assert run["failed_seats"] == ["Skeptic"]
    assert "Skeptic" in run["failure_reason"]


@pytest.mark.asyncio
async def test_chair_error_message_is_kept_and_run_fails():
    fake, _ = _scripted(lambda m: {"content": "ok"}, chair={"error": True, "error_message": "x"})
    run = (await _run(fake, _agents(2)))[-1]["run"]
    assert run["status"] == "failed"
    assert run["chair_final"]["error"] is True
    assert run["chair_final"]["error_message"] == "x"
    assert run["failure_reason"] == "chair failed: x"


@pytest.mark.asyncio
async def test_empty_chair_answer_is_not_success():
    fake, _ = _scripted(lambda m: {"content": "ok"}, chair={"content": "   ", "error": False})
    run = (await _run(fake, _agents(2)))[-1]["run"]
    assert run["status"] == "failed"
    assert run["chair_final"]["error_message"] == "chair returned empty content"


@pytest.mark.asyncio
async def test_moderator_failure_is_degraded_with_reason():
    fake, _ = _scripted(lambda m: {"content": "ok"}, moderator={"error": True, "error_message": "mod timeout"})
    run = (await _run(fake, _agents(2)))[-1]["run"]
    assert run["status"] == "degraded"
    assert run["moderator_summary"]["error_message"] == "mod timeout"
    assert "moderator failed: mod timeout" in run["failure_reason"]


@pytest.mark.asyncio
async def test_all_success_is_completed():
    fake, _ = _scripted(lambda m: {"content": "ok"})
    run = (await _run(fake, _agents(2)))[-1]["run"]
    assert run["status"] == "completed"
    assert run["failure_reason"] is None and run["failed_seats"] == []


@pytest.mark.asyncio
async def test_empty_error_message_is_never_a_falsy_error():
    # httpx ReadTimeout stringifies to '' and providers forward it verbatim.
    fake, _ = _scripted(lambda m: {"error": True, "error_message": ""} if m == "mock:skeptic" else {"content": "ok"})
    run = (await _run(fake, _agents(2)))[-1]["run"]
    skeptic = [r for r in run["rounds"][0]["responses"] if r["agent_label"] == "Skeptic"][0]
    assert skeptic["error"]
    assert run["status"] == "degraded"
    ledger = [a for a in run["call_accounting"]["attempts"] if a["model"] == "mock:skeptic"]
    assert all(a["failed"] and a["error_message"] for a in ledger)


@pytest.mark.asyncio
async def test_exception_with_empty_message_keeps_type_name():
    import asyncio

    def seat(m):
        if m == "mock:skeptic":
            raise asyncio.TimeoutError()
        return {"content": "ok"}

    fake, _ = _scripted(seat)
    run = (await _run(fake, _agents(2), num_rounds=1))[-1]["run"]
    skeptic = [r for r in run["rounds"][0]["responses"] if r["agent_label"] == "Skeptic"][0]
    assert skeptic["error"] == "TimeoutError"


@pytest.mark.asyncio
async def test_empty_seat_content_counts_as_failure():
    fake, _ = _scripted(lambda m: {"content": ""} if m == "mock:skeptic" else {"content": "ok"})
    run = (await _run(fake, _agents(2), num_rounds=1))[-1]["run"]
    assert run["status"] == "degraded"
    assert run["failed_seats"] == ["Skeptic"]


@pytest.mark.asyncio
async def test_rounds_are_capped_before_prediction():
    fake, calls = _scripted(lambda m: {"content": "ok"})
    events = await _run(fake, _agents(4), num_rounds=5, budget=14)
    init = events[0]
    assert init["total_rounds"] == MAX_EXECUTED_ROUNDS == 3
    assert init["requested_rounds"] == 5
    assert init["call_accounting"]["predicted_calls"] == 14
    run = events[-1]["run"]
    assert events[-1]["type"] == "chair_complete"
    assert len(run["rounds"]) == 3
    assert run["requested_rounds"] == 5 and run["executed_rounds"] == 3
    assert run["call_accounting"]["attempted_calls"] == 14 == len(calls)


@pytest.mark.asyncio
async def test_budget_rejection_records_reason():
    fake, calls = _scripted(lambda m: {"content": "ok"})
    events = await _run(fake, _agents(3), num_rounds=3, budget=5)
    run = events[-1]["run"]
    assert run["status"] == "budget_exceeded"
    assert "exceed" in run["failure_reason"]
    assert calls == []


# ---------------------------------------------------------------------------
# Capture gate and downstream presentation.
# ---------------------------------------------------------------------------


def _run_dict(status):
    return {
        "run_id": f"run-{status}",
        "conversation_id": "synthetic",
        "question": "synthetic question",
        "status": status,
        "failure_reason": None if status == "completed" else f"{status} reason",
        "failed_seats": ["Skeptic"] if status == "degraded" else [],
        "council": [],
        "rounds": [],
        "moderator_summary": {"model": "mock:m", "content": "summary", "error": False},
        "chair_final": {"model": "mock:c", "content": "answer", "error": False},
        "call_accounting": {"predicted_calls": 4, "attempted_calls": 4, "failed_calls": 0},
        "created_at": "2026-11-01T00:00:00",
        "completed_at": "2026-11-01T00:01:00",
    }


def test_capturable_statuses_match_between_modules():
    assert turn_capture.CAPTURABLE_RUN_STATUSES == CAPTURABLE_RUN_STATUSES == {"completed", "degraded"}


@pytest.mark.parametrize("status", ["failed", "aborted", "budget_exceeded", "running", None])
def test_capture_run_refuses_non_capturable_status(status, monkeypatch):
    append = Mock()
    monkeypatch.setattr(turn_capture, "append_turns", append)
    run = _run_dict("completed")
    run["status"] = status
    assert turn_capture.capture_run(run, session_id="refuse") is None
    append.assert_not_called()


def test_captured_turns_carry_degraded_status():
    out = turn_capture.capture_run(_run_dict("degraded"), session_id="degraded-check")
    meta = [json.loads(line)["meta"] for line in Path(out).read_text().splitlines()]
    assert {m["run_status"] for m in meta} == {"degraded"}
    assert all(m["failed_seats"] == ["Skeptic"] for m in meta)


def _ui_settings():
    return SimpleNamespace(
        council_models=["ollama:a", "ollama:b"],
        chairman_model="ollama:c",
        roundtable_num_rounds=5,
        roundtable_max_parallel=2,
        enabled_providers={"ollama": True},
        direct_provider_toggles={},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal", "status", "captured", "client_event"),
    [
        ("chair_complete", "completed", True, "roundtable_complete"),
        ("chair_complete", "degraded", True, "roundtable_complete"),
        ("chair_complete", "failed", False, "roundtable_error"),
        ("roundtable_failed", "failed", False, "roundtable_error"),
        ("roundtable_aborted", "aborted", False, "roundtable_error"),
        ("roundtable_budget_exceeded", "budget_exceeded", False, "roundtable_error"),
    ],
)
async def test_ui_stream_capture_gate_and_outcome(monkeypatch, terminal, status, captured, client_event):
    run = _run_dict(status)
    seen = {}

    async def fake_roundtable(**kwargs):
        seen.update(kwargs)
        yield {"type": "roundtable_init", "call_accounting": run["call_accounting"]}
        yield {"type": terminal, "run": run, "chair_final": run["chair_final"], "predicted_calls": 4}

    events, mocks, capture = await _collect_main_stream(monkeypatch, _ui_settings(), fake_roundtable)
    assert seen["num_rounds"] == 3  # requested 5, capped before the engine is called
    start = next(e for e in events if e["type"] == "roundtable_start")
    assert start["data"]["num_rounds"] == 3 and start["data"]["requested_rounds"] == 5
    terminal_events = [e for e in events if e["type"] in {"roundtable_complete", "roundtable_error"}]
    assert terminal_events[-1]["type"] == client_event
    assert terminal_events[-1]["data"]["status"] == status
    mocks["save_run"].assert_called_once_with(run)
    metadata = mocks["add_roundtable_message"].call_args.kwargs["metadata"]
    assert metadata["status"] == status
    assert metadata["requested_rounds"] == 5
    if captured:
        capture.assert_called_once_with(run)
    else:
        capture.assert_not_called()


def _v1_settings():
    return SimpleNamespace(
        council_models=["ollama:a", "ollama:b"],
        chairman_model="ollama:c",
        roundtable_num_rounds=1,
        roundtable_max_parallel=2,
        roundtable_timeout_seconds=5.0,
        enabled_providers={"ollama": True},
        direct_provider_toggles={},
    )


def _v1_request(stream):
    return openai_compat.ChatCompletionRequest(
        model="roundtable:fast", messages=[openai_compat.ChatMessage(role="user", content="q")], stream=stream
    )


def _v1_fake(terminal_event):
    async def fake(**kwargs):
        yield {"type": "roundtable_init", "total_rounds": 1, "council_members": [{}, {}]}
        yield {"type": "round_start", "round_number": 1, "round_name": "opening"}
        yield {
            "type": "round_progress",
            "response": {"agent_label": "Skeptic", "error": "synthetic outage", "content": ""},
        }
        yield terminal_event

    return fake


@pytest.mark.asyncio
async def test_v1_failed_council_is_an_error_not_an_answer(monkeypatch):
    run = _run_dict("failed")
    run["failure_reason"] = "no council seat succeeded in round 1"
    monkeypatch.setattr(openai_compat, "get_settings", _v1_settings)
    monkeypatch.setattr(openai_compat, "run_roundtable", _v1_fake({"type": "roundtable_failed", "run": run}))

    with pytest.raises(openai_compat.RoundtableFailedError) as exc:
        await openai_compat.generate_non_streaming_response(_v1_request(False))
    assert "no council seat succeeded" in exc.value.message

    text = "".join([c async for c in openai_compat.generate_openai_stream(_v1_request(True))])
    assert "Skeptic failed: synthetic outage" in text
    assert "Roundtable failed: no council seat succeeded in round 1" in text
    assert "responded" not in text


@pytest.mark.asyncio
async def test_v1_route_returns_502_for_failed_council(monkeypatch):
    async def failing(body, request):
        raise openai_compat.RoundtableFailedError("[Roundtable failed: x]", {"status": "failed", "failed_seats": []})

    monkeypatch.setattr(main, "generate_non_streaming_response", failing)
    response = await main.openai_chat_completions(_v1_request(False), SimpleNamespace())
    assert response.status_code == 502
    body = json.loads(response.body)
    assert body["error"]["type"] == "roundtable_failed"
    assert "choices" not in body


@pytest.mark.asyncio
async def test_v1_chair_failure_after_rounds_is_an_error(monkeypatch):
    run = _run_dict("failed")
    run["failure_reason"] = "chair failed: x"
    run["chair_final"] = {"model": "ollama:c", "content": "", "error": True, "error_message": "x"}
    event = {"type": "chair_complete", "chair_final": run["chair_final"], "run": run}
    monkeypatch.setattr(openai_compat, "get_settings", _v1_settings)
    monkeypatch.setattr(openai_compat, "run_roundtable", _v1_fake(event))
    with pytest.raises(openai_compat.RoundtableFailedError):
        await openai_compat.generate_non_streaming_response(_v1_request(False))
    text = "".join([c async for c in openai_compat.generate_openai_stream(_v1_request(True))])
    assert "Roundtable failed: chair failed: x" in text


@pytest.mark.asyncio
async def test_v1_degraded_answer_is_labelled(monkeypatch):
    run = _run_dict("degraded")
    run["failure_reason"] = "seat failures: Skeptic"
    event = {"type": "chair_complete", "chair_final": run["chair_final"], "run": run}
    monkeypatch.setattr(openai_compat, "get_settings", _v1_settings)
    monkeypatch.setattr(openai_compat, "run_roundtable", _v1_fake(event))
    response = await openai_compat.generate_non_streaming_response(_v1_request(False))
    content = response.choices[0].message.content
    assert content.startswith("[Degraded council: seat failures: Skeptic]")
    assert content.endswith("answer")


def test_isolation_root_is_not_real_sadb():
    assert os.environ["SADB_DATA_DIR"] == str(ISOLATED_SADB_ROOT)
    assert REAL_SADB_ROOT not in ISOLATED_SADB_ROOT.parents


# ---------------------------------------------------------------------------
# WOR-362: an enabled flag or stored key is not spending authority.
# Metered API routes are refused unconditionally; no paid-provider exception.
# ---------------------------------------------------------------------------

METERED = ["openrouter", "openai", "anthropic", "google", "mistral", "deepseek", "groq", "custom"]


def _everything_enabled():
    return SimpleNamespace(
        enabled_providers={
            "openrouter": True,
            "ollama": True,
            "groq": True,
            "direct": True,
            "custom": True,
            "agentcli": True,
        },
        direct_provider_toggles=dict.fromkeys(("openai", "anthropic", "google", "mistral", "deepseek", "groq"), True),
        openrouter_api_key="stored-key",
        openai_api_key="stored-key",
        anthropic_api_key="stored-key",
    )


@pytest.mark.parametrize("provider", METERED)
def test_metered_route_refused_even_when_enabled_and_keyed(provider):
    reason = council.check_model_route(f"{provider}:some-model", _everything_enabled())
    assert reason and "metered" in reason


def test_non_metered_enabled_routes_still_allowed():
    s = _everything_enabled()
    assert council.check_model_route("ollama:llama3.2", s) is None
    assert council.check_model_route("agentcli:claude", s) is None


def test_every_registered_provider_is_classified():
    assert set(council.PROVIDERS) == set(METERED) | {"ollama", "agentcli"}
    assert council.METERED_API_PROVIDERS == frozenset(METERED)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", METERED)
async def test_query_model_never_dispatches_metered_when_enabled(monkeypatch, provider):
    spies = {name: AsyncMock(return_value={"content": "paid inference"}) for name in council.PROVIDERS}
    for name, p in council.PROVIDERS.items():
        monkeypatch.setattr(p, "query", spies[name])
    monkeypatch.setattr(council, "get_settings", _everything_enabled)
    result = await council.query_model(f"{provider}:m", [{"role": "user", "content": "q"}])
    assert result["error"] is True and "metered" in result["error_message"]
    for spy in spies.values():
        spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_openrouter_layers_refuse_even_when_enabled_with_key(monkeypatch):
    from backend import openrouter as openrouter_module

    post = AsyncMock(side_effect=AssertionError("no HTTP request may be made"))
    monkeypatch.setattr("httpx.AsyncClient.post", post)
    enabled = SimpleNamespace(enabled_providers={"openrouter": True}, openrouter_api_key="stored-key")
    monkeypatch.setattr("backend.providers.openrouter.get_settings", lambda: enabled)
    monkeypatch.setattr(openrouter_module, "get_openrouter_api_key", lambda: "stored-key")
    msgs = [{"role": "user", "content": "q"}]
    provider_result = await OpenRouterProvider().query("openrouter:x", msgs)
    module_result = await openrouter_module.query_model("x", msgs)
    assert provider_result["error"] is True and "metered" in provider_result["error_message"]
    assert module_result["error"] is True and "metered" in module_result["error_message"]
    post.assert_not_awaited()


@pytest.mark.parametrize(
    ("models", "chair"),
    [(["openai:gpt-4o", "ollama:a"], "ollama:c"), (["ollama:a", "ollama:b"], "anthropic:claude")],
)
def test_preflight_refuses_explicit_metered_seats_and_chair(models, chair):
    problems = council.preflight_roundtable_models(models, chair, chair, _everything_enabled())
    assert problems and all("metered" in p for p in problems)


@pytest.mark.asyncio
async def test_anthropic_key_check_makes_no_inference_request(monkeypatch):
    from backend.providers.anthropic import AnthropicProvider

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": []}

    post = AsyncMock(side_effect=AssertionError("validate_key must not POST /messages (paid inference)"))
    get = AsyncMock(return_value=FakeResponse())
    monkeypatch.setattr("httpx.AsyncClient.post", post)
    monkeypatch.setattr("httpx.AsyncClient.get", get)
    result = await AnthropicProvider().validate_key("stored-key")
    post.assert_not_awaited()
    get.assert_awaited_once()
    assert get.await_args.args[0].endswith("/models")
    assert result["success"] is True
