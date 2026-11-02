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
async def test_openrouter_provider_refuses_even_with_stored_key(monkeypatch):
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
        with pytest.raises(openai_compat.RoundtableFailedError) as exc:
            await openai_compat.generate_non_streaming_response(request)
        assert exc.value.status_code == 400
        text = exc.value.message
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
    assert seen["num_rounds"] == 5  # the engine receives the request and records both counts
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
    monkeypatch.setattr(openai_compat, "get_settings", _v1_settings)  # routes pass preflight
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
    send = AsyncMock(side_effect=AssertionError("no HTTP request may be made"))
    monkeypatch.setattr("httpx.AsyncClient.post", post)
    monkeypatch.setattr("httpx.AsyncClient.send", send)
    enabled = SimpleNamespace(enabled_providers={"openrouter": True}, openrouter_api_key="stored-key")
    monkeypatch.setattr("backend.providers.openrouter.get_settings", lambda: enabled)
    monkeypatch.setattr(openrouter_module, "get_openrouter_api_key", lambda: "stored-key")
    msgs = [{"role": "user", "content": "q"}]
    provider_result = await OpenRouterProvider().query("openrouter:x", msgs)
    module_result = await openrouter_module.query_model("x", msgs)
    assert provider_result["error"] is True and "metered" in provider_result["error_message"]
    assert module_result["error"] is True and "metered" in module_result["error_message"]
    post.assert_not_awaited()
    send.assert_not_awaited()


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


# ---------------------------------------------------------------------------
# Round-1 review repairs (R1-G1..G20).
# ---------------------------------------------------------------------------


def test_isolation_is_installed_at_conftest_import():
    from tests import conftest

    # Installed at conftest import, before any backend module was loaded.
    assert conftest.BACKEND_LOADED_AT_GUARD_TIME == []
    snap = conftest.GUARDS_AT_IMPORT
    assert snap["socket.connect"] is conftest._guarded_connect
    assert snap["socket.connect_ex"] is conftest._guarded_connect_ex
    assert snap["socket.sendto"] is conftest._guarded_sendto
    if snap["socket.sendmsg"] is not None:
        assert snap["socket.sendmsg"] is conftest._guarded_sendmsg
    for name in (
        "create_connection",
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        assert snap[name] is conftest._blocked, name
        assert getattr(socket, name) is conftest._blocked, name
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(NetworkBlockedError):
        socket.gethostbyname("example.com")
    for name in conftest.SCRUBBED_KEY_ENV_VARS:
        assert name not in os.environ, name


def test_inet_sockets_cannot_send():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        with pytest.raises(NetworkBlockedError):
            udp.sendto(b"x", ("127.0.0.1", 9))
        with pytest.raises(NetworkBlockedError):
            udp.connect_ex(("127.0.0.1", 9))


@pytest.mark.asyncio
async def test_enabled_non_metered_route_dispatches_exactly_once(monkeypatch):
    spy = AsyncMock(return_value={"content": "local answer"})
    monkeypatch.setattr(council.PROVIDERS["ollama"], "query", spy)
    monkeypatch.setattr(council, "get_settings", lambda: _settings(ollama=True))
    result = await council.query_model(" ollama:llama3.2 ", [{"role": "user", "content": "q"}])
    assert result == {"content": "local answer"}
    spy.assert_awaited_once()
    assert spy.await_args.args[0] == "ollama:llama3.2"


@pytest.mark.parametrize("model", ["ollama:", "ollama:   ", "agentcli:"])
def test_known_prefix_without_model_name_is_refused(model):
    reason = council.check_model_route(model, _settings(ollama=True, agentcli=True))
    assert reason and "no model name" in reason


@pytest.mark.asyncio
async def test_moderator_and_chair_exceptions_keep_ledger_reasons():
    def raising(model):
        return {"content": "ok"}

    async def fake(model, messages, timeout=120.0, temperature=0.5):
        if model == "mock:moderator":
            raise RuntimeError("")
        if model == "mock:chair":
            raise ValueError("chair boom")
        return raising(model)

    run = (await _run(fake, _agents(2), num_rounds=1))[-1]["run"]
    attempts = {a["model"]: a for a in run["call_accounting"]["attempts"]}
    assert attempts["mock:moderator"]["failed"] and attempts["mock:moderator"]["error_message"] == "RuntimeError"
    assert attempts["mock:chair"]["error_message"] == "chair boom"
    assert run["status"] == "failed" and run["chair_final"]["error_message"] == "chair boom"


@pytest.mark.asyncio
async def test_cancelled_run_records_reason_and_cancelled_attempts():
    import asyncio

    started = asyncio.Event()

    async def slow(model, messages, timeout=120.0, temperature=0.5):
        started.set()
        await asyncio.sleep(30)
        return {"content": "never"}

    events = []

    async def drive():
        with (
            patch("backend.roundtable.query_model", slow),
            patch("backend.roundtable.get_settings", return_value=_rt_settings()),
        ):
            async for e in run_roundtable(
                conversation_id="c",
                question="q",
                agents=_agents(2),
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                num_rounds=1,
            ):
                events.append(e)

    task = asyncio.create_task(drive())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run = events[-1]["run"]
    assert events[-1]["type"] == "roundtable_aborted"
    assert run["status"] == "aborted" and run["failure_reason"]
    assert run["executed_rounds"] == 0
    assert all(a["error_message"] == "cancelled" for a in run["call_accounting"]["attempts"])


@pytest.mark.asyncio
async def test_executed_rounds_reflect_what_ran():
    fail, _ = _scripted(lambda m: {"error": True, "error_message": "down"})
    assert (await _run(fail, _agents(2)))[-1]["run"]["executed_rounds"] == 1
    ok, _ = _scripted(lambda m: {"content": "ok"})
    budget = (await _run(ok, _agents(3), num_rounds=3, budget=5))[-1]["run"]
    assert budget["executed_rounds"] == 0 and budget["planned_rounds"] == 3
    deep = (await _run(ok, _agents(2), num_rounds=5))[-1]["run"]
    assert (deep["requested_rounds"], deep["planned_rounds"], deep["executed_rounds"]) == (5, 3, 3)


@pytest.mark.asyncio
async def test_whitespace_seat_content_is_a_failure():
    fake, _ = _scripted(lambda m: {"content": "  \n "} if m == "mock:skeptic" else {"content": "ok"})
    run = (await _run(fake, _agents(2), num_rounds=1))[-1]["run"]
    assert run["status"] == "degraded" and run["failed_seats"] == ["Skeptic"]


@pytest.mark.parametrize(
    "chair",
    [{"content": "", "error": False}, {"content": "   ", "error": False}, {"content": "x", "error": True}],
)
def test_capture_refuses_completed_run_without_chair_answer(chair, monkeypatch):
    append = Mock()
    monkeypatch.setattr(turn_capture, "append_turns", append)
    run = _run_dict("completed")
    run["chair_final"] = {"model": "m", **chair}
    assert turn_capture.capture_run(run, session_id="no-chair") is None
    append.assert_not_called()


def test_captured_turns_carry_failure_reason():
    out = turn_capture.capture_run(_run_dict("degraded"), session_id="reason-check")
    meta = [json.loads(line)["meta"] for line in Path(out).read_text().splitlines()]
    assert {m["failure_reason"] for m in meta} == {"degraded reason"}


async def _route(monkeypatch, settings_obj, fake=None, stream=False):
    monkeypatch.setattr(openai_compat, "get_settings", lambda: settings_obj)
    if fake is not None:
        monkeypatch.setattr(openai_compat, "run_roundtable", fake)
    return await main.openai_chat_completions(_v1_request(stream), SimpleNamespace())


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_v1_route_preflight_refusal_is_http_400(monkeypatch, stream):
    bad = SimpleNamespace(**{**vars(_v1_settings()), "council_models": ["ollama:a", "openai:gpt-4o"]})
    roundtable = Mock(side_effect=AssertionError("must not start"))
    response = await _route(monkeypatch, bad, roundtable, stream=stream)
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"]["type"] == "roundtable_refused" and "metered" in body["error"]["message"]
    roundtable.assert_not_called()


def _terminal(event):
    async def fake(**kwargs):
        yield {"type": "roundtable_init", "total_rounds": 1, "council_members": [{}, {}]}
        yield event

    return fake


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "status", "etype"),
    [
        (
            {"type": "roundtable_failed", "run": {**_run_dict("failed"), "failure_reason": "no seat"}},
            502,
            "roundtable_failed",
        ),
        (
            {"type": "roundtable_budget_exceeded", "predicted_calls": 22, "max_calls_per_run": 14, "run": {}},
            429,
            "roundtable_budget_exceeded",
        ),
        ({"type": "roundtable_aborted", "run": _run_dict("aborted")}, 502, "roundtable_aborted"),
    ],
)
async def test_v1_route_non_success_terminals_are_http_errors(monkeypatch, event, status, etype):
    response = await _route(monkeypatch, _v1_settings(), _terminal(event))
    assert response.status_code == status
    body = json.loads(response.body)
    assert body["error"]["type"] == etype and "choices" not in body


@pytest.mark.asyncio
async def test_v1_route_engine_exception_is_http_500(monkeypatch):
    async def exploding(**kwargs):
        raise ValueError("roundtable requires at least one round")
        yield  # pragma: no cover

    response = await _route(monkeypatch, _v1_settings(), exploding)
    assert response.status_code == 500
    assert "at least one round" in json.loads(response.body)["error"]["message"]


@pytest.mark.asyncio
async def test_v1_route_completed_council_is_http_200(monkeypatch):
    run = _run_dict("completed")
    response = await _route(
        monkeypatch,
        _v1_settings(),
        _terminal({"type": "chair_complete", "chair_final": run["chair_final"], "run": run}),
    )
    assert response.choices[0].message.content == "answer"


@pytest.mark.asyncio
async def test_ui_stream_terminal_data_carries_reasons(monkeypatch):
    for terminal, status in (("roundtable_aborted", "aborted"), ("roundtable_budget_exceeded", "budget_exceeded")):
        run = _run_dict(status)

        async def fake(_terminal=terminal, _run=run, **kwargs):
            yield {"type": _terminal, "run": _run, "predicted_calls": 22}

        events, mocks, _ = await _collect_main_stream(monkeypatch, _ui_settings(), fake)
        err = [e for e in events if e["type"] == "roundtable_error"][-1]
        assert err["data"]["status"] == status
        assert err["data"]["failure_reason"] == f"{status} reason"
        assert mocks["add_roundtable_message"].call_args.kwargs["chair_final"] == run["chair_final"]


@pytest.mark.asyncio
async def test_ui_stream_stores_empty_dict_when_chair_final_null(monkeypatch):
    run = _run_dict("failed")
    run["chair_final"] = None

    async def fake(**kwargs):
        yield {"type": "roundtable_failed", "run": run}

    _, mocks, _ = await _collect_main_stream(monkeypatch, _ui_settings(), fake)
    assert mocks["add_roundtable_message"].call_args.kwargs["chair_final"] == {}


# ---------------------------------------------------------------------------
# Round-2 review repairs (R2-F2..F8).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["mock:moderator", "mock:chair"])
async def test_cancel_during_moderator_or_chair_records_cancelled(model):
    import asyncio

    started = asyncio.Event()

    async def fake(m, messages, timeout=120.0, temperature=0.5):
        if m == model:
            started.set()
            await asyncio.sleep(30)
        return {"content": "ok"}

    events = []

    async def drive():
        with (
            patch("backend.roundtable.query_model", fake),
            patch("backend.roundtable.get_settings", return_value=_rt_settings()),
        ):
            async for e in run_roundtable(
                conversation_id="c",
                question="q",
                agents=_agents(2),
                moderator_model="mock:moderator",
                chair_model="mock:chair",
                num_rounds=1,
            ):
                events.append(e)

    task = asyncio.create_task(drive())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run = events[-1]["run"]
    assert events[-1]["type"] == "roundtable_aborted" and run["status"] == "aborted"
    assert run["failure_reason"]
    attempt = [a for a in run["call_accounting"]["attempts"] if a["model"] == model][-1]
    assert attempt["failed"] and attempt["error_message"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "degraded"])
@pytest.mark.parametrize("content", ["   ", "\n\t", None])
async def test_v1_blank_chair_answer_is_never_success(monkeypatch, content, status):
    run = _run_dict(status)
    run["chair_final"] = {"model": "ollama:c", "content": content, "error": False}
    fake = _terminal({"type": "chair_complete", "chair_final": run["chair_final"], "run": run})
    response = await _route(monkeypatch, _v1_settings(), fake)
    assert response.status_code == 502
    error = json.loads(response.body)["error"]
    assert "choices" not in json.loads(response.body)
    assert error["type"] == "roundtable_failed" and error["code"] == "failed"
    assert error["failure_reason"]
    monkeypatch.setattr(openai_compat, "run_roundtable", fake)
    text = "".join([c async for c in openai_compat.generate_openai_stream(_v1_request(True))])
    expected = "the chair returned no answer" if status == "completed" else "degraded reason"
    assert f"Roundtable failed: {expected}" in text and error["failure_reason"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_v1_route_request_shape_refusals_are_http_400(monkeypatch, stream):
    roundtable = Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(openai_compat, "run_roundtable", roundtable)
    no_user = openai_compat.ChatCompletionRequest(
        model="roundtable", messages=[openai_compat.ChatMessage(role="system", content="s")], stream=stream
    )
    monkeypatch.setattr(openai_compat, "get_settings", _v1_settings)
    response = await main.openai_chat_completions(no_user, SimpleNamespace())
    assert response.status_code == 400 and "no user message" in json.loads(response.body)["error"]["message"]

    one_seat = SimpleNamespace(**{**vars(_v1_settings()), "council_models": ["ollama:a"]})
    monkeypatch.setattr(openai_compat, "get_settings", lambda: one_seat)
    response = await main.openai_chat_completions(_v1_request(stream), SimpleNamespace())
    assert response.status_code == 400 and "at least 2 council models" in json.loads(response.body)["error"]["message"]
    roundtable.assert_not_called()


@pytest.mark.asyncio
async def test_v1_generator_preflight_raises_for_direct_callers(monkeypatch):
    bad = SimpleNamespace(**{**vars(_v1_settings()), "council_models": ["ollama:a", "openai:gpt-4o"]})
    monkeypatch.setattr(openai_compat, "get_settings", lambda: bad)
    monkeypatch.setattr(openai_compat, "run_roundtable", Mock(side_effect=AssertionError("must not start")))
    with pytest.raises(openai_compat.RoundtableFailedError) as exc:
        await openai_compat.generate_non_streaming_response(_v1_request(False))
    assert exc.value.status_code == 400 and exc.value.error_type == "roundtable_refused"


@pytest.mark.asyncio
async def test_v1_no_terminal_event_is_an_error(monkeypatch):
    async def silent(**kwargs):
        yield {"type": "roundtable_init", "total_rounds": 1, "council_members": [{}, {}]}

    response = await _route(monkeypatch, _v1_settings(), silent)
    assert response.status_code == 502
    assert json.loads(response.body)["error"]["type"] == "roundtable_incomplete"
    monkeypatch.setattr(openai_compat, "run_roundtable", silent)
    text = "".join([c async for c in openai_compat.generate_openai_stream(_v1_request(True))])
    assert "Roundtable failed: the roundtable ended without an answer" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "reason"),
    [
        (
            {
                "type": "roundtable_budget_exceeded",
                "predicted_calls": 22,
                "max_calls_per_run": 14,
                "run": {"status": "budget_exceeded", "failure_reason": "predicted 22 calls exceed cap"},
            },
            "predicted 22 calls exceed cap",
        ),
        (
            {"type": "roundtable_budget_exceeded", "predicted_calls": 22, "max_calls_per_run": 14},
            "roundtable_max_calls_per_run=14",
        ),
        ({"type": "roundtable_aborted", "run": _run_dict("aborted")}, "aborted reason"),
        ({"type": "roundtable_failed", "run": _run_dict("failed")}, "failed reason"),
    ],
)
async def test_v1_error_bodies_carry_failure_reason(monkeypatch, event, reason):
    response = await _route(monkeypatch, _v1_settings(), _terminal(event))
    assert reason in json.loads(response.body)["error"]["failure_reason"]


@pytest.mark.parametrize("provider", sorted(council.METERED_API_PROVIDERS))
def test_get_provider_for_model_refuses_metered(provider):
    with pytest.raises(council.ModelRouteError, match="metered"):
        council.get_provider_for_model(f"{provider}:some-model")


def test_provider_query_is_referenced_from_one_place_only():
    """Static guard: the only `.query` reference in backend/ is inside council.query_model.

    Counts every attribute reference named `query` (calls, aliases such as `fn = p.query`,
    module-level code) and every `getattr(..., "query")`, in every backend module.
    """
    import ast

    backend_dir = Path(council.__file__).parent
    refs = []
    for path in sorted(backend_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        owner = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    owner.setdefault(id(node), fn.name)
        for node in ast.walk(tree):
            is_attr = isinstance(node, ast.Attribute) and node.attr == "query"
            is_getattr = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "query"
            )
            if is_attr or is_getattr:
                refs.append(f"{path.relative_to(backend_dir)}:{owner.get(id(node), '<module>')}")
    assert refs == ["council.py:query_model"]


@pytest.mark.parametrize("model", ["ollama:llama3.2", "agentcli:claude"])
def test_get_provider_for_model_still_returns_non_metered(model):
    assert council.get_provider_for_model(model) is council.PROVIDERS[model.split(":")[0]]


@pytest.mark.asyncio
async def test_openrouter_parallel_helper_refuses_without_http(monkeypatch):
    from backend import openrouter as openrouter_module

    send = AsyncMock(side_effect=AssertionError("no HTTP request may be made"))
    monkeypatch.setattr("httpx.AsyncClient.send", send)
    models = [f"m{i}" for i in range(7)]  # >= 6 exercises the batched branch too
    results = await openrouter_module.query_models_parallel(models, [{"role": "user", "content": "q"}])
    assert set(results) == set(models)
    assert all(r["error"] is True and "metered" in r["error_message"] for r in results.values())
    send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_v1_whitespace_user_message_is_refused(monkeypatch, stream):
    monkeypatch.setattr(openai_compat, "run_roundtable", Mock(side_effect=AssertionError("must not start")))
    monkeypatch.setattr(openai_compat, "get_settings", _v1_settings)
    blank = openai_compat.ChatCompletionRequest(
        model="roundtable", messages=[openai_compat.ChatMessage(role="user", content=" \n ")], stream=stream
    )
    response = await main.openai_chat_completions(blank, SimpleNamespace())
    assert response.status_code == 400
    error = json.loads(response.body)["error"]
    assert error["type"] == "roundtable_refused" and "no user message" in error["failure_reason"]


@pytest.mark.asyncio
async def test_v1_streaming_abort_carries_reason(monkeypatch):
    fake = _terminal({"type": "roundtable_aborted", "run": _run_dict("aborted")})
    monkeypatch.setattr(openai_compat, "get_settings", _v1_settings)
    monkeypatch.setattr(openai_compat, "run_roundtable", fake)
    text = "".join([c async for c in openai_compat.generate_openai_stream(_v1_request(True))])
    assert "[Roundtable was aborted: aborted reason]" in text
