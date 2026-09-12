"""Mocked subprocess tests for AgentCLIProvider (WOR-397)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.council import PROVIDERS, get_provider_for_model
from backend.providers.agent_cli import AgentCLIProvider, _flatten_messages


def _enabled_settings(binary: str | None = None, enabled: bool = True):
    return SimpleNamespace(
        enabled_providers={"agentcli": enabled},
        agentcli_binary_path=binary,
    )


class _FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = None if hang else returncode
        self._final_code = returncode
        self.killed = False
        self.stdin_payload: bytes | None = None

    async def communicate(self, input: bytes | None = None):
        self.stdin_payload = input
        if self.returncode is None:
            await asyncio.sleep(60)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True
        self.returncode = self._final_code if self._final_code is not None else -9

    async def wait(self):
        return self.returncode


def _result_json(text: str, is_error: bool = False) -> bytes:
    payload = {"type": "result", "is_error": is_error, "result": text}
    return json.dumps(payload).encode("utf-8")


@pytest.fixture(autouse=True)
def guarded_launcher(tmp_path, monkeypatch):
    launch_dir = tmp_path / "scripts" / "agent_launch"
    launch_dir.mkdir(parents=True)
    for name in ("claude-subscription", "grok-subscription", "codex-subscription"):
        (launch_dir / name).write_text("# synthetic launcher; subprocess is mocked\n")
    monkeypatch.setenv("C010_ROOT", str(tmp_path))
    return launch_dir / "claude-subscription"


@pytest.fixture
def provider():
    return AgentCLIProvider()


@pytest.mark.asyncio
async def test_success_parses_json_envelope(provider):
    proc = _FakeProcess(stdout=_result_json("council ok"), returncode=0)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
    ):
        result = await provider.query(
            "agentcli:claude",
            [
                {"role": "system", "content": "You are Builder."},
                {"role": "user", "content": "Propose a plan."},
            ],
            timeout=5.0,
            temperature=0.9,
        )

    assert result == {"content": "council ok", "error": False}
    argv = spawn.await_args.args
    assert Path(argv[0]).name == "claude-subscription"
    assert "--safe-mode" in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--disallowedTools" in argv
    assert "-p" in argv
    kwargs = spawn.await_args.kwargs
    assert kwargs["stdin"] is asyncio.subprocess.PIPE
    assert kwargs["cwd"]
    assert proc.stdin_payload == b"You are Builder.\n\nPropose a plan."
    # CLIs do not accept temperature — it must not appear in argv.
    assert "0.9" not in argv
    assert "--temperature" not in argv


@pytest.mark.asyncio
async def test_timeout_kills_process(provider):
    proc = _FakeProcess(hang=True)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
    ):
        result = await provider.query("agentcli:claude", [{"role": "user", "content": "hi"}], timeout=0.05)

    assert result["error"] is True
    assert "timed out" in result["error_message"]
    assert proc.killed is True


@pytest.mark.asyncio
async def test_cli_missing(provider):
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch(
            "backend.providers.agent_cli.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError("missing")),
        ),
    ):
        result = await provider.query("agentcli:claude", [{"role": "user", "content": "hi"}])

    assert result["error"] is True
    assert "not found" in result["error_message"]


@pytest.mark.asyncio
async def test_auth_failure(provider):
    proc = _FakeProcess(stdout=b"", stderr=b"Error: Not logged in. Please run /login", returncode=1)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
    ):
        result = await provider.query("agentcli:claude", [{"role": "user", "content": "hi"}])

    assert result["error"] is True
    assert "auth failure" in result["error_message"]


@pytest.mark.asyncio
async def test_json_parse_failure(provider):
    proc = _FakeProcess(stdout=b"not-json from claude", returncode=0)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
    ):
        result = await provider.query("agentcli:claude", [{"role": "user", "content": "hi"}])

    assert result["error"] is True
    assert "JSON parse failure" in result["error_message"]


@pytest.mark.asyncio
async def test_disabled_short_circuits_without_spawn(provider):
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings(enabled=False)),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec") as spawn,
    ):
        result = await provider.query("agentcli:claude", [{"role": "user", "content": "hi"}])

    assert result["error"] is True
    assert "disabled" in result["error_message"]
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_get_models_static_list(provider):
    with patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()):
        models = await provider.get_models()
    assert models[0]["id"] == "agentcli:claude"
    assert models[0]["provider"] == "AgentCLI"
    ids = [m["id"] for m in models]
    assert ids == ["agentcli:claude", "agentcli:grok", "agentcli:codex"]


def test_prefix_routes_to_agentcli():
    provider = get_provider_for_model("agentcli:claude")
    assert provider is PROVIDERS["agentcli"]
    assert isinstance(provider, AgentCLIProvider)


def test_flatten_system_and_user():
    prompt = _flatten_messages(
        [
            {"role": "system", "content": "You are Skeptic."},
            {"role": "user", "content": "Critique this."},
        ]
    )
    assert prompt == "You are Skeptic.\n\nCritique this."


@pytest.mark.asyncio
async def test_validate_key_reports_missing_binary(provider, guarded_launcher):
    guarded_launcher.unlink()
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.shutil.which", return_value=None),
        patch("backend.providers.agent_cli.os.path.isfile", return_value=False),
    ):
        result = await provider.validate_key("")

    assert result["success"] is False
    assert "not found" in result["message"]


@pytest.mark.asyncio
async def test_envelope_is_error_true(provider):
    proc = _FakeProcess(stdout=_result_json("rate limited", is_error=True), returncode=0)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
    ):
        result = await provider.query("agentcli:claude", [{"role": "user", "content": "hi"}])

    assert result["error"] is True
    assert "rate limited" in result["error_message"]


def test_api_models_appends_agentcli_when_enabled():
    from fastapi.testclient import TestClient

    from backend.main import app

    settings = SimpleNamespace(
        enabled_providers={"agentcli": True, "openrouter": True},
        agentcli_binary_path=None,
    )
    with (
        patch("backend.openrouter.fetch_models", AsyncMock(return_value=[{"id": "openai/gpt-4o", "name": "GPT-4o"}])),
        patch("backend.main.get_settings", return_value=settings),
        patch("backend.providers.agent_cli.get_settings", return_value=settings),
    ):
        response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["models"]]
    assert "openai/gpt-4o" in ids
    assert "agentcli:claude" in ids
    assert "agentcli:grok" in ids
    assert "agentcli:codex" in ids


def test_api_models_omits_agentcli_when_disabled():
    from fastapi.testclient import TestClient

    from backend.main import app

    settings = SimpleNamespace(
        enabled_providers={"agentcli": False, "openrouter": True},
        agentcli_binary_path=None,
    )
    with (
        patch("backend.openrouter.fetch_models", AsyncMock(return_value=[{"id": "openai/gpt-4o", "name": "GPT-4o"}])),
        patch("backend.main.get_settings", return_value=settings),
        patch("backend.providers.agent_cli.get_settings", return_value=settings),
    ):
        response = TestClient(app).get("/api/models")

    ids = [m["id"] for m in response.json()["models"]]
    assert "agentcli:claude" not in ids


def test_test_provider_allows_empty_key_for_agentcli():
    from fastapi.testclient import TestClient

    from backend.main import app

    with patch(
        "backend.providers.agent_cli.AgentCLIProvider.validate_key",
        AsyncMock(return_value={"success": False, "message": "agentcli binary not found: claude"}),
    ):
        response = TestClient(app).post(
            "/api/settings/test-provider",
            json={"provider_id": "agentcli", "api_key": ""},
        )

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert "not found" in response.json()["message"]


@pytest.mark.asyncio
async def test_raw_binary_override_refused_before_spawn(provider):
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings("claude")),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec") as spawn,
    ):
        result = await provider.query("agentcli:claude", [{"role": "user", "content": "hi"}])
    assert result["error"] is True
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_validate_key_does_not_infer(provider):
    proc = _FakeProcess(
        stdout=json.dumps({"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"}).encode()
    )
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
    ):
        result = await provider.validate_key("")
    assert result["success"] is True
    assert spawn.await_args.args[-2:] == ("auth", "status")
    assert proc.stdin_payload is None


def test_successful_authentication_discussion_is_content():
    from backend.providers.agent_cli import _interpret_cli_result

    assert _interpret_cli_result(0, _result_json("Use authentication for this endpoint.").decode(), "") == {
        "content": "Use authentication for this endpoint.",
        "error": False,
    }


@pytest.mark.asyncio
async def test_disabled_provider_advertises_no_models(provider):
    with patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings(enabled=False)):
        assert await provider.get_models() == []


@pytest.mark.asyncio
async def test_cleanup_signals_group_after_launcher_exits():
    from backend.providers.agent_cli import _kill

    proc = _FakeProcess(returncode=0)
    proc.pid = 987654
    with patch("backend.providers.agent_cli.os.killpg") as killpg:
        await _kill(proc)
    killpg.assert_called_once()


@pytest.mark.parametrize(
    "text",
    [
        'launcher notice\n{"type":"result","is_error":false,"result":"council ok","usage":{"input_tokens":1}}',
        '[{"type":"system","subtype":"init"},{"type":"result","is_error":false,"result":"council ok","usage":{"input_tokens":1}}]',
    ],
)
def test_live_cli_envelope_shapes(text):
    from backend.providers.agent_cli import _interpret_cli_result

    assert _interpret_cli_result(0, text, "") == {"content": "council ok", "error": False}


def test_grok_json_uses_text_field():
    from backend.providers.agent_cli import _interpret_grok_result

    payload = json.dumps({"text": "grok ok", "sessionId": "abc", "stopReason": "end_turn"})
    assert _interpret_grok_result(0, payload, "") == {"content": "grok ok", "error": False}


def test_grok_error_object():
    from backend.providers.agent_cli import _interpret_grok_result

    payload = json.dumps({"type": "error", "message": "Couldn't start session"})
    result = _interpret_grok_result(1, payload, "")
    assert result["error"] is True
    assert "Couldn't start session" in result["error_message"]


def test_codex_item_completed_stream():
    from backend.providers.agent_cli import _interpret_codex_stream

    stream = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "codex ok"},
                }
            ),
            json.dumps({"type": "turn.completed"}),
        ]
    )
    assert _interpret_codex_stream(0, stream, "") == {"content": "codex ok", "error": False}


def test_codex_malformed_stream():
    from backend.providers.agent_cli import _interpret_codex_stream

    result = _interpret_codex_stream(0, "not-json\nstill-not-json", "")
    assert result["error"] is True
    assert "JSON parse failure" in result["error_message"]


@pytest.mark.asyncio
async def test_grok_query_uses_hermetic_prompt_file(provider):
    proc = _FakeProcess(stdout=json.dumps({"text": "isolated ok"}).encode(), returncode=0)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
    ):
        result = await provider.query("agentcli:grok", [{"role": "user", "content": "hello"}])
    assert result == {"content": "isolated ok", "error": False}
    argv = list(spawn.await_args.args)
    assert Path(argv[0]).name == "grok-subscription"
    assert "--hermetic" in argv
    assert argv[argv.index("--model") + 1] == "grok-4.6"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--prompt-file" in argv
    assert "--no-subagents" in argv
    assert "--no-memory" in argv
    assert "--disable-web-search" in argv
    assert spawn.await_args.kwargs["stdin"] is asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_codex_query_uses_exec_json_and_stdin(provider):
    stream = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "codex seat"}})
    proc = _FakeProcess(stdout=stream.encode(), returncode=0)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
    ):
        result = await provider.query("agentcli:codex", [{"role": "user", "content": "hello"}])
    assert result == {"content": "codex seat", "error": False}
    argv = list(spawn.await_args.args)
    assert Path(argv[0]).name == "codex-subscription"
    assert argv[1:4] == ["exec", "--json", "--sandbox"]
    assert "read-only" in argv
    assert "--ephemeral" in argv
    assert "--skip-git-repo-check" in argv
    assert argv[-1] == "-"
    assert spawn.await_args.kwargs["stdin"] is asyncio.subprocess.PIPE
    assert proc.stdin_payload == b"hello"


@pytest.mark.asyncio
async def test_unverified_codex_model_suffix_refused(provider):
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec") as spawn,
    ):
        result = await provider.query("agentcli:codex:o3", [{"role": "user", "content": "hi"}])
    assert result["error"] is True
    assert "Unsupported" in result["error_message"]
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_grok_timeout_cancels(provider):
    proc = _FakeProcess(hang=True)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
    ):
        result = await provider.query("agentcli:grok", [{"role": "user", "content": "hi"}], timeout=0.05)
    assert result["error"] is True
    assert "timed out" in result["error_message"]
    assert proc.killed is True


@pytest.mark.asyncio
async def test_codex_timeout_cancels(provider):
    proc = _FakeProcess(hang=True)
    with (
        patch("backend.providers.agent_cli.get_settings", return_value=_enabled_settings()),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
    ):
        result = await provider.query("agentcli:codex", [{"role": "user", "content": "hi"}], timeout=0.05)
    assert result["error"] is True
    assert "timed out" in result["error_message"]
    assert proc.killed is True


@pytest.mark.asyncio
async def test_validate_grok_models_banner():
    from backend.providers.agent_cli import _validate_grok

    proc = _FakeProcess(stdout=b"You are logged in with grok.com.\nDefault model: grok-4.6\n")
    with patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        result = await _validate_grok("/tmp/grok-subscription")
    assert result["success"] is True
    assert spawn.await_args.args[-2:] == ("--hermetic", "models")


@pytest.mark.asyncio
async def test_validate_codex_login_status():
    from backend.providers.agent_cli import _validate_codex

    proc = _FakeProcess(stdout=b"Logged in using ChatGPT\n")
    with patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        result = await _validate_codex("/tmp/codex-subscription")
    assert result["success"] is True
    assert spawn.await_args.args[-2:] == ("login", "status")


@pytest.mark.asyncio
async def test_validate_grok_rejects_not_logged_in_banner():
    from backend.providers.agent_cli import _validate_grok

    proc = _FakeProcess(stdout=b"Not logged in with grok.com.\n", returncode=0)
    with patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        result = await _validate_grok("/tmp/grok-subscription")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_validate_codex_rejects_not_logged_in_banner():
    from backend.providers.agent_cli import _validate_codex

    proc = _FakeProcess(stdout=b"Not logged in using ChatGPT\n", returncode=0)
    with patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        result = await _validate_codex("/tmp/codex-subscription")
    assert result["success"] is False


def test_codex_nested_content_array():
    from backend.providers.agent_cli import _interpret_codex_stream

    stream = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "content": [{"type": "text", "text": "nested ok"}],
            },
        }
    )
    assert _interpret_codex_stream(0, stream, "") == {"content": "nested ok", "error": False}


def test_codex_last_wins_multiple_agent_messages():
    from backend.providers.agent_cli import _interpret_codex_stream

    stream = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "last"}}),
        ]
    )
    assert _interpret_codex_stream(0, stream, "") == {"content": "last", "error": False}


def test_codex_nonzero_returncode_with_partial_text():
    from backend.providers.agent_cli import _interpret_codex_stream

    stream = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}})
    result = _interpret_codex_stream(1, stream, "boom")
    assert result["error"] is True
    assert "exited 1" in result["error_message"]


def test_grok_json_after_banner_prefix():
    from backend.providers.agent_cli import _interpret_grok_result

    payload = "You are logged in with grok.com.\n" + json.dumps({"text": "after banner"})
    assert _interpret_grok_result(0, payload, "") == {"content": "after banner", "error": False}


@pytest.mark.asyncio
async def test_claude_binary_pin_does_not_block_grok(provider, guarded_launcher):
    proc = _FakeProcess(stdout=json.dumps({"text": "grok despite claude pin"}).encode())
    with (
        patch(
            "backend.providers.agent_cli.get_settings",
            return_value=_enabled_settings(str(guarded_launcher)),
        ),
        patch("backend.providers.agent_cli.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
    ):
        result = await provider.query("agentcli:grok", [{"role": "user", "content": "hi"}])
    assert result["error"] is False
    assert Path(spawn.await_args.args[0]).name == "grok-subscription"


@pytest.mark.asyncio
async def test_budget_zero_fails_fast(mock_query_model, mock_settings, tmp_path, monkeypatch):
    from backend.roundtable import AgentConfig, run_roundtable

    monkeypatch.setenv("SADB_DATA_DIR", str(tmp_path / "sadb"))
    mock_settings.roundtable_max_calls_per_run = 0
    agents = [
        AgentConfig(model="mock:builder", role="builder", label="Builder"),
        AgentConfig(model="mock:skeptic", role="skeptic", label="Skeptic"),
    ]
    with (
        patch("backend.roundtable.query_model", mock_query_model),
        patch("backend.roundtable.get_settings", return_value=mock_settings),
    ):
        types = []
        async for event in run_roundtable(
            conversation_id="budget-0",
            question="q",
            agents=agents,
            moderator_model="mock:moderator",
            chair_model="mock:chair",
            num_rounds=1,
        ):
            types.append(event["type"])
    assert "roundtable_budget_exceeded" in types
    assert mock_query_model.await_count == 0
