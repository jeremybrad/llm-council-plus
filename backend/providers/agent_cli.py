"""Native subscription-CLI provider (WOR-397/398/399).

Seats billed only through C010 guarded launchers — never raw vendor CLIs
and never API keys:

- `agentcli:claude` → `claude-subscription -p --output-format json`
- `agentcli:grok`   → `grok-subscription --hermetic --output-format json`
- `agentcli:codex`  → `codex-subscription exec --json` (installed CLI default model)

Prompt is piped on stdin (Claude/Codex) or `--prompt-file` (Grok; round-3
prompts exceed safe argv length). Tools are disabled and the process runs
in a scratch cwd so a council seat cannot edit the repo. Temperature is
dropped — vendor CLIs do not accept it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any

from ..settings import get_settings
from .base import LLMProvider

# Tools a pure-text deliberation seat must not invoke. Passed as one
# comma-separated --disallowedTools argument (Claude Code yargs form).
_DISALLOWED_TOOLS = (
    "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit,Agent,Skill,Task,TodoWrite,BashOutput,KillShell"
)

_SEAT_LAUNCHERS = {
    "claude": "claude-subscription",
    "grok": "grok-subscription",
    "codex": "codex-subscription",
}

# Verified on this host by `grok-subscription --hermetic models` (no inference).
_GROK_MODEL = "grok-4.6"

_AUTH_MARKERS = (
    "not logged in",
    "please run /login",
    "please login",
    "authentication required",
    "unauthenticated",
    "invalid api key",
    "not authenticated",
    "auth failure",
)


class AgentCLIProvider(LLMProvider):
    """Headless vendor-CLI provider billed to a local subscription seat."""

    async def query(
        self, model_id: str, messages: list[dict[str, str]], timeout: float = 120.0, temperature: float = 0.7
    ) -> dict[str, Any]:
        del temperature  # CLIs do not accept temperature.
        settings = get_settings()
        if not settings.enabled_providers.get("agentcli", False):
            return {"error": True, "error_message": "agentcli provider is disabled"}

        seat = _seat_name(model_id)
        if seat not in _SEAT_LAUNCHERS:
            supported = ", ".join(f"agentcli:{name}" for name in _SEAT_LAUNCHERS)
            return {
                "error": True,
                "error_message": f"Unsupported agentcli seat '{seat}' (supported: {supported})",
            }

        prompt = _flatten_messages(messages)
        if not prompt.strip():
            return {"error": True, "error_message": "Empty prompt"}

        try:
            binary = _resolve_launcher(seat, settings.agentcli_binary_path)
        except ValueError as exc:
            return {"error": True, "error_message": str(exc)}

        if seat == "claude":
            return await _invoke_claude(binary=binary, prompt=prompt, timeout=timeout)
        if seat == "grok":
            return await _invoke_grok(binary=binary, prompt=prompt, timeout=timeout)
        return await _invoke_codex(binary=binary, prompt=prompt, timeout=timeout)

    async def get_models(self) -> list[dict[str, Any]]:
        if not get_settings().enabled_providers.get("agentcli", False):
            return []
        return [
            {"id": "agentcli:claude", "name": "Claude Code [agentcli]", "provider": "AgentCLI", "is_free": True},
            {"id": "agentcli:grok", "name": "Grok Build [agentcli]", "provider": "AgentCLI", "is_free": False},
            {
                "id": "agentcli:codex",
                "name": "Codex CLI default [agentcli]",
                "provider": "AgentCLI",
                "is_free": False,
            },
        ]

    async def validate_key(self, api_key: str) -> dict[str, Any]:
        """Check Claude subscription authentication without inference (WOR-397).

        Grok and Codex seats expose the same check through `_validate_seat`.
        """
        if api_key:
            return {"success": False, "message": "agentcli accepts no API key or binary override"}
        result = await _validate_seat("claude", get_settings().agentcli_binary_path)
        return {"success": bool(result["success"]), "message": str(result["message"])}


def _seat_name(model_id: str) -> str:
    rest = model_id.split(":", 1)[-1] if ":" in model_id else model_id
    # Refuse unverified model suffixes such as agentcli:codex:o3.
    return rest.strip()


def _resolve_binary(configured: str | None) -> str:
    """Claude launcher path. Kept for WOR-397 tests that patch this name."""
    return _resolve_launcher("claude", configured)


def _resolve_launcher(seat: str, configured: str | None) -> str:
    # C010 is the authority; never execute a PATH-selected raw vendor binary.
    name = _SEAT_LAUNCHERS[seat]
    root = Path(
        os.environ.get("C010_ROOT")
        or (Path(os.environ.get("CODELOCAL_ROOT") or Path.home() / "CodeLocal") / "C010_standards")
    )
    launcher = root / "scripts" / "agent_launch" / name
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        # A WOR-397 pin to claude-subscription must not disable grok/codex.
        # An override is honored only when it is exactly this seat's launcher.
        if configured_path != launcher.resolve():
            if seat == "claude":
                raise ValueError("Only the canonical C010 guarded subscription launcher is permitted")
            configured = None
    if not launcher.is_file():
        raise ValueError(f"Canonical guarded agentcli launcher not found: {name}")
    return str(launcher)


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    """Concatenate system + user contents. Roundtable always sends exactly those two."""
    parts: list[str] = []
    for message in messages:
        content = (message.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


async def _invoke_claude(*, binary: str, prompt: str, timeout: float) -> dict[str, Any]:
    argv = [
        binary,
        "-p",
        "--safe-mode",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--no-chrome",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--disallowedTools",
        _DISALLOWED_TOOLS,
        "--max-turns",
        "1",
    ]
    return await _run_cli(argv, prompt=prompt, timeout=timeout, stdin=True, parser="claude")


async def _invoke_grok(*, binary: str, prompt: str, timeout: float) -> dict[str, Any]:
    scratch = tempfile.mkdtemp(prefix="agentcli-grok-")
    prompt_path = Path(scratch) / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    argv = [
        binary,
        "--hermetic",
        "--model",
        _GROK_MODEL,
        "--output-format",
        "json",
        "--prompt-file",
        str(prompt_path),
        "--disable-web-search",
        "--no-subagents",
        "--no-memory",
        "--cwd",
        scratch,
    ]
    try:
        return await _run_cli(argv, prompt=None, timeout=timeout, stdin=False, parser="grok", scratch=scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def _invoke_codex(*, binary: str, prompt: str, timeout: float) -> dict[str, Any]:
    scratch = tempfile.mkdtemp(prefix="agentcli-codex-")
    argv = [
        binary,
        "exec",
        "--json",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "-C",
        scratch,
        "-",
    ]
    try:
        return await _run_cli(argv, prompt=prompt, timeout=timeout, stdin=True, parser="codex", scratch=scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def _run_cli(
    argv: list[str],
    *,
    prompt: str | None,
    timeout: float,
    stdin: bool,
    parser: str,
    scratch: str | None = None,
) -> dict[str, Any]:
    owned_scratch = scratch is None
    scratch_dir = scratch or tempfile.mkdtemp(prefix="agentcli-")
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=scratch_dir,
                env=os.environ.copy(),
                start_new_session=True,
            )
        except FileNotFoundError:
            return {"error": True, "error_message": f"agentcli binary not found: {argv[0]}"}

        payload = prompt.encode("utf-8") if stdin and prompt is not None else None
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(payload), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            await _kill(proc)
            return {"error": True, "error_message": f"agentcli timed out after {timeout}s"}
        except asyncio.CancelledError:
            await _kill(proc)
            raise

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if parser == "codex":
            return _interpret_codex_stream(proc.returncode or 0, stdout, stderr)
        if parser == "grok":
            return _interpret_grok_result(proc.returncode or 0, stdout, stderr)
        return _interpret_cli_result(proc.returncode or 0, stdout, stderr)
    finally:
        if owned_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)


def _interpret_cli_result(returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}".lower()
    if returncode != 0 and _looks_like_auth_failure(combined):
        return {"error": True, "error_message": f"agentcli auth failure: {_brief(stderr or stdout)}"}

    payload = _parse_json_envelope(stdout)
    if payload is None:
        if returncode != 0:
            return {"error": True, "error_message": f"agentcli exited {returncode}: {_brief(stderr or stdout)}"}
        return {"error": True, "error_message": f"agentcli JSON parse failure: {_brief(stdout or stderr)}"}

    if payload.get("type") == "error":
        message = payload.get("message") or payload.get("result") or "CLI reported error"
        return {"error": True, "error_message": str(message)}

    if payload.get("is_error") or payload.get("error") is True:
        message = (
            payload.get("result") or payload.get("error_message") or payload.get("content") or "CLI reported error"
        )
        return {"error": True, "error_message": str(message)}

    content = _extract_content(payload)
    if not content:
        if returncode != 0:
            return {"error": True, "error_message": f"agentcli exited {returncode}: {_brief(stderr or stdout)}"}
        return {"error": True, "error_message": "agentcli returned empty content"}

    if returncode != 0:
        return {"error": True, "error_message": f"agentcli exited {returncode}: {_brief(stderr or content)}"}

    return {"content": content, "error": False}


def _interpret_grok_result(returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    return _interpret_cli_result(returncode, stdout, stderr)


def _interpret_codex_stream(returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}".lower()
    if returncode != 0 and _looks_like_auth_failure(combined):
        return {"error": True, "error_message": f"agentcli auth failure: {_brief(stderr or stdout)}"}

    texts: list[str] = []
    saw_json = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        saw_json = True
        extracted = _codex_event_text(event)
        if extracted:
            texts.append(extracted)

    if texts:
        if returncode != 0:
            return {"error": True, "error_message": f"agentcli exited {returncode}: {_brief(stderr or texts[-1])}"}
        return {"content": texts[-1], "error": False}

    if returncode != 0:
        return {"error": True, "error_message": f"agentcli exited {returncode}: {_brief(stderr or stdout)}"}
    if saw_json:
        return {"error": True, "error_message": "agentcli returned empty content"}
    return {"error": True, "error_message": f"agentcli JSON parse failure: {_brief(stdout or stderr)}"}


def _codex_event_text(event: dict[str, Any]) -> str:
    item = event.get("item")
    if isinstance(item, dict):
        item_type = item.get("type") or item.get("item_type")
        if item_type in {"agent_message", "message", None}:
            extracted = _stringify_content(item.get("text")) or _stringify_content(item.get("content"))
            if extracted:
                return extracted
    event_type = event.get("type")
    if event_type in {"agent_message", "item.completed", "message"}:
        extracted = _stringify_content(event.get("text")) or _stringify_content(event.get("content"))
        if extracted:
            return extracted
    return ""


def _stringify_content(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        return "\n".join(chunks)
    return ""


def _looks_like_auth_failure(combined_lower: str) -> bool:
    return any(marker in combined_lower for marker in _AUTH_MARKERS)


def _parse_json_envelope(stdout: str) -> dict[str, Any] | None:
    # Current Claude JSON may be an event array. Older versions emit a single
    # envelope, sometimes after launcher notices. Decode whole values so nested
    # usage metadata cannot become a false starting brace.
    decoder = json.JSONDecoder()
    text = stdout.strip()
    candidates: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        starts = [i for i in (text.find("{", offset), text.find("[", offset)) if i >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            data, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        offset = end
        if isinstance(data, dict):
            candidates.append(data)
        elif isinstance(data, list):
            candidates.extend(item for item in data if isinstance(item, dict))
    results = [item for item in candidates if item.get("type") == "result"]
    if results:
        return results[-1]
    return candidates[-1] if candidates else None


def _extract_content(payload: dict[str, Any]) -> str:
    for key in ("result", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    message = payload.get("message")
    if isinstance(message, dict):
        inner = message.get("content")
        if isinstance(inner, str) and inner.strip():
            return inner
        if isinstance(inner, list):
            chunks: list[str] = []
            for item in inner:
                if isinstance(item, dict) and item.get("type") in (None, "text"):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
                elif isinstance(item, str) and item:
                    chunks.append(item)
            if chunks:
                return "\n".join(chunks)
    return ""


async def _validate_seat(seat: str, configured: str | None) -> dict[str, str | bool]:
    try:
        binary = _resolve_launcher(seat, configured)
    except ValueError as exc:
        return {"seat": seat, "success": False, "message": str(exc)}
    if seat == "claude":
        return await _validate_claude(binary)
    if seat == "grok":
        return await _validate_grok(binary)
    return await _validate_codex(binary)


async def _validate_claude(binary: str) -> dict[str, str | bool]:
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "auth",
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        return {"seat": "claude", "success": False, "message": "guarded agentcli launcher not found"}
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (TimeoutError, asyncio.TimeoutError):
        await _kill(proc)
        return {"seat": "claude", "success": False, "message": "guarded authentication check timed out"}
    except asyncio.CancelledError:
        await _kill(proc)
        raise
    try:
        auth = json.loads(stdout)
    except (ValueError, TypeError):
        auth = {}
    ok = (
        proc.returncode == 0
        and isinstance(auth, dict)
        and auth.get("loggedIn") is True
        and auth.get("authMethod") == "claude.ai"
        and bool(auth.get("subscriptionType"))
    )
    return {
        "seat": "claude",
        "success": ok,
        "message": "Claude subscription authenticated" if ok else "Subscription authentication unverified",
    }


async def _validate_grok(binary: str) -> dict[str, str | bool]:
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "--hermetic",
            "models",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        return {"seat": "grok", "success": False, "message": "guarded agentcli launcher not found"}
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (TimeoutError, asyncio.TimeoutError):
        await _kill(proc)
        return {"seat": "grok", "success": False, "message": "guarded authentication check timed out"}
    except asyncio.CancelledError:
        await _kill(proc)
        raise
    text = (stdout + stderr).decode("utf-8", errors="replace")
    ok = proc.returncode == 0 and _positive_login_banner(text, "logged in with grok.com")
    return {
        "seat": "grok",
        "success": ok,
        "message": "Grok subscription authenticated" if ok else "Subscription authentication unverified",
    }


async def _validate_codex(binary: str) -> dict[str, str | bool]:
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "login",
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        return {"seat": "codex", "success": False, "message": "guarded agentcli launcher not found"}
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (TimeoutError, asyncio.TimeoutError):
        await _kill(proc)
        return {"seat": "codex", "success": False, "message": "guarded authentication check timed out"}
    except asyncio.CancelledError:
        await _kill(proc)
        raise
    text = (stdout + stderr).decode("utf-8", errors="replace")
    ok = proc.returncode == 0 and _positive_login_banner(text, "logged in using chatgpt")
    return {
        "seat": "codex",
        "success": ok,
        "message": "Codex subscription authenticated" if ok else "Subscription authentication unverified",
    }


def _positive_login_banner(text: str, needle: str) -> bool:
    """Require the positive banner; reject 'Not logged in …' supersets."""
    lower = text.lower()
    if "not logged in" in lower or f"not {needle}" in lower:
        return False
    return needle in lower


def _brief(text: str, limit: int = 300) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed or "(empty)"
    return collapsed[: limit - 3] + "..."


async def _kill(proc: asyncio.subprocess.Process) -> None:
    try:
        if getattr(proc, "pid", None):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except (TimeoutError, asyncio.TimeoutError, ProcessLookupError):
        pass
