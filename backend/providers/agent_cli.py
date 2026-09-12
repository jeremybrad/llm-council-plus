"""Native subscription-CLI provider (Path B / WOR-397).

MVP seat: `agentcli:claude` → headless `claude -p --output-format json`.
Prompt is piped on stdin (round-3 prompts exceed safe argv length). Tools are
disabled and the process runs in a scratch cwd so a council seat cannot edit
the repo. Temperature is dropped — vendor CLIs do not accept it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from typing import Any

from ..settings import get_settings
from .base import LLMProvider

# Tools a pure-text deliberation seat must not invoke. Passed as one
# comma-separated --disallowedTools argument (Claude Code yargs form).
_DISALLOWED_TOOLS = (
    "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit,Agent,Skill,Task,TodoWrite,BashOutput,KillShell"
)

_DEFAULT_BINARY = "claude"
_DEFAULT_MODEL_ID = "agentcli:claude"

# Auth-failure substrings observed in Claude Code stderr / result text.
_AUTH_MARKERS = (
    "not logged in",
    "please run /login",
    "please login",
    "authentication",
    "unauthenticated",
    "invalid api key",
    "not authenticated",
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

        seat = model_id.split(":", 1)[-1] if ":" in model_id else model_id
        if seat != "claude":
            return {
                "error": True,
                "error_message": f"Unsupported agentcli seat '{seat}' (MVP is agentcli:claude)",
            }

        prompt = _flatten_messages(messages)
        if not prompt.strip():
            return {"error": True, "error_message": "Empty prompt"}

        return await _invoke_claude(
            binary=_resolve_binary(settings.agentcli_binary_path),
            prompt=prompt,
            timeout=timeout,
        )

    async def get_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": _DEFAULT_MODEL_ID,
                "name": "Claude Code [agentcli]",
                "provider": "AgentCLI",
                "is_free": True,
            }
        ]

    async def validate_key(self, api_key: str) -> dict[str, Any]:
        """CLI presence + auth smoke test.

        `api_key` is an optional binary-path override (there is no API key).
        Smoke-tests even when the provider toggle is off.
        """
        settings = get_settings()
        binary = _resolve_binary(api_key or settings.agentcli_binary_path)
        if shutil.which(binary) is None and not os.path.isfile(binary):
            return {"success": False, "message": f"agentcli binary not found: {binary}"}

        smoke = await _invoke_claude(
            binary=binary,
            prompt="Reply with the single word pong.",
            timeout=30.0,
        )
        if smoke.get("error"):
            return {"success": False, "message": smoke.get("error_message", "agentcli smoke test failed")}
        return {"success": True, "message": "Claude Code CLI is present and authenticated"}


def _resolve_binary(configured: str | None) -> str:
    path = (configured or "").strip()
    return path or _DEFAULT_BINARY


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
        "--output-format",
        "json",
        "--disallowedTools",
        _DISALLOWED_TOOLS,
        "--max-turns",
        "1",
    ]
    scratch = tempfile.mkdtemp(prefix="agentcli-")
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=scratch,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            return {"error": True, "error_message": f"agentcli binary not found: {binary}"}

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            await _kill(proc)
            return {"error": True, "error_message": f"agentcli timed out after {timeout}s"}

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        return _interpret_cli_result(proc.returncode or 0, stdout, stderr)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _interpret_cli_result(returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}".lower()
    if _looks_like_auth_failure(combined):
        return {"error": True, "error_message": f"agentcli auth failure: {_brief(stderr or stdout)}"}

    payload = _parse_json_envelope(stdout)
    if payload is None:
        if returncode != 0:
            return {"error": True, "error_message": f"agentcli exited {returncode}: {_brief(stderr or stdout)}"}
        return {"error": True, "error_message": f"agentcli JSON parse failure: {_brief(stdout or stderr)}"}

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


def _looks_like_auth_failure(combined_lower: str) -> bool:
    return any(marker in combined_lower for marker in _AUTH_MARKERS)


def _parse_json_envelope(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    # Claude sometimes emits logs before the JSON object. Take the last object.
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_content(payload: dict[str, Any]) -> str:
    for key in ("result", "content"):
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


def _brief(text: str, limit: int = 300) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed or "(empty)"
    return collapsed[: limit - 3] + "..."


async def _kill(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except (TimeoutError, asyncio.TimeoutError, ProcessLookupError):
        pass
