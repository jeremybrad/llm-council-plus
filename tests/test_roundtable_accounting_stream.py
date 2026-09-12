"""Synthetic SSE accounting contract; providers, storage and SADB are stubbed."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend import main, turn_capture


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["chair_complete", "roundtable_aborted", "roundtable_budget_exceeded"])
async def test_stream_preserves_call_accounting(monkeypatch, terminal):
    ledger = {
        "predicted_calls": 6,
        "attempted_calls": 4,
        "failed_calls": 1,
        "max_calls_per_run": 14,
        "quota_units": "unknown",
    }
    if terminal == "roundtable_budget_exceeded":
        ledger.update(attempted_calls=0, failed_calls=0)
    run = {
        "run_id": "synthetic-run",
        "conversation_id": "synthetic",
        "status": "completed",
        "call_accounting": ledger,
        "chair_final": {},
    }

    async def synthetic_roundtable(**kwargs):
        yield {"type": "roundtable_init", "call_accounting": {**ledger, "attempted_calls": 0}}
        yield {
            "type": terminal,
            "run": run,
            "chair_final": {},
            "predicted_calls": 6,
            "max_calls_per_run": 14,
            "quota_units": "unknown",
        }

    monkeypatch.setattr(main, "run_roundtable", synthetic_roundtable)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            council_models=["agentcli:claude", "agentcli:codex"],
            chairman_model="agentcli:claude",
            roundtable_num_rounds=1,
            roundtable_max_parallel=2,
        ),
    )
    monkeypatch.setattr(main.storage, "get_conversation", lambda _: {"messages": [{"role": "user"}]})
    for name in ["add_user_message", "save_run", "add_roundtable_message", "add_error_message"]:
        monkeypatch.setattr(main.storage, name, Mock())
    monkeypatch.setattr(turn_capture, "capture_run", Mock())
    response = await main.send_message_stream(
        "synthetic",
        main.SendMessageRequest(content="synthetic only", execution_mode="roundtable", web_search=False),
        SimpleNamespace(),
    )
    events = []
    async for chunk in response.body_iterator:
        events.append(json.loads(chunk.removeprefix("data: ").strip()))
    accounting = [e for e in events if e["type"] == "roundtable_accounting"]
    assert accounting[0]["data"]["attempted_calls"] == 0
    assert accounting[-1]["data"] == ledger

    evidence_dir = os.environ.get("WOR402_STREAM_EVIDENCE_DIR")
    if evidence_dir:
        Path(evidence_dir, terminal + ".json").write_text(json.dumps(events))
