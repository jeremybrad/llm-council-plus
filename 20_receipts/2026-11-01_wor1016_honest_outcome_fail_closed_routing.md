# WOR-1016 honest run outcome and WOR-362 fail-closed routing

PRD: N/A (WOR-1016, Linear-governed; routing ruling recorded on WOR-362 comment 297e4a19, Jeremy, 2026-11-01)

## Why

On 2026-11-01 an engine product audit at `1d70601` used source reading and mock-only checks. It found that a council where every seat, the moderator and the chair failed was still saved as `completed`. That record was turn-captured to SADB and returned on `/v1` as HTTP 200. Moderator and chair error reasons were dropped. HTTP timeouts produced empty (falsy) seat errors. Turn capture also fired for aborted and budget-rejected runs. Requests for more than 3 rounds ran 3 but were budgeted for N. Separately, `council.get_provider_for_model` sent blank, unprefixed and unknown model IDs to OpenRouter, which is metered, and `enabled_providers.openrouter` defaulted to true.

Jeremy's ruling on WOR-362: reject blank, unprefixed and unknown routes, with no automatic OpenRouter fallback. Neither a stored key nor an enabled flag authorizes paid inference under the $0 metered budget.

## What changed

**Routing** (`council.py`, `providers/openrouter.py`, `settings.py`):
- Model IDs must carry an explicit known prefix, and their provider must be enabled. Direct providers need both the `direct` master flag and their individual toggle.
- `query_model` refuses a route without contacting any provider.
- `OpenRouterProvider` also refuses while disabled.
- The `openrouter` default is now `false`.
- `preflight_roundtable_models` rejects blank seat, moderator and chair configurations before dispatch. Both the UI SSE path and `/v1` call it.

**Outcome** (`roundtable.py`):
- Terminal statuses are `completed`, `degraded`, `failed`, `aborted` and `budget_exceeded`.
- Quorum stop: if no seat succeeds in round 1, the run ends `failed` and skips later rounds, the moderator and the chair.
- Runs record `failed_seats` (in council order), `failure_reason`, `requested_rounds` and `executed_rounds`.
- Failure reasons are never empty. They fall back to the exception type or a fixed message. Empty content counts as a failure.
- Rounds are capped at 3 before prediction.

**Consumers:**
- `main.py` emits outcome data with its terminal events, stores the status in message metadata, and captures only `completed` and `degraded` runs.
- `turn_capture.capture_run` independently refuses any other status and tags each turn with `run_status` and `failed_seats`.
- `/v1` labels degraded answers. A failed council produces an explicit failure notice when streaming, and HTTP 502 with an OpenAI-style error body when not.
- The timeline shows Degraded, Failed and Budget-rejected badges, seat `error` text, and moderator and chair failure reasons.

**Test isolation** (`tests/conftest.py`): at conftest import, before any `backend` import, the suite:
- sets `SADB_DATA_DIR` and `SADB_CLI_PATH` to a disposable root;
- removes metered key environment variables;
- points the data, settings, debug, mode-session and claims paths at that root;
- blocks non-UNIX socket connections and DNS.

## Explicitly enabled metered routes (WOR-362 follow-up, same PR)

Jeremy asked whether refusing the *automatic* fallback is enough, given his rule that a saved key or an enabled toggle is not spending authorization. At head `56ee2d2` it was not:
- `provider_enabled()` still allowed `openrouter` when its flag was true, and the direct providers when `direct` plus their own toggle were true. `OpenRouterProvider` then dispatched.
- Anthropic's settings key check sent a real `POST /messages` request (paid inference) using the stored key.

Synthetic RED: 21 new tests failed at `56ee2d2`.

The repair:
- `METERED_API_PROVIDERS` (`openrouter`, `openai`, `anthropic`, `google`, `mistral`, `deepseek`, `groq`, `custom`) is refused unconditionally in `check_model_route`, so `query_model` and the preflight both refuse it.
- `OpenRouterProvider.query` and the lowest-layer `openrouter.query_model` refuse before any HTTP request.
- The Anthropic key check now lists models instead of sending a message.
- `custom` is refused because an arbitrary endpoint's billing cannot be verified.
- Remaining routes: local `ollama` and guarded `agentcli`.
- Open WOR-362 question, not decided here: whether Ollama Cloud models proxied by a local daemon count as metered.

## Evidence

- The RED baseline at `1d70601` (same scenarios, mocks only) printed:
  - all seats failed → `completed`, with 11 attempted calls and the chair still called;
  - one seat failed → `completed`;
  - chair error → `completed`, with `error_message` None;
  - empty `error_message` → falsy seat error;
  - 5 rounds with 4 seats → budget rejection predicting 22 calls;
  - `''`, `gpt-4o` and `antigravity:x` → `OpenRouterProvider`;
  - default `openrouter` → True;
  - capture not gated on status.
- GREEN: 424 engine tests pass, including 55 new ones in `tests/test_wor1016_outcome_and_routing.py`. Scoped ruff check and format pass. The frontend fixture passes, including when replaying the real `main.py` SSE output. `npm run build` passes.
- Isolation was verified on effective paths, not just environment variables:
  - `resolve_sadb_root()` resolves to the disposable root;
  - storage, run and settings paths sit under it;
  - socket connections raise;
  - captured synthetic turns appeared only under `$TMPDIR/llm-council-tests-*/sadb`.
- A before/after `stat` snapshot of the real `~/SyncedProjects/_SADB_Data/hot` and the real engine `data/` showed no change, and zero files under `_SADB_Data` were newer than the pre-test snapshot.

No model or provider call, quota probe, live web-app run, provider activation or shared-data write was made. WOR-397 keeps live seat acceptance and the earlier synthetic-turn disposition. Exact head, review and triage are recorded on the PR.
