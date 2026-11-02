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

**Test isolation** (`tests/conftest.py`): at conftest import, which pytest performs before importing any test module, the suite:
- sets `SADB_DATA_DIR` and `SADB_CLI_PATH` to a disposable root;
- removes metered key environment variables;
- points the data, settings, debug, mode-session and claims paths at that root;
- blocks non-UNIX socket connect and send calls, `create_connection`, and the resolver functions (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`, `getnameinfo`).

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

## Round-1 independent review (guarded hermetic Grok, grok-4.7, head `56ee2d2`)

The review produced 20 findings, recorded in the PR's review-budget ledger. 18 were adopted and repaired at the next head; 2 were rejected:
- **R1-G6:** a pre-existing duplicate-label defect, kept out of scope.
- **R1-G13:** a by-design preflight stub in plumbing tests.

The repairs:
- **`/v1` HTTP statuses:** a preflight refusal is HTTP 400 on both paths, checked before the response starts. Non-streaming budget rejection is 429, abort and failed council are 502, and an engine exception is 500.
- **Round counts:** the UI passes the requested round count to the engine, which records `requested_rounds`, `planned_rounds` and the actual `executed_rounds`.
- **Non-empty reasons:** the call ledger never stores an empty failure reason, cancelled attempts record `cancelled`, and aborted runs record a reason.
- **SSE reasons:** budget and abort SSE data carry `failure_reason`.
- **Null chair:** a `null` chair answer is stored as `{}`.
- **Empty model names:** a known prefix with an empty model name is refused.
- **Capture:** it also requires a chair answer and stamps `failure_reason`.
- **Import-time isolation:** it now really happens at conftest import (network guard first). It extends the key scrub and the `sendto`/`sendmsg` guard.
- **Tests:** added for each repair.

## Round-2 independent review (guarded hermetic Grok, grok-4.7, head `93495cc`)

The reviewer confirmed the round-1 repairs and raised 8 findings (R2-F1..F8). R2-F1 was rejected: `CallAccountant.close_open` already records `cancelled` for in-flight moderator and chair attempts; its missing test was adopted under R2-F8. The other 7 were adopted:
- **Blank chair answers on `/v1` (F2):** a whitespace or `null` chair answer is a failure: HTTP 502 when non-streaming; when streaming (already HTTP 200 once the body starts) an explicit failure notice.
- **Request-shape refusals (F3, F4):** no user message, or fewer than 2 council models, is HTTP 400 at the route. The non-streaming generator raises a 400 `RoundtableFailedError` for its own preflight instead of returning a 200 "Error" answer. An engine run that ends without a terminal answer is 502 (`roundtable_incomplete`), or a streaming failure notice; the "No response generated." 200 fallback is gone.
- **Reasons in error bodies (F5):** `/v1` error bodies carry `failure_reason`; the budget rejection keeps the run's reason.
- **Defense in depth (F6):** `get_provider_for_model` refuses metered providers, and a static AST test asserts that the only `.query` reference in `backend/` is inside `council.query_model` (round 3 widened it from direct calls to every attribute reference and `getattr(..., "query")`).
- **Docs and dead code (F7):** the remaining OpenRouter inference body was removed; the resolver guards were extended; docs state the guarantees precisely.
- **Test strength (F8):** the import-time isolation test now checks the guards recorded at conftest import, the full key-scrub list and `sendto`/`connect_ex` on an INET socket; the OpenRouter spy also covers `httpx.AsyncClient.send`; moderator and chair cancellation are tested.

19 of the new tests failed against `93495cc` with the new conftest (RED); all pass at the repair head.

## Round-3 independent review (guarded hermetic Grok, grok-4.7, head `8dca2aa`)

The reviewer confirmed R2-F3, F5 and F7 fixed and the rest mostly fixed, and raised 8 findings. Rejected with evidence: R3-2 (the streaming generator's contract is an SSE body; the route already returns 400 before streaming; its comment was corrected), R3-4 (`query_models_parallel` imports `asyncio` locally; now covered by a test) and R3-5 (the network guard raises a `RuntimeError` on purpose so `except OSError` handlers cannot swallow a blocked call). Adopted and repaired:
- **R3-1:** a blank chair answer is reported as `failed` (error `code`) even when the run recorded `completed` or `degraded`.
- **R3-3:** a whitespace-only user message is refused with 400.
- **R3-6:** the static dispatch test counts every `.query` attribute reference and `getattr(..., "query")`, not just direct calls; docs state what it proves.
- **R3-7:** comments, CLAUDE.md and this receipt were corrected.
- **R3-8:** tests assert error `code`/`failure_reason`, drive streaming fakes explicitly, add positive `get_provider_for_model` cases for `ollama`/`agentcli`, cover `query_models_parallel`, the 400 `failure_reason` and the streaming abort reason.

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
- GREEN at the round-1 repair head: 469 engine tests pass, including 100 in `tests/test_wor1016_outcome_and_routing.py`; the first head (`56ee2d2`) had 424 and 55. Scoped ruff check and format pass. The frontend fixture passes, including when replaying the real `main.py` SSE output. `npm run build` passes.
- Isolation was verified on effective paths, not just environment variables:
  - `resolve_sadb_root()` resolves to the disposable root;
  - storage, run and settings paths sit under it;
  - socket connections raise;
  - captured synthetic turns appeared only under `$TMPDIR/llm-council-tests-*/sadb`.
- A before/after `stat` snapshot of the real `~/SyncedProjects/_SADB_Data/hot` and the real engine `data/` showed no change, and zero files under `_SADB_Data` were newer than the pre-test snapshot.

No model or provider call, quota probe, live web-app run, provider activation or shared-data write was made. WOR-397 keeps live seat acceptance and the earlier synthetic-turn disposition. Exact head, review and triage are recorded on the PR.
