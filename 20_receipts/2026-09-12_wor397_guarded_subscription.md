# WOR-397 guarded subscription repair

The original AgentCLI seat launched raw claude with inherited model-billing environment and accepted arbitrary binary overrides. Auth validation launched an inference probe. These violated current C010 subscription-only execution policy.

The seat now resolves only the canonical C010 guarded launcher, rejects raw/foreign paths and API-key override input, uses inference-free auth status, and pins safe-mode, empty tools, strict empty MCP, no browser, and no session persistence. Timeout/cancellation kill the process group.

Evidence: two new tests failed against original implementation and pass after repair; 323 offline mock-subprocess tests pass on macOS Python3.10.18; scoped Ruff passes. Live guarded seat and independent review are separate pending checks. No runtime activation, settings persistence, API-key inference or raw CLI launch performed.

Changed implementation: backend/providers/agent_cli.py, backend/settings.py, tests/test_agent_cli_provider.py, CLAUDE.md.

Independent review round1 F1-F3 confirmed: surviving descendant cleanup after parent exit, false auth classification of successful prose, disabled direct model discovery. Three regressions reproduced red then repaired. Full offline suite326 passes; live process-group synthetic test and re-review follow.

Round2 F4 and Mac live synthetic check exposed JSON nested metadata/log-prefix and current Claude event-array envelope rejection. Both shapes reproduced with synthetic tests before repair; parser now decodes whole JSON values and selects terminal result events.328 offline tests pass. The live attempt was through guarded authenticated claude.ai/max subscription, no metered fallback.
