# WOR-397 guarded subscription repair

The original AgentCLI seat launched raw claude with inherited model-billing environment and accepted arbitrary binary overrides. Auth validation launched an inference probe. These violated current C010 subscription-only execution policy.

The seat now resolves only the canonical C010 guarded launcher, rejects raw/foreign paths and API-key override input, uses inference-free auth status, and pins safe-mode, empty tools, strict empty MCP, no browser, and no session persistence. Timeout/cancellation kill the process group.

Evidence: two new tests failed against original implementation and pass after repair; 323 offline mock-subprocess tests pass on macOS Python3.10.18; scoped Ruff passes. Live guarded seat and independent review are separate pending checks. No runtime activation, settings persistence, API-key inference or raw CLI launch performed.

Changed implementation: backend/providers/agent_cli.py, backend/settings.py, tests/test_agent_cli_provider.py, CLAUDE.md.
