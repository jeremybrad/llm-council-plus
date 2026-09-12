# WOR-402 timeline call-accounting acceptance repair

Existing engine #3 and parent #7 delivered the run ledger and guardrails. The required timeline display was absent: three synthetic endpoint terminal fixtures emitted no accounting, and server-rendering the timeline with a ledger showed no counts. This branch finishes that existing acceptance path.

The SSE adapter now forwards initial and terminal accounting. The App uses a shared pure transition helper, and the timeline renders predicted/attempted/failed invocation snapshots, unknown values and terminal errors. Snapshot timing is explicit; provider quota usage remains unknown. README and CLAUDE recommend roundtable:fast and distinguish each subscription seat's account limits from invocation counts.

Guarded Grok supplied an implementation-only proposal, captured by Betty. Independent review remains a separate required step recorded on the PR. Synthetic endpoint events are replayed through the same App helper into the actual React timeline server render. Completion, abort, budget lockout, missing/invalid ledger, true zero and legacy records are covered. Provider calls, storage and SADB capture are stubbed. No runtime flag, provider implementation, quota measurement, scheduler or production data was changed. WOR-397 remains independent. Parent gitlink integration follows Jeremy's engine merge.

Exact head, validation and independent triage are recorded on the PR.

Independent review1 found WOR402-UI-001: missing/invalid terminal ledger retained the initial zero snapshot. Reproduced RED with valid initial ledger followed by null/array/string terminal accounting and completion/error. The App now forwards these events and the shared state helper invalidates the snapshot to unknown fields. Existing true-zero and valid-ledger behavior is retained. Repair review follows; no unrelated scope added.
