# Receipt: Claims API v0.1 (FastAPI)

- **Date**: 2026-01-12
- **Repo**: llm-council-plus (submodule of C012_round-table)
- **Branch**: main
- **Commit**: 04653d7
- **Type**: feat
- **Why**: Expose the existing claims store + validation as a REST API for integration and testing.

## What changed

### Files created
- `backend/api/__init__.py` — router exports for clean main import
- `backend/api/claims.py` — Claims API router + Pydantic models (v0.1)
- `tests/test_claims_api.py` — HTTP-level tests using FastAPI TestClient

### Files modified
- `backend/main.py` — include router: `app.include_router(claims_router)`

## Endpoints implemented

- `GET /api/claims` — list with filters + pagination + response shaping
- `GET /api/claims/{id}` — fetch one claim (+ include options)
- `POST /api/claims` — create candidate claim
- `PATCH /api/claims/{id}` — update status/temporal fields with audit note
- `POST /api/claims/{id}/evidence` — add evidence, recalc confidence
- `POST /api/claims/{id}/validate` — validate a single claim (SADB integration mocked in tests)

Response shaping controls:
- `include_evidence`
- `include_history`
- `quote_max_len` (default 280)

## Verification

Commands run:
```bash
uv run pytest tests/test_claims.py tests/test_claims_api.py -v --tb=short
```

Result:
- PASS — 61 tests (21 claims store + 40 API tests)

## Stash status
- EMPTY (verified at closeout)

## Notes / follow-ups

- Consider adding a `make verify` / `scripts/verify_*.py` entry point if this repo is promoted into stricter "work pool" style validation.
