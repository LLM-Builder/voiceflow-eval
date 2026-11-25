# VoiceFlow Eval

Platform for designing, simulating, testing and evaluating LLM-powered voice agents. Initial use case: an AI sales
assistant that qualifies leads and books product demos.

## Status (read this first)

| Area | State |
|---|---|
| Agent engine (state machine, provider/scheduler interfaces, idempotent booking, handoff) | Implemented, tested |
| 36 scripted scenarios, 9 conditions, metrics, failure taxonomy, regression gates, CLI | Implemented, tested |
| FastAPI service (auth, tenants, prompts + rollback, runs, compare, export, simulate, retention) | Implemented, tested |
| PostgreSQL schema + SQL migration, SQLite for dev | Implemented; SQLite tested, **PostgreSQL not run here** |
| Docker / Compose / Render / CI files | Written, **not executed here** (no Docker in the build sandbox) |
| Next.js dashboard (7 pages) | **Not built yet** — API and CLI cover the same operations |
| Real LLM provider, real scheduler, telephony/STT | **Not included** — interfaces + mocks only; need credentials |
| Alembic, background worker queue | Not included (plain SQL migration; FastAPI BackgroundTasks) |
| Custom suite CRUD/import | Not included (built-in suite only) |

The repository you described (166 tests, 36-case baseline) was not available to me, so nothing was inspected or preserved.
The 36 scenarios and 67 tests here were authored fresh.

## Local setup
    cd backend
    pip install -r requirements-dev.txt
    pytest -q                       # 67 tests
    ruff check .
    python -m app.evaluation.cli --prompt prompts/candidate_v2.json --baseline prompts/baseline_v1.json --out reports
    set VOICEFLOW_API_KEYS=dev:my-long-random-key     (Windows cmd; use export on bash)
    uvicorn app.main:get_app --factory --reload        # docs at http://localhost:8000/docs

Docker: copy `.env.example` to `.env`, fill it, then `docker compose up --build`.
Render: `render.yaml` (set `VOICEFLOW_API_KEYS` in the dashboard). Apply `backend/migrations/001_init.sql` to PostgreSQL once
(the app also creates tables automatically on SQLite).

## Measured results (mock adapters only)
`MockLLM` is a deterministic rule-based simulator; "weak" vs "full" profiles stand in for two prompt versions. These numbers
validate the pipeline, state machine and metrics, **not the quality of any real LLM**.

| Metric (36 scenarios) | sales-agent v1 (weak) | v2 (full) |
|---|---|---|
| Task completion | 35/36 | 36/36 |
| Intent macro-F1 | 0.624 | 1.000 |
| Handoff F1 | 0.957 | 1.000 |
| Gates | fails intent_macro_f1 | pass |

Candidate v2 vs baseline v1 shows 1 improvement and 0 regressions; reversing the direction flags `handoff-02`
(high severity), which is how regression detection is tested. Structured-output validity is 120/131 for v2 because the
scenarios inject malformed outputs on purpose. User-feedback scores are not measured (null) until real ratings are posted.

## Design notes
* Conversation logic (`app/agent/engine.py`) knows only the `LLMProvider` and `SchedulingAdapter` interfaces.
* A booking is claimed only from a provider confirmation. `book` is never blindly retried: after a timeout the engine
  does a read-only `lookup(idempotency_key)`. Duplicate deliveries are absorbed by the key.
* Slot choices are resolved against offered slots; raw user text never reaches tool calls.
* Failures are classified (`GET /v1/taxonomy`) as agent vs integration.
* Every run stores its exact prompt version, config, seed and provider label.
* Auth: bearer API keys (hashed), tenant-scoped queries, rate limits on auth failures and run creation, request IDs,
  body-size limit, optional transcript redaction, retention purge. Rate limiting is in-process.

## Known limitations
Mock-only results; no frontend yet; single-process rate limiter; no user accounts (API-key tenants only);
runs execute in-process (fine for the mock, add a worker for slow real providers).
