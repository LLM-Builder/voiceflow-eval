"""VoiceFlow Eval API. Every protected query is tenant-scoped; secrets never leave the process."""
from __future__ import annotations

import csv
import datetime as dt
import hmac
import io
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .agent.engine import FLOW_EDGES, AgentEngine, State
from .agent.providers import MockLLM, PromptConfig
from .agent.scheduling import MockScheduler
from .config import Settings, hash_key, load_settings
from .db import (
    audit_events,
    eval_runs,
    make_engine,
    metadata,
    now,
    prompt_versions,
    scenario_executions,
)
from .evaluation.metrics import (
    DEFAULT_REGRESSION,
    DEFAULT_THRESHOLDS,
    compare_runs,
    compute_metrics,
)
from .evaluation.runner import TAXONOMY, run_suite
from .evaluation.scenarios import CONDITIONS, SCENARIOS

log = logging.getLogger("voiceflow")
STRATEGIES = ("zero_shot", "few_shot", "structured_instruction", "role_based")


class Principal(BaseModel):
    tenant_id: str


class RateLimiter:
    """Sliding-window limiter (in-process; use a shared store when running several replicas)."""

    def __init__(self) -> None:
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, window: float = 60.0) -> bool:
        q, t = self.hits[key], time.monotonic()
        while q and t - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(t)
        return True


class PromptIn(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=r"^[\w .-]+$")
    strategy: Literal["zero_shot", "few_shot", "structured_instruction", "role_based"]
    system_prompt: str = Field(min_length=1, max_length=20_000)
    params: dict[str, Any] = Field(default_factory=dict)
    note: str | None = Field(default=None, max_length=255)


class RollbackIn(BaseModel):
    to_version: int = Field(ge=1)


class RunIn(BaseModel):
    prompt_name: str
    prompt_version: int | None = None
    baseline_run_id: str | None = None
    seed: int = Field(default=0, ge=0, le=2**31)
    conditions: list[str] | None = None
    redact: bool = False
    idempotency_key: str | None = Field(default=None, max_length=120)
    thresholds: dict[str, float] = Field(default_factory=dict)
    regression: dict[str, float] = Field(default_factory=dict)


class SimulateIn(BaseModel):
    prompt_name: str | None = None
    utterances: list[str] = Field(min_length=1, max_length=40)
    llm_faults: dict[str, Any] = Field(default_factory=dict)
    scheduler_faults: dict[str, Any] = Field(default_factory=dict)


class FeedbackIn(BaseModel):
    score: int = Field(ge=1, le=5)


def err(status: int, code: str, message: str, details: Any = None) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message, "details": details})


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    engine = make_engine(cfg.database_url)
    metadata.create_all(engine)
    app = FastAPI(title="VoiceFlow Eval API", version="0.1.0")
    limiter = RateLimiter()
    app.state.settings, app.state.engine = cfg, engine

    # ---- middleware / errors -------------------------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.rid = rid
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > cfg.max_body_bytes:
            return JSONResponse(status_code=413, headers={"x-request-id": rid}, content={"error": {
                "code": "payload_too_large", "message": "Request body too large", "request_id": rid}})
        start = time.perf_counter()
        resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        log.info(json.dumps({"rid": rid, "method": request.method, "path": request.url.path,
                             "status": resp.status_code, "ms": round((time.perf_counter() - start) * 1000, 2)}))
        return resp

    def body(request: Request, code: str, message: str, details: Any = None) -> dict:
        return {"error": {"code": code, "message": message, "details": details,
                          "request_id": getattr(request.state, "rid", None)}}

    @app.exception_handler(HTTPException)
    async def http_err(request: Request, exc: HTTPException):
        d = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=body(request, d["code"], d["message"], d.get("details")))

    @app.exception_handler(RequestValidationError)
    async def val_err(request: Request, exc: RequestValidationError):
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content=body(request, "validation_error", "Invalid request", details))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error rid=%s", getattr(request.state, "rid", None))
        return JSONResponse(status_code=500, content=body(request, "internal_error", "Internal server error"))

    # ---- dependencies ------------------------------------------------------------------------
    def db():
        with Session(engine) as s:
            yield s

    def auth(request: Request) -> Principal:
        ip = request.client.host if request.client else "unknown"
        h = request.headers.get("authorization", "")
        token = h[7:] if h.lower().startswith("bearer ") else ""
        digest = hash_key(token) if token else ""
        tenant = next((t for k, t in cfg.api_key_hashes.items() if hmac.compare_digest(k, digest)), None)
        if tenant is None:
            if not limiter.check(f"authfail:{ip}", cfg.auth_fail_limit_per_min):
                raise err(429, "rate_limited", "Too many failed authentication attempts")
            raise err(401, "unauthorized", "Missing or invalid API key")
        return Principal(tenant_id=tenant)

    def audit(s: Session, p: Principal, action: str, target: str | None = None, detail: Any = None) -> None:
        s.execute(insert(audit_events).values(tenant_id=p.tenant_id, actor=p.tenant_id, action=action,
                                              target=target, detail=detail, created_at=now()))

    # ---- health ------------------------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(s: Session = Depends(db)):
        try:
            s.execute(select(1))
        except Exception:
            raise err(503, "db_unavailable", "Database not reachable") from None
        return {"status": "ready", "database": "ok"}

    @app.get("/v1/me")
    def me(p: Principal = Depends(auth)):
        return {"tenant_id": p.tenant_id}

    @app.get("/v1/integrations")
    def integrations(p: Principal = Depends(auth)):
        return {"llm": {"provider": cfg.llm_provider, "mode": "mock" if cfg.llm_provider == "mock" else "unconfigured",
                        "note": "Real providers require credentials and an adapter implementing LLMProvider."},
                "scheduler": {"provider": cfg.scheduler_provider, "mode": "mock"},
                "telephony": {"provider": "simulator", "note": "Real audio/telephony/STT needs provider credentials."},
                "secrets_exposed": False}

    @app.get("/v1/flow")
    def flow(p: Principal = Depends(auth)):
        return {"states": [s.value for s in State], "edges": FLOW_EDGES,
                "terminal": ["CONFIRMED", "HANDOFF", "ENDED"]}

    @app.get("/v1/taxonomy")
    def taxonomy(p: Principal = Depends(auth)):
        return {k: {"category": c, "definition": d} for k, (c, d) in TAXONOMY.items()}

    # ---- prompts ---------------------------------------------------------------------------------------
    def _pv_out(r) -> dict:
        return {k: r[k] for k in ("id", "name", "version", "strategy", "system_prompt", "params", "note")} | {
            "created_at": r["created_at"]}

    def _new_version(s: Session, p: Principal, name: str, strategy: str, sp: str, params: dict, note: str | None):
        cur = s.execute(select(func.max(prompt_versions.c.version)).where(
            prompt_versions.c.tenant_id == p.tenant_id, prompt_versions.c.name == name)).scalar() or 0
        res = s.execute(insert(prompt_versions).values(tenant_id=p.tenant_id, name=name, version=cur + 1,
                                                       strategy=strategy, system_prompt=sp, params=params, note=note,
                                                       created_at=now()))
        audit(s, p, "prompt.create", f"{name}@{cur + 1}")
        s.commit()
        return s.execute(select(prompt_versions).where(prompt_versions.c.id == res.inserted_primary_key[0])).mappings().one()

    @app.post("/v1/prompts", status_code=201)
    def create_prompt(b: PromptIn, p: Principal = Depends(auth), s: Session = Depends(db)):
        return _pv_out(_new_version(s, p, b.name, b.strategy, b.system_prompt, b.params, b.note))

    @app.get("/v1/prompts")
    def list_prompts(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                     p: Principal = Depends(auth), s: Session = Depends(db)):
        q = (select(prompt_versions.c.name, func.max(prompt_versions.c.version).label("latest_version"))
             .where(prompt_versions.c.tenant_id == p.tenant_id).group_by(prompt_versions.c.name)
             .order_by(prompt_versions.c.name).limit(limit).offset(offset))
        return {"items": [dict(r) for r in s.execute(q).mappings()], "limit": limit, "offset": offset}

    @app.get("/v1/prompts/{name}/versions")
    def prompt_versions_list(name: str, p: Principal = Depends(auth), s: Session = Depends(db)):
        rows = s.execute(select(prompt_versions).where(prompt_versions.c.tenant_id == p.tenant_id,
                                                       prompt_versions.c.name == name)
                         .order_by(prompt_versions.c.version.desc())).mappings().all()
        if not rows:
            raise err(404, "not_found", "Prompt not found")
        return {"items": [_pv_out(r) for r in rows]}

    @app.post("/v1/prompts/{name}/rollback", status_code=201)
    def rollback(name: str, b: RollbackIn, p: Principal = Depends(auth), s: Session = Depends(db)):
        old = s.execute(select(prompt_versions).where(
            prompt_versions.c.tenant_id == p.tenant_id, prompt_versions.c.name == name,
            prompt_versions.c.version == b.to_version)).mappings().first()
        if not old:
            raise err(404, "not_found", "Prompt version not found")
        # Rollback never rewrites history: it appends a new version that copies the old content.
        return _pv_out(_new_version(s, p, name, old["strategy"], old["system_prompt"], old["params"],
                                    f"rollback to v{b.to_version}"))

    def _resolve_prompt(s: Session, p: Principal, name: str, version: int | None):
        q = select(prompt_versions).where(prompt_versions.c.tenant_id == p.tenant_id, prompt_versions.c.name == name)
        q = q.where(prompt_versions.c.version == version) if version else q.order_by(prompt_versions.c.version.desc())
        row = s.execute(q).mappings().first()
        if not row:
            raise err(404, "not_found", "Prompt version not found")
        return row

    # ---- scenarios -----------------------------------------------------------------------------------------
    @app.get("/v1/scenarios")
    def scenarios(condition: str | None = None, severity: str | None = None, expected_state: str | None = None,
                  limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0), p: Principal = Depends(auth)):
        items = [s for s in SCENARIOS if (not condition or s.condition == condition)
                 and (not severity or s.severity == severity) and (not expected_state or s.expected_state == expected_state)]
        return {"total": len(items), "conditions": CONDITIONS, "limit": limit, "offset": offset,
                "items": [{"id": s.id, "condition": s.condition, "description": s.description, "severity": s.severity,
                           "expected_state": s.expected_state, "turns": [t for t, _ in s.turns],
                           "fault_injected": s.fault_injected} for s in items[offset:offset + limit]]}

    # ---- runs --------------------------------------------------------------------------------------------------
    def _run_out(r) -> dict:
        return {k: r[k] for k in ("id", "suite", "seed", "status", "provider", "baseline_run_id", "config", "metrics",
                                  "comparison", "error", "created_at", "started_at", "finished_at")} | {
            "prompt_version_id": r["prompt_version_id"]}

    def _get_run(s: Session, p: Principal, run_id: str):
        r = s.execute(select(eval_runs).where(eval_runs.c.id == run_id, eval_runs.c.tenant_id == p.tenant_id)).mappings().first()
        if not r:
            raise err(404, "not_found", "Run not found")
        return r

    def _load_execs(s: Session, run_id: str) -> list[dict]:
        rows = s.execute(select(scenario_executions).where(scenario_executions.c.run_id == run_id)
                         .order_by(scenario_executions.c.id)).mappings().all()
        return [r["scoring"] | {"transcript": r["transcript"]} for r in rows]

    def execute_run(run_id: str, tenant: str, prompt: PromptConfig, cfg_run: dict) -> None:
        with Session(engine) as s:
            try:
                s.execute(update(eval_runs).where(eval_runs.c.id == run_id).values(status="running", started_at=now()))
                s.commit()
                chosen = [sc for sc in SCENARIOS if not cfg_run["conditions"] or sc.condition in cfg_run["conditions"]]

                def cancelled() -> bool:
                    st = s.execute(select(eval_runs.c.status).where(eval_runs.c.id == run_id)).scalar()
                    return st == "cancelled"

                def save(e: dict) -> None:
                    scoring = {k: v for k, v in e.items() if k != "transcript"}
                    s.execute(insert(scenario_executions).values(
                        run_id=run_id, tenant_id=tenant, scenario_id=e["scenario_id"], condition=e["condition"],
                        severity=e["severity"], passed=e["passed"], final_state=e["final_state"],
                        total_latency_ms=e["total_latency_ms"], fields=e["fields"], transcript=e["transcript"],
                        failures=e["failures"], scoring=scoring))
                    s.commit()

                execs = run_suite(prompt, chosen, cfg_run["seed"], cfg_run["redact"], cancelled, save)
                if cancelled():
                    s.execute(update(eval_runs).where(eval_runs.c.id == run_id).values(finished_at=now()))
                    s.commit()
                    return
                metrics = compute_metrics(execs, cfg_run["thresholds"])
                comparison = None
                base_id = cfg_run.get("baseline_run_id")
                if base_id:
                    base_execs = _load_execs(s, base_id)
                    base_metrics = compute_metrics(base_execs)
                    comparison = compare_runs(base_execs, execs, base_metrics, metrics, cfg_run["regression"])
                s.execute(update(eval_runs).where(eval_runs.c.id == run_id).values(
                    status="completed", metrics=metrics, comparison=comparison, finished_at=now(),
                    provider=",".join(metrics["provider"])))
                s.commit()
            except Exception as e:  # persisted so the dashboard can show the failure, not hide it
                log.exception("run failed")
                s.rollback()
                s.execute(update(eval_runs).where(eval_runs.c.id == run_id).values(
                    status="failed", error=f"{type(e).__name__}: {e}", finished_at=now()))
                s.commit()

    @app.post("/v1/runs", status_code=202)
    def create_run(b: RunIn, bg: BackgroundTasks, response: Response, p: Principal = Depends(auth),
                   s: Session = Depends(db)):
        if not limiter.check(f"run:{p.tenant_id}", cfg.run_rate_limit_per_min):
            raise err(429, "rate_limited", "Too many evaluation runs; retry later")
        if b.conditions and (bad := [c for c in b.conditions if c not in CONDITIONS]):
            raise err(422, "invalid_condition", "Unknown conditions", bad)
        if b.idempotency_key:
            ex = s.execute(select(eval_runs).where(eval_runs.c.tenant_id == p.tenant_id,
                                                   eval_runs.c.idempotency_key == b.idempotency_key)).mappings().first()
            if ex:
                response.status_code = 200
                return _run_out(ex)
        pv = _resolve_prompt(s, p, b.prompt_name, b.prompt_version)
        if b.baseline_run_id:
            _get_run(s, p, b.baseline_run_id)  # tenant-scoped existence check
        run_id = str(uuid.uuid4())
        run_cfg = {"seed": b.seed, "conditions": b.conditions, "redact": b.redact, "thresholds": b.thresholds,
                   "regression": b.regression, "baseline_run_id": b.baseline_run_id,
                   "prompt": {"name": pv["name"], "version": pv["version"], "strategy": pv["strategy"],
                              "system_prompt": pv["system_prompt"], "params": pv["params"]},
                   "defaults": {"thresholds": DEFAULT_THRESHOLDS, "regression": DEFAULT_REGRESSION}}
        try:
            s.execute(insert(eval_runs).values(id=run_id, tenant_id=p.tenant_id, prompt_version_id=pv["id"],
                                               baseline_run_id=b.baseline_run_id, suite="builtin", seed=b.seed,
                                               config=run_cfg, status="queued", idempotency_key=b.idempotency_key,
                                               created_at=now()))
            audit(s, p, "run.create", run_id)
            s.commit()
        except IntegrityError:
            s.rollback()
            raise err(409, "conflict", "Duplicate idempotency key") from None
        prompt = PromptConfig(pv["name"], pv["version"], pv["strategy"], pv["system_prompt"], dict(pv["params"]))
        bg.add_task(execute_run, run_id, p.tenant_id, prompt, run_cfg)
        return _run_out(_get_run(s, p, run_id))

    @app.get("/v1/runs")
    def list_runs(status: str | None = None, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  p: Principal = Depends(auth), s: Session = Depends(db)):
        q = select(eval_runs).where(eval_runs.c.tenant_id == p.tenant_id)
        cq = select(func.count()).select_from(eval_runs).where(eval_runs.c.tenant_id == p.tenant_id)
        if status:
            q, cq = q.where(eval_runs.c.status == status), cq.where(eval_runs.c.status == status)
        rows = s.execute(q.order_by(eval_runs.c.created_at.desc()).limit(limit).offset(offset)).mappings().all()
        return {"total": s.execute(cq).scalar(), "limit": limit, "offset": offset, "items": [_run_out(r) for r in rows]}

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str, p: Principal = Depends(auth), s: Session = Depends(db)):
        return _run_out(_get_run(s, p, run_id))

    @app.post("/v1/runs/{run_id}/cancel")
    def cancel_run(run_id: str, p: Principal = Depends(auth), s: Session = Depends(db)):
        r = _get_run(s, p, run_id)
        if r["status"] not in ("queued", "running"):
            raise err(409, "conflict", f"Run already {r['status']}")
        s.execute(update(eval_runs).where(eval_runs.c.id == run_id).values(status="cancelled"))
        audit(s, p, "run.cancel", run_id)
        s.commit()
        return _run_out(_get_run(s, p, run_id))

    @app.delete("/v1/runs/{run_id}", status_code=204)
    def delete_run(run_id: str, p: Principal = Depends(auth), s: Session = Depends(db)):
        _get_run(s, p, run_id)
        s.execute(update(eval_runs).where(eval_runs.c.baseline_run_id == run_id).values(baseline_run_id=None))
        s.execute(delete(scenario_executions).where(scenario_executions.c.run_id == run_id))
        s.execute(delete(eval_runs).where(eval_runs.c.id == run_id, eval_runs.c.tenant_id == p.tenant_id))
        audit(s, p, "run.delete", run_id)
        s.commit()
        return Response(status_code=204)

    @app.get("/v1/runs/{run_id}/executions")
    def list_execs(run_id: str, passed: bool | None = None, condition: str | None = None,
                   limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                   p: Principal = Depends(auth), s: Session = Depends(db)):
        _get_run(s, p, run_id)
        c = scenario_executions.c
        q = select(scenario_executions).where(c.run_id == run_id, c.tenant_id == p.tenant_id)
        if passed is not None:
            q = q.where(c.passed == passed)
        if condition:
            q = q.where(c.condition == condition)
        rows = s.execute(q.order_by(c.id).limit(limit).offset(offset)).mappings().all()
        return {"limit": limit, "offset": offset, "items": [
            {"id": r["id"], "scenario_id": r["scenario_id"], "condition": r["condition"], "severity": r["severity"],
             "passed": r["passed"], "final_state": r["final_state"], "failures": r["failures"],
             "total_latency_ms": r["total_latency_ms"], "feedback_score": r["feedback_score"]} for r in rows]}

    @app.get("/v1/executions/{exec_id}")
    def get_exec(exec_id: int, p: Principal = Depends(auth), s: Session = Depends(db)):
        r = s.execute(select(scenario_executions).where(scenario_executions.c.id == exec_id,
                                                        scenario_executions.c.tenant_id == p.tenant_id)).mappings().first()
        if not r:
            raise err(404, "not_found", "Execution not found")
        return {k: r[k] for k in ("id", "run_id", "scenario_id", "condition", "severity", "passed", "final_state",
                                  "fields", "transcript", "failures", "total_latency_ms", "feedback_score")}

    @app.post("/v1/executions/{exec_id}/feedback")
    def feedback(exec_id: int, b: FeedbackIn, p: Principal = Depends(auth), s: Session = Depends(db)):
        res = s.execute(update(scenario_executions).where(scenario_executions.c.id == exec_id,
                                                          scenario_executions.c.tenant_id == p.tenant_id)
                        .values(feedback_score=b.score))
        if res.rowcount == 0:
            raise err(404, "not_found", "Execution not found")
        s.commit()
        return {"id": exec_id, "feedback_score": b.score}

    @app.get("/v1/compare")
    def compare(baseline_run: str, candidate_run: str, p: Principal = Depends(auth), s: Session = Depends(db)):
        br, cr = _get_run(s, p, baseline_run), _get_run(s, p, candidate_run)
        if br["status"] != "completed" or cr["status"] != "completed":
            raise err(409, "conflict", "Both runs must be completed")
        be, ce = _load_execs(s, baseline_run), _load_execs(s, candidate_run)
        return compare_runs(be, ce, br["metrics"], cr["metrics"]) | {
            "side_by_side": [{"scenario_id": x["scenario_id"], "baseline": x["transcript"], "candidate": y["transcript"]}
                             for x in be for y in ce if x["scenario_id"] == y["scenario_id"]
                             and x["passed"] != y["passed"]]}

    @app.get("/v1/runs/{run_id}/export")
    def export(run_id: str, format: Literal["json", "csv"] = "json", p: Principal = Depends(auth),
               s: Session = Depends(db)):
        r = _get_run(s, p, run_id)
        execs = _load_execs(s, run_id)
        if format == "json":
            return JSONResponse(jsonable_encoder({"run": _run_out(r), "executions": execs}),
                                headers={"content-disposition": f'attachment; filename="run-{run_id}.json"'})
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["scenario_id", "condition", "severity", "passed", "final_state", "handoff_reason", "failure_codes"])
        for e in execs:
            w.writerow([e["scenario_id"], e["condition"], e["severity"], e["passed"], e["final_state"],
                        e["handoff_reason"] or "", ";".join(f["code"] for f in e["failures"])])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"content-disposition": f'attachment; filename="run-{run_id}.csv"'})

    @app.post("/v1/simulate")
    def simulate(b: SimulateIn, p: Principal = Depends(auth), s: Session = Depends(db)):
        pc = PromptConfig()
        if b.prompt_name:
            pv = _resolve_prompt(s, p, b.prompt_name, None)
            pc = PromptConfig(pv["name"], pv["version"], pv["strategy"], pv["system_prompt"], dict(pv["params"]))
        eng = AgentEngine(MockLLM(pc.params.get("mock_profile", "full"), b.llm_faults),
                          MockScheduler(b.scheduler_faults), pc, session_id=uuid.uuid4().hex[:12])
        eng.start()
        for u in b.utterances:
            if eng.state.value in ("CONFIRMED", "HANDOFF", "ENDED"):
                break
            eng.step(u)
        return eng.result() | {"provider": "mock-rule-based", "data_label": "MOCK_ADAPTERS"}

    @app.post("/v1/admin/retention")
    def retention(days: int = Query(ge=1, le=3650), p: Principal = Depends(auth), s: Session = Depends(db)):
        cutoff = now() - dt.timedelta(days=days)
        old = [r for r in s.execute(select(eval_runs.c.id).where(
            eval_runs.c.tenant_id == p.tenant_id, eval_runs.c.created_at < cutoff)).scalars()]
        for rid in old:
            s.execute(update(eval_runs).where(eval_runs.c.baseline_run_id == rid).values(baseline_run_id=None))
            s.execute(delete(scenario_executions).where(scenario_executions.c.run_id == rid))
            s.execute(delete(eval_runs).where(eval_runs.c.id == rid))
        audit(s, p, "retention.purge", None, {"days": days, "runs_deleted": len(old)})
        s.commit()
        return {"runs_deleted": len(old)}

    return app


app = None  # created lazily by the ASGI entrypoint below


def get_app() -> FastAPI:
    return create_app()
