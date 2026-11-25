"""Normalised schema. JSON columns map to JSONB on PostgreSQL. Every tenant-owned table carries tenant_id."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

metadata = MetaData()
JSONX = JSON().with_variant(JSONB(), "postgresql")


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


prompt_versions = Table(
    "prompt_versions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tenant_id", String(64), nullable=False),
    Column("name", String(120), nullable=False),
    Column("version", Integer, nullable=False),
    Column("strategy", String(32), nullable=False),
    Column("system_prompt", Text, nullable=False),
    Column("params", JSONX, nullable=False, default=dict),
    Column("note", String(255)),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
    UniqueConstraint("tenant_id", "name", "version", name="uq_prompt_version"),
)
Index("ix_prompt_tenant_name", prompt_versions.c.tenant_id, prompt_versions.c.name)

eval_runs = Table(
    "eval_runs", metadata,
    Column("id", String(36), primary_key=True),
    Column("tenant_id", String(64), nullable=False),
    Column("prompt_version_id", Integer, ForeignKey("prompt_versions.id"), nullable=False),
    Column("baseline_run_id", String(36), ForeignKey("eval_runs.id", ondelete="SET NULL")),
    Column("suite", String(64), nullable=False),
    Column("seed", Integer, nullable=False, default=0),
    Column("config", JSONX, nullable=False, default=dict),  # exact configuration used, for reproducibility
    Column("status", String(16), nullable=False),  # queued|running|completed|cancelled|failed
    Column("provider", String(64)),
    Column("idempotency_key", String(120)),
    Column("metrics", JSONX),
    Column("comparison", JSONX),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    UniqueConstraint("tenant_id", "idempotency_key", name="uq_run_idempotency"),
)
Index("ix_runs_tenant_created", eval_runs.c.tenant_id, eval_runs.c.created_at)

scenario_executions = Table(
    "scenario_executions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(36), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False),
    Column("tenant_id", String(64), nullable=False),
    Column("scenario_id", String(64), nullable=False),
    Column("condition", String(64), nullable=False),
    Column("severity", String(16), nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("final_state", String(16), nullable=False),
    Column("total_latency_ms", Float),
    Column("fields", JSONX),            # extracted structured fields (redacted if requested)
    Column("transcript", JSONX),        # turn-level events: state transitions, tool calls, latencies
    Column("failures", JSONX),          # failure classifications
    Column("scoring", JSONX),           # raw per-scenario scoring inputs used by the metric engine
    Column("feedback_score", Integer),
    UniqueConstraint("run_id", "scenario_id", name="uq_exec_run_scenario"),
)
Index("ix_exec_run", scenario_executions.c.run_id)
Index("ix_exec_tenant_condition", scenario_executions.c.tenant_id, scenario_executions.c.condition)

audit_events = Table(
    "audit_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tenant_id", String(64), nullable=False),
    Column("actor", String(64), nullable=False),
    Column("action", String(64), nullable=False),
    Column("target", String(120)),
    Column("detail", JSONX),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
)
Index("ix_audit_tenant_created", audit_events.c.tenant_id, audit_events.c.created_at)


def make_engine(url: str) -> Engine:
    kw: dict = {"future": True}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False}
    eng = create_engine(url, **kw)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _fk(dbapi, _):  # enforce foreign keys on SQLite
            dbapi.execute("PRAGMA foreign_keys=ON")
    return eng
