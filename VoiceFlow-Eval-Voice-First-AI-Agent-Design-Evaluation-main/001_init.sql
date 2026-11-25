-- Generated from app/db.py (PostgreSQL). Apply once: psql "$DATABASE_URL" -f migrations/001_init.sql


CREATE TABLE audit_events (
	id SERIAL NOT NULL, 
	tenant_id VARCHAR(64) NOT NULL, 
	actor VARCHAR(64) NOT NULL, 
	action VARCHAR(64) NOT NULL, 
	target VARCHAR(120), 
	detail JSONB, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
)

;

CREATE INDEX ix_audit_tenant_created ON audit_events (tenant_id, created_at);


CREATE TABLE prompt_versions (
	id SERIAL NOT NULL, 
	tenant_id VARCHAR(64) NOT NULL, 
	name VARCHAR(120) NOT NULL, 
	version INTEGER NOT NULL, 
	strategy VARCHAR(32) NOT NULL, 
	system_prompt TEXT NOT NULL, 
	params JSONB NOT NULL, 
	note VARCHAR(255), 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_prompt_version UNIQUE (tenant_id, name, version)
)

;

CREATE INDEX ix_prompt_tenant_name ON prompt_versions (tenant_id, name);


CREATE TABLE eval_runs (
	id VARCHAR(36) NOT NULL, 
	tenant_id VARCHAR(64) NOT NULL, 
	prompt_version_id INTEGER NOT NULL, 
	baseline_run_id VARCHAR(36), 
	suite VARCHAR(64) NOT NULL, 
	seed INTEGER NOT NULL, 
	config JSONB NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	provider VARCHAR(64), 
	idempotency_key VARCHAR(120), 
	metrics JSONB, 
	comparison JSONB, 
	error TEXT, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	started_at TIMESTAMP WITH TIME ZONE, 
	finished_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_run_idempotency UNIQUE (tenant_id, idempotency_key), 
	FOREIGN KEY(prompt_version_id) REFERENCES prompt_versions (id), 
	FOREIGN KEY(baseline_run_id) REFERENCES eval_runs (id) ON DELETE SET NULL
)

;

CREATE INDEX ix_runs_tenant_created ON eval_runs (tenant_id, created_at);


CREATE TABLE scenario_executions (
	id SERIAL NOT NULL, 
	run_id VARCHAR(36) NOT NULL, 
	tenant_id VARCHAR(64) NOT NULL, 
	scenario_id VARCHAR(64) NOT NULL, 
	condition VARCHAR(64) NOT NULL, 
	severity VARCHAR(16) NOT NULL, 
	passed BOOLEAN NOT NULL, 
	final_state VARCHAR(16) NOT NULL, 
	total_latency_ms FLOAT, 
	fields JSONB, 
	transcript JSONB, 
	failures JSONB, 
	scoring JSONB, 
	feedback_score INTEGER, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_exec_run_scenario UNIQUE (run_id, scenario_id), 
	FOREIGN KEY(run_id) REFERENCES eval_runs (id) ON DELETE CASCADE
)

;

CREATE INDEX ix_exec_run ON scenario_executions (run_id);

CREATE INDEX ix_exec_tenant_condition ON scenario_executions (tenant_id, condition);
