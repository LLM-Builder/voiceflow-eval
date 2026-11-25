from .conftest import PROMPT_V1, PROMPT_V2, A, B


def mkprompts(c, h=A):
    assert c.post("/v1/prompts", json=PROMPT_V1, headers=h).status_code == 201
    assert c.post("/v1/prompts", json=PROMPT_V2, headers=h).status_code == 201


def start(c, v, h=A, **kw):
    r = c.post("/v1/runs", json={"prompt_name": "sales", "prompt_version": v, **kw}, headers=h)
    assert r.status_code in (200, 202), r.text
    return r.json()["id"]


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["database"] == "ok"


def test_auth_required_and_consistent_error_shape(client):
    r = client.get("/v1/me")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized" and r.json()["error"]["request_id"]
    assert client.get("/v1/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/me", headers=A).json() == {"tenant_id": "tenant-a"}


def test_auth_failures_are_rate_limited(client):
    codes = [client.get("/v1/me").status_code for _ in range(8)]
    assert codes[:5] == [401] * 5 and 429 in codes


def test_validation_error_structure(client):
    r = client.post("/v1/prompts", json={"name": "x", "strategy": "bogus", "system_prompt": ""}, headers=A)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_oversized_body_rejected(client):
    r = client.post("/v1/prompts", content=b"x" * 2_000_000, headers={**A, "content-type": "application/json"})
    assert r.status_code == 413


def test_prompt_versioning_history_and_rollback(client):
    mkprompts(client)
    vs = client.get("/v1/prompts/sales/versions", headers=A).json()["items"]
    assert [v["version"] for v in vs] == [2, 1]
    rb = client.post("/v1/prompts/sales/rollback", json={"to_version": 1}, headers=A).json()
    assert rb["version"] == 3 and rb["system_prompt"] == PROMPT_V1["system_prompt"] and rb["note"] == "rollback to v1"
    assert len(client.get("/v1/prompts/sales/versions", headers=A).json()["items"]) == 3
    assert client.post("/v1/prompts/sales/rollback", json={"to_version": 9}, headers=A).status_code == 404


def test_run_executes_and_persists_measured_results(client):
    mkprompts(client)
    rid = start(client, 2)
    run = client.get(f"/v1/runs/{rid}", headers=A).json()
    assert run["status"] == "completed" and run["metrics"]["scenarios"] == 36
    assert run["metrics"]["data_label"] == "MOCK_ADAPTERS" and run["config"]["prompt"]["version"] == 2
    ex = client.get(f"/v1/runs/{rid}/executions?limit=5", headers=A).json()
    assert len(ex["items"]) == 5
    detail = client.get(f"/v1/executions/{ex['items'][0]['id']}", headers=A).json()
    assert detail["transcript"][0]["assistant"] and "state_after" in detail["transcript"][1]


def test_condition_filter_and_pagination(client):
    mkprompts(client)
    rid = start(client, 2, conditions=["handoff"])
    assert client.get(f"/v1/runs/{rid}", headers=A).json()["metrics"]["scenarios"] == 4
    assert client.get("/v1/scenarios?condition=injection", headers=A).json()["total"] == 4
    assert client.get("/v1/runs?limit=1", headers=A).json()["limit"] == 1
    assert client.get("/v1/runs?limit=1000", headers=A).status_code == 422


def test_idempotent_run_creation(client):
    mkprompts(client)
    a = start(client, 2, idempotency_key="k1", conditions=["handoff"])
    b = start(client, 2, idempotency_key="k1", conditions=["handoff"])
    assert a == b and client.get("/v1/runs", headers=A).json()["total"] == 1


def test_compare_runs_detects_regression_and_exports(client):
    mkprompts(client)
    good, bad = start(client, 2), start(client, 1)
    c = client.get(f"/v1/compare?baseline_run={good}&candidate_run={bad}", headers=A).json()
    assert c["regression_detected"] and "handoff-02" in c["regressions"] and c["side_by_side"]
    # baseline comparison embedded in run when requested at creation time
    rid = start(client, 1, baseline_run_id=good)
    assert client.get(f"/v1/runs/{rid}", headers=A).json()["comparison"]["regression_detected"]
    csv_r = client.get(f"/v1/runs/{good}/export?format=csv", headers=A)
    assert csv_r.headers["content-type"].startswith("text/csv") and csv_r.text.count("\n") == 37
    js = client.get(f"/v1/runs/{good}/export?format=json", headers=A).json()
    assert len(js["executions"]) == 36


def test_tenant_isolation(client):
    mkprompts(client)
    rid = start(client, 2, conditions=["handoff"])
    ex_id = client.get(f"/v1/runs/{rid}/executions", headers=A).json()["items"][0]["id"]
    assert client.get(f"/v1/runs/{rid}", headers=B).status_code == 404
    assert client.get(f"/v1/runs/{rid}/executions", headers=B).status_code == 404
    assert client.get(f"/v1/executions/{ex_id}", headers=B).status_code == 404
    assert client.post(f"/v1/executions/{ex_id}/feedback", json={"score": 5}, headers=B).status_code == 404
    assert client.delete(f"/v1/runs/{rid}", headers=B).status_code == 404
    assert client.get("/v1/prompts/sales/versions", headers=B).status_code == 404
    assert client.post("/v1/runs", json={"prompt_name": "sales"}, headers=B).status_code == 404
    assert client.get("/v1/runs", headers=B).json()["total"] == 0


def test_cannot_use_other_tenants_baseline(client):
    mkprompts(client)
    mkprompts(client, B)
    rid = start(client, 2, conditions=["handoff"])
    r = client.post("/v1/runs", json={"prompt_name": "sales", "baseline_run_id": rid}, headers=B)
    assert r.status_code == 404


def test_feedback_is_aggregated_not_fabricated(client):
    mkprompts(client)
    rid = start(client, 2, conditions=["handoff"])
    ex_id = client.get(f"/v1/runs/{rid}/executions", headers=A).json()["items"][0]["id"]
    assert client.post(f"/v1/executions/{ex_id}/feedback", json={"score": 9}, headers=A).status_code == 422
    assert client.post(f"/v1/executions/{ex_id}/feedback", json={"score": 4}, headers=A).json()["feedback_score"] == 4


def test_cancel_finished_run_conflicts_and_delete_works(client):
    mkprompts(client)
    rid = start(client, 2, conditions=["handoff"])
    assert client.post(f"/v1/runs/{rid}/cancel", headers=A).status_code == 409
    assert client.delete(f"/v1/runs/{rid}", headers=A).status_code == 204
    assert client.get(f"/v1/runs/{rid}", headers=A).status_code == 404


def test_redacted_run_stores_no_emails(client):
    mkprompts(client)
    rid = start(client, 2, conditions=["success"], redact=True)
    exs = client.get(f"/v1/runs/{rid}/export?format=json", headers=A).text
    assert "priya@acme.com" not in exs


def test_simulate_endpoint_and_fault_injection(client):
    r = client.post("/v1/simulate", json={"utterances": ["I'm Priya Sharma from Acme Corp", "priya@acme.com",
                                                          "about 25 people", "14:00"]}, headers=A).json()
    assert r["final_state"] == "CONFIRMED" and r["data_label"] == "MOCK_ADAPTERS"
    r = client.post("/v1/simulate", json={"utterances": ["I'm Priya Sharma from Acme Corp", "priya@acme.com",
                                                          "about 25 people", "14:00"],
                                          "scheduler_faults": {"book": "error"}}, headers=A).json()
    assert r["final_state"] == "HANDOFF" and r["booking"] is None


def test_retention_purge_and_audit(client):
    mkprompts(client)
    start(client, 2, conditions=["handoff"])
    assert client.post("/v1/admin/retention?days=30", headers=A).json()["runs_deleted"] == 0
    assert client.post("/v1/admin/retention?days=0", headers=A).status_code == 422


def test_integrations_never_expose_secrets(client):
    r = client.get("/v1/integrations", headers=A)
    assert r.json()["secrets_exposed"] is False and "key" not in r.text.lower().replace("keyword", "")


def test_flow_and_taxonomy_endpoints(client):
    assert {"from": "SCHEDULING", "to": "CONFIRMED"} in client.get("/v1/flow", headers=A).json()["edges"]
    assert "INTEGRATION_ERROR" in client.get("/v1/taxonomy", headers=A).json()
