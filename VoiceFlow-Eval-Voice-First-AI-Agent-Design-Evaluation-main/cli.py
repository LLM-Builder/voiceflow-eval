"""Evaluation command: ``python -m app.evaluation.cli --prompt prompts/candidate_v2.json``.

Executes the scripted scenarios, prints a scorecard and writes JSON/CSV reports.
With the mock provider, numbers describe the pipeline and state machine, NOT a real LLM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..agent.providers import PromptConfig
from .metrics import compare_runs, compute_metrics
from .runner import run_suite
from .scenarios import SCENARIOS


def load_prompt(path: str) -> PromptConfig:
    d = json.loads(Path(path).read_text())
    return PromptConfig(name=d["name"], version=d.get("version", 1), strategy=d["strategy"],
                        system_prompt=d.get("system_prompt", ""), params=d.get("params", {}))


def write_reports(out: Path, label: str, execs: list[dict], metrics: dict, prompt: PromptConfig) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}.json").write_text(json.dumps(
        {"prompt": prompt.__dict__, "metrics": metrics, "executions": execs}, indent=2, default=str))
    with (out / f"{label}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "condition", "severity", "passed", "final_state", "expected_state",
                    "handoff_reason", "failure_codes", "total_latency_ms"])
        for e in execs:
            w.writerow([e["scenario_id"], e["condition"], e["severity"], e["passed"], e["final_state"],
                        e["expected_state"], e["handoff_reason"] or "",
                        ";".join(x["code"] for x in e["failures"]), e["total_latency_ms"]])


def fmt(r) -> str:
    v = r["value"] if isinstance(r, dict) else r
    if isinstance(r, dict):
        return "n/a" if v is None else f"{v:.3f} ({r['numerator']}/{r['denominator']})"
    return "n/a" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--baseline", help="baseline prompt json to compare against")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    prompt = load_prompt(a.prompt)
    execs = run_suite(prompt, SCENARIOS, a.seed)
    m = compute_metrics(execs)
    write_reports(out, f"{prompt.name}-v{prompt.version}", execs, m, prompt)
    print(f"== {prompt.name} v{prompt.version} [{prompt.strategy}] data={m['data_label']} scenarios={m['scenarios']}")
    for k in ("task_completion_rate", "field_extraction_accuracy", "booking_success_rate", "duplicate_booking_rate",
              "structured_output_validity", "handoff_reason_accuracy", "recovery_success_rate"):
        print(f"  {k:30s} {fmt(m[k])}")
    print(f"  {'handoff_f1':30s} {fmt(m['handoff_f1'])}")
    print(f"  {'intent_macro_f1':30s} {fmt(m['intent']['macro_f1'])}")
    print(f"  turn latency ms p50/p95   {m['latency']['turn_ms']['p50']}/{m['latency']['turn_ms']['p95']}")
    print(f"  failures: {m['failures_by_code']}")
    print(f"  gates passed: {m['gates']['passed']} failed={m['gates']['failed']}")
    rc = 0 if m["gates"]["passed"] else 1
    if a.baseline:
        bp = load_prompt(a.baseline)
        bexecs = run_suite(bp, SCENARIOS, a.seed)
        bm = compute_metrics(bexecs)
        write_reports(out, f"{bp.name}-v{bp.version}", bexecs, bm, bp)
        cmp = compare_runs(bexecs, execs, bm, m)
        (out / "comparison.json").write_text(json.dumps(cmp, indent=2))
        print(f"== vs baseline {bp.name} v{bp.version}: regressions={cmp['regressions']} "
              f"improvements={len(cmp['improvements'])} regression_detected={cmp['regression_detected']}")
        rc = rc or (2 if cmp["regression_detected"] else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
