"""Pure metric functions. Every rate reports its numerator and denominator.

Not measured here (reported as null): user-feedback scores need real user ratings
(see the feedback endpoint); simulated runs never fabricate them.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..agent.providers import INTENTS

DEFAULT_THRESHOLDS: dict[str, float] = {
    "task_completion_rate": 0.90, "field_extraction_accuracy": 0.90, "booking_success_rate": 0.90,
    "structured_output_validity": 0.90, "handoff_f1": 0.90, "recovery_success_rate": 0.90,
    "intent_macro_f1": 0.80, "duplicate_booking_rate_max": 0.0, "p95_turn_latency_ms_max": 2000.0,
}
DEFAULT_REGRESSION = {"max_regression_rate": 0.05, "max_metric_drop": 0.05, "fail_on_high_severity_regression": 1.0}


def rate(num: int, den: int) -> dict[str, Any]:
    return {"value": (num / den) if den else None, "numerator": num, "denominator": den}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def latency_stats(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95),
            "max": max(values) if values else None, "mean": round(sum(values) / len(values), 3) if values else None}


def intent_metrics(pairs: list[tuple[str, str | None]]) -> dict[str, Any]:
    """Confusion matrix and per-class P/R/F1 over turns with a ground-truth label and a prediction."""
    scored = [(t, p) for t, p in pairs if p is not None]
    labels = [i for i in INTENTS if any(t == i or p == i for t, p in scored)]
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in scored:
        matrix[t][p] += 1
    per_class: dict[str, Any] = {}
    for c in labels:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in labels if t != c)
        fn = sum(matrix[c][p] for p in labels if p != c)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if (tp + fp + fn) else None)
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None and v["support"] > 0]
    return {"labelled_turns": len(pairs), "scored_turns": len(scored), "unscored_llm_unavailable": len(pairs) - len(scored),
            "accuracy": rate(sum(matrix[c][c] for c in labels), len(scored)), "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "per_class": per_class, "confusion_matrix": matrix}


def compute_metrics(execs: list[dict[str, Any]], thresholds: dict[str, float] | None = None,
                    feedback_scores: list[int] | None = None) -> dict[str, Any]:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    n = len(execs)
    booked_expected = [e for e in execs if e["expected_booking"]]
    booked_actual = [e for e in execs if e["booked"]]
    tp = sum(1 for e in execs if e["handoff"] and e["expected_handoff"])
    fp = sum(1 for e in execs if e["handoff"] and not e["expected_handoff"])
    fn = sum(1 for e in execs if not e["handoff"] and e["expected_handoff"])
    h_prec = tp / (tp + fp) if tp + fp else None
    h_rec = tp / (tp + fn) if tp + fn else None
    h_f1 = 2 * h_prec * h_rec / (h_prec + h_rec) if h_prec and h_rec else (0.0 if tp + fp + fn else None)
    exp_h = [e for e in execs if e["expected_handoff"] and e["expected_handoff_reason"]]
    faults = [e for e in execs if e["fault_injected"]]
    turn_lat = [x for e in execs for x in e["turn_latencies_ms"]]
    pairs = [p for e in execs for p in e["intent_pairs"]]
    fail_codes = Counter(f["code"] for e in execs for f in e["failures"])
    by_cond: dict[str, dict[str, int]] = {}
    for e in execs:
        c = by_cond.setdefault(e["condition"], {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += int(e["passed"])
    m: dict[str, Any] = {
        "scenarios": n,
        "task_completion_rate": rate(sum(e["passed"] for e in execs), n),
        "field_extraction_accuracy": rate(sum(e["field_correct"] for e in execs), sum(e["field_expected"] for e in execs)),
        "booking_success_rate": rate(sum(1 for e in booked_expected if e["booked"]), len(booked_expected)),
        "duplicate_booking_rate": rate(sum(1 for e in booked_actual if e["duplicate_bookings"] > 0), len(booked_actual)),
        "duplicate_deliveries_absorbed": sum(e["absorbed_duplicates"] for e in execs),
        "structured_output_validity": rate(sum(e["llm_valid"] for e in execs), sum(e["llm_calls"] for e in execs)),
        "handoff_precision": h_prec, "handoff_recall": h_rec, "handoff_f1": h_f1,
        "handoff_reason_accuracy": rate(sum(1 for e in exp_h if e["handoff_reason"] == e["expected_handoff_reason"]), len(exp_h)),
        "recovery_success_rate": rate(sum(e["passed"] for e in faults), len(faults)),
        "intent": intent_metrics(pairs),
        "latency": {"turn_ms": latency_stats(turn_lat), "end_to_end_ms": latency_stats([e["total_latency_ms"] for e in execs])},
        "user_feedback": ({"mean": sum(feedback_scores) / len(feedback_scores), "count": len(feedback_scores)}
                          if feedback_scores else {"mean": None, "count": 0, "note": "not measured"}),
        "failures_by_code": dict(fail_codes),
        "failures_by_category": dict(Counter(f["category"] for e in execs for f in e["failures"])),
        "pass_by_condition": by_cond,
        "provider": sorted({e["provider"] for e in execs}), "scheduler": sorted({e["scheduler"] for e in execs}),
        "data_label": "MOCK_ADAPTERS" if any("mock" in e["provider"] for e in execs) else "LIVE",
    }
    m["gates"] = evaluate_gates(m, th)
    m["thresholds"] = th
    return m


def evaluate_gates(m: dict[str, Any], th: dict[str, float]) -> dict[str, Any]:
    def at_least(v, t):
        return None if v is None else v >= t
    p95 = m["latency"]["turn_ms"]["p95"]
    checks = {
        "task_completion_rate": at_least(m["task_completion_rate"]["value"], th["task_completion_rate"]),
        "field_extraction_accuracy": at_least(m["field_extraction_accuracy"]["value"], th["field_extraction_accuracy"]),
        "booking_success_rate": at_least(m["booking_success_rate"]["value"], th["booking_success_rate"]),
        "structured_output_validity": at_least(m["structured_output_validity"]["value"], th["structured_output_validity"]),
        "handoff_f1": at_least(m["handoff_f1"], th["handoff_f1"]),
        "recovery_success_rate": at_least(m["recovery_success_rate"]["value"], th["recovery_success_rate"]),
        "intent_macro_f1": at_least(m["intent"]["macro_f1"], th["intent_macro_f1"]),
        "duplicate_booking_rate": (None if m["duplicate_booking_rate"]["value"] is None
                                   else m["duplicate_booking_rate"]["value"] <= th["duplicate_booking_rate_max"]),
        "p95_turn_latency_ms": None if p95 is None else p95 <= th["p95_turn_latency_ms_max"],
    }
    failed = [k for k, v in checks.items() if v is False]
    return {"checks": checks, "failed": failed, "passed": not failed}


SCALAR_KEYS = ["task_completion_rate", "field_extraction_accuracy", "booking_success_rate",
               "structured_output_validity", "recovery_success_rate"]


def compare_runs(base: list[dict[str, Any]], cand: list[dict[str, Any]], base_m: dict[str, Any],
                 cand_m: dict[str, Any], reg: dict[str, float] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_REGRESSION, **(reg or {})}
    b = {e["scenario_id"]: e for e in base}
    c = {e["scenario_id"]: e for e in cand}
    common = sorted(set(b) & set(c))
    regressions = [s for s in common if b[s]["passed"] and not c[s]["passed"]]
    improvements = [s for s in common if not b[s]["passed"] and c[s]["passed"]]
    base_pass = sum(1 for s in common if b[s]["passed"])
    reg_rate = rate(len(regressions), base_pass)
    deltas: dict[str, Any] = {}
    flagged: list[str] = []
    for k in SCALAR_KEYS:
        bv, cv = base_m[k]["value"], cand_m[k]["value"]
        d = None if bv is None or cv is None else cv - bv
        deltas[k] = {"baseline": bv, "candidate": cv, "delta": d}
        if d is not None and d < -cfg["max_metric_drop"]:
            flagged.append(k)
    for k in ("handoff_f1",):
        bv, cv = base_m[k], cand_m[k]
        d = None if bv is None or cv is None else cv - bv
        deltas[k] = {"baseline": bv, "candidate": cv, "delta": d}
        if d is not None and d < -cfg["max_metric_drop"]:
            flagged.append(k)
    bi, ci = base_m["intent"]["macro_f1"], cand_m["intent"]["macro_f1"]
    deltas["intent_macro_f1"] = {"baseline": bi, "candidate": ci, "delta": None if bi is None or ci is None else ci - bi}
    exceeded = reg_rate["value"] is not None and reg_rate["value"] > cfg["max_regression_rate"]
    high = [s for s in regressions if c[s]["severity"] == "high"]
    high_flag = bool(high) and bool(cfg["fail_on_high_severity_regression"])
    return {"high_severity_regressions": high, "scenarios_compared": len(common), "regressions": regressions, "improvements": improvements,
            "regression_rate": reg_rate, "metric_deltas": deltas, "metrics_dropped_beyond_threshold": flagged,
            "regression_detected": bool(exceeded or flagged or high_flag), "regression_config": cfg}
