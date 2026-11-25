import pytest

from app.agent.providers import PromptConfig
from app.evaluation.metrics import (
    compare_runs,
    compute_metrics,
    intent_metrics,
    percentile,
    rate,
)
from app.evaluation.runner import redact_text, run_suite
from app.evaluation.scenarios import CONDITIONS, SCENARIOS

FULL = PromptConfig(params={"mock_profile": "full"})
WEAK = PromptConfig(params={"mock_profile": "weak"})


def test_baseline_shape_36_scenarios_nine_conditions():
    assert len(SCENARIOS) == 36 and len(CONDITIONS) == 9
    assert all(sum(1 for s in SCENARIOS if s.condition == c) == 4 for c in CONDITIONS)


def test_rate_reports_denominator_and_handles_zero():
    assert rate(3, 4) == {"value": 0.75, "numerator": 3, "denominator": 4}
    assert rate(0, 0)["value"] is None


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 95) is None


def test_intent_metrics_confusion_matrix_and_f1():
    m = intent_metrics([("a", "a")] * 0 + [("book_demo", "book_demo"), ("book_demo", "unclear"),
                                           ("unclear", "unclear"), ("goodbye", None)])
    assert m["scored_turns"] == 3 and m["unscored_llm_unavailable"] == 1
    assert m["confusion_matrix"]["book_demo"]["unclear"] == 1
    assert m["per_class"]["book_demo"]["precision"] == 1.0 and m["per_class"]["book_demo"]["recall"] == 0.5


@pytest.fixture(scope="module")
def runs():
    return run_suite(FULL, seed=1), run_suite(WEAK, seed=1)


def test_suite_is_deterministic_for_the_same_seed():
    a = [(e["scenario_id"], e["passed"], e["final_state"]) for e in run_suite(FULL, seed=3)]
    b = [(e["scenario_id"], e["passed"], e["final_state"]) for e in run_suite(FULL, seed=3)]
    assert a == b


def test_metrics_labelled_mock_and_feedback_not_fabricated(runs):
    m = compute_metrics(runs[0])
    assert m["data_label"] == "MOCK_ADAPTERS" and m["user_feedback"]["mean"] is None
    assert m["task_completion_rate"]["denominator"] == 36


def test_no_false_booking_claims_or_duplicates_in_any_run(runs):
    for execs in runs:
        for e in execs:
            assert not any(f["code"] in ("FALSE_BOOKING_CLAIM", "DUPLICATE_BOOKING") for f in e["failures"])


def test_unconfirmed_booking_scenarios_have_no_booking(runs):
    e = next(x for x in runs[0] if x["scenario_id"] == "booking-03")
    assert e["booked"] is False and e["handoff_reason"] == "booking_unconfirmed" and e["passed"]


def test_weak_profile_is_measurably_worse_and_regression_is_detected(runs):
    full, weak = runs
    fm, wm = compute_metrics(full), compute_metrics(weak)
    assert wm["intent"]["macro_f1"] < fm["intent"]["macro_f1"]
    c = compare_runs(full, weak, fm, wm)  # candidate = weak => regression
    assert "handoff-02" in c["regressions"] and c["regression_detected"]


def test_no_regression_when_candidate_equals_baseline(runs):
    fm = compute_metrics(runs[0])
    c = compare_runs(runs[0], runs[0], fm, fm)
    assert not c["regressions"] and not c["regression_detected"]


def test_configurable_thresholds_change_gate_outcome(runs):
    strict = compute_metrics(runs[1], {"intent_macro_f1": 0.99})
    lax = compute_metrics(runs[1], {"intent_macro_f1": 0.1, "task_completion_rate": 0.1,
                                    "handoff_f1": 0.1, "recovery_success_rate": 0.1})
    assert "intent_macro_f1" in strict["gates"]["failed"] and "intent_macro_f1" not in lax["gates"]["failed"]


def test_failure_taxonomy_categories_and_integration_split():
    from app.evaluation.runner import TAXONOMY
    assert TAXONOMY["INTEGRATION_ERROR"][0] == "integration" and TAXONOMY["FIELD_MISMATCH"][0] == "agent"


def test_cancel_callback_stops_run_between_scenarios():
    seen = []
    out = run_suite(FULL, should_cancel=lambda: len(seen) >= 3, on_result=seen.append)
    assert len(out) == 3


def test_redaction_removes_emails_from_transcripts():
    assert "[email]" in redact_text("reach me at a.b@c.com") and "@" not in redact_text("a.b@c.com")
    r = run_suite(FULL, [SCENARIOS[0]], redact=True)[0]
    assert all("@" not in (t["user"] or "") for t in r["transcript"]) and r["fields"]["email"] == "[redacted]"
