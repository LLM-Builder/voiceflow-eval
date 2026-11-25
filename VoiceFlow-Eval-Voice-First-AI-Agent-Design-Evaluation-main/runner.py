"""Runs scripted scenarios through the real engine and scores them against explicit criteria."""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from ..agent.engine import TERMINAL, AgentEngine, State
from ..agent.providers import MockLLM, PromptConfig
from ..agent.scheduling import MockScheduler
from .scenarios import SCENARIOS, Scenario

PROVIDER_REASONS = {"llm_unavailable", "scheduler_unavailable", "booking_failed", "booking_unconfirmed"}

# Documented failure taxonomy: code -> (category, definition)
TAXONOMY: dict[str, tuple[str, str]] = {
    "WRONG_TERMINAL_STATE": ("agent", "Conversation ended in a different terminal state than expected."),
    "INCOMPLETE_CONVERSATION": ("agent", "Script ran out before the conversation reached a terminal state."),
    "FIELD_MISMATCH": ("agent", "A required field's final value differs from the expected value."),
    "HANDOFF_REASON_MISMATCH": ("agent", "Handoff happened for a different reason than expected."),
    "BOOKING_MISMATCH": ("agent", "A booking exists when none was expected, or is missing when expected."),
    "FALSE_BOOKING_CLAIM": ("agent", "Assistant claimed a booking that the provider never confirmed."),
    "DUPLICATE_BOOKING": ("agent", "More than one confirmed booking exists for the same (email, slot)."),
    "FORBIDDEN_CONTENT": ("agent", "Assistant output contained a phrase the scenario forbids."),
    "INTEGRATION_ERROR": ("integration", "Unexpected handoff caused by an LLM or scheduler provider failure."),
}
_CLAIM = re.compile(r"you'?re booked|you are booked|has been booked|booking is confirmed|confirmation id", re.IGNORECASE)


def redact_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "[email]", s)
    return re.sub(r"\+?\d[\d\s().-]{7,}\d", "[phone]", s)


def run_scenario(sc: Scenario, prompt: PromptConfig, seed: int = 0, redact: bool = False) -> dict[str, Any]:
    llm = MockLLM(profile=prompt.params.get("mock_profile", "full"), faults=sc.llm_faults)
    sched = MockScheduler(faults=sc.scheduler_faults)
    eng = AgentEngine(llm, sched, prompt, session_id=f"s{seed}-{sc.id}")
    began = time.perf_counter()
    eng.start()
    pairs: list[tuple[str, str | None]] = []
    for text, label in sc.turns:
        if eng.state in TERMINAL:
            break
        eng.step(text)
        t = eng.turns[-1]
        pairs.append((label, t["intent"]))  # intent None => LLM unavailable that turn
    total_ms = round((time.perf_counter() - began) * 1000, 3)
    res = eng.result()

    failures: list[dict[str, str]] = []

    def fail(code: str, detail: str = "") -> None:
        failures.append({"code": code, "category": TAXONOMY[code][0], "detail": detail})

    if res["final_state"] not in {s.value for s in TERMINAL}:
        fail("INCOMPLETE_CONVERSATION", f"stopped in {res['final_state']}")
    elif res["final_state"] != sc.expected_state:
        if res["handoff_reason"] in PROVIDER_REASONS:
            fail("INTEGRATION_ERROR", res["handoff_reason"])
        fail("WRONG_TERMINAL_STATE", f"expected {sc.expected_state}, got {res['final_state']}")
    if sc.expected_handoff_reason and res["final_state"] == State.HANDOFF.value \
            and res["handoff_reason"] != sc.expected_handoff_reason:
        fail("HANDOFF_REASON_MISMATCH", f"expected {sc.expected_handoff_reason}, got {res['handoff_reason']}")
    for k, v in sc.expected_fields.items():
        if res["fields"].get(k) != v:
            fail("FIELD_MISMATCH", f"{k}: expected {v!r}, got {res['fields'].get(k)!r}")
    if sc.expected_booking != (res["booking"] is not None):
        fail("BOOKING_MISMATCH", f"expected_booking={sc.expected_booking}")
    replies = [t["assistant"] for t in res["transcript"]]
    if res["booking"] is None and any(_CLAIM.search(r) for r in replies):
        fail("FALSE_BOOKING_CLAIM")
    if sched.duplicate_booking_count() > 0:
        fail("DUPLICATE_BOOKING")
    for phrase in sc.forbidden_phrases:
        if any(phrase.lower() in r.lower() for r in replies):
            fail("FORBIDDEN_CONTENT", phrase)

    exp_fields_total = len(sc.expected_fields)
    field_correct = sum(1 for k, v in sc.expected_fields.items() if res["fields"].get(k) == v)
    transcript = res["transcript"]
    if redact:
        pii = ("email", "name")
        for t in transcript:
            t["user"], t["assistant"] = redact_text(t.get("user")), redact_text(t["assistant"])
            t["fields_extracted"] = {k: ("[redacted]" if k in pii else v) for k, v in t.get("fields_extracted", {}).items()}
        res["fields"] = {k: ("[redacted]" if k in pii else v) for k, v in res["fields"].items()}
        if res["handoff_summary"]:
            hs = res["handoff_summary"]
            hs["collected_fields"] = {k: ("[redacted]" if k in pii else v) for k, v in hs["collected_fields"].items()}
            hs["last_user_messages"] = [redact_text(m) for m in hs["last_user_messages"]]
    return {
        "scenario_id": sc.id, "condition": sc.condition, "severity": sc.severity,
        "passed": not failures, "failures": failures,
        "expected_state": sc.expected_state, "final_state": res["final_state"],
        "expected_handoff": sc.expected_state == "HANDOFF", "handoff": res["final_state"] == "HANDOFF",
        "expected_handoff_reason": sc.expected_handoff_reason, "handoff_reason": res["handoff_reason"],
        "handoff_summary": res["handoff_summary"],
        "expected_booking": sc.expected_booking, "booked": res["booking"] is not None, "booking": res["booking"],
        "duplicate_bookings": sched.duplicate_booking_count(), "absorbed_duplicates": sched.absorbed_duplicates,
        "fault_injected": sc.fault_injected, "fields": res["fields"],
        "field_expected": exp_fields_total, "field_correct": field_correct,
        "intent_pairs": pairs, "llm_calls": res["llm_calls"], "llm_valid": res["llm_valid"],
        "total_latency_ms": total_ms, "turn_latencies_ms": [t["latency_ms"] for t in transcript if t["user"]],
        "transcript": transcript, "provider": llm.name, "scheduler": sched.name,
    }


def run_suite(prompt: PromptConfig, scenarios: Iterable[Scenario] | None = None, seed: int = 0,
              redact: bool = False, should_cancel: Callable[[], bool] | None = None,
              on_result: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sc in scenarios or SCENARIOS:
        if should_cancel and should_cancel():
            break
        r = run_scenario(sc, prompt, seed, redact)
        out.append(r)
        if on_result:
            on_result(r)
    return out
