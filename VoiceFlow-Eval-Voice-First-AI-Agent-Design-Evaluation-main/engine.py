"""Conversation state machine for the lead-qualification / demo-booking agent.

Conversation logic is independent of the LLM provider and the scheduler: both are
injected through the interfaces in ``providers.py`` and ``scheduling.py``.
Replies are templated and deterministic; the LLM is used only for structured
analysis (intent + field extraction), which is validated before use.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from .providers import LLMProvider, LLMTimeout, PromptConfig
from .scheduling import SchedulerError, SchedulerTimeout, SchedulingAdapter, SlotTaken


class State(str, Enum):
    QUALIFYING = "QUALIFYING"
    CLARIFYING = "CLARIFYING"
    SCHEDULING = "SCHEDULING"
    CONFIRMED = "CONFIRMED"
    HANDOFF = "HANDOFF"
    ENDED = "ENDED"


TERMINAL = {State.CONFIRMED, State.HANDOFF, State.ENDED}
S = State
ALLOWED: dict[State, set[State]] = {
    S.QUALIFYING: {S.QUALIFYING, S.CLARIFYING, S.SCHEDULING, S.HANDOFF, S.ENDED},
    S.CLARIFYING: {S.CLARIFYING, S.QUALIFYING, S.SCHEDULING, S.HANDOFF, S.ENDED},
    S.SCHEDULING: {S.SCHEDULING, S.CONFIRMED, S.HANDOFF, S.ENDED},
    S.CONFIRMED: set(), S.HANDOFF: set(), S.ENDED: set(),
}
FLOW_EDGES = [  # exposed to the Agent Flow Designer
    {"from": a.value, "to": b.value} for a, bs in ALLOWED.items() for b in sorted(bs, key=lambda x: x.value)
]

REQUIRED = ["name", "email", "company", "team_size"]
ASK = {
    "name": "Could I get your name?",
    "email": "What's the best email address for the calendar invite?",
    "company": "Which company are you with?",
    "team_size": "Roughly how many people are on your team?",
}
FAQ = {
    "pricing": "Pricing depends on team size and needs, and the demo will cover it.",
    "security": "Security details are covered in the demo and in follow-up documentation.",
    "default": "That's a great question, and the demo is the best place to cover it.",
}
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,59}$")
COMPANY_RE = re.compile(r"^[\w][\w &.,'\-]{1,79}$")
VAGUE_COMPANY = {"hmm", "not sure", "idk", "none", "n/a", "skip", "no idea", "dunno"}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


class Analysis(BaseModel):
    intent: Literal["book_demo", "provide_info", "ask_question", "human_request", "goodbye", "injection", "unclear"]
    fields: dict[str, str] = {}


def validate_field(name: str, raw: str) -> tuple[bool, str, str]:
    """Return (ok, normalised_value, error). Also guards tool arguments."""
    v = _CTRL.sub("", str(raw)).strip()[:120]
    if name == "name":
        return (True, v, "") if NAME_RE.match(v) else (False, v, "That doesn't look like a valid name.")
    if name == "email":
        return (True, v.lower(), "") if EMAIL_RE.match(v) else (False, v, "That email address doesn't look valid.")
    if name == "company":
        ok = bool(COMPANY_RE.match(v)) and v.lower() not in VAGUE_COMPANY
        return (True, v, "") if ok else (False, v, "I didn't catch a valid company name.")
    if name == "team_size":
        if re.fullmatch(r"\d{1,6}", v) and 1 <= int(v) <= 100_000:
            return True, str(int(v)), ""
        return False, v, "Please give me a team size as a positive number."
    return False, v, "Unknown field."


class AgentEngine:
    def __init__(self, llm: LLMProvider, scheduler: SchedulingAdapter, prompt: PromptConfig | None = None, *,
                 session_id: str = "session", max_turns: int = 20, max_asks: int = 4, llm_retries: int = 1):
        self.llm, self.scheduler = llm, scheduler
        self.prompt = prompt or PromptConfig()
        self.session_id, self.max_turns, self.max_asks, self.llm_retries = session_id, max_turns, max_asks, llm_retries
        self.state = State.QUALIFYING
        self.fields: dict[str, str] = {}
        self.asked: dict[str, int] = {}
        self.slot_asks = 0
        self.offered: list[str] = []
        self.booking: dict[str, Any] | None = None
        self.handoff_reason: str | None = None
        self.handoff_summary: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.llm_calls = 0
        self.llm_valid = 0
        self.llm_fail_streak = 0
        self._t: dict[str, Any] = {}

    # ---- helpers ----------------------------------------------------------------
    def _go(self, new: State) -> None:
        if new not in ALLOWED[self.state]:
            raise RuntimeError(f"illegal transition {self.state} -> {new}")
        self.state = new

    def _pending(self) -> str | None:
        return next((f for f in REQUIRED if f not in self.fields), None)

    def _ask(self, field: str, prefix: str = "") -> str:
        self.asked[field] = self.asked.get(field, 0) + 1
        if self.asked[field] > self.max_asks:
            return self._handoff("clarification_limit")
        return f"{prefix} {ASK[field]}".strip()

    def _prompt_pending(self) -> str:
        if self.state == State.SCHEDULING:
            return self._offer_text()
        p = self._pending()
        return self._ask(p) if p else self._begin_scheduling()

    def _offer_text(self) -> str:
        opts = "; ".join(f"{i + 1}) {s.replace('T', ' ')}" for i, s in enumerate(self.offered))
        return f"I have these times available: {opts}. Which works best?"

    def _tool(self, name: str, fn, *a):
        start = time.perf_counter()
        rec = {"tool": name, "ok": True, "error": None}
        try:
            return fn(*a)
        except Exception as e:  # recorded then re-raised for the caller's recovery logic
            rec.update(ok=False, error=type(e).__name__)
            raise
        finally:
            rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
            self._t["tool_calls"].append(rec)

    # ---- LLM analysis -------------------------------------------------------------
    def _analyze(self, text: str) -> Analysis | None:
        ctx = {"turn": self._t["idx"], "awaiting": "slot" if self.state == State.SCHEDULING else self._pending(),
               "state": self.state.value}
        for attempt in range(self.llm_retries + 1):  # LLM analysis is read-only: safe to retry
            ctx["attempt"] = attempt
            self.llm_calls += 1
            try:
                raw = self.llm.analyze(text, ctx, self.prompt)
                result = Analysis.model_validate(json.loads(raw))
                self.llm_valid += 1
                return result
            except LLMTimeout:
                err = "llm_timeout"
            except (json.JSONDecodeError, ValidationError, TypeError):
                err = "llm_malformed_output"
            self.errors.append({"turn": ctx["turn"], "type": err, "attempt": attempt})
            self._t["validation_errors"].append({"field": "llm_output", "error": err})
        return None

    # ---- public API ---------------------------------------------------------------
    def start(self) -> str:
        msg = "Hi, thanks for calling! I can help you set up a product demo. " + self._ask("name")
        self.turns.append({"idx": 0, "user": None, "assistant": msg, "intent": None, "state_before": None,
                           "state_after": self.state.value, "tool_calls": [], "validation_errors": [],
                           "fields_extracted": {}, "latency_ms": 0.0, "tags": []})
        return msg

    def step(self, text: str) -> str:
        if self.state in TERMINAL:
            return "This conversation has ended."
        began = time.perf_counter()
        self._t = {"idx": len(self.turns), "user": text, "assistant": "", "intent": None,
                   "state_before": self.state.value, "tool_calls": [], "validation_errors": [],
                   "fields_extracted": {}, "tags": []}
        self.turns.append(self._t)
        reply = self._handle(text)
        if self.state not in TERMINAL and len(self.turns) > self.max_turns:
            reply = self._handoff("max_turns")
        self._t["assistant"] = reply
        self._t["state_after"] = self.state.value
        self._t["latency_ms"] = round((time.perf_counter() - began) * 1000, 3)
        return reply

    # ---- turn handling ------------------------------------------------------------
    def _handle(self, text: str) -> str:
        a = self._analyze(text)
        if a is None:
            self.llm_fail_streak += 1
            self._t["tags"].append("llm_failure")
            if self.llm_fail_streak >= 2:
                return self._handoff("llm_unavailable")
            return "Sorry, I had trouble processing that. Could you rephrase?"
        self.llm_fail_streak = 0
        self._t["intent"] = a.intent
        if a.intent == "injection":
            self._t["tags"].append("injection_refused")
            return "I can't change how I work or share internal details, but I'm happy to help with a demo. " + self._prompt_pending()
        if a.intent == "human_request":
            return self._handoff("user_requested")
        if a.intent == "goodbye":
            self._go(State.ENDED)
            return "No problem, thanks for your time. Goodbye!"
        if a.intent == "ask_question":
            low = text.lower()
            ans = FAQ["pricing"] if re.search(r"pric|cost|how much", low) else FAQ["security"] if "secur" in low else FAQ["default"]
            return ans + " " + self._prompt_pending()
        if self.state == State.SCHEDULING:
            return self._scheduling(a)
        return self._qualify(a)

    def _qualify(self, a: Analysis) -> str:
        bad: list[tuple[str, str]] = []
        for k, raw in a.fields.items():
            if k not in REQUIRED:
                continue
            ok, val, err = validate_field(k, raw)
            if ok:
                self.fields[k] = val
                self.asked[k] = 0
                self._t["fields_extracted"][k] = val
            else:
                self._t["validation_errors"].append({"field": k, "error": err})
                bad.append((k, err))
        if bad:
            self._go(State.CLARIFYING)
            return self._ask(bad[0][0], bad[0][1])
        p = self._pending()
        if p:
            self._go(State.QUALIFYING)
            prefix = "Thanks." if self._t["fields_extracted"] else "Sorry, I didn't catch that."
            return self._ask(p, prefix)
        return self._begin_scheduling()

    def _begin_scheduling(self) -> str:
        slots: list[str] = []
        for _ in range(2):  # availability is a read: bounded retry is safe
            try:
                slots = self._tool("get_availability", self.scheduler.get_availability)
                break
            except SchedulerError:
                slots = []
                self._t["tags"].append("scheduler_error")
        else:
            return self._handoff("scheduler_unavailable")
        if not slots:
            return self._handoff("no_availability")
        self.offered = slots
        if self.state != State.SCHEDULING:
            self._go(State.SCHEDULING)
        return f"Thanks, {self.fields['name'].split()[0]}. " + self._offer_text()

    def _resolve_slot(self, raw: str | None) -> str | None:
        if not raw:
            return None
        if raw.startswith("#") and raw[1:].isdigit():
            i = int(raw[1:]) - 1
            return self.offered[i] if 0 <= i < len(self.offered) else None
        matches = [s for s in self.offered if s.endswith("T" + raw)]
        return matches[0] if len(matches) == 1 else None

    def _scheduling(self, a: Analysis) -> str:
        slot = self._resolve_slot(a.fields.get("slot"))  # never pass raw user text to the tool
        if slot is None:
            self.slot_asks += 1
            if self.slot_asks > self.max_asks:
                return self._handoff("clarification_limit")
            return "Sorry, I didn't catch which time. " + self._offer_text()
        lead = {k: self.fields[k] for k in REQUIRED}
        key = hashlib.sha256(f"{self.session_id}|{lead['email']}|{slot}".encode()).hexdigest()[:24]
        try:
            b = self._tool("book", self.scheduler.book, key, slot, lead)
        except SchedulerTimeout:
            self._t["tags"].append("booking_timeout")
            found = self._tool("lookup", self.scheduler.lookup, key)  # safe read, never blind re-book
            if found is None:
                return self._handoff("booking_unconfirmed")
            b = found
        except SlotTaken:
            self.offered = [s for s in self.offered if s != slot]
            try:
                self.offered = self._tool("get_availability", self.scheduler.get_availability)
            except SchedulerError:
                return self._handoff("scheduler_unavailable")
            if not self.offered:
                return self._handoff("no_availability")
            return "That time was just taken. " + self._offer_text()
        except SchedulerError:
            return self._handoff("booking_failed")
        self.booking = {"confirmation_id": b.confirmation_id, "slot": b.slot, "key": b.key}
        self._go(State.CONFIRMED)
        return f"You're booked for {b.slot.replace('T', ' ')}. Your confirmation ID is {b.confirmation_id}."

    def _handoff(self, reason: str) -> str:
        self.handoff_reason = reason
        self.handoff_summary = {
            "reason": reason, "collected_fields": dict(self.fields),
            "missing_fields": [f for f in REQUIRED if f not in self.fields],
            "turns": len(self.turns) - 1, "state_at_handoff": self.state.value,
            "last_user_messages": [t["user"] for t in self.turns if t.get("user")][-3:],
        }
        if self.state != State.HANDOFF:
            self._go(State.HANDOFF)
        return "I'm connecting you with a teammate who can help. I haven't scheduled anything yet."

    def result(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "final_state": self.state.value, "fields": dict(self.fields),
                "booking": self.booking, "handoff_reason": self.handoff_reason,
                "handoff_summary": self.handoff_summary, "transcript": self.turns, "errors": self.errors,
                "llm_calls": self.llm_calls, "llm_valid": self.llm_valid}
