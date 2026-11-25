"""LLM provider interface and a deterministic, rule-based MOCK provider.

The mock is a *simulator*: results measured with it validate the pipeline and the
state machine, not the quality of any real model. Every run records the provider
name so results are labelled accordingly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMTimeout(Exception):
    pass


@dataclass
class PromptConfig:
    name: str = "default"
    version: int = 1
    strategy: str = "structured_instruction"
    system_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str

    def analyze(self, text: str, context: dict[str, Any], prompt: PromptConfig) -> str:
        """Return a raw JSON string: {"intent": ..., "fields": {...}}."""


INTENTS = ["book_demo", "provide_info", "ask_question", "human_request", "goodbye", "injection", "unclear"]

_INJECTION = re.compile(
    r"ignore (all |any |your |the )?(previous |prior |above )?(instructions|rules)|system prompt|"
    r"you are now|pretend (to be|you)|reveal your (prompt|instructions)|developer mode|"
    r"disregard (your|the|all)", re.IGNORECASE)
_HUMAN_FULL = re.compile(
    r"\b(human|real person|representative|live agent|manager|talk to (a |an )?(someone|person)|"
    r"speak (to|with) (a |an )?(someone|person))\b", re.IGNORECASE)
_HUMAN_WEAK = re.compile(r"\b(human|representative)\b", re.IGNORECASE)
_GOODBYE = re.compile(r"^\s*(bye|goodbye|no thanks|never ?mind|stop)\b|not interested", re.IGNORECASE)
_VAGUE = re.compile(
    r"^\s*(hmm+|um+|uh+|not sure|i'?m not sure|i don'?t know|idk|no idea|dunno|maybe|skip|pass|"
    r"what|ok|okay|yes|yeah|sure)\W*$", re.IGNORECASE)
_QUESTION = re.compile(r"\?|\b(pricing|price|cost|how much|security|soc ?2)\b", re.IGNORECASE)
_BOOK = re.compile(r"\b(demo|book|schedule|meeting|call)\b", re.IGNORECASE)
_EMAIL_TOKEN = re.compile(r"\S+@\S+")
_NAME = re.compile(r"(?i:my name is|i'm|i am|call me)\s+([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2})")
_COMPANY = re.compile(
    r"(?i:work(?:ing)?\s+(?:at|for)|company\s+is|from|with|represent|at)\s+"
    r"([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})")
_TEAM = [
    re.compile(r"(-?\d+)\s*(?:people|employees|reps|seats|users|engineers|salespeople|team members|staff)", re.IGNORECASE),
    re.compile(r"team\s+(?:of|size(?:\s+is)?)\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"(?:about|around|roughly|we have|we're)\s+(-?\d+)\b", re.IGNORECASE),
]
_ORD = {"first": "#1", "1st": "#1", "second": "#2", "2nd": "#2", "third": "#3", "3rd": "#3"}


def _slot_from(text: str) -> str | None:
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text, re.IGNORECASE)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(2).lower() == "pm" else 0)
        return f"{h:02d}:00"
    m = re.search(r"\b(first|1st|second|2nd|third|3rd)\b", text, re.IGNORECASE)
    if m:
        return _ORD[m.group(1).lower()]
    m = re.match(r"^\s*(?:option\s*)?([123])\W*$", text, re.IGNORECASE)
    return f"#{m.group(1)}" if m else None


class MockLLM:
    """Rule-based stand-in for an LLM. ``profile`` simulates a weaker/stronger prompt."""

    name = "mock-rule-based"

    def __init__(self, profile: str = "full", faults: dict[str, Any] | None = None):
        self.profile = profile
        self.faults = faults or {}

    def analyze(self, text: str, context: dict[str, Any], prompt: PromptConfig) -> str:
        f = self.faults
        if f.get("down"):
            raise LLMTimeout("provider unavailable")
        if f.get("kind") and context["turn"] in f.get("turns", []) and (f.get("persistent") or context["attempt"] == 0):
            if f["kind"] == "timeout":
                raise LLMTimeout("timeout")
            if f["kind"] == "malformed":
                return "Sure! Here is the analysis: {not json"
            if f["kind"] == "invalid_schema":
                return json.dumps({"intent": "purchase", "fields": {}})
        return json.dumps(self._analyze(text.strip(), context))

    def _analyze(self, t: str, ctx: dict[str, Any]) -> dict[str, Any]:
        full = self.profile == "full"
        awaiting = ctx.get("awaiting")
        if full and _INJECTION.search(t):
            return {"intent": "injection", "fields": {}}
        if (_HUMAN_FULL if full else _HUMAN_WEAK).search(t):
            return {"intent": "human_request", "fields": {}}
        if _GOODBYE.search(t):
            return {"intent": "goodbye", "fields": {}}
        fields: dict[str, str] = {}
        if awaiting == "slot":
            s = _slot_from(t)
            if s:
                return {"intent": "provide_info", "fields": {"slot": s}}
        else:
            if (m := _EMAIL_TOKEN.search(t)):
                fields["email"] = m.group(0).strip(".,;")
            if (m := _NAME.search(t)):
                fields["name"] = m.group(1)
            if (m := _COMPANY.search(t)):
                fields["company"] = m.group(1).rstrip(".,'")
            for rx in _TEAM:
                if (m := rx.search(t)):
                    fields["team_size"] = m.group(1)
                    break
        if fields:
            return {"intent": "provide_info", "fields": fields}
        if _QUESTION.search(t):
            return {"intent": "ask_question" if full else "unclear", "fields": {}}
        if _VAGUE.match(t):
            return {"intent": "unclear", "fields": {}}
        if _BOOK.search(t) and re.match(r"(?i)\s*(i|we|hello|hi|hey|can|could|please)\b", t):
            return {"intent": "book_demo", "fields": {}}  # a request, not an answer to the pending question
        if awaiting in ("name", "email", "company", "team_size") and len(t.split()) <= 6:
            return {"intent": "provide_info", "fields": {awaiting: t.strip(" .!")}}
        if _BOOK.search(t):
            return {"intent": "book_demo", "fields": {}}
        return {"intent": "unclear", "fields": {}}
