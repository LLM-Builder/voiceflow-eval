"""Scheduling adapter interface plus a deterministic mock.

Safety rules encoded here and in the engine:
* ``get_availability`` and ``lookup`` are read-only and may be retried.
* ``book`` is NOT retried blindly. After an ambiguous timeout the engine calls
  ``lookup(idempotency_key)`` to learn the real outcome.
* ``book`` is idempotent per key, so duplicate deliveries cannot double-book.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol


class SchedulerError(Exception):
    """Definite provider failure."""


class SchedulerTimeout(SchedulerError):
    """Outcome unknown: the booking may or may not have been applied."""


class SlotTaken(SchedulerError):
    pass


@dataclass(frozen=True)
class Booking:
    confirmation_id: str
    slot: str
    key: str
    email: str


class SchedulingAdapter(Protocol):
    name: str

    def get_availability(self) -> list[str]: ...
    def book(self, key: str, slot: str, lead: dict[str, Any]) -> Booking: ...
    def lookup(self, key: str) -> Booking | None: ...


DEFAULT_SLOTS = ["2026-11-02T10:00", "2026-11-02T14:00", "2026-11-03T11:00"]


class MockScheduler:
    """Deterministic in-memory scheduler with fault injection. NOT a real provider."""

    name = "mock-scheduler"

    def __init__(self, faults: dict[str, Any] | None = None, slots: list[str] | None = None):
        self.faults = faults or {}
        self.slots = list(slots or DEFAULT_SLOTS)
        self.taken: dict[str, str] = {}  # slot -> key
        self.store: dict[str, Booking] = {}
        self.absorbed_duplicates = 0
        self.book_calls = 0
        self._slot_taken_fired = False

    def get_availability(self) -> list[str]:
        if self.faults.get("availability") == "error":
            raise SchedulerError("availability unavailable")
        return [s for s in self.slots if s not in self.taken]

    def lookup(self, key: str) -> Booking | None:
        return self.store.get(key)

    def _apply(self, key: str, slot: str, lead: dict[str, Any]) -> Booking:
        if key in self.store:
            self.absorbed_duplicates += 1
            return self.store[key]
        if slot in self.taken:
            raise SlotTaken(slot)
        b = Booking("MOCK-" + hashlib.sha1(key.encode()).hexdigest()[:8].upper(), slot, key,
                    str(lead.get("email", "")))
        self.taken[slot] = key
        self.store[key] = b
        return b

    def book(self, key: str, slot: str, lead: dict[str, Any]) -> Booking:
        self.book_calls += 1
        mode = self.faults.get("book")
        if mode == "error":
            raise SchedulerError("provider error")
        if mode == "timeout_not_applied":
            raise SchedulerTimeout("timeout before apply")
        if mode == "timeout_applied":
            self._apply(key, slot, lead)
            raise SchedulerTimeout("timeout after apply")
        if mode == "slot_taken_once" and not self._slot_taken_fired:
            self._slot_taken_fired = True
            self.taken[slot] = "other-customer"
            raise SlotTaken(slot)
        if mode == "duplicate_delivery":
            self._apply(key, slot, lead)  # first delivery
        return self._apply(key, slot, lead)

    def duplicate_booking_count(self) -> int:
        """Number of (email, slot) pairs that hold more than one confirmed booking."""
        seen: dict[tuple[str, str], int] = {}
        for b in self.store.values():
            seen[(b.email, b.slot)] = seen.get((b.email, b.slot), 0) + 1
        return sum(1 for n in seen.values() if n > 1)
