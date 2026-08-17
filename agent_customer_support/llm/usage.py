"""Opt-in accounting of LLM token usage.

Both providers already return `{"input": n, "output": n}` on every call, and the LLM
facade already reads it to feed Langfuse. This module lets an *in-process* caller --
the eval harness -- collect the same numbers without a Langfuse deployment and without
threading a return value back out through every agent signature.

Nothing collects by default: `record` is a no-op unless a caller has opened a
`collect()` scope, so production pays a tuple lookup per LLM call and nothing else.

    with usage.collect() as u:
        run_the_agent()
    u.input_tokens, u.output_tokens, u.n_calls, u.by_model

Two properties the eval harness depends on:

  * `_ACTIVE` holds a *tuple* of collectors, not one, so scopes nest: an outer
    "whole turn" collector and an inner "just this call" collector both see the call.
  * A collector is a mutable object referenced by the ContextVar. `asyncio.to_thread`
    copies the context, so a ContextVar *set* inside the thread would not escape --
    but *mutating* the object the var already points at does. That is what makes a
    sync LLM call offloaded with `to_thread` visible to a scope opened by its caller.
"""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMCall:
    """One completed LLM call: which model, and what it consumed."""

    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class UsageCollector:
    """Accumulates the calls made inside one `collect()` scope."""

    calls: list[LLMCall] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def by_model(self) -> dict[str, list[int]]:
        """`model -> [input_tokens, output_tokens, n_calls]`.

        A list rather than a tuple so the result survives a JSON/CSV round trip
        unchanged, which is how the eval harness persists it.
        """
        totals: dict[str, list[int]] = {}
        for c in self.calls:
            entry = totals.setdefault(c.model, [0, 0, 0])
            entry[0] += c.input_tokens
            entry[1] += c.output_tokens
            entry[2] += 1
        return totals


_ACTIVE: contextvars.ContextVar[tuple[UsageCollector, ...]] = contextvars.ContextVar(
    "llm_usage_collectors", default=()
)


@contextmanager
def collect() -> Iterator[UsageCollector]:
    """Collect every LLM call made inside this scope."""
    collector = UsageCollector()
    token = _ACTIVE.set(_ACTIVE.get() + (collector,))
    try:
        yield collector
    finally:
        _ACTIVE.reset(token)


def record(model: str, usage: dict | None) -> None:
    """Report one call to every active collector. No-op when none are open.

    `usage` is the provider dict (`{"input": n, "output": n}`), or None when the
    provider returned no usage block -- the call is still counted, with zero tokens,
    because a call that happened is worth seeing even if its size is unknown.
    """
    active = _ACTIVE.get()
    if not active:
        return
    call = LLMCall(
        model=model,
        input_tokens=int((usage or {}).get("input") or 0),
        output_tokens=int((usage or {}).get("output") or 0),
    )
    for collector in active:
        collector.calls.append(call)
