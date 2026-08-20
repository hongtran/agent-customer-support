import contextvars
import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from agent_customer_support.config import get_settings

logger = logging.getLogger(__name__)


class _NoopSpan:
    """Returned when tracing is disabled or a span fails to start."""

    def update(self, **kwargs: Any) -> None:
        return None


_NOOP = _NoopSpan()


@lru_cache(maxsize=1)
def _client():
    """Singleton Langfuse client, or None if unconfigured / import fails."""
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
    except Exception as exc:  # import error or init failure -> tracing off
        logger.warning("Langfuse init failed; tracing disabled: %s", exc)
        return None


def enabled() -> bool:
    return _client() is not None


# Which agent (and which step within it) is running right now. Set by `agent_span`
# and `step`, read by `generation` so an LLM call can be attributed to its agent --
# the LLM facade calls the model without ever learning who asked for it, and the
# name only lives on the parent span, which the facade cannot see. A ContextVar
# carries it without growing a parameter on every agent signature; agents call the
# model synchronously inside their own async method, so it is simply in scope (and
# an `asyncio.to_thread` offload would still read it, since to_thread copies the
# context in -- only a *set* inside the thread would fail to escape).
_AGENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("tracing_agent", default=None)
_STEP: contextvars.ContextVar[str | None] = contextvars.ContextVar("tracing_step", default=None)


def current_labels() -> tuple[str | None, str | None]:
    """`(agent, step)` for the innermost enclosing scope; either may be None."""
    return _AGENT.get(), _STEP.get()


@contextmanager
def trace(
    name: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    input: Any = None,
    metadata: dict | None = None,
) -> Iterator[Any]:
    """Root of one turn. Sets trace-level attributes (session_id groups a whole
    conversation across turns) and opens the root span. No-op when tracing is off.

    Only the Langfuse setup is guarded; exceptions from the wrapped body propagate.
    """
    client = _client()
    if client is None:
        yield _NOOP
        return
    try:
        from langfuse import propagate_attributes

        attr_cm = propagate_attributes(session_id=session_id, user_id=user_id, tags=tags)
        # "chain", not "span": the root is the whole pipeline. The sub-agents beneath
        # it are the `agent`-typed observations.
        span_cm = client.start_as_current_observation(
            name=name, as_type="chain", input=input, metadata=metadata
        )
    except Exception as exc:
        logger.warning("tracing trace '%s' failed to start: %s", name, exc)
        yield _NOOP
        return
    with attr_cm, span_cm as handle:
        yield handle


@contextmanager
def span(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict | None = None,
) -> Iterator[Any]:
    """Nested span. No-op (yields a dummy with .update()) when tracing is off.

    `as_type` is the Langfuse observation type -- "tool", "retriever", "guardrail"
    and friends render with their own icon and can be filtered on in the UI. The
    default keeps a plain span for anything with no better label.

    Only the Langfuse start call is guarded; exceptions raised by the wrapped
    body propagate normally (and are recorded by Langfuse when enabled).
    """
    client = _client()
    if client is None:
        yield _NOOP
        return
    try:
        cm = client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        )
    except Exception as exc:
        logger.warning("tracing span '%s' failed to start: %s", name, exc)
        yield _NOOP
        return
    with cm as handle:
        yield handle


@contextmanager
def agent_span(name: str, *, input: Any = None, metadata: dict | None = None) -> Iterator[Any]:
    """Span for one sub-agent: typed `agent`, and it names every LLM generation
    started inside it (`llm.<name>`).

    The label is reset in a `finally` so an agent that raises cannot leak its name
    onto the next agent's generations.
    """
    token = _AGENT.set(name)
    try:
        with span(f"agent.{name}", as_type="agent", input=input, metadata=metadata) as handle:
            yield handle
    finally:
        _AGENT.reset(token)


@contextmanager
def step(name: str) -> Iterator[None]:
    """Sub-label the LLM calls made inside, e.g. `llm.knowledge.contextualize`.

    Creates no observation of its own -- it exists only to keep a second LLM call
    inside one agent distinguishable from the first, which matters when pointing a
    Langfuse evaluator at a single kind of call.
    """
    token = _STEP.set(name)
    try:
        yield
    finally:
        _STEP.reset(token)


@contextmanager
def generation(
    name: str, *, model: str, input: Any = None, metadata: dict | None = None
) -> Iterator[Any]:
    """LLM generation span (records model + token usage via handle.update). No-op when off.

    The enclosing `agent_span`/`step` labels are folded into the observation name
    (`llm` -> `llm.knowledge.contextualize`) and into its metadata, so a generation
    can be filtered by the agent that made it without reading its parent.
    """
    client = _client()
    if client is None:
        yield _NOOP
        return
    agent, step_name = current_labels()
    name = ".".join(part for part in (name, agent, step_name) if part)
    # Copy: the caller's dict is theirs, and the facade passes a literal it may reuse.
    metadata = dict(metadata or {})
    if agent:
        metadata["agent"] = agent
    if step_name:
        metadata["step"] = step_name
    try:
        cm = client.start_as_current_observation(
            name=name, as_type="generation", model=model, input=input, metadata=metadata
        )
    except Exception as exc:
        logger.warning("tracing generation '%s' failed to start: %s", name, exc)
        yield _NOOP
        return
    with cm as handle:
        yield handle


def flush() -> None:
    """Flush buffered traces (call on shutdown; safe no-op when disabled)."""
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("tracing flush failed: %s", exc)
