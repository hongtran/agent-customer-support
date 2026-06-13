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


@contextmanager
def span(name: str, *, input: Any = None, metadata: dict | None = None) -> Iterator[Any]:
    """Nested span. No-op (yields a dummy with .update()) when tracing is off.

    Only the Langfuse start call is guarded; exceptions raised by the wrapped
    body propagate normally (and are recorded by Langfuse when enabled).
    """
    client = _client()
    if client is None:
        yield _NOOP
        return
    try:
        cm = client.start_as_current_span(name=name, input=input, metadata=metadata)
    except Exception as exc:
        logger.warning("tracing span '%s' failed to start: %s", name, exc)
        yield _NOOP
        return
    with cm as handle:
        yield handle


@contextmanager
def generation(
    name: str, *, model: str, input: Any = None, metadata: dict | None = None
) -> Iterator[Any]:
    """LLM generation span. No-op when tracing is off."""
    client = _client()
    if client is None:
        yield _NOOP
        return
    try:
        cm = client.start_as_current_generation(
            name=name, model=model, input=input, metadata=metadata
        )
    except Exception as exc:
        logger.warning("tracing generation '%s' failed to start: %s", name, exc)
        yield _NOOP
        return
    with cm as handle:
        yield handle


def update_trace(**kwargs: Any) -> None:
    """Set trace-level attributes (e.g. session_id, user_id, tags, output)."""
    client = _client()
    if client is None:
        return
    try:
        client.update_current_trace(**kwargs)
    except Exception as exc:
        logger.warning("tracing update_trace failed: %s", exc)


def flush() -> None:
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("tracing flush failed: %s", exc)
