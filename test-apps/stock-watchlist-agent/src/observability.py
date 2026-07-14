"""Logging for the stock-watchlist agent, with Datadog trace correlation.

Plain Python logging plus ddtrace's logging integration, so every record carries
``dd.trace_id`` / ``dd.span_id`` / ``dd.service`` / ``dd.env`` / ``dd.version``. When the
app runs under tracing (a normal run with ``DD_API_KEY``, or
``ddtrace-experiment run ... --publish``), those ids let each log line pivot to its LLM
Obs trace in Datadog; otherwise they're emitted as ``0`` / empty.
"""
from __future__ import annotations

import logging
import os

import ddtrace


_LOG_FORMAT = (
    "%(asctime)s %(levelname)-5s [%(name)s] "
    "[dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s "
    "dd.service=%(dd.service)s dd.env=%(dd.env)s dd.version=%(dd.version)s] %(message)s"
)

_configured = False


def _level_from_env(default: int) -> int:
    """Resolve the log level from ``LOG_LEVEL`` (e.g. WARNING / ERROR / DEBUG), else ``default``.

    Lets a demo quiet the app's own INFO chatter (and propagated library logs) with
    ``export LOG_LEVEL=WARNING`` — no code change.
    """
    name = os.environ.get("LOG_LEVEL")
    if not name:
        return default
    resolved = logging.getLevelName(name.strip().upper())
    return resolved if isinstance(resolved, int) else default


def setup_logging(level: int | None = None) -> None:
    """Configure root logging once, with Datadog trace-id injection. Idempotent.

    Level precedence: explicit ``level`` arg > ``LOG_LEVEL`` env > INFO. Set
    ``LOG_LEVEL=WARNING`` (or ``ERROR``) to silence the app's INFO logs for a clean demo.
    """
    global _configured
    if _configured:
        return
    if level is None:
        level = _level_from_env(logging.INFO)
    ddtrace.patch(logging=True)  # inject dd.* fields into every LogRecord
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    _configured = True
