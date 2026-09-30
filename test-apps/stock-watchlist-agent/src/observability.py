"""Logging for the stock-watchlist agent, with Datadog trace correlation.

Plain Python logging plus ddtrace's logging integration, so every record carries
``dd.trace_id`` / ``dd.span_id`` / ``dd.service`` / ``dd.env`` / ``dd.version``. When the
app runs under tracing (a normal run with ``DD_API_KEY``, or
``ddtrace-experiment ... --trace``), those ids let each log line pivot to its LLM Obs
trace in Datadog; otherwise they're emitted as ``0`` / empty.
"""
from __future__ import annotations

import logging

import ddtrace


_LOG_FORMAT = (
    "%(asctime)s %(levelname)-5s [%(name)s] "
    "[dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s "
    "dd.service=%(dd.service)s dd.env=%(dd.env)s dd.version=%(dd.version)s] %(message)s"
)

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with Datadog trace-id injection. Idempotent."""
    global _configured
    if _configured:
        return
    ddtrace.patch(logging=True)  # inject dd.* fields into every LogRecord
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    _configured = True
