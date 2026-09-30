"""
Local workaround (auto-imported by Python at startup because this dir is on
sys.path).

Also loads `.env` here so every entrypoint (src.main, capture_*.py,
trace_scenarios.py, ddtrace-experiment) picks up credentials without each
script needing its own load_dotenv() call. Existing environment variables
take precedence over `.env`.

ddtrace 4.10.2's agentless trace export encodes spans with plain json.dumps()
(no default=), and the pydantic_ai integration captures OpenAI `Omit` /
`NOT_GIVEN` sentinels into span metadata. Those aren't JSON-serializable, so a
real agent run crashes in span.finish() with:

    TypeError: Object of type Omit is not JSON serializable

This adds a `default=str` fallback so the FULL trace still encodes and is sent to
Datadog — the only effect is that an omitted-parameter sentinel shows as its
string form instead of crashing the run. Remove once the ddtrace pydantic_ai
integration sanitizes these sentinels (or the encoder gains a default=).
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # never let the workaround break startup
    pass

try:
    import json

    import ddtrace.internal.encoding as _enc

    def _json_dumps_bytes(obj: object) -> bytes:
        # default=str only kicks in for otherwise-unserializable values (Omit, etc.)
        return json.dumps(obj, default=str).encode("utf-8", errors="backslashreplace")

    _enc._json_dumps_bytes = _json_dumps_bytes
except Exception:  # never let the workaround break startup
    pass
