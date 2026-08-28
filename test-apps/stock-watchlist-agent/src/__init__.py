# Configure logging (with Datadog trace correlation) for every entrypoint of this test
# app — normal runs and `ddtrace-experiment` capture/replay all import the `src` package,
# so this is the single place that guarantees logs are visible across all of them.
from .observability import setup_logging

setup_logging()
