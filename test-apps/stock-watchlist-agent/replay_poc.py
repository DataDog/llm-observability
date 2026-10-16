# replay_poc.py — PoC: app.datadoghq.com -> localhost -> re-run the REAL agent as an LLM Obs Experiment
#
# Runs on stock RELEASED ddtrace. The browser reads a trace's replay case off the LLM Obs UI
# (metadata.replay_input + metadata.replay_output) and POSTs it here; this server turns that case
# into a one-record Dataset, runs an LLM Obs Experiment whose task re-runs the current LOCAL code,
# scores it with the app's evaluators, and returns the Experiment URL for the button to deep-link to.
#
# Requires DD_API_KEY (trace ingest) + DD_APP_KEY (datasets/experiments API) + OPENAI_API_KEY in .env.
import asyncio
import time
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY / DD_API_KEY / DD_APP_KEY / DD_SITE before ddtrace imports

from flask import Flask, request, jsonify, make_response
from ddtrace.llmobs import LLMObs

from src.agents.orchestrator import analyze_portfolio  # the real entrypoint (the experiment task)
from ddtrace.llmobs._evaluators import EvaluatorContext
from src.evals import CompletenessEvaluator, sentiment_judge, grounding_judge  # the app's evaluators

LLMObs.enable(ml_app="stock-watchlist-agent", agentless_enabled=True)

ML_APP = "stock-watchlist-agent"
app = Flask(__name__)


@app.after_request
def cors(resp):
    origin = request.headers.get("Origin", "")
    if origin.endswith(".datadoghq.com") or origin.endswith(".datad0g.com"):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Private-Network"] = "true"  # Chrome LNA
    return resp


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return make_response("", 204)
    return jsonify(ok=True, agent="stock-watchlist-agent")


# --------------------------------------------------------------------------- #
# Experiment wiring
# --------------------------------------------------------------------------- #
def _task(input_data, config):
    """Experiment task: re-run the current agent code on the case's input, return the new output.
    (Runs on the developer's laptop — this is the 'replay against local code' step.)"""
    briefing, _ = asyncio.run(analyze_portfolio(input_data["tickers"]))
    return briefing.model_dump()


def _judge_evaluator(app_ev):
    """Adapt a fixed app evaluator (the LLM judges) to the experiment evaluator signature
    (input_data, output_data, expected_output) -> value. Scores the NEW output; the recorded
    expected_output rides along on the dataset record for the old-vs-new view in the UI."""
    def _fn(input_data, output_data, expected_output):
        ctx = EvaluatorContext(input_data=input_data, output_data=output_data, span_id="", trace_id="")
        return app_ev.evaluate(ctx).value
    _fn.__name__ = app_ev.name  # becomes the eval-metric label in the Experiments UI
    return _fn


def completeness(input_data, output_data, expected_output):
    # CompletenessEvaluator needs the requested tickers at construction, so build it per record.
    ev = CompletenessEvaluator(input_data["tickers"])
    ctx = EvaluatorContext(input_data=input_data, output_data=output_data, span_id="", trace_id="")
    return ev.evaluate(ctx).value


EVALUATORS = [
    completeness,                        # programmatic (per-record tickers)
    _judge_evaluator(sentiment_judge),   # LLM judge (boolean)
    _judge_evaluator(grounding_judge),   # LLM judge (score 1-5)
]


@app.route("/replay_trace", methods=["POST", "OPTIONS"])
def replay_trace():
    if request.method == "OPTIONS":
        return make_response("", 204)
    body = request.get_json(silent=True) or {}
    md = body.get("metadata") or {}
    replay_input = md.get("replay_input")
    replay_output = md.get("replay_output")
    trace_id = body.get("trace_id") or "adhoc"

    if not (isinstance(replay_input, dict) and isinstance(replay_input.get("tickers"), list)):
        return make_response(jsonify(error="missing metadata.replay_input.tickers"), 400)
    if replay_output is None:
        return make_response(jsonify(error="missing metadata.replay_output"), 400)

    tickers = replay_input["tickers"]
    dataset_name = f"replay-{trace_id}"

    # Stable dataset per trace: reuse it across replays so every replay of this trace runs over the
    # SAME case, and the resulting experiments compare against each other in the UI. Pull the existing
    # dataset if we've replayed this trace before; otherwise create it seeded with the trace's case.
    try:
        dataset = LLMObs.pull_dataset(dataset_name, project_name=ML_APP)
    except Exception:
        dataset = LLMObs.create_dataset(
            dataset_name,
            project_name=ML_APP,
            description=f"Replay case for trace {trace_id}",
            records=[{"input_data": {"tickers": tickers}, "expected_output": replay_output}],
        )

    # Each replay is its own experiment over that shared dataset (unique name = its own run).
    experiment = LLMObs.experiment(
        f"replay-{trace_id}-{int(time.time())}",
        _task,
        dataset,
        evaluators=EVALUATORS,
        project_name=ML_APP,
        tags={"source": "replay-poc", "replayed_trace_id": trace_id},
    )
    experiment.run()  # runs local code + evaluators, publishes to LLM Obs Experiments (~60-120s)

    return jsonify(
        status="done",
        tickers=tickers,
        replayed_trace_id=trace_id,
        experiment_url=experiment.url,
    )


app.run(port=8787)
