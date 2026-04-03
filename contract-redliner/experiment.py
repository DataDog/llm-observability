"""Offline evaluation pipeline for the contract redliner agent.

Runs run_redliner() against the golden dataset and scores each result
with three evaluators. Results are sent to Datadog LLM Observability
under the experiment name "contract-redliner-eval".

Usage:
    cd contract-redliner/
    python experiment.py
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from ddtrace.llmobs import LLMObs, EvaluatorResult, BaseEvaluator, EvaluatorContext

LLMObs.enable(
    ml_app="contract-redliner",
    api_key=os.environ["DD_API_KEY"],
    site=os.environ.get("DD_SITE", "datadoghq.com"),
    project_name="contract-redliner",
    agentless_enabled=True,
)

# Must import after LLMObs.enable()
from contract_redliner.agent import run_redliner  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset = LLMObs.create_dataset_from_csv(
    csv_path=str(Path(__file__).parent / "golden_dataset.csv"),
    dataset_name="contract-redliner-golden",
    input_data_columns=["contract"],
    expected_output_columns=["expected_proposals"],
    metadata_columns=["category", "severity"],
)

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def task(input_data: dict, config: dict | None = None) -> dict:
    result, _ = run_redliner(input_data["contract"])
    return result.model_dump()


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_RISK_ORDER = {"high": 2, "medium": 1, "low": 0}


def _parse_expected(expected_output: dict) -> list[dict]:
    """Parse the expected_proposals JSON string into a list of dicts."""
    raw = expected_output.get("expected_proposals", "[]")
    if isinstance(raw, list):
        return raw
    return json.loads(raw)


def proposal_count_delta(
    input_data: dict, output_data: dict, expected_output: dict
) -> EvaluatorResult:
    """Absolute difference between expected and actual proposal count.

    Pass if delta == 0, fail otherwise.
    Catches both over-flagging and under-flagging.
    """
    expected = _parse_expected(expected_output)
    actual = (output_data or {}).get("proposals", [])
    delta = abs(len(actual) - len(expected))
    return EvaluatorResult(
        value=delta,
        assessment="pass" if delta == 0 else "fail",
        reasoning=(
            f"Expected {len(expected)} proposal(s), got {len(actual)} "
            f"(delta={delta})."
        ),
    )


def clause_recall(
    input_data: dict, output_data: dict, expected_output: dict
) -> EvaluatorResult:
    """Fraction of expected clause indexes that the agent flagged (0.0–1.0).

    Pass if recall == 1.0 (every expected clause was caught).
    If there are no expected proposals, score is 1.0 / pass.

    Business impact: low recall means risky clauses slip through unreviewed.
    A missed liability cap or non-compliant termination clause can expose the
    company to legal action, financial penalties, or failed audits. In
    contract review, a false negative (missed issue) is almost always more
    costly than a false positive — treat recall as the primary safety metric.
    """
    expected = _parse_expected(expected_output)
    if not expected:
        return EvaluatorResult(
            value=1.0,
            assessment="pass",
            reasoning="No expected proposals — nothing to miss.",
        )

    expected_idxs = {p["clause_index"] for p in expected}
    actual_idxs = {p["clause_index"] for p in (output_data or {}).get("proposals", [])}
    recall = len(actual_idxs & expected_idxs) / len(expected_idxs)
    missed = sorted(expected_idxs - actual_idxs)
    return EvaluatorResult(
        value=round(recall, 4),
        assessment="pass" if recall == 1.0 else "fail",
        reasoning=(
            f"Recalled {len(actual_idxs & expected_idxs)}/{len(expected_idxs)} "
            f"expected clause(s)."
            + (f" Missed clause index(es): {missed}." if missed else "")
        ),
    )


def clause_precision(
    input_data: dict, output_data: dict, expected_output: dict
) -> EvaluatorResult:
    """Fraction of agent-flagged clauses that were actually expected (0.0–1.0).

    Pass if precision == 1.0 (no false positives).
    If the agent flagged nothing, score is 1.0 / pass (nothing to be wrong about).

    Business impact: low precision erodes attorney trust and drives up review
    cost. When the agent floods reviewers with false alarms, lawyers spend time
    investigating non-issues instead of high-risk clauses — slowing deal
    velocity and increasing the chance that real problems are buried in noise.
    A precision-recall tradeoff exists: optimise recall first to avoid missed
    risks, then improve precision to keep the review workload manageable.
    """
    expected = _parse_expected(expected_output)
    expected_idxs = {p["clause_index"] for p in expected}
    actual = (output_data or {}).get("proposals", [])
    actual_idxs = {p["clause_index"] for p in actual}

    if not actual_idxs:
        return EvaluatorResult(
            value=1.0,
            assessment="pass",
            reasoning="Agent flagged no clauses — no false positives.",
        )

    precision = len(actual_idxs & expected_idxs) / len(actual_idxs)
    false_positives = sorted(actual_idxs - expected_idxs)
    return EvaluatorResult(
        value=round(precision, 4),
        assessment="pass" if precision == 1.0 else "fail",
        reasoning=(
            f"Flagged {len(actual_idxs)} clause(s), "
            f"{len(actual_idxs & expected_idxs)} were expected."
            + (f" False positive clause index(es): {false_positives}." if false_positives else "")
        ),
    )


def severity_match(
    input_data: dict, output_data: dict, expected_output: dict
) -> EvaluatorResult:
    """Whether the agent's highest risk level matches the expected max severity.

    Value: 1 (pass) if match, 0 (fail) if not.
    Handles contracts with no expected proposals (expected max = 'none').
    """
    expected = _parse_expected(expected_output)

    if expected:
        expected_max = max(
            (p["risk_level"] for p in expected), key=lambda r: _RISK_ORDER[r]
        )
    else:
        expected_max = "none"

    actual_proposals = (output_data or {}).get("proposals", [])
    if actual_proposals:
        actual_max = max(
            (p["risk_level"] for p in actual_proposals),
            key=lambda r: _RISK_ORDER[r],
        )
    else:
        actual_max = "none"

    match = expected_max == actual_max
    return EvaluatorResult(
        value=1 if match else 0,
        assessment="pass" if match else "fail",
        reasoning=(
            f"Expected max severity: {expected_max}, "
            f"actual max severity: {actual_max}."
        ),
    )


class RevisionQualityEvaluator(BaseEvaluator):
    """LLM-as-judge: scores how closely suggested_revision text matches expected
    revisions in legal intent, risk coverage, and clause structure (1–5).
    Pass if score >= 3.
    """

    def __init__(self):
        super().__init__(name="revision_quality")
        self._client = OpenAI()
        self._model = os.environ.get("OPENAI_MODEL", "gpt-5-nano-2025-08-07")

    def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        output = context.output_data or {}
        expected = context.expected_output or {}

        prompt = (
            "You are a senior contract lawyer reviewing an AI agent's redlining work.\n\n"
            f"AGENT OUTPUT (actual proposals):\n{json.dumps(output, indent=2)}\n\n"
            f"EXPECTED PROPOSALS (ground truth):\n{json.dumps(expected, indent=2)}\n\n"
            "Score the agent's suggested revisions from 1 to 5 based on how closely they "
            "match the expected revisions in legal intent, risk coverage, and clause structure.\n"
            "1 = completely wrong or missing, 5 = near-identical in substance.\n"
            "If there are no expected proposals (clean contract), score 5 if the agent "
            "also flagged nothing, 1 if it over-flagged.\n\n"
            "Reply with a JSON object: {\"score\": <int 1-5>, \"reasoning\": \"<one sentence>\"}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        parsed = json.loads(response.choices[0].message.content)
        score = int(parsed.get("score", 1))
        reasoning = parsed.get("reasoning", "")

        return EvaluatorResult(
            value=score,
            assessment="pass" if score >= 3 else "fail",
            reasoning=reasoning,
        )


revision_quality = RevisionQualityEvaluator()


def flaky(
    input_data: dict, output_data: dict, expected_output: dict
):
    """Randomly raises one of three exception types, returns None, or returns 1.
    Useful for testing how the experiment loop handles evaluator failures.
    """
    outcome = random.choice(["value_error", "runtime_error", "type_error", "none", "one"])
    if outcome == "value_error":
        raise ValueError("FlakyEvaluator: simulated ValueError")
    if outcome == "runtime_error":
        raise RuntimeError("FlakyEvaluator: simulated RuntimeError")
    if outcome == "type_error":
        raise TypeError("FlakyEvaluator: simulated TypeError")
    if outcome == "none":
        return None
    return 1


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    experiment = LLMObs.experiment(
        name="contract-redliner-eval",
        task=task,
        dataset=dataset,
        runs=3,
        evaluators=[clause_recall, clause_precision, proposal_count_delta, severity_match, revision_quality, flaky],
    )
    # In a real experiment, can remove the sample size
    result = experiment.run(jobs=5, sample_size=5)
    print("done")
