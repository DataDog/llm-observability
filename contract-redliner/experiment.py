"""Offline evaluation pipeline for the contract redliner agent.

Runs run_redliner() against the golden dataset and scores each result
with three evaluators. Results are sent to Datadog LLM Observability
under the experiment name "contract-redliner-eval".

Usage:
    cd contract-redliner/
    python experiment.py
"""

import asyncio
import json
from pathlib import Path
from ddtrace.llmobs import LLMObs, EvaluatorResult
from openai import AsyncOpenAI

MODEL = 'gpt-5.4-nano'

LLMObs.enable(
    ml_app="contract-redliner",
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

async def task(input_data: dict, config: dict | None = None) -> list[dict]:
    revisions = await run_redliner(input_data["contract"])
    return [
        {"original_clause": clause, **revision.model_dump()}
        for clause, revision in revisions
    ]


# ---------------------------------------------------------------------------
# Clause matching
# ---------------------------------------------------------------------------

def clauses_match(clause1: str, clause2: str) -> bool:
    """True if either normalized clause contains the other."""
    a = " ".join(clause1.lower().split())
    b = " ".join(clause2.lower().split())
    return a in b or b in a


def _matched_pairs(expected: list[dict], output_data: list[dict]) -> list[tuple[dict, dict]]:
    """Return (expected, actual) pairs where original_clause matches. One match per expected clause."""
    pairs = []
    for exp in expected:
        for item in output_data:
            if clauses_match(exp["original_clause"], item["original_clause"]):
                pairs.append((exp, item))
                break
    return pairs


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

async def clause_recall(input_data, output_data, expected_output) -> EvaluatorResult:
    expected = json.loads(expected_output["expected_proposals"])
    if not expected:
        return EvaluatorResult(value=1.0, assessment="pass")

    flagged_clauses = [item["original_clause"] for item in output_data]
    matched = sum(
        1 for exp in expected
        if any(clauses_match(exp["original_clause"], flagged) for flagged in flagged_clauses)
    )
    score = matched / len(expected)
    return EvaluatorResult(value=score, assessment="pass" if score >= 1.0 else "fail")


async def clause_precision(input_data, output_data, expected_output) -> EvaluatorResult:
    if not output_data:
        return EvaluatorResult(value=1.0, assessment="pass")

    expected_clauses = [item["original_clause"] for item in json.loads(expected_output["expected_proposals"])]
    matched = sum(
        1 for item in output_data
        if any(clauses_match(item["original_clause"], exp) for exp in expected_clauses)
    )
    score = matched / len(output_data)
    return EvaluatorResult(value=score, assessment="pass" if score >= 1.0 else "fail")


async def severity_accuracy(input_data, output_data, expected_output) -> EvaluatorResult:
    pairs = _matched_pairs(json.loads(expected_output["expected_proposals"]), output_data)
    if not pairs:
        return EvaluatorResult(value=1.0, assessment="pass")

    correct = sum(1 for exp, item in pairs if item["risk_level"] == exp["risk_level"])
    score = correct / len(pairs)
    return EvaluatorResult(value=score, assessment="pass" if score >= 1.0 else "fail")


_openai = AsyncOpenAI()


async def _rate_revision(expected: dict, actual: dict) -> float:
    response = await _openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a senior contract lawyer reviewing an AI agent's redlining work.",
            },
            {
                "role": "user",
                "content": (
                    f"EXPECTED REVISION:\n{json.dumps(expected, indent=2)}\n\n"
                    f"AGENT REVISION:\n{json.dumps(actual, indent=2)}\n\n"
                    "Score the agent's suggested revision from 1 to 5 based on how closely it "
                    "matches the expected revision in legal intent, risk coverage, and clause structure. "
                    "1 = completely wrong or missing, 5 = near-identical in substance. "
                    "Reply with ONLY a single integer from 1 to 5."
                ),
            },
        ],
        max_completion_tokens=5,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except (ValueError, AttributeError):
        return 1.0


async def revision_quality(input_data, output_data, expected_output) -> EvaluatorResult:
    pairs = _matched_pairs(json.loads(expected_output["expected_proposals"]), output_data)
    if not pairs:
        return EvaluatorResult(value=1.0, assessment="fail")

    scores = await asyncio.gather(*[_rate_revision(exp, actual) for exp, actual in pairs])
    avg = sum(scores) / len(scores)
    return EvaluatorResult(value=avg, assessment="pass" if avg >= 3.0 else "fail")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main():
    experiment = LLMObs.async_experiment(
        name="contract-redliner-eval",
        task=task,
        dataset=dataset,
        evaluators=[clause_recall, clause_precision, severity_accuracy, revision_quality],
    )
    await experiment.run(jobs=5)
    print(f"done")


if __name__ == "__main__":
    asyncio.run(main())
