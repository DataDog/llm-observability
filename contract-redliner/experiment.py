"""Offline evaluation pipeline for the contract redliner agent.

Runs run_redliner() against the golden dataset and scores each result
with three evaluators. Results are sent to Datadog LLM Observability
under the experiment name "contract-redliner-eval".

Usage:
    cd contract-redliner/
    python experiment.py
"""

import os
from pathlib import Path
from ddtrace.llmobs import LLMObs, EvaluatorResult, LLMJudge, ScoreStructuredOutput

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
# Evaluators
# ---------------------------------------------------------------------------

clause_recall = LLMJudge(
    name="clause_recall",
    provider="openai",
    model=MODEL,
    system_prompt="You are evaluating a contract review agent.",
    user_prompt=(
        "EXPECTED problematic clauses (ground truth):\n{{expected_output}}\n\n"
        "AGENT flagged clauses:\n{{output_data}}\n\n"
        "For each expected clause, determine whether the agent flagged it "
        "(the same clause, identified by its section title). "
        "Score = (number of expected clauses the agent caught) / (total expected clauses). "
        "If there are no expected clauses, score 1.0."
    ),
    structured_output=ScoreStructuredOutput(
        description="Fraction of expected problematic clauses the agent caught (0.0–1.0)",
        min_score=0.0,
        max_score=1.0,
        min_threshold=1.0,
        reasoning=True,
    ),
)

clause_precision = LLMJudge(
    name="clause_precision",
    provider="openai",
    model=MODEL,
    system_prompt="You are evaluating a contract review agent.",
    user_prompt=(
        "EXPECTED problematic clauses (ground truth):\n{{expected_output}}\n\n"
        "AGENT flagged clauses:\n{{output_data}}\n\n"
        "For each clause the agent flagged, determine whether it is actually problematic "
        "(i.e. it appears in the expected list, matched by section title). "
        "Score = (number of flagged clauses that are actually problematic) / (total flagged clauses). "
        "If the agent flagged nothing, score 1.0."
    ),
    structured_output=ScoreStructuredOutput(
        description="Fraction of agent-flagged clauses that are actually problematic (0.0–1.0)",
        min_score=0.0,
        max_score=1.0,
        min_threshold=1.0,
        reasoning=True,
    ),
)


severity_accuracy = LLMJudge(
    name="severity_accuracy",
    provider="openai",
    model=MODEL,
    system_prompt="You are evaluating a contract review agent.",
    user_prompt=(
        "EXPECTED problematic clauses (ground truth):\n{{expected_output}}\n\n"
        "AGENT flagged clauses:\n{{output_data}}\n\n"
        "For each clause the agent flagged, compare its risk_level against the expected "
        "risk_level for that clause (matched by section title). "
        "Score = (number of correctly classified clauses) / (number of expected clauses). "
        "If a clause was flagged but not expected, it does not contribute to the score. "
        "If there are no expected clauses, score 1.0."
    ),
    structured_output=ScoreStructuredOutput(
        description="Fraction of expected clauses whose risk level the agent classified correctly (0.0–1.0)",
        min_score=0.0,
        max_score=1.0,
        min_threshold=1.0,
        reasoning=True,
    ),
)


revision_quality = LLMJudge(
    name="revision_quality",
    provider="openai",
    model=MODEL,
    system_prompt="You are a senior contract lawyer reviewing an AI agent's redlining work.",
    user_prompt=(
        "AGENT OUTPUT (actual proposals):\n{{output_data}}\n\n"
        "EXPECTED PROPOSALS (ground truth):\n{{expected_output}}\n\n"
        "Score the agent's suggested revisions from 1 to 5 based on how closely they "
        "match the expected revisions in legal intent, risk coverage, and clause structure. "
        "1 = completely wrong or missing, 5 = near-identical in substance. "
        "If there are no expected proposals (clean contract), score 5 if the agent "
        "also flagged nothing, 1 if it over-flagged."
    ),
    structured_output=ScoreStructuredOutput(
        description="Quality score for the agent's suggested revisions (1=completely wrong, 5=near-identical in substance)",
        min_score=1,
        max_score=5,
        min_threshold=3,
        reasoning=True,
    ),
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main():
    experiment = LLMObs.async_experiment(
        name="contract-redliner-eval",
        task=task,
        dataset=dataset,
        runs=3,
        evaluators=[clause_recall, clause_precision, severity_accuracy, revision_quality],
    )
    result = await experiment.run(jobs=5)
    print(f"done: {result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
