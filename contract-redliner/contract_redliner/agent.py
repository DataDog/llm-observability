"""Contract redliner agent using Pydantic AI."""
import asyncio
from typing import Literal, Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from ddtrace.llmobs.decorators import agent as llmobs_agent

from .policies import POLICY_DB, policy_index

MODEL = 'gpt-5.4-nano'


class ClauseExtractorResult(BaseModel):
    clauses: list[str]


ClauseExtractor = Agent(
    MODEL,
    name="Clause Extractor",
    output_type=ClauseExtractorResult,
    system_prompt=(
        "Extract clauses from the contract that should be reviewed "
        "against company policies into an array. "
        "Each clause should be a complete paragraph or numbered section."
    )
)


class Revision(BaseModel):
    reasoning: str
    revised_clause: str
    risk_level: Literal["low", "medium", "high"]


class ClauseReviewerResult(BaseModel):
    revision: Optional[Revision]


ClauseReviewer = Agent(
    MODEL,
    name="Clause Reviewer",
    output_type=ClauseReviewerResult,
    system_prompt=(
        "You are a contract redlining agent. Your job is to review the "
        "given contract clause against relevant company policies "
        "and determine if revision is required. Revisions should "
        "not greatly exceed the original clause in length. State your "
        "reasoning concisely.\n\n"
        "Policy Index:\n\n"
        f"{policy_index()}\n\n"
    ),
)


@ClauseReviewer.tool_plain
def get_policy(topic: str, policy: str) -> str:
    """Look up policy text by doc_type and topic."""
    return POLICY_DB.get(topic, {}).get(policy, "policy not found")


@llmobs_agent(name="ContractRedliner")
async def run_redliner(contract_text: str) -> list[tuple[str, Revision]]:
    """Run redliner on contract text."""

    result = await ClauseExtractor.run(contract_text)

    revision_coros = [
        ClauseReviewer.run(clause)
        for clause in result.output.clauses
    ]

    results = await asyncio.gather(*revision_coros)

    revisions = []
    for original_clause, result in zip(result.output.clauses, results):
        revision = result.output.revision
        if revision:
            revisions.append((original_clause, revision))

    return revisions
