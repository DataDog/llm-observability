"""Contract redliner agent using Pydantic AI."""
import asyncio
from dataclasses import dataclass
from typing import Literal, Optional, Union

from pydantic import BaseModel
from pydantic_ai import Agent
from ddtrace.llmobs.decorators import agent as llmobs_agent

from .policies import POLICY_DB, policy_index

MODEL = 'gpt-5.4-nano'


class ProposedRevision(BaseModel):
    reasoning: str
    revised_clause: str
    risk_level: Literal["low", "medium", "high"]


ClauseReviewer = Agent(
    MODEL,
    name="Clause Reviewer",
    output_type=Optional[ProposedRevision],
    system_prompt=(
        "Review the given contract clause against relevant company policies "
        "and determine if revision is required. Revisions should "
        "not greatly exceed the original clause in length. State your "
        "reasoning concisely. Follow these steps:\n"
        " 1. Retrieve relevant policies\n"
        " 2. Read policies that may be relevant\n"
        " 3. If the clause needs revision, return the revision "
        "    along with reasoning and risk level. Otherwise, return "
        "    empty values."
    ),
)


class RetrievedPolicy(BaseModel):
    topic: str
    policy_name: str


PolicyRetriever = Agent(
    MODEL,
    name="Policy Retriever",
    output_type=list[RetrievedPolicy],
    system_prompt=(
        "Return policies that may be relevant for the given contract clause."
        "\n\nPolicy Index:\n\n"
        f"{policy_index()}"
    )
)


@ClauseReviewer.tool_plain
async def retrieve_policies(clause: str) -> list[RetrievedPolicy]:
    result = await PolicyRetriever.run(clause)
    return result.output


@ClauseReviewer.tool_plain
async def read_policy(topic: str, policy: str) -> str:
    return POLICY_DB.get(topic, {}).get(policy, "policy not found")


class ValidatorDecision(BaseModel):
    revision_index: int
    decision: Literal["accept", "reject", "modify"]
    reasoning: str
    final_revision_if_modified: str

Validator = Agent(
    MODEL,
    name="Validator",
    output_type=list[ValidatorDecision],
    system_prompt=(
        "You are a senior contract lawyer performing a final review. "
        "Check all proposed edits for: (1) internal consistency — clauses must not "
        "contradict each other; (2) conciseness — language is succinct; "
        "(3) accurate risk calibration. Approve, modify, or escalate each proposal."
    )
)

class ProposedRevisionWithOriginal(ProposedRevision):
    original_clause: str

RedLiner = Agent(
    MODEL,
    name="RedLiner",
    output_type=list[ProposedRevisionWithOriginal],
    system_prompt=(
        "You are a contract redlining agent. Use the review_clause tool "
        "to review each contract clause against company policies. "
        "Each clause should be a complete paragraph or numbered section. "
        "Afterwards, use the validate_revisions tool to check for overall "
        "consistency and quality of the proposed revisions."
    )
)


@RedLiner.tool_plain
async def review_clause(clause: str) -> Optional[ProposedRevision]:
    result = await ClauseReviewer.run(clause)
    return result.output


@RedLiner.tool_plain
async def validate_revisions(revisions: list[ProposedRevision]) -> list[ValidatorDecision]:
    revision_text = "\n\n".join(
        revision.model_dump_json(indent=2)
        for revision in revisions
    )
    result = await Validator.run(revision_text)
    return result.output


@llmobs_agent(name="ContractRedliner")
async def run_redliner(contract_text: str) -> list[ProposedRevisionWithOriginal]:
    result = await RedLiner.run(contract_text)
    return result.output
