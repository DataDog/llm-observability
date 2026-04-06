"""LLM calls for clause proposal and validation.

Each function is decorated with @llm from ddtrace so it appears as a child
[llm] span inside the [tool] span that calls it.
"""

import os

from openai import OpenAI
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm

from ..primitives.models import Policy, ProposalResult, ValidationResult

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-nano-2025-08-07")
_client = OpenAI()


@llm(model_name=MODEL, model_provider="openai")
def generate_proposal(
    clause_index: int,
    clause_text: str,
    doc_type: str,
    policies: list[Policy],
) -> ProposalResult:
    """Analyze a single clause against relevant policies and propose a rewrite."""
    policy_lines = "\n".join(f"  - [{p.severity.upper()}] {p.rule}" for p in policies)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert contract lawyer performing a redline review. "
                "Analyze the given clause, identify policy violations and legal risks, "
                "and propose a specific revised clause text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Contract type: {doc_type}\n"
                f"Clause #{clause_index}:\n{clause_text}\n\n"
                f"Internal policies to comply with:\n{policy_lines}"
            ),
        },
    ]
    response = _client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=ProposalResult,
    )
    result = response.choices[0].message.parsed
    LLMObs.annotate(input_data=messages, output_data=result.model_dump())
    return result


@llm(model_name=MODEL, model_provider="openai")
def generate_validation(
    original_clauses: list[str],
    proposals: list[ProposalResult],
    doc_type: str,
) -> ValidationResult:
    """Holistically review all proposals for internal consistency and completeness."""
    proposals_text = "\n\n".join(
        f"Clause #{p.clause_index} [{p.risk_level.upper()} risk]:\n"
        f"  Proposed revision: {p.suggested_revision}\n"
        f"  Reasoning: {p.reasoning}"
        for p in proposals
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior contract lawyer performing a final review. "
                "Check all proposed edits for: (1) internal consistency — clauses must not "
                "contradict each other; (2) completeness — no standard provisions are missing; "
                "(3) accurate risk calibration. Approve, modify, or escalate each proposal."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Contract type: {doc_type}\n"
                f"Number of clauses: {len(original_clauses)}\n\n"
                f"Proposed edits:\n{proposals_text}"
            ),
        },
    ]
    response = _client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=ValidationResult,
    )
    result = response.choices[0].message.parsed
    LLMObs.annotate(input_data=messages, output_data=result.model_dump())
    return result
