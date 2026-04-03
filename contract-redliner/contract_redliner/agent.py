"""Contract redliner agent using Pydantic AI.

Two agents:
  - segmenter: classifies document type and splits into clauses (output_type=DocumentSegment)
  - redliner: tool-calling agent that reviews each clause (output_type=RedlineResult)

LLM Observability tracing:
  - run_redliner() → @llmobs_agent (top-level [agent] span)
  - _segment()     → @llmobs_task  ([task] child span)
  - Each @redliner.tool uses `with LLMObs.tool()` for explicit [tool] spans
  - generate_proposal / generate_validation use @llm → [llm] spans nested inside [tool] spans
"""

import os

from pydantic_ai import Agent, RunContext
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import agent as llmobs_agent, task as llmobs_task

from contract_redliner.primitives.models import (
    ContractDeps,
    DocumentSegment,
    ProposalResult,
    RedlineResult,
)
from contract_redliner.primitives.policies import get_policies
from contract_redliner.tools.tools import generate_proposal, generate_validation

MODEL = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-5-nano-2025-08-07')}"

# ---------------------------------------------------------------------------
# Segmenter agent — classify + split into clauses in one structured call
# ---------------------------------------------------------------------------

segmenter = Agent(
    MODEL,
    output_type=DocumentSegment,
    system_prompt=(
        "You are a contract parser. Classify the contract type as one of: "
        "nda, saas, employment, vendor, other. "
        "Then split the contract into discrete, self-contained clauses. "
        "Each clause should be a complete paragraph or numbered section. "
        "Return structured output with doc_type and the list of clause strings."
    ),
)

# ---------------------------------------------------------------------------
# Redliner agent — autonomous tool-calling loop
# ---------------------------------------------------------------------------

redliner = Agent(
    MODEL,
    deps_type=ContractDeps,
    output_type=RedlineResult,
    system_prompt=(
        "You are a contract redlining agent. Your job is to review every clause "
        "and return a complete RedlineResult.\n\n"
        "Follow this process for each clause (0-indexed):\n"
        "1. Call policy_retrieval(clause_topic) to fetch relevant internal policies.\n"
        "2. Call proposal_tool(clause_index, clause_topic) to analyze and propose improvements.\n\n"
        "After all clauses have proposals, call validation_tool(proposals) once for a holistic review.\n\n"
        "The RedlineResult you return must include:\n"
        "- proposals: all ProposalResult objects (one per clause)\n"
        "- risk_summary: counts of high/medium/low risk proposals\n\n"
        "Do not skip any clause. Process them in order."
    ),
)


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

@redliner.tool
def policy_retrieval(ctx: RunContext[ContractDeps], clause_topic: str) -> list[dict]:
    """Retrieve internal policies relevant to a clause topic.

    Args:
        clause_topic: Short description of what the clause covers (e.g. "liability cap", "termination").
    """
    with LLMObs.tool(name="policy_retrieval") as span:
        policies = get_policies(ctx.deps.doc_type, clause_topic)
        LLMObs.annotate(
            span=span,
            input_data={"doc_type": ctx.deps.doc_type, "clause_topic": clause_topic},
            output_data=[p.model_dump() for p in policies],
        )
    return [p.model_dump() for p in policies]


@redliner.tool
def proposal_tool(
    ctx: RunContext[ContractDeps],
    clause_index: int,
    clause_topic: str,
) -> dict:
    """Analyze a clause and propose improvements against internal policies.

    Args:
        clause_index: 0-based index of the clause in the contract.
        clause_topic: Short description of what the clause covers.
    """
    with LLMObs.tool(name="proposal_tool") as span:
        clause_text = ctx.deps.clauses[clause_index]
        policies = get_policies(ctx.deps.doc_type, clause_topic)
        result = generate_proposal(clause_index, clause_text, ctx.deps.doc_type, policies)
        LLMObs.annotate(
            span=span,
            input_data={"clause_index": clause_index, "clause_text": clause_text},
            output_data=result.model_dump(),
        )
    return result.model_dump()


@redliner.tool
def validation_tool(ctx: RunContext[ContractDeps], proposals: list[ProposalResult]) -> dict:
    """Holistically validate all proposed edits for consistency and completeness.

    Args:
        proposals: List of all ProposalResult objects generated so far.
    """
    with LLMObs.tool(name="validation_tool") as span:
        result = generate_validation(ctx.deps.clauses, proposals, ctx.deps.doc_type)
        LLMObs.annotate(
            span=span,
            input_data={
                "proposal_count": len(proposals),
                "doc_type": ctx.deps.doc_type,
                "clause_count": len(ctx.deps.clauses),
            },
            output_data=result.model_dump(),
        )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@llmobs_task
def _segment(contract_text: str) -> DocumentSegment:
    result = segmenter.run_sync(contract_text)
    LLMObs.annotate(
        input_data={"text_length": len(contract_text)},
        output_data=result.output.model_dump(),
    )
    return result.output


@llmobs_agent(name="ContractRedliner")
def run_redliner(contract_text: str) -> tuple[RedlineResult, dict]:
    """Classify, segment, propose edits, validate — return (RedlineResult, span_ctx)."""
    LLMObs.annotate(input_data={"contract_text": contract_text})

    doc = _segment(contract_text)
    deps = ContractDeps(doc_type=doc.doc_type, clauses=doc.clauses)

    agent_result = redliner.run_sync(
        (
            f"Redline this {doc.doc_type} contract. "
            f"It has {len(doc.clauses)} clauses (0-indexed). "
            f"Process every clause and return a complete RedlineResult."
        ),
        deps=deps,
    )

    output = agent_result.output
    LLMObs.annotate(output_data=output.model_dump())
    span_ctx = LLMObs.export_span()

    return output, span_ctx
