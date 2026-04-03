# Contract Redliner

An AI-powered contract review agent that automates legal due diligence by analyzing contracts against company policies, identifying risk areas, and proposing specific clause revisions. Built with Pydantic AI and traced end-to-end with Datadog LLM Observability.

## What it does

The Contract Redliner agent helps legal and business teams review vendor agreements, SaaS contracts, NDAs, and employment contracts by:

- **Identifying risky clauses**: Automatically flags liability caps, data processing terms, termination clauses, and other high-risk provisions that don't meet company standards
- **Proposing specific edits**: Generates concrete revision text for each problematic clause, not just generic feedback
- **Risk-based prioritization**: Classifies issues as high/medium/low risk to focus attention on the most critical items
- **Policy enforcement**: Validates every clause against your internal policy database to catch non-compliant terms before signature

**Business value**: Reduces contract review time from hours to minutes, catches issues human reviewers might miss, and ensures consistent policy enforcement across all agreements.

## Quickstart

**1. Set up Python environment**

```bash
cd contract-redliner/

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**2. Configure environment variables**

```bash
cp .env.example .env   # add your DD_API_KEY and OPENAI_API_KEY
```

**3. Run the agent**

```bash
python main.py
```

The example script (`main.py`) runs the agent on a sample SaaS contract with intentionally problematic clauses:
- Missing uptime SLA commitments
- Overly broad IP ownership terms
- Unreasonably low liability cap ($100)
- Unrestricted data processing rights
- Immediate termination without notice
- Aggressive payment terms (1 day to suspension)

The agent will flag all six clauses, classify them by risk level, and propose compliant revisions. Output is a structured JSON object with proposals and a risk summary.

## How it works

Two Pydantic AI agents orchestrate the review process:

1. **Segmenter agent**:
   - Classifies the contract type (`nda` / `saas` / `employment` / `vendor`)
   - Splits the document into numbered, discrete clauses for analysis
   - Returns a `DocumentSegment` with doc type and clause list

2. **Redliner agent**:
   - Runs an autonomous tool-calling loop, processing each clause sequentially
   - For every clause:
     - Calls `policy_retrieval()` to fetch relevant policies from the internal database
     - Calls `proposal_tool()` to analyze the clause and generate a revision
   - After all proposals: calls `validation_tool()` for a final consistency check across all edits
   - Returns a complete `RedlineResult` with proposals and risk summary

All agent actions, LLM calls, and tool invocations are automatically traced in Datadog LLM Observability for debugging and performance analysis.

## Policy database

The agent enforces company policies stored in `contract_redliner/primitives/policies.py`. Policies are organized by contract type:

- **NDA**: Confidentiality scope, survival terms, return/destruction, injunctive relief
- **SaaS**: Uptime SLA, data processing (GDPR/CCPA), liability caps, IP ownership, termination
- **Employment**: At-will employment, non-compete restrictions, severance, IP assignment
- **Vendor**: Payment terms, indemnification, termination notice, governing law

Each policy includes:
- **topic**: Short identifier (e.g., `sla_uptime`, `liability_cap`)
- **rule**: Specific requirement text the clause must meet
- **severity**: `critical` / `high` / `medium` / `low` classification

**Customization**: Edit `POLICY_DB` to add your organization's policies. The agent automatically filters policies by contract type and uses keyword matching to retrieve relevant policies for each clause.

## Agent tools

| Tool | What it does | LLM call |
|---|---|---|
| `policy_retrieval(clause_topic)` | Keyword-filters `POLICY_DB` for the doc type | — |
| `proposal_tool(clause_index, clause_topic)` | Analyzes the clause against policies, proposes a rewrite | `generate_proposal()` → `ProposalResult` |
| `validation_tool(proposals)` | Holistic review of all proposals for consistency | `generate_validation()` → `ValidationResult` |

## Example output

After running `python main.py`, the agent returns a structured JSON object:

```json
{
  "proposals": [
    {
      "clause_index": 0,
      "risk_level": "high",
      "suggested_revision": "Provider commits to a minimum 99.5% monthly uptime SLA...",
      "reasoning": "Missing SLA violates critical policy requirement..."
    },
    {
      "clause_index": 3,
      "risk_level": "high",
      "suggested_revision": "Customer retains all rights to Customer Data...",
      "reasoning": "Current clause grants Provider unrestricted data rights..."
    }
    // ... more proposals
  ],
  "risk_summary": {
    "high": 4,
    "medium": 1,
    "low": 1
  }
}
```

Each proposal includes:
- **clause_index**: Zero-based position in the contract
- **risk_level**: `high` / `medium` / `low` classification
- **suggested_revision**: Specific replacement text that complies with policy
- **reasoning**: Explanation of the issue and why the revision is necessary

## Using the agent in your code

```python
from contract_redliner import run_redliner

contract_text = """
YOUR CONTRACT TEXT HERE
"""

result = run_redliner(contract_text)
print(f"Found {len(result.proposals)} issues")
print(f"Risk summary: {result.risk_summary}")

for proposal in result.proposals:
    print(f"Clause {proposal.clause_index}: {proposal.risk_level} risk")
    print(f"  Issue: {proposal.reasoning}")
    print(f"  Fix: {proposal.suggested_revision}")
```

## Offline evaluation

**Purpose**: Test the agent systematically on labeled data before deploying to production. Offline evaluation answers critical questions: *Does the agent catch all risky clauses? Does it over-flag safe language? Are the proposed revisions actually correct?*

For a contract review agent, missing a high-risk clause (false negative) is far more costly than over-flagging safe language (false positive) — a missed liability cap or non-compliant data processing term can expose the company to legal action, financial penalties, or failed audits. Offline evaluation quantifies this risk before you rely on the agent for real contracts.

**Run the evaluation**:

```bash
python experiment.py
```

This executes the agent on 20 labeled test contracts from `golden_dataset.csv` (covering NDAs, SaaS, employment, and vendor agreements) and measures performance across five dimensions:

1. **clause_recall** (0.0–1.0): Percentage of risky clauses correctly identified
   - *What you learn*: **Primary safety metric**. Recall = 0.85 means the agent catches 85% of problematic clauses but misses 15% — potentially exposing you to unreviewed risks. In contract review, aim for recall ≥ 0.95.

2. **clause_precision** (0.0–1.0): Percentage of flagged clauses that are actually problematic
   - *What you learn*: How much noise the agent creates. Precision = 0.70 means 30% of flags are false positives, which erodes attorney trust and wastes review time. Balance precision with recall.

3. **proposal_count_delta**: Exact match on number of flagged clauses
   - *What you learn*: Is the agent consistently flagging the right number of issues, or is it missing some / adding noise?

4. **severity_match**: Whether the agent's risk classification matches ground truth
   - *What you learn*: Does the agent correctly prioritize issues? If it marks critical data processing violations as "low risk," reviewers will miss them.

5. **revision_quality** (1–5, LLM-as-judge): How closely the suggested revision matches expected legal language
   - *What you learn*: Are the proposed clause rewrites actually usable, or do they require heavy editing? Score ≥ 3 is considered acceptable.

**Results & iteration**: All experiment runs and per-evaluator metrics are sent to Datadog LLM Observability under the experiment name `contract-redliner-eval`. Compare runs to track improvement across iterations.

**Ideas to test in offline evaluation**:

1. **Upgrade the model** (contract_redliner/agent.py:31)
   - Switch from `gpt-4` → `gpt-5` for better reasoning
   - Hypothesis: Larger models may catch subtle policy violations that smaller models miss
   - Measure: Does recall improve? Does revision_quality increase?

2. **Improve the redliner agent prompt** (contract_redliner/agent.py:57-68)
   - Add explicit instruction: "If a clause looks compliant on first read, reread it carefully against each policy requirement before deciding."
   - Add few-shot examples: "Example: Clause 'Provider will make reasonable efforts' violates SLA policy requiring 99.5% commitment."
   - Hypothesis: Better instructions reduce false negatives (missed risky clauses)
   - Measure: Does recall improve without hurting precision?

3. **Refine tool definitions** (contract_redliner/agent.py:76-136)
   - Add more detailed docstrings with examples: `policy_retrieval(clause_topic)` → "e.g., 'liability cap', 'data processing', 'termination notice'"
   - Add constraints: "clause_topic must be a 2-5 word description, not a full sentence"
   - Hypothesis: Clearer tool interfaces reduce tool misuse (e.g., agent passing full clause text instead of topic)
   - Measure: Check traces — are tools being called correctly? Does recall improve?

4. **Enhance tool prompts** (contract_redliner/tools/tools.py:19-53, 56-95)
   - In `generate_proposal()`: Add explicit instruction to flag clauses even if the violation is subtle: "If in doubt, flag it — a human will review."
   - In `generate_validation()`: Add instruction to escalate if any high-risk clause is missing from proposals
   - Hypothesis: More conservative flagging increases recall (fewer missed issues)
   - Measure: Does recall improve? Does precision drop (more false positives)?

5. **Add policy examples** (contract_redliner/primitives/policies.py)
   - Extend each `Policy` with a `bad_example` and `good_example` field
   - Pass examples to `generate_proposal()` to ground the LLM's judgment
   - Hypothesis: Concrete examples reduce ambiguity in policy interpretation
   - Measure: Does severity_match improve? Does revision_quality increase?

Use the Datadog UI to drill into individual traces where recall or precision failed, inspect the agent's tool calls, and understand *why* it missed a clause or over-flagged safe language. Compare experiments side-by-side to validate which changes actually moved the metrics.

## Observability

All execution is automatically traced in Datadog LLM Observability under `ml_app:contract-redliner`.

**Trace structure**:
```
[agent]  ContractRedliner               ← root span
  [task]   _segment                     ← document classification + splitting
  [tool]   policy_retrieval (×N)        ← per clause, keyword search
  [tool]   proposal_tool (×N)           ← per clause, LLM analysis
    [llm]    generate_proposal           ← OpenAI structured output
  [tool]   validation_tool              ← once, after all proposals
    [llm]    generate_validation         ← OpenAI structured output
```

An external evaluation (`clauses_with_issues`) is submitted to the root span after each run:
- **score**: Number of flagged clauses
- **assessment**: `pass` (clean contract, 0 issues) / `fail` (issues found)

## File structure

```
contract-redliner/
├── contract_redliner/  Main package
│   ├── __init__.py     Package exports (run_redliner, clauses_with_issues)
│   ├── agent.py        Agent definitions, tool registrations, run_redliner()
│   ├── evaluators.py   clauses_with_issues — external eval submitted to Datadog
│   ├── primitives/     Core data models and policy definitions
│   │   ├── __init__.py Package exports
│   │   ├── models.py   ContractDeps, DocumentSegment, ProposalResult, RedlineResult
│   │   └── policies.py POLICY_DB (nda / saas / employment / vendor) + get_policies()
│   └── tools/          LLM-powered tools for analysis
│       ├── __init__.py Package exports
│       └── tools.py    generate_proposal() + generate_validation() — @llm decorated
├── main.py             Entry point: LLMObs.enable() + example contract
├── experiment.py       Offline evaluation pipeline
├── golden_dataset.csv  Test dataset for experiments
├── requirements.txt    Python dependencies
├── .env.example        Environment variables template
└── .gitignore          Excludes .env, .venv, and Python artifacts
```

## Environment variables

```
DD_API_KEY=...
DD_SITE=datadoghq.com
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-nano-2025-08-07
```
