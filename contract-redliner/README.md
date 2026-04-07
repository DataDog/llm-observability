# Contract Redliner

An AI-powered contract review agent that automates legal due diligence by analyzing contracts against company policies, identifying risk areas, and proposing specific clause revisions. Built with Pydantic AI and traced end-to-end with Datadog LLM Observability.

## Quickstart

**1. Set up Python environment**

```bash
cd contract-redliner/

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install ddtrace "pydantic-ai-slim[openai]"
```

**2. Configure environment variables**

Set these env vars:
```bash
export DD_API_KEY=your_datadog_api_key_here
export DD_APP_KEY=your_datadog_app_key_here
export OPENAI_API_KEY=your_openai_api_key_here
```

If you need to use a different value for `DD_SITE` than the 
default `datadoghq.com`, also set:

```bash
export DD_SITE=your_datadog_site
```


**3. Run the agent**

```bash
python main.py
```

The example script (`main.py`) runs the agent on a sample SaaS contract with intentionally problematic clauses. 
All agent actions, LLM calls, and tool invocations are automatically traced in Datadog LLM Observability for debugging and performance analysis.

### Example output

After running `python main.py`, the agent returns a structured JSON object:

```json
original_clause='1. SERVICE LEVELS. Provider will use commercially reasonable efforts to make the Service available. No specific uptime commitment is made. Downtime credits are not provided under any circumstances.'

revision={
  "reasoning": "Clause disclaims any uptime commitment and provides no downtime credits, which conflicts with policy requiring a 99.5% monthly uptime SLA and defined downtime credits. Revise to include minimum uptime and credits while keeping language largely similar and not materially longer.",
  "revised_clause": "1. SERVICE LEVELS. Provider will use commercially reasonable efforts to make the Service available. Provider shall meet a minimum 99.5% monthly uptime SLA, measured on a monthly basis. If Provider fails to meet the SLA, Customer will receive downtime credits of at least 10% of the applicable monthly service fees for each full hour (or pro-rated, where applicable) of excess downtime during the month in which the SLA is not met.",
  "risk_level": "high"
}
```

## Offline evaluation

**Run the evaluation**:

```bash
python experiment.py
```

This executes the agent on 20 labeled test contracts from `golden_dataset.csv` (covering NDAs, SaaS, employment, and vendor agreements) and measures performance across five dimensions:

1. **clause_recall**: Percentage of risky clauses correctly identified

2. **clause_precision**: Percentage of flagged clauses that are actually problematic

4. **severity_accuracy**: Accuracy of the agent's risk classification when it flagged a clause

5. **revision_quality**: For flagged cases that are actually problematic, how closely the suggested revision matches expected legal language (1-5 scale)

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
