"""Entry point for the contract redliner demo.

Run from inside the contract_redliner/ directory:
  python main.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from ddtrace.llmobs import LLMObs

LLMObs.enable(
    ml_app="contract-redliner",
    api_key=os.environ["DD_API_KEY"],
    site=os.environ.get("DD_SITE", "datadoghq.com"),
    agentless_enabled=True,
)

from contract_redliner.agent import run_redliner  # noqa: E402 — import after LLMObs.enable()
from contract_redliner.evaluators import clauses_with_issues

EXAMPLE_CONTRACT = """
SOFTWARE AS A SERVICE AGREEMENT

This Software as a Service Agreement ("Agreement") is entered into as of the Effective Date
between Acme Corp ("Provider") and Customer.

1. SERVICE LEVELS. Provider will use commercially reasonable efforts to make the Service
available. No specific uptime commitment is made. Downtime credits are not provided under
any circumstances.

2. INTELLECTUAL PROPERTY. All improvements, modifications, or derivative works created by
Customer using the Service shall be jointly owned by both parties. Provider may use any
feedback provided by Customer to improve the Service without restriction or compensation.

3. LIMITATION OF LIABILITY. IN NO EVENT SHALL PROVIDER BE LIABLE FOR ANY INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES. Provider's total aggregate
liability for any claims arising under this Agreement shall not exceed $100 USD.

4. DATA PROCESSING. Customer grants Provider a perpetual, irrevocable license to use,
process, and disclose Customer Data for any purpose, including improving Provider's products,
marketing, and sharing with third-party partners without restriction.

5. TERMINATION. Either party may terminate this Agreement immediately upon written notice
for any reason or no reason, without penalty or further obligation.

6. PAYMENT. Customer shall pay all fees within 90 days of invoice. Late payments accrue
interest at 5% per month. Provider may suspend Service after 1 day of non-payment.
"""


if __name__ == "__main__":
    print("Running contract redliner demo...")
    print("Input contract:")
    print(EXAMPLE_CONTRACT)

    result, span_ctx = run_redliner(EXAMPLE_CONTRACT)

    count, assessment, reasoning = clauses_with_issues(result.model_dump())
    LLMObs.submit_evaluation(
        span=span_ctx,
        label="clauses_with_issues",
        metric_type="score",
        value=count,
        assessment=assessment,
        reasoning=reasoning,
    )

    print("Output:")
    print(result.model_dump_json(indent=2))
