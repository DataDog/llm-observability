from __future__ import annotations

import json
from typing import Any

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs._evaluators import (
    BaseEvaluator,
    BooleanStructuredOutput,
    EvaluatorContext,
    EvaluatorResult,
    LLMJudge,
    ScoreStructuredOutput,
)

from src.models import PortfolioBriefing


# --- Programmatic evaluator: completeness ---


class CompletenessEvaluator(BaseEvaluator):
    """Check that every requested ticker appears in the output.

    Works in both modes: pass ``requested_tickers`` explicitly (the standalone
    ``run_evaluations`` path), or omit it and the requested tickers are read from the
    evaluation context — so it can be attached to the inline-experiment boundary via
    ``experiment_start(evaluators=...)``, where each replayed case supplies its own
    ``input_data`` (``{"tickers": [...]}``).
    """

    def __init__(self, requested_tickers: list[str] | None = None):
        super().__init__(name="completeness")
        self.requested_tickers = requested_tickers

    def _resolve_tickers(self, context: EvaluatorContext) -> list[str]:
        if self.requested_tickers is not None:
            return self.requested_tickers
        data = context.input_data
        if isinstance(data, dict):
            return data.get("tickers", []) or []
        if isinstance(data, str):
            return [t.strip() for t in data.split(",") if t.strip()]
        return []

    def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        output = context.output_data
        result_tickers = {a["ticker"].upper() for a in output.get("analyses", [])}
        requested = {t.upper() for t in self._resolve_tickers(context)}
        missing = requested - result_tickers
        passed = len(missing) == 0

        return EvaluatorResult(
            value=passed,
            assessment="pass" if passed else "fail",
            reasoning=(
                "All requested tickers present in output."
                if passed
                else f"Missing tickers: {', '.join(sorted(missing))}"
            ),
        )


# --- LLM judge: sentiment consistency ---

sentiment_judge = LLMJudge(
    name="sentiment_consistency",
    provider="openai",
    model="gpt-4o-mini",
    user_prompt=(
        "Review these stock analyses. For each stock, check whether the sentiment label "
        "(bullish/bearish/neutral) is consistent with the summary and key factors.\n\n"
        "Rules:\n"
        "- 'neutral' is valid when there are mixed positive and negative signals\n"
        "- Only flag clear contradictions (e.g., 'bullish' but summary describes major losses)\n"
        "- When in doubt, consider it consistent\n\n"
        "Analyses:\n{{output_data}}"
    ),
    structured_output=BooleanStructuredOutput(
        description="Whether all sentiment labels are consistent with their narratives",
        reasoning=True,
        pass_when=True,
    ),
)


# --- LLM judge: factual grounding ---

grounding_judge = LLMJudge(
    name="factual_grounding",
    provider="openai",
    model="gpt-4o-mini",
    user_prompt=(
        "Rate the factual grounding of these stock analyses.\n\n"
        "1 = Entirely vague, no specific facts cited\n"
        "2 = Mostly vague with occasional specifics\n"
        "3 = Mix of vague and specific claims\n"
        "4 = Mostly grounded with concrete numbers, dates, and events\n"
        "5 = Thoroughly grounded with specific revenue figures, dates, named events throughout\n\n"
        "Analyses:\n{{output_data}}"
    ),
    structured_output=ScoreStructuredOutput(
        description="How well-grounded the analyses are in specific, verifiable facts",
        min_score=1,
        max_score=5,
        min_threshold=3,
        reasoning=True,
    ),
)


def _submit(
    evaluator_name: str,
    result: EvaluatorResult,
    span_context: dict[str, Any],
    metric_type: str,
) -> None:
    """Submit an evaluation result to LLMObs."""
    LLMObs.submit_evaluation(
        label=evaluator_name,
        metric_type=metric_type,
        value=result.value,
        span=span_context,
        assessment=result.assessment,
        reasoning=result.reasoning,
    )


def run_evaluations(
    briefing: PortfolioBriefing,
    requested_tickers: list[str],
    span_context: dict[str, Any],
) -> None:
    """Run all evaluations on the portfolio briefing and submit results to LLMObs."""
    output_data = json.loads(briefing.model_dump_json())

    context = EvaluatorContext(
        input_data=", ".join(requested_tickers),
        output_data=output_data,
        span_id=span_context["span_id"],
        trace_id=span_context["trace_id"],
    )

    # Programmatic: completeness
    completeness = CompletenessEvaluator(requested_tickers)
    _submit("completeness", completeness.evaluate(context), span_context, "boolean")

    # LLM judge: sentiment consistency
    _submit("sentiment_consistency", sentiment_judge.evaluate(context), span_context, "boolean")

    # LLM judge: factual grounding
    _submit("factual_grounding", grounding_judge.evaluate(context), span_context, "score")
