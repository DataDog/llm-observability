
from dataclasses import dataclass
from ddtrace.llmobs import LLMObs
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
    ReportEvaluator,
    ReportEvaluatorContext,
)
from pydantic_evals.reporting.analyses import ScalarResult

import os

LLMObs.enable(
    project_name="pydantic-demo-project",
    api_key=os.environ["DD_API_KEY"],
    app_key=os.environ["DD_APP_KEY"],
    site=os.environ["DD_SITE"],
    agentless_enabled=True,
)

@dataclass
class ComprehensiveCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        format_valid = self._check_format(ctx.output)

        to_return = {
            'valid_format': EvaluationReason(
                value=format_valid,
                reason='Valid JSON format' if format_valid else 'Invalid JSON format',
            ),
            'quality_score': self._score_quality(ctx.output),  
            'category': self._classify(ctx.output),  
        }
        return to_return

    def _check_format(self, output: str) -> bool:
        return output.startswith('{') and output.endswith('}')

    def _score_quality(self, output: str) -> float:
        return len(output) / 100.0

    def _classify(self, output: str) -> str:
        return 'short' if len(output) < 50 else 'long'


dataset = LLMObs.create_dataset(
    dataset_name="capitals-of-the-world",
    description="Questions about world capitals",
    records=[
        {
            "input_data": {"question": "What is the capital of China?", "output": "Beijing"},       # required, JSON or string
            "expected_output": "Beijing",                                      # optional, JSON or string
            "metadata": {"difficulty": "easy"}                                 # optional, JSON
        },
        {
            "input_data": {"question": "What is the capital of China?", "output": "China"},       # required, JSON or string
            "expected_output": "Beijing",                                      # optional, JSON or string
            "metadata": {"difficulty": "easy"}                                 # optional, JSON
        },
        {
            "input_data": {"question": "Which city serves as the capital of South Africa?", "output": "Pretoria"},
            "expected_output": "Pretoria",
            "metadata": {"difficulty": "medium"}
        }
    ]
)

def my_task(input_data, config) -> str:
    return input_data["output"]

class TotalCasesEvaluator(ReportEvaluator):
    def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
        return ScalarResult(
            title='Total',
            value=len(ctx.report.cases),
            unit='cases',
        )

experiment = LLMObs.experiment(
    name="pydantic-demo",
    task=my_task, 
    dataset=dataset,
    evaluators=[ComprehensiveCheck()],
    summary_evaluators=[TotalCasesEvaluator()],
    description="Determine whether the actual output is factually correct based on the expected output.",
)

results = experiment.run()
print(experiment.url)