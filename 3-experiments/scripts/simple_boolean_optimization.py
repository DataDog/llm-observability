"""Automated Prompt Optimization for Hallucination Detection (Boolean Classification).

CONTEXT AND USE CASE
====================

You have production LLM traffic (e.g., a chatbot, Q&A system, or AI assistant) and you want
to deploy an automated evaluation system that detects hallucinations in the model's responses.

WORKFLOW
========

1. **Data Collection**: Capture representative samples of your LLM's inputs and outputs from
   production traffic

2. **Manual Annotation**: Create a golden dataset by manually reviewing examples and labeling
   them as hallucination (True) or not hallucination (False). This is your ground truth.

3. **Dataset Upload**: Upload your annotated dataset to Datadog LLM Observability using
   `LLMObs.create_dataset()` or the UI.

4. **Run This Script**: Execute this prompt optimization script to automatically:
   - Test your initial detection prompt on the golden dataset
   - Analyze successes and failures
   - Iteratively improve the prompt using an LLM reasoning model
   - Track performance metrics (precision, recall, accuracy, FPR)
   - Stop when target metrics are achieved

5. **Deploy**: Once optimized, deploy the best performing prompt to production for
   automated hallucination detection on live traffic.

WHAT THIS SCRIPT DOES
=====================

This script uses Datadog's Prompt Optimization framework to:
- Load your annotated hallucination dataset
- Define a boolean classification task (hallucination: yes/no)
- Run experiments with different prompt variations
- Use AI-powered optimization (metaprompting) to improve the prompt
- Track metrics across iterations (F1-score, precision, recall)
- Output the best performing prompt for production use

REQUIREMENTS
============

- OpenAI API key (for running the evaluation model and optimization model)
- Datadog API key and App key (for LLM Observability)
- An uploaded dataset with boolean labels

EXAMPLE DATASET FORMAT
======================

Record structure:
{
    "input_data": [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Berlin."}
    ],
    "expected_output": true  # true = hallucination detected, false = no hallucination
}

CUSTOMIZATION
=============

This script works for any boolean output use case.

Adjust these variables to fit your use case:
- DATASET_NAME: Name of your uploaded dataset
- INITIAL_PROMPT: Starting prompt describing problem. i.e. Detect hallucination
- EVALUATION_MODEL_NAME: Model that will perform detection in production
- OPTIMIZATION_MODEL_NAME: Reasoning model for prompt improvement
- MAX_ITERATION: Maximum optimization iterations
- stopping_condition(): Target metrics (precision, accuracy) for early stopping

"""

import json
import os

from ddtrace.llmobs import LLMObs
from openai import OpenAI
from pydantic import BaseModel

# Prevent ddtrace connection refused error
os.environ["DD_TRACE_ENABLED"] = "false"

# Experiment variables
ML_APP = "YOUR_ML_APP"
EXPERIMENT_NAME = "po_test_gpt4o_mini_simple"
EVALUATION_MODEL_NAME = "gpt-4o-mini"
JOBS = 20
RUNS = 1

# Prompt Optimizer variables
MAX_ITERATION = 20
OPTIMIZATION_MODEL_NAME = "o3-mini"

# Dataset variables
PROJECT_NAME = "YOUR_PROJECT_NAME"
DATASET_NAME = "YOUR_DATASET_NAME"
INITIAL_PROMPT = "YOUR_INITIAL_PROMPT"


class BooleanEvaluationResult(BaseModel):
    """Pydantic model for evaluation results."""
    value: bool
    reasoning: str

    @classmethod
    def output_format(cls) -> str:
        """Return JSON schema for output format."""
        return json.dumps(
            {
                "value": "boolean: true or false evaluation result",
                "reasoning": "string: detailed explanation for the evaluation decision"
            },
            indent=3
        )

class OptimizationResult(BaseModel):
    """Pydantic model for optimization results."""
    prompt: str


# You can change this function when you're using another provider
def boolean_task_function(input_data, config):
    """Execute the prediction task on a single input.

    This function represents your production evaluation logic. It takes a
    conversation or text input and returns a boolean prediction along with
    reasoning for the decision.

    The function is called once per dataset record during optimization to test
    how well the current prompt performs.

    Note:
        This uses structured outputs (response_format=BooleanEvaluationResult)
        to ensure consistent, parseable responses from the LLM.
    """
    client = OpenAI()

    # Handle both direct input format and nested format from dataset
    if isinstance(input_data, list):
        formatted_messages = "\n".join([
            f"[{msg['role']}]: {msg['content']}"
            for msg in input_data
            if msg['role'] != 'system'
        ])
    else:
        formatted_messages = input_data

    system_prompt = config["prompt"]
    user_prompt = f"Data:\n{formatted_messages}"
    model_name = EVALUATION_MODEL_NAME

    response = client.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format=BooleanEvaluationResult,
    )

    return response.choices[0].message.parsed


# You can change this function when you're using another provider
def optimization_task_function(system_prompt: str, user_prompt: str, config: dict):
    """Generate an improved detection prompt.

    This function is called after each iteration to analyze successes and
    failures and propose improvements. It uses a reasoning model (o3-mini) to
    understand why the current prompt is failing and suggest a better version.

    The Datadog framework automatically constructs the system_prompt (optimization
    instructions) and user_prompt (current performance data, failure examples) and
    passes them to this function.

    Returns:
        str: The improved detection prompt to test in the next iteration
    """
    client = OpenAI()

    response = client.chat.completions.parse(
        model=OPTIMIZATION_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=OptimizationResult,
    )

    return response.choices[0].message.parsed.prompt


def boolean_evaluator_function(input_data, output_data, expected_output):
    """Classify each prediction into confusion matrix categories.

    These labels are used in the Experiment UI to filter examples and observe
    their distributions
    """
    prediction = output_data.value if hasattr(output_data, 'value') else output_data

    if prediction and expected_output:
        return "true_positive"
    elif prediction and not expected_output:
        return "false_positive"
    elif not prediction and expected_output:
        return "false_negative"
    else:
        return "true_negative"


def labelization_function(individual_result):
    """Categorize results to show diverse examples to the optimization LLM.

    This function is critical for prompt optimization. It labels each result as
    "GOOD EXAMPLE" or "BAD EXAMPLE" based on whether the prediction was correct.

    The optimizer uses these labels to:
    1. Select one random example from each category
    2. Show the optimization LLM both successful and failed cases
    3. Help the LLM understand what works and what doesn't

    By seeing both good and bad examples, the optimization model can identify patterns
    and suggest targeted improvements.

    Note:
        The number of distinct labels determines diversity of examples shown to the
        optimizer. Keep label cardinality low (<10 categories) for best results.
        Label names should be meaningful as they're shown directly to the LLM.
    """
    eval_value = individual_result["evaluations"]["boolean_evaluator_function"]["value"]

    if eval_value in ("true_positive", "true_negative"):
        return "GOOD EXAMPLE"
    else:  # false_positive or false_negative
        return "BAD EXAMPLE"

def boolean_summary_evaluator(inputs, outputs, expected_outputs, evaluations):
    """Calculate aggregate performance metrics across the entire dataset."""
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, prediction in enumerate(outputs):
        pred_value = prediction.value if hasattr(prediction, 'value') else prediction
        expected_value = expected_outputs[i]

        if pred_value and expected_value:
            true_positives += 1
        elif pred_value and not expected_value:
            false_positives += 1
        elif not pred_value and expected_value:
            false_negatives += 1
        else:  # not pred_value and not expected_value
            true_negatives += 1

    # Calculate metrics
    total_positives = true_positives + false_positives
    precision = true_positives / total_positives if total_positives > 0 else 0.0

    actual_positives = true_positives + false_negatives
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0

    total_samples = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0.0

    actual_negatives = false_positives + true_negatives
    fpr = false_positives / actual_negatives if actual_negatives > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
    }


def compute_score(summary_evaluators) -> float:
    """Compute optimization score for ranking iterations (Here F1-Score).

    The optimization framework uses this score to determine which iteration performed
    best. After all iterations complete, the iteration with the highest score is
    selected as the final result.

    Note:
        You can customize this to optimize for different metrics.
    """
    # Implementing F1-Score
    precision = summary_evaluators['boolean_summary_evaluator']['value']['precision']
    recall = summary_evaluators['boolean_summary_evaluator']['value']['recall']
    return 2 * (precision * recall) / (precision + recall)


def stopping_condition(summary_evaluators) -> bool:
    """Determine whether to stop optimization early (before MAX_ITERATION).

    If this function returns True, optimization stops immediately and returns the
    current best prompt. This saves time and API costs when target metrics are
    already achieved.
    """
    precision_condition = summary_evaluators['boolean_summary_evaluator']['value']['precision'] >= 0.9
    accuracy_condition = summary_evaluators['boolean_summary_evaluator']['value']['accuracy'] >= 0.8

    return precision_condition and accuracy_condition


def main():
    """This function orchestrates the entire optimization process:

    1. **Initialize**: Enable Datadog LLM Observability with project settings
    2. **Load Data**: Pull your annotated hallucination dataset from Datadog
    3. **Configure**: Set up the optimization with:
       - Task function (how to run detection)
       - Optimization task (how to improve the prompt)
       - Evaluators (how to measure performance)
       - Scoring (how to rank iterations)
       - Stopping condition (when to stop early)
    4. **Optimize**: Run iterative optimization (up to MAX_ITERATION times):
       - Test current prompt on entire dataset (parallel execution via jobs=JOBS)
       - Compute metrics (precision, recall, accuracy, FPR)
       - Analyze failures and generate improvement suggestions
       - Test improved prompt
       - Repeat until stopping condition met or max iterations reached
    5. **Output**: Display the best performing prompt and all experiment URLs

    The optimization runs in parallel (jobs=20) for fast execution. Each iteration
    is tracked in Datadog LLM Observability for analysis and comparison.
    """
    LLMObs.enable(
        ml_app=ML_APP,
        project_name=PROJECT_NAME
    )

    # Load dataset
    print(f"Loading dataset '{DATASET_NAME}' from LLMObs...")
    dataset = LLMObs.pull_dataset(dataset_name=DATASET_NAME)
    print(f"Loaded dataset with {len(dataset)} records\n")

    # Run prompt optimization
    prompt_optimization = LLMObs._prompt_optimization(
        name=EXPERIMENT_NAME,
        dataset=dataset,
        task=boolean_task_function,
        optimization_task=optimization_task_function,
        evaluators=[boolean_evaluator_function],
        compute_score=compute_score,
        summary_evaluators=[boolean_summary_evaluator],
        labelization_function=labelization_function,
        stopping_condition=stopping_condition,
        max_iterations=MAX_ITERATION,
        config={
            # Mandatory
            "prompt": INITIAL_PROMPT,
            # Optionals
            "model_name": EVALUATION_MODEL_NAME,
            "evaluation_output_format": BooleanEvaluationResult.output_format(),
            "runs": RUNS,
        }
    )

    result = prompt_optimization.run(jobs=JOBS)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nBest prompt:\n{result.best_prompt}")

    # Print all experiment URLs for reference
    print(f"\n{'=' * 80}")
    print("ALL EXPERIMENT URLS")
    print(f"{'=' * 80}")
    print(result.summary())

if __name__ == "__main__":
    main()
