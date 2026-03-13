"""Automated Prompt Optimization for Action Classification (Multi-class Classification).

CONTEXT AND USE CASE
====================

You have an incident management or alerting system that processes events, errors, or
incidents. You want to automatically classify what action should be taken for each
incident (e.g., RESOLVE, IGNORE, ASSIGN).

WORKFLOW
========

1. **Data Collection**: Gather historical incidents and their resolutions from your
   system. Collect the incident description, context, metrics, and what action was
   actually taken to resolve it.

2. **Manual Annotation**: Create a golden dataset by labeling each incident with the
   correct label(s) that should be suggested. Note that some incidents might have
   multiple valid labels (e.g., ["ASSIGN", "RESOLVE"]).

3. **Dataset Upload**: Upload your annotated dataset to Datadog LLM Observability.
   Each record should have the incident details as input and the correct action(s)
   as expected output.

4. **Run This Script**: Execute this prompt optimization script to automatically:
   - Test your initial classification prompt on the golden dataset
   - Analyze misclassifications and confusion patterns
   - Iteratively improve the prompt using an LLM reasoning model
   - Track accuracy across iterations
   - Stop when target accuracy is achieved

5. **Deploy**: Deploy the optimized prompt to production to automatically suggest
   labels for new incidents.

WHAT THIS SCRIPT DOES
=====================

This script uses Datadog's Prompt Optimization framework to:
- Load your annotated classification dataset
- Define a multi-class classification task
- Run experiments with different prompt variations
- Use AI-powered optimization to improve classification accuracy
- Track accuracy across iterations
- Output the best performing prompt for production use

REQUIREMENTS
============

- OpenAI API key (for running the evaluation model and optimization model)
- Datadog API key and App key (for LLM Observability)
- An uploaded dataset with labels

EXAMPLE DATASET FORMAT
======================

Record structure:
{
    "input_data": "Error: Database connection timeout after 30s. CPU usage normal, memory at 85%.",
    "expected_output": ["ASSIGN"]
}

Or conversation format:
{
    "input_data": [
        {"role": "user", "content": "We're seeing high latency on API endpoints"},
        {"role": "assistant", "content": "Checking metrics..."}
    ],
    "expected_output": ["RESOLVE", "TICKET"]  # Multiple valid actions possible
}

CUSTOMIZATION
=============

Adjust these variables to fit your use case:
- DATASET_NAME: Name of your uploaded dataset
- INITIAL_PROMPT: Starting prompt for classification
- EVALUATION_MODEL_NAME: Model that will perform classification in production
- OPTIMIZATION_MODEL_NAME: Reasoning model for prompt improvement
- MAX_ITERATION: Maximum optimization iterations
- stopping_condition(): Target metrics for early stopping

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

class ClassificationEvaluationResult(BaseModel):
    """Pydantic model for evaluation results."""
    value: str
    reasoning: str

    @classmethod
    def output_format(cls) -> str:
        """Return JSON schema for output format."""
        return json.dumps(
            {
                "value": "str: class predicted",
                "reasoning": "string: detailed explanation for the evaluation decision"
            },
            indent=3
        )


class OptimizationResult(BaseModel):
    """Pydantic model for optimization results."""
    prompt: str


# You can change this function when you're using another provider
def classification_task_function(input_data, config):
    """Execute the classification task on a single input.

    This function represents your production classification logic. It takes an incident
    description (or conversation) and predicts which label should be taken, along with
    reasoning for the recommendation.

    The function is called once per dataset record during optimization to test how
    well the current prompt performs at classifying labels.

    Note:
        Uses structured outputs to ensure the model returns a valid label class
        from the predefined set.
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
        response_format=ClassificationEvaluationResult,
    )

    return response.choices[0].message.parsed


# You can change this function when you're using another provider
def optimization_task_function(system_prompt: str, user_prompt: str, config: dict):
    """Generate an improved classification prompt.

    This function is called after each iteration to analyze misclassifications and
    propose improvements. It uses a reasoning model to understand confusion
    patterns and suggest clearer classification criteria.

    The Datadog framework automatically constructs prompts with performance data and
    examples of misclassified incidents.

    Returns:
        str: The improved classification prompt to test in the next iteration
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


def classification_evaluator_function(input_data, output_data, expected_output):
    """Check if the predicted label matches any of the expected labels.

    This label is used in the Experiment UI to filter examples and observe
    its distribution
    """
    return output_data.value in expected_output


def labelization_function(individual_result):
    """Categorize results to show diverse examples to the optimization LLM.

    This function labels each classification result as "GOOD EXAMPLE" (correct action
    predicted) or "BAD EXAMPLE" (wrong label predicted).

    The optimizer uses these labels to:
    1. Select representative examples from each category
    2. Show the optimization LLM both correct and incorrect classifications
    3. Help identify confusion patterns

    For multi-class classification, you could create more granular labels to help
    the optimizer understand specific confusion patterns:
    "CORRECT", "CONFUSED" and "INCORRECT"
    """
    eval_value = individual_result["evaluations"]["classification_evaluator_function"]["value"]
    if eval_value:
        return "GOOD EXAMPLE"
    else:
        return "BAD EXAMPLE"


def accuracy_summary_evaluator(inputs, outputs, expected_outputs, evaluations):
    """Calculate classification accuracy across the entire dataset.

    This summary evaluator computes the percentage of correctly classified incidents.

    **Accuracy** = (Correct Classifications) / (Total Classifications)

    For multi-class problems with multiple valid labels, a prediction is correct if
    it matches any of the expected labels.
    """
    good_predictions = 0
    for i, prediction in enumerate(outputs):
        expected_value = expected_outputs[i]
        if prediction.value in expected_value:
            good_predictions += 1

    return {"accuracy": good_predictions / len(outputs)}


def compute_score(summary_evaluators) -> float:
    """Compute optimization score for ranking iterations (Here F1-Score).

    The optimization framework uses this score to determine which iteration performed
    best. After all iterations complete, the iteration with the highest score is
    selected as the final result.

    Note:
        You can customize this to optimize for different metrics.

    Customization Examples:
        # Weighted accuracy (if some labels are more important)
        accuracy = summary_evaluators['accuracy_summary_evaluator']['value']['accuracy']
        critical_accuracy = summary_evaluators['critical_actions']['value']['accuracy']
        return 0.7 * accuracy + 0.3 * critical_accuracy

        # Macro F1 (if you add per-class metrics)
        return summary_evaluators['macro_f1_evaluator']['value']['f1']

        # Custom business metric
        # Penalize critical misclassifications (e.g., IGNORE when should be CODE_FIX)
    """
    return summary_evaluators['accuracy_summary_evaluator']['value']['accuracy']


def stopping_condition(summary_evaluators) -> bool:
    """Determine whether to stop optimization early (before MAX_ITERATION).

    If this function returns True, optimization stops immediately and returns the
    current best prompt. This is useful when you've achieved your target accuracy
    and don't need to continue iterating.
    """
    accuracy = summary_evaluators['accuracy_summary_evaluator']['value']['accuracy']
    return accuracy >= 0.95


def main():
    """Execute the classification prompt optimization workflow.

    This function orchestrates the entire optimization process:

    1. **Initialize**: Enable Datadog LLM Observability with project settings
    2. **Load Data**: Pull your annotated classification dataset from Datadog
    3. **Configure**: Set up the optimization with:
       - Classification task function (how to predict labels)
       - Optimization task (how to improve the prompt)
       - Evaluators (how to measure correctness)
       - Scoring (how to rank iterations by accuracy)
       - Stopping condition (when to stop early)
    4. **Optimize**: Run iterative optimization (up to MAX_ITERATION times):
       - Test current prompt on entire dataset (parallel execution)
       - Compute accuracy
       - Analyze misclassifications and confusion patterns
       - Generate improved prompt with better classification criteria
       - Test improved prompt
       - Repeat until stopping condition met or max iterations reached
    5. **Output**: Display the best performing prompt and all experiment URLs

    The optimization runs in parallel (jobs=20) for fast execution. Each iteration
    is tracked in Datadog LLM Observability where you can:
    - View per-class performance
    - Analyze confusion patterns
    - Compare prompts across iterations
    - Export the best prompt for production deployment
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
        task=classification_task_function,
        optimization_task=optimization_task_function,
        evaluators=[classification_evaluator_function],
        labelization_function=labelization_function,
        compute_score=best_iteration_computation,
        summary_evaluators=[accuracy_summary_evaluator],
        stopping_condition=stopping_condition,
        max_iterations=MAX_ITERATION,
        config={
            "prompt": INITIAL_PROMPT,
            # Optionals
            "model_name": EVALUATION_MODEL_NAME,
            "evaluation_output_format": ClassificationEvaluationResult.output_format(),
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
