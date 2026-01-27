import json
import os

from ddtrace.llmobs import LLMObs
from openai import OpenAI
from pydantic import BaseModel

# Prevent ddtrace connection refused error
os.environ["DD_TRACE_ENABLED"] = "false"

# Experiment variables
ML_APP = "PromptOptimization"
EXPERIMENT_NAME = "po_test_gpt4o_mini_simple"
EVALUATION_MODEL_NAME = "gpt-4o-mini"
JOBS = 20

# Prompt Optimizer variables
MAX_ITERATION = 20
OPTIMIZATION_MODEL_NAME = "o3-mini"

# Dataset variables
PROJECT_NAME = "PromptOptimization"
DATASET_NAME = "hallucination_boolean"
INITIAL_PROMPT = "Detect hallucination"

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


def boolean_task_function(input_data, config):
    """Call the model to make the prediction"""
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


def optimization_task_function(system_prompt: str, user_prompt: str, config: dict):
    """Call LLM to generate an improved prompt based on evaluation results.

    Args:
        system_prompt: Instructions for the optimization LLM
        user_prompt: Current prompt and performance data
        model: Model name to use for optimization

    Returns:
        dict: Contains "new_prompt" and "reasoning" keys
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
    """Evaluate prediction against expected output.

    Args:
        input_data: Input to the task
        output_data: Output from the task (EvaluationResult)
        expected_output: Expected boolean value

    Returns:
        str: Classification label (true_positive, false_positive, etc.)
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
    """Categorize individual results into good or bad examples.

    Args:
        individual_result: Dict containing "evaluations" key with evaluator results.

    Returns:
        str: "GOOD EXAMPLE" for correct predictions, "BAD EXAMPLE" for incorrect ones.
    """
    eval_value = individual_result["evaluations"]["boolean_evaluator_function"]["value"]

    if eval_value in ("true_positive", "true_negative"):
        return "GOOD EXAMPLE"
    else:  # false_positive or false_negative
        return "BAD EXAMPLE"

def boolean_summary_evaluator(inputs, outputs, expected_outputs, evaluations):
    """Calculate precision, recall, accuracy, and FPR across all evaluations.

    Metrics:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - FPR = FP / (FP + TN)

    Args:
        inputs: List of input data
        outputs: List of task outputs
        expected_outputs: List of expected outputs
        evaluations: List of evaluation results

    Returns:
        dict: Contains "precision", "recall", "accuracy", and "fpr" keys
    """
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


def accuracy_summary_evaluator(inputs, outputs, expected_outputs, evaluations):
    """Calculate accuracy"""
    good_predictions = 0
    for i, prediction in enumerate(outputs):
        pred_value = prediction.value if hasattr(prediction, 'value') else prediction
        expected_value = expected_outputs[i]

        if is_correct(pred_value, expected_value):
            good_predictions += 1

    return {"accuracy": good_predictions / len(outputs)}


def compute_score(summary_evaluators) -> float:
    precision = summary_evaluators['boolean_summary_evaluator']['value']['precision']
    accuracy = summary_evaluators['boolean_summary_evaluator']['value']['accuracy']
    return precision + accuracy


def stopping_condition(summary_evaluators) -> bool:
    """
    [
        {
            "boolean_summary_evaluator": {
                "value": {
                    "accuracy": 0.3
                    "precision": 0.5
                    "recall": 0.2
                    "fpr": 0.8
                },
                "error": "None"
            }
        }
    ]
    """

    precision_condition = summary_evaluators['boolean_summary_evaluator']['value']['precision'] >= 0.9
    accuracy_condition = summary_evaluators['boolean_summary_evaluator']['value']['accuracy'] >= 0.8

    return precision_condition and accuracy_condition


def main():
    """Run prompt optimization experiment."""
    # Enable LLMObs
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
            "runs": 1,
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
