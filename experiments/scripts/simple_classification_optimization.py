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
DATASET_NAME = "suggest_action"
INITIAL_PROMPT = "Predict if one of these actions will be in the output: 'RESOLVE', 'IGNORE', 'ASSIGN', 'TICKET', 'CODE_FIX', 'SCALE UP', 'CONFIG_CHANGE'"


class ClassificationEvaluationResult(BaseModel):
    """Pydantic model for evaluation results."""
    value: str
    reasoning: str

    @classmethod
    def output_format(cls) -> str:
        """Return JSON schema for output format."""
        return json.dumps(
            {
                "value": "str: class predicted: 'RESOLVE', 'IGNORE', 'ASSIGN', 'TICKET',  'CODE_FIX' 'SCALE UP' 'CONFIG_CHANGE'",
                "reasoning": "string: detailed explanation for the evaluation decision"
            },
            indent=3
        )


class OptimizationResult(BaseModel):
    """Pydantic model for optimization results."""
    prompt: str


def classification_task_function(input_data, config):
    """Call GPT-4o mini to predict if conversation is incomplete."""
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


def optimization_task_function(system_prompt: str, user_prompt: str, config: dict):
    """Call LLM to generate an improved prompt based on evaluation results.

    Args:
        system_prompt: Instructions for the optimization LLM
        user_prompt: Current prompt and performance data
        model: Model name to use for optimization

    Returns:
        str: the optimized prompt
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
    """Evaluate prediction against expected output.

    Args:
        input_data: Input to the task
        output_data: Output from the task (EvaluationResult)
        expected_output: Expected boolean value

    Returns:
        bool: whether the evaluation result is in the expected output
    """
    return output_data.value in expected_output


def labelization_function(individual_result):
    """Categorize individual results into good or bad examples.

    Args:
        individual_result: Dict containing "evaluations" key with evaluator results.

    Returns:
        str: "GOOD EXAMPLE" for correct predictions, "BAD EXAMPLE" for incorrect ones.
    """
    eval_value = individual_result["evaluations"]["classification_evaluator_function"]["value"]
    if eval_value:
        return "GOOD EXAMPLE"
    else:
        return "BAD EXAMPLE"


def accuracy_summary_evaluator(inputs, outputs, expected_outputs, evaluations):
    """Calculate accuracy"""
    good_predictions = 0
    for i, prediction in enumerate(outputs):
        expected_value = expected_outputs[i]
        if prediction.value in expected_value:
            good_predictions += 1

    return {"accuracy": good_predictions / len(outputs)}

def is_correct(output, expected_output):
    return expected_output in output

def best_iteration_computation(summary_evaluators) -> float:
    """Compute score for iteration based on precision and accuracy.

    Returns sum of precision and accuracy (max score = 2.0).
    """
    return summary_evaluators['accuracy_summary_evaluator']['value']['accuracy']


def stopping_condition(summary_evaluators) -> bool:
    """Check if optimization should stop.

    Stops when accuracy >= 0.95.
    """
    # For accuracy_summary_evaluator
    accuracy = summary_evaluators['accuracy_summary_evaluator']['value']['accuracy']
    return accuracy >= 0.95


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
            "model_name": EVALUATION_MODEL_NAME,
            "evaluation_output_format": ClassificationEvaluationResult.output_format(),
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
