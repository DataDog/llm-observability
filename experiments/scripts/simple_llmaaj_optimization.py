"""Automated Prompt Optimization for Text Generation (LLM-as-a-Judge).

CONTEXT AND USE CASE
====================

You have a text generation task where you want an LLM to produce descriptions or
summaries based on input data. You want to automatically optimize the generation prompt
by evaluating outputs using another LLM as a judge.

WORKFLOW
========

1. **Data Collection**: Gather representative samples of your input data along with
   any relevant context. This gives the model enough information to understand what
   it should generate.

2. **Manual Annotation**: Create a golden dataset by manually writing high-quality
   expected outputs for representative inputs. Label each with "yes" if it's a good
   example to replicate, or "no" if it's a bad example to avoid.

3. **Dataset Upload**: Upload your annotated dataset to Datadog LLM Observability using
   `LLMObs.create_dataset()` or the UI.

4. **Run This Script**: Execute this prompt optimization script to automatically:
   - Test your initial generation prompt on the golden dataset
   - Use an LLM judge to evaluate similarity between generated and expected outputs
   - Analyze low-scoring outputs and high-scoring outputs
   - Iteratively improve the prompt using an LLM reasoning model
   - Track accuracy metrics (based on similarity scores)
   - Stop when target accuracy is achieved

5. **Deploy**: Once optimized, deploy the best performing prompt to production for
   automated text generation at scale.

WHAT THIS SCRIPT DOES
=====================

This script uses Datadog's Prompt Optimization framework to:
- Load your annotated text generation dataset
- Define a generation task (produce descriptions from input data)
- Use LLM-as-a-Judge evaluation (similarity scoring between generated vs expected)
- Run experiments with different prompt variations
- Use AI-powered optimization (metaprompting) to improve the generation prompt
- Track accuracy across iterations based on similarity thresholds
- Output the best performing prompt for production use

REQUIREMENTS
============

- OpenAI API key (for running the generation model, judge model, and optimization model)
- Datadog API key and App key (for LLM Observability)
- An uploaded dataset with input data and expected descriptions

EXAMPLE DATASET FORMAT
======================

Record structure:
{
    "input_data": {
        "name": "api.request.duration",
        "context": "API endpoint handler that processes user requests",
        "details": "Tracks the time from request receipt to response sent"
    },
    "expected_output": {
        "description": "Measures the time taken to process API requests in seconds",
        "value": "yes",  # "yes" = good example to replicate, "no" = bad example to avoid
        "reasoning": "Clear, concise, specifies unit of measurement"
    }
}

Or simplified format:
{
    "input_data": "Some text input that needs a description",
    "expected_output": {
        "description": "The expected description for this input",
        "value": "yes",
        "reasoning": "Why this is a good description"
    }
}

CUSTOMIZATION
=============

This script works for any text generation task with LLM-as-a-Judge evaluation.

Adjust these variables to fit your use case:
- DATASET_NAME: Name of your uploaded dataset
- INITIAL_PROMPT: Starting prompt for text generation
- EVALUATION_MODEL_NAME: Model that will generate descriptions in production
- JUDGE_MODEL_NAME: Model that will evaluate similarity (can be same or different)
- OPTIMIZATION_MODEL_NAME: Reasoning model for prompt improvement
- MAX_ITERATION: Maximum optimization iterations
- stopping_condition(): Target accuracy threshold for early stopping
- llm_judge_evaluator_function(): Customize the similarity evaluation criteria

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
EXPERIMENT_NAME = "po_test_gpt4o_mini_llm_judge"
EVALUATION_MODEL_NAME = "gpt-4o-mini"
JUDGE_MODEL_NAME = "gpt-4o-mini"
JOBS = 20
RUNS = 1

# Prompt Optimizer variables
MAX_ITERATION = 2
OPTIMIZATION_MODEL_NAME = "o3-mini"

# Dataset variables
PROJECT_NAME = "YOUR_PROJECT_NAME"
DATASET_NAME = "YOUR_DATASET_NAME"
INITIAL_PROMPT = "YOUR_INITIAL_PROMPT"

class DescriptionGenerationResult(BaseModel):
    """Pydantic model for generated description."""
    description: str
    reasoning: str

    @classmethod
    def output_format(cls) -> str:
        """Return JSON schema for output format."""
        return json.dumps(
            {
                "description": "string: clear, concise description",
                "reasoning": "string: explanation of key points in the description"
            },
            indent=3
        )


class SimilarityJudgeResult(BaseModel):
    """Pydantic model for LLM judge similarity evaluation."""
    similarity_score: float
    reasoning: str

    @classmethod
    def output_format(cls) -> str:
        """Return JSON schema for output format."""
        return json.dumps(
            {
                "similarity_score": "float: similarity score between 0.0 and 1.0, where 1.0 means identical meaning and 0.0 means completely unrelated",
                "reasoning": "string: explanation of the similarity assessment"
            },
            indent=3
        )


class OptimizationResult(BaseModel):
    """Pydantic model for optimization results."""
    prompt: str


def description_generation_task_function(input_data, config):
    """Call the model to generate a description."""
    client = OpenAI()

    system_prompt = config["prompt"]
    user_prompt = f"Generate a description for this:\n\n{input_data}"
    model_name = EVALUATION_MODEL_NAME

    response = client.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format=DescriptionGenerationResult,
    )

    return response.choices[0].message.parsed


def optimization_task_function(system_prompt: str, user_prompt: str, config: dict):
    """Call LLM to generate an improved prompt based on evaluation results.

    Args:
        system_prompt: Instructions for the optimization LLM
        user_prompt: Current prompt and performance data
        config: Configuration dictionary

    Returns:
        str: New optimized prompt
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


def llm_judge_evaluator_function(input_data, output_data, expected_output):
    """Use LLM as a judge to evaluate similarity between generated and expected descriptions."""
    client = OpenAI()

    # Extract descriptions
    generated_description = output_data.description if hasattr(output_data, 'description') else str(output_data)
    expected_description = expected_output.get('description', '') if isinstance(expected_output, dict) else str(expected_output)
    expected_value = expected_output.get('value', 'yes') if isinstance(expected_output, dict) else 'yes'

    # Create judge prompt
    # Prompt to be adapted to the current use case
    judge_system_prompt = """You are an expert evaluator comparing two descriptions.

Your task is to assess how similar the generated description is to the expected description in terms of meaning and content.

Return a similarity score between 0.0 and 1.0:
- 1.0 = Identical meaning, same key information
- 0.8-0.9 = Very similar, minor differences in wording
- 0.6-0.7 = Similar core meaning, some missing details
- 0.4-0.5 = Partially similar, significant differences
- 0.2-0.3 = Slightly related, mostly different
- 0.0-0.1 = Completely unrelated
"""

    judge_user_prompt = f"""Generated Description:
{generated_description}

Expected Description:
{expected_description}

Assess the similarity between these descriptions."""

    response = client.chat.completions.parse(
        model=JUDGE_MODEL_NAME,
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_prompt},
        ],
        temperature=0.0,
        response_format=SimilarityJudgeResult,
    )

    judge_result = response.choices[0].message.parsed

    return {
        "similarity_score": judge_result.similarity_score,
        "expected_value": expected_value,
        "reasoning": judge_result.reasoning
    }


def labelization_function(individual_result):
    """Categorize individual results into good or bad examples.

    Args:
        individual_result: Dict containing "evaluations" key with evaluator results.

    Returns:
        dict: Contains "value" (label) and "extra" (reasoning) keys.
    """
    eval_result = individual_result["evaluations"]["llm_judge_evaluator_function"]["value"]
    similarity_score = eval_result["similarity_score"]
    expected_value = eval_result["expected_value"]

    # Convert expected_value to boolean
    if isinstance(expected_value, str):
        is_good_expected = expected_value.lower() in ("yes", "true", "1")
    else:
        is_good_expected = bool(expected_value)

    if is_good_expected:
        # Expected a good description - high similarity is good
        label = "GOOD EXAMPLE" if similarity_score >= 0.5 else "BAD EXAMPLE"
    else:
        # Expected a bad description - low similarity is good
        label = "GOOD EXAMPLE" if similarity_score < 0.50 else "BAD EXAMPLE"

    return label


def accuracy_summary_evaluator(inputs, outputs, expected_outputs, evaluations):
    """Calculate accuracy based on similarity scores and expected values.

    Args:
        inputs: List of input data
        outputs: List of task outputs
        expected_outputs: List of expected outputs
        evaluations: List of evaluation results

    Returns:
        dict: Contains "accuracy", "match_count", "not_match_count", and "total" keys
    """
    match_count = 0
    not_match_count = 0

    for i, prediction in enumerate(evaluations['llm_judge_evaluator_function']):
        similarity_score = prediction['similarity_score']
        expected_value = expected_outputs[i]['value']
        is_good_expected = expected_value == 'yes'

        if is_good_expected:
            if similarity_score >= 0.5:
                match_count += 1
            else:
                not_match_count += 1
        else:
            # Lower similarity to bad description is better
            if similarity_score < 0.50:
                match_count += 1
            else:
                not_match_count += 1

    total = len(outputs)
    accuracy = match_count / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "match_count": match_count,
        "not_match_count": not_match_count,
        "total": total,
    }


def compute_score(summary_evaluators) -> float:
    """Compute the optimization score from accuracy."""
    accuracy = summary_evaluators['accuracy_summary_evaluator']['value']['accuracy']
    return accuracy


def stopping_condition(summary_evaluators) -> bool:
    """Determine if optimization should stop (accuracy > 0.9)."""
    accuracy = summary_evaluators['accuracy_summary_evaluator']['value']['accuracy']
    return accuracy > 0.9


def main():
    """Execute the text generation prompt optimization workflow with LLM-as-a-Judge.

    This function orchestrates the entire optimization process:

    1. **Initialize**: Enable Datadog LLM Observability with project settings
    2. **Load Data**: Pull your annotated text generation dataset from Datadog
    3. **Configure**: Set up the optimization with:
       - Generation task function (how to produce descriptions)
       - Optimization task (how to improve the prompt)
       - LLM Judge evaluator (how to measure similarity to expected outputs)
       - Scoring (how to rank iterations by accuracy)
       - Stopping condition (when to stop early)
    4. **Optimize**: Run iterative optimization (up to MAX_ITERATION times):
       - Test current prompt on entire dataset (parallel execution)
       - Use LLM judge to evaluate similarity scores for each generated output
       - Compute accuracy based on similarity thresholds
       - Analyze low-scoring and high-scoring examples
       - Generate improved prompt based on judge feedback
       - Test improved prompt
       - Repeat until stopping condition met or max iterations reached
    5. **Output**: Display the best performing prompt and all experiment URLs

    The optimization runs in parallel (jobs=20) for fast execution. Each iteration
    is tracked in Datadog LLM Observability where you can:
    - View similarity score distributions
    - Analyze examples that scored high vs low
    - Compare prompts across iterations
    - Review judge reasoning for evaluation decisions
    - Export the best prompt for production deployment
    """
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
        task=description_generation_task_function,
        optimization_task=optimization_task_function,
        evaluators=[llm_judge_evaluator_function],
        compute_score=compute_score,
        summary_evaluators=[accuracy_summary_evaluator],
        labelization_function=labelization_function,
        stopping_condition=stopping_condition,
        max_iterations=MAX_ITERATION,
        config={
            # Mandatory
            "prompt": INITIAL_PROMPT,
            # Optionals
            "model_name": EVALUATION_MODEL_NAME,
            "evaluation_output_format": DescriptionGenerationResult.output_format(),
            "runs": RUNS,
        }
    )

    result = prompt_optimization.run(jobs=JOBS)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nBest prompt:\n{result.best_prompt}")
    print(f"\nBest accuracy: {result.best_score:.4f}")

    # Print all experiment URLs for reference
    print(f"\n{'=' * 80}")
    print("ALL EXPERIMENT URLS")
    print(f"{'=' * 80}")
    print(result.summary())


if __name__ == "__main__":
    main()
