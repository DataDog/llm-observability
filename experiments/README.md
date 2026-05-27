# LLM Observability Experiments

This directory contains hands-on examples for using Datadog's LLM Observability Experimentation SDK. The experiments framework allows you to systematically evaluate and compare LLM applications using datasets, tasks, and evaluators.

## What are LLM Observability Experiments?

LLM Observability Experiments provide a structured way to:

- **Evaluate LLM performance** against datasets with ground truth
- **Compare different models, prompts, or configurations** systematically
- **Track performance metrics** over time and across iterations
- **Identify issues** like hallucinations or poor accuracy before production

## Key SDK classes

| Class | Description |
|-------|-------------|
| `EvaluatorResult` | Rich evaluation result with `value`, `reasoning`, `assessment`, `metadata`, and `tags`. |
| `MultiEvaluatorResult` | Return multiple named metrics from one evaluator call. Each sub-value can be a plain value or an `EvaluatorResult`. Labels default to `"<evaluator_name>-<key>"`; pass `prefix=False` for raw keys. |
| `BaseEvaluator` | Base class for class-based evaluators with custom configuration or state. |
| `BaseSummaryEvaluator` | Base class for evaluators that run once after all records and compute aggregate statistics. |

## Getting started

Explore the notebooks in the `notebooks/` directory or use the Postman collection (`experiments.postman_collection.json`) for submitting data directly to our APIs for cases where we don't support an SDK in your language.
