# Evaluating with TypeSafe's Jev

[Jev](https://docs.typesafe.ai/concepts/system-one) is a decision model. You send it some
evidence and a set of typed questions, and it returns typed answers with probabilities. It
does not generate prose, which makes it a cheap fit for evaluation work where the output
you actually need is a verdict rather than an explanation.

These notebooks wire Jev into both evaluation surfaces in Datadog Agent Observability:
online evals on live spans, and offline evals inside an experiment. One rubric definition
drives both.

The example application is a support agent for a fictional airline. It answers customer
tickets from retrieved policy excerpts, and the policy corpus has deliberate holes, so some
tickets have no grounded answer available. That gives the evaluator something real to catch.

## Prerequisites

- [A Datadog API key and application key](https://docs.datadoghq.com/account_management/api-app-keys)
- [An OpenAI API key](https://platform.openai.com/docs/quickstart/account-setup) for the agent under evaluation
- A [TypeSafe API key](https://typesafe.ai) for the judge

## Setup

#### 1. Activate your virtualenv

```bash
python -m venv venv
source venv/bin/activate
```

#### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

#### 3. Create a `.env` file in this directory

```bash
DD_API_KEY=<your Datadog API key>
DD_APPLICATION_KEY=<your Datadog application key>
DD_SITE=datadoghq.com
OPENAI_API_KEY=<your OpenAI key>
TYPESAFE_API_KEY=<your TypeSafe key>
```

`DD_SITE` defaults to `datadoghq.com`. Set it to match
[your Datadog site](https://docs.datadoghq.com/getting_started/site/) if it differs.

Optional: set `DD_LLMOBS_ML_APP` and `DD_PROJECT` to control where the spans and the
experiment land. Both default to `airline-support-agent`.

#### 4. Launch Jupyter

```bash
jupyter notebook
```

## Notebooks

### 1. Judging a reply with Jev

**[This notebook](./1-jev-rubric.ipynb)** builds an evaluation rubric one question at a
time, sends all five questions in a single request, and reads the answers. It covers the
three question types, why the probability distribution is worth more than the winning
label, and why the composite verdict belongs in your code rather than in the rubric.

Only needs `TYPESAFE_API_KEY`.

### 2. Online evals: scoring live spans

**[This notebook](./2-online-evals.ipynb)** runs the agent, emits spans tagged with a
`turn_id`, then scores those spans from a separate step and attaches the verdicts with
`LLMObs.submit_evaluation`. It also covers how the choice of metric type decides how much
of Jev's output survives into the query layer.

### 3. Offline evals: Jev inside a Datadog experiment

**[This notebook](./3-experiments.ipynb)** runs the same rubric over a dataset as a Datadog
experiment. It shows how to judge each row once instead of once per evaluator, how to check
the judge against your own labels rather than only checking the agent, and how a rubric fix
shows up as a measurable delta between two runs.

## Files

| File | What it holds |
| --- | --- |
| [`vega_air.py`](./vega_air.py) | The application under evaluation: the policy corpus, the tickets, and `answer_ticket` |
| [`jev_rubric.py`](./jev_rubric.py) | The five questions, the thresholds, and the helpers that turn a Jev response into Datadog eval metrics |

Notebook 1 builds the rubric inline so you can read it question by question. Notebooks 2
and 3 import `jev_rubric.py`, so there is one definition rather than three copies.

## Costs

Notebook 1 makes a single Jev request. Notebooks 2 and 3 each run the agent over ten
tickets and judge every reply, so each one costs ten OpenAI completions plus ten Jev
requests. Notebook 3 runs the experiment twice if you follow it to the end.

See TypeSafe's [models page](https://docs.typesafe.ai/models) for current Jev pricing and
request limits.

## Teardown

```bash
deactivate
```
