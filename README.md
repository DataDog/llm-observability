# LLM Observability Jupyter Notebooks

These notebooks introduce you to Datadog's [LLM Observability Python SDK](https://docs.datadoghq.com/tracing/llm_observability/sdk/) using hands-on examples.

For a detailed instrumentation guide, see [Trace an LLM Application](https://docs.datadoghq.com/tracing/llm_observability/trace_an_llm_application/).

## Prerequisites

- [A Datadog API key](https://docs.datadoghq.com/account_management/api-app-keys)
- [An OpenAI API key](https://platform.openai.com/docs/quickstart/account-setup)

## Setup

#### 1. Activate your virtualenv:

```bash
python -m venv myenv
source myenv/bin/activate
```

#### 2. Create a .env file and add the following:

```bash
DD_API_KEY=<YOUR_DATADOG_API_KEY>
DD_SITE=<YOUR_DATADOG_SITE>
DD_LLMOBS_AGENTLESS_ENABLED=1
DD_LLMOBS_ML_APP="onboarding-quickstart"
```

- Note: if [your Datadog site](https://docs.datadoghq.com/getting_started/site/#access-the-datadog-site) (`DD_SITE`) is not provided, the value defaults to `"datadoghq.com"`
- Feel free to update the `DD_LLMOBS_ML_APP` variable to any custom app name.
- `DD_LLMOBS_AGENTLESS_ENABLED=1` is only required if the Datadog Agent is not running. If the agent is running in your production environment, make sure this environment variable is unset.


#### 3. If you don't already have a system-wide OPENAI_API_KEY variable, add one to the .env file:

```bash
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

#### 3. Install shared dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```

#### 4. Launch Jupyter notebooks

You can either start Jupyter on the command line (`jupyter notebook`) to use the web interface, or open your notebook from your preferred code editor (for example, VS Code) and run it there.

## Notebooks

### 1. Tracing a simple LLM call

**[This notebook](./1-llm-span.ipynb)** shows you how to create and trace a simple LLM call.

<img src="./images/llm-span.png" height="350" >

### 2. Tracing an LLM Workflow

**[This notebook](./2-workflow-span.ipynb)** shows you how to create and trace a more complex, static series of steps that involves a tool call in addition to a call to an LLM.

<img src="./images/workflow-span.png" height="350" >

### 3. Tracing an LLM Agent

**[This notebook](./3-agent-span.ipynb)** shows you how to create and trace an LLM powered agent that calls tools and makes decisions based on data about what to do next.

<img src="./images/agent-span.png" height="350" >

### 4. Tracing and evaluating a RAG workflow

**[This notebook](./4-custom-evaluations.ipynb)** shows you how to create, trace, and evaluate a RAG workflow.

<img src="./images/rag-span.png" height="350" >

## Teardown

When you're done with the tutorials, deactivate your virtualenv and return to your system's default Python env:

```bash
deactivate
```
