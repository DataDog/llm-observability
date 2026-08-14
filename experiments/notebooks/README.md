# LLM Observability Experiment Notebooks

These notebooks introduce you to Datadog's LLM Observability Experiment SDK using hands-on examples.

## Notebooks

- `00-basic-datasets.ipynb` — create, modify, push, and pull datasets.
- `01-basic-experiments.ipynb` — run a basic experiment with local evaluators.
- `02-extra-data.ipynb` — load CSV data and evaluate tasks with additional inputs.
- `03-remote-evaluators.ipynb` — use managed LLM-as-Judge evaluators.
- `04-otel-experiments.ipynb` — emit OpenTelemetry GenAI spans from a Python experiment task.

## Prerequisites

- [Datadog API Key](https://docs.datadoghq.com/account_management/api-app-keys)
- [Datadog App Key](https://app.datadoghq.com/organization-settings/application-keys)
- [An OpenAI API key](https://platform.openai.com/docs/quickstart/account-setup)

## Setup

#### 1. Activate your virtualenv:

```bash
virtualenv venv
source venv/bin/activate
```

#### 2. Install shared dependencies from the requirements.in file:

```bash
pip install -r requirements.in
```

#### 3. Create a .env file and add the following:

```bash
DD_API_KEY=<>
DD_APPLICATION_KEY=<>
OPENAI_API_KEY=<>
DD_SITE=<> # Optional: (default: "datadoghq.com" | examples: "us3.datadoghq.com", "eu.datadoghq.com")
DD_TRACE_OTEL_ENABLED=1 # Required for 04-otel-experiments.ipynb; set before importing ddtrace
```

#### 4. Launch Jupyter notebooks

You can either start Jupyter on the command line (jupyter notebook) to use the web interface, or open your notebook from your preferred code editor (for example, VS Code) and run it there.
