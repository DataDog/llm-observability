# [Early Preview] LLM Observability Experiment Notebooks

These notebooks introduce you to Datadog's LLM Observability Experiment SDK using hands-on examples.

## Prerequisites

- [Datadog API Key](https://docs.datadoghq.com/account_management/api-app-keys)
- [Datadog App Key](https://app.datadoghq.com/organization-settings/application-keys)
- [An OpenAI API key](https://platform.openai.com/docs/quickstart/account-setup)
- [An OpenRouter API Key](https://openrouter.ai/settings/keys)
- [Rust](https://rustup.rs/): Required for building the Python wheel during the early preview phase. This dependency will be removed in a future update.

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

#### 3. Install the Python SDK

```bash
pip install git+https://github.com/DataDog/dd-trace-py.git@llm-experiments
```

#### 4. Create a .env file and add the following:

```bash
DD_API_KEY=<>
DD_APPLICATION_KEY=<>
OPENAI_API_KEY=<>
OPENROUTER_API_KEY=<>
DD_SITE=<> # Optional: (default: "datadoghq.com" | examples: "us3.datadoghq.com", "eu.datadoghq.com")
```

#### 5. Launch Jupyter notebooks

You can either start Jupyter on the command line (jupyter notebook) to use the web interface, or open your notebook from your preferred code editor (for example, VS Code) and run it there.
