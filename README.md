# Datadog LLM Observability Cookbooks

Hands-on examples and tutorials for instrumenting, monitoring, and evaluating LLM applications with [Datadog LLM Observability](https://docs.datadoghq.com/llm_observability/).

## What You'll Learn

This repository provides a complete learning path from basic instrumentation to advanced evaluation workflows:

1. **[Tracing](#1-tracing)** - Instrument and trace LLM applications to capture execution flows
2. **[Online Evaluation](#2-online-evaluation)** - Monitor production traces with managed evaluators and custom metrics
3. **[Offline Experiments](#3-experiments)** - Run systematic offline evaluations with datasets, experiments, and CI/CD integration

## Quick Start

### Prerequisites

- **Datadog Account**: [Sign up for free](https://www.datadoghq.com/)
- **API Keys**:
  - [Datadog API Key](https://docs.datadoghq.com/account_management/api-app-keys)
  - [Datadog Application Key](https://app.datadoghq.com/organization-settings/application-keys)
- **LLM Provider**: [OpenAI API Key](https://platform.openai.com/docs/quickstart/account-setup) (used in most examples)
- **Python**: 3.8+ with `ddtrace>=4.3.0`

### Installation

```bash
# Clone the repository
git clone https://github.com/DataDog/llm-observability-cookbooks.git
cd llm-observability-cookbooks

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Navigate to a specific section and install dependencies
cd 1-tracing
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in each section directory:

```bash
DD_API_KEY=<YOUR_DATADOG_API_KEY>
DD_SITE=<YOUR_DATADOG_SITE>  # e.g., datadoghq.com, us3.datadoghq.com
DD_LLMOBS_AGENTLESS_ENABLED=1  # If not using Datadog Agent
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

## Additional Resources

- **Documentation**: [docs.datadoghq.com/llm_observability](https://docs.datadoghq.com/llm_observability/)
- **API Reference**: [docs.datadoghq.com/api/latest/llm-observability](https://docs.datadoghq.com/api/latest/llm-observability/)
- **Python SDK**: [ddtrace Python library](https://ddtrace.readthedocs.io/)
- **Community**: [Datadog Slack](https://chat.datadoghq.com/)

