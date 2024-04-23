# llm-observability onboarding

## Setup

### 1. Activate your virtualenv:
```
python -m venv env
source env/bin/activate
```

### 2. Install dependencies in the requirements.txt file:
`pip install -r requirements.txt`

### 3. Create a .env file and add the following:
```
DD_LLMOBS_ENABLED=true 
DD_LLMOBS_APP_NAME="test-onboarding-app"
DD_API_KEY=[your API key goes here]
```

## Notebooks
1. [Tracing an LLM Model Call](./1-llm-span.ipynb)
2. [Tracing an LLM Workflow](./2-workflow-span.ipynb)
3. [coming soon] Tracing an LLM Agent 


## Teardown
When you're done with the tutorials, deactivate your virtualenv and return to your system's default Python env:
```
deactivate
```
