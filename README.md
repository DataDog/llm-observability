# llm-observability interactive onboarding

## Setup

#### 1. Activate your virtualenv:

```
python -m venv env
source env/bin/activate
```

#### 2. Install dependencies in the requirements.txt file:

`pip install -r requirements.txt`

#### 3. Create a .env file and add the following:

```
DD_LLMOBS_ENABLED=true
DD_LLMOBS_NO_APM=1
DD_LLMOBS_APP_NAME="test-onboarding-app"
DD_API_KEY=[your API key goes here]
```

#### 4. Start your Jupyter notebook
You can either start your notebook on the command line (`jupyter notebook`) or open your notebook from your prefered code editor and run it there (VSCode has a great notebook interface).

## Notebooks

### 1. Tracing a simple LLM call

**[This notebook](./1-llm-span.ipynb)** shows you how to create and trace a simple LLM call.

<img src="./images/llm-span.png" height="350" > 

### 2. Tracing an LLM Workflow

**[This notebook](./2-workflow-span.ipynb)** shows you how to create and trace a more complex, static series of steps that involves a tool call as well as a call to an LLM.

<img src="./images/workflow-span.png" height="350" > 

### 3. Tracing an an LLM Agent

_coming soon_

## Teardown

When you're done with the tutorials, deactivate your virtualenv and return to your system's default Python env:

```
deactivate
```
