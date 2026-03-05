from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from ddtrace.llmobs import LLMObs

import os

LLMObs.enable(
    project_name="deep-eval-demo-project",
    api_key=os.environ["DD_API_KEY"],
    app_key=os.environ["DD_APP_KEY"],
    site=os.environ["DD_SITE"],
    agentless_enabled=True,
)


correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    async_mode=True,
)

dataset = LLMObs.create_dataset(
    dataset_name="capitals-of-the-world",
    description="Questions about world capitals",
    records=[
        {
            "input_data": {"question": "What is the capital of China?", "output": "Beijing"},       # required, JSON or string
            "expected_output": "Beijing",                                      # optional, JSON or string
            "metadata": {"difficulty": "easy"}                                 # optional, JSON
        },
        {
            "input_data": {"question": "What is the capital of China?", "output": "China"},       # required, JSON or string
            "expected_output": "Beijing",                                      # optional, JSON or string
            "metadata": {"difficulty": "easy"}                                 # optional, JSON
        },
        {
            "input_data": {"question": "Which city serves as the capital of South Africa?", "output": "Pretoria"},
            "expected_output": "Pretoria",
            "metadata": {"difficulty": "medium"}
        }
    ]
)

def my_task(input_data, config):
    return input_data["output"]

def num_exact_matches(inputs, outputs, expected_outputs, evaluators_results):
    return evaluators_results["Correctness"].count(True)

experiment = LLMObs.experiment(
    name="deepeval-demo",
    task=my_task, 
    dataset=dataset,
    evaluators=[correctness_metric],
    summary_evaluators=[num_exact_matches], # optional, used to summarize the experiment results
    description="Determine whether the actual output is factually correct based on the expected output.",
)

result = experiment.run()
print(experiment.url)