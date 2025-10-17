import os
from ddtrace.llmobs import LLMObs
from agents import Agent, ModelSettings, function_tool
from dotenv import load_dotenv

load_dotenv()

LLMObs.enable(
    ml_app="tool_selection_demo",
    api_key=os.environ["DD_API_KEY"],
    site=os.environ["DD_SITE"],
    agentless_enabled=True,
)

@function_tool
def add_numbers(a: int, b: int) -> int:
    """
    Adds two numbers together.
    """
    return a + b

@function_tool
def subtract_numbers(a: int, b: int) -> int:
    """
    Subtracts two numbers.
    """
    return a - b
    

# List of tools available to the agent 
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Please use the tools to find the answer.",
    model="o3-mini",
    tools=[
        add_numbers, subtract_numbers
    ],
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for history questions",
    instructions="You provide help with history problems.",
    model="o3-mini",
)

# The triage agent decides which specialized agent to hand off the task to — another type of tool selection covered by this evaluation.
triage_agent = Agent(  
    'openai:gpt-4o',
    model_settings=ModelSettings(temperature=0),
    instructions='What is the sum of 1 to 10?',  
    handoffs=[math_tutor_agent, history_tutor_agent],
)
