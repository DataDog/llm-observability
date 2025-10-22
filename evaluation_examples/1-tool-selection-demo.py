import os
from ddtrace.llmobs import LLMObs
from dotenv import load_dotenv
from agents import Agent, ModelSettings, Runner, function_tool

load_dotenv()

LLMObs.enable(
    # feel free to change the ml_app name to any custom name
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
    
@function_tool  
def multiply_numbers(a: int, b: int) -> int:
    """
    Multiplies two numbers.
    """
    return a * b
@function_tool
def divide_numbers(a: int, b: int) -> int:
    """
    Divides two numbers.
    """
    return a / b


LLMObs.enable(
  ml_app="tool_selection_check",
  api_key=os.environ["DD_API_KEY"],
  site=os.environ["DD_SITE"],
  agentless_enabled=True,
)


math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Please use the tools to find the answer.",
    model="gpt-4o",
    tools=[
        add_numbers, subtract_numbers, multiply_numbers, divide_numbers
    ],
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for history questions",
    instructions="You provide help with history problems.",
    model="gpt-5-nano",
)

triage_agent = Agent(  
    'openai:gpt-4o',
    model_settings=ModelSettings(temperature=0),
    instructions='DO NOT RELY ON YOUR OWN MATHEMATICAL KNOWLEDGE, MAKE SURE TO CALL AVAILABLE TOOLS TO SOLVE EVERY SUBPROBLEM.',  
    handoffs=[math_tutor_agent, history_tutor_agent],
)


result = Runner.run_sync(triage_agent, '''
  Help me solve the following problem:
  What is the sum of the numbers between 1 and 100?
  Make sure you list out all the mathematical operations (addition, subtraction, multiplication, division) in order before you start calling tools in that order.
''')  