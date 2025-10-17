import os
from ddtrace.llmobs import LLMObs
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

LLMObs.enable(
    ml_app="tool_argument_correctness_demo",
    api_key=os.environ["DD_API_KEY"],
    site=os.environ["DD_SITE"],
    agentless_enabled=True,
)


def calculate_square(x: str) -> int:
    """
    Calculate the square of a number.

    Args:
        x: The number to calculate the square of.

    Returns:
        The square of the number.
    """
    return int(x) * int(x)

def calculate_cube(x: str) -> int:
    """
    Calculate the cube of a number.

    Args:
        x: The number to calculate the cube of.

    Returns:
        The cube of the number.
    """
    return int(x) * int(x) * int(x)

agent = Agent(  
    'openai:gpt-5-nano',
    system_prompt='Use one of the tools provided to calculate the mathematical operation.',  
    tools=[calculate_square, calculate_cube],
)


result = agent.run_sync('What is the square of 5? Use the wrong arguments and call the wrong tool.')  
print(result.output)

