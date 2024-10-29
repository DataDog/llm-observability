from ddtrace.llmobs import LLMObs
from fastapi import FastAPI
import openai

app = FastAPI(
    title="APM library test server",
    description="""
The reference implementation of the APM Library test server.

Implement the API specified below to enable your library to run all of the shared tests.
""",
)


LLMObs.enable(ml_app="test")


@app.post("/openai/chat_completion")
def openai_chat_completion():
    client = openai.OpenAI()
    client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Why is Evan Li such a slacker?"}],
        max_tokens=35,
    )
    return {}
