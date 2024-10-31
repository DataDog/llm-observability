import os

import ddtrace
from ddtrace.llmobs import LLMObs
from fastapi import FastAPI
from pydantic import BaseModel

import openai

app = FastAPI(
    title="APM library test server",
    description="""
The reference implementation of the APM Library test server.

Implement the API specified below to enable your library to run all of the shared tests.
""",
)


spans = {}
LLMObs.enable(ml_app="test")
oaiClient = openai.OpenAI()


class TaskRequest(BaseModel):
    name: str
    session_id: str
    ml_app: str


@app.get("/sdk/info")
def sdk_version():
    return {"version": ddtrace.__version__}


@app.post("/sdk/task")
def sdk_task(req: TaskRequest):
    span = LLMObs.task(req.name, req.session_id, req.ml_app)
    spans[span.span_id] = span
    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "meta": {k: v for k, v in span.get_tags().items() if isinstance(v, str)},
        "metrics": {
            k: v for k, v in span.get_tags().items() if isinstance(v, (int, float))
        },
    }


class FinishSpanRequest(BaseModel):
    span_id: int


@app.post("/sdk/finish_span")
def finish_span(req: FinishSpanRequest):
    span = spans[req.span_id]
    span.finish()
    return {}


class ChatCompletionRequest(BaseModel):
    prompt: str


@app.post("/openai/chat_completion")
def openai_chat_completion(req: ChatCompletionRequest):
    oaiClient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=35,
    )
    return {}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=os.environ["PORT"])
