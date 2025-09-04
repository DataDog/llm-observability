import ddtrace.auto

import os
from typing import Optional, Union, List, Literal, Dict, Any

import ddtrace
from ddtrace.internal.telemetry import telemetry_writer
from ddtrace.llmobs import LLMObs
from ddtrace import tracer
import asyncio

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi import Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import openai  # noqa: E402

from google import genai  # noqa: E402

import anthropic  # noqa: E402

app = FastAPI(
    title="APM library test server",
    description="""
The reference implementation of the APM Library test server.

Implement the API specified below to enable your library to run all of the shared tests.
""",
)

proxy_server_url = os.getenv("PROXY_LLM_SERVER_URL")


def get_proxy_url(provider):
    return f"{proxy_server_url}/{provider}"


spans = {}
oaiClient = openai.OpenAI(base_url=get_proxy_url("openai"))
PROJECT_ID = "datadog-sandbox"
genaiClient = genai.Client(http_options={"base_url": get_proxy_url("genai")})
anthropicClient = anthropic.Anthropic(base_url=get_proxy_url("anthropic"))


class TestAnnotation(BaseModel):
    input_data: Optional[Union[dict, str, List[Union[dict, str]]]] = None
    output_data: Optional[Union[dict, str, List[Union[dict, str]]]] = None
    metadata: Optional[dict] = None
    metrics: Optional[dict] = None
    tags: Optional[dict] = None

    explicit_span: Optional[bool] = False


class TestTraceStructure(BaseModel):
    sdk: Union[Literal["llmobs", "tracer"]]

    name: Optional[str] = None
    session_id: Optional[str] = None
    ml_app: Optional[str] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    kind: Optional[str] = None

    annotations: Optional[List[TestAnnotation]] = None
    annotate_after: Optional[bool] = False

    children: Optional[
        List[Union["TestTraceStructure", List["TestTraceStructure"]]]
    ] = None

    export_span: Optional[Literal["explicit", "implicit"]] = None


class TraceStructureRequest(BaseModel):
    trace_structure: TestTraceStructure


@app.get("/sdk/info")
def sdk_version():
    return {"version": ddtrace.__version__}


@app.post("/sdk/trace")
async def sdk_trace(req: TraceStructureRequest):
    maybe_exported_span_ctx = await create_trace(req.trace_structure)
    telemetry_writer.periodic(force_flush=True)
    return maybe_exported_span_ctx or {}


async def create_trace(trace_structure: TestTraceStructure):
    is_llmobs = trace_structure.sdk == "llmobs"
    kind = trace_structure.kind if is_llmobs else None
    make_trace = getattr(LLMObs, kind) if is_llmobs else tracer.trace

    if is_llmobs:
        options = {
            "name": trace_structure.name,
        }
        if trace_structure.session_id:
            options["session_id"] = trace_structure.session_id
        if trace_structure.ml_app:
            options["ml_app"] = trace_structure.ml_app
        if trace_structure.model_name:
            options["model_name"] = trace_structure.model_name
        if trace_structure.model_provider:
            options["model_provider"] = trace_structure.model_provider
    else:
        options = {
            "name": trace_structure.name,
        }

    exported_span_ctx = None

    annotations = trace_structure.annotations
    annotate_after = trace_structure.annotate_after
    span = None

    with make_trace(**options) as _span:
        span = _span

        # apply annotations
        if annotations and not annotate_after:
            apply_annotations(span, annotations)

        # apply export span
        export_span = trace_structure.export_span
        if export_span:
            args = (span,) if export_span == "explicit" else ()
            exported_span_ctx = LLMObs.export_span(*args)

        # trace children
        children = trace_structure.children or []

        for child in children:
            if not isinstance(child, list):
                await create_trace(child)
            else:
                await asyncio.gather(*[create_trace(c) for c in child])

    if annotate_after:
        # this case should always throw
        apply_annotations(span, annotations, annotate_after=True)

    return exported_span_ctx


def apply_annotations(span, annotations: TestAnnotation, annotate_after=False):
    for annotation in annotations:
        options = {
            k: v
            for k, v in annotation.model_dump().items()
            if k not in ("explicit_span",)
        }
        if annotation.explicit_span or annotate_after:
            options["span"] = span
        LLMObs.annotate(**options)


@app.post("/sdk/export_span_with_fake_span")
async def export_span_with_fake_span():
    fake_span = 5
    exported_span_ctx = LLMObs.export_span(fake_span)
    return {"exported_span_ctx": exported_span_ctx}


class EvaluationMetricRequest(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    span_with_tag_value: Optional[dict] = None

    label: Optional[str] = None
    metric_type: Optional[str] = None
    value: Optional[Union[str, int, float]] = None

    tags: Optional[dict] = None
    ml_app: Optional[str] = None
    timestamp_ms: Optional[Any] = None
    metadata: Optional[Dict[str, object]] = None


@app.post("/sdk/submit_evaluation_metric")
async def submit_evaluation_metric(req: EvaluationMetricRequest):
    try:
        joining_options = {}

        if req.trace_id or req.span_id:
            joining_options["span"] = {
                "trace_id": req.trace_id,
                "span_id": req.span_id,
            }

        if req.span_with_tag_value:
            joining_options["span_with_tag_value"] = req.span_with_tag_value

        LLMObs.submit_evaluation_for(
            label=req.label,
            metric_type=req.metric_type,
            value=req.value,
            tags=req.tags,
            ml_app=req.ml_app,
            timestamp_ms=req.timestamp_ms,
            metadata=req.metadata,
            **joining_options,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        telemetry_writer.periodic(force_flush=True)


@app.post("/sdk/inject_distributed_headers")
def sdk_inject_distributed_headers():
    span = LLMObs.task("task_span")
    LLMObs.inject_distributed_headers({}, span)
    telemetry_writer.periodic(force_flush=True)
    return {}


@app.post("/sdk/activate_distributed_headers")
def sdk_activate_distributed_headers(req: Request):
    LLMObs.activate_distributed_headers(req.headers)
    telemetry_writer.periodic(force_flush=True)
    return {}


@app.post("/sdk/flush")
def sdk_flush():
    LLMObs.flush()
    telemetry_writer.periodic(force_flush=True)
    return {}


class FinishSpanRequest(BaseModel):
    span_id: int


@app.post("/sdk/finish_span")
def finish_span(req: FinishSpanRequest):
    span = spans[req.span_id]
    span.finish()
    telemetry_writer.periodic(force_flush=True)
    return {}


class CompletionRequest(BaseModel):
    prompt: Union[str, List[str]]
    model: Optional[str] = None
    parameters: Optional[dict] = None
    system_instruction: Optional[str] = None
    tools: Optional[list] = None
    tool_config: Optional[dict] = None


class EmbeddingRequest(BaseModel):
    input: str
    model: str


class GenAiGenerateContentRequest(BaseModel):
    model: str
    contents: Any
    config: Optional[dict] = None


class GenAiEmbedContentRequest(BaseModel):
    model: str
    contents: Any


class ResponsesCreateRequest(BaseModel):
    model: str
    input: Union[str, List[dict]]
    parameters: dict
    tools: Optional[list] = None


class AnthropicCreateRequest(BaseModel):
    model: str
    messages: List[Any]
    system: Optional[Union[str, List[Any]]] = None
    parameters: Optional[dict] = None
    tools: Optional[List[dict]] = None
    stream: Optional[bool] = False
    stream_as_method: Optional[bool] = False


@app.post("/openai/chat_completion")
def openai_chat_completion(req: CompletionRequest):
    tools = {"tools": req.tools, "tool_choice": "auto"} if req.tools else {}
    parameters = req.parameters or {}
    stream = parameters.get("stream", False)

    if not stream:
        del parameters["stream"]

    response = oaiClient.chat.completions.create(
        model=req.model,
        messages=[{"role": "user", "content": req.prompt}],
        **tools,
        **parameters,
    )

    if stream:
        # consume the stream
        for _ in response:
            pass
    telemetry_writer.periodic(force_flush=True)
    return {"response": response}


@app.post("/openai/completion")
def openai_completion(req: CompletionRequest):
    response = oaiClient.completions.create(
        model=req.model,
        prompt=req.prompt,
        **req.parameters,
    )
    return {"response": response}


@app.post("/openai/embedding")
def openai_embedding(req: EmbeddingRequest):
    response = oaiClient.embeddings.create(
        model=req.model,
        input=req.input,
    )
    return {"response": response}


@app.post("/openai/responses/create")
def openai_responses_create(req: ResponsesCreateRequest):
    parameters = req.parameters or {}
    tools = dict(tools=req.tools, tool_choice="auto") if req.tools else {}
    stream = parameters.get("stream", False)

    if not stream:
        del parameters["stream"]

    response = oaiClient.responses.create(
        model=req.model,
        input=req.input,
        **tools,
        **parameters,
    )

    if stream:
        # consume the stream
        for _ in response:
            pass

    return {"response": response}


@app.post("/genai/generate_content")
def genai_generate_content(req: GenAiGenerateContentRequest):
    config = req.config or {}
    stream = config.pop("stream", False)

    kwargs = {"model": req.model, "contents": req.contents, "config": config}

    if stream:
        for _ in genaiClient.models.generate_content_stream(**kwargs):
            pass
    else:
        genaiClient.models.generate_content(**kwargs)

    return {}


@app.post("/genai/embed_content")
def genai_embed_content(req: GenAiEmbedContentRequest):
    genaiClient.models.embed_content(model=req.model, contents=req.contents)
    return {}


@app.post("/anthropic/create")
def anthropic_create(req: AnthropicCreateRequest):
    stream_as_method = req.stream_as_method
    stream = req.stream
    tools = {"tools": req.tools} if req.tools else {}

    kwargs = dict(model=req.model, messages=req.messages, **req.parameters, **tools)
    if req.system:
        kwargs["system"] = req.system

    if stream:
        if stream_as_method:
            with anthropicClient.messages.stream(**kwargs) as stream:
                for _ in stream.text_stream:
                    pass
        else:
            resp = anthropicClient.messages.create(**kwargs, stream=True)

            for _ in resp:
                pass  # consume the stream
    else:
        anthropicClient.messages.create(**kwargs)

    return {}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))
