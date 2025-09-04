import json
import time
import requests

from typing import Optional
from typing import Union
from typing import Any
from typing import Dict
from typing import List

from test.utils import (
    TestSpan as TestTraceStructure,
)  # importing as TestTraceStructure for clarity, the Span should represent a root span (trace)


class InstrumentationClient:
    """Client to query the shared interface to the instrumentation libraries."""

    def __init__(self, url: str, test_lang: str):
        self._base_url = url
        self._session = requests.Session()
        self._test_lang = test_lang

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def __repr__(self):
        return f"InstrumentationClient(test_lang={self._test_lang!r})"

    def wait_to_start(self):
        """Wait for the server to start."""
        for i in range(6000):
            try:
                resp = self._session.get(self._url("/sdk/info"))
                if 200 <= resp.status_code < 300:
                    return resp.json()
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.01)
        else:
            assert False, "Test server did not start"

    def sdk_trace(
        self, trace_structure: TestTraceStructure, raise_on_error: bool = True
    ):
        resp = self._session.post(
            self._url("/sdk/trace"),
            json={"trace_structure": trace_structure.model_dump(exclude_none=True)},
        )
        if raise_on_error:
            resp.raise_for_status()
        return resp.json()

    def export_span(self, err=None):
        resp = self._session.post(self._url("/sdk/export_span"), json={"err": err})
        resp.raise_for_status()
        return resp.json()

    def inject_distributed_headers(self):
        resp = self._session.post(self._url("/sdk/inject_distributed_headers"))
        resp.raise_for_status()
        return resp.json()

    def activate_distributed_headers(self):
        resp = self._session.post(
            self._url("/sdk/activate_distributed_headers"),
            headers={
                "x-datadog-trace-id": "16823502017678563351",
                "x-datadog-parent-id": "5716834657549619909",
                "x-datadog-sampling-priority": "1",
                "x-datadog-tags": "_dd.p.llmobs_trace_id=684330eb00000000a6be47d8109fe8d0,_dd.p.llmobs_parent_id=5716834657549619909,_dd.p.dm=-0,_dd.p.tid=6813c72400000000",
                "traceparent": "00-6813c72400000000e97915f349cc5817-4f5646d4fae312c5-01",
                "tracestate": "dd=p:4f5646d4fae312c5;s:1;t.dm:-0;t.llmobs_parent_id:5716834657549619909;t.tid:6813c72400000000",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def flush(self):
        resp = self._session.post(self._url("/sdk/flush"))
        resp.raise_for_status()
        return resp.json()

    def export_span_with_fake_span(self):
        resp = self._session.post(
            self._url("/sdk/export_span_with_fake_span"),
        )
        resp.raise_for_status()
        return resp.json()

    def submit_evaluation_metric(
        self,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        span_with_tag_value: Optional[dict] = None,
        label: Optional[str] = None,
        metric_type: Optional[str] = None,
        value: Optional[Union[str, int, float]] = None,
        tags: Optional[dict] = None,
        ml_app: Optional[str] = None,
        timestamp_ms: Optional[Any] = None,
        metadata: Optional[Dict[str, object]] = None,
        raise_on_error: bool = True,
    ):
        resp = self._session.post(
            self._url("/sdk/submit_evaluation_metric"),
            json={
                "trace_id": trace_id,
                "span_id": span_id,
                "span_with_tag_value": span_with_tag_value,
                "label": label,
                "metric_type": metric_type,
                "value": value,
                "tags": tags,
                "ml_app": ml_app,
                "timestamp_ms": timestamp_ms,
                "metadata": metadata,
            },
        )
        if raise_on_error:
            resp.raise_for_status()
        return resp.json()

    def openai_chat_completion(
        self,
        model: str,
        prompt: str,
        parameters: Optional[dict] = None,
        tools: Optional[dict] = None,
    ):
        resp = self._session.post(
            self._url("/openai/chat_completion"),
            json={
                "prompt": prompt,
                "model": model,
                "parameters": parameters,
                "tools": tools,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def openai_completion(self, model: str, prompt: str, parameters: dict):
        resp = self._session.post(
            self._url("/openai/completion"),
            json={
                "prompt": prompt,
                "model": model,
                "parameters": parameters,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def openai_embedding(self, model: str, input: str):
        resp = self._session.post(
            self._url("/openai/embedding"),
            json={
                "input": input,
                "model": model,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def openai_responses_create(
        self,
        model: str,
        input: str | list[dict],
        parameters: dict,
        tools: Optional[dict] = None,
    ):
        resp = self._session.post(
            self._url("/openai/responses/create"),
            json={
                "model": model,
                "input": input,
                "parameters": parameters,
                "tools": tools,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def vertexai_completion(
        self,
        prompt: any,
        parameters: dict = None,
        system_instruction: str = None,
        asynchronous: bool = None,
        tools: list = None,
        tool_config: dict = None,
    ):
        url = (
            "/vertexai/completion" if not asynchronous else "/vertexai/completion_async"
        )
        resp = self._session.post(
            self._url(url),
            json={
                "prompt": prompt,
                "parameters": parameters,
                "system_instruction": system_instruction,
                "tools": tools,
                "tool_config": tool_config,
            },
        )
        resp.raise_for_status()

    def vertexai_chat_completion(
        self,
        prompt: any,
        parameters: dict = None,
        system_instruction: str = None,
        asynchronous: str = None,
        tools: list = None,
        tool_config: dict = None,
    ):
        url = (
            "/vertexai/chat_completion"
            if not asynchronous
            else "/vertexai/chat_completion_async"
        )
        resp = self._session.post(
            self._url(url),
            json={
                "prompt": prompt,
                "parameters": parameters,
                "system_instruction": system_instruction,
                "tools": tools,
                "tool_config": tool_config,
            },
        )
        resp.raise_for_status()

    def genai_generate_content(
        self,
        model: str,
        contents: Any,  # string, content, part, or list of contents or parts
        config: dict = None,
    ):
        resp = self._session.post(
            self._url("/genai/generate_content"),
            json=dict(
                model=model,
                contents=contents,
                config=config,
            ),
        )

        resp.raise_for_status()
        return resp.json()

    def genai_embed_content(
        self,
        model: str,
        contents: Any,  # string, content, part, or list of contents or parts
    ):
        resp = self._session.post(
            self._url("/genai/embed_content"),
            json=dict(
                model=model,
                contents=contents,
            ),
        )

        resp.raise_for_status()
        return resp.json()

    def anthropic_create(
        self,
        model: str,
        messages: List[Any],
        system: Optional[Union[str, List[Any]]] = None,
        parameters: Optional[dict] = None,
        tools: Optional[List[dict]] = None,
        stream_as_method: Optional[bool] = False,
        stream: Optional[bool] = False,
        raise_on_error: bool = True,
    ):
        resp = self._session.post(
            self._url("/anthropic/create"),
            json={
                "model": model,
                "messages": messages,
                "system": system,
                "parameters": parameters,
                "tools": tools,
                "stream_as_method": stream_as_method,
                "stream": stream,
            },
        )
        if raise_on_error:
            resp.raise_for_status()

        try:
            return resp.json()
        except json.JSONDecodeError:
            return {}
