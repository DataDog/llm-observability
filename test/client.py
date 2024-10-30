import time
import requests

from typing import TypedDict
from typing import Optional


class Span(TypedDict):
    trace_id: int
    span_id: int
    parent_id: int
    name: str
    start_ns: int
    duration: int
    status: str
    meta: dict
    metrics: dict

    def get_tag(self, key: str):
        return self.meta.get(key)


class InstrumentationClient:
    """Client to query the shared interface to the instrumentation libraries."""

    def __init__(self, url: str):
        self._base_url = url
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def wait_to_start(self):
        """Wait for the server to start."""
        for i in range(100):
            try:
                resp = self._session.get(self._url("/test"))
                if resp.status_code == 404:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)

    def sdk_task(
        self,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        ml_app: Optional[str] = None,
    ) -> Span:
        resp = self._session.post(
            self._url("/sdk/task"),
            json={"name": name, "session_id": session_id, "ml_app": ml_app},
        )
        resp.raise_for_status()
        return resp.json()

    def finish_span(self, span_id: int):
        resp = self._session.post(
            self._url("/sdk/finish_span"), json={"span_id": span_id}
        )
        resp.raise_for_status()

    def openai_chat_completion(self, prompt: str):
        resp = self._session.post(
            self._url("/openai/chat_completion"),
            json={
                "prompt": prompt,
            },
        )
        resp.raise_for_status()
