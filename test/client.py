import requests


class InstrumentationClient:
    def __init__(self, url: str):
        self._base_url = url
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def openai_chat_completion(self, prompt: str):
        resp = self._session.post(self._url("/openai/chat_completion"))
        resp.raise_for_status()
