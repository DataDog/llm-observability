import pytest
from unittest import mock

from test import supported, unsupported
from test.utils import assert_llmobs_span_event, assert_apm_span

get_current_weather_func = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": [
                    "celsius",
                    "fahrenheit",
                ],
            },
        },
        "required": ["location"],
    },
}

weather_tool = {"function_declarations": [get_current_weather_func]}

TOOLS = [weather_tool]

TOOL_CONFIG = {
    "functionCallingConfig": {
        "mode": "ANY",
        "allowedFunctionNames": ["get_current_weather"],
    }
}

GENERATION_CONFIG = {
    "stop_sequences": ["x"],
    "max_output_tokens": 1000,
    "temperature": 1.0,
}


class TestVertexaiApm:
    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("asynchronous", [True, False])
    def test_completion(self, test_lang, test_client, test_agent, stream, asynchronous):
        if test_lang == "nodejs" and asynchronous:
            pytest.skip(
                "Node.js VertexAI SDK does not implement the `generateContentAsync` method. `generateContent` is already asynchronous."
            )
        test_client.vertexai_completion(
            prompt="How are you today?",
            parameters={"generation_config": GENERATION_CONFIG, "stream": stream},
            system_instruction="Please answer every question with a one sentence response.",
            asynchronous=asynchronous,
        )
        traces = test_agent.wait_for_num_traces(num=1)

        resource = None
        if test_lang == "python":
            resource = (
                "GenerativeModel.generate_content_async"
                if asynchronous
                else "GenerativeModel.generate_content"
            )
        elif test_lang == "nodejs":
            resource = (
                "GenerativeModel.generateContentStream"
                if stream
                else "GenerativeModel.generateContent"
            )

        assert_apm_span(
            span=traces[0][0],
            name="vertexai.request",
            resource=resource,
            tags=[
                ("vertexai.request.model", "gemini-1.5-flash-002"),
                ("vertexai.request.provider", "google"),
            ],
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_completion_error(self, test_client, test_agent):
        with pytest.raises(Exception):
            test_client.vertexai_completion(
                prompt="How are you today?",
                parameters={
                    "generation_config": GENERATION_CONFIG,
                    "candidate_count": 2,  # candidate_count is not a valid keyword argument
                },
            )
        traces = test_agent.wait_for_num_traces(num=1)

        assert traces[0][0]["name"] == "vertexai.request"
        assert (
            traces[0][0]["resource"] == "GenerativeModel.generate_content"
            or traces[0][0]["resource"] == "GenerativeModel.generateContent"
        )
        assert traces[0][0]["error"] == 1
        assert traces[0][0]["meta"]["vertexai.request.model"] == "gemini-1.5-flash-002"

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_completion_multiple_messages(self, test_lang, test_client, test_agent):
        test_client.vertexai_completion(
            prompt=[
                "Hello World!",
                "How are you today?",
            ],
            parameters={"generation_config": GENERATION_CONFIG},
            system_instruction="Please answer every question with a one sentence response.",
        )
        traces = test_agent.wait_for_num_traces(num=1)

        assert_apm_span(
            span=traces[0][0],
            name="vertexai.request",
            resource="GenerativeModel.generate_content"
            if test_lang == "python"
            else "GenerativeModel.generateContent",
            tags=[
                ("vertexai.request.model", "gemini-1.5-flash-002"),
                ("vertexai.request.provider", "google"),
            ],
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_completion_tool(self, test_lang, test_client, test_agent):
        test_client.vertexai_completion(
            prompt="What is the weather like in Boston, MA in Fahrenheit?",
            parameters={"generation_config": GENERATION_CONFIG},
            tools=TOOLS,
            tool_config=TOOL_CONFIG,
        )
        traces = test_agent.wait_for_num_traces(num=1)

        assert_apm_span(
            span=traces[0][0],
            name="vertexai.request",
            resource="GenerativeModel.generate_content"
            if test_lang == "python"
            else "GenerativeModel.generateContent",
            tags=[
                ("vertexai.request.model", "gemini-1.5-flash-002"),
                ("vertexai.request.provider", "google"),
            ],
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("asynchronous", [True, False])
    def test_chat_completion(
        self, test_lang, test_client, test_agent, stream, asynchronous
    ):
        if test_lang == "nodejs" and asynchronous:
            pytest.skip(
                "Node.js VertexAI SDK does not implement the `sendMessageAsync` method. `sendMessage` is already asynchronous."
            )
        test_client.vertexai_chat_completion(
            prompt="How are you today?",
            parameters={"generation_config": GENERATION_CONFIG, "stream": stream},
            asynchronous=asynchronous,
        )
        traces = test_agent.wait_for_num_traces(num=1)

        resource = None
        if test_lang == "python":
            resource = (
                "ChatSession.send_message_async"
                if asynchronous
                else "ChatSession.send_message"
            )
        elif test_lang == "nodejs":
            resource = (
                "ChatSession.sendMessageStream" if stream else "ChatSession.sendMessage"
            )

        assert_apm_span(
            span=traces[0][0],
            name="vertexai.request",
            resource=resource,
            tags=[
                ("vertexai.request.model", "gemini-1.5-flash-002"),
                ("vertexai.request.provider", "google"),
            ],
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_chat_completion_error(self, test_client, test_agent):
        with pytest.raises(Exception):
            test_client.vertexai_chat_completion(
                prompt="How are you today?",
                parameters={
                    "generation_config": GENERATION_CONFIG,
                    "candidate_count": 2,  # candidate_count is not a valid keyword argument
                },
            )
        traces = test_agent.wait_for_num_traces(num=1)

        assert traces[0][0]["name"] == "vertexai.request"
        assert (
            traces[0][0]["resource"] == "ChatSession.send_message"
            or traces[0][0]["resource"] == "ChatSession.sendMessage"
        )
        assert traces[0][0]["error"] == 1
        assert traces[0][0]["meta"]["vertexai.request.model"] == "gemini-1.5-flash-002"

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_chat_completion_multiple_messages(
        self, test_lang, test_client, test_agent
    ):
        test_client.vertexai_chat_completion(
            prompt=[
                "Hello World!",
                "How are you today?",
            ],
            parameters={"generation_config": GENERATION_CONFIG},
        )
        traces = test_agent.wait_for_num_traces(num=1)

        assert_apm_span(
            span=traces[0][0],
            name="vertexai.request",
            resource="ChatSession.sendMessage"
            if test_lang == "nodejs"
            else "ChatSession.send_message",
            tags=[
                ("vertexai.request.model", "gemini-1.5-flash-002"),
                ("vertexai.request.provider", "google"),
            ],
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_chat_completion_tool(self, test_lang, test_client, test_agent):
        test_client.vertexai_chat_completion(
            prompt="What is the weather like in Boston, MA in Fahrenheit?",
            parameters={"generation_config": GENERATION_CONFIG},
            tools=TOOLS,
            tool_config=TOOL_CONFIG,
        )
        traces = test_agent.wait_for_num_traces(num=1)

        assert_apm_span(
            span=traces[0][0],
            name="vertexai.request",
            resource="ChatSession.sendMessage"
            if test_lang == "nodejs"
            else "ChatSession.send_message",
            tags=[
                ("vertexai.request.model", "gemini-1.5-flash-002"),
                ("vertexai.request.provider", "google"),
            ],
        )


class TestVertexaiLLMObs:
    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("asynchronous", [True, False])
    def test_completion(self, test_lang, test_client, test_agent, stream, asynchronous):
        if test_lang == "nodejs" and asynchronous:
            pytest.skip(
                "Node.js VertexAI SDK does not implement the `generateContentAsync` method. `generateContent` is already asynchronous."
            )
        test_client.vertexai_completion(
            prompt="How are you today?",
            parameters={"generation_config": GENERATION_CONFIG, "stream": stream},
            asynchronous=asynchronous,
        )
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        name = None
        if test_lang == "python":
            name = (
                "GenerativeModel.generate_content_async"
                if asynchronous
                else "GenerativeModel.generate_content"
            )
        elif test_lang == "nodejs":
            name = (
                "GenerativeModel.generateContentStream"
                if stream
                else "GenerativeModel.generateContent"
            )

        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        inputs = [{"content": "How are you today?"}]
        if test_lang == "nodejs":
            inputs[0]["role"] = "user"

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name=name,
            span_kind="llm",
            input=inputs,
            output=[{"content": mock.ANY, "role": "model"}],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_completion_error(self, test_lang, test_client, test_agent):
        with pytest.raises(Exception):
            test_client.vertexai_completion(
                prompt="How are you today?",
                parameters={
                    "generation_config": GENERATION_CONFIG,
                    "candidate_count": 2,  # candidate_count is not a valid keyword argument
                },
            )
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name="GenerativeModel.generate_content"
            if test_lang == "python"
            else "GenerativeModel.generateContent",
            span_kind="llm",
            input=[{"content": "How are you today?"}],
            output=[{"content": ""}],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={},
            status="error",
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_completion_multiple_messages(self, test_lang, test_client, test_agent):
        test_client.vertexai_completion(
            prompt=[
                "Hello World!",
                "How are you today?",
            ],
            parameters={"generation_config": GENERATION_CONFIG},
        )
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        inputs = [{"content": "Hello World!"}, {"content": "How are you today?"}]
        if test_lang == "nodejs":  # nodejs has to set role for multiple inputs
            for inp in inputs:
                inp["role"] = "user"

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name="GenerativeModel.generate_content"
            if test_lang == "python"
            else "GenerativeModel.generateContent",
            span_kind="llm",
            input=inputs,
            output=[{"content": mock.ANY, "role": "model"}],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_completion_tool(self, test_lang, test_client, test_agent):
        test_client.vertexai_completion(
            prompt="What is the weather like in Boston, MA in Fahrenheit?",
            parameters={"generation_config": GENERATION_CONFIG},
            tools=TOOLS,
            tool_config=TOOL_CONFIG,
        )
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        inputs = [{"content": "What is the weather like in Boston, MA in Fahrenheit?"}]
        if test_lang == "nodejs":
            inputs[0]["role"] = "user"

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name="GenerativeModel.generate_content"
            if test_lang == "python"
            else "GenerativeModel.generateContent",
            span_kind="llm",
            input=inputs,
            output=[
                {
                    "content": "",
                    "role": "model",
                    "tool_calls": [
                        {
                            "name": "get_current_weather",
                            "arguments": {
                                "location": "Boston, MA",
                                "unit": "fahrenheit",
                            },
                        }
                    ],
                }
            ],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("asynchronous", [True, False])
    def test_chat_completion(
        self, test_lang, test_client, test_agent, stream, asynchronous
    ):
        if test_lang == "nodejs" and asynchronous:
            pytest.skip(
                "Node.js VertexAI SDK does not implement the `sendMessageAsync` method. `sendMessage` is already asynchronous."
            )
        test_client.vertexai_chat_completion(
            prompt="How are you today?",
            parameters={"generation_config": GENERATION_CONFIG, "stream": stream},
            asynchronous=asynchronous,
        )
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        name = None
        if test_lang == "python":
            name = (
                "ChatSession.send_message_async"
                if asynchronous
                else "ChatSession.send_message"
            )
        elif test_lang == "nodejs":
            name = (
                "ChatSession.sendMessageStream" if stream else "ChatSession.sendMessage"
            )

        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name=name,
            span_kind="llm",
            input=[{"content": "How are you today?"}],
            output=[{"content": mock.ANY, "role": "model"}],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_chat_completion_error(self, test_lang, test_client, test_agent):
        with pytest.raises(Exception):
            test_client.vertexai_chat_completion(
                prompt="How are you today?",
                parameters={
                    "generation_config": GENERATION_CONFIG,
                    "candidate_count": 2,  # candidate_count is not a valid keyword argument
                },
            )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name="ChatSession.send_message"
            if test_lang == "python"
            else "ChatSession.sendMessage",
            span_kind="llm",
            input=[{"content": "How are you today?"}],
            output=[{"content": ""}],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={},
            status="error",
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_chat_completion_multiple_messages(
        self, test_lang, test_client, test_agent
    ):
        test_client.vertexai_chat_completion(
            prompt=[
                "Hello World!",
                "How are you today?",
            ],
            parameters={"generation_config": GENERATION_CONFIG},
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name="ChatSession.send_message"
            if test_lang == "python"
            else "ChatSession.sendMessage",
            span_kind="llm",
            input=[{"content": "Hello World!"}, {"content": "How are you today?"}],
            output=[{"content": mock.ANY, "role": "model"}],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("nodejs", version="5.0.0")
    @supported("python", version="2.18.0")
    @unsupported("java", reason="Java SDK does not auto-instrument VertexAI")
    def test_chat_completion_tool(self, test_lang, test_client, test_agent):
        test_client.vertexai_chat_completion(
            prompt="What is the weather like in Boston, MA in Fahrenheit?",
            parameters={"generation_config": GENERATION_CONFIG},
            tools=TOOLS,
            tool_config=TOOL_CONFIG,
        )
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        assert len(reqs) == 1
        assert reqs[0][0]["event_type"] == "span"
        assert len(reqs[0][0]["spans"]) == 1

        span_event = reqs[0][0]["spans"][0]
        assert_llmobs_span_event(
            span_event,
            name="ChatSession.send_message"
            if test_lang == "python"
            else "ChatSession.sendMessage",
            span_kind="llm",
            input=[
                {"content": "What is the weather like in Boston, MA in Fahrenheit?"}
            ],
            output=[
                {
                    "content": "",
                    "role": "model",
                    "tool_calls": [
                        {
                            "name": "get_current_weather",
                            "arguments": {
                                "location": "Boston, MA",
                                "unit": "fahrenheit",
                            },
                        }
                    ],
                }
            ],
            model_name="gemini-1.5-flash-002",
            model_provider="google",
            metadata={"temperature": 1.0, "max_output_tokens": 1000},
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )
