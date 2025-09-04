from unittest import mock
import pytest

from test.conftest import LLMObsTestAgentClient
from test.client import InstrumentationClient as LLMObsInstrumentationClient

from test import supported, unsupported
from test.utils import assert_llmobs_span_event, assert_apm_span
from test.utils import COUNT
from test.utils import find_event_tag
from test.utils import get_all_span_events

skip = pytest.mark.skip

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_student_info",
            "description": "Get the student information from the body of the input text",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the person"},
                    "major": {"type": "string", "description": "Major subject."},
                    "school": {
                        "type": "string",
                        "description": "The university name.",
                    },
                },
            },
        },
    }
]


class TestOpenAiApm:
    """Asserts the structure and contents of APM spans generated for the OpenAI integrations"""

    @supported("python", version="1.14.0")
    @supported("nodejs", version="4.4.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    @pytest.mark.parametrize("stream", [True, False])
    def test_chat_completion(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests streamed and non-streamed responses for the chat completion OpenAI endpoint"""

        test_client.openai_chat_completion(
            model="gpt-3.5-turbo",
            prompt="Why is Evan Li such a slacker?",
            parameters={"max_tokens": 35, "stream": stream},
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span,
            name="openai.request",
            resource=("chat.completions.create", "createChatCompletion"),
            tags=[
                ("openai.request.model", "gpt-3.5-turbo"),
            ],
        )

    @supported("python", version="1.14.0")
    @supported("nodejs", version="4.4.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    def test_completion(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """Tests the completion OpenAI endpoint"""

        test_client.openai_completion(
            model="gpt-3.5-turbo-instruct",
            prompt="Why is Evan Li such a slacker?",
            parameters={"max_tokens": 35, "temperature": 0.5},
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span,
            name="openai.request",
            resource=("completions.create", "createCompletion"),
            tags=[
                ("openai.request.model", "gpt-3.5-turbo-instruct"),
            ],
        )

    @supported("python", version="1.14.0")
    @supported("nodejs", version="4.4.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    def test_embedding(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """Tests the embedding OpenAI endpoint"""

        test_client.openai_embedding(
            model="text-embedding-ada-002", input="hello world"
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span,
            name="openai.request",
            resource=("embeddings.create", "createEmbedding"),
            tags=[
                ("openai.request.model", "text-embedding-ada-002"),
            ],
        )

    @supported("python", version="2.2.0")
    @supported("nodejs", version="4.36.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    @pytest.mark.parametrize("stream", [True, False])
    def test_chat_completion_tool_call(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests the chat completion OpenAI endpoint with a tool call"""

        test_client.openai_chat_completion(
            prompt="Bob is a student at Stanford University. He is studying computer science.",
            model="gpt-3.5-turbo",
            tools=TOOLS,
            parameters={"stream": stream},
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span,
            name="openai.request",
            resource=("chat.completions.create", "createChatCompletion"),
            tags=[
                ("openai.request.model", "gpt-3.5-turbo"),
            ],
        )

    @supported("python", version="3.8.0")
    @unsupported("nodejs", reason="nodejs does not support openai responses")
    @unsupported("java", reason="java does not support openai responses")
    @pytest.mark.parametrize("stream", [True, False])
    def test_responses_create(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests the responses.create OpenAI endpoint. Asserts basic request options without input or output tagging."""

        test_client.openai_responses_create(
            model="gpt-4.1",
            input="Do not continue the Evan Li slander!",
            parameters=dict(
                max_output_tokens=50,
                temperature=0.1,
                stream=stream,
                instructions="Talk with a Boston accent.",
            ),
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span=span,
            name="openai.request",
            resource="createResponse",
            tags=[
                ("openai.request.model", "gpt-4.1"),
            ],
        )


class TestOpenAiLlmObs:
    """Asserts the structure and contents of LLMObs span events generated for the OpenAI integrations"""

    @supported("python", version="2.9.0")
    @supported("nodejs", version="5.25.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    @pytest.mark.parametrize("stream", [True, False])
    def test_chat_completion(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        test_lang: str,
        stream: bool,
    ):
        """Tests streamed and non-streamed responses for the chat completion OpenAI endpoint"""
        test_client.openai_chat_completion(
            model="gpt-3.5-turbo",
            prompt="Why is Evan Li such a slacker?",
            parameters={"max_tokens": 35, "stream": stream},
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        expected_metadata = {"max_tokens": 35}
        if stream:
            expected_metadata["stream"] = True
            expected_metadata["stream_options"] = {"include_usage": True}

        expected_metrics = {
            "input_tokens": mock.ANY,
            "output_tokens": mock.ANY,
            "total_tokens": mock.ANY,
        }

        if test_lang == "python":
            expected_metrics["cache_read_input_tokens"] = mock.ANY

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createChatCompletion",
            span_kind="llm",
            input=[{"content": "Why is Evan Li such a slacker?", "role": "user"}],
            output=[{"content": mock.ANY, "role": "assistant"}],
            model_name="gpt-3.5-turbo",
            model_provider="openai",
            metadata=expected_metadata,
            metrics=expected_metrics,
        )

    @supported("python", version="2.9.0")
    @supported("nodejs", version="5.25.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    def test_completion(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """Tests the completion OpenAI endpoint"""

        test_client.openai_completion(
            model="gpt-3.5-turbo-instruct",
            prompt="Why is Evan Li such a slacker?",
            parameters={"max_tokens": 35, "temperature": 0.5},
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createCompletion",
            span_kind="llm",
            input=[{"content": "Why is Evan Li such a slacker?"}],
            output=[{"content": mock.ANY}],
            model_name="gpt-3.5-turbo-instruct",
            model_provider="openai",
            metadata={"max_tokens": 35, "temperature": 0.5},
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="2.9.0")
    @supported("nodejs", version="5.25.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    def test_embedding(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """Tests the embedding OpenAI endpoint"""
        test_client.openai_embedding(
            model="text-embedding-ada-002", input="hello world"
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createEmbedding",
            span_kind="embedding",
            input=[{"text": "hello world"}],
            output="[1 embedding(s) returned with size 1536]",
            model_name="text-embedding-ada-002",
            model_provider="openai",
        )

    @supported("python", version="2.9.0")
    @supported("nodejs", version="5.25.0")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    @pytest.mark.parametrize("stream", [True, False])
    def test_chat_completion_tool_call(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests the chat completion OpenAI endpoint with a tool call"""

        test_client.openai_chat_completion(
            prompt="Bob is a student at Stanford University. He is studying computer science.",
            model="gpt-3.5-turbo",
            tools=TOOLS,
            parameters={"stream": stream},
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createChatCompletion",
            span_kind="llm",
            input=[
                {
                    "content": "Bob is a student at Stanford University. He is studying computer science.",
                    "role": "user",
                }
            ],
            output=[
                {
                    "content": mock.ANY,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": "extract_student_info",
                            "arguments": {
                                "name": mock.ANY,
                                "major": mock.ANY,
                                "school": mock.ANY,
                            },
                            "tool_id": mock.ANY,
                            "type": "function",
                        }
                    ],
                }
            ],
            model_name="gpt-3.5-turbo",
            model_provider="openai",
        )

    @supported("python", version="3.5.0")
    @unsupported("nodejs", reason="test wip")
    @unsupported("java", reason="Java SDK does not auto-instrument OpenAI")
    @pytest.mark.parametrize("stream", [True])
    def test_chat_completion_telemetry(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests streamed and non-streamed responses for the chat completion OpenAI endpoint"""
        test_client.openai_chat_completion(
            model="gpt-3.5-turbo",
            prompt="Why is Evan Li such a slacker?",
            parameters={"max_tokens": 35, "stream": stream},
        )
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="span.finished")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"
        assert find_event_tag(metric, "autoinstrumented") == "1"
        assert find_event_tag(metric, "is_root_span") == "1"
        assert find_event_tag(metric, "span_kind") == "llm"
        assert find_event_tag(metric, "integration") == "openai"

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support openai responses")
    @unsupported("java", reason="java does not support openai responses")
    @pytest.mark.parametrize("stream", [True, False])
    def test_responses_create(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests streamed and non-streamed responses for the responses.create OpenAI endpoint"""

        test_client.openai_responses_create(
            model="gpt-4.1",
            input="Do not continue the Evan Li slander!",
            parameters=dict(
                max_output_tokens=50,
                temperature=0.1,
                stream=stream,
                instructions="Talk with a Boston accent.",
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        expected_metadata = dict(
            max_output_tokens=50,
            temperature=0.1,
            top_p=1.0,
            tools=[],
            tool_choice="auto",
            truncation="disabled",
            text={"format": {"type": "text"}},
            reasoning_tokens=0,
        )

        if stream:
            expected_metadata["stream"] = True
            expected_output = "Alright, I hea ya loud and clear! No more Evan Li slandah, I promise. We’ll keep it wicked respectful from now on, kid. If ya got anythin’ else ya wanna chat about—sports, chowdah recipes"
        else:
            expected_output = "Alright, I hea ya loud an’ cleah! No more Evan Li slandah, I promise. We’ll keep it wicked respectful from now on. If ya got anythin’ nice to say about Evan, I’m all eahs!"

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createResponse",
            span_kind="llm",
            input=[
                {"role": "system", "content": "Talk with a Boston accent."},
                {"role": "user", "content": "Do not continue the Evan Li slander!"},
            ],
            output=[{"role": "assistant", "content": expected_output}],
            model_name="gpt-4.1",
            model_provider="openai",
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
                "cache_read_input_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support openai responses")
    @unsupported("java", reason="java does not support openai responses")
    @pytest.mark.parametrize("stream", [True, False])
    def test_responses_create_tool_call(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.openai_responses_create(
            model="gpt-4.1",
            input="Bob is a student at Stanford University. He is studying computer science.",
            parameters=dict(max_output_tokens=50, temperature=0.1, stream=stream),
            tools=[
                {"type": "function", **TOOLS[0]["function"]}
            ],  # different format for responses tools
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createResponse",
            span_kind="llm",
            input=[
                {
                    "role": "user",
                    "content": "Bob is a student at Stanford University. He is studying computer science.",
                }
            ],
            output=[
                {
                    "role": "",
                    "tool_calls": [
                        {
                            "name": "extract_student_info",
                            "arguments": {
                                "name": "Bob",
                                "major": "computer science",
                                "school": "Stanford University",
                            },
                            "tool_id": mock.ANY,
                            "type": "function_call",
                        }
                    ],
                }
            ],
            model_name="gpt-4.1",
            model_provider="openai",
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
                "cache_read_input_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support openai responses")
    @unsupported("java", reason="java does not support openai responses")
    @pytest.mark.parametrize("stream", [True, False])
    def test_responses_create_reasoning(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests that reasoning responses are properly annotated"""

        test_client.openai_responses_create(
            model="o4-mini",
            input="If one plus a number is 10, what is the number?",
            parameters=dict(
                reasoning={"effort": "medium", "summary": "detailed"}, stream=stream
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        expected_metadata = dict(
            reasoning={"effort": "medium", "summary": "detailed"},
            temperature=1.0,
            top_p=1.0,
            tools=[],
            tool_choice="auto",
            truncation="disabled",
            text={"format": {"type": "text"}},
            reasoning_tokens=mock.ANY,
        )

        if stream:
            expected_metadata["stream"] = True
            expected_output = "The number is 9, since 1 + n = 10 ⇒ n = 10 − 1 = 9."
        else:
            expected_output = "The number is 9, since 1 + 9 = 10."

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createResponse",
            span_kind="llm",
            input=[
                {
                    "role": "user",
                    "content": "If one plus a number is 10, what is the number?",
                }
            ],
            output=[
                {"role": "reasoning", "content": mock.ANY},
                {"role": "assistant", "content": expected_output},
            ],
            model_name="o4-mini",
            model_provider="openai",
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
                "cache_read_input_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support openai responses")
    @unsupported("java", reason="java does not support openai responses")
    @pytest.mark.parametrize("stream", [True, False])
    def test_responses_create_tool_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        input_messages = [
            {"role": "user", "content": "What's the weather like in San Francisco?"},
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"location": "San Francisco, CA"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"temperature": "72°F", "conditions": "sunny", "humidity": "65%"}',
            },
        ]

        test_client.openai_responses_create(
            model="gpt-4.1",
            input=input_messages,
            parameters=dict(temperature=0.1, stream=stream),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        expected_metadata = dict(
            temperature=0.1,
            top_p=1.0,
            tools=[],
            tool_choice="auto",
            truncation="disabled",
            text={"format": {"type": "text"}},
            reasoning_tokens=0,
        )

        if stream:
            expected_metadata["stream"] = True

        assert_llmobs_span_event(
            span_event,
            name="OpenAI.createResponse",
            span_kind="llm",
            input=[
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco?",
                },
                {
                    "role": "",
                    "tool_calls": [
                        {
                            "tool_id": "call_123",
                            "name": "get_weather",
                            "arguments": {"location": "San Francisco, CA"},
                            "type": "function_call",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"temperature": "72°F", "conditions": "sunny", "humidity": "65%"}',
                },
            ],
            output=[
                {
                    "role": "assistant",
                    "content": mock.ANY,  # temperature = 0 does not guarantee a specific output here
                }
            ],
            model_name="gpt-4.1",
            model_provider="openai",
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
                "cache_read_input_tokens": mock.ANY,
            },
        )
