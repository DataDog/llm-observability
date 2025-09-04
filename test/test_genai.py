from unittest import mock
import pytest
import json

from test.conftest import LLMObsTestAgentClient
from test.client import InstrumentationClient as LLMObsInstrumentationClient

from test import supported, unsupported
from test.utils import assert_apm_span, assert_llmobs_span_event


GET_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for",
            },
        },
    },
}


def _format_expected_genai_generate_content_metadata(**metadata):
    expected_metadata = {
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "candidate_count": None,
        "max_output_tokens": None,
        "stop_sequences": None,
        "response_logprobs": None,
        "logprobs": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "seed": None,
        "response_mime_type": None,
        "safety_settings": None,
        "automatic_function_calling": None,
        "tools": None,
    }

    for key, value in metadata.items():
        if value is not None:
            expected_metadata[key] = value

    return expected_metadata


class TestGenAiApm:
    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        """Tests the generate_content and generate_content_stream endpoints."""
        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents="Why did the chicken cross the road?",
            config=dict(
                temperature=0.1,
                max_output_tokens=50,
                stream=stream,
            ),
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        span_resource = (
            "Models.generate_content_stream" if stream else "Models.generate_content"
        )

        assert_apm_span(
            span=span,
            name="google_genai.request",
            resource=span_resource,
            tags=[
                ("google_genai.request.model", "gemini-2.0-flash"),
                ("google_genai.request.provider", "google"),
            ],
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    def test_embed_content(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """Tests the embed_content endpoint."""
        test_client.genai_embed_content(
            model="gemini-embedding-001",
            contents="Why did the chicken cross the road?",
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span=span,
            name="google_genai.request",
            resource="Models.embed_content",
            tags=[
                ("google_genai.request.model", "gemini-embedding-001"),
                ("google_genai.request.provider", "google"),
            ],
        )


class TestGenAiLlmObs:
    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents="Why did the chicken cross the road?",
            config=dict(
                temperature=0.1,
                max_output_tokens=50,
                stream=stream,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            max_output_tokens=50,
            temperature=0.1,
        )

        expected_output_content = "This is a classic joke! Here are a few possible answers, ranging from the traditional to the more absurd:\n\n*   **The Classic:** To get to the other side.\n*   **The Logical:** Because there was a road there."

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.0-flash",
            model_provider="google",
            input=[{"role": "user", "content": "Why did the chicken cross the road?"}],
            output=[{"role": "assistant", "content": expected_output_content}],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_multiple_strings_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents=["Why did the chicken cross the road?", "What is 2 + 2?"],
            config=dict(
                temperature=0.1,
                max_output_tokens=50,
                stream=stream,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            max_output_tokens=50,
            temperature=0.1,
        )

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.0-flash",
            model_provider="google",
            input=[
                {"role": "user", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            output=[
                {
                    "role": "assistant",
                    "content": "Okay, here are the answers to your questions:\n\n*   **Why did the chicken cross the road?** To get to the other side.\n\n*   **What is 2 + 2?** 4",
                }
            ],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_parts_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents=[
                {"text": "Why did the chicken cross the road?"},
                {"text": "What is 2 + 2?"},
            ],
            config=dict(
                temperature=0.1,
                max_output_tokens=50,
                stream=stream,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            max_output_tokens=50,
            temperature=0.1,
        )

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.0-flash",
            model_provider="google",
            input=[
                {"role": "user", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            output=[
                {
                    "role": "assistant",
                    "content": "Okay, here are the answers to your questions:\n\n*   **Why did the chicken cross the road?** To get to the other side.\n\n*   **What is 2 + 2?** 4",
                }
            ],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_content_block_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents=[
                {
                    "parts": [{"text": "Why did the chicken cross the road?"}],
                    "role": "user",
                },
            ],
            config=dict(
                temperature=0.1,
                max_output_tokens=50,
                stream=stream,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            max_output_tokens=50,
            temperature=0.1,
        )

        expected_output_content = "This is a classic joke! Here are a few possible answers, ranging from the traditional to the more absurd:\n\n*   **The Classic:** To get to the other side.\n*   **The Logical:** Because there was a road there."

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.0-flash",
            model_provider="google",
            input=[{"role": "user", "content": "Why did the chicken cross the road?"}],
            output=[
                {
                    "role": "assistant",
                    "content": expected_output_content,
                }
            ],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_reasoning_output(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        test_lang: str,
        stream: bool,
    ):
        if test_lang == "python" and stream:
            pytest.skip(
                "python does not report `reasoning` role for streamed responses"
            )

        test_client.genai_generate_content(
            model="gemini-2.5-pro",
            contents="If x + 9 = 10, what is the value of x?",
            config=dict(
                thinking_config=dict(
                    thinking_budget=1024,
                    include_thoughts=True,
                ),
                stream=stream,
                temperature=0.1,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            temperature=0.1,
        )

        reasoning_content = (
            "**Deconstructing a Simple Linear Equation**\n\nOkay, so the user wants to solve for 'x' in the equation `x + 9 = 10`. "
            "This is a classic one-step linear equation, nothing complicated. My approach is pretty straightforward: isolate 'x'.\n\n"
            "First, I look at the equation. 'x' has a '+ 9' attached to it.  To get 'x' by itself, I need to undo that addition.  "
            "The inverse operation of addition is subtraction, so I need to subtract 9 from both sides of the equation. This is "
            "key to maintaining the equation's balance.\n\nHere's how I'd break it down:\n\n1.  Start with the given: `x + 9 = 10`\n2.  "
            "Subtract 9 from both sides: `x + 9 - 9 = 10 - 9`\n3.  Simplify:  This gives me `x = 1`\n\nNow, to make sure I'm right, "
            "I always check my work. I'll substitute the value I found back into the original equation.  So, I replace 'x' with '1' in "
            "the original: `1 + 9 = 10`.  Yep, that checks out; 1 plus 9 does indeed equal 10. The equation `10 = 10` is true, "
            "so I'm confident my answer is correct.  Simple as that!\n"
        )
        assistant_content = "To find the value of x, you need to get x by itself on one side of the equation.\n\nGiven the equation:\nx + 9 = 10\n\nSubtract 9 from both sides of the equation:\nx + 9 - 9 = 10 - 9\nx = 1\n\nSo, the value of **x is 1**."

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.5-pro",
            model_provider="google",
            input=[
                {"role": "user", "content": "If x + 9 = 10, what is the value of x?"}
            ],
            output=[
                {"role": "reasoning", "content": reasoning_content},
                {"role": "assistant", "content": assistant_content},
            ],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_reasoning_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        input_messages = [
            {
                "parts": [{"text": "If x + 9 = 10, what is the value of x?"}],
                "role": "user",
            },
            {
                "parts": [
                    {"text": "Since 1 + 9 = 10, the value of x is 1.", "thought": True}
                ],
                "role": "model",
            },
            {"parts": [{"text": "The value of x is 1."}], "role": "model"},
            {"parts": [{"text": "What is that number plus 3?"}], "role": "user"},
        ]

        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents=input_messages,
            config=dict(
                stream=stream,
                temperature=0.1,
                max_output_tokens=50,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            max_output_tokens=50,
            temperature=0.1,
        )

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.0-flash",
            model_provider="google",
            input=[
                {"role": "user", "content": "If x + 9 = 10, what is the value of x?"},
                {
                    "role": "reasoning",
                    "content": "Since 1 + 9 = 10, the value of x is 1.",
                },
                {"role": "assistant", "content": "The value of x is 1."},
                {"role": "user", "content": "What is that number plus 3?"},
            ],
            output=[
                {
                    "role": "assistant",
                    "content": "The number is 1. Adding 3 to it gives 1 + 3 = 4.\n\nSo the answer is 4.\n",
                }
            ],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_with_tools(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents="What is the weather in Tokyo?",
            config=dict(
                max_output_tokens=50,
                stream=stream,
                tools=[{"function_declarations": [GET_WEATHER_TOOL]}],
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            max_output_tokens=50,
            tools=[{"function_declarations": [GET_WEATHER_TOOL]}],
        )

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.0-flash",
            model_provider="google",
            input=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            output=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "get_weather", "arguments": {"location": "Tokyo"}}
                    ],
                }
            ],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_with_tool_response(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        input_messages = [
            {"parts": [{"text": "What is the weather in Tokyo?"}], "role": "user"},
            {
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"location": "Tokyo"},
                            "id": "abc123",
                        }
                    }
                ],
                "role": "model",
            },
            {
                "parts": [
                    {
                        "function_response": {
                            "name": "get_weather",
                            "response": {"weather": "sunny", "temperature": "78°F"},
                            "id": "abc123",
                        }
                    }
                ],
                "role": "user",
            },
        ]

        test_client.genai_generate_content(
            model="gemini-2.0-flash",
            contents=input_messages,
            config=dict(
                stream=stream,
                temperature=0.1,
                max_output_tokens=50,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        expected_metadata = _format_expected_genai_generate_content_metadata(
            max_output_tokens=50,
            temperature=0.1,
        )

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="llm",
            model_name="gemini-2.0-flash",
            model_provider="google",
            input=[
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "get_weather", "arguments": {"location": "Tokyo"}}
                    ],
                },
                {
                    "role": "tool",
                    "content": "{'weather': 'sunny', 'temperature': '78°F'}",
                    "tool_id": "abc123",
                },
            ],
            output=[
                {
                    "role": "assistant",
                    "content": "The weather in Tokyo is sunny with a temperature of 78°F.\n",
                }
            ],
            metadata=expected_metadata,
            metrics={
                "input_tokens": mock.ANY,
                "output_tokens": mock.ANY,
                "total_tokens": mock.ANY,
            },
        )

    @supported("python", version="3.12.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    @pytest.mark.parametrize("stream", [True, False])
    def test_generate_content_executable_code(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.genai_generate_content(
            model="gemini-2.5-flash",
            contents="What is the sum of the first 50 prime numbers? Generate and run code for the calculation, and make sure you get all 50.",
            config=dict(
                tools=[{"code_execution": {}}],
                stream=stream,
            ),
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        if stream:
            assert len(span_event["meta"]["output"]["messages"]) == 3
            expected_output_messages = [
                {"role": "assistant", "content": mock.ANY},
                {
                    "role": "assistant",
                    "content": {"language": mock.ANY, "code": mock.ANY},
                },
                {
                    "role": "assistant",
                    "content": {"outcome": mock.ANY, "output": mock.ANY},
                },
            ]
        else:
            #  non-streamed responses return the outcome of the third block as the content of the last block for some reason
            assert len(span_event["meta"]["output"]["messages"]) == 4
            expected_output_messages = [
                {"role": "assistant", "content": mock.ANY},
                {
                    "role": "assistant",
                    "content": {"language": mock.ANY, "code": mock.ANY},
                },
                {
                    "role": "assistant",
                    "content": {"outcome": mock.ANY, "output": mock.ANY},
                },
                {"role": "assistant", "content": mock.ANY},
            ]

        actual_output_messages = span_event["meta"]["output"]["messages"]

        #  order is not guaranteed for output messages from the Gemini API, so we do a check over each set of actual & expected messages
        for expected_message in expected_output_messages:
            found = False
            for actual_message in actual_output_messages:
                try:
                    assert actual_message["role"] == expected_message["role"]
                    try:
                        assert (
                            json.loads(actual_message["content"])
                            == expected_message["content"]
                        )
                    except json.JSONDecodeError:
                        assert actual_message["content"] == expected_message["content"]
                    found = True
                    break
                except:  # noqa: E722
                    continue
            assert found, (
                f"Did not find expected message {expected_message} in {actual_output_messages}"
            )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    def test_embed_content(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.genai_embed_content(
            model="gemini-embedding-001",
            contents="Why did the chicken cross the road?",
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        assert_llmobs_span_event(
            span_event,
            span_kind="embedding",
            name="google_genai.request",
            model_name="gemini-embedding-001",
            model_provider="google",
            input=[{"text": "Why did the chicken cross the road?"}],
            output="[1 embedding(s) returned with size 3072]",
            metadata={},
            metrics={},
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    def test_embed_content_multiple_strings_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.genai_embed_content(
            model="gemini-embedding-001",
            contents=["Why did the chicken cross the road?", "What is 2 + 2?"],
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="embedding",
            model_name="gemini-embedding-001",
            model_provider="google",
            input=[
                {"text": "Why did the chicken cross the road?"},
                {"text": "What is 2 + 2?"},
            ],
            output="[2 embedding(s) returned with size 3072]",
            metadata={},
            metrics={},
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    def test_embed_content_parts_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.genai_embed_content(
            model="gemini-embedding-001",
            contents=[
                {"text": "Why did the chicken cross the road?"},
                {"text": "What is 2 + 2?"},
            ],
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="embedding",
            model_name="gemini-embedding-001",
            model_provider="google",
            input=[
                {"text": "Why did the chicken cross the road?"},
                {"text": "What is 2 + 2?"},
            ],
            output="[2 embedding(s) returned with size 3072]",
            metadata={},
            metrics={},
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support genai")
    @unsupported("java", reason="java does not support genai")
    def test_embed_content_content_block_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.genai_embed_content(
            model="gemini-embedding-001",
            contents=[
                {
                    "parts": [
                        {"text": "Why did the chicken cross the road?"},
                        {"text": "What is 2 + 2?"},
                    ],
                    "role": "user",
                },
            ],
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        assert len(reqs) == 1
        assert len(reqs[0][0]["spans"]) == 1
        span_event = reqs[0][0]["spans"][0]

        assert_llmobs_span_event(
            span_event,
            name="google_genai.request",
            span_kind="embedding",
            model_name="gemini-embedding-001",
            model_provider="google",
            input=[
                {"text": "Why did the chicken cross the road?"},
                {"text": "What is 2 + 2?"},
            ],
            output="[1 embedding(s) returned with size 3072]",
            metadata={},
            metrics={},
        )
