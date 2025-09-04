import json
import pytest
from unittest import mock

from test.conftest import LLMObsTestAgentClient
from test.client import InstrumentationClient as LLMObsInstrumentationClient

from test import supported
from test import unsupported

from test.utils import assert_apm_span
from test.utils import assert_llmobs_span_event
from test.utils import get_all_span_events

DEFAULT_PARAMETERS = dict(
    max_tokens=100,
    temperature=0.5,
)

DEFAULT_MODEL = "claude-3-7-sonnet-20250219"

EXPECTED_TOKENS = dict(
    input_tokens=mock.ANY,
    output_tokens=mock.ANY,
    total_tokens=mock.ANY,
    cache_write_input_tokens=mock.ANY,
    cache_read_input_tokens=mock.ANY,
)

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    }
]


class TestAnthropicApm:
    @supported("python", version="2.10.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span=span,
            name="anthropic.request",
            resource="Messages.create",
            tags=[
                ("anthropic.request.model", DEFAULT_MODEL),
            ],
        )

    @supported("python", version="2.10.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not have the stream helper method")
    def test_create_stream_method(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """Tests that the `stream` method is properly traced, and not just the `create` method with `stream=True`"""
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=True,
            stream_as_method=True,
        )

        traces = test_agent.wait_for_num_traces(num=1)
        span = traces[0][0]

        assert_apm_span(
            span=span,
            name="anthropic.request",
            resource="Messages.stream",
            tags=[
                ("anthropic.request.model", DEFAULT_MODEL),
            ],
        )


class TestAnthropicLlmObs:
    @supported("python", version="2.10.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name=DEFAULT_MODEL,
            model_provider="anthropic",
            input=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            output=[
                {
                    "role": "assistant",
                    "content": "2+2 = 4",
                }
            ],
            metadata=DEFAULT_PARAMETERS,
            metrics=EXPECTED_TOKENS,
        )

    @supported("python", version="2.10.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not have the stream helper method")
    def test_create_stream_method(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=True,
            stream_as_method=True,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name=DEFAULT_MODEL,
            model_provider="anthropic",
            input=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            output=[
                {
                    "role": "assistant",
                    "content": "2+2 = 4",
                }
            ],
            metadata=DEFAULT_PARAMETERS,
            metrics=EXPECTED_TOKENS,
        )

    @supported("python", version="2.10.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_content_block(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many ships are there in the fleet?",
                        }
                    ],
                },
            ],
            system="You are a helpful assistant who speaks like a pirate.",
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        if stream:
            expected_output = "Arr, me hearty! I don't have the specific information about which fleet ye be referrin' to. There be many fleets sailin' the seven seas!\n\nIf ye be askin' about a particular navy or merchant fleet, I'd need ye to specify which one ye mean. Could ye share more details about the fleet ye be inquirin' about, and I'll do me best to help ye with that information, savvy?"
        else:
            expected_output = "Arr, me hearty! I don't be havin' the exact count of ships in yer fleet, as ye haven't mentioned which fleet ye be referrin' to. \n\nThere be many fleets sailin' the seven seas - naval fleets, merchant fleets, fishin' fleets, and even pirate fleets (me favorite kind, yarr!).\n\nIf ye be wantin' to know about a specific fleet, just give me a"

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name=DEFAULT_MODEL,
            model_provider="anthropic",
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who speaks like a pirate.",
                },
                {
                    "role": "user",
                    "content": "How many ships are there in the fleet?",
                },
            ],
            output=[
                {
                    "role": "assistant",
                    "content": expected_output,
                }
            ],
            metadata=DEFAULT_PARAMETERS,
            metrics=EXPECTED_TOKENS,
        )

    @supported("python", version="2.10.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_error(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model="bad-model",
            messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
            raise_on_error=False,
        )
        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name="bad-model",
            model_provider="anthropic",
            input=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            output=[],
            metadata=DEFAULT_PARAMETERS,
            metrics={},
            status="error",
        )

    @supported("python", version="3.4.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_multiple_system_prompts(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            system=[
                {
                    "type": "text",
                    "text": "You are a helpful assistant who speaks like Yoda.",
                },
                {
                    "type": "text",
                    "text": "You also only speak in exactly 7 word sentences.",
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": "Explain in a few sentences what a pizza is.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "And, what ingredients do I need to make one?",
                        }
                    ],
                },
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        if stream:
            expected_output = "A delicious food, the pizza truly is.\n\nFlat dough with toppings, baked until done.\n\nCircular shape, it usually takes on.\n\nFor pizza, these ingredients you will need:\n\nFlour, water, yeast for the dough.\n\nTomato sauce spread across the base.\n\nCheese on top, melts when baked.\n\nToppings of choice, add them now.\n\nInto hot oven, the pizza"
        else:
            expected_output = "Round dough with toppings, a pizza is.\n\nDelicious meal enjoyed by many, it is.\n\nSauce, cheese, and toppings create perfect harmony.\n\nFor pizza dough, flour and yeast needed.\n\nWater, salt, olive oil complete the base.\n\nTomato sauce spread on dough, you must.\n\nCheese and favorite toppings added with care.\n\nIn hot oven, bake until golden brown."

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name=DEFAULT_MODEL,
            model_provider="anthropic",
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who speaks like Yoda.",
                },
                {
                    "role": "system",
                    "content": "You also only speak in exactly 7 word sentences.",
                },
                {
                    "role": "user",
                    "content": "Explain in a few sentences what a pizza is.",
                },
                {
                    "role": "user",
                    "content": "And, what ingredients do I need to make one?",
                },
            ],
            output=[
                {"role": "assistant", "content": expected_output},
            ],
            metadata=DEFAULT_PARAMETERS,
            metrics=EXPECTED_TOKENS,
        )

    @supported("python", version="2.11.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_with_tools(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "What is the weather in New York City?"},
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
            tools=TOOLS,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name=DEFAULT_MODEL,
            model_provider="anthropic",
            input=[
                {"role": "user", "content": "What is the weather in New York City?"},
            ],
            output=[
                {
                    "role": "assistant",
                    "content": "I can help you check the current weather in New York City. Let me get that information for you right away.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "arguments": {
                                "location": "New York City, NY",
                            },
                            "tool_id": mock.ANY,
                            "type": "tool_use",
                        }
                    ],
                },
            ],
            metadata=DEFAULT_PARAMETERS,
            metrics=EXPECTED_TOKENS,
        )

    @supported("python", version="3.12.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_tool_result(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "What is the weather in New York City?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I can help you check the current weather in New York City. Let me get that information for you right away.",
                        },
                        {
                            "type": "tool_use",
                            "name": "get_weather",
                            "input": {"location": "New York City"},
                            "id": "call_123",
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": json.dumps(
                                {
                                    "location": "New York City",
                                    "temperature": "70F",
                                    "description": "Sunny",
                                }
                            ),
                        }
                    ],
                },
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        if stream:
            expected_output = "The current weather in New York City is 70°F and sunny."
        else:
            expected_output = "Currently in New York City, it's 70°F and sunny."

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name=DEFAULT_MODEL,
            model_provider="anthropic",
            input=[
                {"content": "What is the weather in New York City?", "role": "user"},
                {
                    "content": "I can help you check the current weather in New York City. Let me get that information for you right away.",
                    "role": "assistant",
                },
                {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "arguments": {"location": "New York City"},
                            "tool_id": "call_123",
                            "type": "tool_use",
                        }
                    ],
                },
                {
                    "content": "",
                    "tool_results": [
                        {
                            "result": json.dumps(
                                {
                                    "location": "New York City",
                                    "temperature": "70F",
                                    "description": "Sunny",
                                }
                            ),
                            "tool_id": "call_123",
                            "type": "tool_result",
                        }
                    ],
                    "role": "user",
                },
            ],
            output=[
                {
                    "role": "assistant",
                    "content": expected_output,
                }
            ],
            metadata=DEFAULT_PARAMETERS,
            metrics=EXPECTED_TOKENS,
        )

    @supported("python", version="2.11.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_redact_image_input(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://tinyurl.com/yxbpd5p8",
                            },
                        },
                    ],
                }
            ],
            parameters=DEFAULT_PARAMETERS,
            stream=stream,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        if stream:
            expected_output = 'This is an image of an orange tabby cat relaxing on a wooden floor. The cat has distinctive orange and cream striped fur with white markings on its face. It appears to be resting peacefully with its eyes closed and is in the classic "loaf" position where its paws are tucked underneath its body. The cat has pointed ears and whiskers, and the warm lighting gives its fur a golden glow against the light wooden flooring.'
        else:
            expected_output = "This is an image of an orange tabby cat resting on a wooden floor. The cat has distinctive orange and cream striped fur with white markings, particularly around its face. It appears to be relaxed with its eyes closed, in a typical loaf position where its paws are tucked underneath its body. The cat has pointed ears and whiskers visible against its peaceful expression. The warm tones of the cat's fur contrast nicely with the cooler tones of the wooden"

        assert_llmobs_span_event(
            span_event,
            name="anthropic.request",
            span_kind="llm",
            model_name=DEFAULT_MODEL,
            model_provider="anthropic",
            input=[
                {"role": "user", "content": "What is this image?"},
                {"role": "user", "content": "([IMAGE DETECTED])"},
            ],
            output=[
                {
                    "role": "assistant",
                    "content": expected_output,
                }
            ],
            metadata=DEFAULT_PARAMETERS,
            metrics=EXPECTED_TOKENS,
        )

    @supported("python", version="3.11.0")
    @unsupported("nodejs", reason="nodejs does not support anthropic")
    @unsupported("java", reason="java does not support anthropic")
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_prompt_caching(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        stream: bool,
    ):
        parameters = DEFAULT_PARAMETERS.copy()
        parameters["extra_headers"] = {"anthropic-beta": "prompt-caching-2024-07-31"}

        large_system_prompt = [
            {
                "type": "text",
                "text": "Speak only like an annoyed Bostonian. "
                + "farewell" * (2 * 1024),
                "cache_control": {
                    "type": "ephemeral",
                },
            }
        ]

        #  write cached tokens
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            system=large_system_prompt,
            messages=[
                {"role": "user", "content": "Where is the nearest Dunkin Donuts?"}
            ],
            parameters=parameters,
            stream=stream,
        )

        #  read cached tokens
        test_client.anthropic_create(
            model=DEFAULT_MODEL,
            system=large_system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": "What is the best place to visit in Boston?",
                }
            ],
            parameters=parameters,
            stream=stream,
        )

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (write_span_event, read_span_event) = get_all_span_events(reqs)

        assert write_span_event["metrics"]["cache_write_input_tokens"] == 6156
        assert read_span_event["metrics"]["cache_read_input_tokens"] == 6156
