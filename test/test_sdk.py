from test import supported
from test import unsupported
import pytest

from test.conftest import LLMObsTestAgentClient
from test.client import InstrumentationClient as LLMObsInstrumentationClient

from typing import Dict, Union
from test.utils import COUNT
from test.utils import DIST
from test.utils import find_event_tag
from test.utils import get_all_span_events
from test.utils import get_evaluation_metrics
from test.utils import get_io_value_from_span_event
from test.utils import TestLLMObsSpan, TestApmSpan, TestAnnotation


# Define all span kinds and their required parameters
SPAN_KINDS = [
    {"kind": "task", "requires_model": False},
    {"kind": "llm", "requires_model": True},
    {"kind": "workflow", "requires_model": False},
    {"kind": "agent", "requires_model": False},
    {"kind": "retrieval", "requires_model": False},
    {"kind": "embedding", "requires_model": True},
    {"kind": "tool", "requires_model": False},
]

SIX_MB = 6 * 1024 * 1024


class TestEnablement:
    @unsupported("python", reason="python does not default ml_app")
    @unsupported("nodejs", reason="nodejs does not default ml_app")
    @unsupported("java", reason="java reports more spans than expected")
    @pytest.mark.parametrize("test_client_ml_app", ["test-ml-app", "", None])
    def test_ml_app(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        test_client_ml_app: str,
    ):
        llmobs_span = TestLLMObsSpan(kind="task")
        test_client.sdk_trace(llmobs_span)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        span_event_ml_app = find_event_tag(span_event, "ml_app")
        if test_client_ml_app:
            assert span_event_ml_app == test_client_ml_app
        else:
            assert span_event_ml_app == "test-service"  # defaults to service


class TestTracing:
    @unsupported(
        "java",
        reason="Java SDK does not send the _dd.tracer_version payload field",  # TODO: fix me
    )
    def test_assert_top_level_span_event_payload_fields(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that the top-level span event payload fields are present and correct
        """
        llmobs_span = TestLLMObsSpan(kind="task")

        test_client.sdk_trace(llmobs_span)
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        is_using_batched_requests = isinstance(reqs[0], list)

        required_fields = {
            "event_type": "span",
            "_dd.stage": "raw",
            "_dd.tracer_version": None,  # Just check existence
            "spans": None,  # Just check existence
        }

        span_payloads_to_check = (
            [span for req in reqs for span in req]
            if is_using_batched_requests
            else reqs
        )

        for span_payload in span_payloads_to_check:
            for field, expected_value in required_fields.items():
                assert field in span_payload, f"Missing required field: {field}"
                if expected_value is not None:
                    assert span_payload[field] == expected_value, (
                        f"Invalid value for {field}"
                    )

            if is_using_batched_requests:
                assert len(span_payload["spans"]) == 1, (
                    "Batched span must have exactly one span in the spans struct"
                )

    @pytest.mark.parametrize(
        "kind_info",
        SPAN_KINDS,
        ids=lambda span_kind_info: f"test_{span_kind_info['kind']}_span",
    )
    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_trace_spans(
        self,
        kind_info: Dict[str, Union[str, bool]],
        test_lang: str,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests the agent, workflow, task, tool, llm, embedding, and retrieval span
        types. Asserts that each span kind can be started and populated with the proper
        arguments on start.
        """
        if test_lang == "java" and kind_info["kind"] in ("embedding", "retrieval"):
            pytest.skip("Java SDK does not support embedding or retrieval spans")

        kind = kind_info["kind"]
        requires_model = kind_info["requires_model"]

        llmobs_span = TestLLMObsSpan(
            kind=kind,
            name=f"test_{kind}",
            session_id="test_id",
            ml_app="test_app",
        )

        if requires_model:
            llmobs_span.model_name = "test_model"
            llmobs_span.model_provider = "test_provider"

        test_client.sdk_trace(llmobs_span)
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        (span_event,) = get_all_span_events(reqs)

        if requires_model:
            assert span_event["meta"]["model_name"] == "test_model"
            assert span_event["meta"]["model_provider"] == "test_provider"

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_sets_parentage_simple(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that span parentage is propagated from a parent LLMObs span to
        a child LLMObs span.
        """
        trace_structure = TestLLMObsSpan(
            kind="workflow",
            name="test-workflow",
            children=[
                TestLLMObsSpan(
                    kind="task",
                    name="test-task",
                )
            ],
        )
        test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        workflow_span, task_span = get_all_span_events(reqs)

        assert workflow_span["name"] == "test-workflow"
        assert task_span["name"] == "test-task"
        assert workflow_span["parent_id"] == "undefined"
        assert task_span["parent_id"] == workflow_span["span_id"]

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_sets_parentage_apm_spans(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that parentage is properly maintained when an APM span is started
        in between LLMObs spans.
        """
        trace_structure = TestLLMObsSpan(
            kind="workflow",
            name="test-workflow",
            children=[
                TestApmSpan(
                    name="apm-span",
                    children=[
                        TestLLMObsSpan(
                            kind="task",
                            name="test-task",
                        )
                    ],
                )
            ],
        )
        test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        workflow_span, task_span = get_all_span_events(reqs)

        assert workflow_span["name"] == "test-workflow"
        assert task_span["name"] == "test-task"
        assert workflow_span["parent_id"] == "undefined"
        assert task_span["parent_id"] == workflow_span["span_id"]

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_sets_parentage_apm_root_span(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that LLMObs spans started underneath an APM root span both their parent
        IDs set to "undefined"
        """
        trace_structure = TestApmSpan(
            name="apm-root-span",
            children=[
                TestLLMObsSpan(
                    kind="workflow",
                    name="test-workflow",
                ),
                TestLLMObsSpan(
                    kind="workflow",
                    name="test-workflow-2",
                ),
            ],
        )
        test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        workflow_span, workflow_span_2 = get_all_span_events(reqs)

        assert workflow_span["name"] == "test-workflow"
        assert workflow_span_2["name"] == "test-workflow-2"
        assert workflow_span["parent_id"] == "undefined"
        assert workflow_span_2["parent_id"] == "undefined"

    @unsupported(
        "java", reason="Java SDK does not propagate ML apps to children"
    )  # TODO: fix me
    def test_ml_app_propagates_to_children(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that ML apps are propagated down through to child LLMObs spans
        """
        trace_structure = TestLLMObsSpan(
            kind="workflow",
            ml_app="overridden-ml-app",
            children=[
                TestLLMObsSpan(
                    kind="task",
                    ml_app="overridden-ml-app-2",
                ),
                TestLLMObsSpan(
                    kind="task",
                ),
            ],
        )
        test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        parent_workflow_span, child_task_span, child_task_span_2 = get_all_span_events(
            reqs
        )

        assert find_event_tag(parent_workflow_span, "ml_app") == "overridden-ml-app"
        assert find_event_tag(child_task_span, "ml_app") == "overridden-ml-app-2"
        assert find_event_tag(child_task_span_2, "ml_app") == "overridden-ml-app"

    @unsupported(
        "java", reason="Java SDK does not propagate session_id to children"
    )  # TODO: fix me
    def test_session_id_propagates_to_children(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that session IDs are propagated down through to child LLMObs spans
        """
        trace_structure = TestLLMObsSpan(
            kind="workflow",
            session_id="overridden-session-id",
            children=[
                TestLLMObsSpan(
                    kind="task",
                    session_id="overridden-session-id-2",
                ),
                TestLLMObsSpan(
                    kind="task",
                ),
            ],
        )
        test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        parent_workflow_span, child_task_span, child_task_span_2 = get_all_span_events(
            reqs
        )

        assert (
            find_event_tag(parent_workflow_span, "session_id")
            == "overridden-session-id"
        )
        assert (
            find_event_tag(child_task_span, "session_id") == "overridden-session-id-2"
        )
        assert (
            find_event_tag(child_task_span_2, "session_id") == "overridden-session-id"
        )

    @unsupported(
        "java",
        reason="Java SDK does not propagate session_id to children through APM spans",  # TODO: fix me
    )
    def test_session_id_propagates_through_apm_spans(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that ML apps are propagated down through to child LLMObs spans through
        intermixed APM spans
        """
        trace_structure = TestLLMObsSpan(
            kind="workflow",
            session_id="test-session-id",
            children=[
                TestApmSpan(
                    name="apm-span",
                    children=[
                        TestLLMObsSpan(
                            kind="task",
                        )
                    ],
                )
            ],
        )
        test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        test_workflow_span, test_task_span = get_all_span_events(reqs)

        assert find_event_tag(test_workflow_span, "session_id") == "test-session-id"
        assert find_event_tag(test_task_span, "session_id") == "test-session-id"


class TestAnnotate:
    @unsupported("java", reason="Java SDK annotations require a non-null span")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_annotate_no_span_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that annotating without an active LLMObs span throws an exception
        """
        trace_structure = TestApmSpan(
            name="apm-span",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                )
            ],
        )
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported("java", reason="Java SDK can only annotate LLMObs spans")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_annotate_non_llmobs_span_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that annotating when passed an APM span explicitly throws an exception
        """
        trace_structure = TestApmSpan(
            name="apm-span",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                    explicit_span=True,
                )
            ],
        )
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported(
        "java",
        reason="Java SDK does not throw an exception when trying to annotate a finished span",  # TODO: fix me
    )
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_annotate_finished_span_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that annotating a finished span throws an exception
        """
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test-task",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                )
            ],
            annotate_after=True,
        )
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_annotate_non_model_span(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that annotating a non-model spans formats input and output values
        correctly
        """
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test-task",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                    tags={"foo": "bar"},
                )
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert get_io_value_from_span_event(span_event, "input", "value") == "hello"
        assert get_io_value_from_span_event(span_event, "output", "value") == "world"
        assert find_event_tag(span_event, "foo") == "bar"

    @unsupported(
        "java", reason="Java SDK defaults empty role to 'unknown'"
    )  # TODO: fix me (maybe?)
    def test_annotate_llm_span_simple(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that annotating an LLM span with simple strings for input and output are
        properly formatted on the span's input and output messages
        """
        trace_structure = TestLLMObsSpan(
            kind="llm",
            name="test-llm",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                )
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        input_messages = get_io_value_from_span_event(span_event, "input", "messages")
        output_messages = get_io_value_from_span_event(span_event, "output", "messages")

        assert len(input_messages) == 1
        assert input_messages[0]["content"] == "hello"

        assert len(output_messages) == 1
        assert output_messages[0]["content"] == "world"

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_annotate_llm_span_with_messages(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that annotating LLM spans with message objects are properly formatted on
        the spans input and output messages
        """
        trace_structure = TestLLMObsSpan(
            kind="llm",
            name="test-llm",
            annotations=[
                TestAnnotation(
                    input_data=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "world"},
                        {"role": "user", "content": "foo"},
                    ],
                    output_data=[
                        {"role": "assistant", "content": "bar"},
                    ],
                )
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert get_io_value_from_span_event(span_event, "input", "messages") == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "foo"},
        ]
        assert get_io_value_from_span_event(span_event, "output", "messages") == [
            {"role": "assistant", "content": "bar"},
        ]

    @unsupported("java", reason="Java SDK has strict typing for messages")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_annotate_llm_span_with_malformed_messages_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that annotating an LLM span with malformed message content throws an exception
        """
        trace_structure = TestLLMObsSpan(
            kind="llm",
            name="test-llm",
            annotations=[TestAnnotation(input_data=[{"content": "hello", "role": 5}])],
        )
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported("java", reason="Java SDK does not support embedding spans")
    def test_annotate_embedding_span_simple(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that annotating an embedding span with string inputs and outputs
        formats input documents and output values correctly
        """
        trace_structure = TestLLMObsSpan(
            kind="embedding",
            name="test-embedding",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                )
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        (span_event,) = get_all_span_events(reqs)

        assert get_io_value_from_span_event(span_event, "input", "documents") == [
            {"text": "hello"}
        ]
        assert get_io_value_from_span_event(span_event, "output", "value") == "world"

    @unsupported("java", reason="Java SDK does not support embedding spans")
    def test_annotate_embedding_span_with_documents(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that annotating an embedding span with input documents and output value are
        properly formatted on the span's input documents and output value
        """
        trace_structure = TestLLMObsSpan(
            kind="embedding",
            name="test-embedding",
            annotations=[
                TestAnnotation(
                    input_data=[
                        {"text": "hello", "name": "foo", "id": "bar", "score": 0.678},
                        {"text": "world", "name": "baz", "score": 0.321},
                    ],
                    output_data="embedded 2 documents",
                )
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        (span_event,) = get_all_span_events(reqs)

        assert get_io_value_from_span_event(span_event, "input", "documents") == [
            {"text": "hello", "name": "foo", "id": "bar", "score": 0.678},
            {"text": "world", "name": "baz", "score": 0.321},
        ]
        assert (
            get_io_value_from_span_event(span_event, "output", "value")
            == "embedded 2 documents"
        )

    @unsupported("java", reason="Java SDK does not support embedding spans")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_annotate_embedding_span_with_malformed_documents_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that annotating an embedding span with malformed input documents throws
        an exception.
        """
        trace_structure = TestLLMObsSpan(
            kind="embedding",
            name="test-embedding",
            annotations=[TestAnnotation(input_data=[{"text": 5}])],
        )
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported("java", reason="Java SDK does not support retrieval spans")
    def test_annotate_retrieval_span_simple(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that annotating a retrieval span with string inputs and outputs
        formats input values and output documents correctly
        """
        trace_structure = TestLLMObsSpan(
            kind="retrieval",
            name="test-retrieval",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                )
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        (span_event,) = get_all_span_events(reqs)

        assert get_io_value_from_span_event(span_event, "input", "value") == "hello"
        assert get_io_value_from_span_event(span_event, "output", "documents") == [
            {"text": "world"}
        ]

    @unsupported("java", reason="Java SDK does not support retrieval spans")
    def test_annotate_retrieval_span_with_documents(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that annotating a retrieval span with input values and output documents are
        properly formatted on the span's input values and output documents
        """
        trace_structure = TestLLMObsSpan(
            kind="retrieval",
            name="test-retrieval",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data=[
                        {"text": "world", "name": "foo", "id": "bar", "score": 0.678},
                        {"text": "foo", "name": "baz", "score": 0.321},
                    ],
                )
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)

        (span_event,) = get_all_span_events(reqs)

        assert get_io_value_from_span_event(span_event, "input", "value") == "hello"
        assert get_io_value_from_span_event(span_event, "output", "documents") == [
            {"text": "world", "name": "foo", "id": "bar", "score": 0.678},
            {"text": "foo", "name": "baz", "score": 0.321},
        ]

    @unsupported("java", reason="Java SDK does not support retrieval spans")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_annotate_retrieval_span_with_malformed_documents_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that annotating a retrieval span with malformed output documents throws
        an exception
        """
        trace_structure = TestLLMObsSpan(
            kind="retrieval",
            name="test-retrieval",
            annotations=[TestAnnotation(output_data=[{"text": 5}])],
        )
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_annotate_merges_and_updates_metadata(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that subsequent annotations to metadata merge and update the span's
        metadata
        """
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test-task",
            annotations=[
                TestAnnotation(metadata={"foo": "bar"}),
                TestAnnotation(metadata={"foo": "quux", "bar": "baz"}),
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert span_event["meta"]["metadata"]["foo"] == "quux"
        assert span_event["meta"]["metadata"]["bar"] == "baz"

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_annotate_merges_and_updates_metrics(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that subsequent annotations to metrics merge and update the span's
        metrics
        """
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test-task",
            annotations=[
                TestAnnotation(metrics={"foo": 5}),
                TestAnnotation(metrics={"foo": 10, "bar": 20}),
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert span_event["metrics"]["foo"] == 10
        assert span_event["metrics"]["bar"] == 20

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    @unsupported(
        "nodejs",
        reason="Node.js SDK has a bug where tag values are not merged properly",
    )
    def test_annotate_merges_and_updates_tags(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that subsequent annotations to tags merge and update the span's
        tags
        """
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test-task",
            annotations=[
                TestAnnotation(tags={"foo": "bar"}),
                TestAnnotation(tags={"baz": "qux", "foo": "quux"}),
            ],
        )
        test_client.sdk_trace(trace_structure)
        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        assert find_event_tag(span_event, "foo") == "quux"
        assert find_event_tag(span_event, "baz") == "qux"


class TestExportSpan:
    @unsupported("java", reason="Java SDK does not support exporting spans")
    def test_export_span(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that the current span is exported with the correct span and trace ID
        from the inferred active span
        """
        trace_structure = TestLLMObsSpan(kind="task", export_span="implicit")
        exported_span_ctx = test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        exported_trace_id = exported_span_ctx.get(
            "trace_id", exported_span_ctx.get("traceId")
        )
        exported_span_id = exported_span_ctx.get(
            "span_id", exported_span_ctx.get("spanId")
        )

        assert exported_trace_id == span_event["trace_id"]
        assert exported_span_id == span_event["span_id"]

    @unsupported("java", reason="Java SDK does not support exporting spans")
    def test_export_span_explicit_span(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that the explicitly specified span is exported with the correct span
        and trace IDs
        """
        trace_structure = TestLLMObsSpan(kind="task", export_span="explicit")
        exported_span_ctx = test_client.sdk_trace(trace_structure)

        reqs = test_agent.wait_for_llmobs_requests(num=1)
        (span_event,) = get_all_span_events(reqs)

        exported_trace_id = exported_span_ctx.get(
            "trace_id", exported_span_ctx.get("traceId")
        )
        exported_span_id = exported_span_ctx.get(
            "span_id", exported_span_ctx.get("spanId")
        )

        assert span_event["trace_id"] == exported_trace_id
        assert span_event["span_id"] == exported_span_id

    @unsupported("java", reason="Java SDK does not support exporting spans")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_export_span_missing_span_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that exporting a non-existent LLMObs span throws
        """
        trace_structure = TestApmSpan(name="test-span", export_span="implicit")
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported("java", reason="Java SDK does not support exporting spans")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_export_span_is_not_llm_obs_span_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that exporting an APM span throws an exception
        """
        trace_structure = TestApmSpan(name="test-span", export_span="explicit")
        with pytest.raises(Exception):
            test_client.sdk_trace(trace_structure)

    @unsupported("java", reason="Java SDK does not support exporting spans")
    @unsupported("python", reason="Python SDK logs instead of raises")
    def test_export_span_is_not_span_throws(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that exporting a fake span (whose type is not a Span) throws
        """
        with pytest.raises(Exception):
            test_client.export_span_with_fake_span()


class TestEvaluations:
    @unsupported(
        "java",
        reason="Needs support for specifying trace_id and span_id separately from an entire span object",
    )
    def test_submit_evaluation_metric_simple_categorical(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that submitting a categorical evaluation metric populates the correct evaluation
        metric payload fields
        """
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            label="foo",
            metric_type="categorical",
            value="bar",
            tags={"baz": "qux"},
            ml_app="test_ml_app",
            timestamp_ms=1234567890,
        )

        reqs = test_agent.wait_for_llmobs_evaluations_requests(num=1)
        (evaluation_metric,) = get_evaluation_metrics(reqs)

        if evaluation_metric.get("join_on", None) is not None:
            assert evaluation_metric["join_on"]["span"]["trace_id"] == "123"
            assert evaluation_metric["join_on"]["span"]["span_id"] == "456"
        else:
            assert evaluation_metric["trace_id"] == "123"
            assert evaluation_metric["span_id"] == "456"

        assert evaluation_metric["label"] == "foo"
        assert evaluation_metric["metric_type"] == "categorical"
        assert evaluation_metric["categorical_value"] == "bar"
        assert evaluation_metric["ml_app"] == "test_ml_app"
        assert evaluation_metric["timestamp_ms"] == 1234567890
        assert evaluation_metric["ml_app"] == "test_ml_app"

        assert find_event_tag(evaluation_metric, "baz") == "qux"
        assert find_event_tag(evaluation_metric, "ddtrace.version") is not None

    @unsupported(
        "java",
        reason="Needs support for specifying trace_id and span_id separately from an entire span object",
    )
    def test_submit_evaluation_metric_simple_score(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that submitting a score evaluation metric populates the correct evaluation
        metric payload fields
        """
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            label="foo",
            metric_type="score",
            ml_app="test_ml_app",
            value=0.5,
        )

        reqs = test_agent.wait_for_llmobs_evaluations_requests(num=1)
        (evaluation_metric,) = get_evaluation_metrics(reqs)

        if evaluation_metric.get("join_on", None) is not None:
            assert evaluation_metric["join_on"]["span"]["trace_id"] == "123"
            assert evaluation_metric["join_on"]["span"]["span_id"] == "456"
        else:
            assert evaluation_metric["trace_id"] == "123"
            assert evaluation_metric["span_id"] == "456"

        assert evaluation_metric["label"] == "foo"
        assert evaluation_metric["metric_type"] == "score"
        assert evaluation_metric["score_value"] == 0.5
        assert evaluation_metric["ml_app"] == "test_ml_app"

        assert find_event_tag(evaluation_metric, "ddtrace.version") is not None

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_submit_evaluation_infers_ml_app(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that submitting an evaluation metric when there is a current active LLMObs
        span infers the ml_app tag to set
        """
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            label="foo",
            metric_type="score",
            value=0.5,
        )

        reqs = test_agent.wait_for_llmobs_evaluations_requests(num=1)
        (evaluation_metric,) = get_evaluation_metrics(reqs)

        assert (
            evaluation_metric["ml_app"] == "test-ml-app"
        )  # inferred from the global configuration

    @unsupported(
        "java",
        reason="Needs support for specifying trace_id and span_id separately from an entire span object",
    )
    def test_submit_evaluation_throws_for_missing_trace_id(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting an evaluation metric with a missing trace ID throws
        an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                span_id="456",
                label="foo",
                metric_type="score",
                value=0.5,
            )

    @unsupported(
        "java",
        reason="Needs support for specifying trace_id and span_id separately from an entire span object",
    )
    def test_submit_evaluation_throws_for_missing_span_id(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting an evaluation metric with a missing span ID throws
        an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                label="foo",
                metric_type="score",
                value=0.5,
            )

    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    def test_submit_evaluation_defaults_timestamp(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        """
        Tests that submitting an evaluation metric without a timestamp correctly defaults it
        """
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            label="foo",
            metric_type="score",
            value=0.5,
        )

        reqs = test_agent.wait_for_llmobs_evaluations_requests(num=1)
        (evaluation_metric,) = get_evaluation_metrics(reqs)

        assert evaluation_metric["timestamp_ms"] is not None

    @unsupported(
        "java", reason="Java SDK does not support setting timestamp manually"
    )  # TODO: fix me
    def test_submit_evaluation_throws_for_negative_timestamp(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting an evaluation metric with a negative timestamp
        throws an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                span_id="456",
                label="foo",
                metric_type="score",
                value=0.5,
                timestamp_ms=-1,
            )

    @unsupported(
        "java", reason="Java SDK does not support setting timestamp manually"
    )  # TODO: fix me
    def test_submit_evaluation_throws_for_non_number_timestamp(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting an evaluation metric with a non-number timestamp
        throws an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                span_id="456",
                label="foo",
                metric_type="score",
                value=0.5,
                timestamp_ms="not a number",
            )

    @unsupported(
        "java",
        reason="Java SDK does not throw an exception for missing label",  # TODO: fix me
    )
    def test_submit_evaluation_throws_for_missing_label(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting an evaluation metric with a missing label throws an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                span_id="456",
                metric_type="score",
                value=0.5,
            )

    @unsupported(
        "java",
        reason="Java SDK does not support setting metric types directly. It is inferred through the value type.",
    )
    def test_submit_evaluation_throws_for_bad_metric_type(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting an evaluation metric for an incorrect metric type (not
        oneof categorical,score) throws an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                span_id="456",
                label="foo",
                metric_type="bad",
                value=0.5,
            )

    @unsupported("java", reason="Java SDK will treat this as a score metric")
    def test_submit_evaluation_throws_for_categorical_with_non_string_value(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting a categorical evaluation metric with a non-string value
        throws an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                span_id="456",
                label="foo",
                metric_type="categorical",
                value=0.5,
            )

    @unsupported("java", reason="Java SDK will treat this as a categorical metric")
    def test_submit_evaluation_throws_for_score_with_non_number_value(
        self, test_agent, test_client: LLMObsInstrumentationClient
    ):
        """
        Tests that submitting a score evaluation metric with a non-number value
        throws an exception
        """
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                span_id="456",
                label="foo",
                metric_type="score",
                value="not a number",
            )

    @unsupported("java", reason="Needs support for joining on a key")
    @unsupported("nodejs", reason="Needs support for joining on a key")
    def test_submit_evaluation_metric_with_joining_key(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.submit_evaluation_metric(
            span_with_tag_value={
                "tag_key": "foo",
                "tag_value": "bar",
            },
            label="foo",
            metric_type="score",
            value=0.5,
        )

        reqs = test_agent.wait_for_llmobs_evaluations_requests(num=1)
        (evaluation_metric,) = get_evaluation_metrics(reqs)

        assert evaluation_metric["join_on"]["tag"]["key"] == "foo"
        assert evaluation_metric["join_on"]["tag"]["value"] == "bar"

    @unsupported("java", reason="Needs support for joining on a key")
    @unsupported("nodejs", reason="Needs support for joining on a key")
    def test_submit_evaluation_metric_with_joining_key_and_span_throws(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                trace_id="123",
                span_id="456",
                span_with_tag_value={
                    "tag_key": "foo",
                    "tag_value": "bar",
                },
            )

    @unsupported("java", reason="Needs support for joining on a key")
    @unsupported("nodejs", reason="Needs support for joining on a key")
    def test_submit_evaluation_metric_with_joining_key_without_tag_key_throws(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                span_with_tag_value={
                    "tag_value": "bar",
                },
            )

    @unsupported("java", reason="Needs support for joining on a key")
    @unsupported("nodejs", reason="Needs support for joining on a key")
    def test_submit_evaluation_metric_with_joining_key_without_tag_value_throws(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                span_with_tag_value={
                    "tag_key": "foo",
                },
            )

    @unsupported("java", reason="Needs support for joining on a key")
    @unsupported("nodejs", reason="Needs support for joining on a key")
    def test_submit_evaluation_metric_with_joining_key_non_string_tag_key_throws(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                span_with_tag_value={
                    "tag_key": 1,
                    "tag_value": "bar",
                },
            )

    @unsupported("java", reason="Needs support for joining on a key")
    @unsupported("nodejs", reason="Needs support for joining on a key")
    def test_submit_evaluation_metric_with_joining_key_non_string_tag_value_throws(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        with pytest.raises(Exception):
            test_client.submit_evaluation_metric(
                span_with_tag_value={
                    "tag_key": "foo",
                    "tag_value": 1,
                },
            )


class TestTelemetryMetrics:
    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_span_start(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(kind="task")
        test_client.sdk_trace(trace_structure)
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="span.start")
        assert metric["type"] == COUNT

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_span_finish(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(kind="task", session_id="test_id")
        test_client.sdk_trace(trace_structure)
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="span.finished")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"
        assert find_event_tag(metric, "autoinstrumented") == "0"
        assert find_event_tag(metric, "has_session_id") == "1"
        assert find_event_tag(metric, "is_root_span") == "1"
        assert find_event_tag(metric, "span_kind") == "task"
        assert find_event_tag(metric, "integration") == "n/a"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_span_raw_size(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(kind="task")
        test_client.sdk_trace(trace_structure)
        event, metric = test_agent.wait_for_telemetry_metric(
            metric_name="span.raw_size"
        )
        assert event["request_type"] == DIST
        assert find_event_tag(metric, "error") == "0"
        assert find_event_tag(metric, "autoinstrumented") == "0"
        assert find_event_tag(metric, "span_kind") == "task"
        assert find_event_tag(metric, "integration") == "n/a"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_span_size(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(kind="task")
        test_client.sdk_trace(trace_structure)
        event, metric = test_agent.wait_for_telemetry_metric(metric_name="span.size")
        assert event["request_type"] == DIST
        assert find_event_tag(metric, "error") == "0"
        assert find_event_tag(metric, "autoinstrumented") == "0"
        assert find_event_tag(metric, "truncated") == "0"
        assert find_event_tag(metric, "span_kind") == "task"
        assert find_event_tag(metric, "integration") == "n/a"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_span_size_truncated(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test_task",
            annotations=[
                TestAnnotation(input_data="A" * SIX_MB, output_data="A" * SIX_MB),
            ],
        )
        test_client.sdk_trace(trace_structure)
        event, metric = test_agent.wait_for_telemetry_metric(metric_name="span.size")
        assert event["request_type"] == DIST
        assert find_event_tag(metric, "error") == "0"
        assert find_event_tag(metric, "autoinstrumented") == "0"
        assert find_event_tag(metric, "truncated") == "1"
        assert find_event_tag(metric, "span_kind") == "task"
        assert find_event_tag(metric, "integration") == "n/a"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_annotate(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test_task",
            annotations=[TestAnnotation(metadata={"hello": "world"})],
        )
        test_client.sdk_trace(trace_structure)
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="annotations")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_annotate_no_span_err(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestApmSpan(
            name="apm-span",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                )
            ],
        )
        test_client.sdk_trace(trace_structure, raise_on_error=False)
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="annotations")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "1"
        assert find_event_tag(metric, "error_type") == "invalid_span_no_active_spans"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_annotate_span_finished_err(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(
            kind="task",
            name="test-task",
            annotations=[
                TestAnnotation(
                    input_data="hello",
                    output_data="world",
                )
            ],
            annotate_after=True,
        )
        test_client.sdk_trace(trace_structure, raise_on_error=False)
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="annotations")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "1"
        assert find_event_tag(metric, "error_type") == "invalid_finished_span"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_submit_eval(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            label="foo",
            metric_type="categorical",
            value="bar",
        )
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="evals_submitted")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_submit_eval_invalid_span_err(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.submit_evaluation_metric(
            trace_id="123",
            label="foo",
            metric_type="categorical",
            value="bar",
            raise_on_error=False,
        )
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="evals_submitted")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "1"
        assert find_event_tag(metric, "error_type") == "invalid_span"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_submit_eval_invalid_metric_label_err(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            raise_on_error=False,
        )
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="evals_submitted")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "1"
        assert find_event_tag(metric, "error_type") == "invalid_metric_label"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_submit_eval_invalid_metric_type_err(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            label="foo",
            metric_type="blah",
            raise_on_error=False,
        )
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="evals_submitted")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "1"
        assert find_event_tag(metric, "error_type") == "invalid_metric_type"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_submit_eval_invalid_metric_value_err(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            metric_type="categorical",
            label="foo",
            value=1,
            raise_on_error=False,
        )
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="evals_submitted")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "1"
        assert find_event_tag(metric, "error_type") == "invalid_metric_value"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_export_span(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestLLMObsSpan(kind="task", export_span="implicit")
        test_client.sdk_trace(trace_structure)
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="spans_exported")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"

    @supported("python", version="3.5.1")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_export_span_err(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        trace_structure = TestApmSpan(name="test-span", export_span="implicit")
        test_client.sdk_trace(trace_structure, raise_on_error=False)
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="spans_exported")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "1"
        assert find_event_tag(metric, "error_type") == "no_active_span"

    @supported("python", version="3.5.1")
    @unsupported(
        "nodejs", reason="does not have public methods for distributed tracing"
    )
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_inject_headers(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.inject_distributed_headers()
        _, metric = test_agent.wait_for_telemetry_metric(
            metric_name="inject_distributed_headers"
        )
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"

    @supported("python", version="3.5.1")
    @unsupported(
        "nodejs", reason="does not have public methods for distributed tracing"
    )
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_activate_headers(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.activate_distributed_headers()
        _, metric = test_agent.wait_for_telemetry_metric(
            metric_name="activate_distributed_headers"
        )
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"

    @unsupported("python", reason="incorrect metric name, will be corrected in 3.7.0")
    @supported("nodejs", version="5.50.0")
    @unsupported("java", reason="does not submit telemetry for API methods")
    def test_flush(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
    ):
        test_client.flush()
        _, metric = test_agent.wait_for_telemetry_metric(metric_name="user_flush")
        assert metric["type"] == COUNT
        assert find_event_tag(metric, "error") == "0"


class TestLLMObsConnectivity:
    @unsupported("java", reason="Java SDK is not officially released yet for CI")
    @supported("python", version="3.8.0")
    @supported("nodejs", version="5.52.0")
    @pytest.mark.parametrize("test_agent_connectivity_mode", ["tcp", "uds", "api"])
    def test_sends_span_events(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        test_agent_connectivity_mode: str,
    ):
        """
        This test verifies different connectivity modes for the LLMObs SDKs can send span events.
        For API connectivity, the test verifies that the SDK does not crash.
        """
        llmobs_span = TestLLMObsSpan(kind="task")
        test_client.sdk_trace(llmobs_span)

        # we cannot assert the request was made to the intake endpoint for api mode
        if test_agent_connectivity_mode != "api":
            reqs = test_agent.wait_for_llmobs_requests(num=1)
            span_events = get_all_span_events(reqs)
            assert len(span_events) == 1

    @unsupported(
        "java", reason="Java SDK does not send evaluation metrics via UDS"
    )  # TODO: fix me
    @supported("python", version="3.8.0")
    @supported("nodejs", version="5.52.0")
    @pytest.mark.parametrize("test_agent_connectivity_mode", ["tcp", "uds", "api"])
    def test_sends_evaluation_metrics(
        self,
        test_client: LLMObsInstrumentationClient,
        test_agent: LLMObsTestAgentClient,
        test_agent_connectivity_mode: str,
    ):
        """
        This test verifies different connectivity modes for the LLMObs SDKs can send evaluation metrics.
        For API connectivity, the test verifies that the SDK does not crash.
        """
        test_client.submit_evaluation_metric(
            trace_id="123",
            span_id="456",
            label="foo",
            metric_type="categorical",
            value="bar",
        )

        if test_agent_connectivity_mode != "api":
            reqs = test_agent.wait_for_llmobs_evaluations_requests(num=1)
            evaluation_metrics = get_evaluation_metrics(reqs)
            assert len(evaluation_metrics) == 1
