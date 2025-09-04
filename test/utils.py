from typing import Literal
from typing import Optional
from typing import Union
from typing import Dict
from typing import Tuple
from typing import Any
from typing import List
from pydantic import BaseModel


COUNT = "count"
DIST = "distributions"


LLMObsRequests = List[List[Dict[str, Any]]]


def assert_llmobs_span_event(
    span_event: dict,
    name: str,
    span_kind: Literal[
        "llm", "agent", "workflow", "task", "tool", "embedding", "retrieval"
    ],
    input: Union[list, str],
    output: Union[list, str] = None,
    model_name: Optional[str] = None,
    model_provider: Optional[str] = None,
    metadata: Optional[dict] = None,
    metrics: Optional[Union[Dict[str, int], Dict[str, float]]] = None,
    parent_id: Optional[str] = "undefined",
    status: Optional[str] = None,
):
    assert span_event["name"].lower() == name.lower(), f"{span_event['name']} != {name}"
    assert span_event["meta"]["span.kind"] == span_kind, (
        f"{span_event['meta']['span.kind']} != {span_kind}"
    )
    assert span_event["parent_id"] == parent_id, (
        f"{span_event['parent_id']} != {parent_id}"
    )

    if span_kind in ("llm", "embedding"):
        assert model_name in span_event["meta"]["model_name"], (
            f"{span_event['meta']['model_name']} != {model_name}"
        )
        assert span_event["meta"]["model_provider"] == model_provider, (
            f"{span_event['meta']['model_provider']} != {model_provider}"
        )
        if span_kind == "llm":
            if not isinstance(input, list):
                raise ValueError(
                    f"Input list to test against must be a list, got {type(input)}"
                )
            for i, message in enumerate(input):
                input_message = span_event["meta"]["input"]["messages"][i]
                if "content" not in input_message:
                    assert "content" not in message or message["content"] is None, (
                        "No content in actual input message"
                    )
                else:
                    assert (
                        span_event["meta"]["input"]["messages"][i]["content"]
                        == message["content"]
                    ), (
                        f"{span_event['meta']['input']['messages'][i]['content']} != {message['content']}"
                    )

                span_event_role = span_event["meta"]["input"]["messages"][i].get("role")
                if "role" in message or (
                    span_event_role is not None and span_event_role != ""
                ):
                    assert (
                        span_event["meta"]["input"]["messages"][i]["role"]
                        == message["role"]
                    ), (
                        f"{span_event['meta']['input']['messages'][i]['role']} != {message['role']}"
                    )

                tool_calls = message.get("tool_calls")
                if tool_calls:
                    assert tool_calls == input_message.get("tool_calls"), (
                        f"{tool_calls} != {input_message.get('tool_calls')}"
                    )

                tool_results = message.get("tool_results")
                if tool_results:
                    assert tool_results == input_message.get("tool_results"), (
                        f"{tool_results} != {input_message.get('tool_results')}"
                    )

            if not isinstance(output, list):
                raise ValueError(
                    f"Output list to test against must be a list, got {type(output)}"
                )
            for i, message in enumerate(output):
                output_message = span_event["meta"]["output"]["messages"][i]
                if "content" not in output_message:
                    assert "content" not in message or message["content"] is None, (
                        "No content in actual output message"
                    )
                else:
                    assert (
                        span_event["meta"]["output"]["messages"][i]["content"]
                        == message["content"]
                    ), (
                        f"{span_event['meta']['output']['messages'][i]['content']} != {message['content']}"
                    )

                span_event_role = span_event["meta"]["output"]["messages"][i].get(
                    "role"
                )
                if "role" in message or (
                    span_event_role is not None and span_event_role != ""
                ):
                    assert (
                        span_event["meta"]["output"]["messages"][i]["role"]
                        == message["role"]
                    ), (
                        f"{span_event['meta']['output']['messages'][i]['role']} != {message['role']}"
                    )

                tool_calls = message.get("tool_calls")
                if tool_calls:
                    assert tool_calls == output_message.get("tool_calls"), (
                        f"{tool_calls} != {output_message.get('tool_calls')}"
                    )

                tool_results = message.get("tool_results")
                if tool_results:
                    assert tool_results == output_message.get("tool_results"), (
                        f"{tool_results} != {output_message.get('tool_results')}"
                    )

        elif span_kind == "embedding":
            assert span_event["meta"]["input"]["documents"] == input, (
                f"{span_event['meta']['input']['documents']} != {input}"
            )
            assert span_event["meta"]["output"]["value"] == output, (
                f"{span_event['meta']['output']['value']} != {output}"
            )
    elif span_kind == "retrieval":
        assert span_event["meta"]["input"]["value"] == input, (
            f"{span_event['meta']['input']['value']} != {input}"
        )
        assert span_event["meta"]["output"]["documents"] == output, (
            f"{span_event['meta']['output']['documents']} != {output}"
        )
    else:
        assert span_event["meta"]["input"]["value"] == input, (
            f"{span_event['meta']['input']['value']} != {input}"
        )
        assert span_event["meta"]["output"]["value"] == output, (
            f"{span_event['meta']['output']['value']} != {output}"
        )

    if metadata:
        assert span_event["meta"]["metadata"] == metadata, (
            f"{span_event['meta']['metadata']} != {metadata}"
        )

    if metrics:
        assert span_event["metrics"] == metrics, f"{span_event['metrics']} != {metrics}"

    if status:
        assert span_event["status"] == status


AcceptedTags = Optional[List[Union[Tuple[str, Any], List[Tuple[str, Any]]]]]


def assert_apm_span(
    span: dict,
    name: str,
    resource: Tuple[str],
    parent_id: Optional[str] = 0,
    tags: AcceptedTags = [],
):
    assert span["name"] == name, f"Expected {name}, got {span['name']}"
    assert span["resource"] in resource, (
        f"Expected {span['resource']} to be in {resource}"
    )  # in the case of resource name differences
    assert span["parent_id"] == parent_id, (
        f"Expected {parent_id}, got {span['parent_id']}"
    )

    for tag in tags:
        if isinstance(tag, list):
            matches = [_tag_matches(span, t) for t in tag]
            any_match = any(match[0] for match in matches)
            assert any_match, f"Mismatched tags: {[match[1] for match in matches]}"
        else:
            assert _tag_matches(span, tag)


def _tag_matches(span, tag) -> Tuple[bool, str]:
    k, v = tag
    maybe_meta = span["meta"].get(k)
    maybe_metric = span["metrics"].get(k)

    if not (maybe_meta or maybe_metric):
        return False, f"Expected {k} to be present in span meta or metrics"

    if isinstance(v, list):
        results = []
        for val in v:
            try:
                results.append(
                    (maybe_meta == val or maybe_meta == str(val))
                    or (maybe_metric == val or maybe_metric == float(val))
                )
            except:  # noqa E722
                results.append(False)
        return any(
            results
        ), f"Expected {k} to be one of {v}, got {maybe_meta} and {maybe_metric}"

    return (
        (maybe_meta == v or maybe_meta == str(v))
        or (maybe_metric == v or maybe_metric == float(v))
    ), f"Expected {k} to be {v}, got {maybe_meta} and {maybe_metric}"


def find_event_tag(event, tag):
    """Find a tag in a span event or telemetry metric event."""
    tags = event["tags"]
    for t in tags:
        k, v = t.split(":")
        if k == tag:
            return v


def get_all_span_events(reqs: LLMObsRequests) -> List[Dict[str, Any]]:
    """Returns all of the span events from the requests, flattened out and sorted by start time"""
    using_batched_requests = isinstance(reqs[0], list)
    if using_batched_requests:
        events = [span for req in reqs for span in req]  # flatten
        return sorted([e["spans"][0] for e in events], key=lambda x: x["start_ns"])
    else:
        events = [span for req in reqs for span in req["spans"]]
        return sorted(events, key=lambda x: x["start_ns"])


def get_evaluation_metrics(reqs: LLMObsRequests) -> Dict[str, Any]:
    """Returns the all evaluation metrics from the requests, flattened out and sorted by timestamp"""
    return sorted(
        [
            eval_metric
            for req in reqs
            for eval_metric in req["data"]["attributes"]["metrics"]
        ],
        key=lambda x: x["timestamp_ms"],
    )


def get_io_value_from_span_event(
    span_event: dict,
    kind: Literal["input", "output"],
    property: Literal["messages", "documents", "value"],
) -> List[Dict[str, Any]]:
    meta = span_event["meta"]
    from_kind_directly = kind in meta

    return meta[kind][property] if from_kind_directly else meta[f"{kind}.{property}"]


class TestAnnotation(BaseModel):
    __test__ = False

    input_data: Optional[Union[dict, str, List[Union[dict, str]]]] = None
    output_data: Optional[Union[dict, str, List[Union[dict, str]]]] = None
    metadata: Optional[dict] = None
    metrics: Optional[dict] = None
    tags: Optional[dict] = None

    explicit_span: Optional[bool] = False


class TestSpan(BaseModel):
    __test__ = False

    sdk: Literal["llmobs", "tracer"]
    name: Optional[str] = None
    children: Optional[
        Union[
            List[Union["TestApmSpan", "TestLLMObsSpan"]],
            List[List[Union["TestApmSpan", "TestLLMObsSpan"]]],
        ]
    ] = None

    # These fields are specifically for the LLMObs SDK but can be applied on ApmSpans as well
    # to test erroring behavior of the LLMObs.annotate and LLMObs.export_span methods
    annotations: Optional[List["TestAnnotation"]] = None
    annotate_after: Optional[bool] = None
    export_span: Optional[Literal["explicit", "implicit"]] = None


class TestApmSpan(TestSpan):
    __test__ = False

    sdk: Literal["tracer"] = "tracer"
    name: str


class TestLLMObsSpan(TestSpan):
    __test__ = False

    sdk: Literal["llmobs"] = "llmobs"
    kind: Literal["llm", "agent", "workflow", "task", "tool", "embedding", "retrieval"]
    session_id: Optional[str] = None
    ml_app: Optional[str] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
