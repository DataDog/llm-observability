from . import unsupported


@unsupported("nodejs", reason="nodejs doesn't implement the task method")
def test_trace_task(test_client, test_agent):
    span = test_client.sdk_task(
        name="test_task", session_id="test_id", ml_app="test_app"
    )
    test_client.finish_span(span["span_id"])
    traces = test_agent.wait_for_num_traces(num=1)
    assert traces[0][0]["name"] == "test_task"
    assert traces[0][0]["meta"]["_ml_obs.session_id"] == "test_id"
    assert traces[0][0]["meta"]["_ml_obs.meta.ml_app"] == "test_app"
