from . import supported


@supported("python", version="1.14.0")
@supported("nodejs", version="4.4.0")
def test_chat_completion(test_lang, test_client, test_agent):
    test_client.openai_chat_completion(prompt="Why is Evan Li such a slacker?")
    traces = test_agent.wait_for_num_traces(num=1)
    assert (
        traces[0][0]["meta"]["openai.request.messages.0.content"]
        == "Why is Evan Li such a slacker?"
    ), traces
    assert traces[0][0]["name"] == "openai.request"

    if test_lang != "nodejs":
        reqs = test_agent.llmobs_requests()
        assert len(reqs) == 1
