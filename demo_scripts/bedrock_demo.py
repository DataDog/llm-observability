import boto3


def _get_bedrock_client():
    session = boto3.Session(profile_name="")
    client_params = {}
    return session.client("bedrock-runtime", **client_params)


def invoke_meta_model():
    boto3_bedrock = _get_bedrock_client()
    modelId = "meta.llama2-13b-chat-v1"
    accept = "application/json"
    contentType = "application/json"

    prompt_data = "Explain like I'm a five-year old: what does Datadog do?"
    body = json.dumps(
        {
            "prompt": prompt_data,
            "temperature": 0.9,
            "top_p": 1.0,
            "max_gen_len": 60,
        }
    )
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = response.get("body").read()
    print(response_body)


if __name__ == "__main__":
    invoke_meta_model()
