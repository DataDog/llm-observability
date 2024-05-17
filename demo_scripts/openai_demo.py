import json
import openai
import requests

SEARCH_ENDPOINT = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MAX_RESULTS = 5
FETCH_MET_URLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_met_urls",
        "description": "Submits a query to the MET API and returns urls of relevant artworks",
        "parameters": {
            "type": "object",
            "properties": {
                "query_parameters": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Represents the users query. Required. Add as many search terms from the query as you can. 'medieval portraits', 'french impressionist paintings', etc.",
                        },
                        "title": {
                            "type": "boolean",
                            "description": "Limits the query to only apply to the title field.",
                        },
                        "tags": {
                            "type": "boolean",
                            "description": "Limits the query to only apply to the tags field.",
                        },
                        "isOnView": {
                            "type": "boolean",
                            "description": "Returns objects that match the query and are on view in the museum.",
                        },
                        "artistOrCulture": {
                            "type": "boolean",
                            "description": "Returns objects that match the query, specifically searching against the artist name or culture field for objects.",
                        },
                        "medium": {
                            "type": "string",
                            "description": 'Returns objects that match the query and are of the specified medium or object type. Examples include: "Ceramics", "Furniture", "Paintings", "Sculpture", "Textiles", etc.',
                        },
                        "geoLocation": {
                            "type": "string",
                            "description": 'Returns objects that match the query and the specified geographic location. Examples include: "Europe", "France", "Paris", "China", "New York", etc.',
                        },
                        "dateBegin": {
                            "type": "number",
                            "description": "You must use both dateBegin and dateEnd, or neither. Returns objects that match the query and fall between the dateBegin and dateEnd parameters. Examples include: dateBegin=1700&dateEnd=1800 for objects from 1700 A.D. to 1800 A.D., dateBegin=-100&dateEnd=100 for objects between 100 B.C. to 100 A.D.",
                        },
                        "dateEnd": {
                            "type": "number",
                            "description": "You must use both dateBegin and dateEnd, or neither. Returns objects that match the query and fall between the dateBegin and dateEnd parameters. Examples include: dateBegin=1700&dateEnd=1800 for objects from 1700 A.D. to 1800 A.D., dateBegin=-100&dateEnd=100 for objects between 100 B.C. to 100 A.D.",
                        },
                    },
                    "required": ["q"],
                },
            },
        },
    },
}


def openai_chat_completion():
    messages = [
        {"role": "system", "content": "Your task is to guess which Spongebob episode the given quote is from, and describe the context of the corresponding scene."},
        {"role": "user", "content": "Oh brother, this guy stinks!"},
    ]
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.9,
        max_tokens=50,
    )
    print(resp.choices[0].message.content)


def openai_chat_completion_stream():
    messages = [
        {"role": "system", "content": "Your task is to guess which Spongebob episode the given quote is from, and describe the context of the corresponding scene."},
        {"role": "user", "content": "How many times do I have to teach you this lesson, old man?"},
    ]
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.9,
        max_tokens=50,
        stream=True,
    )
    for chunk in resp:
        print(chunk.choices[0].delta.content)


def fetch_met_urls(query_parameters):
    response = requests.get(SEARCH_ENDPOINT, params=query_parameters)
    response.raise_for_status()
    object_ids = response.json().get("objectIDs")
    objects_to_return = object_ids[:MAX_RESULTS] if object_ids else []
    urls = [
        f"https://www.metmuseum.org/art/collection/search/{objectId}"
        for objectId in objects_to_return
    ]
    return urls


def parse_query(message):
    system_prompt = """
    Example query inputs and outputs for the fetch_met_urls function:

    query: medieval french tapestry painting
    output: {'q': 'medieval french tapestry painting', geoLocation: 'France', medium: 'Textiles', dateBegin: 1000, dateEnd: 1500}

    query: etruscan urns
    output: {'q': 'etruscan urn', geoLocation: 'Italy', medium: 'Travertine'}

    query: Cambodian hats from the 18th and 19th centuries
    output: {'q': 'Cambodian hats', geolocation: 'Cambodia', 'dateBegin': 1700, 'dateEnd': 1900}

    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response_message = (
        oai_client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            tools=[FETCH_MET_URLS_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "fetch_met_urls"}},
        )
        .choices[0]
        .message
    )
    arguments = {}
    if response_message.tool_calls:
        arguments = json.loads(response_message.tool_calls[0].function.arguments)
    print(arguments["query_parameters"])
    return arguments["query_parameters"]


def find_artworks(question):
    query = parse_query(question)
    print("Parsed query parameters", query)
    urls = fetch_met_urls(query)
    return urls

if __name__ == "__main__":
    openai_chat_completion()
    # openai_chat_completion_stream()
    # find_artworks("paintings of the french revolution")