import langchain
import langchain_core
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

# Requires OpenAI API key to be set in the environment variable $OPENAI_API_KEY


def invoke_llm():
    llm = OpenAI(temperature=0)
    ans = llm("Can you tell me a joke with puns about bees?")
    print(ans)


def invoke_chat_model():
    chat = ChatOpenAI(temperature=0, max_tokens=256)
    messages = chat(
        [
            SystemMessage(content="Respond like a frat boy."),
            HumanMessage(content="What's the fastest way to get to Boston from New York City?"),
        ]
    )
    print(messages)


def invoke_chain():
    prompt1 = langchain_core.prompts.ChatPromptTemplate.from_template("what is the city {person} is from? Respond with the name of the city only.")
    prompt2 = langchain_core.prompts.ChatPromptTemplate.from_template("what country is the city {city} in? Respond in {language}")

    model = langchain_openai.ChatOpenAI(temperature=0, max_tokens=256)
    chain1 = prompt1 | model | langchain_core.output_parsers.StrOutputParser()
    chain2 = prompt2 | model | langchain_core.output_parsers.StrOutputParser()

    complete_chain = {"city": chain1, "language": itemgetter("language")} | chain2

    complete_chain.invoke({"person": "Spongebob Squarepants", "language": "Spanish"})


def submit_evaluation():
    evaluations = {
        "user_satisfaction": 4,
        "profanity": 1,
        "topic": "cartoons"
    }
    return evaluations


if __name__ == "__main__":
    invoke_llm()
    # invoke_chat_model()
    # invoke_chain()
