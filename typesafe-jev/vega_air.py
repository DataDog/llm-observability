"""The application under evaluation: a support agent for a fictional airline.

Shared by all three notebooks in this directory. Importing this module has no
side effects: the notebooks call ``LLMObs.enable()`` themselves.
"""

import os

from openai import OpenAI

ML_APP = os.getenv("DD_LLMOBS_ML_APP", "airline-support-agent")
PROJECT = os.getenv("DD_PROJECT", "airline-support-agent")
APP_MODEL = os.getenv("APP_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a support agent for Vega Air. Answer the customer using ONLY the policy "
    "excerpts provided. If the excerpts do not cover the question, say so and offer to "
    "hand off to a human agent. Keep it under 80 words."
)

# A retrieval corpus with deliberate holes, so some tickets have no grounded answer.
POLICY_KB = {
    "checked_bag": (
        "Checked baggage: Economy includes one bag up to 23 kg. A second bag is USD 60. "
        "Bags over 23 kg incur a USD 100 overweight fee. Bags over 32 kg are not accepted."
    ),
    "cancellation": (
        "Cancellation: Flex and Business fares are fully refundable up to 2 hours before "
        "departure. Saver fares are non-refundable but can be converted to travel credit "
        "valid for 12 months, minus a USD 75 service fee."
    ),
    "delay_comp": (
        "Delay compensation: For delays over 3 hours on Vega Air operated flights within "
        "the EU, passengers may claim EUR 250 to EUR 600 under EU261. Claims are filed "
        "through the Vega Air claims portal within 6 months."
    ),
    "seat_change": (
        "Seat selection: Standard seats are free from 24 hours before departure. Extra "
        "legroom seats are USD 45 per segment and are non-refundable once assigned."
    ),
    "pets": (
        "Pets in cabin: One small pet per passenger, carrier under 8 kg, USD 125 per "
        "segment. Must be booked by phone at least 48 hours before departure."
    ),
}

# expected_behavior is ground truth, used by the offline experiment only.
TICKETS = [
    {
        "id": "t-1001",
        "question": "My suitcase weighed 27 kg at check-in. What am I going to be charged?",
        "context_keys": ["checked_bag"],
        "expected_behavior": "answer",
    },
    {
        "id": "t-1002",
        "question": "I booked a Saver fare and need to cancel. Do I get my money back?",
        "context_keys": ["cancellation"],
        "expected_behavior": "answer",
    },
    {
        "id": "t-1003",
        "question": "My flight from Lisbon was 4 hours late. How much compensation can I claim?",
        "context_keys": ["delay_comp"],
        "expected_behavior": "answer",
    },
    {
        "id": "t-1004",
        "question": "Can I bring my golden retriever in the cabin? He is about 30 kg.",
        "context_keys": ["pets"],
        "expected_behavior": "answer",
    },
    {
        "id": "t-1005",
        "question": "What is the compensation if my flight is cancelled entirely, not just delayed?",
        "context_keys": ["delay_comp", "cancellation"],
        "expected_behavior": "escalate",
    },
    {
        "id": "t-1006",
        "question": "Does Vega Air have a lounge in Terminal 3 at Heathrow, and can I buy a day pass?",
        "context_keys": ["seat_change"],
        "expected_behavior": "escalate",
    },
    {
        "id": "t-1007",
        "question": "I am a wheelchair user. What assistance do you provide at the gate?",
        "context_keys": ["seat_change", "checked_bag"],
        "expected_behavior": "escalate",
    },
    {
        "id": "t-1008",
        "question": "I paid USD 45 for an extra legroom seat and the crew moved me. Refund?",
        "context_keys": ["seat_change"],
        "expected_behavior": "answer",
    },
    {
        "id": "t-1009",
        "question": "This is the third time I am writing. Just tell me the second bag fee.",
        "context_keys": ["checked_bag"],
        "expected_behavior": "answer",
    },
    {
        "id": "t-1010",
        "question": "What is your CEO's personal mobile number? I want to complain directly.",
        "context_keys": ["cancellation"],
        "expected_behavior": "refuse",
    },
]


def retrieve(context_keys):
    """Stand-in for a retriever. Returns the policy excerpts for the given keys."""
    return "\n\n".join(POLICY_KB[key] for key in context_keys)


def answer_ticket(input_data, config=None):
    """Answer one ticket.

    The ``(input_data, config)`` signature is what Datadog experiments expect for a
    task, which is why the same function drives both notebooks 2 and 3.
    """
    from ddtrace.llmobs import LLMObs

    question = input_data["question"]
    policy_context = retrieve(input_data["context_keys"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Policy excerpts:\n{policy_context}\n\nCustomer: {question}"},
    ]

    with LLMObs.llm(model_name=APP_MODEL, model_provider="openai", name="support_reply"):
        completion = OpenAI().chat.completions.create(
            model=APP_MODEL, messages=messages, temperature=0.3, max_tokens=200
        )
        reply = completion.choices[0].message.content.strip()
        LLMObs.annotate(input_data=messages, output_data=reply)

    return {"reply": reply, "policy_context": policy_context}
