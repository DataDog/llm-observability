"""One Jev request judges one support reply on five atomic questions.

Notebook 1 builds this rubric from scratch and explains each part. Notebooks 2
and 3 import it, so the same rubric definition drives online evals and offline
experiments.

    export TYPESAFE_API_KEY=...
"""

from typesafe_sdk import Choice, Noul, NoulCriteria, Score, TypeSafeClient

# Pinned rather than jev-latest: the thresholds below were calibrated against this
# version, and an alias moves when a release ships.
JEV_MODEL = "jev-1.13.0"

GROUNDED_THRESHOLD = 0.70
ANSWERED_THRESHOLD = 0.60
HANDOFF_THRESHOLD = 0.60
CONFIDENCE_THRESHOLD = 0.50
IMPACT_THRESHOLD = 1.5

QUESTIONS = {
    "grounded": Noul(
        instructions={
            "question": (
                "Is every factual claim about Vega Air policy in `reply` stated in, or "
                "directly restated from, `policy_context`?"
            ),
            "inspect": "reply",
            "scope": [
                "Only policy claims count: fees, amounts, deadlines, weight limits, eligibility.",
                "Ignore greetings, apologies, and offers to hand off to a human agent.",
                "A reply that states no policy claims at all is grounded.",
            ],
        },
        criteria=NoulCriteria(
            true="Every policy claim in `reply` appears in `policy_context`.",
            false=(
                "At least one policy claim in `reply` is absent from `policy_context`, "
                "contradicts it, or changes a number, fee, or deadline."
            ),
        ),
    ),
    "answers_question": Noul(
        instructions={
            "question": "Does `reply` give the customer the specific information asked for in `question`?",
            "inspect": "reply",
            "scope": [
                "Judge whether the information was supplied, not whether it is correct.",
                "A reply that only promises to look into it has not supplied the information.",
            ],
        },
        criteria=NoulCriteria(
            true="`reply` states the fee, rule, amount, or answer that `question` asked for.",
            false="`reply` deflects, asks a counter-question, or only describes next steps.",
        ),
    ),
    "offers_handoff": Noul(
        instructions={
            "question": "Does `reply` offer to transfer the customer to a human agent or another team?",
            "inspect": "reply",
        },
        criteria=NoulCriteria(
            true="`reply` offers a human agent, a specialist team, or a support contact.",
            false="`reply` makes no such offer.",
        ),
    ),
    "failure_mode": Choice(
        instructions={
            "question": "What is the single biggest problem with `reply`?",
            "scope": (
                "Pick `none` when the reply is fine. Pick `unclear` only when the reply "
                "is too short or too garbled to judge."
            ),
        },
        criteria={
            "none": "The reply is accurate, on-policy, and useful.",
            "unsupported_claim": "The reply states a fee, rule, or number that is not in `policy_context`.",
            "missed_handoff": (
                "`policy_context` does not cover the question and the reply neither says so "
                "nor offers a human agent."
            ),
            "partial_answer": "The reply covers part of the question and silently drops the rest.",
            "unsafe_request": (
                "The reply complies with a request for personal data or something outside "
                "support scope."
            ),
            "unclear": "The reply is too short or too garbled to judge.",
        },
    ),
    "customer_impact": Score(
        instructions={
            "question": "If this reply were sent as-is, how badly would it affect the customer?",
            "scope": "Rate the consequence for the customer, not the tone of the reply.",
        },
        criteria=[
            "No harm. The customer gets what they need.",
            "Mild friction. The customer must ask again or look elsewhere.",
            "Real cost. The customer acts on wrong information or is stranded without a route forward.",
            "Serious harm. The customer loses money, misses travel, or their privacy is breached.",
        ],
    ),
}

# Offline only. Dataset rows carry a ground-truth label, so the experiment can ask
# Jev the same question the label answers and measure how often the two agree.
BEHAVIOR_QUESTION = {
    "behavior": Choice(
        instructions={
            "question": "Which of these best describes what `reply` actually did?",
            "inspect": "reply",
        },
        criteria={
            "answer": "It gave the customer the policy information they asked for.",
            "escalate": "It said the policy does not cover this and pointed to a human or another team.",
            "refuse": "It declined the request as out of scope or as something it must not provide.",
        },
    )
}


def make_client():
    return TypeSafeClient(model=JEV_MODEL)


def judge(client, question, policy_context, reply, extra_questions=None):
    """One request, five answers. Jev scores each question in parallel against one state."""
    return client.system_one(
        state={"question": question, "policy_context": policy_context, "reply": reply},
        questions={**QUESTIONS, **(extra_questions or {})},
    )


def verdict(result):
    """Flatten a Jev response into the numbers the Datadog eval metrics are built from.

    Jev does not do arithmetic, so the composite pass/fail is computed here in code.
    """
    grounded = result.nouls["grounded"].noul
    answered = result.nouls["answers_question"].noul
    handoff = result.nouls["offers_handoff"].noul
    failure = result.choices["failure_mode"]
    impact = result.scores["customer_impact"]

    # Handled correctly means grounded, and either answered or routed to a human.
    handled = grounded >= GROUNDED_THRESHOLD and (
        answered >= ANSWERED_THRESHOLD or handoff >= HANDOFF_THRESHOLD
    )

    return {
        "model": result.model,
        "grounded": grounded,
        "answers_question": answered,
        "offers_handoff": handoff,
        "failure_mode": failure.choice,
        "failure_mode_confidence": failure.confidence,
        "failure_mode_probabilities": failure.probabilities,
        "customer_impact": impact.score,
        "customer_impact_confidence": impact.confidence,
        "handled_correctly": handled,
        "input_tokens": result.usage.input_tokens,
    }


def _top_three(probabilities):
    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])[:3]
    return "; ".join(f"{name}={p:.2f}" for name, p in ranked)


def eval_metrics(v):
    """Map one verdict onto Datadog eval metrics.

    Probabilities are submitted as scores rather than collapsed to booleans. The
    threshold lives in `assessment`, so it can be re-cut later without re-running
    the judge over the whole backlog.
    """
    tags = {"judge": "typesafe-jev", "judge_model": v["model"]}
    return [
        {
            "label": "jev_grounded",
            "metric_type": "score",
            "value": round(v["grounded"], 4),
            "assessment": "pass" if v["grounded"] >= GROUNDED_THRESHOLD else "fail",
            "reasoning": f"P(grounded)={v['grounded']:.2f}, threshold={GROUNDED_THRESHOLD}",
            "tags": tags,
        },
        {
            "label": "jev_answers_question",
            "metric_type": "score",
            "value": round(v["answers_question"], 4),
            "assessment": "pass" if v["answers_question"] >= ANSWERED_THRESHOLD else "fail",
            "reasoning": f"P(answered)={v['answers_question']:.2f}, threshold={ANSWERED_THRESHOLD}",
            "tags": tags,
        },
        {
            "label": "jev_failure_mode",
            "metric_type": "categorical",
            "value": v["failure_mode"],
            "reasoning": _top_three(v["failure_mode_probabilities"]),
            "tags": tags,
        },
        {
            "label": "jev_failure_mode_confidence",
            "metric_type": "score",
            "value": round(v["failure_mode_confidence"], 4),
            "assessment": "pass" if v["failure_mode_confidence"] >= CONFIDENCE_THRESHOLD else "fail",
            "reasoning": "Below 0.5 the label is a close call between options; route to review.",
            "tags": tags,
        },
        {
            "label": "jev_customer_impact",
            "metric_type": "score",
            "value": round(v["customer_impact"], 4),
            "assessment": "pass" if v["customer_impact"] < IMPACT_THRESHOLD else "fail",
            "reasoning": f"0=no harm, 3=serious harm. confidence={v['customer_impact_confidence']:.2f}",
            "tags": tags,
        },
        {
            "label": "jev_handled_correctly",
            "metric_type": "boolean",
            "value": v["handled_correctly"],
            "assessment": "pass" if v["handled_correctly"] else "fail",
            "reasoning": (
                f"grounded={v['grounded']:.2f} answered={v['answers_question']:.2f} "
                f"handoff={v['offers_handoff']:.2f}"
            ),
            "tags": tags,
        },
    ]
