from contract_redliner.primitives.models import Policy

POLICY_DB: dict[str, list[Policy]] = {
    "nda": [
        Policy(
            topic="confidentiality_scope",
            rule="Definition of Confidential Information must explicitly exclude: (a) public domain information, (b) independently developed information, (c) information received from a third party without restriction, (d) information required to be disclosed by law.",
            severity="critical",
        ),
        Policy(
            topic="term_survival",
            rule="Post-termination confidentiality survival must not exceed 3 years for general confidential information. Trade secrets may survive longer only if explicitly identified as such.",
            severity="high",
        ),
        Policy(
            topic="return_destruction",
            rule="Receiving party must have the option to destroy (not only return) Confidential Information and must provide written certification of destruction upon request.",
            severity="medium",
        ),
        Policy(
            topic="injunctive_relief",
            rule="Injunctive relief clauses must be mutual and must not waive the right to a jury trial.",
            severity="low",
        ),
    ],
    "saas": [
        Policy(
            topic="sla_uptime",
            rule="Minimum 99.5% monthly uptime SLA is required. Downtime credits must be defined: at minimum 10% credit for each full hour of excess downtime.",
            severity="critical",
        ),
        Policy(
            topic="data_processing",
            rule="A Data Processing Addendum (DPA) compliant with GDPR/CCPA is required whenever Customer data includes EU or California resident personal data.",
            severity="critical",
        ),
        Policy(
            topic="liability_cap",
            rule="Provider's aggregate liability must be capped at no more than 12 months of fees paid by Customer in the 12 months preceding the claim.",
            severity="high",
        ),
        Policy(
            topic="ip_ownership",
            rule="Platform IP and all improvements to the platform remain exclusively with Provider. Customer retains full ownership of Customer Data.",
            severity="high",
        ),
        Policy(
            topic="termination",
            rule="Termination for convenience requires a minimum 30 days written notice. Immediate termination is only permitted for material uncured breach.",
            severity="medium",
        ),
    ],
    "employment": [
        Policy(
            topic="at_will",
            rule="At-will employment clause is required for US-based employees. The clause must state that either party may terminate the relationship at any time for any lawful reason.",
            severity="critical",
        ),
        Policy(
            topic="non_compete",
            rule="Non-compete restrictions must not exceed 12 months post-termination and must be limited to a 50-mile geographic radius or specific named competitors.",
            severity="high",
        ),
        Policy(
            topic="severance",
            rule="Severance policy must provide a minimum of 2 weeks' base salary per year of service, subject to a signed release of claims.",
            severity="medium",
        ),
        Policy(
            topic="ip_assignment",
            rule="Employee must assign all work-product, inventions, and IP created during the course of employment to the company, excluding pre-existing IP listed in an exhibit.",
            severity="high",
        ),
    ],
    "vendor": [
        Policy(
            topic="payment_terms",
            rule="Payment terms must be Net 30 from the invoice date. Late payment interest must not exceed 1.5% per month.",
            severity="medium",
        ),
        Policy(
            topic="indemnification",
            rule="Indemnification obligations must be mutual. Unilateral indemnification (vendor-only) is not acceptable without corresponding liability protections.",
            severity="high",
        ),
        Policy(
            topic="termination_notice",
            rule="Termination for convenience requires a minimum 30 days written notice. Auto-renewal clauses must include a 60-day opt-out window.",
            severity="medium",
        ),
        Policy(
            topic="governing_law",
            rule="Governing law must be the State of Delaware or the counterparty's primary state of incorporation. Arbitration clauses must specify JAMS or AAA as the arbitration body.",
            severity="low",
        ),
    ],
}


def get_policies(doc_type: str, topic_hint: str = "") -> list[Policy]:
    """Return policies for doc_type, filtered by keyword overlap with topic_hint.

    Falls back to all policies for the doc type if no keyword match is found.
    Falls back to vendor policies if doc_type is unrecognised.
    """
    base = POLICY_DB.get(doc_type, POLICY_DB["vendor"])
    if not topic_hint:
        return base
    words = set(topic_hint.lower().split())
    filtered = [
        p for p in base
        if words & set(p.topic.replace("_", " ").split())
    ]
    return filtered or base
