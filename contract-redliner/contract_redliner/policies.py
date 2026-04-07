POLICY_DB: dict[str, dict[str, str]] = {
    "nda": {
        "confidentiality_scope": "Definition of Confidential Information must explicitly exclude: (a) public domain information, (b) independently developed information, (c) information received from a third party without restriction, (d) information required to be disclosed by law.",
        "term_survival": "Post-termination confidentiality survival must not exceed 3 years for general confidential information. Trade secrets may survive longer only if explicitly identified as such.",
        "return_destruction": "Receiving party must have the option to destroy (not only return) Confidential Information and must provide written certification of destruction upon request.",
        "injunctive_relief": "Injunctive relief clauses must be mutual and must not waive the right to a jury trial.",
    },
    "saas": {
        "sla_uptime": "Minimum 99.5% monthly uptime SLA is required. Downtime credits must be defined: at minimum 10% credit for each full hour of excess downtime.",
        "data_processing": "A Data Processing Addendum (DPA) compliant with GDPR/CCPA is required whenever Customer data includes EU or California resident personal data.",
        "liability_cap": "Provider's aggregate liability must be capped at no more than 12 months of fees paid by Customer in the 12 months preceding the claim.",
        "ip_ownership": "Platform IP and all improvements to the platform remain exclusively with Provider. Customer retains full ownership of Customer Data.",
        "termination": "Termination for convenience requires a minimum 30 days written notice. Immediate termination is only permitted for material uncured breach.",
    },
    "employment": {
        "at_will": "At-will employment clause is required for US-based employees. The clause must state that either party may terminate the relationship at any time for any lawful reason.",
        "non_compete": "Non-compete restrictions must not exceed 12 months post-termination and must be limited to a 50-mile geographic radius or specific named competitors.",
        "severance": "Severance policy must provide a minimum of 2 weeks' base salary per year of service, subject to a signed release of claims.",
        "ip_assignment": "Employee must assign all work-product, inventions, and IP created during the course of employment to the company, excluding pre-existing IP listed in an exhibit.",
    },
    "vendor": {
        "payment_terms": "Payment terms must be Net 30 from the invoice date. Late payment interest must not exceed 1.5% per month.",
        "indemnification": "Indemnification obligations must be mutual. Unilateral indemnification (vendor-only) is not acceptable without corresponding liability protections.",
        "termination_notice": "Termination for convenience requires a minimum 30 days written notice. Auto-renewal clauses must include a 60-day opt-out window.",
        "governing_law": "Governing law must be the State of Delaware or the counterparty's primary state of incorporation. Arbitration clauses must specify JAMS or AAA as the arbitration body.",
    },
}

def policy_index() -> str:
    """Get index of available policies."""
    lines = []
    for topic, policies in POLICY_DB.items():
        lines.append(f"topic: {topic}")
        for policy in policies:
            lines.append(f" - policy: {policy}")
    return "\n".join(lines)