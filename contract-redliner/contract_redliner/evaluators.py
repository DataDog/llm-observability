"""Online Evaluators executed after the agent has run."""


def clauses_with_issues(output_data: dict) -> tuple[int, str, str]:
    """Count clauses with issues in the agent output.

    Returns (count, assessment, reasoning) for use with LLMObs.submit_evaluation().
    assessment: 'pass' if 0 issues, 'fail' otherwise.
    """
    proposals = (output_data or {}).get("proposals", [])
    count = len(proposals)
    assessment = "pass" if count == 0 else "fail"
    reasoning = f"Agent flagged {count} clause(s) with issues."
    return count, assessment, reasoning
