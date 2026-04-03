from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


@dataclass
class ContractDeps:
    """Runtime context passed to all agent tools via RunContext.deps."""
    doc_type: str
    clauses: list[str]


class DocumentSegment(BaseModel):
    """Output of the segmenter agent."""
    doc_type: Literal["nda", "saas", "employment", "vendor", "other"]
    clauses: list[str]


class Policy(BaseModel):
    topic: str
    rule: str
    severity: Literal["critical", "high", "medium", "low"]


class ProposalResult(BaseModel):
    clause_index: int
    risk_level: Literal["low", "medium", "high"]
    suggested_revision: str
    reasoning: str


class ValidationEdit(BaseModel):
    clause_index: int
    final_risk_level: Literal["low", "medium", "high"]
    approved: bool
    modifications: str | None
    final_revision: str
    validation_notes: str


class ValidationResult(BaseModel):
    edits: list[ValidationEdit]


class RedlineResult(BaseModel):
    """Final structured output returned by the redliner agent."""
    proposals: list[ProposalResult]
    risk_summary: dict[str, int]  # {"high": n, "medium": n, "low": n}
