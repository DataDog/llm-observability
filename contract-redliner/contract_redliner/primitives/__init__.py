"""Primitives for the contract redliner agent.

This module contains core data models and policy definitions.
"""

from .models import (
    ContractDeps,
    DocumentSegment,
    Policy,
    ProposalResult,
    ValidationEdit,
    ValidationResult,
    RedlineResult,
)
from .policies import POLICY_DB, get_policies

__all__ = [
    "ContractDeps",
    "DocumentSegment",
    "Policy",
    "ProposalResult",
    "ValidationEdit",
    "ValidationResult",
    "RedlineResult",
    "POLICY_DB",
    "get_policies",
]
