from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]

@dataclass(slots=True)
class GenResult:
    code: str
    description: str
    tests: str = ""  # pytest-style or __main__ demo calls that actually exercise `code`
    assumptions: list[str] = field(default_factory=list)
    clarification_needed: str | None = None

@dataclass(slots=True)
class CheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    confidence: Confidence = "low"
    error: str | None = None
    notes: list[str] = field(default_factory=list)

@dataclass(slots=True)
class AgentResult:
    code: str
    description: str
    assumptions: list[str]
    passed: bool
    confidence: Confidence
    issues: list[str]
    retries_taken: int
    tests: str = ""
    clarification_needed: str | None = None
    pipeline_notes: list[str] = field(default_factory=list)  # cleanup/infra notes, NOT reviewer findings



class GenerateRequest(BaseModel):
    query: str = Field(min_length=1)


class AgentResponse(BaseModel):
    code: str
    description: str
    assumptions: list[str]
    passed: bool
    confidence: Confidence
    issues: list[str]
    retries_taken: int