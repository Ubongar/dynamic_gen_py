from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


Confidence = Literal["high", "medium", "low"]


@dataclass(slots=True)
class GenResult:
    code: str
    description: str
    assumptions: list[str] = field(default_factory=list)


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

