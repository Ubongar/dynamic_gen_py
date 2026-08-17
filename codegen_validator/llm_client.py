from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from groq import Groq
from pydantic import BaseModel, ValidationError

from .models import Confidence, GenResult


LOGGER = logging.getLogger(__name__)


class LLMClientError(Exception):
    pass


class _GenResultSchema(BaseModel):
    code: str
    description: str
    assumptions: list[str]


class _LogicResultSchema(BaseModel):
    correct: bool
    issues: list[str]
    confidence: Confidence


@dataclass(slots=True)
class LogicReviewResult:
    correct: bool
    issues: list[str]
    confidence: Confidence


class LLMClient:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile") -> None:
        self._client = Groq(api_key=api_key)
        self._model = model

    def generate_code(self, query: str, system_prompt: str) -> GenResult:
        payload = self._chat_json(system_prompt=system_prompt, user_prompt=query)
        return self._parse_gen_result(payload)

    def repair_code(self, prompt: str) -> GenResult:
        payload = self._chat_json(
            system_prompt=(
                "You are a precise Python code generator and repair assistant. "
                "Return strict JSON matching: {code, description, assumptions}."
            ),
            user_prompt=prompt,
        )
        return self._parse_gen_result(payload)

    def review_logic(self, prompt: str) -> LogicReviewResult:
        payload = self._chat_json(
            system_prompt=(
                "You are a rigorous code reviewer. You are NOT executing this code, "
                "you are reading it. Return JSON: {correct: bool, issues: [...], "
                "confidence: 'high'|'medium'|'low'}."
            ),
            user_prompt=prompt,
        )
        try:
            parsed = _LogicResultSchema.model_validate(payload)
        except ValidationError as exc:
            raise LLMClientError(f"Invalid logic review schema: {exc}") from exc
        return LogicReviewResult(
            correct=parsed.correct,
            issues=parsed.issues,
            confidence=parsed.confidence,
        )

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self._log_stage("llm_call_start")
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        self._log_stage("llm_call_end")
        content = response.choices[0].message.content
        if content is None:
            raise LLMClientError("LLM returned empty content.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"LLM did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError("LLM JSON response was not an object.")
        return parsed

    @staticmethod
    def _parse_gen_result(payload: dict[str, Any]) -> GenResult:
        try:
            parsed = _GenResultSchema.model_validate(payload)
        except ValidationError as exc:
            raise LLMClientError(f"Invalid generator schema: {exc}") from exc
        return GenResult(
            code=parsed.code,
            description=parsed.description,
            assumptions=parsed.assumptions,
        )

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())

