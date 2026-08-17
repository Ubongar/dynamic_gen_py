# pyright: reportMissingImports=false
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import groq  # type: ignore[import-not-found]  # Import the base groq module to catch its errors
from groq import Groq  # type: ignore[import-not-found]
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import Confidence, GenResult


LOGGER = logging.getLogger(__name__)


class LLMClientError(Exception):
    pass


class _GenResultSchema(BaseModel):
    code: str
    tests: str = ""
    description: str
    assumptions: list[str]
    clarification_needed: str | None = None  # Allow the LLM to return null or a string


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
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b") -> None:
        self._client = Groq(api_key=api_key)
        self._model = model

    @staticmethod
    def _parse_gen_result(payload: dict[str, Any]) -> GenResult:
        try:
            parsed = _GenResultSchema.model_validate(payload)
        except ValidationError as exc:
            raise LLMClientError(f"Invalid generator schema: {exc}") from exc
        return GenResult(
            code=parsed.code,
            tests=parsed.tests,
            description=parsed.description,
            assumptions=parsed.assumptions,
            clarification_needed=parsed.clarification_needed,
        )

    def generate_code(self, query: str, system_prompt: str) -> GenResult:
        payload = self._chat_json(system_prompt=system_prompt, user_prompt=query)
        return self._parse_gen_result(payload)

    def repair_code(self, prompt: str) -> GenResult:
        payload = self._chat_json(
            system_prompt=(
                "You are an elite Python code repair assistant. The previous generated code failed static validation, "
                "execution, or logic checks. "
                "CRITICAL CONSTRAINTS:\n"
                "1. Fix the reported validator issues directly without stripping out existing functionality or comments.\n"
                "2. Maintain strict syntax safety by using raw strings (r\"...\") for any docstrings, regex, and math.\n"
                "3. Ensure the repaired code retains type hints and graceful try/except error handling.\n"
                "3b. If the code opens a sqlite3.Connection to a real file, it MUST be explicitly closed with "
                "conn.close() in a finally block (or via contextlib.closing). `with connection:` only handles the "
                "transaction, it does not close the connection, leaving it open causes file-lock errors.\n"
                "4. The 'tests' field must call every function in 'code' with real inputs and assert on results, it "
                "must not be left empty.\n"
                "5. Return ONLY strict JSON matching: {\"code\": \"string\", \"tests\": \"string\", "
                "\"description\": \"string\", \"assumptions\": [\"string\"], \"clarification_needed\": \"string or null\"}. "
                "Ensure 'assumptions' is a JSON array. DO NOT include markdown code block wrappers. "
                "6. NEVER escape apostrophes/single quotes with a backslash (\\' is invalid JSON and will be rejected "
                "outright), write them plain instead."
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

    @retry(
        # Only retry on actual Groq API issues (rate limits, timeouts, 500 errors)
        retry=retry_if_exception_type((groq.APIError, groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError)),
        # Wait 1s, then 2s, then 4s, etc., up to 10 seconds between retries
        wait=wait_exponential(multiplier=1, min=1, max=10),
        # Give up after 4 total attempts
        stop=stop_after_attempt(4),
        reraise=True
    )
    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self._log_stage("llm_call_start")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                # NOTE: max_tokens is RESERVED budget, not just actual usage, Groq's
                # tokens-per-minute limit is checked against (prompt_tokens + max_tokens),
                # not what's actually generated. Keep this comfortably under your account's
                # TPM cap minus a typical prompt size, raise it only alongside checking your
                # actual TPM limit (see Groq console > Settings > Billing).
                max_tokens=4096
            )
        except groq.APIStatusError as exc:
            if isinstance(exc, (groq.RateLimitError, groq.InternalServerError)):
                # These ARE meant to be retried by the @retry decorator wrapping this
                # method (see retry_if_exception_type above), re-raise unchanged so
                # tenacity still sees the original type and retries as configured.
                raise
            # Everything else here (BadRequestError from malformed model JSON, the
            # 413 tokens-per-minute case, etc.) is not fixed by retrying an identical
            # request, so convert it to a clean, catchable error instead of letting
            # the raw API exception propagate all the way up and crash the CLI.
            self._log_stage("llm_call_api_status_error")
            raise LLMClientError(
                f"The LLM provider rejected the request (status {exc.status_code}). "
                f"Common causes: malformed JSON from the model, or the request exceeding "
                f"your account's tokens-per-minute limit. Raw provider error: {exc!s}"
            ) from exc
        self._log_stage("llm_call_end")
        content = response.choices[0].message.content
        if content is None:
            raise LLMClientError("LLM returned empty content.")
        try:
            parsed = json.loads(content)

            # Defensive Check 1: If the LLM double-encoded the JSON into a string, parse it again
            if isinstance(parsed, str):
                parsed = json.loads(parsed)

            # Defensive Check 2: If the LLM wrapped the object in a list, extract the first item
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]

        except json.JSONDecodeError as exc:
            raise LLMClientError(f"LLM did not return valid JSON. \nRaw Output: {content}\nError: {exc}") from exc

        # Final strict validation with raw output debugging
        if not isinstance(parsed, dict):
            raise LLMClientError(f"LLM JSON response was not an object. Type received: {type(parsed)}\nRaw Output: {content}")

        return parsed

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())