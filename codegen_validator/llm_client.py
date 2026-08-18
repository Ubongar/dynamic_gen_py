import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError, InternalServerError
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
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self._model = model or os.getenv("OPENAI_MODEL", "olori-image")

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
                "must not be left empty. DO NOT import the generated classes/functions in the tests field; the test code "
                "is appended directly to the main code during validation and they are already in scope.\n "
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
        # Removed LLMClientError so fatal errors (like 400 Bad Request) fail fast
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, InternalServerError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
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
                max_tokens=8192
            )
        except APIStatusError as exc:
            # Let tenacity automatically retry rate limits and internal server errors
            if isinstance(exc, (RateLimitError, InternalServerError)):
                raise
            
            # For all other API errors (e.g., 400, 401, 404), format a detailed message and fail immediately
            self._log_stage("llm_call_api_status_error")
            status = getattr(exc, "status_code", None)
            response_data = getattr(exc, "response", None)
            
            details = []
            if status is not None:
                details.append(f"status {status}")
            if response_data is not None:
                details.append(f"response: {response_data}")
            details_str = f" ({', '.join(details)})" if details else ""
            
            raise LLMClientError(
                f"The LLM provider rejected the request{details_str}: {exc}"
            ) from exc
            
        except Exception as exc:
            # Fallback for completely unexpected network or system errors
            self._log_stage("llm_call_unexpected_error")
            raise LLMClientError(
                f"LLM request failed ({type(exc).__name__}): {exc!r}"
            ) from exc

        self._log_stage("llm_call_end")
        
        if not response.choices or not response.choices[0].message:
            raise LLMClientError("Server returned an empty or malformed payload.")

        message = response.choices[0].message
        content = message.content

        if not content:
            reasoning = getattr(message, 'reasoning', None)
            if reasoning:
                raise LLMClientError(
                    "The model ran out of tokens while 'thinking' and did not output the final code. "
                    "Try increasing max_tokens further."
                )
            raise LLMClientError("LLM returned empty content. No code was generated.")

        # Strip possible markdown code fences if the model adds ```json wrappers
        clean_content = content.strip()
        if clean_content.startswith("```"):
            lines = clean_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_content = "\n".join(lines).strip()

        try:
            parsed = json.loads(clean_content)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]
        except json.JSONDecodeError as exc:
            truncated_output = content[:500] + "\n...[truncated]" if len(content) > 500 else content
            raise LLMClientError(f"LLM did not return valid JSON.\nError: {exc}\nRaw Output:\n{truncated_output}") from exc

        if not isinstance(parsed, dict):
            raise LLMClientError(f"LLM JSON response was not an object. Type received: {type(parsed)}")

        return parsed
    
    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())