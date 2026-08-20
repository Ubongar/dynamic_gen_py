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

# Baseline budget for a single request. OpenAI's real rate limits are far
# higher than the previous provider's, so this can sit higher without
# risking the earlier reservation-based 413s.
_BASE_MAX_TOKENS = 16384

# Hard ceiling we will never exceed even when escalating after a truncation
# failure. Raise this only if you're on a model whose context window can't
# fit prompt + this many output tokens.
_MAX_TOKENS_CEILING = 32768

# How many times to retry with a bigger budget after a truncation/
# reasoning-exhaustion failure before giving up and surfacing the error.
_TOKEN_BUDGET_RETRIES = 2

# How much to add to max_tokens on each retry after a truncation failure.
_TOKEN_BUDGET_STEP = 2048

# Appended to every system prompt to discourage the model from spending its
# token budget on long internal reasoning before it ever writes the JSON.
_BUDGET_DISCIPLINE_SUFFIX = (
    "\n\nIMPORTANT: Keep any internal reasoning brief. Do not deliberate at "
    "length before producing output, move directly to writing the final "
    "JSON object so the full response fits within the available token budget."
)


class LLMClientError(Exception):
    pass


class _TokenBudgetExceededError(LLMClientError):
    """
    Raised internally when the model exhausts its token budget, either by
    using it all on hidden reasoning (empty content) or by getting cut off
    mid-write (unparsable/truncated JSON). This is caught and retried with
    a larger max_tokens by _chat_json before ever reaching the caller. If
    still failing after retries, it is re-raised as a plain LLMClientError.
    """
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


class _CleanupResultSchema(BaseModel):
    code: str


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
            apply_budget_discipline=False,
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
        
    def cleanup_code(self, code: str) -> str:
        payload = self._chat_json(
            system_prompt=(
                "You are an expert Python developer. Your objective is to clean up validated code "
                "by removing sandbox-specific testing artifacts.\n"
                "CRITICAL CONSTRAINTS:\n"
                "1. Remove ONLY `_stub_missing_package`, `unittest.mock` imports/patches, and stubbed dummy connections.\n"
                "2. DO NOT alter the core business logic, type hints, or error handling.\n"
                "3. Return ONLY strict JSON matching this schema: {\"code\": \"string\"}.\n"
                "4. DO NOT wrap the output in markdown code blocks."
            ),
            user_prompt=f"Clean this code:\n\n{code}",
        )
        try:
            parsed = _CleanupResultSchema.model_validate(payload)
            return parsed.code
        except ValidationError as exc:
            raise LLMClientError(f"Invalid cleanup schema: {exc}") from exc

    def _chat_json(self, system_prompt: str, user_prompt: str, apply_budget_discipline: bool = True) -> dict[str, Any]:
        """
        Outer wrapper around _chat_json_once that adapts max_tokens on the
        fly. If a call fails because the model ran out of budget while
        thinking or got cut off mid-JSON, that is not a fatal error, it is
        a signal to retry the SAME request with more headroom. Any other
        LLMClientError (bad schema, 4xx that isn't 413, etc.) is not
        retried here and propagates immediately, retrying those would just
        burn time reproducing the same failure.
        """
        max_tokens = _BASE_MAX_TOKENS
        last_error: LLMClientError | None = None

        for attempt in range(_TOKEN_BUDGET_RETRIES + 1):
            try:
                return self._chat_json_once(system_prompt, user_prompt, max_tokens, apply_budget_discipline)
            except _TokenBudgetExceededError as exc:
                last_error = exc
                if attempt == _TOKEN_BUDGET_RETRIES or max_tokens >= _MAX_TOKENS_CEILING:
                    break
                max_tokens = min(max_tokens + _TOKEN_BUDGET_STEP, _MAX_TOKENS_CEILING)
                self._log_stage(f"llm_call_retry_larger_budget_{max_tokens}")
                continue
            except (APIConnectionError, RateLimitError, InternalServerError) as exc:
                # tenacity's retries inside _chat_json_once were exhausted and
                # reraise=True let the ORIGINAL OpenAI exception through. Every
                # caller of this class only knows about LLMClientError, so an
                # unconverted SDK exception here would crash agent.py outright
                # instead of producing a normal failed AgentResult.
                raise LLMClientError(
                    f"LLM request failed after retries ({type(exc).__name__}): {exc}"
                ) from exc

        assert last_error is not None
        raise LLMClientError(
            f"{last_error} (gave up after {_TOKEN_BUDGET_RETRIES + 1} attempts, "
            f"final max_tokens={max_tokens}). The request likely needs more output "
            "than the account's token budget currently allows for a single call."
        ) from last_error

    @retry(
        # Removed LLMClientError so fatal errors (like 400 Bad Request) fail fast.
        # Capped at 2 (was 4): this already runs inside the outer budget-escalation
        # loop in _chat_json, 4 inner x 3 outer meant a single generate() call could
        # fire up to 12 real API requests with exponential backoff on a bad day.
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, InternalServerError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(2),
        reraise=True
    )
    def _chat_json_once(
        self, system_prompt: str, user_prompt: str, max_tokens: int, apply_budget_discipline: bool = True
    ) -> dict[str, Any]:
        self._log_stage("llm_call_start")
        effective_system_prompt = system_prompt
        if apply_budget_discipline:
            effective_system_prompt = system_prompt + _BUDGET_DISCIPLINE_SUFFIX
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": effective_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens
            )
        except APIStatusError as exc:
            # Let tenacity automatically retry rate limits and internal server errors
            if isinstance(exc, (RateLimitError, InternalServerError)):
                raise

            status = getattr(exc, "status_code", None)

            if status == 413:
                # The provider reserves prompt+max_tokens against the account's
                # TPM budget up front. This is not a truncation failure, bumping
                # max_tokens further will only make it worse, so this is
                # deliberately NOT raised as _TokenBudgetExceededError.
                self._log_stage("llm_call_token_budget_rejected")
                raise LLMClientError(
                    f"The provider rejected this request for exceeding the account's token "
                    f"budget/rate limit (413) at max_tokens={max_tokens}. This needs a smaller "
                    "prompt or a higher TPM limit on the account, retrying with more tokens "
                    "would make this worse, not better."
                ) from exc

            # For all other API errors (e.g., 400, 401, 404), format a detailed message and fail immediately
            self._log_stage("llm_call_api_status_error")
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

        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = getattr(choice, "finish_reason", None)

        # finish_reason == "length" is the standard OpenAI signal that the
        # response was cut off because max_tokens ran out, this covers both
        # empty content (budget spent entirely on reasoning tokens, which on
        # real OpenAI reasoning models are billed but not surfaced as visible
        # content) and content that exists but stops mid-sentence/mid-JSON.
        if finish_reason == "length":
            raise _TokenBudgetExceededError(
                f"The response was truncated (finish_reason='length') before "
                f"completing (max_tokens={max_tokens})."
            )

        if not content:
            # Fallback for non-OpenAI-standard providers that expose a
            # separate 'reasoning' field instead of (or alongside) finish_reason.
            reasoning = getattr(message, 'reasoning', None)
            if reasoning:
                raise _TokenBudgetExceededError(
                    f"The model ran out of tokens while 'thinking' and did not output the "
                    f"final code (max_tokens={max_tokens})."
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
            # A response that finished with a broken JSON body (unterminated
            # string, missing closing brace, "expecting value" at the tail,
            # etc.) is almost always the same root cause as the empty-content
            # case above: the model ran out of room mid-write. Treat it the
            # same way, retry with a bigger budget, rather than failing hard.
            raise _TokenBudgetExceededError(
                f"LLM response was truncated before valid JSON completed "
                f"(max_tokens={max_tokens}).\nError: {exc}\nRaw Output:\n{truncated_output}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMClientError(f"LLM JSON response was not an object. Type received: {type(parsed)}")

        return parsed

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())