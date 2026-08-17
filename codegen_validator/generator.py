from __future__ import annotations

import logging
from datetime import datetime, timezone

from .llm_client import LLMClient
from .models import GenResult


LOGGER = logging.getLogger(__name__)

GENERATOR_PROMPT = (
    "You are a precise Python code generator. Given a natural language request, "
    "produce a single Python function or script. Include a brief description of "
    "what the code does and list any assumptions made about ambiguous parts of "
    "the request (e.g. assumed data types, assumed function signatures, assumed "
    "external resources like a database or API the code references). Return "
    "strict JSON matching this schema: {code, description, assumptions}."
)


class Generator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def generate(self, query: str) -> GenResult:
        self._log_stage("generate_start")
        result = self._llm_client.generate_code(query=query, system_prompt=GENERATOR_PROMPT)
        self._log_stage("generate_end")
        return result

    def repair(self, query: str, code: str, issues: list[str]) -> GenResult:
        self._log_stage("repair_start")
        prompt = (
            "The previous code needs targeted fixes.\n"
            f"Original query:\n{query}\n\n"
            f"Current code:\n{code}\n\n"
            f"Validator issues:\n{issues}\n\n"
            "Return strict JSON matching this schema: {code, description, assumptions}. "
            "Only change what is needed to address the listed issues."
        )
        result = self._llm_client.repair_code(prompt=prompt)
        self._log_stage("repair_end")
        return result

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())

