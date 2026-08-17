from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException

from .agent import CodeAgent
from .generator import Generator
from .llm_client import LLMClient, LLMClientError
from .models import AgentResponse, GenerateRequest
from .validator import Validator


LOGGER = logging.getLogger(__name__)


def create_agent() -> CodeAgent:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required.")
    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    llm_client = LLMClient(api_key=api_key, model=model)
    generator = Generator(llm_client=llm_client)
    validator = Validator(llm_client=llm_client)
    return CodeAgent(generator=generator, validator=validator, max_retries=3)


def create_app() -> FastAPI:
    app = FastAPI(title="codegen_validator")

    @app.post("/generate", response_model=AgentResponse)
    def generate(request: GenerateRequest, agent: CodeAgent = Depends(create_agent)) -> AgentResponse:
        LOGGER.info("api_generate_start %s", datetime.now(timezone.utc).isoformat())
        try:
            result = agent.run(request.query)
        except LLMClientError as exc:
            raise HTTPException(status_code=500, detail=f"LLM response validation failed: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        LOGGER.info("api_generate_end %s", datetime.now(timezone.utc).isoformat())
        return AgentResponse(
            code=result.code,
            description=result.description,
            assumptions=result.assumptions,
            passed=result.passed,
            confidence=result.confidence,
            issues=result.issues,
            retries_taken=result.retries_taken,
        )

    return app


app = create_app()

