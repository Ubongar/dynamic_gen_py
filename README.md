"# dynamic_gen_py" 

## codegen_validator

Python project implementing a code generation + static/logic validation pipeline:

- `codegen_validator/generator.py` — generation + repair
- `codegen_validator/validator.py` — AST static check + LLM logic check
- `codegen_validator/agent.py` — retry orchestration
- `codegen_validator/api.py` — FastAPI `/generate`
- `codegen_validator/cli.py` — CLI over HTTP API
- `codegen_validator/tests/query_harness.py` — query benchmark harness

## Setup with uv

```bash
uv sync
```

## Run API

```bash
export GROQ_API_KEY=your_key
uv run uvicorn codegen_validator.api:app --reload
```

## Run CLI

```bash
uv run python -m codegen_validator.cli "Write a function that sums a list"
```

## Run query harness

```bash
uv run python -m codegen_validator.tests.query_harness --base-url http://127.0.0.1:8000
```"
