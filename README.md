# dynamic_gen_py

A Python project for generating code from a natural-language prompt, validating it with static checks and runtime execution, and repairing failures through repeated LLM passes.

## Repository layout

- [pyproject.toml](pyproject.toml) — project metadata and Python dependencies
- [query_harness.py](query_harness.py) — batch regression harness for prompt validation
- [codegen_validator/__init__.py](codegen_validator/__init__.py) — package marker
- [codegen_validator/agent.py](codegen_validator/agent.py) — high-level orchestration for generation, validation, repair, and retry loops
- [codegen_validator/cli.py](codegen_validator/cli.py) — command-line entry point
- [codegen_validator/generator.py](codegen_validator/generator.py) — prompt-to-code generation and repair prompt logic
- [codegen_validator/llm_client.py](codegen_validator/llm_client.py) — LLM provider wrapper, currently supports OpenAI-compatible endpoints and Groq compatibility fallback
- [codegen_validator/models.py](codegen_validator/models.py) — result and validation data structures
- [codegen_validator/validator.py](codegen_validator/validator.py) — syntax checking, subprocess execution validation, and logic review
- [codegen_validator/tests/test_smoke.py](codegen_validator/tests/test_smoke.py) — smoke tests for static validation and defaults

## Setup

### With uv

```bash
uv sync
```

### With pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

## Environment variables

The project is configured for an OpenAI-compatible endpoint by default.

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=http://your-host:port/v1
export OPENAI_MODEL=your-model-name
```

It also accepts legacy Groq-style variables if needed:

```bash
export GROQ_API_KEY=...
export GROQ_MODEL=...
```

### PowerShell (Windows) quick setup

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
$env:OPENAI_BASE_URL = "http://your-host:port/v1"
$env:OPENAI_MODEL = "your-model-name"
```

To load them from a local `.env` file in PowerShell:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match "^\s*#" -or $_ -match "^\s*$") { return }
    $name, $value = $_ -split "=", 2
    [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
}
```

## Run the CLI

```bash
python -m codegen_validator.cli "Write a function that divides two numbers and handles division by zero"
```

or with environment loaded from a shell:

```bash
set -a
source .env
set +a
python -m codegen_validator.cli "Write a function that sums a list"
```

## Run the regression harness

```bash
python query_harness.py
```

This runs the full benchmark set and prints PASS/FAIL results for each prompt category.

## How the pipeline works

1. The user sends a natural-language request to the CLI or harness.
2. [codegen_validator/generator.py](codegen_validator/generator.py) asks the LLM for Python code and tests.
3. [codegen_validator/validator.py](codegen_validator/validator.py) verifies syntax, executes the generated code/tests, and asks the model to review logic.
4. If validation fails, [codegen_validator/agent.py](codegen_validator/agent.py) triggers a repair cycle and retries.
5. The final result includes generated code, tests, assumptions, and pass/fail quality metadata.

## Notes

- The project is designed for prompt-driven code synthesis and validation, not for a standalone web app.
- The current runtime is local and environment-driven, so the model provider and endpoint are configured through environment variables rather than hardcoded values.
- The project intentionally uses real LLM calls in the harness so it can catch regressions before demos or submissions.
