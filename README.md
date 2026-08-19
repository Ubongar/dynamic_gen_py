# dynamic_gen_py

A local Python pipeline that turns a natural-language request into Python code, generates executable tests alongside it, validates both, and asks an LLM to repair failures through bounded retry cycles. It is a command-line tool, not a web service.

## Repository layout

- [pyproject.toml](pyproject.toml) — project metadata and Python dependencies
- [query_harness.py](query_harness.py) — 17-query batch regression harness with categorized results and append-only run history
- [codegen_validator/__init__.py](codegen_validator/__init__.py) — package marker
- [codegen_validator/agent.py](codegen_validator/agent.py) — high-level orchestration for generation, validation, repair, and retry loops
- [codegen_validator/cli.py](codegen_validator/cli.py) — command-line entry point
- [codegen_validator/generator.py](codegen_validator/generator.py) — prompt-to-code generation and repair prompt logic
- [codegen_validator/llm_client.py](codegen_validator/llm_client.py) — OpenAI-compatible chat client, JSON parsing, Pydantic response validation, and transient-error retries
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

The project uses the OpenAI Python client against an OpenAI-compatible chat-completions endpoint. `OPENAI_API_KEY` is required. `OPENAI_BASE_URL` is optional and can point to a local or hosted compatible provider. The default model is `olori-image` unless `OPENAI_MODEL` is set.

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=http://your-host:port/v1  # optional
export OPENAI_MODEL=your-model-name              # optional
```

### PowerShell (Windows) quick setup

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
$env:OPENAI_BASE_URL = "http://your-host:port/v1"
$env:OPENAI_MODEL = "your-model-name"
```

The application also calls `load_dotenv()`, so a local `.env` file can provide these variables. Do not commit that file or API keys.

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
python query_harness.py --output results.json
```

This runs all 17 benchmark queries across the `easy`, `medium`, `hard`, `ambiguous`, `edge_case`, and `adversarial` categories and prints a summary as it runs. Each query makes real LLM calls. Use `--max-retries 2` to change the retry limit or `--no-json-file` to avoid writing the results file.

The default output path is `result.json` (the command above uses `results.json`). Every run is appended to that file as a historical entry. If the file contains one older result object, it is wrapped into a history list; if it is empty or invalid JSON, the harness starts a new history. Each query result includes the prompt, pass/fail metadata, confidence, retries, latency, clarification request, issues, crash details, extracted code, and a serialized raw agent result. Strings in the serialized result are truncated to 4,000 characters.

A provider or unexpected exception is isolated to that query and recorded with `crashed=true` and a `CRASH` status, so later queries still run. Validation failures are recorded separately as `passed=false` with `crashed=false`. The process exits non-zero only when at least one query crashes; ordinary validation failures do not change the exit code.

## Current architecture

```text
CLI or query_harness.py
            |
            v
        CodeAgent.run(query)
            |
            +--> Generator.generate(query) ----+
            |                                  |
            |                                  v
            |                            LLMClient
            |                         (JSON + Pydantic)
            |                                  |
            |                                  v
            |                            GenResult
            |                                  |
            +--> clarification_needed? ------> return AgentResult
            |
            +--> Validator.static_check(code)
            |       ast.parse + pyflakes
            |
            +--> Validator.execution_check(code + tests)
            |       temporary Python file + subprocess
            |       10-second timeout; 256 MB POSIX memory limit
            |
            +--> Validator.logic_check(code, description, query)
                    LLM-based, read-only review
            |
            +--> pass: return AgentResult
            |
            +--> failure: Generator.repair(...) and retry
```

### Request and response flow

1. The CLI or harness creates one shared `LLMClient`, then injects it into `Generator` and `Validator`, which are injected into `CodeAgent`.
2. `Generator.generate()` asks the model for strict JSON containing `code`, `tests`, `description`, `assumptions`, and optional `clarification_needed`. `LLMClient` parses the JSON and validates its schema with Pydantic.
3. If clarification is requested, `CodeAgent` returns immediately with `passed=False`; no generated code is validated.
4. Otherwise, each attempt runs in order: AST syntax parsing and pyflakes, generated-code-plus-tests execution in a temporary subprocess, then an LLM logic review. Execution is intentionally not a full security sandbox: generated code can access the filesystem and network.
5. Any failed check is sent to `Generator.repair()` with the original query, current code, and validator issues. The agent repeats until it passes or reaches `max_retries` (three by default).
6. The CLI prints code, generated tests, description, assumptions, status, confidence, retries, and issues. The harness additionally records latency, crash details, and the serialized raw agent result as a bounded JSON history. It searches several conventional result-field names when extracting code, with `code` as the first choice.

### Data contracts

- `GenResult` — generator output plus tests, assumptions, and optional clarification.
- `CheckResult` — validator status, issues, confidence, error, and notes.
- `AgentResult` — final code, tests, metadata, pass/fail status, confidence, retry count, and optional clarification.
- `GenerateRequest` and `AgentResponse` — Pydantic models available for structured API-facing integration, although this repository currently exposes only the CLI and harness entry points.

## Notes

- The project is designed for prompt-driven code synthesis and validation, not for a standalone web app.
- The current runtime is local and environment-driven, so the model provider, endpoint, and model are configured through environment variables rather than hardcoded values.
- Generated tests are appended directly to generated code during execution; they must call the generated functions/classes without importing them.
- Temporary execution files are removed after each check. POSIX runs apply CPU and memory limits; Windows still relies on the subprocess timeout and is not a complete sandbox.
- The project intentionally uses real LLM calls in the harness so it can catch regressions before demos or submissions.
