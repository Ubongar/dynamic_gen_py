# dynamic_gen_py

A local Python pipeline that turns a natural-language request into Python code, generates executable tests alongside it, validates both, and asks an LLM to repair failures through bounded retry cycles. It is a command-line tool, not a web service.

## Repository layout

- [pyproject.toml](pyproject.toml) — project metadata and Python dependencies
- [query_harness.py](query_harness.py) — 17-query batch regression harness with categorized results and append-only run history
- [codegen_validator/__init__.py](codegen_validator/__init__.py) — package marker
- [codegen_validator/agent.py](codegen_validator/agent.py) — high-level orchestration for generation, validation, repair, retry, and post-pass cleanup
- [codegen_validator/cli.py](codegen_validator/cli.py) — command-line entry point
- [codegen_validator/generator.py](codegen_validator/generator.py) — prompt-to-code generation, repair prompt logic, and local mock-stripping cleanup
- [codegen_validator/llm_client.py](codegen_validator/llm_client.py) — OpenAI-compatible chat client, JSON parsing, Pydantic response validation, adaptive token-budget retries, and transient-error retries
- [codegen_validator/models.py](codegen_validator/models.py) — result and validation data structures
- [codegen_validator/validator.py](codegen_validator/validator.py) — syntax checking, subprocess execution validation, environment-vs-code-bug detection, and logic review
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

If pip itself is missing from the venv (`No module named pip`), run:

```bash
python -m ensurepip --upgrade
```

## Environment variables

The project uses the OpenAI Python client against an OpenAI-compatible chat-completions endpoint. `OPENAI_API_KEY` is required. `OPENAI_BASE_URL` is optional and can point to a local or hosted compatible provider. The default model is `olori-image` unless `OPENAI_MODEL` is set.

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=http://your-host:port/v1  # optional
export OPENAI_MODEL=your-model-name              # optional
```

### Optional: Groq for cleanup fallback only

If set, `GROQ_API_KEY` enables a second, Groq-backed `LLMClient` used **only** as a fallback for the post-pass cleanup step, never for `generate`, `repair`, or `logic_check`. Cleanup normally runs locally via AST stripping with no network call at all; the LLM fallback only fires if the generated code deviated from the expected mock-stubbing template, and its output is always re-validated with `static_check` before being kept. This cannot reduce overall accuracy, since it never touches the code-writing or code-judging path.

```bash
export GROQ_API_KEY=your_groq_key_here
export GROQ_MODEL=openai/gpt-oss-120b            # optional, this is the default
export GROQ_BASE_URL=https://api.groq.com/openai/v1  # optional, this is the default
```

### Optional: lighter model for logic review

`OPENAI_REVIEW_MODEL` is opt-in and unset by default. If set, `logic_check` uses this model instead of the primary `OPENAI_MODEL`. Leave it unset unless you've verified the lighter model's review quality is acceptable for your use case, this is the one knob that can trade accuracy for speed if misused.

```bash
export OPENAI_REVIEW_MODEL=your-lighter-model   # optional, defaults to OPENAI_MODEL if unset
```

### PowerShell (Windows) quick setup

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
$env:OPENAI_BASE_URL = "http://your-host:port/v1"
$env:OPENAI_MODEL = "your-model-name"
$env:GROQ_API_KEY = "your_groq_key_here"          # optional
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

Add `--verbose` to also print non-blocking reviewer commentary on a PASS (hidden by default, since it isn't a real issue):

```bash
python -m codegen_validator.cli "Write a function that sums a list" --verbose
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

This runs all 17 benchmark queries across the `easy`, `medium`, `hard`, `ambiguous`, `edge_case`, and `adversarial` categories and prints a summary as it runs. Each query makes real LLM calls. Use `--max-retries 2` to change the retry limit (must be 1 or higher, `0` is rejected since it would skip validation entirely) or `--no-json-file` to avoid writing the results file.

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
            |                    (JSON + Pydantic + adaptive
            |                     token-budget retry)
            |                                  |
            |                                  v
            |                            GenResult
            |                                  |
            +--> clarification_needed? ------> return AgentResult
            |
            +--> Validator.static_check(code)
            |       ast.parse + pyflakes (undefined names fail;
            |       unused imports stay informational)
            |
            +--> Validator.execution_check(code + tests)
            |       temporary Python file + subprocess
            |       10-second timeout; 256 MB POSIX memory limit
            |       detects "missing pip in sandbox" as an
            |       ENVIRONMENT issue, not a code bug, and skips
            |       repair for it instead of wasting an LLM call
            |
            +--> Validator.logic_check(code, description, query)
                    LLM-based, read-only review
            |
            +--> pass: Generator.cleanup(code)
            |       local AST-based mock-stripping first (no
            |       network call); LLM fallback only if the code
            |       deviated from the expected template; cleaned
            |       code is re-validated with static_check before
            |       being kept, otherwise the original validated
            |       code ships instead
            |       --> return AgentResult (cleanup notes go in
            |           pipeline_notes, kept separate from real
            |           reviewer issues)
            |
            +--> failure: Generator.repair(...) and retry
                    (an LLM provider failure here returns a normal
                    failed AgentResult instead of crashing the run)
```

### Request and response flow

1. The CLI or harness creates one shared `LLMClient`, then injects it into `Generator` and `Validator`, which are injected into `CodeAgent`. Optionally, a second Groq-backed `LLMClient` is injected into `Generator` as a cleanup-only fallback, and a third, lighter OpenAI `LLMClient` can be injected into `Validator` for `logic_check` (opt-in, see Environment variables above).
2. `Generator.generate()` asks the model for strict JSON containing `code`, `tests`, `description`, `assumptions`, and optional `clarification_needed`. `LLMClient` parses the JSON and validates its schema with Pydantic. If the model runs out of token budget mid-response (either no content at all, or a truncated/unparsable JSON body), `LLMClient` automatically retries the same request with a larger `max_tokens` budget, up to a capped ceiling, before giving up.
3. If a request names a specific technology (e.g. "using MySQL"), the generator prompt requires the real driver code, never a silent substitution to something locally runnable like sqlite3. Since this sandbox has no live external services, the model is instructed to mock the connection layer with `unittest.mock` instead, using a generic, verified helper (`_stub_missing_package`) for packages that aren't installed here at all. That helper only stubs a package if the real one fails to import, so on a machine where the real driver is installed, the real driver is used.
4. If clarification is requested, `CodeAgent` returns immediately with `passed=False`; no generated code is validated.
5. Otherwise, each attempt runs in order: AST syntax parsing and pyflakes, generated-code-plus-tests execution in a temporary subprocess, then an LLM logic review. Execution is intentionally not a full security sandbox: generated code can access the filesystem and network.
6. Any failed check is sent to `Generator.repair()` with the original query, current code, and validator issues, unless the failure is a sandbox environment issue (e.g. no working `pip`), in which case it's reported directly without spending a repair call trying to fix code that was never broken. A provider failure during `repair()` also short-circuits to a normal failed result rather than an unhandled exception.
7. On a pass, `Generator.cleanup()` strips the mock-stubbing scaffolding so the shipped code reads as real, direct driver code. This happens locally via AST transformation whenever possible; an LLM call is only used as a fallback, and its output is always re-checked with `static_check` before being trusted, an unsafe cleanup result is discarded in favor of the original validated code.
8. The agent repeats validation until it passes or reaches `max_retries` (three by default, must be at least one).
9. The CLI prints code, generated tests, description, assumptions, status, confidence, retries, reviewer issues (hidden on PASS unless `--verbose`), and separately, any pipeline notes from the cleanup step.

### Data contracts

- `GenResult` — generator output plus tests, assumptions, and optional clarification.
- `CheckResult` — validator status, issues, confidence, error, and notes.
- `AgentResult` — final code, tests, metadata, pass/fail status, confidence, retry count, optional clarification, and `pipeline_notes` for cleanup/infrastructure notes kept separate from real reviewer findings.
- `GenerateRequest` and `AgentResponse` — Pydantic models available for structured API-facing integration, although this repository currently exposes only the CLI and harness entry points.

## Notes

- The project is designed for prompt-driven code synthesis and validation, not for a standalone web app.
- The current runtime is local and environment-driven, so the model provider, endpoint, and model are configured through environment variables rather than hardcoded values.
- Generated tests are appended directly to generated code during execution; they must call the generated functions/classes without importing them. Printed tests reflect the pre-cleanup code and may reference mock objects the cleaned code shown above no longer defines.
- Temporary execution files are removed after each check. POSIX runs apply CPU and memory limits (with a few seconds of headroom over the wall-clock timeout, since CPU-seconds and wall-clock time measure different things). Windows still relies on the subprocess timeout and is not a complete sandbox.
- The project intentionally uses real LLM calls in the harness so it can catch regressions before demos or submissions.
- Accuracy-critical steps (`generate`, `repair`, `logic_check` by default) always use the primary model. Only the cleanup fallback (optional Groq) and logic review (optional lighter model, opt-in only) can be pointed at a different, faster model, and both changes require an explicit environment variable, nothing changes by default.