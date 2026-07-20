# Prompt Ninja

Prompt Ninja turns plain prompts to testable application artifacts. Instead of scattering critical prompts across
source code, chat history, and one-off documents, each prompt lives in a
readable `*.prompt.toml` file with a purpose, model, typed inputs, output
contract, version, and semantic regression tests.

A prompt can therefore be reviewed like a code change, tested before release,
and maintained as requirements and models evolve. The question becomes “does
this version still satisfy its contract?” rather than “which wording worked
last time?”

Prompt Ninja has three ways to work:

| Interface | Best for |
| --- | --- |
| **UI** | Describing a desired behavior and letting the LLM-assisted Board of Prompts create and evaluate a prompt |
| **CLI** | Generating, validating, testing, updating, and automating prompts in CI |
| **Python API** | Loading versioned prompts in an application, rendering typed inputs, validating outputs, and observing real runs with hooks |

The Board of Prompts is the creation workflow inside Prompt Ninja. It extracts
requirements, drafts competing candidates, synthesizes a result, generates a
test fixture, and compiles a validated prompt artifact. The output is not just
generated text—it is ready to version, test, and improve.

## Why Prompt Ninja

Prompt Ninja removes the need to depend on a dedicated prompt engineer for every
AI behavior change. A team specifies the desired behavior, constraints, and
expected output; LLMs collaboratively generate the prompt implementation.

That prompt is then treated like software: stored as `*.prompt.toml`, versioned,
validated, and tested with pytest-style semantic test cases. This makes prompts
easier to review, automate, maintain, and share across a team.

The important long-term piece is continuous quality control. Even a strong
prompt can drift as models, data, business rules, or user expectations change.
The platform can repeatedly test real behavior against its contract, detect
degradation, and eventually propose—or safely run—a controlled self-update loop.

Two ways to frame it:

- **For engineers:** we make prompts into testable, maintainable application
  artifacts instead of handcrafted text.
- **For DevOps/platform:** we build the control plane around AI behavior—model
  configuration, test automation, versioning, rollout safety, monitoring, drift
  detection, and eventually managed prompt updates.

## Features

- Create prompts through an LLM-assisted UI with editable requirements and up
  to five reference files.
- Compare multiple prompt candidates and synthesize a tested final result.
- Store prompts in structured, human-readable `*.prompt.toml` files.
- Declare LLM models, typed variables, defaults, output contracts, and
  semantic tests alongside the prompt.
- Validate TOML structure, required template variables, input types, defaults,
  model paths, and output declarations before a prompt runs.
- Render JSON, dictionaries, typed lists, dates, datetimes, dynamic values, and
  Pydantic models predictably.
- Inject output instructions and Pydantic JSON Schema automatically.
- Run semantic regression tests with thresholds, targeted test selection,
  progress reporting, verbose diagnostics, and plain CI output.
- Update prompt files from natural-language feedback while preserving a
  `.bak` copy of the previous version.
- Load an entire prompt directory once with `PromptCollection`.
- Browse live OpenRouter models with pricing and structured-output capability.
- Observe sampled production runs through request, response, and error hooks.

## Quick start

### Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Node.js and npm for the UI
- An OpenRouter API key for generation, execution, semantic tests, and fixes

### Backend and CLI

```bash
cd backend
uv sync --no-editable
cp .env.example .env
```

Add your key to `backend/.env`:

```dotenv
OPENROUTER_API_KEY=your-key
```

Start the API:

```bash
uv run --no-editable uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The CLI is installed by `uv sync`:

```bash
uv run --no-editable prompt-ninja --help
uv run --no-editable prompt-ninja validate prompts
```

The non-editable install keeps the generated console command reliable on
macOS/Python environments that ignore hidden editable-install `.pth` files. If
the virtual environment is activated, the shorter `prompt-ninja ...` form also
works.

### UI

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite. In the UI, describe the behavior you want,
optionally add reference files, review the enhanced brief, select models, and
follow each Board of Prompts stage.

Uploaded files are processed in memory and are not persisted by the API.

## Understanding `prompt.toml`

Prompt definitions use the `*.prompt.toml` extension. The checked-in examples
live in [`backend/prompts`](backend/prompts); the
[`greeting.prompt.toml`](backend/prompts/greeting.prompt.toml) file is the
smallest complete example of specification `1.2`.

### File anatomy

```toml
[metadata]
spec_version = "1.2"
name = "project-summary"
description = "Creates a concise project summary."
used_by = ["backend/app/summary.py"]
version = "1.0.0"
output = "String"

[llm_model]
provider = "openrouter"
name = "google/gemini-2.5-flash"

[prompt]
system = "Summarize the supplied update for {{audience}}."
user = "{{update}}"

[[variables]]
name = "update"
type = "string"
description = "The project update to summarize."
required = true

[[variables]]
name = "audience"
type = "string"
description = "Who will read the summary."
required = false
default = "the project team"

[testing]
pass_threshold = 0.95

[[tests]]
name = "summarizes a project update"
variable.update = "The launch moved to Friday; Priya owns the checklist."
variable.audience = "leadership"
expected_output = """
A concise leadership-ready summary that preserves the Friday launch date and
Priya's ownership.
"""
```

Read the file from top to bottom:

- `[metadata]` identifies and versions the prompt, records its consumers, and
  declares its output.
- `[llm_model]` selects the default provider and model.
- `[prompt]` contains the system and user templates.
- Each `[[variables]]` table defines one typed template input.
- `[testing]` sets the minimum semantic score required to pass.
- Each `[[tests]]` table supplies inputs and natural-language success criteria.

Every `metadata.used_by` entry must be a repository-relative consumer path such
as `backend/app/agents.py`.

### Variables and rendering

Variable declarations support:

- `string` or `str`
- `integer` or `int`
- `number` or `float`
- `boolean` or `bool`
- `json`
- `dict`
- `list[T]`, including nested supported types
- `date`
- `datetime`
- `dynamic`
- A dotted Pydantic model path such as `app.models.Person`

Legacy `array` and `object` declarations remain supported. Optional variables
may provide a type-correct `default`; required variables produce an actionable
error when missing.

Prompt Ninja validates runtime values before rendering. JSON and dictionaries
use `json.dumps`, typed lists become comma-separated values, dates and datetimes
use their standard representations, and Pydantic models use their representation.
Templates can request an explicit format:

```toml
[prompt]
user = """
JSON: {{payload | json}}
Items: {{items | csv}}
Debug representation: {{person | repr}}
Text: {{value | str}}
"""
```

For a Pydantic input, keep TOML declarative and use an inline table in a test:

```toml
[[variables]]
name = "person"
type = "app.models.Person"
description = "The person to greet."
required = true

[[tests]]
name = "greets a typed person"
variable.person = { name = "Ada", role = "Engineer" }
expected_output = "Greets Ada appropriately."
```

Prompt Ninja resolves the model path and calls `Person.model_validate(...)`; it
does not evaluate Python expressions from TOML.

### Output contracts

`metadata.output` accepts:

- `String`
- `BigInt`
- A dotted path to a Pydantic `BaseModel`, such as
  `app.models.GreetingResult`

Prompt Ninja injects the output contract into the prepared system message on
every run. Pydantic outputs include the model's JSON Schema; `String` and
`BigInt` receive their corresponding format instructions. Prompt authors can
therefore describe behavior and quality without duplicating response-schema
instructions in the template.

When OpenRouter returns a structured response, Prompt Ninja reads the parsed
Pydantic object from the Responses API and validates it at the model boundary.

### Semantic tests

Each `[[tests]]` table defines runtime inputs with `variable.<name>` keys and
describes semantic correctness in `expected_output`:

```toml
[testing]
pass_threshold = 0.95

[[tests]]
name = "French translation contract"
variable.source = "Hello, world!"
expected_output = """
An accurate French translation that preserves the meaning and uses natural
French phrasing.
"""
```

The OpenRouter-backed judge compares the actual response with these criteria,
returns a score from `0.0` to `1.0`, and explains its rationale. The test passes
when the score meets `testing.pass_threshold`.

Run one prompt, a directory, or a specific test:

```bash
prompt-ninja test --prompt prompts/greeting.prompt.toml
prompt-ninja test-prompts --prompts-dir prompts
prompt-ninja test-prompts --prompt-name greeting --test-name "formal greeting"
```

## Python integration

### Load and run one prompt

`PromptNinja.from_file(...)` loads and validates a definition before it can be
used:

```python
import asyncio

from app.prompt_ninja import PromptNinja

prompt = PromptNinja.from_file("prompts/greeting.prompt.toml")


async def main():
    result = await prompt.run_openrouter({"name": "Ada", "tone": "friendly"})
    print(result.result)


asyncio.run(main())
```

The declared `app.models.GreetingResult` output means `result` is already a
validated Pydantic object.

### Load a prompt directory once

Use `PromptCollection` when an application owns several prompts:

```python
from app.prompt_ninja import PromptCollection

prompts = PromptCollection(dir="prompts")

greeting = prompts.greeting
brief_enhancer = prompts.brief_enhancer  # metadata.name = "brief-enhancer"
compiler = prompts["prompt_compiler"]    # exact metadata.name
```

The collection eagerly loads and validates every `*.prompt.toml` file at
construction. Later lookups do not read the filesystem again. Create a new
collection to pick up file changes. Duplicate names, ambiguous dot aliases, and
reserved aliases are rejected.

The bundled application creates `app.prompt_catalog.PROMPTS` once during
process startup and shares it across the council, brief enhancer, test harness,
and built-in CLI helpers.

### Runtime model overrides

Override a model for one call without changing the versioned TOML:

```python
from app.prompt_ninja import PromptRuntimeOptions


async def run_with_model():
    return await prompt.run_openrouter(
        {"name": "Ada"},
        runtime=PromptRuntimeOptions(model="openai/gpt-4.1-mini"),
    )
```

The override affects only that execution.

## CLI reference

Run `prompt-ninja --help` for the command list. Every command also supports
`--help`.

### `generate`

```text
prompt-ninja generate [OPTIONS]
```

- `--goal TEXT` — goal to generate a prompt for.
- `--config PATH` — configuration TOML; defaults to `prompt-ninja.toml`.
- `--output PATH` — destination `*.prompt.toml` path.

`generate` uses the Board of Prompts and writes a validated prompt artifact.

### `test`

```text
prompt-ninja test [OPTIONS]
```

- `--prompt PATH` — prompt file to test.
- `--judge-model TEXT` — semantic judge model; defaults to
  `google/gemini-2.5-flash`.
- `-n, --test-name TEXT` — run one named `[[tests]]` case.
- `-v, --verbose` — show inputs, criteria, actual output, rationale, and errors.
- `--plain` — disable Rich tables and progress output for CI.

### `test-prompts`

```text
prompt-ninja test-prompts [OPTIONS]
```

- `-t, --prompts-dir PATH` — directory to load as a prompt collection;
  defaults to `prompts`.
- `-p, --prompt-name TEXT` — run only this exact `metadata.name`.
- `--judge-model TEXT` — semantic judge model; defaults to
  `google/gemini-2.5-flash`.
- `-n, --test-name TEXT` — run one named test across the selected prompts.
- `-v, --verbose` — show detailed inputs, outputs, rationale, and errors.
- `--plain` — disable Rich tables and progress output for CI.

### `update`

```text
prompt-ninja update [OPTIONS] PROMPT_FILE FEEDBACK
```

- `PROMPT_FILE` — `*.prompt.toml` file to revise.
- `FEEDBACK` — requested change in natural language.
- `--model TEXT` — updater model; defaults to
  `google/gemini-2.5-flash`.

The returned TOML must validate before it is written. The original is preserved
beside it as a `.bak` file.

### `validate`

```text
prompt-ninja validate [OPTIONS] PATH
```

- `PATH` — one prompt file or a directory of prompt files.
- `--fix` — ask the LLM updater to repair invalid files.
- `--model TEXT` — model used by `--fix`; defaults to
  `google/gemini-2.5-flash`.

Validation checks the TOML specification and imports every declared output or
variable model path. `--fix` requires `OPENROUTER_API_KEY`.

### `ui`

```text
prompt-ninja ui [OPTIONS]
```

- `--port INTEGER` — API port from `1` through `65535`; defaults to `8000`.

### Examples

```bash
prompt-ninja generate --goal "Summarize legal documents in plain English"
prompt-ninja generate --config prompt-ninja.toml --output prompts/legal-summary.prompt.toml
prompt-ninja test --prompt prompts/greeting.prompt.toml --verbose
prompt-ninja test-prompts --prompts-dir prompts --prompt-name judge --plain
prompt-ninja update prompts/judge.prompt.toml "Reject responses that invent facts"
prompt-ninja validate prompts
prompt-ninja validate prompts/judge.prompt.toml --fix
prompt-ninja ui --port 8000
```

Generation, execution, semantic tests, updates, and `validate --fix` require
`OPENROUTER_API_KEY`. Plain validation does not.

## Runtime observability and hooks

`OpenRouterPromptClient` accepts hooks on each execution. A hook receives a
`PromptRunEvent` for:

- `request` — the rendered prompt is about to be sent.
- `response` — a parsed and validated output was returned.
- `error` — execution or output validation failed.

Each event includes a `run_id`, prompt name, provider, model, rendered system
and user messages, and the eventual output or error. Request and terminal
events share the same `run_id`.

`SamplingRunHook` selects a percentage of requests and forwards the complete
event sequence for each selected run to one sink:

```python
from app.prompt_ninja import (
    OpenRouterPromptClient,
    PromptNinja,
    PromptRuntimeOptions,
    SamplingRunHook,
)

prompt = PromptNinja.from_file("prompts/greeting.prompt.toml")


async def queue_for_review(event):
    payload = event.model_dump()
    payload["system"] = redact(payload["system"])
    payload["user"] = redact(payload["user"])
    await review_queue.publish(payload)


async def run_observed():
    client = OpenRouterPromptClient()
    hook = SamplingRunHook(queue_for_review, sample_rate=0.05)

    try:
        prepared = prompt.prepare({"name": "Ada"})
        return await client.execute(
            prompt,
            prepared,
            runtime=PromptRuntimeOptions(model="google/gemini-2.5-flash"),
            hooks=(hook,),
        )
    finally:
        await client.aclose()
```

Hook failures are isolated and never fail the model request. Because hooks can
receive real inputs and outputs, redact or encrypt sensitive data before it is
stored.

### Controlled auto-fixing

Hooks supply the evidence for an auto-fix system; they do not edit production
prompts themselves. A safe loop is:

1. Sample and redact real request, response, and error events.
2. Detect a recurring failure through human review, a business rule, or an
   evaluator prompt.
3. Copy the promoted prompt to a branch or staging path and give the updater
   the failure evidence as feedback.
4. Validate the candidate and run both its embedded tests and the broader
   regression suite.
5. Compare the candidate with the current version and require an approval or
   promotion policy.
6. Retain the previous version and a tested rollback path.

This creates a measured improvement loop without allowing an unreviewed model
response to redefine production behavior.

## OpenRouter configuration

The application uses OpenRouter's OpenAI-compatible Responses API. Configure it
in `backend/.env`:

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_DEFAULT_MODEL=google/gemini-2.5-flash
OPENROUTER_ALLOWED_MODELS=
OPENROUTER_HTTP_REFERER=http://localhost:5173
OPENROUTER_APP_TITLE=Prompt Ninja
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:5174
```

- `OPENROUTER_DEFAULT_MODEL` is selected until the live catalog loads.
- `OPENROUTER_ALLOWED_MODELS` optionally restricts the model catalog with a
  comma-separated allowlist.
- `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` provide optional
  OpenRouter attribution.
- `FRONTEND_ORIGINS` controls allowed browser origins for the API.

The UI loads the live catalog from OpenRouter, displays input and output pricing
per million tokens, and limits selection to models that advertise structured
output support.

## Development checks

Run the backend suite from the repository root:

```bash
uv run --no-editable --directory backend pytest
```

The dependency graphs are pinned by `backend/uv.lock` and
`frontend/package-lock.json`.

## Platform direction and safety

Prompt Ninja already provides the artifacts, tests, update mechanism, and
runtime evidence needed for continuous prompt quality control. The next layer
is scheduled or event-driven evaluation: detect drift as models, data, business
rules, and user expectations change, then propose a candidate update.

Promotion should remain a controlled release decision. Fully autonomous updates
are appropriate only after strong evaluation, approval policy, observability,
versioning, and rollback are in place.
