# Reference

Detailed specification, Python API, CLI, and observability docs for Prompt
Ninja. Start with the top-level [README](../README.md) for the pitch and
quick start; this file is for building against the library or writing new
`*.prompt.toml` files.

## `prompt.toml` specification

Prompt definitions use the `*.prompt.toml` extension. The checked-in examples
live in [`backend/prompts`](../backend/prompts); the
[`greeting.prompt.toml`](../backend/prompts/greeting.prompt.toml) file is the
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
- A dotted Pydantic model path such as `myapp.models.Person`

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
type = "myapp.models.Person"
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
  `myapp.models.GreetingResult`

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
when the score meets `testing.pass_threshold`. A failed case always prints the
complete rationale and can suggest a prompt change, a test-case change, or both.
The judge receives the rendered prompt and test input so those suggestions are
grounded in the executed case.

Each prompt execution and judge request gets up to three total API attempts for
transient failures such as connection errors, timeouts, rate limits, and server
errors. A completed response that fails its semantic threshold is not retried.

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

from prompt_ninja import PromptNinja

prompt = PromptNinja.from_file("prompts/greeting.prompt.toml")


async def main():
    result = await prompt.run_openrouter({"name": "Ada", "tone": "friendly"})
    print(result.result)


asyncio.run(main())
```

The declared `myapp.models.GreetingResult` output means `result` is already a
validated Pydantic object.

### Load a prompt directory once

Use `PromptCollection` when an application owns several prompts:

```python
from prompt_ninja import PromptCollection

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
from prompt_ninja import PromptRuntimeOptions


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

### Examples

```bash
prompt-ninja generate --goal "Summarize legal documents in plain English"
prompt-ninja generate --config prompt-ninja.toml --output prompts/legal-summary.prompt.toml
prompt-ninja test --prompt prompts/greeting.prompt.toml --verbose
prompt-ninja test-prompts --prompts-dir prompts --prompt-name judge --plain
prompt-ninja update prompts/judge.prompt.toml "Reject responses that invent facts"
prompt-ninja validate prompts
prompt-ninja validate prompts/judge.prompt.toml --fix
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
from prompt_ninja import (
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

The first application hook performs deterministic online evaluation on every
successful call:

```python
from app.hooks import EveryNRunEvalHook, OpenRouterRunJudge

evaluations = []  # Replace with a database, queue, or observability sink.
eval_hook = EveryNRunEvalHook(
    judge=OpenRouterRunJudge(),
    sink=evaluations.append,
  every=1,
)

result = await client.execute(prompt, prepared, hooks=(eval_hook,))
```

Each `RunEvaluation` contains the rendered system prompt, input, output, model,
run ID, a 0–1 score, and the judge rationale. Only successful responses count
toward the interval; request and error events are ignored. Judging runs in a
tracked background task, so evaluation does not delay the original prompt
response. Call `await eval_hook.drain()` when a test or shutdown path must wait
for pending evaluations to finish.

The demo also attaches `TokenUsageCostHook` to Creator 2. Successful response
events include normalized `input_tokens`, `output_tokens`, and `total_tokens`
from the OpenAI Responses API. The hook combines those counts with OpenRouter's
per-token model catalogue prices and stores a per-run cost estimate without
making another LLM call or delaying the original response. The Hooks page and
`GET /api/hooks` expose both examples together.

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

The UI loads OpenRouter's live text-model catalog and displays input and output
pricing per million tokens. The initial board uses an OpenAI creator, a
DeepSeek creator, a Gemini creator, and GPT-5.6 Terra as the judge.
