# Prompt Ninja · Board of Prompts

Prompt Ninja is a small FastAPI + static React/Chakra UI app for turning a
free-flow request into a production-ready prompt through a visible Board of
Prompts review. Users can attach up to five reference files, address them as
`File #1` through `File #5`, review and edit an LLM-enhanced structured brief,
and explicitly confirm it before the board starts.

## Run locally

```bash
cd backend
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend is a Vite + React + Chakra UI npm project. `frontend/package-lock.json` and `backend/uv.lock` pin the install graphs for reproducible local runs.

The intake enhancer and every board stage—requirements, all three creators,
final synthesis, and validation—use TOML-defined LLM prompts and require
`OPENAI_API_KEY`; without it, the API reports an error rather than creating
local fallback prompts. The FastAPI service is stateless: uploaded files are
read in memory and are not persisted.

Each board prompt declares its own default model in TOML. The UI starts with
`OPENAI_DEFAULT_MODEL` selected for each configurable stage and only offers the
comma-separated models in `OPENAI_ALLOWED_MODELS`; add a model there only after
confirming that the API project can invoke it. Restart the backend after
changing either setting.

## Prompt Ninja prompt specifications

Prompt definitions live in `backend/prompts` and use the `*.prompt.toml` extension. [requirements.prompt.toml](backend/prompts/requirements.prompt.toml), [creator-1.prompt.toml](backend/prompts/creator-1.prompt.toml), [creator-2.prompt.toml](backend/prompts/creator-2.prompt.toml), [creator-3.prompt.toml](backend/prompts/creator-3.prompt.toml), and [judge.prompt.toml](backend/prompts/judge.prompt.toml) define the board's provider-backed stages. [greeting.prompt.toml](backend/prompts/greeting.prompt.toml) remains a minimal executable example of the `1.0` specification.

`PromptNinja` validates a prompt before it reaches a model, resolves its declared
Pydantic output class, renders declared variables, and executes embedded tests.
Structured response validation is performed once by that Pydantic class through
the OpenAI parser.

Every `prompt.used_in` entry must be a repository-relative consumer file path such as `backend/app/agents.py`, not a package or application name.

The canonical TOML `output` value is `String`, `BigInt`, or a dotted import
path to a Pydantic `BaseModel`, for example:

```toml
output = "app.prompt_compiler.CompiledPromptResult"
```

Prompt Ninja resolves model paths with Python import and attribute lookup, then
passes the class directly to `responses.parse(text_format=...)`. It reads the
validated object from `response.output_parsed`; it does not normalize alternate
model-generated schemas.

```python
from app.prompt_ninja import PromptNinja

prompt = PromptNinja.from_file("prompts/greeting.prompt.toml")

def executor(prepared):
    # Send prepared.system and prepared.user to prepared.provider/prepared.model.
    return '{"result": "Hello, Ada!"}'

assert prompt.run_tests(executor).passed
```

Run the library checks with:

```bash
cd backend
uv run pytest
```

## CLI

From `backend`, the packaged CLI exposes generation, testing, prompt updates, and the web API:

```bash
prompt-ninja generate --goal "Summarize legal docs into plain English"
prompt-ninja generate # reads goal from prompt-ninja.toml
prompt-ninja test --prompt prompts/my-prompt.prompt.toml
prompt-ninja test-prompts --prompts-dir prompts --prompt-name judge --verbose
prompt-ninja update prompts/judge.prompt.toml "Make echo-trap tests score 0.0"
prompt-ninja validate prompts/judge.prompt.toml
prompt-ninja validate ./prompts --fix
prompt-ninja ui --port 8000
```

`generate` writes a validated `*.prompt.toml` artifact. `test` and `test-prompts` execute embedded test cases using the configured LLM, then score the expected-versus-actual result with an LLM judge; both require `OPENAI_API_KEY`. `update` validates the model's replacement before writing it and keeps a `.bak` copy.

`validate` checks a file or directory of prompt files, imports every declared
output-model path, verifies that its attribute exists and subclasses Pydantic
`BaseModel`, and prints actionable errors for CI. `validate --fix` gives those
errors to the LLM updater, requires a valid replacement output declaration, and
preserves each original as a `.bak` file.

### Semantic prompt tests

Define each test as a `[[tests]]` TOML table. `input` values must match declared `[[variables]]`; `expected_output` is a natural-language contract that the LLM judge scores semantically. Tests pass when the score meets `[testing].pass_threshold`, which defaults to `0.95`.

```toml
[testing]
pass_threshold = 0.95

[[tests]]
name = "French translation contract"
expected_output = """
A JSON array of input/expected_output pairs that translates English to French.
"""

[tests.input]
goal = "Translate English to French"
extra = "\n\nKeep summaries under 5 bullets"
```

Use TOML multiline strings (`"""..."""`) for longer inputs or expectations. The CLI also supports the short CI-friendly forms:

```bash
prompt-ninja test-prompts -t ./prompts
prompt-ninja test-prompts -p judge
prompt-ninja test-prompts -t ./prompts -v
```

### Runtime model overrides and auto-fix samples

`PromptNinja` uses the OpenAI Responses API through `OpenAIPromptClient`. Pass `PromptRuntimeOptions` to override the model for one run without changing the checked-in TOML. Sampling and output-length controls are left to the selected model. Attach a `SamplingRunHook` to receive matching request and response events for a sampled set of real runs—an ideal input for a queue-backed auto-fix workflow.

```python
from app.prompt_ninja import OpenAIPromptClient, PromptNinja, PromptRuntimeOptions, SamplingRunHook

prompt = PromptNinja.from_file("prompts/my-prompt.prompt.toml")

async def queue_for_review(event):
    # event contains the rendered system/user messages and eventual output/error.
    await review_queue.publish(event.model_dump())

client = OpenAIPromptClient()
hook = SamplingRunHook(queue_for_review, sample_rate=0.05)
prepared = prompt.prepare({"input": "Example source"})
result = await client.execute(
    prompt,
    prepared,
    runtime=PromptRuntimeOptions(model="gpt-5.6-sol"),
    hooks=(hook,),
)
```

Because hooks receive real prompt inputs and outputs, redact or encrypt sensitive fields before persisting them. Hook failures are isolated and never fail the model run.

When a stage has an application model, pass it as `output_model` and the client returns that validated Pydantic object instead of an untyped dictionary. The Board of Prompts uses this for `RequirementsResult`, so malformed requirements fail at the model boundary rather than being normalized later.
