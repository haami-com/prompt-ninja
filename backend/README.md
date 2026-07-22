# Prompt Ninja

**Let LLMs engineer the prompts. Prompt Ninja makes prompt creation a repeatable engineering process — versioned, validated, and tested in CI, not trial-and-error.**

Describe the behavior you want, and a Board of Prompts — multiple
LLMs drafting, comparing, and judging candidates — produces a
`*.prompt.toml` file with a purpose, model, typed inputs, output contract, and
semantic regression tests baked in.

Instead of scattering critical prompts across source code, chat history, and
one-off docs, every prompt becomes something you can review like a code
change, test before release, and re-validate as models and requirements
change. The question becomes "does this version still satisfy its contract?"
rather than "which wording worked last time?"

## Why

A strong prompt can drift as models, data, or expectations change. Most teams
have no way to notice until a user does. Prompt Ninja treats prompts as
software: stored in version control, validated against a schema, and scored
by an LLM judge against natural-language test cases — so regressions show up
in CI, not production.

- **For engineers:** testable, maintainable application artifacts instead of
  handcrafted text buried in application code.
- **For platform/DevOps:** a control plane around AI behavior — model
  configuration, test automation, versioning, rollout safety, and drift
  monitoring.

## How it works

The **Board of Prompts** is the creation workflow:

1. **Enhance** — an LLM turns a rough request into a structured brief
   (outcome, context, constraints, expected output).
2. **Draft** — three creator models independently draft candidate prompts.
3. **Judge** — a judge model synthesizes the strongest ideas into one prompt.
4. **Validate** — embedded semantic contracts test and compile the final prompt,
   versioned `*.prompt.toml` artifact.

Three ways to use the result:

| Interface | Best for |
| --- | --- |
| **UI** | Describing a desired behavior and watching the Board of Prompts create and evaluate a prompt |
| **CLI** | Generating, validating, testing, updating, and automating prompts in CI |
| **Python API** | Loading versioned prompts in an application, rendering typed inputs, validating outputs, and observing real runs with hooks |

---

# Setup

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Node.js and npm (only for the Board of Prompts UI)
- An OpenRouter API key — needed for generation, execution, semantic tests, and
  `--fix`. **Loading, rendering, and validating prompts work with no key**, so
  you can explore a good part of the project offline.

## Option A — run from this repository

This is the path to use for reviewing the project, and the one every command
below is verified against.

```bash
git clone https://github.com/haami-com/prompt-ninja.git
cd prompt-ninja/backend
uv sync --no-editable
cp .env.example .env
```

Add your key to `backend/.env`:

```dotenv
OPENROUTER_API_KEY=your-key
```

Check the CLI is working:

```bash
uv run --no-editable prompt-ninja --help
```

> **Why `--no-editable`:** the generated console script is unreliable on macOS
> Python environments that ignore hidden editable-install `.pth` files. If you
> activate the virtual environment (`source .venv/bin/activate`), the shorter
> `prompt-ninja ...` form works too, and the rest of this README uses that form
> for readability.

## Option B — install the package into your own project

```bash
pip install prompt-ninja          # CLI + Python API
pip install 'prompt-ninja[server]'  # adds the FastAPI Board of Prompts backend
export OPENROUTER_API_KEY=your-key
```

## Run the UI

The backend serves the API; Vite serves the UI in development. In one terminal:

```bash
cd backend
uv run --no-editable uvicorn prompt_ninja.main:app --reload --host 127.0.0.1 --port 8000
```

In another:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints. The **Board** page takes a plain-language request
(plus up to five optional reference files) and runs every stage live. The
**Hooks** page then shows the quality scores, judge rationales, token counts,
and cost estimates recorded from that run. Uploaded files are processed in
memory and are never persisted.

## Run the tests

```bash
uv run --no-editable --directory backend pytest
```

99 deterministic tests, no API key and no network required — provider calls are
injected as fake executors. See
[Two layers of testing](#decision-4-two-layers-of-testing) for why.

---

# Try it with the included sample

[`examples/`](examples/) contains a ready-made artifact so you can see the
format and the workflow without generating anything first: a support-ticket
triage prompt, its Pydantic output contract, and six sample tickets.

| File | What it is |
| --- | --- |
| [`examples/prompts/ticket-triage.prompt.toml`](examples/prompts/ticket-triage.prompt.toml) | The artifact: instructions, model, typed inputs, output contract, and four semantic tests |
| [`examples/prompts/ticket_triage_models.py`](examples/prompts/ticket_triage_models.py) | The Pydantic model every response must validate against |
| [`examples/sample-tickets.json`](examples/sample-tickets.json) | Six sample tickets, including two designed to be tempting to misclassify |
| [`examples/run_triage.py`](examples/run_triage.py) | Loads the artifact and classifies all six |

Run these from the `examples/` directory, so the artifact's
`prompts.ticket_triage_models.TicketTriage` output path is importable:

```bash
cd examples
```

**Validate the artifact — no API key needed.** This checks the TOML schema,
variable declarations, defaults, template references, model settings, and that
the output model actually imports:

```bash
prompt-ninja validate prompts
# VALID prompts/ticket-triage.prompt.toml
```

**Run the semantic tests.** These call the model and score each response
against a natural-language expectation (requires `OPENROUTER_API_KEY`):

```bash
prompt-ninja test --prompt prompts/ticket-triage.prompt.toml --verbose
```

Two of the four cases exist to pin down failures that are easy to regress into:

- `never invents an affected component` — the ticket names no system, so
  `affected_component` must be exactly `unknown` rather than a plausible guess.
- `does not escalate on tone alone` — an angry ticket with no concrete impact
  must not escalate. Severity should track evidence, not volume.

**Classify the sample tickets from Python:**

```bash
python run_triage.py
```

Each `result` comes back as a validated `TicketTriage` instance, not raw JSON —
a malformed response, a bad enum value, or a missing field fails at the
contract boundary instead of reaching your code.

**Then change something and watch a test catch it.** Delete the
"Never invent an affected component" paragraph from the `[prompt]` system block
and re-run `prompt-ninja test`. That is the whole point of the format: the
expectation survives the edit.

---

# From a description to a tested prompt

### 1. Describe what the prompt should do

`generate` turns a plain-language goal into a versioned `*.prompt.toml` file
with model settings, typed inputs, an output contract, and semantic tests:

```bash
mkdir -p prompts
prompt-ninja generate \
  --goal "Turn release notes into a concise customer update" \
  --output prompts/customer-update.prompt.toml
```

Open the generated file and review it like source code. The instructions live
under `[prompt]`; `[[variables]]` defines the inputs; and each `[[tests]]` block
defines behavior that must remain stable.

When the requested output is structured JSON, generation also creates an
importable companion module such as `prompts/customer_update_models.py` and
sets `metadata.output` to its Pydantic model path. Run the command from the
project root and keep the generated TOML, model module, and `__init__.py`
together. If no consumer file is known yet, `metadata.used_by` is `[]`.
Generated structured test cases store `expected_output` as an inline TOML
object that validates against the same Pydantic model. Text-output tests keep a
natural-language semantic expectation.

### 2. Add or change expectations

Add semantic cases directly to the artifact without prescribing exact output
wording. For example:

```toml
[[tests]]
name = "avoids internal terminology"
variable.release_notes = "The API gateway migration begins July 30."
expected_output = "A customer-friendly update that preserves the date and does not use internal engineering terminology."
```

### 3. Validate or repair the artifact

Validation is local and checks the TOML schema, variables, defaults, model
settings, output contract, and importable Pydantic paths:

```bash
prompt-ninja validate prompts/customer-update.prompt.toml
prompt-ninja validate prompts
```

If validation fails, `--fix` asks Prompt Ninja's versioned repair prompt to
produce a corrected artifact. The replacement must validate before it is
written, and the original is preserved as `.bak`:

```bash
prompt-ninja validate prompts/customer-update.prompt.toml --fix
```

### 4. Run semantic regression tests

Run one prompt, one named expectation, or every prompt in the directory:

```bash
prompt-ninja test --prompt prompts/customer-update.prompt.toml --verbose
prompt-ninja test \
  --prompt prompts/customer-update.prompt.toml \
  --test-name "avoids internal terminology"
prompt-ninja test-prompts --prompts-dir prompts --plain
```

Test commands exit non-zero on failure, and `--plain` produces CI-friendly
output. Failed cases always show the complete judge rationale and suggest a
prompt change, a test-case change, or both when appropriate.

### 5. Update the prompt from feedback

Ask Prompt Ninja to revise the prompt implementation without manually rewriting
it. Tests and the pass threshold are protected during the rewrite. The candidate
must pass the complete preserved contract before it replaces the artifact:

```bash
prompt-ninja update \
  prompts/customer-update.prompt.toml \
  "Preserve dates and avoid internal engineering terminology"
```

The command prints complete test diagnostics. On success it writes the candidate
and backs up the previous version; on failure the original file is untouched.

### 6. Load it in your application

```python
from prompt_ninja import PromptCollection, PromptNinja

prompt = PromptNinja.from_file("prompts/customer-update.prompt.toml")
result = await prompt.run_openrouter({"release_notes": "Search launches July 30."})

prompts = PromptCollection(dir="prompts")
customer_update = prompts.customer_update
```

Run `prompt-ninja COMMAND --help` for every CLI option, or see the
[complete command reference](docs/reference.md#cli-reference).

---

# How this was built: Codex and GPT-5.6

Two distinct things are worth separating: **GPT-5.6 is a component of the
product**, and **Codex is how the product was implemented**.

## GPT-5.6 inside the product

The Board of Prompts is deliberately multi-vendor. Three creator models draft
independently, because three samples from one model converge on one house style
and give the judge little to choose between. GPT-5.6 holds the two roles where
that matters most — one of the drafting seats, and both judging seats:

| Stage | Model | Where it is configured |
| --- | --- | --- |
| Requirements / brief | `google/gemini-2.5-flash` | [`requirements.prompt.toml`](backend/prompt_ninja/prompts/requirements.prompt.toml) |
| Creator 1 | **`openai/gpt-5.6-luna`** | [`creator-1.prompt.toml`](backend/prompt_ninja/prompts/creator-1.prompt.toml) |
| Creator 2 | `deepseek/deepseek-v4-flash` | [`creator-2.prompt.toml`](backend/prompt_ninja/prompts/creator-2.prompt.toml) |
| Creator 3 | `google/gemini-3.5-flash` | [`creator-3.prompt.toml`](backend/prompt_ninja/prompts/creator-3.prompt.toml) |
| Judge / synthesis | **`openai/gpt-5.6-terra`** | [`judge.prompt.toml`](backend/prompt_ninja/prompts/judge.prompt.toml) |
| Runtime quality hook | **`openai/gpt-5.6-terra`** | [`sampled-run-judge.prompt.toml`](backend/prompt_ninja/prompts/sampled-run-judge.prompt.toml) |

Judging is the hardest job on the board. It has to read three full candidate
prompts, work out which parts of each actually serve the brief, and merge them
into one artifact that still satisfies a schema — and the same reasoning runs
again at runtime, where the quality hook scores live responses and has to
justify the score. GPT-5.6 Terra is the default for both
([`model_config.py`](backend/prompt_ninja/model_config.py)). Every model is
overridable per stage in the UI and per prompt in the TOML; these are defaults,
not hard-coding.

## Codex as the implementation partner

Codex generated most of the implementation across both sides of the project.
On the backend: the TOML models, validation, prompt runtime, CLI, semantic test
runner, update safeguards, hooks, and FastAPI endpoints. On the frontend: the
React/Vite application, including the Board of Prompts, the contract workspace,
the TOML export flow, and the Hooks page.

**Where it accelerated the work most:** the mechanical breadth. Five CLI
commands with consistent Rich output, a variable system covering ten-plus types
with four render filters, ten API endpoints plus a health check, and a React
workspace — implemented in parallel with the design work rather than after it.
Codex was most effective when a task had a clear behavioral goal and a focused
validation step attached to it.

**The loop was deliberate, not one-shot:** define the next capability, describe
its behavior and constraints, implement it with Codex, run the tests or the
frontend build, correct the approach where it diverged, tighten the
requirements, then move to the next layer. Product direction and implementation
order were decided first, not delegated.

## Key decisions

### Decision 1: the expectation is the source of truth

The inversion the whole project rests on. You describe the behavior you want;
the LLM engineers the prompt text. That makes prompt wording an *implementation*
— something that can be regenerated, judged, and safely rewritten — while the
expectation stays stable in `[[tests]]`. Every other decision follows from this
one.

### Decision 2: prompt policy in TOML, guarantees in Python

**This was the hardest thing to communicate to Codex.** Prompt Ninja is a tool
that uses prompts to generate, test, repair, and update other prompts, and those
internal prompts should themselves use the Prompt Ninja format. Codex would
repeatedly flatten them back into ordinary Python string constants, or migrate
semantic behavior out of a prompt artifact and into Python control flow —
locally reasonable, but it dissolved the entire premise.

The boundary had to be restated constantly: **semantic policy belongs in
versioned, tested `*.prompt.toml` files; deterministic guarantees — schema
validation, test preservation, safe file promotion — belong in Python.** The
project dogfoods the result. Every prompt the Board, compiler, updater, repair
flow, and judges use is itself a `*.prompt.toml` artifact with declared inputs,
an output contract, and tests: [`backend/prompt_ninja/prompts/`](backend/prompt_ninja/prompts/).

### Decision 3: one conversion pipeline for typed values

Simple string substitution works until a value is a boolean, number, date, list,
dict, or Pydantic model — and optional-vs-required-vs-default pressures push
conditional logic into every caller. The tricky part was moving safely between
serialized and real objects in both directions: a date arrives as an ISO string,
a list carries typed items, a dict represents a Pydantic input, and a response
starts as JSON text but must end as the declared output model.

It resolved into a single pipeline: parse the TOML, coerce every input to its
declared Python type, render it consistently (with explicit `str`, `repr`,
`json`, and `csv` filters), parse the response, and validate it into the
declared Pydantic object before returning. Unknown variables, invalid defaults,
wrongly typed fixtures, and schema mismatches now fail at validation time
instead of surprising you in production.

### Decision 4: two layers of testing

Model output is nondeterministic. Exact string assertions are too brittle;
mocking everything only proves the Python wiring works. So there are two layers:

- **Deterministic unit tests** inject fake provider executors to simulate
  success, errors, retries, structured responses, and hook events. They cover
  parsing, coercion, rendering, validation, and control flow with no network
  calls — this is the 99-test suite that runs in CI.
- **Embedded `[[tests]]` cases** run the real prompt and judge the response
  against a natural-language expectation. Pydantic validates the structure
  first; the judge then evaluates meaning and constraints.

This split is also what makes protected updates possible: a candidate prompt
runs the full behavioral contract before it is allowed to replace the current
version.

### Decision 5: choose the stack before generating

The clearest lesson from building this way. At the start of a generated project
you have to fix the packages, frameworks, versions, and tools you expect. Left
open, Codex will pick a library that doesn't fit, use an outdated API,
hand-roll something a proven package already does, or choose an approach that
works at first and becomes hard to maintain.

Naming Pydantic for contracts, FastAPI for the API, Click for the CLI,
OpenRouter through an OpenAI-compatible client, and React + Vite + Chakra UI for
the frontend gave Codex a consistent foundation to build against. Specifying
*ownership boundaries* mattered as much as naming libraries — see Decision 2.

The summary: treat Codex as an implementation partner, not an automatic
architect. Decide the stack and constraints first, hand over work in a
deliberate order, review what comes back, and run focused tests or builds after
every meaningful change.

---

# Deploy to Fly.io

The root [`Dockerfile`](Dockerfile) builds the Vite frontend, installs the
backend with a frozen `uv sync` from [`backend/uv.lock`](backend/uv.lock), and
serves both from one non-root FastAPI container. [`fly.toml`](fly.toml)
configures port `8080`, HTTPS, automatic machine start/stop, and `/health`
checks.

Fly app names are globally unique. Change the `app` value in `fly.toml` if
`prompt-ninja` is unavailable, then create the app, set the provider key, and
deploy from the repository root:

```bash
fly apps create prompt-ninja
fly secrets set OPENROUTER_API_KEY=your-key
fly deploy
```

The hosted instance at [prompt-ninja.fly.dev](https://prompt-ninja.fly.dev)
runs exactly this.

The secret is stored by Fly and is not built into the image. To verify the
same image locally:

```bash
docker build -t prompt-ninja .
docker run --rm -p 8080:8080 \
  -e OPENROUTER_API_KEY=your-key \
  prompt-ninja
```

Open `http://localhost:8080`; the API remains available under `/api` and its
health check at `/health`.

# Features

- LLM-assisted UI to draft prompts from a plain request plus reference files.
- Multiple competing prompt candidates, synthesized into one tested result.
- Prompts stored as structured, human-readable `*.prompt.toml` files with
  models, typed variables, output contracts, and semantic tests alongside
  the prompt.
- Validation of TOML structure, template variables, types, defaults, model
  paths, and output declarations before a prompt ever runs.
- Semantic regression testing via an LLM judge, with thresholds, targeted
  test selection, and CI-friendly output.
- Natural-language prompt updates that preserve semantic contracts, test the
  candidate before promotion, and keep a `.bak` of the previous version.
- A browser contract workspace for editing expectations, running every test,
  inspecting full failures and suggestions, updating the prompt, and rerunning.
- Live OpenRouter model browsing with pricing and structured-output filtering.
- Extensible runtime hooks, demonstrated with non-blocking quality evaluation
  for Creator 1 and token/cost telemetry for Creator 2.

# Project layout

```
backend/prompt_ninja/        the installable package
  core.py                    spec models, rendering, validation, runtime
  cli.py                     the five prompt-ninja commands
  agents.py                  the Board of Prompts pipeline
  main.py                    FastAPI API
  hooks/                     quality-evaluation and usage/cost hooks
  prompts/                   the prompts Prompt Ninja uses on itself
backend/tests/               99 deterministic tests, no network
examples/                    runnable sample artifact and data
frontend/src/                React + Vite Board and Hooks pages
docs/reference.md            full specification and API reference
```

# Platform direction

Prompt Ninja already provides the artifacts and tests needed for safer prompt
change. The next layer is continuous quality control: run those contracts
against real behavior over time, detect drift, and produce a candidate update
when quality degrades. Runtime hooks already capture quality scores, failure
rationales, inputs, outputs, and usage — the evidence a self-improving loop
needs. Promotion stays a controlled release decision, not an unreviewed model
action — see [`docs/reference.md`](docs/reference.md#controlled-auto-fixing)
for the proposed safe-update loop.

# Further reading

[`docs/reference.md`](docs/reference.md) covers the full `prompt.toml`
specification, variable types and rendering, output contracts, the Python API
(`PromptNinja`, `PromptCollection`, runtime overrides), the complete CLI
reference, observability hooks, and OpenRouter configuration.
