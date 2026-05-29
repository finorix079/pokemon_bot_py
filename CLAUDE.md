# CLAUDE.md

Orientation for Claude Code working in this repo. Read this before exploring.

## What this repo is

`pokemon-bot` is an interactive Python REPL chatbot over the [PokéAPI](https://pokeapi.co/),
ported from the TypeScript `/api/chat-stream` route in `open-react-template`. Entry point:
`python -m pokemon_bot` (defined in `pokemon_bot/__main__.py`).

The pipeline is a multi-stage LLM workflow (query refinement → planner → validator →
iterative executor → streamed final answer) backed by three model providers:

- **Anthropic Claude** (`claude-sonnet-4-5-20250929`) — planner, validator, executor synthesis, query refinement, placeholder resolver, final-answer streaming
- **Kimi** (`kimi-k2-turbo-preview`) — per-step useful-data extraction, intent classifier
- **OpenAI SDK** — kept only as the HTTP wrapper for the Kimi client (`KIMI_BASE_URL`); no OpenAI model is actually called

All prompts in `pokemon_bot/prompts/` are byte-identical to the TS source; tests in
`tests/test_executor_prompt.py` anchor that. Do not "improve" prompts without explicit
instruction — see the migration note in `README.md` for the one allowed Claude-tuning patch.

## Layout (high level)

```
pokemon_bot/
├── __main__.py        # REPL boot, observability init, start_trace/end_trace per turn
├── chat/              # handler, planner, executor, validators, message_utils, session
├── services/          # PokéAPI clients, dynamic API request, RAG retrieval, parameter mapper
├── tools/pokemon_tools.py  # named-tool registry (camelCase: apiService, queryRefinement, searchPokemon, fetchPokemonDetails, searchMove, searchBerry, searchAbility)
├── utils/             # ai_handler (LLM clients), query_refinement, cli_stream
├── schemas/ai.py      # pydantic models ported from Zod
└── prompts/           # bundled prompt assets — DO NOT edit verbatim text
tests/                 # pytest, asyncio_mode=auto, mocks all three LLM providers
```

`README.md` has the full file-by-file TS → Python provenance table.

## Conventions specific to this repo

- **Tool names are camelCase, not snake_case.** They match the TS `ed_tools.ts` exports
  exactly because the agent_tools map and planner LLM output look them up by string.
  Do not rename.
- **Tracing is opt-in.** With `ELASTICDASH_SERVER_URL` unset, `init_observability`,
  `start_trace`, `end_trace`, and `wrap_tool` all become no-ops. Don't assume tracing
  side effects in tests.
- **`openai_chat_completion(...)` is a dispatcher that forwards to Claude.** The name is
  preserved for historical call sites. Don't rewrite call sites to call the Anthropic
  client directly without a reason.
- **Sessions live in memory only** (`pending_plans` dict). They do not survive process exit.
- **The "extra stats" bug in `generate_final_answer` is intentional.** It is carried over
  from the TS source per the verbatim-prompt requirement. Do not "fix" it unless asked.

## Running and testing

```bash
.venv/bin/python -m pokemon_bot          # run REPL
.venv/bin/python -m pytest               # run smoke + prompt-anchor tests
```

Tests mock all three LLM providers and the PokéAPI HTTP layer; they don't need real keys.

## ElasticDash MCP — trace investigation workflow

This repo ships an MCP config (`.mcp.json`) for the `elasticdash-mcp` server. When Claude
Code is invoked in this directory, the `mcp__elasticdash-mcp__*` tools become available:
`search_traces`, `get_trace_details`, `rerun_step`, `get_recent_traces`.

Use these when the user describes a runtime issue with the bot ("the agent gave a wrong
answer when I asked X", "a user reported Y", "it returned too many stats"). The MCP server
itself instructs you to read `ed_tools.{ts,js,py}` and `ed_workflows.{ts,js,py}` first to
learn valid tool names — for this repo those names are the camelCase exports in
`pokemon_bot/tools/pokemon_tools.py` plus the workflow name `pokemon_chat` (registered in
`pokemon_bot/__main__.py:_start_trace_safe`).

### Required: verify the trace by rerunning it step-by-step

When you have searched traces and believe you have identified the right one (typically
after `search_traces` → `get_trace_details`), **you MUST verify reproducibility before
acting on it**. The agent code, prompts, tools, or upstream APIs may have changed since
the trace was recorded, so a "matching" trace is not enough — you have to confirm it
still reproduces locally.

Procedure:

1. From `get_trace_details`, enumerate each step (tool calls + LLM calls) in order.
2. For each step, call `mcp__elasticdash-mcp__rerun_step` with the captured input.
3. Collect `recorded_output` vs `rerun_output` per step.
4. Print a comparison table to the user before drawing any conclusion. Required columns:

   | # | Step (tool/LLM) | Input summary | Recorded output | Rerun output | Match? |
   |---|---|---|---|---|---|

   - `Match?` is one of `✓ identical`, `≈ semantically equal` (note why), or `✗ diverged`.
   - Keep `Input summary` / outputs short — truncate long JSON to the first ~120 chars and
     note the truncation. Full payloads stay in tool results, not the table.
   - If any step has `✗ diverged`, that is the suspect step. Flag it explicitly and stop
     rather than continuing as if the trace is still ground truth.

5. Only after the table is shown should you propose a fix, file a bug, or answer the user's
   original question based on the trace.

Do not skip the rerun-and-compare step even when the trace looks obviously right — the
whole point of the check is to catch silent drift between recorded and current behavior.

If the user explicitly says "skip rerun" or "just analyze the trace", honor that.

## Git hygiene

- Main branch is `main`. There are no protected-branch or CI rules beyond local pytest.
- The user's git identity is `TerryAtRocketVentures`.
- Do not commit `.env`. `.env.example` is the template that lives in the repo.
