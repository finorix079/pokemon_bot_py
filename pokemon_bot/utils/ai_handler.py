"""utils/ai_handler.py — LLM clients + agentic planner/executor.

Ported from open-react-template/utils/aiHandler.ts (770 lines).

What's preserved:
- The Kimi (`kimi_chat_completion`) and Anthropic
  (`anthropic_chat_completion`) wrappers with the same kwargs
  (`messages`, `model`, `temperature`, `max_tokens`, `system_prompt`,
  `session_id`) and the same default models.
- The `openai_chat_completion` public name — kept as a dispatcher that
  always forwards to Claude so every existing call site keeps working
  unchanged. See `.temp/plan.md` "Switch chatbot from OpenAI to Claude"
  for migration notes. The OpenAI client and the `AI_PROVIDER` env-var
  branch are gone.
- The `agent_tools` registry (apiService + queryRefinement + the five
  PokéAPI tools), with the same descriptions.
- The mid-trace replay machinery: `extract_task_outputs`,
  `resolve_task_input` (incl. `{$task.N.output.path}` placeholders and the
  `{$ref: "taskId.output.path"}` object form), `serialize_agent_state`,
  `deserialize_agent_state`, `executor_agent`, `planner_agent`,
  `resume_agent_from_trace`, `run_agentic_flow`.

What's intentionally dropped:
- ElasticDash `wrapAI` telemetry — replaced with a passthrough decorator.
- Langfuse OTel SDK init + `observeOpenAI` wrapper — the CLI has no
  observability backend.
- `LangfuseSpan` / `LangfuseObservation` arguments — replaced with `None`
  sentinels so call sites keep their shape without needing tracing.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional, TypeVar

from .. import config as _config

T = TypeVar("T", bound=Callable[..., Any])

# ============================================================
# wrap_ai — ElasticDash LLM-call telemetry
# ============================================================
#
# Two complementary mechanisms cover the LLM surface:
#
# 1. **httpx auto-interception** (preferred). `init_observability` in
#    `pokemon_bot/__main__.py` installs ElasticDash's httpx interceptor,
#    which captures every request matching its provider heuristics
#    (`/v1/chat/completions`, `anthropic.com`, `/v1/messages`, etc.).
#    The Python `anthropic` and `openai` SDKs both ride on httpx, so
#    `anthropic_chat_completion`, `kimi_chat_completion` (OpenAI client
#    against the Moonshot endpoint), and `anthropic_stream_text` are
#    all auto-recorded. No manual decorator needed.
#
# 2. **Explicit `wrap_ai`** (kept for parity and for any future LLM
#    client that bypasses httpx). The helper below delegates to
#    `elasticdash_sdk.interceptors.wrap_ai`, which records the full
#    function-level input/output pair as a single `ai` WorkflowEvent.
#    Use it only for providers the httpx interceptor doesn't cover —
#    otherwise the dashboard will see a duplicate event per LLM call.


try:
    from elasticdash_sdk.interceptors import wrap_ai as _ed_wrap_ai  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — SDK is a runtime dep but stay resilient
    _ed_wrap_ai = None  # type: ignore[assignment]


def wrap_ai(name: str, fn: T, options: Optional[dict[str, str]] = None) -> T:
    """Record an LLM call as an `ai` event in the active trace.

    Forwards to `elasticdash_sdk.interceptors.wrap_ai`. `options` carries
    optional provider/model metadata — only `provider` is used by the SDK;
    `model` is informational and the recorded event keys on `name` instead.
    Falls back to a passthrough if the SDK is missing.
    """
    if _ed_wrap_ai is None:
        return fn
    provider = (options or {}).get("provider")
    return _ed_wrap_ai(name, fn, provider=provider)  # type: ignore[return-value]


# ============================================================
# Agent types (dataclasses replacing TS interfaces)
# ============================================================


TaskStatus = Literal["pending", "in-progress", "completed", "failed"]
PlanStatus = Literal["planning", "executing", "completed", "failed", "paused"]


@dataclass
class AgentTask:
    """Single step in an agent's execution plan.

    Ported from TS `AgentTask`. `tool` is a string key into `agent_tools`.
    `input` may contain `{$task.N.output[.path]}` placeholders.
    """

    id: str
    description: str
    tool: str
    input: Any
    status: TaskStatus = "pending"
    output: Any = None
    startedAt: Optional[int] = None
    completedAt: Optional[int] = None
    error: Optional[str] = None


@dataclass
class AgentPlan:
    """Full agent execution plan."""

    id: str
    goal: str
    tasks: list[AgentTask]
    status: PlanStatus = "planning"
    currentTaskIndex: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEvent:
    """A workflow trace event captured during agent execution."""

    taskId: str
    toolName: str
    input: Any
    output: Any
    status: Literal["completed", "failed"]
    startedAt: Optional[int] = None
    completedAt: Optional[int] = None
    error: Optional[str] = None
    agentTaskId: Optional[str] = None
    agentTaskIndex: Optional[int] = None


@dataclass
class AgentState:
    """JSON-serialisable agent state for mid-trace resumption."""

    plan: AgentPlan
    trace: list[WorkflowEvent] = field(default_factory=list)
    resumeFromTaskIndex: int = 0


# ============================================================
# Agent tool registry (lazy — avoids circular import with pokemon_tools)
# ============================================================


@dataclass
class AgentTool:
    """One entry in the `agent_tools` registry."""

    name: str
    execute: Callable[[Any, Any], Any]
    """Callable signature: `execute(input, parent_obs)` — parent_obs is always None in the CLI port."""

    description: str = ""


_agent_tools_cache: Optional[dict[str, AgentTool]] = None


def _build_agent_tools() -> dict[str, AgentTool]:
    """Build the agent tool registry. Lazily imports `pokemon_tools` to
    avoid a circular import (`pokemon_tools` may import other utils)."""
    # Lazy import so that `from pokemon_bot.utils.ai_handler import ...`
    # doesn't pull in the entire tool stack on module load.
    from ..tools import pokemon_tools as pt

    def _wrap(tool_name: str, fn: Callable[[Any], Any], description: str) -> AgentTool:
        async def _execute(input_: Any, _parent_obs: Any = None) -> Any:
            return await pt.maybe_await(fn(input_))

        return AgentTool(name=tool_name, execute=_execute, description=description)

    return {
        "apiService": _wrap("apiService", pt.apiService, "Dynamic API request tool"),
        "queryRefinement": _wrap(
            "queryRefinement", pt.queryRefinement, "Refine and clarify user queries"
        ),
        "searchPokemon": _wrap(
            "searchPokemon",
            pt.searchPokemon,
            "Search Pokémon by name or list by page via PokéAPI",
        ),
        "fetchPokemonDetails": _wrap(
            "fetchPokemonDetails",
            pt.fetchPokemonDetails,
            "Fetch full Pokémon details (stats, types, abilities, moves) via PokéAPI",
        ),
        "searchMove": _wrap(
            "searchMove",
            pt.searchMove,
            "Search moves by name or list by page via PokéAPI",
        ),
        "searchBerry": _wrap(
            "searchBerry",
            pt.searchBerry,
            "Search berries by name or list by page via PokéAPI",
        ),
        "searchAbility": _wrap(
            "searchAbility",
            pt.searchAbility,
            "Search abilities by name or list by page via PokéAPI",
        ),
    }


def get_agent_tools() -> dict[str, AgentTool]:
    """Return the agent tool registry, building it on first access."""
    global _agent_tools_cache
    if _agent_tools_cache is None:
        _agent_tools_cache = _build_agent_tools()
    return _agent_tools_cache


# ============================================================
# Mid-trace replay utilities
# ============================================================


def extract_task_outputs(plan: AgentPlan) -> dict[str, Any]:
    """Extract all task outputs into a flat map keyed by both zero-based
    index (as a string) and task id. Mirrors TS `extractTaskOutputs`."""
    outputs: dict[str, Any] = {}
    for index, task in enumerate(plan.tasks):
        if task.output is not None:
            outputs[str(index)] = task.output
            outputs[task.id] = task.output
    return outputs


_PLACEHOLDER_RE = re.compile(r"\{\$task\.(\d+)\.output(?:\.([^}]+))?\}")
_FULL_PLACEHOLDER_RE = re.compile(r"^\{\$task\.(\d+)\.output(?:\.([^}]+))?\}$")


def resolve_task_input(input_: Any, previous_outputs: dict[str, Any]) -> Any:
    """Recursively resolve `{$task.N.output[.dotPath]}` placeholders.

    Mirrors TS `resolveTaskInput`. Single full-string placeholder preserves
    the original type; inline placeholders are stringified. Also supports
    the elasticdash `{$ref: "taskId.output.path"}` object form.
    """

    def walk_path(root: Any, path: str, task_index: str) -> Any:
        cursor: Any = root
        for part in path.split("."):
            if cursor is None or not isinstance(cursor, dict):
                raise ValueError(
                    f'resolve_task_input: cannot traverse path "{path}" for task '
                    f'{task_index} — reached non-object at "{part}"'
                )
            cursor = cursor.get(part)
        return cursor

    def resolve_string(s: str) -> Any:
        m = _FULL_PLACEHOLDER_RE.match(s)
        if m:
            index, path = m.group(1), m.group(2)
            if index not in previous_outputs:
                raise ValueError(
                    f"resolve_task_input: no output found for task index {index}"
                )
            task_output = previous_outputs[index]
            return walk_path(task_output, path, index) if path else task_output

        def replace(match: re.Match[str]) -> str:
            index = match.group(1)
            path = match.group(2)
            if index not in previous_outputs:
                raise ValueError(
                    f"resolve_task_input: no output found for task index {index}"
                )
            task_output = previous_outputs[index]
            value = walk_path(task_output, path, index) if path else task_output
            return "" if value is None else str(value)

        return _PLACEHOLDER_RE.sub(replace, s)

    def deep_resolve(val: Any) -> Any:
        if isinstance(val, str):
            return resolve_string(val)
        if isinstance(val, list):
            return [deep_resolve(item) for item in val]
        if isinstance(val, dict):
            ref = val.get("$ref")
            if isinstance(ref, str):
                parts = ref.split(".")
                task_id = parts[0]
                path_parts = parts[1:]
                cursor: Any = previous_outputs.get(task_id)
                for part in path_parts:
                    if part == "output":
                        continue
                    if cursor is None:
                        return None
                    cursor = cursor.get(part) if isinstance(cursor, dict) else None
                return cursor
            return {k: deep_resolve(v) for k, v in val.items()}
        return val

    return deep_resolve(input_)


def serialize_agent_state(
    plan: AgentPlan, trace: Optional[list[WorkflowEvent]] = None
) -> AgentState:
    """Serialise a (partial or complete) plan + trace events into
    `AgentState`. The `resumeFromTaskIndex` is the index of the first
    non-completed task (or `len(tasks)` if all complete)."""
    if trace is None:
        trace = []
    first_incomplete = next(
        (i for i, t in enumerate(plan.tasks) if t.status != "completed"), -1
    )
    resume_from = len(plan.tasks) if first_incomplete == -1 else first_incomplete
    cloned_tasks = [
        AgentTask(
            id=t.id,
            description=t.description,
            tool=t.tool,
            input=t.input,
            status=t.status,
            output=t.output,
            startedAt=t.startedAt,
            completedAt=t.completedAt,
            error=t.error,
        )
        for t in plan.tasks
    ]
    cloned_plan = AgentPlan(
        id=plan.id,
        goal=plan.goal,
        tasks=cloned_tasks,
        status=plan.status,
        currentTaskIndex=plan.currentTaskIndex,
        context=dict(plan.context),
        metadata=dict(plan.metadata),
    )
    return AgentState(plan=cloned_plan, trace=trace, resumeFromTaskIndex=resume_from)


def deserialize_agent_state(raw: Any) -> AgentState:
    """Validate and hydrate an `AgentState` from a parsed JSON value.

    Mirrors TS `deserializeAgentState`, including the check that tasks
    marked `completed` before the resume index have an output, and that
    all referenced tools exist in `agent_tools`.
    """
    if not isinstance(raw, (dict, AgentState)):
        raise ValueError("deserialize_agent_state: state must be a non-null object")
    if isinstance(raw, AgentState):
        return raw

    plan_raw = raw.get("plan")
    if not plan_raw or not isinstance(plan_raw, dict) or not isinstance(plan_raw.get("tasks"), list):
        raise ValueError("deserialize_agent_state: invalid state — missing plan.tasks")
    resume_from = int(raw.get("resumeFromTaskIndex", 0) or 0)
    trace_raw = raw.get("trace", []) or []

    tasks = [
        AgentTask(
            id=t["id"],
            description=t.get("description", ""),
            tool=t["tool"],
            input=t.get("input"),
            status=t.get("status", "pending"),
            output=t.get("output"),
            startedAt=t.get("startedAt"),
            completedAt=t.get("completedAt"),
            error=t.get("error"),
        )
        for t in plan_raw["tasks"]
    ]
    plan = AgentPlan(
        id=plan_raw.get("id", "plan-unknown"),
        goal=plan_raw.get("goal", ""),
        tasks=tasks,
        status=plan_raw.get("status", "planning"),
        currentTaskIndex=plan_raw.get("currentTaskIndex", 0),
        context=plan_raw.get("context", {}) or {},
        metadata=plan_raw.get("metadata", {}) or {},
    )

    tools = get_agent_tools()
    for i in range(min(resume_from, len(plan.tasks))):
        task = plan.tasks[i]
        if task.status == "completed" and task.output is None:
            raise ValueError(
                f'deserialize_agent_state: task "{task.id}" is marked completed but has no output'
            )
        if task.tool not in tools:
            raise ValueError(
                f'deserialize_agent_state: unknown tool "{task.tool}" for task "{task.id}"'
            )

    trace = [
        WorkflowEvent(
            taskId=e.get("taskId", ""),
            toolName=e.get("toolName", ""),
            input=e.get("input"),
            output=e.get("output"),
            status=e.get("status", "completed"),
            startedAt=e.get("startedAt"),
            completedAt=e.get("completedAt"),
            error=e.get("error"),
            agentTaskId=e.get("agentTaskId"),
            agentTaskIndex=e.get("agentTaskIndex"),
        )
        for e in trace_raw
    ]
    return AgentState(plan=plan, trace=trace, resumeFromTaskIndex=resume_from)


# ============================================================
# Executor / planner agents
# ============================================================


async def executor_agent(plan: AgentPlan, resume_from: int = 0) -> AgentPlan:
    """Execute the plan sequentially, with optional mid-plan resumption.

    Tasks before `resume_from` are skipped — their cached outputs are
    preserved for placeholder resolution in subsequent tasks.
    Mirrors TS `executorAgent`.
    """
    plan.status = "executing"
    plan.currentTaskIndex = resume_from
    API_CALL_LIMIT = 50
    api_call_count = 0
    print(
        f'[executor_agent] Running plan "{plan.id}" from task {resume_from} '
        f"({len(plan.tasks)} total)"
    )

    tools = get_agent_tools()

    for i, task in enumerate(plan.tasks):
        plan.currentTaskIndex = i

        if i < resume_from:
            print(f'[executor_agent] Skipping cached task "{task.id}" (index {i})')
            continue

        if api_call_count >= API_CALL_LIMIT:
            plan.status = "failed"
            print(f"[executor_agent] API call limit ({API_CALL_LIMIT}) reached")
            break

        # Resolve {$task.N.output.path} placeholders against previous outputs
        prefix_plan = AgentPlan(
            id=plan.id,
            goal=plan.goal,
            tasks=plan.tasks[:i],
            status=plan.status,
            currentTaskIndex=plan.currentTaskIndex,
            context=plan.context,
            metadata=plan.metadata,
        )
        previous_outputs = extract_task_outputs(prefix_plan)
        try:
            task.input = resolve_task_input(task.input, previous_outputs)
        except Exception as err:  # noqa: BLE001
            task.status = "failed"
            task.error = f"Placeholder resolution failed: {err}"
            task.completedAt = int(time.time() * 1000)
            api_call_count += 1
            continue

        task.status = "in-progress"
        task.startedAt = int(time.time() * 1000)

        tool = tools.get(task.tool)
        if tool is None:
            task.status = "failed"
            task.error = f'Unknown tool: "{task.tool}"'
            task.completedAt = int(time.time() * 1000)
            print(f'[executor_agent] Unknown tool "{task.tool}" for task "{task.id}"')
            api_call_count += 1
            continue

        try:
            task.output = await tool.execute(task.input, None)
            task.status = "completed"
            task.completedAt = int(time.time() * 1000)
            print(f'[executor_agent] Completed task "{task.id}"')
        except Exception as err:  # noqa: BLE001
            task.status = "failed"
            task.error = str(err)
            task.completedAt = int(time.time() * 1000)
            print(f'[executor_agent] Task "{task.id}" failed: {task.error}')
        api_call_count += 1

    if plan.status != "failed":
        plan.status = (
            "completed"
            if all(t.status == "completed" for t in plan.tasks)
            else "failed"
        )
        plan.currentTaskIndex = len(plan.tasks)
    return plan


async def planner_agent(
    user_query: str, context: Optional[dict[str, Any]] = None
) -> AgentPlan:
    """Generate an initial plan from a user query by running a query
    refinement step. Mirrors TS `plannerAgent`."""
    if context is None:
        context = {}
    plan = AgentPlan(
        id=f"plan-{int(time.time() * 1000)}",
        goal=user_query,
        status="planning",
        currentTaskIndex=0,
        context=context,
        metadata={"createdAt": int(time.time() * 1000), "userQuery": user_query},
        tasks=[
            AgentTask(
                id="task-planning-1",
                description="Refine and clarify user query",
                tool="queryRefinement",
                input={"userInput": user_query, "userToken": context.get("userToken")},
                status="pending",
            )
        ],
    )
    return await executor_agent(plan)


async def resume_agent_from_trace(state: AgentState) -> AgentPlan:
    """Resume an agent plan from `state.resumeFromTaskIndex`."""
    validated = deserialize_agent_state(state)
    print(
        f'[resume_agent_from_trace] Resuming plan "{validated.plan.id}" from '
        f"task {validated.resumeFromTaskIndex}"
    )
    return await executor_agent(validated.plan, validated.resumeFromTaskIndex)


async def run_agentic_flow(_root_span: Any, plan: AgentPlan) -> AgentPlan:
    """Stub agentic flow entry point. For new code, prefer `executor_agent(plan)`."""
    tools = get_agent_tools()
    for task in plan.tasks:
        task.status = "in-progress"
        tool = tools.get(task.tool)
        if tool is None:
            task.status = "failed"
            task.error = f'Unknown tool: "{task.tool}"'
            continue
        try:
            task.output = await tool.execute(task.input, None)
            task.status = "completed"
        except Exception as err:  # noqa: BLE001
            task.status = "failed"
            task.output = err
    return plan


# ============================================================
# LLM client wrappers
# ============================================================


ChatMessage = dict[str, str]  # {"role": "user" | "assistant" | "system", "content": "..."}


def _split_system(
    messages: Iterable[ChatMessage], system_prompt: str
) -> tuple[Optional[str], list[ChatMessage]]:
    """Split off the system prompt (Anthropic-style top-level kwarg).

    If `system_prompt` is provided, it takes precedence and any system
    messages in the list are dropped. Otherwise the first system message
    (if any) is used.
    """
    msgs = list(messages)
    if system_prompt:
        msgs = [m for m in msgs if m.get("role") != "system"]
        return system_prompt, msgs
    sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
    if sys_msg is None:
        return None, msgs
    return sys_msg.get("content", ""), [m for m in msgs if m.get("role") != "system"]


async def kimi_chat_completion(
    *,
    messages: list[ChatMessage],
    model: str = "kimi-k2-turbo-preview",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system_prompt: str = "",
    session_id: Optional[str] = None,
) -> str:
    """Call Kimi (Moonshot) via the OpenAI-compatible API.

    Mirrors TS `kimiChatCompletion`. The TS code hard-pins `model =
    'kimi-k2-turbo-preview'` regardless of the caller's argument, and we
    preserve that quirk for behaviour parity.
    """
    _ = session_id  # observability sessionId is ignored in the CLI port
    model = "kimi-k2-turbo-preview"
    # Lazy import — openai SDK isn't required for unit tests that mock this.
    from openai import OpenAI  # type: ignore[import-not-found]

    cfg = _config.load()
    client = OpenAI(api_key=cfg.kimi_api_key, base_url=cfg.kimi_base_url)
    chat_messages: list[ChatMessage] = (
        [{"role": "system", "content": system_prompt}, *messages]
        if system_prompt
        else list(messages)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=chat_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (response.choices[0].message.content or "").strip()
        return content
    except Exception as error:  # noqa: BLE001
        print(f"Error in kimi_chat_completion: {error}")
        raise


async def anthropic_chat_completion(
    *,
    messages: list[ChatMessage],
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.0,
    max_tokens: int = 256,
    system_prompt: str = "",
    session_id: Optional[str] = None,
) -> str:
    """Call Anthropic Claude. The single non-streaming chat-completion path
    for the port (used by planner, validator, executor synthesis, etc.).
    Mirrors TS `anthropicChatCompletion`.
    """
    _ = session_id
    from anthropic import Anthropic  # type: ignore[import-not-found]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")

    system, user_msgs = _split_system(messages, system_prompt)
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            system=system or "",
            messages=[
                {"role": m["role"], "content": m.get("content", "")}
                for m in user_msgs
                if m.get("role") in ("user", "assistant")
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # response.content is a list of content blocks; concatenate text blocks.
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts).strip()
    except Exception as error:  # noqa: BLE001
        print(f"Error in anthropic_chat_completion: {error}")
        raise


async def openai_chat_completion(**kwargs: Any) -> str:
    """Chat-completion dispatcher — now Claude-only.

    Historical name kept so every call site (planner, validator,
    executor synthesis, query refinement, placeholder resolver, etc.)
    keeps working without a sweep. The implementation simply forwards
    to `anthropic_chat_completion`. The original OpenAI client code
    and the `AI_PROVIDER` env-var branch have been removed; see
    `.temp/plan.md` "Switch chatbot from OpenAI to Claude" for the
    migration rationale.
    """
    return await anthropic_chat_completion(**kwargs)


# Note: `kimi_chat_completion`, `anthropic_chat_completion`, and
# `openai_chat_completion` are intentionally NOT wrapped here. Their
# underlying HTTP calls go through httpx (the `anthropic` and `openai`
# Python SDKs both build on it), and ElasticDash's `install_ai_interceptor`
# — installed inside `init_observability` — patches
# `httpx.AsyncClient.send` / `httpx.Client.send` to record every request
# whose URL matches its provider heuristics (`anthropic.com`,
# `/v1/messages`, `/v1/chat/completions`). Wrapping these functions
# again would emit duplicate events for the same call.


# ============================================================
# Streaming helper for the final-answer step
# ============================================================


async def anthropic_stream_text(
    *,
    messages: list[ChatMessage],
    model: str = "claude-sonnet-4-5-20250929",
    on_delta: Optional[Callable[[str], None]] = None,
) -> tuple[str, int]:
    """Stream a Claude response, invoking `on_delta(token)` for each text
    delta. Returns `(full_text, output_tokens)`.

    Replaces the Vercel AI SDK `streamText({ model, messages })` +
    `for await (const delta of textStream)` pattern in
    `chat-stream/route.ts → anthropicFinalAnswer`.
    """
    from anthropic import Anthropic  # type: ignore[import-not-found]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")

    system, user_msgs = _split_system(messages, "")
    client = Anthropic(api_key=api_key)

    text_parts: list[str] = []
    output_tokens = 0
    with client.messages.stream(
        model=model,
        system=system or "",
        messages=[
            {"role": m["role"], "content": m.get("content", "")}
            for m in user_msgs
            if m.get("role") in ("user", "assistant")
        ],
        max_tokens=2048,
    ) as stream:
        for text in stream.text_stream:
            text_parts.append(text)
            if on_delta is not None:
                on_delta(text)
        final = stream.get_final_message()
        try:
            output_tokens = int(getattr(final.usage, "output_tokens", 0) or 0)
        except Exception:  # noqa: BLE001
            output_tokens = 0
    return "".join(text_parts), output_tokens


__all__ = [
    "wrap_ai",
    "AgentTask",
    "AgentPlan",
    "WorkflowEvent",
    "AgentState",
    "AgentTool",
    "TaskStatus",
    "PlanStatus",
    "get_agent_tools",
    "extract_task_outputs",
    "resolve_task_input",
    "serialize_agent_state",
    "deserialize_agent_state",
    "executor_agent",
    "planner_agent",
    "resume_agent_from_trace",
    "run_agentic_flow",
    "kimi_chat_completion",
    "anthropic_chat_completion",
    "openai_chat_completion",
    "anthropic_stream_text",
]
