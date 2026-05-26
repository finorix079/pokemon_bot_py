"""chat/handler.py — chat pipeline orchestration for the CLI.

Ported from open-react-template/app/api/chat-stream/route.ts (750 lines).

Key adaptations from the TS source:
- No HTTP, no SSE, no Vercel AI data-stream wire framing. Stage updates
  and final-answer tokens are emitted via callbacks (`on_status`,
  `on_token`); the CLI binds them to `utils.cli_stream` writers.
- No Langfuse `startActiveObservation` wrapper. The CLI runs without
  observability.
- No plan-approval gate (`writePlan` / `storePendingPlan` / `/api/approve`
  polling). Pokémon tools are read-only, so the TS `allReadOnly` branch
  always fires; we just inline its body and drop the write-op branch
  entirely. Approval-gate code lives in the TS file from L575-693.

Pipeline preserved verbatim:
  1. summarize_messages → filter_plan_messages
  2. follow-up detection (regex + length heuristic + pronoun check)
  3. conversation context build (token-capped slice of recent messages)
  4. query refinement
  5. RAG: get_all_matched_apis → get_top_k_results
  6. planner_agent (warm-up trace observation)
  7. run_planner_with_inputs (LLM-driven plan)
  8. detect_resolution_vs_execution → optional table-only re-plan
  9. clarification short-circuit
 10. execute_iterative_planner (read-only branch)
 11. stream_final_answer
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Optional

from ..schemas.ai import ChatStreamRequest
from ..services.chat_planner_service import (
    RequestContext,
    get_all_matched_apis,
    get_top_k_results,
    serialize_useful_data_in_order,
)
from ..tools.pokemon_tools import queryRefinement as _queryRefinementTool, maybe_await
from ..utils.ai_handler import anthropic_stream_text, planner_agent
from ..utils.query_refinement import handle_query_concepts_and_needs
from .executor import execute_iterative_planner
from .message_utils import filter_plan_messages, summarize_messages, summarize_message
from .planner_utils import run_planner_with_inputs
from .session import generate_session_id, pending_plans
from .validators import detect_resolution_vs_execution


# ---------------------------------------------------------------------------
# Follow-up query detection (preserved from chat-stream/route.ts:347-352)
# ---------------------------------------------------------------------------


_FOLLOW_UP_PREFIX_RE = re.compile(
    r"^(what about|how about|and|also|more|details?|show me|tell me more|"
    r"what else|the same|similarly|like that|its|their|his|her)",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(it|them|that|this|those|these)\b", re.IGNORECASE)


def _is_follow_up(content: str) -> bool:
    stripped = content.strip()
    return bool(
        _FOLLOW_UP_PREFIX_RE.match(stripped)
        or len(stripped) < 20
        or _PRONOUN_RE.search(stripped)
    )


def _estimate_tokens(text: str) -> int:
    return -(-len(text) // 4)  # ceil(len / 4)


# ---------------------------------------------------------------------------
# Final-answer streaming (replaces anthropicFinalAnswer + streamFinalAnswer)
# ---------------------------------------------------------------------------


def _format_step(step: dict[str, Any], idx: int) -> str:
    """Format one executed step as a markdown block.

    Ported from TS `formatStep` (chat-stream/route.ts:86-101).
    """
    num = idx + 1
    desc = step.get("description") or f"Step {num}"
    api = step.get("api") or {}
    method = (api.get("method") or "").upper()
    path = api.get("path") or ""
    sql = (api.get("requestBody") or {}).get("query")

    lines = [f"{num}. **{desc}**"]
    if method and path:
        lines.append(f"   `{method} {path}`")
    if sql:
        lines.append(f"   ```sql\n   {sql.strip()}\n   ```")
    return "\n".join(lines)


def _format_executed_steps_summary(executed_steps: list[dict[str, Any]]) -> str:
    """Render `executedSteps` as a "Steps taken" markdown block.

    Ported from TS `formatExecutedStepsSummary` (chat-stream/route.ts:107-111).
    """
    if not executed_steps:
        return ""
    formatted = [_format_step(entry.get("step") or {}, idx) for idx, entry in enumerate(executed_steps)]
    return "**Steps taken:**\n\n" + "\n\n".join(formatted)


async def _stream_final_answer(
    executor_message: str,
    refined_query: str,
    useful_data: Optional[str],
    executed_steps_summary: Optional[str],
    on_token: Callable[[str], None],
) -> dict[str, Any]:
    """Generate + stream the final answer.

    Replaces TS `streamFinalAnswer` (chat-stream/route.ts:176-201). The
    Anthropic system prompt and user-content shape are preserved
    byte-for-byte; only the streaming mechanism differs (Python anthropic
    SDK's `messages.stream` vs Vercel AI SDK's `streamText`).
    """
    user_content = (
        f"Question: {refined_query}\n\nData collected:\n{useful_data}\n\n"
        f"Execution result: {executor_message}"
        if useful_data
        else f"Question: {refined_query}\n\nExecution result: {executor_message}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Synthesise the execution result into "
                "a clear, concise answer for the user."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    text, tokens = await anthropic_stream_text(
        messages=messages, on_delta=on_token
    )
    if executed_steps_summary:
        suffix = f"\n\n---\n\n{executed_steps_summary}"
        on_token(suffix)
        full = f"{text}{suffix}"
    else:
        full = text
    return {"text": full, "tokens": tokens}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class PipelineResult(dict):
    """Final result returned by `run_chat_pipeline`. Marker class so the
    REPL can distinguish "got an answer" from "got a clarification"
    without isinstance gymnastics."""


async def run_chat_pipeline(
    request: ChatStreamRequest,
    *,
    user_token: str = "",
    on_status: Callable[[str], None] = lambda _msg: None,
    on_token: Callable[[str], None] = lambda _tok: None,
    on_result: Callable[[str], None] = lambda _msg: None,
    on_error: Callable[[str], None] = lambda _msg: None,
) -> PipelineResult:
    """Run the chatbot pipeline on a `ChatStreamRequest`.

    Ports `app/api/chat-stream/route.ts` POST handler. Returns a
    `PipelineResult` dict carrying the final state. The caller is
    expected to bind `on_status` / `on_token` / `on_result` / `on_error`
    to terminal writers (see `pokemon_bot/__main__.py`).
    """
    messages = [m.model_dump() for m in request.messages]
    message_id = str(uuid.uuid4())
    if not messages:
        on_error("messages array is required")
        return PipelineResult(error="messages array is required")

    session_id = request.sessionId or generate_session_id(messages)

    # Find the latest user message
    user_message = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    if user_message is None:
        on_error("No user message found")
        return PipelineResult(error="No user message found")

    # Guard: already a plan awaiting approval for this session (approval
    # gate is otherwise dead code in the CLI port; kept for parity).
    if session_id in pending_plans:
        msg = (
            "A plan is already awaiting your approval. "
            "Please use the Approve or Reject buttons."
        )
        on_result(msg)
        return PipelineResult(message=msg, sessionId=session_id, messageId=message_id)

    request_context = RequestContext()

    # --- Summarise & clean conversation context ---
    on_status("Analysing your request…")
    summarised_messages = await summarize_messages(messages, "")
    cleaned_messages = filter_plan_messages(summarised_messages)

    is_follow_up = _is_follow_up(user_message.get("content", ""))
    conversation_context = ""
    MAX_CONTEXT_TOKENS = 800
    MAX_CONTEXT_MESSAGES = 10

    if len(cleaned_messages) > 1:
        context_depth = (
            min(MAX_CONTEXT_MESSAGES, len(cleaned_messages) - 1) if is_follow_up else 1
        )
        # Slice mirrors TS `cleanedMessages.slice(-1 - contextDepth, -1)`
        start = -1 - context_depth
        recent_messages = cleaned_messages[start:-1] if start < -1 else cleaned_messages[:-1]
        context_messages: list[dict[str, Any]] = []
        for msg in recent_messages:
            if msg.get("role") == "assistant" and len(msg.get("content", "")) > 800:
                context_messages.append(await summarize_message(msg, ""))
            else:
                context_messages.append(msg)
        temp_context = "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in context_messages
        )
        if _estimate_tokens(temp_context) > MAX_CONTEXT_TOKENS and context_messages:
            last = context_messages[-1]
            temp_context = f"{last.get('role')}: {last.get('content')}"
        conversation_context = temp_context

    # --- Query refinement ---
    on_status("Refining query…")
    query_with_context = (
        f"Previous context:\n{conversation_context}\n\nCurrent query: {user_message['content']}"
        if conversation_context
        else user_message["content"]
    )

    refinement = await maybe_await(
        _queryRefinementTool(
            {"userInput": query_with_context, "userToken": user_token}
        )
    )
    refined_query = refinement.get("refinedQuery", user_message["content"])
    concepts = refinement.get("concepts") or []
    api_needs = refinement.get("apiNeeds") or []
    entities = refinement.get("entities") or []
    intent_type = refinement.get("intentType") or "FETCH"
    reference_task = refinement.get("referenceTask")
    final_deliverable = refined_query
    handle_query_concepts_and_needs(concepts, api_needs)

    # --- RAG search ---
    on_status("Searching relevant data sources…")
    all_matched_apis = await get_all_matched_apis(
        entities=entities, intentType=intent_type, context=request_context
    )
    top_k_results = await get_top_k_results(all_matched_apis, 20)
    useful_data_str = serialize_useful_data_in_order(request_context)

    # --- Planning ---
    on_status("Building execution plan…")
    await planner_agent(refined_query, {"userToken": user_token})

    try:
        planner_result = await run_planner_with_inputs(
            topKResults=top_k_results,
            refinedQuery=refined_query,
            apiKey="",
            usefulData=useful_data_str,
            conversationContext=conversation_context,
            finalDeliverable=final_deliverable,
            intentType=intent_type,
            entities=entities,
            requestContext=request_context,
            referenceTask=reference_task,
        )
        actionable_plan = planner_result["actionablePlan"]
        planner_raw_response = planner_result["planResponse"]
    except Exception as err:  # noqa: BLE001
        if "No tables selected for SQL generation" in str(err):
            msg = "No relevant tables found for this query"
            on_result(msg)
            return PipelineResult(message=msg, refinedQuery=refined_query)
        raise

    if isinstance(actionable_plan, dict) and actionable_plan.get("impossible"):
        msg = actionable_plan.get("message") or "Unable to process this query."
        on_result(msg)
        return PipelineResult(
            message=msg,
            final=True,
            reason=actionable_plan.get("reason"),
            refinedQuery=refined_query,
        )

    # --- Resolution vs execution detection ---
    query_intent = await detect_resolution_vs_execution(
        refined_query, actionable_plan, ""
    )
    if query_intent == "resolution":
        table_only_results = [
            item
            for item in top_k_results
            if isinstance(item.get("id"), str)
            and (item["id"].startswith("table-") or item["id"] == "sql-query")
        ]
        replan_result = await run_planner_with_inputs(
            topKResults=table_only_results,
            refinedQuery=refined_query,
            apiKey="",
            usefulData=useful_data_str,
            conversationContext=conversation_context,
            finalDeliverable=final_deliverable,
            intentType="FETCH",
            entities=entities,
            requestContext=request_context,
            referenceTask=reference_task,
        )
        actionable_plan = replan_result["actionablePlan"]
        planner_raw_response = replan_result["planResponse"]
        if isinstance(actionable_plan, dict) and actionable_plan.get("impossible"):
            msg = actionable_plan.get("message") or "Unable to process this query."
            on_result(msg)
            return PipelineResult(message=msg, final=True, refinedQuery=refined_query)

    # --- Clarification needed ---
    if isinstance(actionable_plan, dict) and actionable_plan.get("needs_clarification"):
        msg = (
            actionable_plan.get("clarification_question")
            or "Could you clarify your request?"
        )
        on_result(msg)
        return PipelineResult(message=msg, refinedQuery=refined_query)

    execution_plan = (
        actionable_plan.get("execution_plan") if isinstance(actionable_plan, dict) else None
    )

    if execution_plan:
        # Pokémon tools are read-only, so the TS `allReadOnly` branch always
        # fires. The user-decided plan explicitly drops the approval gate.
        on_status("Running read-only queries…")
        exec_request_context = RequestContext()
        result = await execute_iterative_planner(
            refined_query,
            top_k_results,
            planner_raw_response,
            "",
            user_token,
            final_deliverable,
            {},
            conversation_context,
            entities,
            exec_request_context,
            50,
            None,
            None,
        )
        if result.get("error"):
            msg = result.get("clarification_question") or result.get("error")
            on_result(msg)
            return PipelineResult(
                message=msg, error=result.get("error"), refinedQuery=refined_query
            )

        on_status("Generating answer…")
        executed_steps = result.get("executedSteps") or []
        steps_summary = (
            _format_executed_steps_summary(executed_steps) if executed_steps else None
        )
        final = await _stream_final_answer(
            executor_message=result.get("message")
            or str(result.get("accumulatedResults") or []),
            refined_query=refined_query,
            useful_data=None,
            executed_steps_summary=steps_summary,
            on_token=on_token,
        )
        return PipelineResult(
            message=final["text"],
            tokens=final["tokens"],
            refinedQuery=refined_query,
            sessionId=session_id,
            messageId=message_id,
        )

    # --- No execution steps — handle "goal already completed" path ---
    if (
        isinstance(actionable_plan, dict)
        and isinstance(actionable_plan.get("message"), str)
        and "goal completed" in actionable_plan["message"].lower()
    ):
        on_status("Generating answer…")
        final = await _stream_final_answer(
            executor_message="",
            refined_query=refined_query,
            useful_data=useful_data_str,
            executed_steps_summary=None,
            on_token=on_token,
        )
        return PipelineResult(
            message=final["text"],
            tokens=final["tokens"],
            refinedQuery=refined_query,
            sessionId=session_id,
            messageId=message_id,
        )

    msg = "Plan does not include an execution plan."
    on_result(msg)
    return PipelineResult(message=msg, refinedQuery=refined_query)


__all__ = ["run_chat_pipeline", "PipelineResult"]
