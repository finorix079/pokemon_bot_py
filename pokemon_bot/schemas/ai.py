"""schemas/ai.py — pydantic v2 schemas for structured LLM outputs and CLI inputs.

Ported from open-react-template/schemas/ai.ts (Zod). The original schemas are
used with the Vercel AI SDK's `generateObject()` for JSON-schema-constrained
LLM output; in this Python port they serve the same role and also validate
inputs that come in from the REPL or tests.

Pydantic v2 conventions:
- `BaseModel` replaces `z.object`.
- `Literal[...]` replaces `z.enum`.
- `dict[str, Any]` replaces `z.record(z.string(), z.unknown())`.
- `Optional[T] = None` replaces `z.nullable().optional()`.
- `model_config = ConfigDict(extra="ignore")` mirrors the implicit Zod
  behaviour of dropping unknown keys without raising.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

IntentType = Literal["FETCH", "MODIFY"]


# ---------------------------------------------------------------------------
# Query refinement (utils/queryRefinement.ts)
# ---------------------------------------------------------------------------


class QueryRefinement(BaseModel):
    """Structured output from `clarify_and_refine_user_input`.

    Replaces regex-based field extraction from a freeform LLM response.
    """

    model_config = ConfigDict(extra="ignore")

    refinedQuery: str
    """Clearer, normalised version of the user's question."""

    language: str
    """ISO 639-1 language code (e.g. "EN", "ZH")."""

    concepts: list[str] = Field(default_factory=list)
    """Key domain concepts present in the query."""

    apiNeeds: list[str] = Field(default_factory=list)
    """API functionalities required to answer the query."""

    entities: list[str] = Field(default_factory=list)
    """Entity names / data sources that need investigation."""

    intentType: IntentType
    """Whether this is a read-only or mutation query."""


# ---------------------------------------------------------------------------
# Planner intent analysis (app/api/chat/planner.ts — Step 1)
# ---------------------------------------------------------------------------


class Intent(BaseModel):
    """The next-step intent object returned by the intent-analysis LLM call.

    Replaces brittle JSON repair (`replace(/，/g, ',')` etc.) with a schema.
    """

    model_config = ConfigDict(extra="ignore")

    description: str
    """One-sentence description of the single most critical next action."""

    type: IntentType
    """Whether this next action is a read or a mutation."""


# ---------------------------------------------------------------------------
# Execution plan (app/api/chat/planner.ts — Step 3)
# ---------------------------------------------------------------------------


class ApiCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    method: str
    parameters: Optional[dict[str, Any]] = None
    requestBody: Optional[dict[str, Any]] = None


class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_number: int
    description: str
    api: Optional[ApiCall] = None
    depends_on_step: Optional[int] = None


class ExecutionPlan(BaseModel):
    """The complete execution plan returned by the planner LLM.

    Replaces `match(/\\{[\\s\\S]*\\}/)` + `JSON.parse()` from the TS planner.
    """

    model_config = ConfigDict(extra="ignore")

    needs_clarification: bool
    clarification_question: Optional[str] = None
    execution_plan: Optional[list[ExecutionStep]] = None
    message: Optional[str] = None
    """Set when the goal is already complete."""

    impossible: Optional[bool] = None
    """Set to true when the system cannot handle the query."""

    reason: Optional[str] = None
    selected_tools_spec: Optional[list[Any]] = None


# ---------------------------------------------------------------------------
# Planner SQL/schema validation (app/api/chat/planner.ts — Step 3 validation)
# ---------------------------------------------------------------------------


class PlannerValidation(BaseModel):
    """Validation result from the SQL/schema consistency checker.

    Replaces `match(/\\{[\\s\\S]*\\}/)` + `JSON.parse()` on the validator
    response. The redundant `propertyNames` field is kept to mirror the TS
    schema's workaround for OpenAI structured-output strictness.
    """

    model_config = ConfigDict(extra="ignore")

    needs_clarification: bool
    reason: Optional[str] = None
    clarification_question: Optional[str] = None
    propertyNames: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Table selection (app/api/chat/plannerUtils.ts)
# ---------------------------------------------------------------------------


class TableSelection(BaseModel):
    """Table-selection decision returned by Kimi when choosing which DB
    tables to use for SQL generation.

    Replaces `match(/\\{[\\s\\S]*\\}/)` + `JSON.parse()`.
    """

    model_config = ConfigDict(extra="ignore")

    selected_tables: list[str] = Field(default_factory=list)
    """Table IDs or names chosen for this query."""

    focus_columns: dict[str, list[str]] = Field(default_factory=dict)
    """Specific columns of interest per table."""

    reasoning: str = ""
    """Brief explanation of the selection."""


# ---------------------------------------------------------------------------
# Goal validation (app/api/chat/validators.ts)
# ---------------------------------------------------------------------------


class NeedMoreActions(BaseModel):
    """Decision returned by `validateNeedMoreActions`.

    Replaces `match(/\\{[\\s\\S]*\\}/)` + `JSON.parse()` on the validator
    response.
    """

    model_config = ConfigDict(extra="ignore")

    needsMoreActions: bool
    reason: str
    missing_requirements: Optional[list[str]] = None
    suggested_next_action: Optional[str] = None
    useful_data: Optional[str] = None
    item_not_found: Optional[bool] = None


# ---------------------------------------------------------------------------
# CLI / API input validation
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """One message in a chat conversation."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: str


class ChatStreamRequest(BaseModel):
    """Request body schema for the chat pipeline.

    In the TS project this validated POST /api/chat-stream; here it
    represents one REPL submission (a list of messages built up across
    multiple turns plus an optional sessionId).
    """

    model_config = ConfigDict(extra="ignore")

    messages: list[ChatMessage]
    sessionId: Optional[str] = None


# ApproveRequest is intentionally omitted: the plan-approval gate is
# explicitly out of scope for the CLI port (see .temp/plan.md, Risks #5).
