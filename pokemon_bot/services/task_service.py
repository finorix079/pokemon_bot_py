"""services/task_service.py — task / message dataclasses.

Ported from open-react-template/services/taskService.ts (pure types, no logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class TaskStepApi:
    path: str
    method: str
    parameters: Optional[dict[str, Any]] = None
    requestBody: Optional[dict[str, Any]] = None


@dataclass
class TaskStep:
    stepOrder: int
    stepType: int
    stepContent: str
    stepJsonContent: Optional[dict[str, Any]] = None
    api: Optional[TaskStepApi] = None
    depends_on_step: Optional[int] = None


@dataclass
class TaskPayload:
    taskName: str
    taskType: int
    taskContent: str
    taskSteps: list[TaskStep] = field(default_factory=list)
    originalQuery: Optional[str] = None
    planResponse: Optional[str] = None


@dataclass
class SavedTask(TaskPayload):
    id: int = 0
    createdAt: str = ""
    steps: Optional[list[TaskStep]] = None


@dataclass
class PlanStep:
    step_number: Optional[int] = None
    description: Optional[str] = None
    api: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    requestBody: Optional[dict[str, Any]] = None
    depends_on_step: Optional[int] = None


@dataclass
class PlanSummary:
    goal: Optional[str] = None
    phase: Optional[str] = None
    steps: Optional[list[PlanStep]] = None
    selected_apis: Optional[list[Any]] = None


@dataclass
class Message:
    """Pipeline-level message dataclass.

    Mirrors TS `Message` from taskService.ts (the richer variant — note this
    is distinct from `schemas.ai.ChatMessage` which is the strict CLI input
    shape). The pipeline reads/writes these fields when shuttling plan
    metadata between turns.
    """

    role: Literal["user", "assistant"]
    content: str
    awaitingApproval: Optional[bool] = None
    sessionId: Optional[str] = None
    planSummary: Optional[PlanSummary] = None
    planResponse: Optional[str] = None
    refinedQuery: Optional[str] = None
    planningDurationMs: Optional[int] = None
    usedReferencePlan: Optional[bool] = None


__all__ = [
    "TaskStep",
    "TaskStepApi",
    "TaskPayload",
    "SavedTask",
    "PlanStep",
    "PlanSummary",
    "Message",
]
