from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class IntentKind(StrEnum):
    TASK = "task"
    NOTE = "note"
    MIXED = "mixed"
    QUERY = "query"
    CLARIFY = "clarify"
    PRIVATE = "private"


class ExecutionStatus(StrEnum):
    PLANNED = "planned"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class MessageEnvelope:
    platform: str
    account_id: str
    user_id: str
    chat_id: str
    chat_type: str
    message_id: str
    text: str
    received_at: datetime
    media_paths: tuple[str, ...] = ()
    media_types: tuple[str, ...] = ()

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return self.platform, self.account_id, self.message_id

    @property
    def sender_key(self) -> str:
        return f"{self.platform}:{self.account_id}:{self.user_id}"


@dataclass(frozen=True)
class TaskDraft:
    title: str
    due_date: str = ""
    due_time: str = ""
    priority: str = "none"
    category: str = ""
    tags: tuple[str, ...] = ()
    description: str = ""
    reminder_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NoteDraft:
    title: str
    body: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    target_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskQuery:
    mode: str = "today"
    keyword: str = ""


@dataclass(frozen=True)
class TaskReference:
    """The minimum local context required to operate on one exact Dida task."""

    task_id: str
    title: str
    category: str = ""
    project_id: str = ""
    status: str = ""


@dataclass(frozen=True)
class IntentPlan:
    kind: IntentKind
    tasks: tuple[TaskDraft, ...] = ()
    notes: tuple[NoteDraft, ...] = ()
    query: TaskQuery | None = None
    confidence: float = 1.0
    clarification: str = ""

    @property
    def action_count(self) -> int:
        return len(self.tasks) + len(self.notes)


@dataclass(frozen=True)
class ActionResult:
    action: str
    status: ExecutionStatus
    summary: str
    destination: str = ""
    external_id: str = ""
    error: str = ""
    preview: str = ""
    task_refs: tuple[TaskReference, ...] = ()

    @property
    def successful(self) -> bool:
        return self.status in {ExecutionStatus.PLANNED, ExecutionStatus.SUCCEEDED}


@dataclass(frozen=True)
class HandlingResult:
    status: ExecutionStatus
    reply: str
    results: tuple[ActionResult, ...] = field(default_factory=tuple)
    duplicate: bool = False
    llm_called: bool = False
    suppressed: bool = False
