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


class ClarificationReason(StrEnum):
    NONE = "none"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    MISSING_TASK_BODY = "missing_task_body"
    MISSING_REMINDER_DATE = "missing_reminder_date"
    MISSING_REMINDER_TIME = "missing_reminder_time"
    MISSING_REMINDER_DATE_TIME = "missing_reminder_date_time"
    MISSING_RECURRENCE_COUNT = "missing_recurrence_count"
    MISSING_RECURRENCE_DETAILS = "missing_recurrence_details"
    UNSUPPORTED_RECURRENCE = "unsupported_recurrence"
    SEMANTIC_MISMATCH = "semantic_mismatch"


@dataclass(frozen=True)
class ReminderRecurrence:
    """A local Weixin reminder series, never a Dida recurrence rule.

    Weekly rules are finite and use ``weekday``/``count``.  A daily rule is a
    rolling local series: ``count`` and ``weekday`` are zero and ``times``
    contains one or more unique ``HH:MM`` wall-clock slots.
    """

    frequency: str = "weekly"
    interval: int = 1
    weekday: int = 0
    count: int = 0
    times: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    @property
    def conversation_key(self) -> str:
        return f"{self.platform}:{self.account_id}:{self.user_id}:{self.chat_id}"


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
    reminder_recurrence: ReminderRecurrence | None = None

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
    clarification_reason: ClarificationReason = ClarificationReason.NONE

    @property
    def action_count(self) -> int:
        return len(self.tasks) + len(self.notes)


@dataclass(frozen=True)
class PendingTaskClarification:
    reason: ClarificationReason
    task: TaskDraft
    reminder_date: str = ""
    reminder_time: str = ""
    source_message_id: str = ""
    # "ambiguous" means conflicting periods still need explicit clarification.
    reminder_period: str = ""
    last_received_at: str = ""


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
