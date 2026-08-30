from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .models import (
    ActionResult,
    ClarificationReason,
    ExecutionStatus,
    MessageEnvelope,
    PendingTaskClarification,
    ReminderRecurrence,
    TaskDraft,
    TaskReference,
)


@dataclass(frozen=True)
class ClaimResult:
    is_new: bool
    state: str
    retrying_failed_operations: bool = False
    content_matches: bool = True


@dataclass(frozen=True)
class OperationClaim:
    should_run: bool
    previous: ActionResult | None = None


@dataclass(frozen=True)
class ContextLookup:
    candidates: tuple[TaskReference, ...]
    expired: bool = False


@dataclass(frozen=True)
class PendingTaskClaim:
    pending: PendingTaskClarification | None
    state: str = "none"


@dataclass(frozen=True)
class ReminderRecord:
    row_id: int
    task: TaskReference
    platform: str
    account_id: str
    user_id: str
    chat_id: str
    source_message_id: str
    reminder_at: datetime
    attempts: int


class ReminderRouteConflictError(RuntimeError):
    """Refuse to move an existing reminder to a different delivery route."""


class IdempotencyLedger:
    """Durable local state without credentials or complete inbound message bodies.

    A pending clarification keeps only a short-lived, minimized structured task
    draft so a reply such as ``上午9点`` can be joined to the preceding request.
    """

    def __init__(self, path: Path | str):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,
            timeout=10,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS messages (
                    platform TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    sender_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    action_count INTEGER NOT NULL DEFAULT 0,
                    last_error_code TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (platform, account_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS operations (
                    platform TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    operation_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    state TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    destination TEXT NOT NULL DEFAULT '',
                    external_id TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    task_status TEXT NOT NULL DEFAULT '',
                    error_code TEXT NOT NULL DEFAULT '',
                    preview TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (platform, account_id, message_id, operation_key),
                    FOREIGN KEY (platform, account_id, message_id)
                        REFERENCES messages(platform, account_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS private_latches (
                    sender_key TEXT PRIMARY KEY,
                    source_message_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS daily_runs (
                    local_date TEXT NOT NULL,
                    job_name TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    PRIMARY KEY (local_date, job_name)
                );
                CREATE TABLE IF NOT EXISTS task_context (
                    sender_hash TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    task_status TEXT NOT NULL DEFAULT '',
                    context_kind TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    reminder_at TEXT NOT NULL DEFAULT '',
                    observed_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (sender_hash, batch_id, task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_task_context_sender_time
                    ON task_context(sender_hash, observed_at DESC);
                CREATE TABLE IF NOT EXISTS pending_completion (
                    sender_hash TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    task_status TEXT NOT NULL DEFAULT '',
                    source_message_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (sender_hash, ordinal)
                );
                CREATE TABLE IF NOT EXISTS pending_task_clarifications (
                    conversation_hash TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    draft_json TEXT NOT NULL,
                    reminder_date TEXT NOT NULL DEFAULT '',
                    reminder_time TEXT NOT NULL DEFAULT '',
                    source_message_id TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending',
                    claimed_by_message_id TEXT NOT NULL DEFAULT '',
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    platform TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    reminder_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL DEFAULT '',
                    delivered_at TEXT NOT NULL DEFAULT '',
                    delivered_message_id TEXT NOT NULL DEFAULT '',
                    last_error_code TEXT NOT NULL DEFAULT '',
                    UNIQUE(task_id, reminder_at)
                );
                CREATE INDEX IF NOT EXISTS idx_reminders_due
                    ON reminders(status, next_attempt_at, reminder_at);
                CREATE TABLE IF NOT EXISTS pending_reminder_actions (
                    conversation_hash TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS acknowledgements (
                    sender_hash TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    acknowledged_at TEXT NOT NULL,
                    PRIMARY KEY (sender_hash, message_id)
                );
                """
            )
            self._purge_expired_pending_tasks_locked(datetime.now(timezone.utc))
            self._purge_pending_reminder_actions_locked(datetime.now(timezone.utc))
            operation_columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(operations)")
            }
            if "project_id" not in operation_columns:
                self._connection.execute(
                    "ALTER TABLE operations ADD COLUMN project_id TEXT NOT NULL DEFAULT ''"
                )
            if "task_status" not in operation_columns:
                self._connection.execute(
                    "ALTER TABLE operations ADD COLUMN task_status TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def claim(self, message: MessageEnvelope) -> ClaimResult:
        now = self._now()
        content_fingerprint = "\x1f".join(
            (
                message.text or "",
                *message.media_types,
                *message.media_paths,
            )
        )
        content_hash = self._hash(content_fingerprint)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    """
                    SELECT state, content_hash FROM messages
                    WHERE platform = ? AND account_id = ? AND message_id = ?
                    """,
                    message.identity_key,
                ).fetchone()
                if existing is not None:
                    state = str(existing["state"])
                    matches = str(existing["content_hash"]) == content_hash
                    retryable = matches and state in {
                        ExecutionStatus.FAILED.value,
                        ExecutionStatus.PARTIAL.value,
                    }
                    if retryable:
                        self._connection.execute(
                            """
                            UPDATE messages SET state = 'processing', updated_at = ?
                            WHERE platform = ? AND account_id = ? AND message_id = ?
                            """,
                            (now, *message.identity_key),
                        )
                    self._connection.execute("COMMIT")
                    return ClaimResult(
                        retryable,
                        state,
                        retrying_failed_operations=retryable,
                        content_matches=matches,
                    )
                self._connection.execute(
                    """
                    INSERT INTO messages (
                        platform, account_id, message_id, sender_hash, content_hash,
                        received_at, state, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'processing', ?)
                    """,
                    (
                        message.platform,
                        message.account_id,
                        message.message_id,
                        self._hash(message.sender_key),
                        content_hash,
                        message.received_at.isoformat(),
                        now,
                    ),
                )
                self._connection.execute("COMMIT")
                return ClaimResult(True, "processing")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def finish(
        self,
        message: MessageEnvelope,
        status: ExecutionStatus,
        action_count: int = 0,
        error_code: str = "",
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE messages
                SET state = ?, action_count = ?, last_error_code = ?, updated_at = ?
                WHERE platform = ? AND account_id = ? AND message_id = ?
                """,
                (
                    status.value,
                    int(action_count),
                    error_code[:80],
                    self._now(),
                    *message.identity_key,
                ),
            )

    def claim_operation(
        self,
        message: MessageEnvelope,
        operation_key: str,
        action: str,
        *,
        retry_failed: bool = True,
    ) -> OperationClaim:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT * FROM operations
                    WHERE platform = ? AND account_id = ? AND message_id = ?
                      AND operation_key = ?
                    """,
                    (*message.identity_key, operation_key),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO operations(
                            platform, account_id, message_id, operation_key,
                            action, state, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'processing', ?)
                        """,
                        (*message.identity_key, operation_key, action, self._now()),
                    )
                    self._connection.execute("COMMIT")
                    return OperationClaim(True)
                if (
                    retry_failed
                    and str(row["state"]) == ExecutionStatus.FAILED.value
                ):
                    self._connection.execute(
                        """
                        UPDATE operations
                        SET state = 'processing', attempts = attempts + 1, updated_at = ?
                        WHERE platform = ? AND account_id = ? AND message_id = ?
                          AND operation_key = ?
                        """,
                        (self._now(), *message.identity_key, operation_key),
                    )
                    self._connection.execute("COMMIT")
                    return OperationClaim(True)
                self._connection.execute("COMMIT")
                return OperationClaim(False, self._operation_result(row))
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def has_operation(self, message: MessageEnvelope, operation_key: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM operations
                WHERE platform = ? AND account_id = ? AND message_id = ?
                  AND operation_key = ?
                LIMIT 1
                """,
                (*message.identity_key, operation_key),
            ).fetchone()
        return row is not None

    @staticmethod
    def _operation_result(row: sqlite3.Row) -> ActionResult:
        try:
            status = ExecutionStatus(str(row["state"]))
        except ValueError:
            status = ExecutionStatus.UNCERTAIN
        task_refs: tuple[TaskReference, ...] = ()
        if str(row["action"]) in {"task", "complete"} and str(row["external_id"]):
            task_refs = (
                TaskReference(
                    task_id=str(row["external_id"]),
                    title=str(row["summary"]),
                    category=str(row["destination"]),
                    project_id=str(row["project_id"]),
                    status=str(row["task_status"]),
                ),
            )
        return ActionResult(
            action=str(row["action"]),
            status=status,
            summary=str(row["summary"]),
            destination=str(row["destination"]),
            external_id=str(row["external_id"]),
            error=str(row["error_code"]),
            preview=str(row["preview"]),
            task_refs=task_refs,
        )

    def finish_operation(
        self,
        message: MessageEnvelope,
        operation_key: str,
        result: ActionResult,
    ) -> None:
        task_ref = result.task_refs[0] if result.task_refs else None
        with self._lock:
            self._connection.execute(
                """
                UPDATE operations SET
                    action = ?, state = ?, summary = ?, destination = ?,
                    external_id = ?, project_id = ?, task_status = ?,
                    error_code = ?, preview = ?, updated_at = ?
                WHERE platform = ? AND account_id = ? AND message_id = ?
                  AND operation_key = ?
                """,
                (
                    result.action,
                    result.status.value,
                    result.summary[:500],
                    result.destination[:200],
                    result.external_id[:300],
                    (task_ref.project_id if task_ref else "")[:300],
                    (task_ref.status if task_ref else "")[:50],
                    result.error[:200],
                    "",
                    self._now(),
                    *message.identity_key,
                    operation_key,
                ),
            )

    def arm_private_latch(self, sender_key: str, source_message_id: str, expires_at: datetime) -> None:
        sender_hash = self._hash(sender_key)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO private_latches(sender_key, source_message_id, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(sender_key) DO UPDATE SET
                    source_message_id = excluded.source_message_id,
                    expires_at = excluded.expires_at
                """,
                (sender_hash, source_message_id, expires_at.isoformat()),
            )

    def consume_private_latch(self, sender_key: str, now: datetime) -> bool:
        sender_hash = self._hash(sender_key)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT expires_at FROM private_latches WHERE sender_key = ?",
                    (sender_hash,),
                ).fetchone()
                self._connection.execute(
                    "DELETE FROM private_latches WHERE sender_key = ?",
                    (sender_hash,),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        if row is None:
            return False
        try:
            return datetime.fromisoformat(str(row["expires_at"])) >= now
        except ValueError:
            return False

    def record_acknowledgement(self, message: MessageEnvelope) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO acknowledgements(sender_hash, message_id, acknowledged_at)
                VALUES (?, ?, ?)
                """,
                (self._hash(message.sender_key), message.message_id, self._now()),
            )

    def record_task_context(
        self,
        sender_key: str,
        refs: Iterable[TaskReference],
        *,
        batch_id: str,
        source_message_id: str,
        observed_at: datetime,
        ttl_seconds: int,
        context_kind: str,
        reminder_at: datetime | None = None,
    ) -> None:
        expires_at = observed_at + timedelta(seconds=ttl_seconds)
        sender_hash = self._hash(sender_key)
        with self._lock:
            for ref in refs:
                if not ref.task_id or not ref.title:
                    continue
                completed = self._connection.execute(
                    "SELECT 1 FROM task_context WHERE sender_hash = ? AND task_id = ? AND task_status = 'completed' LIMIT 1",
                    (sender_hash, ref.task_id),
                ).fetchone()
                if completed:
                    continue
                self._connection.execute(
                    """
                    INSERT INTO task_context(
                        sender_hash, batch_id, task_id, title, category, project_id,
                        task_status, context_kind, source_message_id, reminder_at,
                        observed_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sender_hash, batch_id, task_id) DO UPDATE SET
                        title = excluded.title,
                        category = excluded.category,
                        project_id = excluded.project_id,
                        task_status = excluded.task_status,
                        context_kind = excluded.context_kind,
                        source_message_id = excluded.source_message_id,
                        reminder_at = excluded.reminder_at,
                        observed_at = excluded.observed_at,
                        expires_at = excluded.expires_at
                    """,
                    (
                        sender_hash,
                        batch_id,
                        ref.task_id,
                        ref.title[:500],
                        ref.category[:200],
                        ref.project_id[:300],
                        ref.status[:50],
                        context_kind[:40],
                        source_message_id[:300],
                        reminder_at.isoformat() if reminder_at else "",
                        observed_at.isoformat(),
                        expires_at.isoformat(),
                    ),
                )

    @staticmethod
    def _row_to_ref(row: sqlite3.Row) -> TaskReference:
        return TaskReference(
            task_id=str(row["task_id"]),
            title=str(row["title"]),
            category=str(row["category"]),
            project_id=str(row["project_id"]),
            status=str(row["task_status"]),
        )

    def recent_task_context(self, sender_key: str, now: datetime) -> ContextLookup:
        sender_hash = self._hash(sender_key)
        with self._lock:
            latest = self._connection.execute(
                """
                SELECT batch_id, expires_at FROM task_context
                WHERE sender_hash = ?
                ORDER BY observed_at DESC, rowid DESC LIMIT 1
                """,
                (sender_hash,),
            ).fetchone()
            if latest is not None:
                if self._pending_expired(latest["expires_at"], now):
                    return ContextLookup((), True)
                rows = self._connection.execute(
                    """
                    SELECT * FROM task_context
                    WHERE sender_hash = ? AND batch_id = ?
                      AND task_status != 'completed' AND expires_at >= ?
                    ORDER BY rowid
                    """,
                    (sender_hash, str(latest["batch_id"]), now.isoformat()),
                ).fetchall()
                return ContextLookup(tuple(self._row_to_ref(row) for row in rows))
            expired = self._connection.execute(
                """
                SELECT 1 FROM task_context
                WHERE sender_hash = ? AND task_status != 'completed'
                LIMIT 1
                """,
                (sender_hash,),
            ).fetchone()
        return ContextLookup((), expired is not None)

    def find_task_context(
        self, sender_key: str, title: str, now: datetime
    ) -> tuple[TaskReference, ...]:
        sender_hash = self._hash(sender_key)
        wanted = title.casefold().strip()
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM task_context
                WHERE sender_hash = ? AND task_status != 'completed' AND expires_at >= ?
                ORDER BY observed_at DESC
                """,
                (sender_hash, now.isoformat()),
            ).fetchall()
        refs = [self._row_to_ref(row) for row in rows]
        exact = [ref for ref in refs if ref.title.casefold().strip() == wanted]
        candidates = exact or [ref for ref in refs if wanted in ref.title.casefold()]
        unique: dict[str, TaskReference] = {}
        for ref in candidates:
            unique.setdefault(ref.task_id, ref)
        return tuple(unique.values())

    def set_pending_completion(
        self,
        sender_key: str,
        refs: Iterable[TaskReference],
        source_message_id: str,
        expires_at: datetime,
    ) -> None:
        sender_hash = self._hash(sender_key)
        with self._lock:
            self._connection.execute(
                "DELETE FROM pending_completion WHERE sender_hash = ?", (sender_hash,)
            )
            for ordinal, ref in enumerate(refs, start=1):
                self._connection.execute(
                    """
                    INSERT INTO pending_completion(
                        sender_hash, ordinal, task_id, title, category, project_id,
                        task_status, source_message_id, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sender_hash,
                        ordinal,
                        ref.task_id,
                        ref.title[:500],
                        ref.category[:200],
                        ref.project_id[:300],
                        ref.status[:50],
                        source_message_id[:300],
                        expires_at.isoformat(),
                    ),
                )

    def pending_completion(
        self, sender_key: str, now: datetime
    ) -> tuple[TaskReference, ...]:
        sender_hash = self._hash(sender_key)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM pending_completion
                WHERE sender_hash = ? AND expires_at >= ? ORDER BY ordinal
                """,
                (sender_hash, now.isoformat()),
            ).fetchall()
        return tuple(self._row_to_ref(row) for row in rows)

    def clear_pending_completion(self, sender_key: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM pending_completion WHERE sender_hash = ?",
                (self._hash(sender_key),),
            )

    @staticmethod
    def _pending_draft_json(pending: PendingTaskClarification) -> str:
        # Persist only what is required to resume a clarification.  In
        # particular, do not retain model-produced descriptions, categories,
        # tags, or other free-form copies of the inbound message.
        task = pending.task
        minimized = TaskDraft(
            title=task.title[:300],
            due_date=task.due_date,
            due_time=task.due_time,
            reminder_at=task.reminder_at,
            reminder_recurrence=task.reminder_recurrence,
        )
        payload = minimized.to_dict()
        payload["reminder_period"] = pending.reminder_period
        payload["last_received_at"] = pending.last_received_at
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _pending_expired(expires_at: object, now: datetime) -> bool:
        try:
            parsed = datetime.fromisoformat(str(expires_at))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc) < now.astimezone(timezone.utc)

    def _purge_expired_pending_tasks_locked(
        self,
        now: datetime,
        *,
        exclude_conversation_hash: str = "",
    ) -> int:
        rows = self._connection.execute(
            "SELECT conversation_hash, expires_at FROM pending_task_clarifications"
        ).fetchall()
        stale = tuple(
            (str(row["conversation_hash"]),)
            for row in rows
            if str(row["conversation_hash"]) != exclude_conversation_hash
            and self._pending_expired(row["expires_at"], now)
        )
        if stale:
            self._connection.executemany(
                "DELETE FROM pending_task_clarifications WHERE conversation_hash = ?",
                stale,
            )
        return len(stale)

    def purge_expired_pending_tasks(self, now: datetime | None = None) -> int:
        """Remove expired clarification payloads across every conversation."""

        with self._lock:
            self._purge_pending_reminder_actions_locked(now or datetime.now(timezone.utc))
            return self._purge_expired_pending_tasks_locked(
                now or datetime.now(timezone.utc)
            )

    @staticmethod
    def _pending_from_row(row: sqlite3.Row) -> PendingTaskClarification:
        raw = json.loads(str(row["draft_json"]))
        if not isinstance(raw, dict):
            raise ValueError("invalid pending task draft")
        recurrence_raw = raw.get("reminder_recurrence")
        recurrence = None
        if isinstance(recurrence_raw, dict):
            recurrence = ReminderRecurrence(
                frequency=str(recurrence_raw.get("frequency", "weekly")),
                interval=int(recurrence_raw.get("interval", 1)),
                weekday=int(recurrence_raw.get("weekday", 0)),
                count=int(recurrence_raw.get("count", 0)),
            )
        task = TaskDraft(
            title=str(raw.get("title", ""))[:300],
            due_date=str(raw.get("due_date", ""))[:10],
            due_time=str(raw.get("due_time", ""))[:5],
            priority=str(raw.get("priority", "none"))[:20],
            category=str(raw.get("category", ""))[:100],
            tags=tuple(str(item)[:50] for item in raw.get("tags", [])[:5]),
            description=str(raw.get("description", ""))[:2000],
            reminder_at=str(raw.get("reminder_at", ""))[:40],
            reminder_recurrence=recurrence,
        )
        reason_raw = str(row["reason"])
        try:
            reason = ClarificationReason(reason_raw)
        except ValueError:
            reason = ClarificationReason.SEMANTIC_MISMATCH
        return PendingTaskClarification(
            reason=reason,
            task=task,
            reminder_date=str(row["reminder_date"]),
            reminder_time=str(row["reminder_time"]),
            source_message_id=str(row["source_message_id"]),
            reminder_period=str(raw.get("reminder_period", ""))[:10],
            last_received_at=str(raw.get("last_received_at", ""))[:40],
        )

    def peek_pending_task(self, conversation_key: str, now: datetime) -> PendingTaskClarification | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM pending_task_clarifications WHERE conversation_hash = ?",
                (self._hash(conversation_key),),
            ).fetchone()
            if row is None or self._pending_expired(row["expires_at"], now):
                return None
            try:
                return self._pending_from_row(row)
            except (ValueError, TypeError, KeyError):
                return None

    def set_pending_task(
        self,
        conversation_key: str,
        pending: PendingTaskClarification,
        expires_at: datetime,
    ) -> None:
        conversation_hash = self._hash(conversation_key)
        updated_at = self._now()
        with self._lock:
            self._purge_expired_pending_tasks_locked(datetime.now(timezone.utc))
            self._connection.execute(
                """
                INSERT INTO pending_task_clarifications(
                    conversation_hash, reason, draft_json, reminder_date,
                    reminder_time, source_message_id, state,
                    claimed_by_message_id, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', '', ?, ?)
                ON CONFLICT(conversation_hash) DO UPDATE SET
                    reason = excluded.reason,
                    draft_json = excluded.draft_json,
                    reminder_date = excluded.reminder_date,
                    reminder_time = excluded.reminder_time,
                    source_message_id = excluded.source_message_id,
                    state = 'pending',
                    claimed_by_message_id = '',
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                WHERE pending_task_clarifications.state = 'pending'
                """,
                (
                    conversation_hash,
                    pending.reason.value,
                    self._pending_draft_json(pending),
                    pending.reminder_date,
                    pending.reminder_time,
                    pending.source_message_id[:300],
                    expires_at.astimezone(timezone.utc).isoformat()
                    if expires_at.tzinfo is not None
                    else expires_at.replace(tzinfo=timezone.utc).isoformat(),
                    updated_at,
                ),
            )

    def claim_pending_task(
        self,
        conversation_key: str,
        message_id: str,
        now: datetime,
    ) -> PendingTaskClaim:
        conversation_hash = self._hash(conversation_key)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT * FROM pending_task_clarifications
                    WHERE conversation_hash = ?
                    """,
                    (conversation_hash,),
                ).fetchone()
                if row is None:
                    self._purge_expired_pending_tasks_locked(now)
                    self._connection.execute("COMMIT")
                    return PendingTaskClaim(None)
                if self._pending_expired(row["expires_at"], now):
                    self._connection.execute(
                        "DELETE FROM pending_task_clarifications WHERE conversation_hash = ?",
                        (conversation_hash,),
                    )
                    self._purge_expired_pending_tasks_locked(now)
                    self._connection.execute("COMMIT")
                    return PendingTaskClaim(None, "expired")
                # Clean other silent conversations as part of any clarification
                # activity while preserving the live target row.
                self._purge_expired_pending_tasks_locked(
                    now, exclude_conversation_hash=conversation_hash
                )
                if str(row["state"]) != "pending":
                    self._connection.execute("COMMIT")
                    return PendingTaskClaim(None, str(row["state"]))
                last_received = json.loads(str(row["draft_json"])).get("last_received_at", "")
                if last_received and now < datetime.fromisoformat(last_received):
                    self._connection.execute("COMMIT")
                    return PendingTaskClaim(None, "stale")
                self._connection.execute(
                    """
                    UPDATE pending_task_clarifications
                    SET state = 'claimed', claimed_by_message_id = ?, updated_at = ?
                    WHERE conversation_hash = ? AND state = 'pending'
                    """,
                    (message_id[:300], self._now(), conversation_hash),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        try:
            return PendingTaskClaim(self._pending_from_row(row), "claimed")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.clear_pending_task(conversation_key)
            return PendingTaskClaim(None, "invalid")

    def release_pending_task(
        self,
        conversation_key: str,
        message_id: str,
        pending: PendingTaskClarification | None = None,
    ) -> bool:
        conversation_hash = self._hash(conversation_key)
        assignments = "state = 'pending', claimed_by_message_id = '', updated_at = ?"
        parameters: list[object] = [self._now()]
        if pending is not None:
            assignments += ", reason = ?, draft_json = ?, reminder_date = ?, reminder_time = ?"
            parameters.extend(
                (
                    pending.reason.value,
                    self._pending_draft_json(pending),
                    pending.reminder_date,
                    pending.reminder_time,
                )
            )
        parameters.extend((conversation_hash, message_id[:300]))
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE pending_task_clarifications SET {assignments}
                WHERE conversation_hash = ? AND state = 'claimed'
                  AND claimed_by_message_id = ?
                """,
                tuple(parameters),
            )
        return cursor.rowcount == 1

    def mark_pending_task_uncertain(
        self, conversation_key: str, message_id: str
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE pending_task_clarifications
                SET state = 'uncertain', draft_json = '{}', reason = '',
                    reminder_date = '', reminder_time = '', source_message_id = '',
                    updated_at = ?
                WHERE conversation_hash = ? AND claimed_by_message_id = ?
                """,
                (self._now(), self._hash(conversation_key), message_id[:300]),
            )

    def complete_pending_task(self, conversation_key: str, message_id: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                DELETE FROM pending_task_clarifications
                WHERE conversation_hash = ? AND claimed_by_message_id = ?
                """,
                (self._hash(conversation_key), message_id[:300]),
            )

    def clear_pending_task(self, conversation_key: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM pending_task_clarifications WHERE conversation_hash = ?",
                (self._hash(conversation_key),),
            )

    def abandon_pending_task(self, conversation_key: str) -> None:
        """Let a clearly new request replace only an unclaimed clarification."""

        with self._lock:
            self._connection.execute(
                """
                DELETE FROM pending_task_clarifications
                WHERE conversation_hash = ? AND state = 'pending'
                """,
                (self._hash(conversation_key),),
            )

    def mark_task_completed(self, sender_key: str, task_id: str, *, conversation_key: str = "") -> None:
        sender_hash = self._hash(sender_key)
        route = sender_key.split(":", 2)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    UPDATE task_context SET task_status = 'completed'
                    WHERE sender_hash = ? AND task_id = ?
                    """,
                    (sender_hash, task_id),
                )
                conversation_hashes = {self._hash(conversation_key)} if conversation_key else set()
                if len(route) == 3:
                    chats = self._connection.execute(
                        "SELECT DISTINCT chat_id FROM reminders WHERE task_id = ? AND platform = ? AND account_id = ? AND user_id = ?",
                        (task_id, *route),
                    ).fetchall()
                    conversation_hashes.update(self._hash(f"{sender_key}:{chat[0]}") for chat in chats)
                    self._connection.execute(
                        """
                        UPDATE reminders SET status = 'cancelled'
                        WHERE task_id = ? AND platform = ? AND account_id = ?
                          AND user_id = ?
                          AND status IN ('pending', 'failed', 'delivering')
                        """,
                        (task_id, route[0], route[1], route[2]),
                    )
                else:
                    self._connection.execute(
                        """
                        UPDATE reminders SET status = 'cancelled'
                        WHERE task_id = ?
                          AND status IN ('pending', 'failed', 'delivering')
                        """,
                        (task_id,),
                    )
                for conversation_hash in conversation_hashes:
                    self._connection.execute("UPDATE task_context SET task_status = 'completed' WHERE sender_hash = ? AND task_id = ?", (conversation_hash, task_id))
                    action = self._connection.execute("SELECT payload FROM pending_reminder_actions WHERE conversation_hash = ?", (conversation_hash,)).fetchone()
                    if action and json.loads(action[0]).get("task", {}).get("task_id") == task_id:
                        self._connection.execute("DELETE FROM pending_reminder_actions WHERE conversation_hash = ?", (conversation_hash,))
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _reminder_route(row: sqlite3.Row) -> tuple[str, str, str, str]:
        return (
            str(row["platform"]),
            str(row["account_id"]),
            str(row["user_id"]),
            str(row["chat_id"]),
        )

    def reminder_snapshot(self, task_id: str, message: MessageEnvelope) -> tuple[tuple[object, ...], ...]:
        """Version token and schedule, restricted to the exact delivery route."""
        with self._lock:
            rows = self._connection.execute(
                """SELECT id, status, reminder_at, source_message_id FROM reminders
                   WHERE task_id = ? AND platform = ? AND account_id = ?
                     AND user_id = ? AND chat_id = ? ORDER BY id""",
                (task_id, message.platform, message.account_id, message.user_id, message.chat_id),
            ).fetchall()
            return tuple(tuple(row) for row in rows)

    def cancel_reminders(self, task_id: str, message: MessageEnvelope, *, scope: str,
                         expected_snapshot: tuple[tuple[object, ...], ...]) -> tuple[int, bool]:
        """Cancel unsent local notifications only; never complete a Dida task."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = self.reminder_snapshot(task_id, message)
                if current != expected_snapshot:
                    raise ValueError("提醒状态已经变化，请重新确认取消范围")
                unresolved = any(row[1] in {"sending", "uncertain"} for row in current)
                eligible = sorted((row for row in current if row[1] in {"pending", "failed", "delivering"}), key=lambda row: str(row[2]))
                if scope == "next":
                    # Never silently cancel the *following* occurrence when the
                    # next one may already be in flight.
                    if unresolved:
                        raise ValueError("本次提醒正在发送或结果待确认，未取消其他次数")
                    eligible = eligible[:1]
                changed = 0
                for row in eligible:
                    changed += self._connection.execute(
                        "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status IN ('pending', 'failed', 'delivering')",
                        (row[0],),
                    ).rowcount
                self._connection.execute("COMMIT")
                return changed, unresolved
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def set_pending_reminder_action(self, message: MessageEnvelope, payload: dict[str, object], expires_at: datetime) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO pending_reminder_actions VALUES (?, ?, ?)",
                (self._hash(message.conversation_key), json.dumps(payload, ensure_ascii=False), expires_at.isoformat()),
            )

    def pending_reminder_action(self, message: MessageEnvelope) -> dict[str, object] | None:
        with self._lock:
            self._purge_pending_reminder_actions_locked(message.received_at)
            row = self._connection.execute("SELECT payload FROM pending_reminder_actions WHERE conversation_hash = ?", (self._hash(message.conversation_key),)).fetchone()
            return json.loads(row[0]) if row else None

    def _purge_pending_reminder_actions_locked(self, now: datetime) -> None:
        rows = self._connection.execute("SELECT conversation_hash, expires_at FROM pending_reminder_actions").fetchall()
        for row in rows:
            if self._pending_expired(row["expires_at"], now):
                self._connection.execute("DELETE FROM pending_reminder_actions WHERE conversation_hash = ?", (row["conversation_hash"],))

    def clear_pending_reminder_action(self, message: MessageEnvelope) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM pending_reminder_actions WHERE conversation_hash = ?", (self._hash(message.conversation_key),))

    def begin_reminder_delivery(self, row_ids: Iterable[int]) -> bool:
        """Atomically cross the last cancellable boundary before network I/O."""
        ids = tuple(row_ids)
        if not ids:
            return False
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                marks = ",".join("?" for _ in ids)
                rows = self._connection.execute(f"SELECT status FROM reminders WHERE id IN ({marks})", ids).fetchall()
                ready = len(rows) == len(ids) and all(row[0] == "delivering" for row in rows)
                if ready:
                    self._connection.execute(f"UPDATE reminders SET status = 'sending' WHERE id IN ({marks})", ids)
                else:
                    # A merged batch changed after claim. Retry only the still
                    # eligible entries; never send stale content containing a cancelled item.
                    self._connection.execute(f"UPDATE reminders SET status = 'failed' WHERE id IN ({marks}) AND status = 'delivering'", ids)
                self._connection.execute("COMMIT")
                return ready
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def enqueue_reminders(
        self,
        message: MessageEnvelope,
        task: TaskReference,
        reminder_ats: Iterable[datetime],
        *,
        replace_existing: bool = False,
        expected_snapshot: tuple[tuple[object, ...], ...] | None = None,
    ) -> tuple[int, tuple[int, ...]]:
        """Atomically enqueue a finite reminder series.

        The existing `(task_id, reminder_at)` key remains the durable
        idempotency boundary. A collision from another delivery route is
        rejected instead of silently transferring the reminder to that route.
        """

        reminder_texts = tuple(item.isoformat() for item in reminder_ats)
        if not reminder_texts:
            raise ValueError("提醒时间不能为空")
        if len(set(reminder_texts)) != len(reminder_texts):
            raise ValueError("提醒时间不能重复")
        if not task.task_id:
            raise ValueError("提醒缺少稳定 task_id")

        route = (message.platform, message.account_id, message.user_id, message.chat_id)
        placeholders = ", ".join("?" for _ in reminder_texts)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = self.reminder_snapshot(task.task_id, message)
                if expected_snapshot is not None and current != expected_snapshot:
                    raise ValueError("提醒状态已经变化，请重新确认后再修改")
                if replace_existing and any(row[1] in {"sending", "uncertain"} for row in current):
                    raise ValueError("原提醒正在发送或结果待确认，不能保证撤回，因此未改动")
                existing_rows = self._connection.execute(
                    f"""
                    SELECT * FROM reminders
                    WHERE task_id = ? AND reminder_at IN ({placeholders})
                    """,
                    (task.task_id, *reminder_texts),
                ).fetchall()
                existing = {
                    str(row["reminder_at"]): row for row in existing_rows
                }
                for row in existing_rows:
                    if self._reminder_route(row) != route:
                        raise ReminderRouteConflictError(
                            "相同任务和时间已绑定到其他微信会话"
                        )

                if replace_existing:
                    protected_statuses = {
                        str(row["status"])
                        for row in existing_rows
                    }
                    if not protected_statuses or not protected_statuses.issubset(
                        {"sent", "cancelled"}
                    ):
                        self._connection.execute(
                            f"""
                            UPDATE reminders SET status = 'rescheduled'
                            WHERE task_id = ?
                              AND platform = ? AND account_id = ?
                              AND user_id = ? AND chat_id = ?
                              AND reminder_at NOT IN ({placeholders})
                              AND status IN ('pending', 'failed', 'delivering')
                            """,
                            (
                                task.task_id,
                                *route,
                                *reminder_texts,
                            ),
                        )

                changed = 0
                row_ids: list[int] = []
                for reminder_text in reminder_texts:
                    row = existing.get(reminder_text)
                    if row is None:
                        cursor = self._connection.execute(
                            """
                            INSERT INTO reminders(
                                task_id, title, category, project_id, platform,
                                account_id, user_id, chat_id, source_message_id,
                                reminder_at, status, next_attempt_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                            """,
                            (
                                task.task_id,
                                task.title[:500],
                                task.category[:200],
                                task.project_id[:300],
                                *route,
                                message.message_id,
                                reminder_text,
                                reminder_text,
                            ),
                        )
                        changed += 1
                        row_ids.append(int(cursor.lastrowid))
                        continue

                    row_id = int(row["id"])
                    row_ids.append(row_id)
                    self._connection.execute(
                        """
                        UPDATE reminders SET title = ?, category = ?, project_id = ?
                        WHERE id = ?
                        """,
                        (
                            task.title[:500],
                            task.category[:200],
                            task.project_id[:300],
                            row_id,
                        ),
                    )
                    if replace_existing and str(row["status"]) in {
                        "failed",
                        "rescheduled",
                    }:
                        activation = self._connection.execute(
                            """
                            UPDATE reminders SET status = 'pending',
                                next_attempt_at = ?, last_error_code = '',
                                source_message_id = ?
                            WHERE id = ? AND status IN ('failed', 'rescheduled')
                            """,
                            (reminder_text, message.message_id, row_id),
                        )
                        changed += activation.rowcount

                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return changed, tuple(row_ids)

    def enqueue_reminder(
        self,
        message: MessageEnvelope,
        task: TaskReference,
        reminder_at: datetime,
        *,
        replace_existing: bool = False,
    ) -> tuple[bool, int]:
        changed, row_ids = self.enqueue_reminders(
            message,
            task,
            (reminder_at,),
            replace_existing=replace_existing,
        )
        return changed > 0, row_ids[0]

    def active_reminder_count(
        self,
        task_id: str,
        message: MessageEnvelope | None = None,
    ) -> int:
        route_clause = ""
        parameters: tuple[object, ...] = (task_id,)
        if message is not None:
            route_clause = (
                " AND platform = ? AND account_id = ? AND user_id = ? AND chat_id = ?"
            )
            parameters = (
                task_id,
                message.platform,
                message.account_id,
                message.user_id,
                message.chat_id,
            )
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS total FROM reminders
                WHERE task_id = ? AND status IN ('pending', 'failed', 'delivering')
                """
                + route_clause,
                parameters,
            ).fetchone()
        return int(row["total"]) if row is not None else 0

    def claim_due_reminders(
        self, now: datetime, *, limit: int = 100, stale_seconds: int = 600
    ) -> tuple[ReminderRecord, ...]:
        now_text = now.isoformat()
        stale_before = (now - timedelta(seconds=stale_seconds)).isoformat()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    UPDATE reminders SET status = 'uncertain',
                        last_error_code = 'delivery-outcome-unknown'
                    WHERE status IN ('delivering', 'sending') AND last_attempt_at < ?
                    """,
                    (stale_before,),
                )
                rows = self._connection.execute(
                    """
                    SELECT * FROM reminders
                    WHERE status IN ('pending', 'failed')
                      AND reminder_at <= ? AND next_attempt_at <= ?
                    ORDER BY reminder_at, id LIMIT ?
                    """,
                    (now_text, now_text, int(limit)),
                ).fetchall()
                ids = [int(row["id"]) for row in rows]
                for row_id in ids:
                    self._connection.execute(
                        """
                        UPDATE reminders SET status = 'delivering', attempts = attempts + 1,
                            last_attempt_at = ? WHERE id = ?
                        """,
                        (now_text, row_id),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return tuple(
            ReminderRecord(
                row_id=int(row["id"]),
                task=TaskReference(
                    task_id=str(row["task_id"]),
                    title=str(row["title"]),
                    category=str(row["category"]),
                    project_id=str(row["project_id"]),
                ),
                platform=str(row["platform"]),
                account_id=str(row["account_id"]),
                user_id=str(row["user_id"]),
                chat_id=str(row["chat_id"]),
                source_message_id=str(row["source_message_id"]),
                reminder_at=datetime.fromisoformat(str(row["reminder_at"])),
                attempts=int(row["attempts"]) + 1,
            )
            for row in rows
        )

    def mark_reminders_sent(
        self, row_ids: Iterable[int], delivered_at: datetime, delivered_message_id: str = ""
    ) -> None:
        with self._lock:
            for row_id in row_ids:
                self._connection.execute(
                    """
                    UPDATE reminders SET status = 'sent', delivered_at = ?,
                        delivered_message_id = ?, last_error_code = ''
                    WHERE id = ? AND status IN ('pending', 'failed', 'delivering', 'sending')
                    """,
                    (delivered_at.isoformat(), delivered_message_id[:300], int(row_id)),
                )

    def mark_reminders_failed(
        self,
        row_ids: Iterable[int],
        error_code: str,
        retry_at: datetime,
    ) -> None:
        with self._lock:
            for row_id in row_ids:
                self._connection.execute(
                    """
                    UPDATE reminders SET status = 'failed', last_error_code = ?,
                        next_attempt_at = ?
                    WHERE id = ? AND status IN ('pending', 'failed', 'delivering', 'sending')
                    """,
                    (error_code[:80], retry_at.isoformat(), int(row_id)),
                )

    def mark_reminders_uncertain(
        self,
        row_ids: Iterable[int],
        error_code: str = "delivery-outcome-unknown",
    ) -> None:
        """Record an ambiguous send result as terminal and never auto-retry it.

        The compare-and-set condition preserves a concurrent cancellation or
        reschedule instead of allowing a late sender result to overwrite it.
        """

        safe_code = (
            error_code
            if error_code
            in {
                "delivery-timeout",
                "delivery-cancelled",
                "adapter-reported-failure",
                "delivery-exception",
                "delivery-outcome-unknown",
            }
            else "delivery-outcome-unknown"
        )
        with self._lock:
            for row_id in row_ids:
                self._connection.execute(
                    """
                    UPDATE reminders SET status = 'uncertain', last_error_code = ?
                    WHERE id = ? AND status IN ('delivering', 'sending')
                    """,
                    (safe_code, int(row_id)),
                )

    def reminder_status(self, task_id: str, reminder_at: datetime) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM reminders WHERE task_id = ? AND reminder_at = ?",
                (task_id, reminder_at.isoformat()),
            ).fetchone()
        return str(row["status"]) if row else None

    def claim_daily_run(self, local_date: str, job_name: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO daily_runs(local_date, job_name, claimed_at)
                VALUES (?, ?, ?)
                """,
                (local_date, job_name, self._now()),
            )
            return cursor.rowcount == 1

    def state_for(self, message: MessageEnvelope) -> str | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT state FROM messages
                WHERE platform = ? AND account_id = ? AND message_id = ?
                """,
                message.identity_key,
            ).fetchone()
        return str(row["state"]) if row else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
