from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .models import ActionResult, ExecutionStatus, MessageEnvelope, TaskReference


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


class IdempotencyLedger:
    """Durable local state. It never stores inbound message bodies or credentials."""

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
                CREATE TABLE IF NOT EXISTS acknowledgements (
                    sender_hash TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    acknowledged_at TEXT NOT NULL,
                    PRIMARY KEY (sender_hash, message_id)
                );
                """
            )
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
                SELECT batch_id FROM task_context
                WHERE sender_hash = ? AND task_status != 'completed' AND expires_at >= ?
                ORDER BY observed_at DESC LIMIT 1
                """,
                (sender_hash, now.isoformat()),
            ).fetchone()
            if latest is not None:
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

    def mark_task_completed(self, sender_key: str, task_id: str) -> None:
        sender_hash = self._hash(sender_key)
        with self._lock:
            self._connection.execute(
                """
                UPDATE task_context SET task_status = 'completed'
                WHERE sender_hash = ? AND task_id = ?
                """,
                (sender_hash, task_id),
            )
            self._connection.execute(
                """
                UPDATE reminders SET status = 'cancelled'
                WHERE task_id = ? AND status IN ('pending', 'failed', 'delivering')
                """,
                (task_id,),
            )

    def enqueue_reminder(
        self,
        message: MessageEnvelope,
        task: TaskReference,
        reminder_at: datetime,
        *,
        replace_existing: bool = False,
    ) -> tuple[bool, int]:
        reminder_text = reminder_at.isoformat()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if replace_existing:
                    self._connection.execute(
                        """
                        UPDATE reminders SET status = 'rescheduled'
                        WHERE task_id = ? AND reminder_at != ?
                          AND status IN ('pending', 'failed', 'delivering')
                        """,
                        (task.task_id, reminder_text),
                    )
                cursor = self._connection.execute(
                    """
                    INSERT OR IGNORE INTO reminders(
                        task_id, title, category, project_id, platform, account_id,
                        user_id, chat_id, source_message_id, reminder_at, status,
                        next_attempt_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        task.task_id,
                        task.title[:500],
                        task.category[:200],
                        task.project_id[:300],
                        message.platform,
                        message.account_id,
                        message.user_id,
                        message.chat_id,
                        message.message_id,
                        reminder_text,
                        reminder_text,
                    ),
                )
                created = cursor.rowcount == 1
                reactivated = False
                if cursor.rowcount == 0:
                    self._connection.execute(
                        """
                        UPDATE reminders SET
                            title = ?, category = ?, project_id = ?, platform = ?,
                            account_id = ?, user_id = ?, chat_id = ?,
                            source_message_id = ?
                        WHERE task_id = ? AND reminder_at = ?
                        """,
                        (
                            task.title[:500],
                            task.category[:200],
                            task.project_id[:300],
                            message.platform,
                            message.account_id,
                            message.user_id,
                            message.chat_id,
                            message.message_id,
                            task.task_id,
                            reminder_text,
                        ),
                    )
                    if replace_existing:
                        activation = self._connection.execute(
                            """
                            UPDATE reminders SET status = 'pending',
                                next_attempt_at = ?, last_error_code = ''
                            WHERE task_id = ? AND reminder_at = ?
                              AND status IN ('failed', 'rescheduled')
                            """,
                            (reminder_text, task.task_id, reminder_text),
                        )
                        reactivated = activation.rowcount == 1
                row = self._connection.execute(
                    "SELECT id FROM reminders WHERE task_id = ? AND reminder_at = ?",
                    (task.task_id, reminder_text),
                ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return created or reactivated, int(row["id"])

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
                    UPDATE reminders SET status = 'failed', next_attempt_at = ?
                    WHERE status = 'delivering' AND last_attempt_at < ?
                    """,
                    (now_text, stale_before),
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
                        delivered_message_id = ?, last_error_code = '' WHERE id = ?
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
                        next_attempt_at = ? WHERE id = ?
                    """,
                    (error_code[:80], retry_at.isoformat(), int(row_id)),
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
