from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable

from .config import SecretarySettings
from .ledger import IdempotencyLedger, ReminderRecord
from .models import ActionResult, ExecutionStatus, MessageEnvelope, TaskDraft, TaskReference


ReminderSender = Callable[[ReminderRecord, str], str]


@dataclass(frozen=True)
class DispatchStats:
    claimed: int = 0
    sent: int = 0
    failed: int = 0
    merged_messages: int = 0


class ReminderQueue:
    def __init__(self, settings: SecretarySettings, ledger: IdempotencyLedger):
        self.settings = settings
        self.ledger = ledger

    def schedule(
        self,
        draft: TaskDraft,
        task: TaskReference,
        message: MessageEnvelope,
        *,
        replace_existing: bool = False,
    ) -> ActionResult:
        try:
            reminder_at = datetime.fromisoformat(draft.reminder_at)
        except ValueError:
            return ActionResult(
                action="reminder",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination="微信",
                error="提醒时间格式无效",
            )
        if reminder_at.tzinfo is None:
            return ActionResult(
                action="reminder",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination="微信",
                error="提醒时间缺少 Asia/Shanghai 时区",
            )
        reminder_at = reminder_at.astimezone(self.settings.tz)
        if reminder_at < message.received_at.astimezone(self.settings.tz) - timedelta(minutes=1):
            return ActionResult(
                action="reminder",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination="微信",
                error="提醒时间已经过去，请重新指定",
            )
        if not self.settings.dry_run and not self.settings.reminders_enabled:
            return ActionResult(
                action="reminder",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination="微信",
                error="本地微信提醒调度器尚未获准启用",
            )
        created, row_id = self.ledger.enqueue_reminder(
            message, task, reminder_at, replace_existing=replace_existing
        )
        status = ExecutionStatus.PLANNED if self.settings.dry_run else ExecutionStatus.SUCCEEDED
        label = reminder_at.strftime("%Y-%m-%d %H:%M")
        return ActionResult(
            action="reminder",
            status=status if created else ExecutionStatus.SKIPPED,
            summary=f"{label}｜{task.title}",
            destination="微信",
            external_id=f"reminder:{row_id}",
            preview=f"task_id={task.task_id}\nreminder_at={reminder_at.isoformat()}",
            task_refs=(task,),
        )


class ReminderScheduler:
    """Project-local queue dispatcher; disabled unless explicitly enabled in config."""

    def __init__(self, settings: SecretarySettings, ledger: IdempotencyLedger):
        self.settings = settings
        self.ledger = ledger
        self._sender: ReminderSender | None = None
        self._sender_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def attach(self, sender: ReminderSender) -> None:
        with self._sender_lock:
            self._sender = sender
        if self.settings.reminders_enabled and self._thread is None:
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="wechat-secretary-reminders",
            )
            self._thread.start()

    def stop(self, timeout: float = 5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._sender_lock:
                sender = self._sender
            if sender is not None:
                self.poll_once(sender, datetime.now(self.settings.tz))
            self._stop.wait(self.settings.reminder_poll_seconds)

    @staticmethod
    def _route_key(record: ReminderRecord) -> tuple[str, str, str, str]:
        return record.platform, record.account_id, record.user_id, record.chat_id

    def _record_sent_context(
        self,
        records: Iterable[ReminderRecord],
        now: datetime,
        batch_id: str,
        delivered_message_id: str,
    ) -> None:
        records = tuple(records)
        if not records:
            return
        first = records[0]
        sender_key = f"{first.platform}:{first.account_id}:{first.user_id}"
        self.ledger.record_task_context(
            sender_key,
            (record.task for record in records),
            batch_id=batch_id,
            source_message_id=delivered_message_id or batch_id,
            observed_at=now,
            ttl_seconds=self.settings.completion_context_ttl_seconds,
            context_kind="reminder",
            reminder_at=min(record.reminder_at for record in records),
        )

    def _deliver_group(
        self,
        sender: ReminderSender,
        records: tuple[ReminderRecord, ...],
        content: str,
        now: datetime,
        batch_id: str,
    ) -> bool:
        try:
            delivered_message_id = sender(records[0], content) or ""
        except Exception as exc:
            self.ledger.mark_reminders_failed(
                (record.row_id for record in records),
                f"send-{type(exc).__name__}",
                now + timedelta(seconds=self.settings.reminder_retry_seconds),
            )
            return False
        self.ledger.mark_reminders_sent(
            (record.row_id for record in records), now, delivered_message_id
        )
        self._record_sent_context(records, now, batch_id, delivered_message_id)
        return True

    def poll_once(self, sender: ReminderSender, now: datetime) -> DispatchStats:
        due = self.ledger.claim_due_reminders(now)
        if not due:
            return DispatchStats()
        routes: dict[tuple[str, str, str, str], list[ReminderRecord]] = defaultdict(list)
        for record in due:
            routes[self._route_key(record)].append(record)

        sent = failed = merged = 0
        threshold = now - timedelta(seconds=self.settings.reminder_overdue_merge_seconds)
        for route_records in routes.values():
            old = tuple(record for record in route_records if record.reminder_at < threshold)
            recent = tuple(record for record in route_records if record.reminder_at >= threshold)
            if old:
                titles = [record.task.title for record in old[:10]]
                remainder = len(old) - len(titles)
                content = "抱歉，刚刚有几条提醒晚到了，我现在补给你：\n" + "\n".join(
                    f"- {title}" for title in titles
                )
                if remainder:
                    content += f"\n- 另有 {remainder} 项"
                ok = self._deliver_group(
                    sender,
                    old,
                    content,
                    now,
                    "reminder-merged:" + ",".join(str(item.row_id) for item in old),
                )
                sent += len(old) if ok else 0
                failed += 0 if ok else len(old)
                merged += 1
            for record in recent:
                title = record.task.title.strip().rstrip("。！!？?，,；; ") or "这件事"
                content = f"提醒你一下，别忘了{title}。"
                ok = self._deliver_group(
                    sender,
                    (record,),
                    content,
                    now,
                    f"reminder:{record.row_id}",
                )
                sent += 1 if ok else 0
                failed += 0 if ok else 1
        return DispatchStats(
            claimed=len(due), sent=sent, failed=failed, merged_messages=merged
        )
