from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable

from .config import SecretarySettings
from .ledger import IdempotencyLedger, ReminderRecord, ReminderRouteConflictError
from .models import ActionResult, ExecutionStatus, MessageEnvelope, TaskDraft, TaskReference


ReminderSender = Callable[[ReminderRecord, str], str]
logger = logging.getLogger(__name__)

_SAFE_DELIVERY_REASON_CODES = frozenset(
    {
        "route-unavailable",
        "transport-not-ready",
        "dispatch-loop-closed",
        "dispatch-submit-failed",
        "delivery-timeout",
        "delivery-cancelled",
        "adapter-reported-failure",
        "delivery-exception",
        "delivery-outcome-unknown",
    }
)


class ReminderDeliveryError(RuntimeError):
    """Base class for delivery outcomes with a sanitized persistent reason."""

    default_reason_code = "delivery-outcome-unknown"

    def __init__(self, reason_code: str = ""):
        candidate = str(reason_code or "").strip().casefold()
        self.reason_code = (
            candidate
            if candidate in _SAFE_DELIVERY_REASON_CODES
            else self.default_reason_code
        )
        # Never surface an adapter/provider exception string through this object.
        super().__init__("reminder delivery outcome was not successful")


class ReminderDeliveryPreSendError(ReminderDeliveryError):
    """The sender has reliable evidence that no delivery attempt was made."""

    default_reason_code = "route-unavailable"


class ReminderDeliveryUncertainError(ReminderDeliveryError):
    """A delivery attempt may have succeeded and therefore must not be retried."""

    default_reason_code = "delivery-outcome-unknown"

_WEEKDAY_NAMES = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "日",
}


@dataclass(frozen=True)
class DispatchStats:
    claimed: int = 0
    sent: int = 0
    failed: int = 0
    uncertain: int = 0
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
        expected_snapshot: tuple[tuple[object, ...], ...] | None = None,
    ) -> ActionResult:
        try:
            reminder_at = datetime.fromisoformat(draft.reminder_at)
        except (TypeError, ValueError):
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

        recurrence = draft.reminder_recurrence
        reminder_ats = (reminder_at,)
        if recurrence is not None:
            frequency = str(recurrence.frequency or "").strip().casefold()
            if (
                frequency != "weekly"
                or recurrence.interval != 1
                or recurrence.weekday not in _WEEKDAY_NAMES
                or recurrence.count < 2
                or recurrence.count > 52
            ):
                return ActionResult(
                    action="reminder",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination="微信",
                    error="重复提醒规则无效；目前只支持每周一次、共 2 到 52 次",
                )
            if reminder_at.isoweekday() != recurrence.weekday:
                return ActionResult(
                    action="reminder",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination="微信",
                    error="首次提醒日期与每周重复星期不一致",
                )
            reminder_ats = tuple(
                reminder_at + timedelta(weeks=index)
                for index in range(recurrence.count)
            )

        try:
            changed, row_ids = self.ledger.enqueue_reminders(
                message,
                task,
                reminder_ats,
                replace_existing=replace_existing,
                expected_snapshot=expected_snapshot,
            )
        except ReminderRouteConflictError:
            return ActionResult(
                action="reminder",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination="微信",
                error="相同任务和时间已绑定到其他微信会话，没有覆盖原提醒",
            )
        except ValueError as exc:
            return ActionResult(
                action="reminder",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination="微信",
                error=str(exc),
            )

        status = ExecutionStatus.PLANNED if self.settings.dry_run else ExecutionStatus.SUCCEEDED
        if recurrence is None:
            label = reminder_at.strftime("%Y-%m-%d %H:%M")
        else:
            weekday = _WEEKDAY_NAMES[recurrence.weekday]
            label = (
                f"{reminder_at.strftime('%Y-%m-%d %H:%M')} 起，"
                f"每周{weekday}，共 {len(reminder_ats)} 次"
            )
        preview_times = "\n".join(item.isoformat() for item in reminder_ats)
        return ActionResult(
            action="reminder",
            status=status if changed else ExecutionStatus.SKIPPED,
            summary=f"{label}｜{task.title}",
            destination="微信",
            external_id=f"reminder:{row_ids[0]}",
            preview=f"task_id={task.task_id}\nreminder_ats=\n{preview_times}",
            task_refs=(task,),
        )


class ReminderScheduler:
    """Project-local queue dispatcher; disabled unless explicitly enabled in config."""

    def __init__(self, settings: SecretarySettings, ledger: IdempotencyLedger):
        self.settings = settings
        self.ledger = ledger
        self._sender: ReminderSender | None = None
        self._sender_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def attach(self, sender: ReminderSender) -> None:
        with self._sender_lock:
            self._sender = sender
        if not self.settings.reminders_enabled:
            return
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="wechat-secretary-reminders",
            )
            self._thread.start()

    def stop(self, timeout: float = 5) -> None:
        self._stop.set()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(self.settings.tz)
            try:
                self.ledger.purge_expired_pending_tasks(now)
            except Exception as exc:
                logger.warning(
                    "Pending clarification cleanup failed: %s", type(exc).__name__
                )
            with self._sender_lock:
                sender = self._sender
            if sender is not None:
                try:
                    self.poll_once(sender, now)
                except Exception as exc:
                    logger.exception(
                        "Reminder scheduler poll failed: %s", type(exc).__name__
                    )
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
            f"{sender_key}:{first.chat_id}",
            (record.task for record in records),
            batch_id=batch_id, source_message_id=delivered_message_id or batch_id,
            observed_at=now, ttl_seconds=self.settings.completion_context_ttl_seconds,
            context_kind="reminder", reminder_at=min(record.reminder_at for record in records),
        )
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
    ) -> str:
        if not self.ledger.begin_reminder_delivery(record.row_id for record in records):
            return "skipped"
        try:
            delivered_message_id = sender(records[0], content) or ""
        except ReminderDeliveryPreSendError as exc:
            self.ledger.mark_reminders_failed(
                (record.row_id for record in records),
                exc.reason_code,
                now + timedelta(seconds=self.settings.reminder_retry_seconds),
            )
            logger.warning("Reminder delivery stopped before send: %s", exc.reason_code)
            return "failed"
        except ReminderDeliveryUncertainError as exc:
            self.ledger.mark_reminders_uncertain(
                (record.row_id for record in records), exc.reason_code
            )
            logger.warning("Reminder delivery result is uncertain: %s", exc.reason_code)
            return "uncertain"
        except Exception as exc:
            # A generic sender exception does not prove that the platform did
            # not accept the message.  Failing closed here prevents an
            # automatic retry from producing a duplicate reminder.
            self.ledger.mark_reminders_uncertain(
                (record.row_id for record in records),
                "delivery-outcome-unknown",
            )
            logger.warning(
                "Reminder delivery raised an unclassified exception: %s",
                type(exc).__name__,
            )
            return "uncertain"
        self.ledger.mark_reminders_sent(
            (record.row_id for record in records), now, delivered_message_id
        )
        self._record_sent_context(records, now, batch_id, delivered_message_id)
        return "sent"

    def poll_once(self, sender: ReminderSender, now: datetime) -> DispatchStats:
        due = self.ledger.claim_due_reminders(now)
        if not due:
            return DispatchStats()
        routes: dict[tuple[str, str, str, str], list[ReminderRecord]] = defaultdict(list)
        for record in due:
            routes[self._route_key(record)].append(record)

        sent = failed = uncertain = merged = 0
        threshold = now - timedelta(seconds=self.settings.reminder_overdue_merge_seconds)
        for route_records in routes.values():
            old = tuple(record for record in route_records if record.reminder_at < threshold)
            recent = tuple(record for record in route_records if record.reminder_at >= threshold)
            if old:
                entries = [
                    (
                        record.reminder_at.astimezone(self.settings.tz).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        record.task.title,
                    )
                    for record in old[:10]
                ]
                remainder = len(old) - len(entries)
                content = (
                    "抱歉，以下提醒未按时送达，我现在补给你（原计划时间如下）：\n"
                    + "\n".join(
                        f"- {planned_at}｜{title}" for planned_at, title in entries
                    )
                )
                if remainder:
                    content += f"\n- 另有 {remainder} 项"
                outcome = self._deliver_group(
                    sender,
                    old,
                    content,
                    now,
                    "reminder-merged:" + ",".join(str(item.row_id) for item in old),
                )
                sent += len(old) if outcome == "sent" else 0
                failed += len(old) if outcome == "failed" else 0
                uncertain += len(old) if outcome == "uncertain" else 0
                merged += 1
            for record in recent:
                title = record.task.title.strip().rstrip("。！!？?，,；; ") or "这件事"
                content = f"提醒你一下，别忘了{title}。"
                outcome = self._deliver_group(
                    sender,
                    (record,),
                    content,
                    now,
                    f"reminder:{record.row_id}",
                )
                sent += 1 if outcome == "sent" else 0
                failed += 1 if outcome == "failed" else 0
                uncertain += 1 if outcome == "uncertain" else 0
        return DispatchStats(
            claimed=len(due),
            sent=sent,
            failed=failed,
            uncertain=uncertain,
            merged_messages=merged,
        )
