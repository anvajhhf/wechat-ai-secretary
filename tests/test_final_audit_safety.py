from __future__ import annotations

import ast
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from wechat_secretary.config import SecretarySettings
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.models import (
    ActionResult,
    ClarificationReason,
    ExecutionStatus,
    IntentKind,
    IntentPlan,
    MessageEnvelope,
    NoteDraft,
    PendingTaskClarification,
    ReminderRecurrence,
    TaskDraft,
)
from wechat_secretary.obsidian import ObsidianExecutor
from wechat_secretary.private_inbox import PrivateInboxExecutor
from wechat_secretary.reminders import ReminderQueue
from wechat_secretary.service import SecretaryService


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = SecretarySettings(
    project_root=ROOT,
    dry_run=True,
    allowed_users=frozenset({"wx-user-1"}),
    account_id="dry-account",
)
TZ = SETTINGS.tz
NOW = datetime(2026, 8, 24, 9, 0, tzinfo=TZ)


def message(message_id: str, text: str, when: datetime = NOW) -> MessageEnvelope:
    return MessageEnvelope(
        platform="weixin",
        account_id="dry-account",
        user_id="wx-user-1",
        chat_id="chat-1",
        chat_type="dm",
        message_id=message_id,
        text=text,
        received_at=when,
    )


class StaticTaskClassifier:
    call_count = 0

    def classify(self, *args: object, **kwargs: object) -> IntentPlan:
        del args, kwargs
        self.call_count += 1
        return IntentPlan(
            kind=IntentKind.TASK,
            tasks=(TaskDraft("买B2M抗体"),),
            confidence=0.95,
        )


class FailedBeforeSendDida:
    def __init__(self) -> None:
        self.create_calls = 0

    def create_task(
        self, task: TaskDraft, incoming: MessageEnvelope
    ) -> ActionResult:
        del incoming
        self.create_calls += 1
        return ActionResult(
            action="task",
            status=ExecutionStatus.FAILED,
            summary=task.title,
            error="滴答连接不可用，创建请求未发出，任务未创建。",
        )

    @staticmethod
    def health_summary() -> dict[str, str]:
        return {}


class StaticPlanClassifier:
    def __init__(self, plan: IntentPlan) -> None:
        self.plan = plan
        self.call_count = 0

    def classify(self, *args: object, **kwargs: object) -> IntentPlan:
        del args, kwargs
        self.call_count += 1
        return self.plan


class RecordingNotes:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def available_links(content: str) -> tuple[str, ...]:
        del content
        return ()

    def save(self, note: NoteDraft, incoming: MessageEnvelope) -> ActionResult:
        del incoming
        self.calls += 1
        return ActionResult(
            action="note",
            status=ExecutionStatus.PLANNED,
            summary=note.title,
        )


class FinalAuditSafetyTests(unittest.TestCase):
    def test_model_cannot_authorize_writes_for_questions_or_status_text(self) -> None:
        dida = FailedBeforeSendDida()
        ledger = IdempotencyLedger(":memory:")
        self.addCleanup(ledger.close)
        classifier = StaticPlanClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(
                    TaskDraft(
                        "买B2M抗体",
                        due_date="2026-08-25",
                        due_time="15:00",
                        reminder_at="2026-08-25T15:00+08:00",
                    ),
                ),
                confidence=0.99,
            )
        )
        service = SecretaryService(
            settings=SETTINGS,
            ledger=ledger,
            classifier=classifier,
            dida=dida,
            obsidian=ObsidianExecutor(SETTINGS),
            private_inbox=PrivateInboxExecutor(SETTINGS),
            reminders=ReminderQueue(SETTINGS, ledger),
        )

        for index, text in enumerate(
            (
                "为什么明天3点提醒我买B2M抗体？",
                "系统会在明天3点提醒我买B2M抗体",
                "B2M抗体今天到了",
            )
        ):
            with self.subTest(text=text):
                result = service.handle(message(f"adversarial-task-{index}", text))
                self.assertIs(result.status, ExecutionStatus.SKIPPED)
                self.assertEqual((), result.results)

        self.assertEqual(0, dida.create_calls)

        notes = RecordingNotes()
        note_ledger = IdempotencyLedger(":memory:")
        self.addCleanup(note_ledger.close)
        note_service = SecretaryService(
            settings=SETTINGS,
            ledger=note_ledger,
            classifier=StaticPlanClassifier(
                IntentPlan(
                    kind=IntentKind.NOTE,
                    notes=(NoteDraft("到货", "B2M抗体今天到了"),),
                    confidence=0.99,
                )
            ),
            dida=FailedBeforeSendDida(),
            obsidian=notes,
            private_inbox=PrivateInboxExecutor(SETTINGS),
            reminders=ReminderQueue(SETTINGS, note_ledger),
        )
        note_result = note_service.handle(message("adversarial-note", "B2M抗体今天到了"))
        self.assertIs(note_result.status, ExecutionStatus.SKIPPED)
        self.assertEqual(0, notes.calls)

    def test_failed_followup_replay_does_not_latch_pending_uncertain(self) -> None:
        ledger = IdempotencyLedger(":memory:")
        self.addCleanup(ledger.close)
        dida = FailedBeforeSendDida()
        service = SecretaryService(
            settings=SETTINGS,
            ledger=ledger,
            classifier=StaticTaskClassifier(),
            dida=dida,
            obsidian=ObsidianExecutor(SETTINGS),
            private_inbox=PrivateInboxExecutor(SETTINGS),
            reminders=ReminderQueue(SETTINGS, ledger),
        )
        source = message("source", "下周二提醒我买B2M抗体")
        service.handle(source)
        followup = message("followup", "上午9点", NOW + timedelta(minutes=1))

        first = service.handle(followup)
        replay = service.handle(followup)
        pending = ledger.claim_pending_task(
            source.conversation_key, "probe", NOW + timedelta(minutes=2)
        )

        self.assertIs(first.status, ExecutionStatus.FAILED)
        self.assertIs(replay.status, ExecutionStatus.UNCERTAIN)
        self.assertEqual(1, dida.create_calls)
        self.assertEqual("claimed", pending.state)
        self.assertIsNotNone(pending.pending)

    def test_pending_payload_is_minimized_scrubbed_and_globally_expired(self) -> None:
        ledger = IdempotencyLedger(":memory:")
        self.addCleanup(ledger.close)
        base = datetime.now(TZ)
        pending = PendingTaskClarification(
            reason=ClarificationReason.MISSING_REMINDER_TIME,
            task=TaskDraft(
                "买B2M抗体",
                category="SECRET-CATEGORY",
                tags=("SECRET-TAG",),
                description="SECRET-DESCRIPTION",
                reminder_recurrence=ReminderRecurrence(
                    frequency="weekly", interval=1, weekday=2, count=3
                ),
            ),
            reminder_date="2026-09-01",
            source_message_id="source",
        )
        conversation = message("source", "").conversation_key
        ledger.set_pending_task(conversation, pending, base + timedelta(minutes=10))
        row = ledger._connection.execute(
            "SELECT draft_json FROM pending_task_clarifications"
        ).fetchone()
        serialized = str(row["draft_json"])

        self.assertIn("买B2M抗体", serialized)
        self.assertNotIn("SECRET-DESCRIPTION", serialized)
        self.assertNotIn("SECRET-CATEGORY", serialized)
        self.assertNotIn("SECRET-TAG", serialized)

        claimed = ledger.claim_pending_task(conversation, "reply", base)
        self.assertIsNotNone(claimed.pending)
        ledger.mark_pending_task_uncertain(conversation, "reply")
        scrubbed = ledger._connection.execute(
            "SELECT draft_json, reminder_date, source_message_id "
            "FROM pending_task_clarifications"
        ).fetchone()
        self.assertEqual({}, json.loads(str(scrubbed["draft_json"])))
        self.assertEqual("", scrubbed["reminder_date"])
        self.assertEqual("", scrubbed["source_message_id"])

        expired_conversation = replace(
            message("expired", ""), chat_id="expired-chat"
        ).conversation_key
        ledger.set_pending_task(
            expired_conversation, pending, base - timedelta(seconds=1)
        )
        self.assertEqual(1, ledger.purge_expired_pending_tasks(base))
        remaining = ledger._connection.execute(
            "SELECT COUNT(*) AS count FROM pending_task_clarifications"
        ).fetchone()
        self.assertEqual(1, int(remaining["count"]))

    def test_gateway_ready_hook_is_fired_again_after_reconnect(self) -> None:
        runtime = ROOT / "runtime" / "hermes-agent" / "gateway" / "run.py"
        tree = ast.parse(runtime.read_text(encoding="utf-8"))
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GatewayRunner"
        )
        reconnect = next(
            node
            for node in runner.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_platform_reconnect_watcher"
        )
        triggers = {
            keyword.value.value
            for node in ast.walk(reconnect)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_notify_gateway_ready_hooks"
            for keyword in node.keywords
            if keyword.arg == "trigger"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        }
        self.assertIn("platform_reconnect", triggers)


if __name__ == "__main__":
    unittest.main()
