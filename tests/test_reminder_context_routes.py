"""All reminder entry points must target this conversation's exact task ID."""
from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timedelta

import test_voice_reminder_conversation as fixtures
from wechat_secretary.models import (
    ActionResult,
    ExecutionStatus,
    IntentKind,
    IntentPlan,
    TaskQuery,
    TaskReference,
)


NOW = fixtures.NOW


class QueryOnlyClassifier:
    def __init__(self):
        self.call_count = 0

    def classify(self, *args, **kwargs):
        self.call_count += 1
        return IntentPlan(kind=IntentKind.QUERY, query=TaskQuery("today"))


class ReminderContextRoutesTests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service
    message = staticmethod(fixtures.VoiceReminderConversationTests.message)

    def setup_a(self):
        service, classifier, media, dida, ledger = self.make_service()
        incoming = self.message("context-create-a", "今天下午三点提醒我任务A", media, voice=False)
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual("pending", ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
        return service, classifier, media, dida, ledger, incoming

    def test_relative_adjustment_does_not_use_another_chat_sender_context(self):
        service, classifier, media, dida, ledger, source = self.setup_a()
        snapshot = ledger.reminder_snapshot("voice-task-1", source)
        elsewhere = replace(
            self.message("context-other-chat", "半小时后提醒我", media, voice=False, minutes_later=1),
            chat_id="different-chat-same-user",
        )
        result = service.handle(elsewhere)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertFalse(result.results)
        self.assertEqual(snapshot, ledger.reminder_snapshot("voice-task-1", source))
        self.assertEqual(0, ledger.active_reminder_count("voice-task-1", elsewhere))
        self.assertEqual(0, classifier.call_count)
        self.assertEqual(1, len(dida.tasks))

    def test_standalone_chinese_relative_adjustment_preserves_seconds(self):
        service, classifier, media, dida, ledger, _ = self.setup_a()
        incoming = replace(
            self.message("context-relative-seconds", "二十分钟后提醒我", media, voice=False),
            received_at=NOW + timedelta(minutes=1, seconds=45),
        )
        result = service.handle(incoming)
        expected = incoming.received_at + timedelta(minutes=20)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual("pending", ledger.reminder_status("voice-task-1", expected))
        self.assertIsNone(ledger.reminder_status("voice-task-1", expected.replace(second=0)))
        self.assertEqual("rescheduled", ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
        self.assertEqual(1, ledger.active_reminder_count("voice-task-1", incoming))
        self.assertEqual(0, classifier.call_count)
        self.assertFalse(result.llm_called)
        self.assertEqual(1, len(dida.tasks))

    def test_querying_existing_task_b_updates_controls_without_touching_task_a(self):
        service, _, media, dida, ledger, source = self.setup_a()
        task_b = TaskReference("existing-task-b", "已有任务B", "Inbox", "inbox")
        original_b_at = NOW.replace(hour=17, minute=0)
        ledger.enqueue_reminder(source, task_b, original_b_at)
        dida.query_tasks = lambda query: ActionResult(
            "query", ExecutionStatus.PLANNED, "查询到已有任务B", task_refs=(task_b,)
        )
        classifier = QueryOnlyClassifier()
        service.classifier = classifier
        query_message = self.message("context-query-b", "查一下今天有哪些任务", media, voice=False, minutes_later=1)
        queried = service.handle(query_message)
        self.assertEqual(ExecutionStatus.PLANNED, queried.status)
        self.assertEqual((task_b,), ledger.recent_task_context(query_message.conversation_key, query_message.received_at).candidates)
        self.assertEqual((task_b,), ledger.recent_task_context(query_message.sender_key, query_message.received_at).candidates)
        update = self.message("context-update-queried-b", "改成四点", media, voice=False, minutes_later=2)
        service.handle(update)
        self.assertEqual("pending", ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
        self.assertEqual("rescheduled", ledger.reminder_status(task_b.task_id, original_b_at))
        self.assertEqual("pending", ledger.reminder_status(task_b.task_id, NOW.replace(hour=16, minute=0)))
        self.assertIsNone(ledger.reminder_status("voice-task-1", NOW.replace(hour=16, minute=0)))
        self.assertEqual(1, classifier.call_count)
        self.assertEqual(1, len(dida.tasks))

    def test_named_binding_of_task_b_updates_controls_without_touching_task_a(self):
        service, classifier, media, dida, ledger, _ = self.setup_a()
        task_b = TaskReference("existing-task-b", "已有任务B", "Inbox", "inbox")
        looked_up = []

        def exact_refs(title):
            looked_up.append(title)
            return (task_b,)

        dida.exact_active_task_references = exact_refs
        named = self.message("context-bind-b", "补设提醒：今天下午五点｜已有任务B", media, voice=False, minutes_later=1)
        bound = service.handle(named)
        self.assertEqual(ExecutionStatus.PLANNED, bound.status)
        self.assertEqual(["已有任务B"], looked_up)
        self.assertEqual((task_b,), ledger.recent_task_context(named.conversation_key, named.received_at).candidates)
        self.assertEqual((task_b,), ledger.recent_task_context(named.sender_key, named.received_at).candidates)
        update = self.message("context-update-bound-b", "改成四点", media, voice=False, minutes_later=2)
        service.handle(update)
        self.assertEqual("pending", ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
        self.assertEqual("rescheduled", ledger.reminder_status(task_b.task_id, NOW.replace(hour=17, minute=0)))
        self.assertEqual("pending", ledger.reminder_status(task_b.task_id, NOW.replace(hour=16, minute=0)))
        self.assertIsNone(ledger.reminder_status("voice-task-1", NOW.replace(hour=16, minute=0)))
        self.assertEqual(0, classifier.call_count)
        self.assertEqual(1, len(dida.tasks))

    def test_relative_adjustment_rejects_snapshot_changed_before_enqueue(self):
        service, classifier, media, _, ledger, source = self.setup_a()
        schedule = service.reminders.schedule
        competing_time = NOW.replace(hour=17, minute=0)

        def concurrent_change(draft, task, incoming, **kwargs):
            ledger.enqueue_reminder(source, task, competing_time, replace_existing=True)
            return schedule(draft, task, incoming, **kwargs)

        service.reminders.schedule = concurrent_change
        incoming = self.message("context-relative-race", "半小时后提醒我", media, voice=False, minutes_later=1)
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertEqual("pending", ledger.reminder_status("voice-task-1", competing_time))
        self.assertIsNone(ledger.reminder_status("voice-task-1", incoming.received_at + timedelta(minutes=30)))
        self.assertEqual(0, classifier.call_count)

    def test_completed_plain_task_without_any_reminder_cannot_be_revived(self):
        service, classifier, media, dida, ledger = self.make_service()
        created = self.message("context-plain-create", "待办：提交报告", media, voice=False)
        initial = service.handle(created)
        self.assertEqual(ExecutionStatus.PLANNED, initial.status)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual("", dida.tasks[0].reminder_at)
        self.assertEqual((), ledger.reminder_snapshot("voice-task-1", created))
        completed = []

        def complete_task(task):
            completed.append(task.task_id)
            return ActionResult("complete", ExecutionStatus.PLANNED, "已完成提交报告")

        dida.complete_task = complete_task
        done = self.message("context-plain-complete", "已完成", media, voice=False, minutes_later=1)
        result = service.handle(done)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(["voice-task-1"], completed)
        self.assertEqual((), ledger.recent_task_context(done.sender_key, done.received_at).candidates)
        self.assertEqual((), ledger.recent_task_context(done.conversation_key, done.received_at).candidates)
        before = classifier.call_count
        later = self.message("context-no-revive", "半小时后提醒我", media, voice=False, minutes_later=2)
        rejected = service.handle(later)
        self.assertEqual(ExecutionStatus.SKIPPED, rejected.status)
        self.assertEqual((), ledger.reminder_snapshot("voice-task-1", later))
        self.assertEqual(before, classifier.call_count)
        self.assertEqual(1, len(dida.tasks))

    def test_completion_marks_explicit_conversation_without_reminder_route(self):
        _, _, media, _, ledger = self.make_service()
        source = self.message("context-complete-no-route", "", media, voice=False)
        task = TaskReference("plain-task-a", "无提醒任务A")
        other = TaskReference("plain-task-b", "仍未完成任务B")
        for key in (source.sender_key, source.conversation_key):
            ledger.record_task_context(
                key, (task, other), batch_id=source.message_id,
                source_message_id=source.message_id, observed_at=NOW,
                ttl_seconds=600, context_kind="task-create",
            )
        ledger.set_pending_reminder_action(source, {
            "kind": "update", "task": {"task_id": task.task_id},
        }, NOW + timedelta(minutes=5))
        self.assertEqual((), ledger.reminder_snapshot(task.task_id, source))
        ledger.mark_task_completed(source.sender_key, task.task_id, conversation_key=source.conversation_key)
        self.assertEqual((other,), ledger.recent_task_context(source.sender_key, NOW).candidates)
        self.assertEqual((other,), ledger.recent_task_context(source.conversation_key, NOW).candidates)
        self.assertIsNone(ledger.pending_reminder_action(source))
        self.assertEqual((), ledger.reminder_snapshot(task.task_id, source))


if __name__ == "__main__":
    unittest.main()
