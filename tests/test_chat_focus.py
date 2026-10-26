"""Deterministic chat/reminder interleavings without live threads or sockets."""
from __future__ import annotations

from collections import deque
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_chat_conversation as fixtures
from wechat_secretary.hermes_plugin import GatewayBridge
from wechat_secretary.models import ExecutionStatus, HandlingResult, TaskReference
from wechat_secretary.reminders import ReminderScheduler


NOW = fixtures.NOW


class ChatFocusInterleavingTests(fixtures.ChatServiceFixture):
    @staticmethod
    def record_notification(ledger, key, reference, observed_at, *, batch="notification"):
        ledger.record_task_context(
            key, (reference,), batch_id=batch, source_message_id=f"delivered-{batch}",
            observed_at=observed_at, ttl_seconds=3600,
            context_kind="reminder-delivery", reminder_at=observed_at,
        )

    def test_reminder_recorded_during_generation_is_inactive_after_chat_delivery(self):
        service, chat, store, _, _ = self.make_service()
        incoming = self.message(service, "聊天：讨论一下阅读习惯")
        reminder = TaskReference("synthetic-during-chat", "交还借阅书籍", "Inbox", "synthetic-project")
        due_at = NOW + timedelta(seconds=5)
        service.ledger.enqueue_reminder(incoming, reminder, due_at)
        scheduler = ReminderScheduler(service.settings, service.ledger)
        sender = Mock(return_value="synthetic-reminder-receipt")
        original_respond = chat.respond

        def respond_during_reminder(content, history, *, image_inputs=()):
            stats = scheduler.poll_once(sender, NOW + timedelta(seconds=10))
            self.assertEqual(1, stats.sent)
            self.assertEqual((reminder,), service.ledger.recent_task_context(
                incoming.conversation_key, NOW + timedelta(seconds=11)
            ).candidates)
            return original_respond(content, history, image_inputs=image_inputs)

        chat.respond = respond_during_reminder
        with patch("wechat_secretary.service.datetime") as wall_clock:
            wall_clock.now.return_value = NOW
            result = service.handle(incoming)
            wall_clock.now.return_value = NOW + timedelta(seconds=20)
            service.record_reply_delivery(incoming, result, confirmed=True)

        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual((), service.ledger.recent_task_context(
            incoming.conversation_key, NOW + timedelta(seconds=30)
        ).candidates)
        self.assertEqual((reminder,), service.ledger.find_task_context(
            incoming.conversation_key, reminder.title, NOW + timedelta(seconds=30)
        ))
        self.assertEqual("sent", service.ledger.reminder_status(reminder.task_id, due_at))
        sender.assert_called_once()
        self.assertEqual(2, len(self.history(store, incoming)))
        self.assertNotIn(reminder.title, repr(self.history(store, incoming)))

    def check_late_notification(self, delta):
        service, _, _, _, _ = self.make_service()
        incoming = self.message(service, "synthetic")
        barrier = NOW + timedelta(seconds=20)
        reference = TaskReference("late-notification", "晚到的提醒上下文")
        service.ledger.reset_conversation_focus(incoming.conversation_key, barrier)
        self.record_notification(service.ledger, incoming.conversation_key, reference, barrier + delta)
        after = barrier + timedelta(seconds=30)
        self.assertEqual((), service.ledger.recent_task_context(incoming.conversation_key, after).candidates)
        self.assertEqual((reference,), service.ledger.find_task_context(incoming.conversation_key, reference.title, after))

    def test_late_notification_observed_before_reset_cannot_restore_anonymous_target(self):
        self.check_late_notification(timedelta(seconds=-1))

    def test_late_notification_observed_exactly_at_reset_cannot_restore_anonymous_target(self):
        self.check_late_notification(timedelta(0))

    def test_new_notification_after_reset_can_establish_new_anonymous_target(self):
        service, _, _, _, _ = self.make_service()
        incoming = self.message(service, "synthetic")
        barrier = NOW + timedelta(seconds=20)
        reference = TaskReference("new-notification", "新送达的提醒")
        service.ledger.reset_conversation_focus(incoming.conversation_key, barrier)
        self.record_notification(service.ledger, incoming.conversation_key, reference, barrier + timedelta(seconds=1))
        self.assertEqual((reference,), service.ledger.recent_task_context(
            incoming.conversation_key, barrier + timedelta(seconds=2)
        ).candidates)

    def test_older_reset_cannot_move_notification_barrier_backwards(self):
        service, _, _, _, _ = self.make_service()
        incoming = self.message(service, "synthetic")
        reference = TaskReference("between-resets", "不应恢复的提醒")
        service.ledger.reset_conversation_focus(incoming.conversation_key, NOW + timedelta(seconds=20))
        service.ledger.reset_conversation_focus(incoming.conversation_key, NOW + timedelta(seconds=5))
        self.record_notification(service.ledger, incoming.conversation_key, reference, NOW + timedelta(seconds=10))
        self.assertEqual((), service.ledger.recent_task_context(
            incoming.conversation_key, NOW + timedelta(seconds=30)
        ).candidates)

    def check_queued_task_order(self, queued_at, *, expect_anonymous):
        service, _, _, dida, _ = self.make_service()
        source = self.message(service, "synthetic notification source")
        notification = TaskReference("notification-task", "旧事项通知", "Inbox", "synthetic-project")
        service.ledger.enqueue_reminder(source, notification, NOW)
        scheduler = ReminderScheduler(service.settings, service.ledger)
        poll_at = NOW + timedelta(seconds=10)

        def send_with_interleaved_chat_and_queued_task(record, content):
            # The reminder sender is suspended here while the conversation
            # worker finishes a chat and processes a previously queued request.
            with patch("wechat_secretary.service.datetime") as wall_clock:
                wall_clock.now.return_value = NOW + timedelta(seconds=20)
                self.exchange(service, "聊天：聊聊学习方法", received_at=NOW + timedelta(seconds=2))
            self.exchange(service, "待办：提交新报告", received_at=queued_at)
            return "synthetic-late-delivery-receipt"

        stats = scheduler.poll_once(send_with_interleaved_chat_and_queued_task, poll_at)
        self.assertEqual(1, stats.sent)
        self.assertEqual(1, len(dida.created))
        at = NOW + timedelta(seconds=30)
        recent = service.ledger.recent_task_context(source.conversation_key, at).candidates
        if expect_anonymous:
            self.assertEqual((dida.references[0],), recent)
        else:
            self.assertEqual((), recent)
        self.assertEqual((dida.references[0],), service.ledger.find_task_context(
            source.conversation_key, "提交新报告", at
        ))
        row = service.ledger._connection.execute(
            "SELECT observed_at, context_kind FROM task_context WHERE sender_hash = ? AND task_id = ?",
            (service.ledger._hash(source.conversation_key), notification.task_id),
        ).fetchone()
        self.assertEqual(poll_at.isoformat(), row["observed_at"])
        self.assertEqual("inactive", row["context_kind"])
        _, result = self.exchange(service, "搞定了", received_at=at)
        if expect_anonymous:
            self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
            self.assertEqual(["提交新报告"], [ref.title for ref in dida.completed])
        else:
            self.assertEqual(ExecutionStatus.SKIPPED, result.status)
            self.assertEqual([], dida.completed)
            self.exchange(service, "完成：提交新报告", received_at=at + timedelta(seconds=1))
            self.assertEqual(["提交新报告"], [ref.title for ref in dida.completed])

    def test_late_inactive_notification_conservatively_blocks_older_queued_anonymous_target(self):
        # Poll time is newer than the queued request's receipt time. Retain the
        # chronology guard even though the request was executed more recently.
        self.check_queued_task_order(NOW + timedelta(seconds=5), expect_anonymous=False)

    def test_new_request_observed_after_poll_keeps_its_fresh_anonymous_target(self):
        self.check_queued_task_order(NOW + timedelta(seconds=15), expect_anonymous=True)


class ChatGatewayBookkeepingTests(fixtures.ChatServiceFixture):
    def test_delivery_bookkeeping_exception_never_sends_second_reply_or_blocks_next_message(self):
        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed):
                service, _, _, _, _ = self.make_service()
                messages = [self.message(service, f"聊天：合成问题 {index}") for index in range(2)]
                results = [HandlingResult(ExecutionStatus.SUCCEEDED, f"合成答复 {index}") for index in range(2)]
                service.handle = Mock(side_effect=results)
                service.record_reply_delivery = Mock(side_effect=RuntimeError("synthetic bookkeeping fault"))
                bridge = GatewayBridge(service, Mock())
                key = messages[0].conversation_key
                pending = deque()
                for incoming in messages:
                    self.assertTrue(bridge._slots.acquire(blocking=False))
                    event = SimpleNamespace(
                        source=SimpleNamespace(platform="weixin", chat_id=incoming.chat_id),
                        message_id=incoming.message_id,
                    )
                    pending.append((object(), object(), event, incoming, ""))
                bridge._queues[key] = pending
                with patch("wechat_secretary.hermes_plugin._schedule_reply", return_value=confirmed) as send:
                    bridge._drain_conversation(key)

                self.assertEqual([result.reply for result in results], [call.args[3] for call in send.call_args_list])
                self.assertEqual(2, service.handle.call_count)
                self.assertEqual(2, service.record_reply_delivery.call_count)
                self.assertEqual([confirmed, confirmed], [call.kwargs["confirmed"] for call in service.record_reply_delivery.call_args_list])
                self.assertNotIn(key, bridge._queues)
                # Every consumed work slot is returned even after a failure.
                for _ in range(service.settings.worker_limit):
                    self.assertTrue(bridge._slots.acquire(blocking=False))
                self.assertFalse(bridge._slots.acquire(blocking=False))
