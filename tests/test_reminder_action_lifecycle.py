from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timedelta

import test_reminder_actions as action_fixtures
from wechat_secretary.reminders import ReminderScheduler


NOW = action_fixtures.NOW
SETTINGS = action_fixtures.SETTINGS


class ReminderActionLifecycleTests(unittest.TestCase):
    # Borrow fixture helpers only. Importing/inheriting a TestCase would make
    # discovery execute its tests a second time in this module.
    make_service = action_fixtures.ReminderActionTests.make_service
    message = staticmethod(action_fixtures.ReminderActionTests.message)
    fixture = action_fixtures.ReminderActionTests.fixture
    send = action_fixtures.ReminderActionTests.send

    def send_at(self, text, received_at):
        self.seq += 1
        message = self.message(f"lifecycle-{self.seq}", text, self.media, voice=False)
        self.last = replace(message, received_at=received_at)
        return self.service.handle(self.last)

    def deliver(self, when, *, during_send=None):
        sent = []

        def sender(record, content):
            if during_send is not None:
                during_send()
            sent.append((record.task.task_id, content))
            return f"lifecycle-delivered-{len(sent)}"

        result = ReminderScheduler(SETTINGS, self.ledger).poll_once(sender, when)
        self.assertEqual(1, result.sent)
        self.assertEqual(1, len(sent))
        return result

    def test_same_second_creations_modify_only_the_latest_task(self):
        self.fixture()
        first_message = self.last
        self.send("今天下午四点提醒我买咖啡")
        self.assertEqual(first_message.received_at, self.last.received_at)
        self.send("改成五点")

        self.assertEqual(2, len(self.dida.tasks))
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
        self.assertEqual("rescheduled", self.ledger.reminder_status("voice-task-2", NOW.replace(hour=16, minute=0)))
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-2", NOW.replace(hour=17, minute=0)))
        self.assertIsNone(self.ledger.reminder_status("voice-task-1", NOW.replace(hour=17, minute=0)))

    def test_sent_reminder_refreshes_expired_conversation_context_for_append(self):
        self.fixture()
        delivery_at = max(
            NOW.replace(hour=15, minute=0),
            NOW + timedelta(seconds=SETTINGS.completion_context_ttl_seconds + 1),
        )
        old_context = self.ledger.recent_task_context(self.last.conversation_key, delivery_at)
        self.assertEqual((), old_context.candidates)

        self.deliver(delivery_at)
        refreshed = self.ledger.recent_task_context(self.last.conversation_key, delivery_at)
        self.assertEqual(("voice-task-1",), tuple(ref.task_id for ref in refreshed.candidates))
        result = self.send_at("再提醒三次，每隔20分钟", delivery_at + timedelta(seconds=5))

        self.assertIn("追加", result.reply)
        self.assertEqual(1, len(self.dida.tasks))
        self.assertEqual(3, self.ledger.active_reminder_count("voice-task-1", self.last))
        self.assertEqual(4, len(self.ledger.reminder_snapshot("voice-task-1", self.last)))

    def test_completed_sent_task_cannot_be_revived_by_append(self):
        self.fixture()
        delivery_at = NOW.replace(hour=15, minute=0)
        self.deliver(delivery_at)
        self.ledger.mark_task_completed(self.last.sender_key, "voice-task-1")
        before = self.ledger.reminder_snapshot("voice-task-1", self.last)

        self.send_at("再提醒三次，每隔20分钟", delivery_at + timedelta(minutes=1))

        self.assertEqual(before, self.ledger.reminder_snapshot("voice-task-1", self.last))
        self.assertEqual(0, self.ledger.active_reminder_count("voice-task-1", self.last))
        self.assertEqual((), self.ledger.recent_task_context(self.last.conversation_key, self.last.received_at).candidates)

    def test_completed_latest_task_does_not_fall_back_to_an_older_active_task(self):
        self.fixture()
        self.send_at("今天下午四点提醒我买咖啡", NOW + timedelta(minutes=1))
        self.ledger.mark_task_completed(self.last.sender_key, "voice-task-2")
        first_before = self.ledger.reminder_snapshot("voice-task-1", self.last)
        second_before = self.ledger.reminder_snapshot("voice-task-2", self.last)

        for index, text in enumerate(("再提醒三次，每隔20分钟", "改成五点", "刚才那个不要了"), 2):
            with self.subTest(text=text):
                self.send_at(text, NOW + timedelta(minutes=index))
                self.assertEqual(first_before, self.ledger.reminder_snapshot("voice-task-1", self.last))
                self.assertEqual(second_before, self.ledger.reminder_snapshot("voice-task-2", self.last))

        self.assertEqual(1, self.ledger.active_reminder_count("voice-task-1", self.last))
        self.assertEqual(0, self.ledger.active_reminder_count("voice-task-2", self.last))

    def test_late_delivery_success_does_not_reactivate_a_completed_task(self):
        self.fixture()
        delivery_at = NOW.replace(hour=15, minute=0)
        sender_key = self.last.sender_key
        self.deliver(
            delivery_at,
            during_send=lambda: self.ledger.mark_task_completed(sender_key, "voice-task-1"),
        )
        before = self.ledger.reminder_snapshot("voice-task-1", self.last)

        self.send_at("再提醒三次，每隔20分钟", delivery_at + timedelta(minutes=1))

        self.assertEqual(before, self.ledger.reminder_snapshot("voice-task-1", self.last))
        self.assertEqual(0, self.ledger.active_reminder_count("voice-task-1", self.last))
        self.assertEqual((), self.ledger.recent_task_context(self.last.conversation_key, self.last.received_at).candidates)
        self.assertEqual((), self.ledger.recent_task_context(sender_key, self.last.received_at).candidates)

    def test_older_cancel_cannot_clear_a_newer_pending_update(self):
        self.fixture()
        self.send_at("改成明天下午四点多", NOW + timedelta(minutes=2))
        pending = self.ledger.pending_reminder_action(self.last)
        self.assertIsNotNone(pending)
        snapshot = self.ledger.reminder_snapshot("voice-task-1", self.last)

        for text in ("算了", "刚才那个不要了"):
            with self.subTest(text=text):
                result = self.send_at(text, NOW + timedelta(minutes=1))
                self.assertIn("早于", result.reply)
                self.assertEqual(pending, self.ledger.pending_reminder_action(self.last))
                self.assertEqual(snapshot, self.ledger.reminder_snapshot("voice-task-1", self.last))

        self.send_at("五点", NOW + timedelta(minutes=3))
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", (NOW + timedelta(days=1)).replace(hour=17, minute=0)))

    def test_update_with_recurrence_controls_is_not_downgraded_to_one_off(self):
        self.fixture()
        before = self.ledger.reminder_snapshot("voice-task-1", self.last)
        for text in ("改成每周二上午九点", "改成明天下午四点，共三次"):
            with self.subTest(text=text):
                result = self.send(text)
                self.assertIn("重复", result.reply)
                self.assertEqual(before, self.ledger.reminder_snapshot("voice-task-1", self.last))
                self.assertEqual(1, len(self.dida.tasks))

    def test_pending_update_recurrence_reply_keeps_original_reminder(self):
        self.fixture()
        before = self.ledger.reminder_snapshot("voice-task-1", self.last)
        self.send("改成明天下午四点多")
        for text in ("每周二上午九点", "共三次"):
            with self.subTest(text=text):
                result = self.send(text)
                self.assertIn("重复", result.reply)
                self.assertEqual(before, self.ledger.reminder_snapshot("voice-task-1", self.last))
                self.assertEqual(1, len(self.dida.tasks))


if __name__ == "__main__":
    unittest.main()
