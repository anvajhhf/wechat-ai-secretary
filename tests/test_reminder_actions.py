from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timedelta

import test_voice_reminder_conversation as fixtures
from test_voice_reminder_conversation import NOW, SETTINGS
from wechat_secretary.models import ExecutionStatus
from wechat_secretary.reminders import ReminderScheduler


class ReminderActionTests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service
    message = staticmethod(fixtures.VoiceReminderConversationTests.message)

    def fixture(self):
        service, classifier, media, dida, ledger = self.make_service()
        self.service, self.media, self.dida, self.ledger = service, media, dida, ledger
        self.seq = 0
        return self.send("今天下午三点提醒我买牛奶")

    def send(self, text, **kwargs):
        self.seq += 1
        message = self.message(f"action-{self.seq}", text, self.media, voice=False, **kwargs)
        self.last = message
        return self.service.handle(message)

    def test_update_inherits_date_and_period_without_creating_task(self):
        self.fixture()
        self.send("改成四点")
        self.assertEqual(1, len(self.dida.tasks))
        self.assertEqual("rescheduled", self.ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", NOW.replace(hour=16, minute=0)))

    def test_cancel_does_not_complete_task(self):
        self.fixture()
        result = self.send("刚才那个不要了")
        self.assertIn("未完成", result.reply)
        self.assertEqual(0, self.ledger.active_reminder_count("voice-task-1", self.last))
        self.assertEqual(1, len(self.ledger.recent_task_context(self.last.conversation_key, NOW).candidates))

    def test_date_only_correction_preserves_clock(self):
        self.fixture()
        self.send("不是今天，是明天")
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", (NOW + timedelta(days=1)).replace(hour=15, minute=0)))

    def test_vague_update_keeps_new_date_during_clarification(self):
        self.fixture()
        self.send("改成明天下午四点多")
        self.send("三点")
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", (NOW + timedelta(days=1)).replace(hour=15, minute=0)))

    def test_abandoned_append_is_not_resumed(self):
        self.fixture()
        self.send("再提醒三次")
        self.send("算了")
        self.send("每隔20分钟")
        self.assertEqual(1, self.ledger.active_reminder_count("voice-task-1", self.last))

    def test_explicit_update_reply_keeps_partial_date(self):
        self.fixture()
        self.send("改成明天下午四点多")
        self.send("改成五点")
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", (NOW + timedelta(days=1)).replace(hour=17, minute=0)))

    def test_period_alone_cannot_repair_invalid_clock(self):
        self.fixture()
        self.send("改成明天25:00")
        self.send("下午")
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
        self.assertEqual(1, len(self.ledger.reminder_snapshot("voice-task-1", self.last)))

    def test_conflicting_period_requires_explicit_disambiguation(self):
        self.fixture()
        self.send("改成明天下午或者晚上")
        self.send("五点")
        self.assertEqual(1, len(self.ledger.reminder_snapshot("voice-task-1", self.last)))
        self.send("下午")
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", (NOW + timedelta(days=1)).replace(hour=17, minute=0)))

    def test_invalid_approximate_count_never_becomes_ten(self):
        self.fixture()
        self.send("再十一二次，每天")
        self.send("再三次，每隔十一二分钟")
        self.assertEqual(1, self.ledger.active_reminder_count("voice-task-1", self.last))

    def test_relative_update_keeps_seconds(self):
        self.fixture()
        incoming = replace(self.message("seconds", "改成二十分钟后", self.media, voice=False), received_at=NOW.replace(second=45))
        self.service.handle(incoming)
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", incoming.received_at + timedelta(minutes=20)))

    def test_append_asks_interval_and_continues_original_task(self):
        self.fixture()
        first = self.send("再提醒我三次")
        self.assertIn("每隔多久", first.reply)
        self.send("每隔20分钟")
        self.assertEqual(1, len(self.dida.tasks))
        self.assertEqual(4, self.ledger.active_reminder_count("voice-task-1", self.last))
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", NOW.replace(hour=16, minute=0)))
        # Replayed WeChat messages must not append again.
        self.service.handle(self.last)
        self.assertEqual(4, self.ledger.active_reminder_count("voice-task-1", self.last))

    def test_cancel_series_clarification(self):
        self.fixture()
        self.send("再提醒三次，每天这个时间")
        result = self.send("取消刚才那个提醒")
        self.assertIn("整个系列", result.reply)
        self.send("本次")
        self.assertEqual(3, self.ledger.active_reminder_count("voice-task-1", self.last))

    def test_expired_and_other_chat_do_not_modify(self):
        self.fixture()
        other = replace(self.message("other", "改成四点", self.media, voice=False), chat_id="other-chat")
        self.service.handle(other)
        self.send("改成四点", minutes_later=SETTINGS.completion_context_ttl_seconds // 60 + 1)
        self.assertEqual("pending", self.ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))

    def test_missing_body_can_resume(self):
        service, _, media, dida, ledger = self.make_service()
        service.handle(self.message("empty-body", "今天下午三点提醒我", media, voice=False))
        service.handle(self.message("body", "买牛奶", media, voice=False))
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual("买牛奶", dida.tasks[0].title)
        self.assertIn("T15:00", dida.tasks[0].reminder_at)

    def test_claimed_but_cancelled_is_never_sent(self):
        self.fixture()
        due = self.ledger.claim_due_reminders(NOW.replace(hour=15, minute=0))
        self.send("取消刚才那个提醒")
        sent = []
        outcome = ReminderScheduler(SETTINGS, self.ledger)._deliver_group(lambda *args: sent.append(args), due, "old", NOW, "test")
        self.assertEqual("skipped", outcome)
        self.assertEqual([], sent)

    def test_sending_cannot_claim_recall_or_reschedule(self):
        self.fixture()
        due = self.ledger.claim_due_reminders(NOW.replace(hour=15, minute=0))
        self.assertTrue(self.ledger.begin_reminder_delivery(record.row_id for record in due))
        result = self.send("刚才那个不要了")
        self.assertEqual(ExecutionStatus.UNCERTAIN, result.status)
        self.assertIn("无法保证撤回", result.reply)
        self.send("改成四点")
        self.assertIsNone(self.ledger.reminder_status("voice-task-1", NOW.replace(hour=16, minute=0)))


if __name__ == "__main__":
    unittest.main()
