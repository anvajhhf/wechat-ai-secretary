"""Bounded post-create corrections, with in-memory storage and fake executors."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import test_pending_corrections_round3 as pending_fixtures
import test_voice_reminder_conversation as fixtures
from wechat_secretary.models import ExecutionStatus, TaskReference
from wechat_secretary.reminder_actions import parse_reminder_action


NOW = pending_fixtures.NOW
BODY = "分选试剂和询价"


class PostCreateCorrectionsTests(unittest.TestCase):
    # Reuse helpers, not the TestCase itself, to avoid duplicate discovery.
    make_service = fixtures.VoiceReminderConversationTests.make_service
    setUp = pending_fixtures.PendingCorrectionsRound3Tests.setUp

    def prepare_service(self):
        (
            self.service, self.classifier, self.media, self.dida, self.ledger,
        ) = self.make_service()
        self.seq = 0

    def send(self, text, *, voice=False, seconds=None, chat_id=None):
        self.seq += 1
        self.last = pending_fixtures.PendingCorrectionsRound3Tests.incoming(
            f"post-create-{self.seq}", text, self.media,
            seconds=self.seq if seconds is None else seconds,
            chat_id=chat_id, voice=voice,
        )
        return self.service.handle(self.last)

    def start(self, text=f"明天下午三点提醒我{BODY}"):
        self.prepare_service()
        result = self.send(text, seconds=0)
        self.source = self.last
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(1, len(self.dida.tasks))
        self.assertEqual(0, self.classifier.call_count)
        self.assertIsNone(self.ledger.peek_pending_task(
            self.source.conversation_key, self.source.received_at,
        ))
        self.original_at = datetime.fromisoformat(self.dida.tasks[0].reminder_at)
        self.assertEqual("pending", self.ledger.reminder_status(
            "voice-task-1", self.original_at,
        ))
        return self.ledger.reminder_snapshot("voice-task-1", self.source)

    def assert_rescheduled(self, result, expected_at):
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertFalse(result.llm_called)
        self.assertEqual(0, self.classifier.call_count)
        self.assertEqual(1, len(self.dida.tasks), "A correction must not recreate the task")
        self.assertEqual(BODY, self.dida.tasks[0].title)
        self.assertEqual("rescheduled", self.ledger.reminder_status(
            "voice-task-1", self.original_at,
        ))
        self.assertEqual("pending", self.ledger.reminder_status(
            "voice-task-1", datetime.fromisoformat(expected_at),
        ))
        rows = self.ledger.reminder_snapshot("voice-task-1", self.source)
        self.assertEqual(2, len(rows), "One historical row and one replacement only")
        self.assertEqual(1, self.ledger.active_reminder_count("voice-task-1", self.source))
        self.assertIsNone(self.ledger.pending_reminder_action(self.last))

    def assert_original_unchanged(self, before, *, task_count=1):
        self.assertEqual(before, self.ledger.reminder_snapshot("voice-task-1", self.source))
        self.assertEqual(task_count, len(self.dida.tasks))

    def test_unpunctuated_and_punctuated_clock_corrections_inherit_afternoon(self):
        for voice in (False, True):
            for text in (
                "不是三点是两点",
                "不是三点，是两点。",
                "不是三点,是两点",
                "不是三点，而是两点",
            ):
                with self.subTest(voice=voice, text=text):
                    self.start()
                    result = self.send(text, voice=voice)
                    self.assert_rescheduled(result, "2026-09-01T14:00+08:00")

    def test_date_correction_keeps_existing_hour_and_minute(self):
        for voice in (False, True):
            for text in ("不是明天是后天", "不是明天，是后天。"):
                with self.subTest(voice=voice, text=text):
                    self.start(f"明天下午三点半提醒我{BODY}")
                    result = self.send(text, voice=voice)
                    self.assert_rescheduled(result, "2026-09-02T15:30+08:00")

    def test_explicit_replacement_period_overrides_inherited_period(self):
        self.start()
        result = self.send("不是下午三点是上午两点")
        self.assert_rescheduled(result, "2026-09-01T02:00+08:00")

    def test_clock_correction_can_inherit_morning_without_flipping_to_pm(self):
        self.start(f"明天上午三点提醒我{BODY}")
        result = self.send("不是三点是两点")
        self.assert_rescheduled(result, "2026-09-01T02:00+08:00")

    def test_correction_replay_does_not_recreate_task_or_schedule(self):
        self.start()
        result = self.send("不是三点是两点", voice=True)
        self.assert_rescheduled(result, "2026-09-01T14:00+08:00")
        before = self.ledger.reminder_snapshot("voice-task-1", self.source)
        media_calls = tuple(self.media.calls)
        self.service.handle(self.last)
        self.assert_original_unchanged(before)
        self.assertEqual(media_calls, tuple(self.media.calls))
        self.assertEqual(0, self.classifier.call_count)

    def test_embedded_dates_quotes_and_questions_are_not_update_commands(self):
        for text in (
            "他说不是明天是后天",
            "讨论的日期不是明天是后天",
            "不是三点是两点开会",
            "不是明天是后天询价",
            "不是明天，是后天询价",
            "“不是三点是两点”",
            '"不是明天是后天"',
            "不是三点是两点？",
            "不是三点是两点吗",
            "是不是明天是后天",
            "不是明天，是后天还是大后天？",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_reminder_action(text))
                before = self.start()
                self.send(text)
                self.assert_original_unchanged(before)
                self.assertIsNone(self.ledger.pending_reminder_action(self.last))

    def test_repeat_fields_are_not_treated_as_a_single_reminder_correction(self):
        for text in (
            "不是三点是每天下午两点",
            "不是三点是明天下午两点共三次",
            "不是明天是每周二上午九点",
        ):
            with self.subTest(text=text):
                before = self.start()
                self.send(text)
                self.assert_original_unchanged(before)

    def test_full_new_request_with_correction_words_does_not_edit_old_reminder(self):
        before = self.start()
        result = self.send("明天下午四点提醒我检查不是明天是后天的文字")
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assert_original_unchanged(before, task_count=2)
        self.assertEqual("检查不是明天是后天的文字", self.dida.tasks[1].title)
        self.assertEqual("2026-09-01T16:00+08:00", self.dida.tasks[1].reminder_at)
        self.assertEqual(0, self.classifier.call_count)

    def test_cancellation_words_cannot_be_smuggled_through_correction(self):
        for text in (
            "不是三点是取消",
            "不是三点是不用了",
            "不是取消是两点",
            "不是三点是刚才那个不要了",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_reminder_action(text))
                before = self.start()
                self.send(text)
                self.assert_original_unchanged(before)
                self.assertIsNone(self.ledger.pending_reminder_action(self.last))

    def test_missing_context_is_refused_without_model_or_task_creation(self):
        for text in ("不是三点是两点", "不是明天是后天"):
            with self.subTest(text=text):
                self.prepare_service()
                result = self.send(text)
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                self.assertFalse(result.llm_called)
                self.assertIn("唯一", result.reply)
                self.assertEqual(0, self.classifier.call_count)
                self.assertEqual([], self.dida.tasks)
                self.assertIsNone(self.ledger.pending_reminder_action(self.last))

    def test_multiple_task_candidates_require_disambiguation(self):
        before = self.start()
        task_a = TaskReference("voice-task-1", BODY, "Inbox", "inbox")
        task_b = TaskReference("different-existing-task", "另一个事项", "Inbox", "inbox")
        self.ledger.enqueue_reminder(self.source, task_b, self.original_at)
        before_b = self.ledger.reminder_snapshot(task_b.task_id, self.source)
        self.ledger.record_task_context(
            self.source.conversation_key, (task_a, task_b),
            batch_id="synthetic-multiple-candidates",
            source_message_id="synthetic-query",
            observed_at=NOW + timedelta(seconds=1),
            ttl_seconds=fixtures.SETTINGS.completion_context_ttl_seconds,
            context_kind="task-query",
        )
        result = self.send("不是三点是两点", seconds=2)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertIn("唯一", result.reply)
        self.assert_original_unchanged(before)
        self.assertEqual(before_b, self.ledger.reminder_snapshot(task_b.task_id, self.source))
        self.assertEqual(0, self.classifier.call_count)

    def test_multiple_active_occurrences_are_not_silently_collapsed(self):
        self.start()
        self.send("再提醒三次，每隔20分钟")
        before = self.ledger.reminder_snapshot("voice-task-1", self.source)
        self.assertEqual(4, self.ledger.active_reminder_count("voice-task-1", self.source))
        result = self.send("不是三点是两点")
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertIn("单次", result.reply)
        self.assert_original_unchanged(before)
        self.assertEqual(0, self.classifier.call_count)

    def test_other_conversation_cannot_borrow_same_sender_context(self):
        before = self.start()
        result = self.send("不是三点是两点", chat_id="other-post-create-chat")
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assert_original_unchanged(before)
        self.assertEqual(0, self.ledger.active_reminder_count("voice-task-1", self.last))
        self.assertEqual(0, self.classifier.call_count)

    def test_expired_context_cannot_modify_old_reminder(self):
        before = self.start()
        result = self.send(
            "不是三点是两点",
            seconds=fixtures.SETTINGS.completion_context_ttl_seconds + 1,
        )
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assert_original_unchanged(before)
        self.assertEqual(0, self.classifier.call_count)

    def test_delayed_message_cannot_modify_a_newer_context(self):
        before = self.start()
        result = self.send("不是三点是两点", seconds=-1)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assert_original_unchanged(before)
        self.assertEqual(0, self.classifier.call_count)

    def test_past_clock_keeps_original_and_cancelling_clarification_is_safe(self):
        before = self.start(f"今天晚上八点提醒我{BODY}")
        result = self.send("不是八点是七点")
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertIn("已经过去", result.reply)
        self.assert_original_unchanged(before)
        self.assertIsNotNone(self.ledger.pending_reminder_action(self.last))
        cancelled = self.send("取消")
        self.assertIn("原提醒保持不变", cancelled.reply)
        self.assert_original_unchanged(before)
        self.assertIsNone(self.ledger.pending_reminder_action(self.last))
        self.assertEqual(0, self.classifier.call_count)

    def test_cancelled_reminder_cannot_be_revived_by_clock_correction(self):
        self.start()
        self.send("取消刚才那个提醒")
        before = self.ledger.reminder_snapshot("voice-task-1", self.source)
        result = self.send("不是三点是两点")
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assert_original_unchanged(before)
        self.assertEqual(0, self.ledger.active_reminder_count("voice-task-1", self.source))
        self.assertEqual(0, self.classifier.call_count)


if __name__ == "__main__":
    unittest.main()
