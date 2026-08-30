from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from wechat_secretary.models import ClarificationReason, IntentKind, IntentPlan
from wechat_secretary.semantic_guard import (
    looks_like_pending_body,
    looks_like_pending_followup,
    resume_pending_task,
    validate_plan_semantics,
)


NOW = datetime(2026, 8, 24, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
EMPTY = IntentPlan(IntentKind.CLARIFY, confidence=0.1)


def start(text):
    return validate_plan_semantics(text, EMPTY, NOW, expected_kind=IntentKind.TASK)


class PendingReminderFieldTests(unittest.TestCase):
    def test_missing_body_keeps_schedule_and_body_reply_completes_it(self):
        first = start("今天下午三点的时候提醒我。")
        self.assertIs(first.reason, ClarificationReason.MISSING_TASK_BODY)
        self.assertEqual("", first.pending.task.title)
        self.assertEqual("2026-08-24", first.pending.reminder_date)
        self.assertEqual("15:00", first.pending.reminder_time)
        completed = resume_pending_task(first.pending, "买牛奶。", NOW)
        self.assertTrue(completed.ready)
        self.assertEqual("买牛奶", completed.plan.tasks[0].title)
        self.assertEqual("2026-08-24T15:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_missing_body_preserves_relative_deadline_not_reanchored_reply_time(self):
        first = start("二十分钟后提醒我")
        self.assertEqual("09:20", first.pending.reminder_time)
        completed = resume_pending_task(first.pending, "喝水", NOW.replace(minute=5))
        self.assertTrue(completed.ready)
        self.assertEqual("2026-08-24T09:20+08:00", completed.plan.tasks[0].reminder_at)

    def test_relative_schedule_preserves_seconds_on_create_and_body_followup(self):
        now = NOW.replace(second=45)
        created = validate_plan_semantics("二十分钟后提醒我喝水", EMPTY, now, expected_kind=IntentKind.TASK)
        self.assertEqual("2026-08-24T09:20:45+08:00", created.plan.tasks[0].reminder_at)
        first = validate_plan_semantics("二十分钟后提醒我", EMPTY, now, expected_kind=IntentKind.TASK)
        completed = resume_pending_task(first.pending, "喝水", now.replace(minute=20, second=15))
        self.assertTrue(completed.ready)
        self.assertEqual("2026-08-24T09:20:45+08:00", completed.plan.tasks[0].reminder_at)

    def test_missing_body_and_clock_can_be_filled_in_either_order(self):
        first = start("明天下午提醒我")
        for replies in (("买牛奶", "三点"), ("三点", "买牛奶")):
            with self.subTest(replies=replies):
                second = resume_pending_task(first.pending, replies[0], NOW)
                self.assertFalse(second.ready)
                completed = resume_pending_task(second.pending, replies[1], NOW)
                self.assertTrue(completed.ready)
                self.assertEqual("买牛奶", completed.plan.tasks[0].title)
                self.assertEqual("2026-08-25T15:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_bare_hour_inherits_saved_afternoon(self):
        first = start("明天下午提醒我买牛奶")
        self.assertEqual("下午", first.pending.reminder_period)
        completed = resume_pending_task(first.pending, "三点", NOW)
        self.assertTrue(completed.ready)
        self.assertEqual("2026-08-25T15:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_explicit_period_replaces_saved_afternoon(self):
        first = start("明天下午提醒我买牛奶")
        second = resume_pending_task(first.pending, "上午", NOW)
        self.assertEqual("上午", second.pending.reminder_period)
        completed = resume_pending_task(second.pending, "十点", NOW)
        self.assertEqual("2026-08-25T10:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_negative_date_correction_replaces_rejected_date(self):
        first = start("今天下午提醒我买牛奶")
        self.assertTrue(looks_like_pending_followup("不是今天，是明天。"))
        second = resume_pending_task(first.pending, "不是今天，是明天。", NOW)
        self.assertFalse(second.ready)
        self.assertEqual("2026-08-25", second.pending.reminder_date)
        self.assertEqual("下午", second.pending.reminder_period)
        completed = resume_pending_task(second.pending, "四点", NOW)
        self.assertEqual("2026-08-25T16:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_invalid_or_fuzzy_clock_preserves_other_fields(self):
        first = start("明天下午提醒我买牛奶")
        for text in ("25:00", "三点左右", "四点多"):
            with self.subTest(text=text):
                invalid = resume_pending_task(first.pending, text, NOW)
                self.assertFalse(invalid.ready)
                self.assertEqual("2026-08-25", invalid.pending.reminder_date)
                self.assertEqual("下午", invalid.pending.reminder_period)
                self.assertEqual("", invalid.pending.reminder_time)
                completed = resume_pending_task(invalid.pending, "四点", NOW)
                self.assertEqual("2026-08-25T16:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_date_conflict_keeps_unambiguous_clock(self):
        first = start("今天或明天下午四点提醒我买牛奶")
        self.assertFalse(first.ready)
        self.assertEqual("", first.pending.reminder_date)
        self.assertEqual("16:00", first.pending.reminder_time)
        completed = resume_pending_task(first.pending, "明天", NOW)
        self.assertEqual("2026-08-25T16:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_clock_conflict_keeps_unambiguous_date(self):
        first = start("明天下午三点或四点提醒我买牛奶")
        self.assertFalse(first.ready)
        self.assertEqual("2026-08-25", first.pending.reminder_date)
        self.assertEqual("", first.pending.reminder_time)
        completed = resume_pending_task(first.pending, "四点", NOW)
        self.assertEqual("2026-08-25T16:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_conflicting_period_is_not_forgotten_by_a_bare_hour_followup(self):
        first = start("明天下午或者晚上四点提醒我买牛奶")
        self.assertFalse(first.ready)
        second = resume_pending_task(first.pending, "三点", NOW)
        self.assertFalse(second.ready)
        self.assertEqual("", second.pending.reminder_time)
        self.assertEqual("ambiguous", second.pending.reminder_period)
        completed = resume_pending_task(second.pending, "下午三点", NOW)
        self.assertTrue(completed.ready)
        self.assertEqual("2026-08-25T15:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_ambiguous_midnight_never_becomes_noon_or_wrong_day(self):
        first = start("明天晚上十二点提醒我喝水")
        self.assertFalse(first.ready)
        self.assertEqual("2026-08-25", first.pending.reminder_date)
        self.assertEqual("", first.pending.reminder_time)
        completed = resume_pending_task(first.pending, "凌晨十二点", NOW)
        self.assertEqual("2026-08-25T00:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_past_clock_is_preserved_for_explicit_new_date(self):
        first = start("今天上午八点提醒我买牛奶")
        self.assertFalse(first.ready)
        self.assertEqual("08:00", first.pending.reminder_time)
        completed = resume_pending_task(first.pending, "不是今天，是明天", NOW)
        self.assertEqual("2026-08-25T08:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_explicit_24_hour_clock_replaces_old_period_even_when_past(self):
        first = start("今天下午提醒我买牛奶")
        second = resume_pending_task(first.pending, "09:00", NOW)
        self.assertFalse(second.ready)
        self.assertEqual("上午", second.pending.reminder_period)
        completed = resume_pending_task(second.pending, "明天十点", NOW)
        self.assertTrue(completed.ready)
        self.assertEqual("2026-08-25T10:00+08:00", completed.plan.tasks[0].reminder_at)

    def test_repeat_count_is_not_task_body_and_is_not_silently_one_off(self):
        first = start("再提醒我三次")
        self.assertFalse(first.ready)
        self.assertEqual("", first.pending.task.title)
        second = resume_pending_task(first.pending, "今天下午四点", NOW)
        self.assertFalse(second.ready)
        self.assertEqual("", second.pending.task.title)
        self.assertFalse(start("今天下午四点提醒我买牛奶，共三次").ready)

    def test_count_followup_requires_frequency_not_new_title(self):
        first = start("今天提醒我买牛奶")
        self.assertTrue(looks_like_pending_followup("还要提醒三次"))
        second = resume_pending_task(first.pending, "还要提醒三次", NOW)
        self.assertFalse(second.ready)
        self.assertEqual("买牛奶", second.pending.task.title)
        self.assertEqual(3, second.pending.task.reminder_recurrence.count)
        self.assertIn("频率", second.question)

    def test_complex_weekdays_rejected_without_truncation_and_recoverable(self):
        for schedule in ("每周二和四", "每周二、周四", "周一到五", "每个工作日"):
            with self.subTest(schedule=schedule):
                first = start(f"{schedule}上午九点提醒我买牛奶，共三次")
                self.assertFalse(first.ready)
                self.assertIs(first.reason, ClarificationReason.UNSUPPORTED_RECURRENCE)
                self.assertEqual("买牛奶", first.pending.task.title)
                self.assertEqual(0, first.pending.task.reminder_recurrence.weekday)
                self.assertEqual("09:00", first.pending.reminder_time)
                completed = resume_pending_task(first.pending, "每周二", NOW)
                self.assertTrue(completed.ready)
                self.assertEqual(2, completed.plan.tasks[0].reminder_recurrence.weekday)
                self.assertEqual(3, completed.plan.tasks[0].reminder_recurrence.count)

    def test_body_only_recurring_reply_keeps_all_recurrence_fields(self):
        first = start("每周二上午九点提醒我，共三次")
        self.assertIs(first.reason, ClarificationReason.MISSING_TASK_BODY)
        completed = resume_pending_task(first.pending, "买牛奶", NOW)
        self.assertTrue(completed.ready)
        self.assertEqual("2026-08-25T09:00+08:00", completed.plan.tasks[0].reminder_at)
        self.assertEqual(3, completed.plan.tasks[0].reminder_recurrence.count)

    def test_non_field_or_unrelated_question_does_not_replace_draft(self):
        first = start("今天提醒我买牛奶")
        for text in ("你下午四点有空吗", "不是买奶，是开会", "这是另外一个问题"):
            with self.subTest(text=text):
                self.assertFalse(looks_like_pending_followup(text))
                reply = resume_pending_task(first.pending, text, NOW)
                self.assertFalse(reply.ready)
                self.assertEqual(first.pending, reply.pending)

    def test_body_filter_excludes_questions_new_requests_cancellation_and_meta(self):
        for text in ("买牛奶", "查一下本地技能有没有更好的", "事项是：给老师回电话"):
            with self.subTest(text=text):
                self.assertTrue(looks_like_pending_body(text))
        for text in ("你下午四点有空吗", "取消", "刚才那个不要了", "共三次", "明天下午三点", "下午买牛奶", "帮我记一下这个结论", "今天提醒我买奶", "好的", "这也太蠢了"):
            with self.subTest(text=text):
                self.assertFalse(looks_like_pending_body(text))


if __name__ == "__main__":
    unittest.main()
