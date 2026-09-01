from __future__ import annotations

import unittest
from contextlib import closing
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.models import (
    ClarificationReason, IntentKind, IntentPlan, ReminderRecurrence, TaskDraft,
)
from wechat_secretary.semantic_guard import (
    extract_task_semantics, looks_like_pending_correction,
    resume_pending_task, validate_plan_semantics,
)


NOW = datetime(2026, 8, 30, 19, 11, tzinfo=ZoneInfo("Asia/Shanghai"))
EMPTY = IntentPlan(IntentKind.TASK)
BODY = "分选试剂盒询价"


def start(text: str, plan: IntentPlan = EMPTY):
    return validate_plan_semantics(text, plan, NOW, expected_kind=IntentKind.TASK)


class FrozenLedgerDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


class RecurrenceScopeRound3Tests(unittest.TestCase):
    def test_exact_screenshot_is_a_grounded_one_off(self):
        result = start(f"明天下午两点，提醒我{BODY}。")
        self.assertTrue(result.ready, result.question)
        self.assertEqual("2026-08-31T14:00+08:00", result.plan.tasks[0].reminder_at)
        self.assertEqual(BODY, result.plan.tasks[0].title)
        self.assertIsNone(result.plan.tasks[0].reminder_recurrence)

    def test_daily_source_is_repeated_accurately_without_inventing_count(self):
        result = start("每天下午两点，提醒我分选试剂和询价。")
        self.assertFalse(result.ready)
        self.assertIn("每天", result.question)
        self.assertIn("14:00", result.question)
        self.assertNotIn("事项和次数", result.question)
        self.assertNotIn("还需要明确频率", result.question)
        self.assertIn("就明天一次", result.question)
        self.assertEqual("unsupported", result.pending.task.reminder_recurrence.frequency)
        self.assertEqual(0, result.pending.task.reminder_recurrence.count)
        self.assertEqual("", result.pending.reminder_date)

    def test_other_unsupported_frequencies_are_not_called_daily(self):
        for frequency in ("每月", "每年", "隔周", "每两周"):
            with self.subTest(frequency=frequency):
                result = start(f"{frequency}下午两点提醒我{BODY}")
                self.assertFalse(result.ready)
                self.assertIn(f"“{frequency}”", result.question)
                self.assertNotIn("事项和次数", result.question)

    def test_counts_in_body_or_quotes_do_not_create_recurrence(self):
        for body in (
            "查看“共三次询价”的记录",
            "检查‘总共三次’实验的试剂盒",
            "查看《提醒我三次》的转写结果",
            "检查连续三次实验失败的记录",
            "连续三次询价",
            "共三次报价",
            "整理实验结果，记得共三次报价",
            "统计本次实验结果共三次",
        ):
            with self.subTest(body=body):
                text = f"明天下午两点，提醒我{body}。"
                signals = extract_task_semantics(text, NOW)
                result = start(text)
                self.assertIsNone(signals.repeat_count)
                self.assertFalse(signals.recurrence_requested)
                self.assertTrue(result.ready, result.question)
                self.assertEqual(body.replace("，", " "), result.plan.tasks[0].title)
                self.assertIsNone(result.plan.tasks[0].reminder_recurrence)

    def test_front_loaded_body_recurrence_words_are_not_controls(self):
        for body in ("检查每天备份的日志", "查看共三次询价的记录", "统计总共三次"):
            with self.subTest(body=body):
                result = start(f"{body}，明天下午两点提醒我")
                self.assertTrue(result.ready, result.question)
                self.assertEqual(body, result.plan.tasks[0].title)
                self.assertIsNone(result.plan.tasks[0].reminder_recurrence)

    def test_existing_delimited_total_count_grammars_remain_supported(self):
        for tail in ("共三次", "总共3次", "一共三次", "连续三周", "连续三次", "三次，总共"):
            with self.subTest(tail=tail):
                result = start(f"每周二下午两点提醒我{BODY}，{tail}")
                self.assertTrue(result.ready, result.question)
                self.assertEqual(BODY, result.plan.tasks[0].title)
                self.assertEqual(3, result.plan.tasks[0].reminder_recurrence.count)

    def test_direct_reminder_count_with_no_body_is_preserved(self):
        result = start("再提醒我三次")
        self.assertFalse(result.ready)
        self.assertEqual("", result.pending.task.title)
        self.assertEqual(3, result.pending.task.reminder_recurrence.count)
        first = start(f"明天提醒我{BODY}")
        result = resume_pending_task(first.pending, "还要提醒三次", NOW)
        self.assertFalse(result.ready)
        self.assertEqual(3, result.pending.task.reminder_recurrence.count)

    def test_single_occurrence_per_week_is_not_a_total_count(self):
        result = start(f"每周二下午两点提醒我一次{BODY}")
        self.assertFalse(result.ready)
        self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
        self.assertEqual(0, result.pending.task.reminder_recurrence.count)

    def test_outer_count_does_not_erase_inner_quoted_count(self):
        result = start("每周二下午两点提醒我查看‘共三次’的记录，共四次")
        self.assertTrue(result.ready, result.question)
        self.assertEqual("查看‘共三次’的记录", result.plan.tasks[0].title)
        self.assertEqual(4, result.plan.tasks[0].reminder_recurrence.count)

    def test_explicit_unbounded_control_still_fails_closed(self):
        for prefix in ("一直", "长期", "重复", "循环"):
            with self.subTest(prefix=prefix):
                result = start(f"每周二下午两点{prefix}提醒我{BODY}")
                self.assertFalse(result.ready)
                self.assertIn("提醒", result.question)

    def test_explicit_one_off_corrections_keep_body_and_time(self):
        pending = start(f"每天下午两点提醒我{BODY}").pending
        for text in (
            "不是每天，是明天", "明天不是每天", "明天，不是每天",
            "就明天一次", "明天提醒我一次", "只在明天提醒一次",
            "改成一次性，明天下午两点", "改成单次，明天下午两点",
            "明天下午两点，只提醒一次", "不是每天，而是明天下午两点",
        ):
            with self.subTest(text=text):
                self.assertTrue(looks_like_pending_correction(text))
                result = resume_pending_task(pending, text, NOW)
                self.assertTrue(result.ready, result.question)
                task = result.plan.tasks[0]
                self.assertEqual(BODY, task.title)
                self.assertEqual("2026-08-31T14:00+08:00", task.reminder_at)
                self.assertIsNone(task.reminder_recurrence)

    def test_single_day_reply_does_not_silently_cancel_real_repeat(self):
        pending = start(f"每天下午两点提醒我{BODY}").pending
        for text in ("明天", "明天下午两点"):
            with self.subTest(text=text):
                self.assertFalse(looks_like_pending_correction(text))
                result = resume_pending_task(pending, text, NOW)
                self.assertFalse(result.ready)
                self.assertEqual("unsupported", result.pending.task.reminder_recurrence.frequency)
                self.assertIn("一次", result.question)
                self.assertIn("开始日期", result.question)

    def test_one_off_without_date_keeps_clock_and_asks_only_date(self):
        pending = start(f"每天下午两点提醒我{BODY}").pending
        first = resume_pending_task(pending, "只提醒一次", NOW)
        self.assertFalse(first.ready)
        self.assertIs(first.reason, ClarificationReason.MISSING_REMINDER_DATE)
        self.assertIsNone(first.pending.task.reminder_recurrence)
        self.assertEqual("14:00", first.pending.reminder_time)
        result = resume_pending_task(first.pending, "明天", NOW)
        self.assertTrue(result.ready, result.question)
        self.assertEqual("2026-08-31T14:00+08:00", result.plan.tasks[0].reminder_at)

    def test_weekday_date_is_allowed_for_explicit_one_off(self):
        pending = start(f"每天下午两点提醒我{BODY}").pending
        for text in ("改成一次性，下周二下午两点", "下周二提醒我一次"):
            with self.subTest(text=text):
                self.assertTrue(looks_like_pending_correction(text))
                result = resume_pending_task(pending, text, NOW)
                self.assertTrue(result.ready, result.question)
                self.assertEqual("2026-09-01T14:00+08:00", result.plan.tasks[0].reminder_at)
                self.assertIsNone(result.plan.tasks[0].reminder_recurrence)

    def test_correction_helper_rejects_body_quotes_questions_and_non_single_counts(self):
        for text in (
            "明天下午两点提醒我买一次性手套", "只提醒一次可以吗？",
            "检查‘不是每天，是明天’的转写结果", "‘就明天一次’",
            "不是每天，是明天三次", "不是每天，是明天共三次",
            "不是每天，是明天共0次", "每周二提醒我一次",
            "改成一次性，每周二下午两点", "不是牛奶，是试剂盒",
            "明天提醒我一次买试剂盒", "就明天一次，另外提醒我买牛奶",
            "不是每天，是取消", "不是每天，是不用了", "取消，不是每天",
            "不是每天，是下午两点", "不是每周二，是周三",
        ):
            with self.subTest(text=text):
                self.assertFalse(looks_like_pending_correction(text))

    def test_unknown_count_never_becomes_explicit_zero_after_frequency_followup(self):
        pending = start(f"每天下午两点提醒我{BODY}").pending
        result = resume_pending_task(pending, "每周二", NOW)
        self.assertFalse(result.ready)
        self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
        self.assertNotIn("需要在2到52", result.question)
        result = resume_pending_task(result.pending, "共三次", NOW)
        self.assertTrue(result.ready, result.question)
        self.assertEqual(3, result.plan.tasks[0].reminder_recurrence.count)

    def test_explicit_zero_survives_ledger_roundtrip_and_followup(self):
        first = start(f"每周二下午两点提醒我{BODY}，共0次")
        self.assertFalse(first.ready)
        self.assertIn("2到52", first.question)
        self.assertEqual(-1, first.pending.task.reminder_recurrence.count)
        with closing(IdempotencyLedger(":memory:")) as ledger:
            with patch("wechat_secretary.ledger.datetime", FrozenLedgerDateTime):
                ledger.set_pending_task("count-zero", first.pending, NOW + timedelta(minutes=10))
                pending = ledger.peek_pending_task("count-zero", NOW)
            self.assertEqual(-1, pending.task.reminder_recurrence.count)
            result = resume_pending_task(pending, "下午三点", NOW)
            self.assertFalse(result.ready)
            self.assertIn("2到52", result.question)
            self.assertEqual(-1, result.pending.task.reminder_recurrence.count)
            recovered = resume_pending_task(result.pending, "共三次", NOW)
            self.assertTrue(recovered.ready, recovered.question)
            self.assertEqual(3, recovered.plan.tasks[0].reminder_recurrence.count)

    def test_unknown_count_survives_legacy_zero_roundtrip(self):
        first = start(f"每周二下午两点提醒我{BODY}")
        self.assertEqual(0, first.pending.task.reminder_recurrence.count)
        with closing(IdempotencyLedger(":memory:")) as ledger:
            with patch("wechat_secretary.ledger.datetime", FrozenLedgerDateTime):
                ledger.set_pending_task("count-unknown", first.pending, NOW + timedelta(minutes=10))
                pending = ledger.peek_pending_task("count-unknown", NOW)
            result = resume_pending_task(pending, "下午三点", NOW)
            self.assertFalse(result.ready)
            self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)

    def test_model_schedules_are_cleared_before_any_pending_is_preserved(self):
        hallucinated = IntentPlan(IntentKind.TASK, tasks=(TaskDraft(
            BODY, reminder_at="2040-01-01T01:00+08:00",
            reminder_recurrence=ReminderRecurrence(weekday=2, count=7),
        ),))
        for text in (
            f"明天提醒我{BODY}", f"下午两点提醒我{BODY}",
            f"提醒我{BODY}", f"明天25:00提醒我{BODY}",
            f"明天或后天下午两点提醒我{BODY}",
            f"明天下午两点或三点提醒我{BODY}",
        ):
            with self.subTest(text=text):
                result = start(text, hallucinated)
                self.assertFalse(result.ready)
                self.assertIsNotNone(result.pending)
                self.assertEqual("", result.pending.task.reminder_at)
                self.assertIsNone(result.pending.task.reminder_recurrence)
                self.assertEqual("", result.plan.tasks[0].reminder_at)
                self.assertIsNone(result.plan.tasks[0].reminder_recurrence)

    def test_missing_body_drops_model_fields_but_keeps_source_clock(self):
        hallucinated = IntentPlan(IntentKind.TASK, tasks=(TaskDraft(
            "幻觉事项", reminder_at="2040-01-01T01:00+08:00",
            reminder_recurrence=ReminderRecurrence(weekday=2, count=7),
        ),))
        result = start("明天下午两点提醒我", hallucinated)
        self.assertFalse(result.ready)
        self.assertEqual("", result.pending.task.title)
        self.assertEqual("", result.pending.task.reminder_at)
        self.assertIsNone(result.pending.task.reminder_recurrence)
        self.assertEqual("2026-08-31", result.pending.reminder_date)
        self.assertEqual("14:00", result.pending.reminder_time)

    def test_sourced_recurrence_survives_conflict_without_model_contamination(self):
        hallucinated = IntentPlan(IntentKind.TASK, tasks=(TaskDraft(
            BODY, reminder_at="2040-01-01T01:00+08:00",
            reminder_recurrence=ReminderRecurrence(weekday=5, count=7),
        ),))
        result = start(f"每周二下午两点或三点提醒我{BODY}，共三次", hallucinated)
        self.assertFalse(result.ready)
        self.assertEqual("", result.pending.task.reminder_at)
        self.assertEqual(2, result.pending.task.reminder_recurrence.weekday)
        self.assertEqual(3, result.pending.task.reminder_recurrence.count)

    def test_one_off_correction_does_not_reuse_rejected_or_past_time(self):
        pending = start(f"每天下午两点提醒我{BODY}").pending
        for text in ("改成一次性，明天25:00", "改成一次性，明天下午两点多"):
            with self.subTest(text=text):
                result = resume_pending_task(pending, text, NOW)
                self.assertFalse(result.ready)
                self.assertEqual("", result.pending.reminder_time)
                self.assertIsNone(result.pending.task.reminder_recurrence)
        result = resume_pending_task(pending, "今天提醒我一次", NOW)
        self.assertFalse(result.ready)
        self.assertIn("已经过去", result.question)
        self.assertIsNone(result.pending.task.reminder_recurrence)


if __name__ == "__main__":
    unittest.main()
