from __future__ import annotations

import re
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from wechat_secretary.models import (
    ClarificationReason,
    IntentKind,
    IntentPlan,
    PendingTaskClarification,
    TaskDraft,
)
from wechat_secretary.semantic_guard import (
    looks_like_pending_followup,
    resume_pending_task,
    validate_plan_semantics,
)


NOW = datetime(2026, 8, 30, 13, 25, tzinfo=ZoneInfo("Asia/Shanghai"))


def compact_title(value: str) -> str:
    return re.sub(r"[\s，,。；;！？!?：:、]+", "", value)


class ReminderScheduleScopeTests(unittest.TestCase):
    """Exercise the pure guard: no network, ledger, or production writes."""

    @staticmethod
    def validate(text: str):
        # The model's guessed clock/body must never authorize a real schedule.
        plan = IntentPlan(
            kind=IntentKind.TASK,
            tasks=(
                TaskDraft(
                    title="模型猜测的事项",
                    reminder_at="2026-08-31T09:00+08:00",
                ),
            ),
        )
        return validate_plan_semantics(text, plan, NOW, expected_kind=IntentKind.TASK)

    @staticmethod
    def pending(*, reminder_date: str = "2026-08-31", reminder_time: str = ""):
        return PendingTaskClarification(
            reason=(
                ClarificationReason.MISSING_REMINDER_TIME
                if reminder_date
                else ClarificationReason.MISSING_REMINDER_DATE
            ),
            task=TaskDraft("买牛奶"),
            reminder_date=reminder_date,
            reminder_time=reminder_time,
            source_message_id="schedule-scope-source",
        )

    def assert_clarifies(self, result) -> None:
        self.assertFalse(result.ready)
        self.assertEqual(IntentKind.CLARIFY, result.plan.kind)
        self.assertTrue(result.question)

    def test_clock_like_words_and_quoted_titles_are_not_schedule(self):
        for body in (
            "阅读三点估计方法",
            "检查三点水汉字",
            "阅读《下午三点》",
            "阅读“下午三点”这篇文章",
            "购买三点水字帖",
            "三点水字帖要买",
        ):
            with self.subTest(body=body):
                result = self.validate(f"明天提醒我{body}")
                self.assert_clarifies(result)
                self.assertEqual(ClarificationReason.MISSING_REMINDER_TIME, result.reason)
                self.assertIsNotNone(result.pending)
                self.assertEqual("2026-08-31", result.pending.reminder_date)
                self.assertEqual("", result.pending.reminder_time)
                self.assertEqual(body, result.pending.task.title)

    def test_three_schedule_positions_keep_exact_time_and_task_body(self):
        body = "让ChatGPT优化本地生信技能"
        for text in (
            f"今天下午三点的时候提醒我{body}",
            f"提醒我今天下午三点的时候{body}",
            f"{body}，今天下午三点的时候提醒我",
        ):
            with self.subTest(text=text):
                result = self.validate(text)
                self.assertTrue(result.ready, result.question)
                self.assertEqual(IntentKind.TASK, result.plan.kind)
                self.assertEqual(1, len(result.plan.tasks))
                task = result.plan.tasks[0]
                self.assertEqual("2026-08-30T15:00+08:00", task.reminder_at)
                self.assertEqual(compact_title(body), compact_title(task.title))

    def test_invalid_replacement_date_never_reuses_pending_date(self):
        for reply in (
            "2026-02-30下午三点",
            "2026-13-01下午三点",
            "13月3日下午三点",
            "8月32日下午三点",
        ):
            with self.subTest(reply=reply):
                self.assertTrue(looks_like_pending_followup(reply))
                result = resume_pending_task(self.pending(), reply, NOW)
                self.assert_clarifies(result)
                self.assertIsNotNone(result.pending)
                self.assertEqual("", result.pending.reminder_date)
                self.assertEqual("买牛奶", result.pending.task.title)

    def test_multiple_schedule_options_in_new_request_require_clarification(self):
        for text in (
            "明天下午三点或四点提醒我买牛奶",
            "明天下午三点下午四点提醒我买牛奶",
            "明天或后天下午三点提醒我买牛奶",
            "2026-08-31或2026-09-01下午三点提醒我买牛奶",
        ):
            with self.subTest(text=text):
                self.assert_clarifies(self.validate(text))

    def test_multiple_replacement_dates_or_times_cannot_pick_first_option(self):
        for reply in (
            "明天后天下午三点",
            "2026-08-31、2026-09-01下午三点",
            "下午三点下午四点",
            "下午三点、下午四点",
        ):
            with self.subTest(reply=reply):
                self.assertTrue(looks_like_pending_followup(reply))
                self.assert_clarifies(resume_pending_task(self.pending(), reply, NOW))

    def test_bare_period_reply_clears_old_exact_time(self):
        for reply in ("下午", "明天下午"):
            with self.subTest(reply=reply):
                pending = self.pending(reminder_date="", reminder_time="09:00")
                self.assertTrue(looks_like_pending_followup(reply))
                result = resume_pending_task(pending, reply, NOW)
                self.assert_clarifies(result)
                self.assertIsNotNone(result.pending)
                self.assertEqual("", result.pending.reminder_time)
                self.assertEqual("买牛奶", result.pending.task.title)


if __name__ == "__main__":
    unittest.main()
