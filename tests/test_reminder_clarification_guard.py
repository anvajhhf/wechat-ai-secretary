from __future__ import annotations

import unittest
from datetime import timedelta

from wechat_secretary.models import (
    ClarificationReason,
    ExecutionStatus,
    IntentKind,
    IntentPlan,
    NoteDraft,
    PendingTaskClarification,
    ReminderRecurrence,
    TaskDraft,
    TaskQuery,
)
from wechat_secretary.semantic_guard import (
    looks_like_pending_followup,
    resume_pending_task,
    validate_plan_semantics,
)

from test_natural_routing import NOW, StaticClassifier, make_message, make_service


EMPTY_PLAN = IntentPlan(
    kind=IntentKind.CLARIFY,
    confidence=0.2,
    clarification="请补充具体事项。",
)


class ReminderClarificationGuardTests(unittest.TestCase):
    def test_empty_model_result_still_preserves_explicit_reminder_body(self) -> None:
        service, dida, ledger = make_service(StaticClassifier(EMPTY_PLAN))
        self.addCleanup(ledger.close)

        first = service.handle(make_message("empty-model", "今天提醒我查一下本地技能有哪些。"))
        self.assertIs(first.status, ExecutionStatus.SKIPPED)
        self.assertIn("几点", first.reply)
        self.assertEqual([], dida.create_calls)

        completed = service.handle(make_message(
            "clock-only", "下午四点的时候。", NOW + timedelta(minutes=1)
        ))
        self.assertIs(completed.status, ExecutionStatus.PLANNED)
        self.assertFalse(completed.llm_called)
        self.assertEqual(1, len(dida.create_calls))
        self.assertEqual("查一下本地技能有哪些", dida.create_calls[0].title)
        self.assertEqual("2026-08-24T16:00+08:00", dida.create_calls[0].reminder_at)

    def test_empty_model_result_with_complete_source_needs_no_clarification(self) -> None:
        guard = validate_plan_semantics(
            "今天下午三点的时候提醒我让ChatGPT看看网上有没有更好的技能。",
            EMPTY_PLAN, NOW, expected_kind=IntentKind.TASK,
        )
        self.assertTrue(guard.ready)
        self.assertIs(guard.plan.kind, IntentKind.TASK)
        self.assertEqual("2026-08-24T15:00+08:00", guard.plan.tasks[0].reminder_at)

    def test_fallback_drops_model_invented_other_actions(self) -> None:
        unrelated = IntentPlan(
            kind=IntentKind.NOTE,
            notes=(NoteDraft("外部指令", "不存在的笔记"),),
            query=TaskQuery(),
        )
        guard = validate_plan_semantics(
            "今天下午四点提醒我查一下技能怎么优化", unrelated, NOW,
            expected_kind=IntentKind.TASK,
        )
        self.assertTrue(guard.ready)
        self.assertEqual(1, len(guard.plan.tasks))
        self.assertEqual((), guard.plan.notes)
        self.assertIsNone(guard.plan.query)

    def test_fallback_does_not_authorize_unrouted_or_negated_input(self) -> None:
        for text in (
            "你会在今天下午三点提醒我买牛奶吗？",
            "为什么今天下午三点提醒我查一下天气？",
            "不要今天下午三点提醒我买牛奶",
        ):
            with self.subTest(text=text):
                service, dida, ledger = make_service(StaticClassifier(EMPTY_PLAN))
                self.addCleanup(ledger.close)
                result = service.handle(make_message("not-a-request", text))
                self.assertIs(result.status, ExecutionStatus.SKIPPED)
                self.assertEqual([], dida.create_calls)

    def test_schedule_scaffolding_is_not_a_task_body(self) -> None:
        for text in (
            "今天下午三点的时候提醒我",
            "提醒我今天下午三点的时候",
            "今天下午四点的时候，记得提醒我。",
            "今天下午三点的时候，请提醒我",
            "明天下午3点务必提醒我",
            "今天下午三点提醒我一下",
        ):
            for plan in (EMPTY_PLAN, IntentPlan(IntentKind.TASK, tasks=(TaskDraft("幻觉事项"),))):
                with self.subTest(text=text, plan=plan):
                    guard = validate_plan_semantics(text, plan, NOW, expected_kind=IntentKind.TASK)
                    self.assertFalse(guard.ready)
                    self.assertIs(guard.reason, ClarificationReason.MISSING_TASK_BODY)
                    self.assertIsNotNone(guard.pending)
                    self.assertEqual("", guard.pending.task.title)

    def test_chinese_leading_and_trailing_schedule_removed_from_title(self) -> None:
        for text in (
            "提醒我今天下午三点的时候查一下技能有哪些",
            "查一下技能有哪些，今天下午三点的时候提醒我",
        ):
            with self.subTest(text=text):
                guard = validate_plan_semantics(text, EMPTY_PLAN, NOW, expected_kind=IntentKind.TASK)
                self.assertTrue(guard.ready)
                self.assertEqual("查一下技能有哪些", guard.plan.tasks[0].title)

    def test_pure_spoken_fields_accept_punctuation_but_not_questions(self) -> None:
        for text in (
            "今天下午四点。", "就下午三点半吧。", "改成下午四点的时候。",
            "明天上午九点十五分！", "下午三点左右。", "明天25:00。", "连续三次。",
        ):
            with self.subTest(text=text):
                self.assertTrue(looks_like_pending_followup(text))
        for text in (
            "今天下午四点有空吗？", "明天三点的天气怎么样", "下午三点买牛奶。",
            "下午四点以后都行。", "会议持续3小时。", "三点水怎么写？", "取消吗？",
        ):
            with self.subTest(text=text):
                self.assertFalse(looks_like_pending_followup(text))

    def test_invalid_or_approximate_replacement_never_reuses_old_clock(self) -> None:
        for recurrence in (None, ReminderRecurrence(weekday=2, count=3)):
            for text in ("明天25:00。", "明天15:70。", "明天下午四点多。", "明天下午三点左右。"):
                with self.subTest(recurrence=recurrence, text=text):
                    pending = PendingTaskClarification(
                        reason=ClarificationReason.MISSING_REMINDER_DATE,
                        task=TaskDraft("查一下技能有哪些", reminder_recurrence=recurrence),
                        reminder_time="16:00",
                    )
                    guard = resume_pending_task(pending, text, NOW)
                    self.assertFalse(guard.ready)
                    self.assertIs(guard.reason, ClarificationReason.MISSING_REMINDER_TIME)
                    self.assertIsNotNone(guard.pending)
                    self.assertEqual("", guard.pending.reminder_time)
                    self.assertEqual("查一下技能有哪些", guard.pending.task.title)

    def test_spoken_cancel_clears_pending_without_creating(self) -> None:
        for text in ("取消。", "算了！", "不用了。", "不设置了。"):
            with self.subTest(text=text):
                service, dida, ledger = make_service(StaticClassifier(EMPTY_PLAN))
                self.addCleanup(ledger.close)
                source = make_message("pending-source", "今天提醒我查一下技能有哪些。")
                service.handle(source)
                cancelled = service.handle(make_message("cancel", text, NOW + timedelta(minutes=1)))
                self.assertIs(cancelled.status, ExecutionStatus.SKIPPED)
                self.assertIn("已取消", cancelled.reply)
                self.assertEqual([], dida.create_calls)
                self.assertIsNone(ledger.claim_pending_task(
                    source.conversation_key, "probe", NOW + timedelta(minutes=2)
                ).pending)


if __name__ == "__main__":
    unittest.main()
