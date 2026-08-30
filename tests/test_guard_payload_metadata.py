from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import test_natural_routing as service_fixtures

from wechat_secretary.models import ExecutionStatus, IntentKind, IntentPlan, NoteDraft, TaskDraft, TaskQuery
from wechat_secretary.semantic_guard import validate_plan_semantics


NOW = datetime(2026, 8, 30, 13, 25, tzinfo=ZoneInfo("Asia/Shanghai"))
MODEL_PLAN = IntentPlan(IntentKind.TASK, tasks=(TaskDraft("模型错误的标题", priority="high"),))


def task_from(text):
    decision = validate_plan_semantics(text, MODEL_PLAN, NOW, expected_kind=IntentKind.TASK)
    if not decision.ready:
        raise AssertionError(decision.question)
    return decision.plan.tasks[0]


class GuardPayloadMetadataTests(unittest.TestCase):
    def test_query_scope_clarification_is_preserved_with_zero_writes(self):
        question = "你想查询今天还是本周的任务？"
        plan = IntentPlan(IntentKind.CLARIFY, clarification=question)
        decision = validate_plan_semantics("查一下我的任务", plan, NOW, expected_kind=IntentKind.QUERY)
        self.assertFalse(decision.ready)
        self.assertEqual(question, decision.question)
        self.assertIsNone(decision.pending)
        service, dida, ledger = service_fixtures.make_service(service_fixtures.StaticClassifier(plan))
        self.addCleanup(ledger.close)
        result = service.handle(service_fixtures.make_message("query-scope", "查一下我的任务"))
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(question, result.reply)
        self.assertEqual([], dida.create_calls)
        self.assertEqual((), result.results)

    def test_query_clarification_cannot_smuggle_writes_or_query_execution(self):
        clean = IntentPlan(IntentKind.CLARIFY, clarification="你想查询今天还是本周？")
        for plan in (
            replace(clean, tasks=(TaskDraft("未授权任务"),)),
            replace(clean, notes=(NoteDraft("未授权笔记", "正文"),)),
            replace(clean, query=TaskQuery()),
            replace(clean, clarification="  "),
        ):
            with self.subTest(plan=plan):
                decision = validate_plan_semantics("查一下我的任务", plan, NOW, expected_kind=IntentKind.QUERY)
                self.assertFalse(decision.ready)
                self.assertIsNone(decision.pending)
                self.assertIn("不是可靠的纯查询结构", decision.question)

    def test_multiple_independent_reminders_never_merge_into_one_timed_task(self):
        for text in (
            "今天下午三点提醒我买牛奶，下午四点提醒我买咖啡",
            "今天下午三点提醒我买牛奶。明天下午四点提醒我买咖啡",
            "今天下午三点提醒我买牛奶；另外明天下午四点提醒我买咖啡",
        ):
            for tasks in ((TaskDraft("买牛奶"),), (TaskDraft("买牛奶"), TaskDraft("买咖啡"))):
                with self.subTest(text=text, task_count=len(tasks)):
                    decision = validate_plan_semantics(text, IntentPlan(IntentKind.TASK, tasks=tasks), NOW, expected_kind=IntentKind.TASK)
                    self.assertFalse(decision.ready)
                    self.assertIsNone(decision.pending)
                    self.assertIn("分开", decision.question)

    def test_mixed_note_and_two_independent_reminders_also_require_separation(self):
        text = "帮我记一下实验结论A，另外今天下午三点提醒我买牛奶，下午四点提醒我买咖啡"
        plan = IntentPlan(IntentKind.MIXED, tasks=(TaskDraft("买牛奶"),), notes=(NoteDraft("实验结论", "实验结论A"),))
        decision = validate_plan_semantics(text, plan, NOW, expected_kind=IntentKind.MIXED)
        self.assertFalse(decision.ready)
        self.assertIsNone(decision.pending)
        self.assertIn("分开", decision.question)

    def test_quoted_or_future_question_reminder_words_are_not_extra_requests(self):
        for body in (
            "阅读《明天下午四点提醒我买咖啡》",
            "问导师能否明天下午四点提醒我交材料",
        ):
            with self.subTest(body=body):
                self.assertEqual(body, task_from(f"今天下午三点提醒我{body}").title)
        text = "帮我记一下今天下午三点提醒我买牛奶，明天下午四点提醒我买咖啡"
        decision = validate_plan_semantics(text, IntentPlan(IntentKind.NOTE), NOW, expected_kind=IntentKind.NOTE)
        self.assertTrue(decision.ready)
        self.assertEqual((), decision.plan.tasks)

    def test_future_question_keeps_its_interrogative_particle(self):
        for body in ("问导师材料交了吗", "询问导师材料交了吗", "确认材料交了吗", "问导师材料在哪呢"):
            with self.subTest(body=body):
                task = task_from(f"今天下午三点提醒我{body}？")
                self.assertEqual(body, task.title)
                self.assertEqual("2026-08-30T15:00+08:00", task.reminder_at)

    def test_polite_request_does_not_strip_future_question(self):
        task = task_from("请今天下午三点提醒我问导师这样可以吗？")
        self.assertEqual("问导师这样可以吗", task.title)
        task = task_from("今天下午三点提醒我问导师材料交了吗，可以吗？")
        self.assertEqual("问导师材料交了吗", task.title)

    def test_outer_courtesy_is_not_part_of_an_ordinary_task(self):
        for text in (
            "你能今天下午三点提醒我买牛奶吗？",
            "今天下午三点提醒我买牛奶，好吗？",
            "今天下午三点提醒我买牛奶可以吗？",
        ):
            with self.subTest(text=text):
                self.assertEqual("买牛奶", task_from(text).title)

    def test_negative_importance_and_body_nouns_do_not_raise_priority(self):
        for body in (
            "买牛奶，不重要、不紧急",
            "查一下重要蛋白有哪些",
            "阅读紧急刹车操作手册",
            "告诉导师这项工作非常重要",
            "研究高优先级调度策略",
            "阅读《高优先级》",
            "查一下日志，紧急",
        ):
            with self.subTest(body=body):
                self.assertEqual("none", task_from(f"今天下午三点提醒我{body}").priority)

    def test_only_explicit_standalone_priority_instructions_are_applied(self):
        for instruction, expected in (
            ("高优先级", "high"),
            ("优先级：高", "high"),
            ("请把任务的优先级设为高", "high"),
            ("设置为中优先级", "medium"),
            ("这个任务的优先级设为低", "low"),
            ("优先级为普通", "none"),
        ):
            with self.subTest(instruction=instruction):
                self.assertEqual(expected, task_from(f"今天下午三点提醒我买牛奶，{instruction}").priority)

    def test_negated_or_conflicting_priority_commands_do_not_raise_priority(self):
        for instruction in (
            "不要设为高优先级",
            "优先级不要高",
            "不是高优先级",
            "高优先级，不要高优先级",
            "高优先级，低优先级",
            "备注里写着优先级高",
            "请阅读“优先级设为高”这条记录",
        ):
            with self.subTest(instruction=instruction):
                self.assertEqual("none", task_from(f"今天下午三点提醒我买牛奶，{instruction}").priority)


if __name__ == "__main__":
    unittest.main()
