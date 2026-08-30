from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from test_secretary import message, settings
from test_colloquial_schedule_round2 import NOW
from wechat_secretary.dida import DidaExecutor
from wechat_secretary.classifier import HeuristicClassifier
from wechat_secretary.models import ExecutionStatus, IntentKind, IntentPlan, TaskDraft
from wechat_secretary.semantic_guard import validate_plan_semantics


class WriteContractRound2Tests(unittest.TestCase):
    def test_offline_demo_honors_query_route_without_write_drafts(self):
        classifier = HeuristicClassifier(settings())
        plan = classifier.classify(
            message("demo-query", "查一下我今天有哪些任务"),
            "查一下我今天有哪些任务", IntentKind.QUERY, (), (),
        )
        self.assertEqual(IntentKind.QUERY, plan.kind)
        self.assertEqual("today", plan.query.mode)
        self.assertEqual((), plan.tasks)
        self.assertEqual((), plan.notes)

    def create(self, fields, draft=None):
        calls = []

        def caller(server, tool, arguments, timeout):
            calls.append(tool)
            node = {"id": "created", "title": "提交报告", "projectId": "inbox", "status": 0}
            if tool == "get_task_by_id":
                node.update(fields)
            return {"ok": True, "structuredContent": node}

        executor = DidaExecutor(settings(
            dry_run=False, dida_mapping_confirmed=True, dida_schema_confirmed=True,
        ), caller)
        with patch.dict(os.environ, {"SECRETARY_DIDA_CREATES_APPROVED": "1"}):
            result = executor.create_task(draft or TaskDraft(
                "提交报告", due_date="2026-08-31", due_time="16:30", priority="high",
            ), message("contract-round2", "待办：提交报告"))
        self.assertEqual(["create_task", "get_task_by_id"], calls)
        return result

    def test_required_fields_must_be_verified_before_reporting_success(self):
        complete = {"dueDate": "2026-08-31T16:30:00+08:00", "isAllDay": False, "priority": 5}
        variants = [
            {}, {**complete, "dueDate": "2026-08-31T04:30:00+08:00"},
            {**complete, "dueDate": "2026-08-31T16:30:00"},
            {**complete, "dueDate": "bad"}, {**complete, "isAllDay": True},
            {**complete, "priority": 1}, {**complete, "priority": True},
        ]
        variants.extend({key: val for key, val in complete.items() if key != missing}
                        for missing in complete)
        for fields in variants:
            with self.subTest(fields=fields):
                result = self.create(fields)
                self.assertEqual(ExecutionStatus.UNCERTAIN, result.status)
                self.assertIn("不要重试", result.error)

    def test_equivalent_utc_readback_is_accepted(self):
        for due in ("2026-08-31T08:30:00Z", "2026-08-31T08:30:00.000+0000"):
            result = self.create({"dueDate": due, "isAllDay": False, "priority": 5})
            self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)

    def test_all_day_deadline_is_verified_and_plain_task_needs_no_clock(self):
        draft = TaskDraft("提交报告", due_date="2026-08-31")
        result = self.create({"dueDate": "2026-08-30T16:00:00Z", "isAllDay": True}, draft)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        result = self.create({"dueDate": "2026-08-30T16:00:00Z", "isAllDay": False}, draft)
        self.assertEqual(ExecutionStatus.UNCERTAIN, result.status)
        self.assertEqual(ExecutionStatus.SUCCEEDED, self.create({}, TaskDraft("提交报告")).status)

    def test_category_requires_an_explicit_outer_metadata_instruction(self):
        for text, title, expected in (
            ("明天下午四点提醒我阅读《工作》", "阅读《工作》", ""),
            ("明天下午四点提醒我提交报告，不要归到工作", "提交报告", ""),
            ("明天下午四点提醒我提交报告，分类：工作", "提交报告", "工作"),
            ("明天下午四点提醒我提交报告，归入工作清单", "提交报告", "工作"),
            ("明天下午四点提醒我提交报告，高优先级，工作", "提交报告", "工作"),
            ("明天下午四点提醒我提交报告，分类：工作，不要归到工作", "提交报告", ""),
            ("明天下午四点提醒我提交报告，工作，分类不要工作", "提交报告", ""),
            ("明天下午四点提醒我提交报告，分类：工作，分类不是工作", "提交报告", ""),
            ("明天下午四点提醒我提交报告，分类：工作，不是工作", "提交报告", ""),
            ("明天下午四点提醒我提交报告，分类：工作，分类：个人", "提交报告", ""),
        ):
            with self.subTest(text=text):
                plan = IntentPlan(IntentKind.TASK, tasks=(TaskDraft(title, category="工作"),))
                decision = validate_plan_semantics(text, plan, NOW, expected_kind=IntentKind.TASK)
                self.assertTrue(decision.ready, decision.question)
                self.assertEqual(expected, decision.plan.tasks[0].category)


if __name__ == "__main__":
    unittest.main()
