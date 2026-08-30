"""Independent route/response-boundary audit: no real model or external writes."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import test_voice_reminder_conversation as fixtures
from wechat_secretary.classifier import HermesStructuredClassifier, plan_from_mapping
from wechat_secretary.models import ExecutionStatus, IntentKind
from wechat_secretary.request_scope import has_multiple_reminder_requests, outer_reminder_match
from wechat_secretary.routing import detect_route_hint, is_non_action_task_utterance


class IntentAuditRound2Tests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service
    message = staticmethod(fixtures.VoiceReminderConversationTests.message)

    def test_punctuation_in_schedule_only_prefix_preserves_request(self):
        for separator in ("，", ",", "、", "。", ".", "；", ";", "！", "!", "\n"):
            for schedule in ("今天下午4点半", "明天下午四点半", "二十分钟后"):
                text = f"{schedule}{separator}提醒我看一下微信助手。"
                with self.subTest(text=text):
                    self.assertIsNotNone(outer_reminder_match(text))
                    self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
                    self.assertFalse(is_non_action_task_utterance(text))

    def test_frontloaded_request_allows_clause_punctuation(self):
        for separator in ("，", ",", "。", ".", "；", ";", "！", "!", "\n"):
            text = f"看一下微信助手{separator}今天下午4点半提醒我。"
            with self.subTest(text=text):
                self.assertIsNotNone(outer_reminder_match(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
        for text in (
            "买1.5升牛奶，2026.09.01下午四点半提醒我",
            "看一下微信助手.2026.09.01下午四点半提醒我",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(outer_reminder_match(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)

    def test_punctuation_never_skips_untrusted_prefix_or_question(self):
        for separator in ("，", "。", ";", "\n"):
            for prefix in ("他说", "转发", "示例", "昨天的原话", "“同事说”", "不要", "为什么"):
                text = f"{prefix}{separator}今天下午四点半{separator}提醒我看一下微信助手。"
                with self.subTest(text=text):
                    self.assertIsNone(outer_reminder_match(text))
                    self.assertIsNone(detect_route_hint(text).kind)
        for text in (
            "今天下午四点半？提醒我看一下微信助手。",
            "今天下午四点半?提醒我看一下微信助手。",
            "“今天下午四点半。提醒我看一下微信助手。”",
            "看过微信助手。今天下午四点半提醒我。",
            "他说看一下微信助手。今天下午四点半提醒我。",
            "买牛奶。交报告。今天下午四点半提醒我。",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_route_hint(text).kind)

    def test_query_requires_an_outer_read_request(self):
        for text in (
            "查一下今天有哪些任务",
            "帮我查一下今天有哪些任务",
            "麻烦你看看明天的待办",
            "你能帮我查一下我的任务吗？",
            "今天的任务有哪些？",
            "我想知道任务有哪些",
            "找一下我的笔记",
            "查一下API开发相关的任务",
            "查一下任务：编写API文档",
        ):
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.QUERY)
        for text in (
            "他说：查一下今天有哪些任务",
            "转发：帮我查一下今天有哪些任务",
            "示例：今天的任务有哪些？",
            "“同事说”查一下今天有哪些任务",
            "“查一下今天有哪些任务”",
            "不要查一下今天的任务",
            "昨天已经查一下任务",
            "你知道如何查询任务吗？",
            "查询任务的接口怎么用？",
            "帮我查一下任务查询的API",
        ):
            with self.subTest(text=text):
                self.assertIsNot(detect_route_hint(text).kind, IntentKind.QUERY)
        self.assertIs(detect_route_hint("帮我记下来：查一下今天有哪些任务").kind, IntentKind.NOTE)
        self.assertIs(detect_route_hint("明天下午三点提醒我查一下今天有哪些任务").kind, IntentKind.TASK)

    def test_independent_reminders_with_split_schedules_stay_multiple(self):
        for separator in ("，", "。", ";", "\n"):
            text = f"明天下午三点。提醒我买牛奶{separator}明天下午四点。提醒我交报告"
            with self.subTest(text=text):
                self.assertTrue(has_multiple_reminder_requests(text))

    def test_null_and_nonlist_optional_metadata_do_not_crash_or_coerce(self):
        for metadata in (None, 123, "工作", {"工作": "secret"}, [None, 5, {"x": "y"}]):
            with self.subTest(metadata=metadata):
                task = plan_from_mapping({"kind": "task", "tasks": [{"title": "提交报告", "tags": metadata}]}).tasks[0]
                note = plan_from_mapping({"kind": "note", "notes": [{"body": "事实正文", "tags": metadata, "links": metadata}]}, allowed_links=("工作",)).notes[0]
                self.assertEqual((), task.tags)
                self.assertEqual((), note.tags)
                self.assertEqual((), note.links)

    def test_metadata_only_accepts_existing_string_candidates(self):
        plan = plan_from_mapping({
            "kind": "note",
            "notes": [{"body": "事实正文", "tags": ["工作", {"x": 1}, "工作"], "links": ["工作", "不存在", None]}],
        }, allowed_links=("工作",))
        self.assertEqual(("工作",), plan.notes[0].tags)
        self.assertEqual(("工作",), plan.notes[0].links)

    def test_nonfinite_confidence_never_becomes_write_confidence(self):
        for confidence in (float("nan"), float("inf"), float("-inf"), "nan", "Infinity", {}, []):
            with self.subTest(confidence=confidence):
                plan = plan_from_mapping({"kind": "task", "tasks": [{"title": "提交报告"}], "confidence": confidence})
                self.assertTrue(math.isfinite(plan.confidence))
                self.assertEqual(0.0, plan.confidence)

    def test_structural_and_partial_drafts_fail_closed(self):
        for key, draft in (("tasks", {"title": "提交报告"}), ("notes", {"body": "事实正文"})):
            for value in ("not-a-list", {}, [draft, None], [draft, {}], [draft] * 4):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        plan_from_mapping({"kind": "mixed", key: value})

    def test_query_without_forced_route_still_needs_valid_scope(self):
        for query in (None, {}, {"mode": "yesterday", "keyword": ""}, {"mode": "today"}, {"mode": "search", "keyword": ""}):
            with self.subTest(query=query):
                plan = plan_from_mapping({"kind": "query", "query": query})
                self.assertIs(plan.kind, IntentKind.CLARIFY)
                self.assertIsNone(plan.query)

    def test_reported_query_cannot_read_tasks_even_if_model_says_query(self):
        for text in ("他说：查一下今天有哪些任务", "转发：查一下今天有哪些任务", "不要查一下今天的任务"):
            with self.subTest(text=text):
                service, _, media, dida, _ = self.make_service()
                dida.query_tasks = lambda *_: self.fail("An unrequested read must not access tasks")
                llm = SimpleNamespace(complete_structured=lambda **_: SimpleNamespace(parsed={
                    "kind": "query", "query": {"mode": "today", "keyword": ""}, "confidence": 1.0,
                }))
                service.classifier = HermesStructuredClassifier(SimpleNamespace(llm=llm), service.settings)
                result = service.handle(self.message("reported-query", text, media, voice=False))
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                self.assertEqual((), result.results)
                self.assertEqual([], dida.tasks)

    def test_invalid_model_drafts_cannot_reach_partial_service_writes(self):
        for key, text, draft in (
            ("tasks", "待办：提交报告，分类：工作", {"title": "提交报告"}),
            ("notes", "笔记：会议结论：A方案可行", {"body": "会议结论：A方案可行"}),
        ):
            for value in ([draft] * 4, [draft, None], [draft, {}]):
                with self.subTest(key=key, value=value):
                    service, _, media, dida, ledger = self.make_service()
                    service.obsidian = SimpleNamespace(
                        available_links=lambda _: (),
                        save=lambda *_: self.fail("Malformed draft must not write any note"),
                    )
                    llm_calls = []

                    def structured(**kwargs):
                        llm_calls.append(kwargs)
                        return SimpleNamespace(parsed={key: value, "confidence": 1.0})

                    service.classifier = HermesStructuredClassifier(SimpleNamespace(llm=SimpleNamespace(complete_structured=structured)), service.settings)
                    incoming = self.message("malformed-drafts", text, media, voice=False)
                    result = service.handle(incoming)
                    self.assertEqual(ExecutionStatus.FAILED, result.status)
                    self.assertEqual(1, len(llm_calls))
                    self.assertTrue(result.llm_called)
                    self.assertEqual([], dida.tasks)
                    self.assertEqual((), result.results)
                    self.assertEqual(0, ledger.active_reminder_count("voice-task-1", incoming))


if __name__ == "__main__":
    unittest.main()
