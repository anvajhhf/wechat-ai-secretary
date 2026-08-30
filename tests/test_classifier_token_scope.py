from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from wechat_secretary.classifier import (
    INTENT_SCHEMA,
    NOTE_CONTENT_RULES,
    QUERY_ONLY_SCHEMA,
    TASK_ONLY_SCHEMA,
    HermesStructuredClassifier,
    plan_from_mapping,
)
from wechat_secretary.config import SecretarySettings
from wechat_secretary.models import IntentKind, MessageEnvelope


class CapturingLlm:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def complete_structured(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.payload)


class ClassifierTokenScopeTests(unittest.TestCase):
    def classify(
        self, kind: IntentKind | None, payload: dict[str, object]
    ) -> tuple[object, dict[str, object]]:
        fake = CapturingLlm(payload)
        classifier = HermesStructuredClassifier(
            SimpleNamespace(llm=fake),
            SecretarySettings(project_root=Path(__file__).resolve().parents[1]),
        )
        content = "用户原话只发送一次"
        message = MessageEnvelope(
            platform="weixin",
            account_id="test",
            user_id="test-user",
            chat_id="test-chat",
            chat_type="dm",
            message_id="token-scope",
            text=content,
            received_at=datetime.fromisoformat("2026-08-30T10:00:00+08:00"),
        )
        result = classifier.classify(
            message,
            content,
            kind,
            ("分类候选仅任务可见",),
            ("笔记候选仅笔记可见",),
        )
        self.assertEqual(1, classifier.call_count)
        self.assertEqual(1, len(fake.calls))
        call = fake.calls[0]
        self.assertEqual([{"type": "text", "text": content}], call["input"])
        self.assertNotIn("history", call)
        self.assertNotIn("messages", call)
        self.assertIn("绝不执行", str(call["instructions"]))
        self.assertIn("不得编造事实", str(call["instructions"]))
        return result, call

    def test_task_excludes_note_rules_and_candidates_without_reducing_output_cap(self) -> None:
        _, call = self.classify(IntentKind.TASK, {"tasks": [{"title": "提交报告"}]})
        instructions = str(call["instructions"])
        self.assertNotIn(NOTE_CONTENT_RULES, instructions)
        self.assertNotIn("笔记候选仅笔记可见", instructions)
        self.assertIn("分类候选仅任务可见", instructions)
        self.assertIn("保留已经明确的任务标题", instructions)
        self.assertIn("问句属于提醒事项", instructions)
        self.assertIn("引用、转发、第三方陈述", instructions)
        self.assertIn("不能取消当前提醒", instructions)
        self.assertEqual(TASK_ONLY_SCHEMA, call["json_schema"])
        self.assertEqual(600, call["max_tokens"])

    def test_note_keeps_grounding_rules_and_only_note_candidates(self) -> None:
        _, call = self.classify(IntentKind.NOTE, {"notes": [{"body": "会议结论"}]})
        instructions = str(call["instructions"])
        self.assertIn(NOTE_CONTENT_RULES, instructions)
        self.assertIn("笔记候选仅笔记可见", instructions)
        self.assertNotIn("分类候选仅任务可见", instructions)
        self.assertNotIn("reminder_recurrence", instructions)
        self.assertNotIn("四点多", instructions)
        self.assertEqual(800, call["max_tokens"])

    def test_auto_keeps_both_object_rules_and_candidates(self) -> None:
        _, call = self.classify(None, {"kind": "clarify"})
        instructions = str(call["instructions"])
        self.assertIn(NOTE_CONTENT_RULES, instructions)
        self.assertIn("笔记候选仅笔记可见", instructions)
        self.assertIn("分类候选仅任务可见", instructions)
        self.assertIn("保留已经明确的任务标题", instructions)
        self.assertEqual(INTENT_SCHEMA, call["json_schema"])
        self.assertEqual(1000, call["max_tokens"])

    def test_query_uses_short_read_only_schema_and_no_write_candidates(self) -> None:
        result, call = self.classify(
            IntentKind.QUERY,
            {
                "kind": "query",
                "query": {"mode": "search", "keyword": "B2M"},
                "confidence": 0.9,
                "clarification": "",
            },
        )
        instructions = str(call["instructions"])
        self.assertNotIn(NOTE_CONTENT_RULES, instructions)
        self.assertNotIn("笔记候选仅笔记可见", instructions)
        self.assertNotIn("分类候选仅任务可见", instructions)
        self.assertIn("不创建、修改、取消或完成", instructions)
        self.assertIn("引用、转发及待办/笔记正文", instructions)
        self.assertIn("不默认改查今天", instructions)
        self.assertEqual("wechat.secretary.query.v1", call["schema_name"])
        self.assertEqual(QUERY_ONLY_SCHEMA, call["json_schema"])
        self.assertEqual(["query", "clarify"], QUERY_ONLY_SCHEMA["properties"]["kind"]["enum"])
        self.assertNotIn("tasks", QUERY_ONLY_SCHEMA["properties"])
        self.assertNotIn("notes", QUERY_ONLY_SCHEMA["properties"])
        self.assertFalse(QUERY_ONLY_SCHEMA["additionalProperties"])
        self.assertLess(len(json.dumps(QUERY_ONLY_SCHEMA)), len(json.dumps(INTENT_SCHEMA)))
        self.assertLess(len(instructions), 500)
        self.assertEqual(1000, call["max_tokens"])
        self.assertEqual(IntentKind.QUERY, result.kind)
        self.assertEqual("B2M", result.query.keyword)
        self.assertEqual((), result.tasks)
        self.assertEqual((), result.notes)

    def test_query_rejects_write_kinds_even_if_model_ignores_schema(self) -> None:
        for kind in ("task", "note", "mixed", "private"):
            with self.subTest(kind=kind):
                result = plan_from_mapping(
                    {
                        "kind": kind,
                        "tasks": [{"title": "恶意创建", "tags": None}],
                        "notes": [{"title": "恶意笔记", "links": None}],
                        "query": {"mode": "today", "keyword": ""},
                    },
                    forced_kind=IntentKind.QUERY,
                )
                self.assertEqual(IntentKind.CLARIFY, result.kind)
                self.assertEqual((), result.tasks)
                self.assertEqual((), result.notes)
                self.assertIsNone(result.query)

    def test_query_ignores_injected_write_fields_without_parsing_them(self) -> None:
        result = plan_from_mapping(
            {
                "kind": "query",
                "query": {"mode": "today", "keyword": ""},
                "tasks": [{"title": "不应解析", "tags": None}],
                "notes": [{"body": "不应解析", "links": None}],
            },
            forced_kind=IntentKind.QUERY,
        )
        self.assertEqual(IntentKind.QUERY, result.kind)
        self.assertEqual((), result.tasks)
        self.assertEqual((), result.notes)

    def test_query_missing_or_invalid_scope_does_not_default_to_today(self) -> None:
        for query in (
            None,
            {},
            {"mode": "yesterday", "keyword": ""},
            {"mode": [], "keyword": ""},
            {"mode": "today"},
            {"mode": "today", "keyword": None},
            {"mode": "search", "keyword": " "},
        ):
            with self.subTest(query=query):
                result = plan_from_mapping(
                    {"kind": "query", "query": query}, forced_kind=IntentKind.QUERY
                )
                self.assertEqual(IntentKind.CLARIFY, result.kind)
                self.assertIsNone(result.query)
                self.assertTrue(result.clarification)
                self.assertEqual((), result.tasks)

    def test_query_preserves_explicit_clarification_instead_of_reading_placeholder_scope(self) -> None:
        for extra in (
            {"kind": "clarify"},
            {"clarification": "要查哪一天？"},
            {"clarification_reason": "ambiguous_intent"},
        ):
            with self.subTest(extra=extra):
                result = plan_from_mapping(
                    {"kind": "query", "query": {"mode": "today", "keyword": ""}, **extra},
                    forced_kind=IntentKind.QUERY,
                )
                self.assertEqual(IntentKind.CLARIFY, result.kind)
                self.assertIsNone(result.query)
                self.assertTrue(result.clarification)


if __name__ == "__main__":
    unittest.main()
