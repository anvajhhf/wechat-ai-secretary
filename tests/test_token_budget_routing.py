"""Token-saving routing must preserve the same write and schedule boundaries."""
from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import test_voice_reminder_conversation as fixtures
from wechat_secretary.media import PreparedImage, PreparedMedia
from wechat_secretary.models import ActionResult, ExecutionStatus, IntentKind, IntentPlan, TaskDraft, TaskQuery


NOW = fixtures.NOW


class BudgetRecordingClassifier:
    """No model connection: record that richer extraction was actually retained."""

    def __init__(self, plan: IntentPlan) -> None:
        self.plan = plan
        self.calls = []

    @property
    def call_count(self):
        return len(self.calls)

    def classify(self, message, content, forced_kind, categories, links, **kwargs):
        self.calls.append((content, forced_kind, categories, {**kwargs, "links": links}))
        return self.plan


class TokenBudgetRoutingTests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service
    message = staticmethod(fixtures.VoiceReminderConversationTests.message)

    def test_exact_source_grounded_reminders_cost_zero_classifier_calls(self):
        cases = (
            ("今天下午三点提醒我买牛奶", "买牛奶", NOW.replace(hour=15, minute=0), 1),
            ("待办：今天下午三点提醒我买牛奶", "买牛奶", NOW.replace(hour=15, minute=0), 1),
            ("提醒我明天下午四点买牛奶", "买牛奶", (NOW + timedelta(days=1)).replace(hour=16, minute=0), 1),
            ("买牛奶，明天下午四点提醒我", "买牛奶", (NOW + timedelta(days=1)).replace(hour=16, minute=0), 1),
            (f"今天下午三点提醒我{fixtures.BODY}", fixtures.BODY, NOW.replace(hour=15, minute=0), 1),
            ("今天下午三点提醒我问导师材料交了吗？", "问导师材料交了吗", NOW.replace(hour=15, minute=0), 1),
            ("今天下午三点提醒我取消会议提醒", "取消会议提醒", NOW.replace(hour=15, minute=0), 1),
            ("二十分钟后提醒我喝水", "喝水", NOW + timedelta(minutes=20), 1),
            ("每周二上午九点提醒我买抗体，共三次", "买抗体", datetime(2026, 9, 1, 9, tzinfo=NOW.tzinfo), 3),
        )
        for voice in (False, True):
            for text, title, when, count in cases:
                with self.subTest(text=text, voice=voice):
                    service, classifier, media, dida, ledger = self.make_service()
                    incoming = self.message("budget-exact", text, media, voice=voice)
                    result = service.handle(incoming)
                    self.assertEqual(0, classifier.call_count)
                    self.assertFalse(result.llm_called)
                    self.assertEqual(ExecutionStatus.PLANNED, result.status)
                    self.assertEqual(1, len(dida.tasks))
                    task = dida.tasks[0]
                    self.assertEqual(fixtures.compact_body(title), fixtures.compact_body(task.title))
                    self.assertEqual(when, datetime.fromisoformat(task.reminder_at))
                    self.assertEqual("", task.due_date)
                    self.assertEqual("", task.due_time)
                    self.assertEqual(count, ledger.active_reminder_count("voice-task-1", incoming))
                    for index in range(count):
                        self.assertEqual("pending", ledger.reminder_status("voice-task-1", when + timedelta(weeks=index)))

    def test_missing_fields_and_clarification_are_zero_call_end_to_end(self):
        cases = (
            (("明天下午提醒我买牛奶", "三点"), "买牛奶", (NOW + timedelta(days=1)).replace(hour=15, minute=0)),
            (("今天下午三点提醒我", "买牛奶"), "买牛奶", NOW.replace(hour=15, minute=0)),
            (("明天下午提醒我", "买牛奶", "三点"), "买牛奶", (NOW + timedelta(days=1)).replace(hour=15, minute=0)),
            (("4点多提醒我买牛奶", "今天下午四点"), "买牛奶", NOW.replace(hour=16, minute=0)),
            (("今天下午提醒我买牛奶", "不是今天，是明天", "四点"), "买牛奶", (NOW + timedelta(days=1)).replace(hour=16, minute=0)),
        )
        for sequence, title, when in cases:
            with self.subTest(sequence=sequence):
                service, classifier, media, dida, ledger = self.make_service()
                for index, text in enumerate(sequence):
                    incoming = self.message(f"budget-fields-{index}", text, media, minutes_later=index, voice=False)
                    result = service.handle(incoming)
                    self.assertEqual(0, classifier.call_count)
                    self.assertFalse(result.llm_called)
                    if index < len(sequence) - 1:
                        self.assertEqual([], dida.tasks)
                        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                self.assertEqual(ExecutionStatus.PLANNED, result.status)
                self.assertEqual(1, len(dida.tasks))
                self.assertEqual(title, dida.tasks[0].title)
                self.assertEqual(when, datetime.fromisoformat(dida.tasks[0].reminder_at))
                self.assertEqual("pending", ledger.reminder_status("voice-task-1", when))

    def test_update_cancel_and_duplicate_messages_never_reinvoke_classifier(self):
        service, classifier, media, dida, ledger = self.make_service()
        sequence = (
            "今天下午三点提醒我买牛奶",
            "改成明天下午四点多",
            "改成五点",
            "取消刚才那个提醒",
        )
        for index, text in enumerate(sequence):
            incoming = self.message(f"budget-control-{index}", text, media, voice=False)
            result = service.handle(incoming)
            snapshot = ledger.reminder_snapshot("voice-task-1", incoming)
            duplicate = service.handle(incoming)
            self.assertTrue(duplicate.duplicate)
            self.assertEqual(snapshot, ledger.reminder_snapshot("voice-task-1", incoming))
            self.assertFalse(result.llm_called)
            self.assertEqual(0, classifier.call_count)
            self.assertEqual(1, len(dida.tasks))
            if index == 2:
                when = (NOW + timedelta(days=1)).replace(hour=17, minute=0)
                self.assertEqual("pending", ledger.reminder_status("voice-task-1", when))
        self.assertEqual(0, ledger.active_reminder_count("voice-task-1", incoming))
        self.assertEqual(1, len(ledger.recent_task_context(incoming.conversation_key, NOW).candidates))

    def test_duplicate_voice_message_also_avoids_retranscription(self):
        service, classifier, media, dida, ledger = self.make_service()
        incoming = self.message("budget-voice-duplicate", "今天下午三点提醒我买牛奶", media)
        first = service.handle(incoming)
        repeated = service.handle(incoming)
        self.assertFalse(first.llm_called)
        self.assertTrue(repeated.duplicate)
        self.assertEqual(0, classifier.call_count)
        self.assertEqual([incoming.message_id], media.calls)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual(1, ledger.active_reminder_count("voice-task-1", incoming))

    def test_append_count_and_interval_clarification_are_zero_call(self):
        service, classifier, media, dida, ledger = self.make_service()
        for index, text in enumerate(("今天下午三点提醒我买牛奶", "再提醒三次", "每隔二十分钟")):
            incoming = self.message(f"budget-append-{index}", text, media, voice=False)
            result = service.handle(incoming)
            self.assertFalse(result.llm_called)
            self.assertEqual(0, classifier.call_count)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual(4, ledger.active_reminder_count("voice-task-1", incoming))
        for minute in (0, 20, 40, 60):
            when = NOW.replace(hour=15, minute=0) + timedelta(minutes=minute)
            self.assertEqual("pending", ledger.reminder_status("voice-task-1", when))

    def test_missing_recurrence_count_is_local_and_keeps_all_occurrences(self):
        service, classifier, media, dida, ledger = self.make_service()
        initial = self.message("budget-weekly-incomplete", "每周二上午九点提醒我买抗体", media, voice=False)
        first = service.handle(initial)
        self.assertEqual(ExecutionStatus.SKIPPED, first.status)
        self.assertEqual([], dida.tasks)
        self.assertEqual(0, classifier.call_count)
        followup = self.message("budget-weekly-count", "共三次", media, voice=False)
        completed = service.handle(followup)
        self.assertFalse(completed.llm_called)
        self.assertEqual(0, classifier.call_count)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual(3, ledger.active_reminder_count("voice-task-1", followup))
        first_at = datetime(2026, 9, 1, 9, tzinfo=NOW.tzinfo)
        for index in range(3):
            self.assertEqual("pending", ledger.reminder_status("voice-task-1", first_at + timedelta(weeks=index)))

    def test_quoted_or_question_clock_does_not_consume_a_local_draft(self):
        for nonanswer in ("“下午三点”", "下午三点是什么意思？", "为什么要下午三点提醒我？"):
            with self.subTest(nonanswer=nonanswer):
                service, classifier, media, dida, ledger = self.make_service()
                first = self.message("budget-draft", "明天提醒我买牛奶", media, voice=False)
                service.handle(first)
                self.assertEqual(0, classifier.call_count)
                side_message = self.message("budget-draft-question", nonanswer, media, minutes_later=1, voice=False)
                service.handle(side_message)
                self.assertEqual([], dida.tasks)
                self.assertEqual(0, ledger.active_reminder_count("voice-task-1", side_message))
                before = classifier.call_count
                answer = self.message("budget-draft-answer", "下午三点", media, minutes_later=2, voice=False)
                result = service.handle(answer)
                self.assertFalse(result.llm_called)
                self.assertEqual(before, classifier.call_count)
                self.assertEqual(1, len(dida.tasks))
                self.assertEqual("买牛奶", dida.tasks[0].title)
                self.assertEqual("pending", ledger.reminder_status("voice-task-1", (NOW + timedelta(days=1)).replace(hour=15, minute=0)))

    def test_explicit_category_keeps_rich_extraction_and_metadata(self):
        service, _, media, dida, ledger = self.make_service()
        service.settings = replace(service.settings, category_map={"工作": "project-work"})
        classifier = BudgetRecordingClassifier(IntentPlan(
            kind=IntentKind.TASK,
            tasks=(TaskDraft("提交报告", category="工作"),),
        ))
        service.classifier = classifier
        incoming = self.message("budget-category", "明天下午三点提醒我提交报告，分类：工作", media, voice=False)
        result = service.handle(incoming)
        self.assertEqual(1, classifier.call_count)
        self.assertTrue(result.llm_called)
        self.assertIn("工作", classifier.calls[0][2])
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual("工作", dida.tasks[0].category)
        expected = (NOW + timedelta(days=1)).replace(hour=15, minute=0)
        self.assertEqual(expected, datetime.fromisoformat(dida.tasks[0].reminder_at))
        self.assertEqual("pending", ledger.reminder_status("voice-task-1", expected))

    def test_deadlines_tags_and_compound_requests_keep_classifier_path(self):
        for text in (
            "明天下午三点提醒我提交报告，最迟后天完成",
            "明天下午三点提醒我提交报告，截止日期是后天",
            "明天下午三点提醒我提交报告，标签：项目A",
            "明天下午三点提醒我买牛奶；明天下午四点提醒我取快递",
            "明天下午三点提醒我买牛奶，明天下午四点提醒我取快递",
            "明天下午三点提醒我买牛奶。明天下午四点提醒我取快递",
            "明天下午三点提醒我买牛奶，另外明天下午四点提醒我取快递",
            "明天下午三点提醒我买牛奶\n明天下午四点提醒我取快递",
        ):
            with self.subTest(text=text):
                service, classifier, media, _, _ = self.make_service()
                result = service.handle(self.message("budget-rich", text, media, voice=False))
                self.assertEqual(1, classifier.call_count)
                self.assertTrue(result.llm_called)
                self.assertEqual(IntentKind.TASK, classifier.calls[0][1])

    def test_multiple_scheduled_tasks_are_not_collapsed_into_a_single_fast_write(self):
        for separator in ("；", "，", "。"):
            with self.subTest(separator=separator):
                service, _, media, dida, ledger = self.make_service()
                classifier = BudgetRecordingClassifier(IntentPlan(
                    kind=IntentKind.TASK,
                    tasks=(TaskDraft("买牛奶"), TaskDraft("取快递")),
                ))
                service.classifier = classifier
                incoming = self.message("budget-multi", f"明天下午三点提醒我买牛奶{separator}明天下午四点提醒我取快递", media, voice=False)
                result = service.handle(incoming)
                self.assertEqual(1, classifier.call_count)
                self.assertTrue(result.llm_called)
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                self.assertEqual([], dida.tasks)
                self.assertEqual(0, ledger.active_reminder_count("voice-task-1", incoming))

    def test_quoted_inner_reminder_is_only_outer_task_content(self):
        for body in (
            "阅读“明天下午四点提醒我买牛奶”这句话",
            "把“明天下午四点提醒我取快递”写进笔记",
            "阅读《明天下午四点提醒我买牛奶》",
            "转述‘明天下午四点通知我开会’",
        ):
            with self.subTest(body=body):
                service, classifier, media, dida, ledger = self.make_service()
                incoming = self.message("budget-inner-quote", f"今天下午三点提醒我{body}", media, voice=False)
                result = service.handle(incoming)
                self.assertEqual(0, classifier.call_count)
                self.assertFalse(result.llm_called)
                self.assertEqual(["task", "reminder"], [item.action for item in result.results])
                self.assertEqual(1, len(dida.tasks))
                self.assertEqual(body, dida.tasks[0].title)
                self.assertEqual(1, ledger.active_reminder_count("voice-task-1", incoming))
                self.assertEqual("pending", ledger.reminder_status("voice-task-1", NOW.replace(hour=15, minute=0)))
                self.assertIsNone(ledger.reminder_status("voice-task-1", (NOW + timedelta(days=1)).replace(hour=16, minute=0)))

    def test_image_reminder_keeps_image_input_and_classifier(self):
        service, _, media, _, _ = self.make_service()
        prepared = PreparedMedia(images=(PreparedImage(b"fake-image", "image/png", "fixture.png", "fixture-only"),))
        service.media = SimpleNamespace(prepare=lambda incoming: prepared)
        classifier = BudgetRecordingClassifier(IntentPlan(kind=IntentKind.TASK, tasks=(TaskDraft("处理图片里的事项"),)))
        service.classifier = classifier
        incoming = replace(
            self.message("budget-image", "明天下午三点提醒我处理图片里的事项", media, voice=False),
            media_paths=("fake-memory-image.png",), media_types=("image/png",),
        )
        result = service.handle(incoming)
        self.assertEqual(1, classifier.call_count)
        self.assertTrue(result.llm_called)
        self.assertEqual(prepared.image_inputs, classifier.calls[0][3]["image_inputs"])

    def test_task_query_does_not_collect_note_links(self):
        service, _, media, dida, _ = self.make_service()
        service.obsidian = SimpleNamespace(available_links=lambda _: self.fail("查询任务不应扫描笔记双链"))
        queries = []

        def query_tasks(query):
            queries.append(query)
            return ActionResult("query", ExecutionStatus.PLANNED, "仅查询，不创建事项")

        dida.query_tasks = query_tasks
        classifier = BudgetRecordingClassifier(IntentPlan(kind=IntentKind.QUERY, query=TaskQuery("today")))
        service.classifier = classifier
        result = service.handle(self.message("budget-query", "查一下今天有哪些任务", media, voice=False))
        self.assertTrue(result.llm_called)
        self.assertEqual(1, classifier.call_count)
        self.assertEqual(IntentKind.QUERY, classifier.calls[0][1])
        self.assertEqual((), classifier.calls[0][3]["links"])
        self.assertEqual([TaskQuery("today")], queries)
        self.assertEqual([], dida.tasks)

    def test_quoted_reported_question_and_negated_reminders_never_fast_write(self):
        for text in (
            "“明天下午三点提醒我买牛奶”",
            "《明天下午三点提醒我买牛奶》",
            "转发：明天下午三点提醒我买牛奶",
            "他说：明天下午三点提醒我买牛奶",
            "为什么明天下午三点提醒我买牛奶？",
            "明天下午三点提醒我买牛奶了吗？",
            "你会在明天下午三点提醒我买牛奶吗？",
            "不要明天下午三点提醒我买牛奶",
            "明天下午三点提醒我买牛奶，算了，不用提醒我了",
            "笔记：明天下午三点提醒我买牛奶",
        ):
            with self.subTest(text=text):
                service, _, media, dida, ledger = self.make_service()
                incoming = self.message("budget-not-action", text, media, voice=False)
                result = service.handle(incoming)
                self.assertEqual([], dida.tasks)
                self.assertEqual(0, ledger.active_reminder_count("voice-task-1", incoming))
                self.assertFalse(any(item.action == "task" for item in result.results))


if __name__ == "__main__":
    unittest.main()
