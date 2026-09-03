from __future__ import annotations

import unittest

from wechat_secretary.models import IntentKind
from wechat_secretary.request_scope import (
    has_multiple_reminder_requests,
    has_negated_reminder,
    mask_quoted_text,
    note_request_match,
    note_source_body,
    outer_reminder_match,
    reminder_marker,
)
from wechat_secretary.routing import detect_route_hint, is_non_action_task_utterance


class NaturalRequestScopeTests(unittest.TestCase):
    def test_future_confirmation_question_is_not_a_reminder_status_query(self) -> None:
        from test_voice_reminder_conversation import VoiceReminderConversationTests

        fixture = VoiceReminderConversationTests()
        self.addCleanup(fixture.doCleanups)
        service, classifier, media, dida, _ = fixture.make_service()
        text = "今天下午三点提醒我确认材料交了吗？"
        result = service.handle(fixture.message("confirm-question", text, media, voice=False))
        self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
        self.assertFalse(is_non_action_task_utterance(text))
        self.assertEqual(1, len(dida.tasks), result.reply)
        self.assertEqual("确认材料交了吗", dida.tasks[0].title)
        self.assertIn("T15:00", dida.tasks[0].reminder_at)
        self.assertEqual(0, classifier.call_count)
        for status in (
            "今天下午三点提醒我确认了吗？",
            "今天下午三点提醒成功了吗？",
            "今天下午三点提醒我交报告成功了吗？",
        ):
            with self.subTest(status=status):
                self.assertIsNone(detect_route_hint(status).kind)

    def test_chinese_relative_requests_use_shared_schedule_tokens(self) -> None:
        for schedule in ("二十分钟后", "两个小时后", "半个小时后", "一小时三十分钟后", "大约二十分钟后"):
            text = f"{schedule}提醒一下我查一下任务有哪些"
            with self.subTest(text=text):
                self.assertIsNotNone(outer_reminder_match(text))
                self.assertFalse(is_non_action_task_utterance(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)

    def test_compact_day_period_reminders_route_without_weakening_boundaries(self) -> None:
        for schedule in ("今早八点", "今晨八点", "今晚七点半", "今夜十一点", "明早八点", "明晨八点", "明晚七点半", "明夜十一点"):
            text = f"{schedule}提醒我看申请书和文献"
            with self.subTest(text=text):
                self.assertIsNotNone(outer_reminder_match(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
        for text in (
            "她说今晚七点半提醒我看文献",
            "不要今晚七点半提醒我看文献",
            "今晚七点半提醒我看文献了吗？",
            "“今晚七点半提醒我看文献”",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_route_hint(text).kind)

    def test_front_loaded_action_is_owned_by_the_tail_reminder(self) -> None:
        for text in (
            "买牛奶，明天下午四点提醒我",
            "买牛奶，二十分钟后提醒我",
            "查一下任务有哪些，明天下午四点提醒一下我",
            "取消会议提醒，明天下午四点提醒我",
            "问导师材料交了吗，明天下午四点提醒我",
            "提交报告，麻烦你明天下午四点提醒我，好吗？",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(outer_reminder_match(text))
                self.assertFalse(is_non_action_task_utterance(text))
                self.assertFalse(has_negated_reminder(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)

    def test_relative_and_front_loaded_forms_preserve_non_action_boundaries(self) -> None:
        for text in (
            "他说二十分钟后提醒我喝水",
            "转发：二十分钟后提醒我喝水",
            "“二十分钟后提醒我喝水”",
            "二十分钟后提醒我喝水了吗？",
            "不要二十分钟后提醒我喝水",
            "他说买牛奶，明天下午四点提醒我",
            "转发：买牛奶，明天下午四点提醒我",
            "“买牛奶”，明天下午四点提醒我",
            "买牛奶，明天下午四点提醒我了吗？",
            "买牛奶这件事怎么删除，明天下午四点提醒我",
            "买过牛奶，明天下午四点提醒我",
            "买牛奶，明天下午四点不用提醒我",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_route_hint(text).kind)
                if reminder_marker(text) is not None:
                    self.assertTrue(is_non_action_task_utterance(text))

    def test_multiple_reminders_require_independent_unquoted_request_clauses(self) -> None:
        for text in (
            "今天下午三点提醒我买牛奶，下午四点提醒我交报告",
            "今天下午三点提醒我买牛奶。同时明天下午四点通知我交报告",
            "今天下午三点提醒我买牛奶；明天下午四点叫我提交报告",
            "买牛奶，今天下午三点提醒我\n交报告，明天下午四点提醒我",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_multiple_reminder_requests(text))
        for text in (
            "今天下午三点提醒我问小王为什么下午四点提醒我交报告",
            "今天下午三点提醒我记录“下午四点提醒我交报告”这句话",
            "今天下午三点提醒我买牛奶，导师下午四点提醒我交报告",
            "今天下午三点提醒我买牛奶，下午四点提醒我交报告了吗？",
            "今天下午三点提醒我买牛奶，下午四点不要提醒我交报告",
            "帮我记下来：今天下午三点提醒我买牛奶，下午四点提醒我交报告",
            "转发：今天下午三点提醒我买牛奶，导师下午四点提醒我交报告",
            "《今天下午三点提醒我买牛奶，下午四点提醒我交报告》",
        ):
            with self.subTest(text=text):
                self.assertFalse(has_multiple_reminder_requests(text))

    def test_reminder_synonyms_preserve_the_marker_and_body(self) -> None:
        for marker in ("提醒我", "提醒一下我", "提醒我一下", "通知我", "通知一下我", "叫我", "叫一下我"):
            text = f"麻烦你明天下午三点{marker}查一下本地生信技能有哪些"
            with self.subTest(marker=marker):
                match = reminder_marker(text)
                self.assertIsNotNone(match)
                self.assertEqual(marker, text[match.start():match.end()])
                self.assertEqual("查一下本地生信技能有哪些", text[match.end():])
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
                self.assertFalse(is_non_action_task_utterance(text))

    def test_future_cancellation_and_nested_negation_are_task_content(self) -> None:
        for text in (
            "明天下午三点提醒一下我取消今天的会议提醒",
            "明天下午三点提醒我删除手机里的提醒",
            "明天下午三点通知我不要忘记买抗体",
            "明天下午三点叫我问导师为什么不用提醒我开会",
            "明天下午三点提醒我联系导师，告诉他不用提醒我开会",
            "明天下午三点提醒我问导师材料交了吗？",
        ):
            with self.subTest(text=text):
                self.assertFalse(has_negated_reminder(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)

    def test_real_cancellation_still_blocks_task_creation(self) -> None:
        for text in (
            "不要明天下午三点提醒一下我买牛奶",
            "取消明天下午三点通知一下我买牛奶",
            "不用再叫我买牛奶",
            "明天下午三点提醒我买牛奶，算了，不用通知一下我了",
            "明天下午三点提醒我买牛奶，取消这个提醒",
            "明天下午三点提醒我，不用提醒我了",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_negated_reminder(text))
                self.assertIsNone(detect_route_hint(text).kind)

    def test_note_request_owns_lookup_reminder_and_negated_content(self) -> None:
        for lead in ("记下来", "帮我记下", "请帮我记录一下", "记个笔记", "把这个记下来"):
            for body in ("查一下任务有哪些", "不要提醒我买牛奶", "明天三点提醒我买牛奶"):
                text = f"{lead}：{body}"
                with self.subTest(text=text):
                    self.assertIsNotNone(note_request_match(text))
                    self.assertEqual(body, note_source_body(text))
                    self.assertIs(detect_route_hint(text).kind, IntentKind.NOTE)
                    self.assertFalse(has_negated_reminder(text))
                    self.assertIsNone(reminder_marker(text))

    def test_note_quoted_payload_is_preserved_not_executed(self) -> None:
        text = "帮我记下来：“明天三点提醒我取消会议提醒”"
        self.assertEqual("“明天三点提醒我取消会议提醒”", note_source_body(text))
        self.assertIs(detect_route_hint(text).kind, IntentKind.NOTE)
        self.assertIsNone(reminder_marker(text))
        self.assertEqual(len(text), len(mask_quoted_text(text)))
        self.assertEqual("会议结论：A方案可行", note_source_body("会议结论：A方案可行"))
        self.assertEqual("会议结论：A方案可行", note_source_body("记下来：会议结论：A方案可行"))

    def test_complex_weekdays_still_reach_semantic_clarification(self) -> None:
        for prefix in ("每周一到五", "每周二和四", "每周二、周四", "每个工作日", "工作日", "隔周"):
            with self.subTest(prefix=prefix):
                self.assertIs(detect_route_hint(f"{prefix}上午九点提醒我买牛奶").kind, IntentKind.TASK)

    def test_independent_mixed_clause_is_distinct_from_note_body(self) -> None:
        text = "帮我记一下查任务的方法，另外明天下午三点提醒一下我买牛奶"
        self.assertIs(detect_route_hint(text).kind, IntentKind.MIXED)
        match = reminder_marker(text)
        self.assertIsNotNone(match)
        self.assertEqual("提醒一下我", text[match.start():match.end()])
        self.assertEqual("买牛奶", text[match.end():])
        self.assertIs(
            detect_route_hint("帮我记一下查任务的方法，另外导师明天提醒我买牛奶").kind,
            IntentKind.NOTE,
        )

    def test_status_forwarding_and_quotes_never_become_creation_requests(self) -> None:
        for text in (
            "转发：明天下午三点提醒一下我买牛奶",
            "他说：明天下午三点通知一下我买牛奶",
            "导师明天下午三点叫我提交报告",
            "系统明天下午三点会通知我交报告",
            "为什么明天下午三点通知一下我买牛奶？",
            "明天下午三点提醒一下我查天气了吗？",
            "明天下午三点叫我买牛奶这件事怎么取消？",
            "《通知一下我》",
            "“明天下午三点提醒一下我买牛奶”",
            "“同事说”明天下午三点提醒我买牛奶",
            "记下来了吗？",
            "记录一下是什么意思？",
            "叫我小王",
            "请叫我小王",
            "麻烦你叫我王老师",
        ):
            with self.subTest(text=text):
                self.assertNotIn(detect_route_hint(text).kind, (IntentKind.TASK, IntentKind.NOTE, IntentKind.MIXED))

    def test_polite_questions_are_requests_but_status_questions_are_not(self) -> None:
        for text in (
            "你能明天下午三点提醒一下我买牛奶吗？",
            "麻烦你明天下午三点通知一下我买牛奶，好吗？",
            "明天下午三点叫我买牛奶，可以吗？",
        ):
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
                self.assertIsNotNone(outer_reminder_match(text))
        self.assertIsNone(detect_route_hint("你会在明天下午三点通知一下我买牛奶吗？").kind)


if __name__ == "__main__":
    unittest.main()
