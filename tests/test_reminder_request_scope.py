from __future__ import annotations

import unittest

from wechat_secretary.models import IntentKind
from wechat_secretary.routing import detect_route_hint, is_non_action_task_utterance


class ReminderRequestScopeTests(unittest.TestCase):
    def test_screenshot_reminders_own_their_question_word_payload(self) -> None:
        cases = (
            "4点多的时候，记得提醒我让拆GPT，查一下本地的生信技能有哪些，要不要优化，网上有没有更好用的。",
            "今天下午3点的时候，提醒我让ChatGPT优化一下本地的生信相关的技能，或者网上有没有更好的。",
            "今天下午4点的时候提醒我让ChatGPT优化一下本地的生信相关技能，或者看看网上有没有更好的技能。",
            "今天下午四点的时候，提醒我查一下本地的技能有哪些。",
            "明天下午3点提醒我研究为什么实验失败，以及该怎么优化。",
            "明天下午3点提醒我问导师明天开会吗？",
            "明天下午3点提醒我确认任务是否创建成功。",
            "明天下午3点提醒我查一下导师明天提交了哪些报告。",
            "不要忘记明天下午3点提醒我研究为什么实验失败。",
            "每周上午9点提醒我买牛奶，共3次",
            "每周都上午9点提醒我买牛奶，共3次",
            "下午提醒我查一下本地生信技能有哪些",
            "大约下午三点提醒我查一下本地生信技能有哪些",
            "明天下午三点左右的时候提醒我查一下本地生信技能有哪些",
            "明天25:00提醒我查一下本地生信技能有哪些",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
                self.assertFalse(is_non_action_task_utterance(text))

    def test_embedded_task_lookup_does_not_steal_reminder_route(self) -> None:
        for text in (
            "明天下午3点提醒我查一下任务有哪些。",
            "明天下午3点提醒我看看笔记，确认实验记录是否完整。",
            "今天下午四点提醒我列出任务，再比较哪些需要优化。",
        ):
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
        for text in ("查一下任务有哪些", "看看笔记", "任务有哪些"):
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.QUERY)

    def test_outer_questions_and_reported_reminders_do_not_authorize_tasks(self) -> None:
        cases = (
            "为什么明天下午3点提醒我查一下生信技能有哪些？",
            "为什么明天下午3点提醒我查一下任务？",
            "你会在今天下午3点提醒我优化技能吗？",
            "系统会在今天下午3点提醒我优化技能",
            "导师明天下午3点提醒我提交报告",
            "导师会在明天下午3点提醒我提交报告",
            "谢谢你刚才提醒我提交报告",
            "今天下午3点提醒我什么？",
            "今天下午3点提醒我是谁说的？",
            "提醒我这个功能怎么用？",
            "今天下午3点提醒我买牛奶了吗？",
            "今天下午3点提醒我买牛奶吗？",
            "今天下午3点提醒我买牛奶是什么意思？",
            "今天下午三点提醒我查一下天气了吗？",
            "今天下午三点提醒我优化本地技能了吗？",
            "下午三点提醒我买牛奶这件事为什么没成功？",
            "下午三点提醒我买牛奶这件事怎么删除？",
            "我想知道下午三点提醒我买牛奶有没有成功",
            "我想知道下午三点提醒我买牛奶",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(detect_route_hint(text).kind)
                self.assertTrue(is_non_action_task_utterance(text))

    def test_stop_intent_is_not_overridden_by_scoped_payload(self) -> None:
        for text in (
            "不要在今天下午3点提醒我查一下本地的技能有哪些。",
            "不要在今天下午3点提醒我买牛奶。",
            "不用明天下午3点再提醒我查一下任务。",
            "请勿再提醒我买牛奶",
            "取消今天下午3点提醒我查一下生信技能有哪些",
            "今天下午3点提醒我买牛奶，算了，不用提醒我了",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_route_hint(text).kind)
                self.assertTrue(is_non_action_task_utterance(text))

    def test_quoted_examples_and_titles_do_not_become_reminder_commands(self) -> None:
        for text in (
            "《明天3点提醒我买牛奶》",
            "他说“明天3点提醒我买牛奶”",
            '"明天3点提醒我查一下任务有哪些"',
            "明天3点看《提醒我买牛奶》",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_route_hint(text).kind)
        self.assertIs(
            detect_route_hint("明天3点提交《提醒我买牛奶》观后感").kind,
            IntentKind.TASK,
        )
        quoted_stop = detect_route_hint("明天3点提醒我提交《不要提醒我》观后感")
        self.assertIs(quoted_stop.kind, IntentKind.TASK)
        self.assertEqual(("reminder-request",), quoted_stop.evidence)


if __name__ == "__main__":
    unittest.main()
