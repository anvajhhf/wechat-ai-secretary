from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta

import test_voice_reminder_conversation as fixtures
from wechat_secretary.models import ExecutionStatus, IntentKind, IntentPlan, TaskDraft
from wechat_secretary.semantic_guard import extract_task_semantics, resume_pending_task, validate_plan_semantics


NOW = datetime(2026, 8, 30, 15, 29, tzinfo=fixtures.SETTINGS.tz)
EMPTY = IntentPlan(IntentKind.TASK)


class ColloquialScheduleTests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service
    message = staticmethod(fixtures.VoiceReminderConversationTests.message)

    def handle(self, service, media, text, index=0, *, voice=False):
        message = replace(self.message(f'colloquial-{index}', text, media, voice=voice), received_at=NOW + timedelta(seconds=index))
        return service.handle(message)

    def test_screenshot_first_utterance_creates_today_1630_without_followup(self):
        for voice in (False, True):
            with self.subTest(voice=voice):
                service, classifier, media, dida, ledger = self.make_service()
                result = self.handle(service, media, '提醒我4点半看一下微信助手，下午4点半。', voice=voice)
                self.assertEqual(ExecutionStatus.PLANNED, result.status, result.reply)
                self.assertEqual(0, classifier.call_count)
                self.assertEqual(1, len(dida.tasks))
                self.assertEqual('看一下微信助手', dida.tasks[0].title)
                self.assertEqual('2026-08-30T16:30+08:00', dida.tasks[0].reminder_at)

    def test_sentence_punctuation_does_not_change_complete_request(self):
        for separator in ('，', ',', '。', '.', '；', ';', '、', '\n'):
            service, classifier, media, dida, _ = self.make_service()
            result = self.handle(service, media, f'今天下午4点半{separator}提醒我看一下微信助手。')
            self.assertEqual(ExecutionStatus.PLANNED, result.status, (separator, result.reply))
            self.assertEqual('2026-08-30T16:30+08:00', dida.tasks[0].reminder_at)
            self.assertEqual('看一下微信助手', dida.tasks[0].title)
            self.assertEqual(0, classifier.call_count)

    def test_tail_only_schedule_and_front_loaded_body_keep_clean_title(self):
        for text in (
            '提醒我看一下微信助手，今天下午4点半。',
            '提醒我看一下微信助手。今天下午4点半。',
            '看一下微信助手。今天下午4点半提醒我。',
            '今天下午4点半提醒我看一下微信助手，下午4点半。',
        ):
            with self.subTest(text=text):
                service, _, media, dida, _ = self.make_service()
                result = self.handle(service, media, text)
                self.assertEqual(ExecutionStatus.PLANNED, result.status, result.reply)
                self.assertEqual('看一下微信助手', dida.tasks[0].title)
                self.assertEqual('2026-08-30T16:30+08:00', dida.tasks[0].reminder_at)

    def test_bare_clock_does_not_become_explicit_early_morning(self):
        service, classifier, media, dida, _ = self.make_service()
        for index, text in enumerate(('提醒我4点半看一下微信助手', '今天')):
            result = self.handle(service, media, text, index)
            self.assertEqual([], dida.tasks)
            self.assertNotIn('已经过去', result.reply)
            self.assertIn('上午', result.reply)
        result = self.handle(service, media, '下午', 2)
        self.assertEqual(ExecutionStatus.PLANNED, result.status, result.reply)
        self.assertEqual(0, classifier.call_count)
        self.assertEqual('2026-08-30T16:30+08:00', dida.tasks[0].reminder_at)

    def test_today_default_only_for_future_unambiguous_clock(self):
        for text in ('下午4点半提醒我看微信', '16:30提醒我看微信'):
            decision = validate_plan_semantics(text, EMPTY, NOW, expected_kind=IntentKind.TASK)
            self.assertTrue(decision.ready, decision.question)
            self.assertEqual('2026-08-30T16:30+08:00', decision.plan.tasks[0].reminder_at)
        for text in ('下午三点提醒我看微信', '明天下午三点或四点提醒我看微信'):
            decision = validate_plan_semantics(text, EMPTY, NOW, expected_kind=IntentKind.TASK)
            self.assertFalse(decision.ready)

    def test_equivalent_repetition_refines_but_conflicting_times_do_not_write(self):
        for text in ('今天今天下午4点半提醒我看微信，下午4点半', '今天16:30提醒我看微信，下午四点半'):
            decision = validate_plan_semantics(text, EMPTY, NOW, expected_kind=IntentKind.TASK)
            self.assertTrue(decision.ready, decision.question)
            self.assertEqual('2026-08-30T16:30+08:00', decision.plan.tasks[0].reminder_at)
        for text in ('今天下午4点半提醒我看微信，下午五点', '今天下午四点提醒我看微信，明天下午四点'):
            self.assertFalse(validate_plan_semantics(text, EMPTY, NOW, expected_kind=IntentKind.TASK).ready)

    def test_body_times_and_quoted_times_are_not_schedule_corrections(self):
        for body in ('查看明天下午四点的会议安排', '阅读《下午三点》', '告诉导师明天下午三点开会', '看报告，里面写着明天下午三点', '发消息，内容是“明天下午三点”'):
            text = f'今天下午4点半提醒我{body}'
            decision = validate_plan_semantics(text, EMPTY, NOW, expected_kind=IntentKind.TASK)
            self.assertTrue(decision.ready, decision.question)
            self.assertEqual('2026-08-30T16:30+08:00', decision.plan.tasks[0].reminder_at)
            self.assertEqual(fixtures.compact_body(body), fixtures.compact_body(decision.plan.tasks[0].title))

    def test_period_only_tail_refines_existing_clock(self):
        decision = validate_plan_semantics('今天4点半提醒我看微信，下午', EMPTY, NOW, expected_kind=IntentKind.TASK)
        self.assertTrue(decision.ready, decision.question)
        self.assertEqual('2026-08-30T16:30+08:00', decision.plan.tasks[0].reminder_at)

    def test_missing_body_today_default_stays_anchored_across_midnight(self):
        first = validate_plan_semantics('晚上十一点提醒我', EMPTY, NOW, expected_kind=IntentKind.TASK)
        self.assertEqual('2026-08-30', first.pending.reminder_date)
        result = resume_pending_task(first.pending, '喝水', NOW + timedelta(days=1))
        self.assertFalse(result.ready)
        self.assertEqual('2026-08-30', result.pending.reminder_date)

    def test_model_cannot_query_without_an_outer_query_request(self):
        from wechat_secretary.models import TaskQuery
        plan = IntentPlan(IntentKind.QUERY, query=TaskQuery())
        decision = validate_plan_semantics('你好', plan, NOW, expected_kind=None)
        self.assertFalse(decision.ready)


if __name__ == '__main__':
    unittest.main()
