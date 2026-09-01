"""ASR may omit commas: bounded totals must still respect payload boundaries."""
from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch

import test_voice_reminder_conversation as fixtures
from wechat_secretary.models import ClarificationReason, ExecutionStatus, IntentKind, IntentPlan
from wechat_secretary.semantic_guard import extract_task_semantics, resume_pending_task, validate_plan_semantics


NOW = datetime(2026, 8, 31, 19, 11, tzinfo=fixtures.SETTINGS.tz)
FIRST_AT = datetime(2026, 9, 1, 9, tzinfo=fixtures.SETTINGS.tz)
EMPTY = IntentPlan(IntentKind.TASK)
WEEKLY = "每周二上午九点提醒我"


class FrozenLedgerDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


def guard(text):
    return validate_plan_semantics(text, EMPTY, NOW, expected_kind=IntentKind.TASK)


class AsrPunctuationTests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service

    def setUp(self):
        self.enterContext(patch("wechat_secretary.ledger.datetime", FrozenLedgerDateTime))

    def incoming(self, text, media, *, voice=False, message_id="punctuation", seconds=0):
        return replace(
            fixtures.VoiceReminderConversationTests.message(message_id, text, media, voice=voice),
            received_at=NOW + timedelta(seconds=seconds),
        )

    def test_attached_weekly_total_variants_keep_clean_body(self):
        for tail in ("共三次", "总共三次", "一共三次", "共3次", "共三次吧", "总共三次吧。"):
            with self.subTest(tail=tail):
                result = guard(f"{WEEKLY}采购抗体{tail}")
                self.assertTrue(result.ready, result.question)
                task = result.plan.tasks[0]
                self.assertEqual("采购抗体", task.title)
                self.assertEqual(FIRST_AT, datetime.fromisoformat(task.reminder_at))
                self.assertEqual(3, task.reminder_recurrence.count)
                self.assertEqual(2, task.reminder_recurrence.weekday)

    def test_attached_total_uses_outer_schedule_in_other_positions(self):
        for text in (
            "提醒我每周二上午九点采购抗体共三次",
            "提醒我采购抗体共三次，每周二上午九点",
            "提醒我采购抗体共三次。每周二上午九点。",
        ):
            with self.subTest(text=text):
                result = guard(text)
                self.assertTrue(result.ready, result.question)
                self.assertEqual("采购抗体", result.plan.tasks[0].title)
                self.assertEqual(3, result.plan.tasks[0].reminder_recurrence.count)

    def test_delimited_and_attached_totals_produce_the_same_task(self):
        expected = guard(f"{WEEKLY}采购抗体，共三次").plan.tasks[0]
        for separator in ("", "，", ",", "；", ";", "。", "\n"):
            with self.subTest(separator=separator):
                result = guard(f"{WEEKLY}采购抗体{separator}共三次")
                self.assertTrue(result.ready, result.question)
                self.assertEqual(expected, result.plan.tasks[0])

    def test_weekly_total_and_title_use_the_same_extraction_boundary(self):
        body = "查看‘共三次’的实验记录"
        text = f"{WEEKLY}{body}共四次"
        signals = extract_task_semantics(text, NOW)
        result = guard(text)
        self.assertEqual(4, signals.repeat_count)
        self.assertTrue(result.ready, result.question)
        self.assertEqual(body, result.plan.tasks[0].title)
        self.assertEqual(4, result.plan.tasks[0].reminder_recurrence.count)

    def test_one_off_counting_body_is_never_converted_to_a_series(self):
        for body in ("统计共三次报价", "统计实验结果共三次", "采购抗体共三次"):
            with self.subTest(body=body):
                result = guard(f"明天下午两点提醒我{body}")
                self.assertTrue(result.ready, result.question)
                task = result.plan.tasks[0]
                self.assertEqual(body, task.title)
                self.assertIsNone(task.reminder_recurrence)
                self.assertEqual("2026-09-01T14:00+08:00", task.reminder_at)

    def test_weekly_body_count_followed_by_prose_is_not_a_total(self):
        for body in (
            "统计共三次报价", "采购抗体共三次后再询价",
            "检查共三次实验失败的记录", "采购抗体总共三次的报价单",
        ):
            with self.subTest(body=body):
                text = f"{WEEKLY}{body}"
                self.assertIsNone(extract_task_semantics(text, NOW).repeat_count)
                result = guard(text)
                self.assertFalse(result.ready)
                self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
                self.assertEqual(body, result.pending.task.title)

    def test_counting_activities_with_a_final_count_remain_ambiguous(self):
        for body in ("统计共三次", "统计实验次数共三次", "累计实验总共三次", "计算报价一共三次"):
            with self.subTest(body=body):
                result = guard(f"{WEEKLY}{body}")
                self.assertFalse(result.ready)
                self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
                self.assertEqual(body, result.pending.task.title)

    def test_polite_prefixes_do_not_bypass_counting_activity_protection(self):
        for body in ("请统计共三次", "帮我统计共三次", "麻烦帮我累计实验共三次", "请你计算报价一共三次"):
            with self.subTest(body=body):
                result = guard(f"{WEEKLY}{body}")
                self.assertFalse(result.ready)
                self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
                self.assertEqual(body, result.pending.task.title)

    def test_counting_words_inside_a_body_do_not_block_a_valid_attached_total(self):
        for body in ("采购统计表", "查看统计结果", "请采购统计表"):
            with self.subTest(body=body):
                result = guard(f"{WEEKLY}{body}共三次")
                self.assertTrue(result.ready, result.question)
                self.assertEqual(body, result.plan.tasks[0].title)
                self.assertEqual(3, result.plan.tasks[0].reminder_recurrence.count)

    def test_quoted_final_count_is_not_an_outer_total(self):
        for body in ("查看“总共三次”", "阅读《共三次》", "查看‘共三次’的记录"):
            with self.subTest(body=body):
                result = guard(f"{WEEKLY}{body}")
                self.assertFalse(result.ready)
                self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
                self.assertEqual(body, result.pending.task.title)

    def test_masking_trailing_quotes_cannot_turn_an_inner_count_into_a_suffix(self):
        for body in (
            "采购抗体共三次“实验记录”",
            "采购抗体总共三次《采购记录》",
            "采购抗体共三次‘新的计划’",
        ):
            with self.subTest(body=body):
                result = guard(f"{WEEKLY}{body}")
                self.assertFalse(result.ready)
                self.assertIs(result.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
                self.assertEqual(body, result.pending.task.title)

    def test_body_weekday_or_quoted_weekday_does_not_authorize_attached_total(self):
        for body in ("查看每周二采购抗体共三次的计划", "查看‘每周二’的采购计划共三次"):
            with self.subTest(body=body):
                text = f"明天下午两点提醒我{body}"
                result = guard(text)
                self.assertTrue(result.ready, result.question)
                self.assertEqual(body, result.plan.tasks[0].title)
                self.assertIsNone(result.plan.tasks[0].reminder_recurrence)

    def test_unsupported_or_multiple_weekdays_do_not_enable_attached_total(self):
        for schedule in ("每天下午两点", "每周二和四上午九点", "每周二上午九点每周四上午九点"):
            with self.subTest(schedule=schedule):
                text = f"{schedule}提醒我采购抗体共三次"
                self.assertIsNone(extract_task_semantics(text, NOW).repeat_count)
                self.assertFalse(guard(text).ready)

    def test_explicit_zero_and_over_limit_attached_totals_are_rejected(self):
        for count in ("0", "零", "53", "五十三", "999"):
            with self.subTest(count=count):
                result = guard(f"{WEEKLY}采购抗体共{count}次")
                self.assertFalse(result.ready)
                self.assertIs(result.reason, ClarificationReason.UNSUPPORTED_RECURRENCE)
                self.assertIn("2到52", result.question)
                self.assertEqual("采购抗体", result.pending.task.title)
                if count in {"0", "零"}:
                    self.assertEqual(-1, result.pending.task.reminder_recurrence.count)

    def test_conflicting_outer_totals_never_choose_the_first_value(self):
        for tail in ("共三次，共四次", "，共三次，共四次", "共0次，共三次", "总共三次，一共四次"):
            with self.subTest(tail=tail):
                text = f"{WEEKLY}采购抗体{tail}"
                signals = extract_task_semantics(text, NOW)
                self.assertTrue(signals.repeat_count_conflict)
                self.assertIsNone(signals.repeat_count)
                result = guard(text)
                self.assertFalse(result.ready)
                self.assertIn("总次数", result.question)
                self.assertIn("不同", result.question)
                self.assertEqual("采购抗体", result.pending.task.title)
                self.assertEqual(0, result.pending.task.reminder_recurrence.count)

    def test_equivalent_outer_totals_are_accepted_and_quoted_counts_stay_in_body(self):
        for body, tail in (
            ("采购抗体", "共三次，共3次"),
            ("查看‘共五次’的记录", "共三次，共3次"),
            ("一次采购抗体", "，共三次，共3次"),
        ):
            with self.subTest(body=body):
                result = guard(f"{WEEKLY}{body}{tail}")
                self.assertTrue(result.ready, result.question)
                self.assertEqual(body.removeprefix("一次"), result.plan.tasks[0].title)
                self.assertEqual(3, result.plan.tasks[0].reminder_recurrence.count)

    def test_pending_count_conflict_drops_old_total_until_one_clear_count_arrives(self):
        first = guard("每周二提醒我采购抗体共三次")
        self.assertFalse(first.ready)
        self.assertEqual(3, first.pending.task.reminder_recurrence.count)
        for correction in ("上午九点，共三次，共四次", "上午九点，三次或四次", "上午九点，三次、四次"):
            with self.subTest(correction=correction):
                conflict = resume_pending_task(first.pending, correction, NOW)
                self.assertFalse(conflict.ready)
                self.assertIn("冲突", conflict.question)
                self.assertEqual(0, conflict.pending.task.reminder_recurrence.count)
                self.assertEqual("09:00", conflict.pending.reminder_time)
                still_missing = resume_pending_task(conflict.pending, "上午十点", NOW)
                self.assertFalse(still_missing.ready)
                self.assertIs(still_missing.reason, ClarificationReason.MISSING_RECURRENCE_COUNT)
                recovered = resume_pending_task(still_missing.pending, "共五次", NOW)
                self.assertTrue(recovered.ready, recovered.question)
                self.assertEqual(5, recovered.plan.tasks[0].reminder_recurrence.count)
                self.assertEqual("2026-09-01T10:00+08:00", recovered.plan.tasks[0].reminder_at)

    def test_relative_time_shortcut_cannot_bypass_a_pending_count_conflict(self):
        first = guard("明天提醒我采购抗体")
        conflict = resume_pending_task(first.pending, "二十分钟后，共三次，共四次", NOW)
        self.assertFalse(conflict.ready)
        self.assertIn("冲突", conflict.question)
        self.assertEqual(0, conflict.pending.task.reminder_recurrence.count)
        self.assertEqual("", conflict.pending.task.reminder_at)

    def test_invalid_number_shapes_do_not_silently_shrink_to_a_usable_integer(self):
        for count in ("3.5", "-1", "三四", "1000"):
            with self.subTest(count=count):
                result = guard(f"{WEEKLY}采购抗体共{count}次")
                self.assertFalse(result.ready)

    def test_ambiguous_and_malformed_chinese_totals_are_not_valid_integers(self):
        for count in ("三四", "两三", "三十十", "十十", "二十零"):
            with self.subTest(count=count):
                text = f"{WEEKLY}采购抗体共{count}次"
                signals = extract_task_semantics(text, NOW)
                self.assertTrue(signals.repeat_count_invalid)
                self.assertIsNone(signals.repeat_count)
                result = guard(text)
                self.assertFalse(result.ready)
                self.assertIn("明确整数", result.question)
                self.assertEqual(0, result.pending.task.reminder_recurrence.count)

    def test_normal_chinese_tens_and_units_remain_usable(self):
        for count, expected in (("三十", 30), ("二十三", 23), ("十三", 13), ("两", 2)):
            with self.subTest(count=count):
                result = guard(f"{WEEKLY}采购抗体共{count}次")
                self.assertTrue(result.ready, result.question)
                self.assertEqual(expected, result.plan.tasks[0].reminder_recurrence.count)

    def test_unparseable_pending_totals_invalidate_the_old_count(self):
        first = guard("每周二提醒我采购抗体共三次")
        for text in (
            "上午九点，三四次", "上午九点，共三四次",
            "上午九点，共两三次", "上午九点，共三十十次",
            "上午九点，共三次，共两三次",
        ):
            with self.subTest(text=text):
                unclear = resume_pending_task(first.pending, text, NOW)
                self.assertFalse(unclear.ready)
                self.assertIn("明确整数", unclear.question)
                self.assertEqual(0, unclear.pending.task.reminder_recurrence.count)
                still_missing = resume_pending_task(unclear.pending, "上午十点", NOW)
                self.assertFalse(still_missing.ready)
                self.assertEqual(0, still_missing.pending.task.reminder_recurrence.count)
                recovered = resume_pending_task(still_missing.pending, "共二十三次", NOW)
                self.assertTrue(recovered.ready, recovered.question)
                self.assertEqual(23, recovered.plan.tasks[0].reminder_recurrence.count)

    def test_unclear_count_like_words_inside_a_body_are_not_schedule_evidence(self):
        for body in ("查看三四次报价的记录", "查看‘共三十十次’的转写结果"):
            with self.subTest(body=body):
                text = f"明天下午两点提醒我{body}"
                self.assertFalse(extract_task_semantics(text, NOW).repeat_count_invalid)
                result = guard(text)
                self.assertTrue(result.ready, result.question)
                self.assertEqual(body, result.plan.tasks[0].title)
                self.assertIsNone(result.plan.tasks[0].reminder_recurrence)

    def test_a_total_never_supplies_the_missing_task_body(self):
        result = guard(f"{WEEKLY}共三次")
        self.assertFalse(result.ready)
        self.assertIs(result.reason, ClarificationReason.MISSING_TASK_BODY)
        self.assertEqual("", result.pending.task.title)
        self.assertEqual(3, result.pending.task.reminder_recurrence.count)

    def test_text_and_fake_voice_create_exactly_one_task_three_reminders_zero_model_calls(self):
        for voice in (False, True):
            with self.subTest(voice=voice):
                service, classifier, media, dida, ledger = self.make_service()
                incoming = self.incoming(f"{WEEKLY}采购抗体共三次", media, voice=voice)
                result = service.handle(incoming)
                self.assertIs(result.status, ExecutionStatus.PLANNED, result.reply)
                self.assertFalse(result.llm_called)
                self.assertEqual(0, classifier.call_count)
                self.assertEqual(1, len(dida.tasks))
                self.assertEqual("采购抗体", dida.tasks[0].title)
                self.assertEqual(3, ledger.active_reminder_count("voice-task-1", incoming))
                for index in range(3):
                    self.assertEqual("pending", ledger.reminder_status("voice-task-1", FIRST_AT + timedelta(weeks=index)))
                self.assertIsNone(ledger.peek_pending_task(incoming.conversation_key, NOW))

    def test_duplicate_punctuationless_request_does_not_create_another_series(self):
        service, classifier, media, dida, ledger = self.make_service()
        incoming = self.incoming(f"{WEEKLY}采购抗体共三次", media, voice=True)
        service.handle(incoming)
        repeated = service.handle(incoming)
        self.assertTrue(repeated.duplicate)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual(3, ledger.active_reminder_count("voice-task-1", incoming))
        self.assertEqual(0, classifier.call_count)

    def test_zero_and_over_limit_service_requests_make_no_external_writes(self):
        for count in ("0", "零", "53"):
            with self.subTest(count=count):
                service, classifier, media, dida, ledger = self.make_service()
                incoming = self.incoming(f"{WEEKLY}采购抗体共{count}次", media, voice=True)
                result = service.handle(incoming)
                self.assertIs(result.status, ExecutionStatus.SKIPPED)
                self.assertIn("2到52", result.reply)
                self.assertEqual([], dida.tasks)
                self.assertEqual(0, classifier.call_count)
                self.assertEqual(0, ledger.active_reminder_count("voice-task-1", incoming))

    def test_service_persists_count_conflict_without_old_count_revival(self):
        service, classifier, media, dida, ledger = self.make_service()
        sequence = (
            "每周二提醒我采购抗体共三次",
            "上午九点，三次或四次",
            "上午十点",
            "共五次",
        )
        for index, text in enumerate(sequence):
            incoming = self.incoming(text, media, message_id=f"count-conflict-{index}", seconds=index)
            result = service.handle(incoming)
            if index < 3:
                self.assertIs(result.status, ExecutionStatus.SKIPPED, result.reply)
                self.assertEqual([], dida.tasks)
                if index > 0:
                    pending = ledger.peek_pending_task(incoming.conversation_key, incoming.received_at)
                    self.assertEqual(0, pending.task.reminder_recurrence.count)
            else:
                self.assertIs(result.status, ExecutionStatus.PLANNED, result.reply)
                self.assertEqual(1, len(dida.tasks))
                self.assertEqual("采购抗体", dida.tasks[0].title)
                self.assertEqual(5, ledger.active_reminder_count("voice-task-1", incoming))
        self.assertEqual(0, classifier.call_count)

    def test_service_persists_an_unparseable_total_as_missing_not_as_old_value(self):
        for count in ("三四", "两三", "三十十"):
            with self.subTest(count=count):
                service, classifier, media, dida, ledger = self.make_service()
                sequence = (
                    "每周二提醒我采购抗体共三次",
                    f"上午九点，共{count}次",
                    "上午十点",
                    "共五次",
                )
                for index, text in enumerate(sequence):
                    incoming = self.incoming(text, media, message_id=f"unclear-count-{index}", seconds=index)
                    result = service.handle(incoming)
                    if index < 3:
                        self.assertIs(result.status, ExecutionStatus.SKIPPED, result.reply)
                        self.assertEqual([], dida.tasks)
                        if index > 0:
                            pending = ledger.peek_pending_task(incoming.conversation_key, incoming.received_at)
                            self.assertEqual(0, pending.task.reminder_recurrence.count)
                    else:
                        self.assertIs(result.status, ExecutionStatus.PLANNED, result.reply)
                        self.assertEqual(1, len(dida.tasks))
                        self.assertEqual(5, ledger.active_reminder_count("voice-task-1", incoming))
                self.assertEqual(0, classifier.call_count)


if __name__ == "__main__":
    unittest.main()
