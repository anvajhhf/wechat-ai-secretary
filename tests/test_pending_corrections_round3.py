"""Service-level correction regressions; all executors and storage are local fakes."""
from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch

import test_voice_reminder_conversation as fixtures
from wechat_secretary.media import PreparedImage, PreparedMedia
from wechat_secretary.models import (
    ExecutionStatus,
    IntentKind,
    IntentPlan,
    ReminderRecurrence,
    TaskDraft,
)


NOW = datetime(2026, 8, 31, 19, 11, tzinfo=fixtures.SETTINGS.tz)
BODY = "分选试剂和询价"
DAILY_REQUEST = f"每天下午两点，提醒我{BODY}。"
ONE_OFF_AT = "2026-09-01T14:00+08:00"


class FrozenLedgerDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


class HallucinatedRecurrenceClassifier:
    """A model mistake must not gain authority through a clarification draft."""

    def __init__(self):
        self.call_count = 0

    def classify(self, *args, **kwargs):
        self.call_count += 1
        return IntentPlan(
            kind=IntentKind.TASK,
            tasks=(TaskDraft(
                "买抗体",
                reminder_at="2026-09-02T09:00+08:00",
                reminder_recurrence=ReminderRecurrence(
                    frequency="weekly", weekday=2, count=3,
                ),
            ),),
            confidence=0.99,
        )


class PendingCorrectionsRound3Tests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service

    def setUp(self):
        # Ledger maintenance also reads wall time when writing a different
        # conversation. Keep that clock aligned with the synthetic receipts.
        self.enterContext(patch("wechat_secretary.ledger.datetime", FrozenLedgerDateTime))

    @staticmethod
    def incoming(message_id, text, media, *, seconds=0, chat_id=None, voice=False):
        message = fixtures.VoiceReminderConversationTests.message(
            message_id, text, media, voice=voice,
        )
        changes = {"received_at": NOW + timedelta(seconds=seconds)}
        if chat_id is not None:
            changes["chat_id"] = chat_id
        return replace(message, **changes)

    def start_daily(self, *, seconds=0):
        service, classifier, media, dida, ledger = self.make_service()
        # Spoken ``明天`` can still be transcribed as ``每天``.  Keep the
        # correction-safety fixture on the voice path, while typed daily
        # requests are now deliberately supported without this confirmation.
        source = self.incoming(
            "daily-source", DAILY_REQUEST, media, seconds=seconds, voice=True,
        )
        handled = service.handle(source)
        self.assertEqual(ExecutionStatus.SKIPPED, handled.status)
        pending = ledger.peek_pending_task(source.conversation_key, source.received_at)
        self.assertIsNotNone(pending)
        self.assertEqual(BODY, pending.task.title)
        self.assertEqual("unsupported", pending.task.reminder_recurrence.frequency)
        self.assertEqual("14:00", pending.reminder_time)
        self.assertEqual([], dida.tasks)
        self.assertEqual(0, classifier.call_count)
        return service, classifier, media, dida, ledger, source

    def assert_single(self, dida, ledger, incoming, *, title=BODY, at=ONE_OFF_AT):
        self.assertEqual(1, len(dida.tasks))
        task = dida.tasks[0]
        self.assertEqual(fixtures.compact_body(title), fixtures.compact_body(task.title))
        self.assertEqual(at, task.reminder_at)
        self.assertIsNone(task.reminder_recurrence)
        self.assertEqual(1, ledger.active_reminder_count("voice-task-1", incoming))
        self.assertIsNone(ledger.peek_pending_task(incoming.conversation_key, incoming.received_at))

    def test_ten_explicit_single_corrections_keep_body_and_clock_without_model(self):
        for text in (
            "不是每天，是明天",
            "明天，不是每天",
            "就明天一次",
            "明天提醒我一次",
            "只在明天提醒一次",
            "只提醒一次，明天下午两点",
            "改成一次性，明天下午两点",
            "改成单次，明天下午两点",
            "明天下午两点，只提醒一次",
            "不是每天，而是明天下午两点",
        ):
            with self.subTest(text=text):
                service, classifier, media, dida, ledger, _ = self.start_daily()
                incoming = self.incoming("single-correction", text, media, seconds=1)
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.PLANNED, result.status)
                self.assertFalse(result.llm_called)
                self.assert_single(dida, ledger, incoming)
                self.assertEqual(0, classifier.call_count)

    def test_single_correction_without_date_asks_only_for_missing_date(self):
        service, classifier, media, dida, ledger, _ = self.start_daily()
        single = self.incoming("single-only", "只提醒一次", media, seconds=1)
        first = service.handle(single)
        self.assertEqual(ExecutionStatus.SKIPPED, first.status)
        pending = ledger.peek_pending_task(single.conversation_key, single.received_at)
        self.assertIsNotNone(pending)
        self.assertEqual(BODY, pending.task.title)
        self.assertEqual("14:00", pending.reminder_time)
        self.assertIsNone(pending.task.reminder_recurrence)
        self.assertEqual([], dida.tasks)
        self.assertNotIn("每周", first.reply)
        day = self.incoming("single-date", "明天", media, seconds=2)
        result = service.handle(day)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assert_single(dida, ledger, day)
        self.assertEqual(0, classifier.call_count)

    def test_date_alone_does_not_silently_convert_a_real_daily_request(self):
        for text in ("明天", "明天下午两点"):
            with self.subTest(text=text):
                service, classifier, media, dida, ledger, _ = self.start_daily()
                incoming = self.incoming("date-only", text, media, seconds=1)
                handled = service.handle(incoming)
                self.assertEqual(ExecutionStatus.SKIPPED, handled.status)
                pending = ledger.peek_pending_task(incoming.conversation_key, incoming.received_at)
                self.assertEqual(BODY, pending.task.title)
                self.assertIsNotNone(pending.task.reminder_recurrence)
                self.assertEqual("unsupported", pending.task.reminder_recurrence.frequency)
                self.assertEqual([], dida.tasks)
                self.assertEqual(0, classifier.call_count)

    def test_single_correction_accepts_a_specific_weekday_date(self):
        for text in ("改成一次性，下周二下午两点", "下周二提醒我一次"):
            with self.subTest(text=text):
                service, classifier, media, dida, ledger, _ = self.start_daily()
                incoming = self.incoming("single-next-week", text, media, seconds=1)
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.PLANNED, result.status)
                self.assert_single(dida, ledger, incoming, at="2026-09-08T14:00+08:00")
                self.assertEqual(0, classifier.call_count)

    def test_replacement_still_requesting_multiple_or_zero_times_is_not_one_off(self):
        for text in (
            "不是每天，是明天三次",
            "不是每天，是明天共三次",
            "不是每天，是明天共0次",
            "明天共三次，不是每天",
        ):
            with self.subTest(text=text):
                service, _, media, dida, ledger, source = self.start_daily()
                incoming = self.incoming("not-one-off", text, media, seconds=1)
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                self.assertEqual([], dida.tasks)
                pending = ledger.peek_pending_task(source.conversation_key, incoming.received_at)
                self.assertIsNotNone(pending)
                self.assertEqual(BODY, pending.task.title)
                self.assertIsNotNone(pending.task.reminder_recurrence)

    def test_cancellation_in_a_correction_is_never_converted_to_creation(self):
        for text in (
            "不是每天，是取消",
            "不是每天，是不用了",
            "不是每天，是不设置了",
            "取消，不是每天",
        ):
            with self.subTest(text=text):
                service, _, media, dida, ledger, _ = self.start_daily()
                day = self.incoming("known-date", "明天", media, seconds=1)
                service.handle(day)
                pending = ledger.peek_pending_task(day.conversation_key, day.received_at)
                self.assertEqual("2026-09-01", pending.reminder_date)
                incoming = self.incoming("negative-correction", text, media, seconds=2)
                result = service.handle(incoming)
                self.assertNotEqual(ExecutionStatus.PLANNED, result.status)
                self.assertEqual([], dida.tasks)

    def test_full_new_request_replaces_old_draft_instead_of_inheriting_it(self):
        service, classifier, media, dida, ledger, _ = self.start_daily()
        incoming = self.incoming(
            "independent-new", "明天下午两点，提醒我分选试剂盒询价。", media, seconds=1,
        )
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assert_single(dida, ledger, incoming, title="分选试剂盒询价")
        self.assertEqual(0, classifier.call_count)

    def test_correction_words_inside_new_task_body_do_not_target_old_draft(self):
        for body in (
            "检查“不是每天，是明天”的转写结果",
            "检查只提醒一次的测试记录",
        ):
            with self.subTest(body=body):
                service, classifier, media, dida, ledger, _ = self.start_daily()
                incoming = self.incoming(
                    "new-body", f"明天下午三点提醒我{body}", media, seconds=1,
                )
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.PLANNED, result.status)
                self.assert_single(dida, ledger, incoming, title=body, at="2026-09-01T15:00+08:00")
                self.assertEqual(0, classifier.call_count)

    def test_quoted_reported_and_question_corrections_never_execute_draft(self):
        for text in (
            "“不是每天，是明天”",
            "他说不是每天，是明天",
            "示例：就明天一次",
            "是不是只提醒一次？",
            "不要改成一次性，明天下午两点",
        ):
            with self.subTest(text=text):
                service, _, media, dida, ledger, source = self.start_daily()
                original = ledger.peek_pending_task(source.conversation_key, source.received_at)
                incoming = self.incoming("untrusted-correction", text, media, seconds=1)
                result = service.handle(incoming)
                self.assertNotEqual(ExecutionStatus.PLANNED, result.status)
                self.assertEqual([], dida.tasks)
                self.assertEqual(original, ledger.peek_pending_task(source.conversation_key, incoming.received_at))

    def test_unsupported_frequency_reply_does_not_invent_a_count(self):
        service, _, media, _, _, _ = self.start_daily()
        # A separate message ID exercises the actual initial response again.
        result = service.handle(self.incoming(
            "daily-again", DAILY_REQUEST, media, seconds=1, voice=True,
        ))
        self.assertNotIn("事项和次数", result.reply)
        self.assertNotIn("次数已保留", result.reply)
        self.assertIn("每天", result.reply)

    def test_correction_after_cancellation_does_not_resurrect_draft(self):
        service, _, media, dida, ledger, _ = self.start_daily()
        cancel = self.incoming("cancel-draft", "取消", media, seconds=1)
        service.handle(cancel)
        self.assertIsNone(ledger.peek_pending_task(cancel.conversation_key, cancel.received_at))
        incoming = self.incoming("after-cancel", "明天提醒我一次", media, seconds=2)
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual([], dida.tasks)
        pending = ledger.peek_pending_task(incoming.conversation_key, incoming.received_at)
        self.assertTrue(pending is None or not pending.task.title)

    def test_expired_correction_cannot_reuse_the_old_body(self):
        service, _, media, dida, ledger, _ = self.start_daily()
        incoming = self.incoming(
            "expired-correction", "明天提醒我一次", media,
            seconds=service.settings.task_clarification_ttl_seconds + 1,
        )
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual([], dida.tasks)
        pending = ledger.peek_pending_task(incoming.conversation_key, incoming.received_at)
        self.assertTrue(pending is None or not pending.task.title)

    def test_correction_in_another_conversation_cannot_reuse_the_draft(self):
        service, _, media, dida, ledger, source = self.start_daily()
        original = ledger.peek_pending_task(source.conversation_key, source.received_at)
        incoming = self.incoming(
            "other-chat-correction", "明天提醒我一次", media, seconds=1,
            chat_id="another-conversation",
        )
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual([], dida.tasks)
        self.assertEqual(original, ledger.peek_pending_task(source.conversation_key, incoming.received_at))

    def test_out_of_order_correction_keeps_newer_draft_unchanged(self):
        service, classifier, media, dida, ledger, source = self.start_daily(seconds=30)
        original = ledger.peek_pending_task(source.conversation_key, source.received_at)
        incoming = self.incoming("stale-correction", "明天提醒我一次", media, seconds=10)
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertIn("早于", result.reply)
        self.assertEqual([], dida.tasks)
        self.assertEqual(original, ledger.peek_pending_task(source.conversation_key, source.received_at))
        self.assertEqual(0, classifier.call_count)

    def test_claimed_and_uncertain_drafts_are_not_reexecuted_by_corrections(self):
        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain):
                service, classifier, media, dida, ledger, source = self.start_daily()
                claimed = ledger.claim_pending_task(source.conversation_key, "in-flight", source.received_at)
                self.assertIsNotNone(claimed.pending)
                if uncertain:
                    ledger.mark_pending_task_uncertain(source.conversation_key, "in-flight")
                incoming = self.incoming("claimed-correction", "明天提醒我一次", media, seconds=1)
                result = service.handle(incoming)
                expected = ExecutionStatus.UNCERTAIN if uncertain else ExecutionStatus.SKIPPED
                self.assertEqual(expected, result.status)
                self.assertEqual([], dida.tasks)
                self.assertEqual(0, classifier.call_count)

    def test_duplicate_correction_creates_exactly_one_task_and_reminder(self):
        service, classifier, media, dida, ledger, _ = self.start_daily()
        incoming = self.incoming("duplicate-correction", "明天提醒我一次", media, seconds=1)
        first = service.handle(incoming)
        second = service.handle(incoming)
        self.assertEqual(ExecutionStatus.PLANNED, first.status)
        self.assertTrue(second.duplicate)
        self.assert_single(dida, ledger, incoming)
        self.assertEqual(0, classifier.call_count)

    def test_model_recurrence_is_removed_before_incomplete_draft_is_persisted(self):
        for text, followup in (
            ("明天提醒我买抗体，分类科研", "下午两点"),
            ("下午两点提醒我买抗体，分类科研", "明天"),
            ("明天25点提醒我买抗体，分类科研", "下午两点"),
            ("明天下午两点，后天下午两点提醒我买抗体，分类科研", "明天"),
        ):
            with self.subTest(text=text):
                service, _, media, dida, ledger = self.make_service()
                classifier = HallucinatedRecurrenceClassifier()
                service.classifier = classifier
                source = self.incoming("model-incomplete", text, media)
                first = service.handle(source)
                self.assertEqual(ExecutionStatus.SKIPPED, first.status)
                pending = ledger.peek_pending_task(source.conversation_key, source.received_at)
                self.assertIsNotNone(pending)
                self.assertIsNone(pending.task.reminder_recurrence)
                self.assertEqual("", pending.task.reminder_at)
                incoming = self.incoming("model-followup", followup, media, seconds=1)
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.PLANNED, result.status)
                self.assertFalse(result.llm_called)
                self.assertEqual(1, len(dida.tasks))
                self.assertEqual(ONE_OFF_AT, dida.tasks[0].reminder_at)
                self.assertIsNone(dida.tasks[0].reminder_recurrence)
                self.assertEqual(1, ledger.active_reminder_count("voice-task-1", incoming))
                self.assertEqual(1, classifier.call_count)

    def test_counts_inside_body_or_quotes_do_not_create_recurring_reminders(self):
        for body in (
            "检查“总共三次”实验的试剂盒",
            "检查连续三次实验失败的记录",
            "检查“提醒我三次”的转写结果",
        ):
            with self.subTest(body=body):
                service, classifier, media, dida, ledger = self.make_service()
                incoming = self.incoming("body-count", f"明天下午两点提醒我{body}", media)
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.PLANNED, result.status)
                self.assert_single(dida, ledger, incoming, title=body)
                self.assertEqual(0, classifier.call_count)

    def test_voice_clarification_echoes_only_the_actual_transcript_without_model(self):
        service, classifier, media, dida, _ = self.make_service()
        incoming = self.incoming("voice-daily", DAILY_REQUEST, media, voice=True)
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(f"我听到的是：{DAILY_REQUEST}", result.reply.splitlines()[0])
        self.assertIn("每天", result.reply.splitlines()[1])
        self.assertEqual(0, classifier.call_count)
        self.assertFalse(result.llm_called)
        self.assertEqual([], dida.tasks)

    def test_typed_daily_request_is_created_without_voice_echo(self):
        service, classifier, media, dida, _ = self.make_service()
        result = service.handle(self.incoming("typed-daily", DAILY_REQUEST, media))
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertNotIn("我听到的是：", result.reply)
        self.assertEqual(0, classifier.call_count)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual("daily", dida.tasks[0].reminder_recurrence.frequency)

    def test_voice_echo_is_whitespace_normalized_and_limited_to_80_characters(self):
        service, classifier, media, _, _ = self.make_service()
        transcript = "\n每天  下午两点，\t提醒我" + "检查试剂盒询价记录" * 14 + "。\n"
        incoming = self.incoming("long-voice", transcript, media, voice=True)
        result = service.handle(incoming)
        normalized = " ".join(transcript.split())
        expected = f"我听到的是：{normalized[:80]}…"
        self.assertEqual(expected, result.reply.splitlines()[0])
        self.assertEqual(0, classifier.call_count)
        self.assertFalse(result.llm_called)

    def test_mixed_typed_caption_and_voice_clarification_has_no_voice_echo(self):
        service, _, media, dida, _ = self.make_service()
        incoming = replace(
            self.incoming("typed-plus-voice", DAILY_REQUEST, media, voice=True),
            text="待办：这段是单独输入的文字",
        )
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertNotIn("我听到的是：", result.reply)
        self.assertEqual([], dida.tasks)

    def test_image_and_multi_attachment_clarifications_have_no_voice_echo(self):
        for with_image in (False, True):
            with self.subTest(with_image=with_image):
                service, _, media, dida, _ = self.make_service()
                incoming = self.incoming("image-or-multiple", DAILY_REQUEST, media, voice=True)
                images = ()
                if with_image:
                    images = (PreparedImage(b"fake-not-decoded", "image/png", "fake.png", "fake-hash"),)
                else:
                    incoming = replace(
                        incoming,
                        media_paths=("fake-audio/a.silk", "fake-audio/b.silk"),
                        media_types=("audio/silk", "audio/silk"),
                    )
                media.prepare = lambda _: PreparedMedia(transcript_text=DAILY_REQUEST, images=images)
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                self.assertNotIn("我听到的是：", result.reply)
                self.assertEqual([], dida.tasks)

    def test_private_and_spoken_private_never_echo_transcript_or_content(self):
        secret = "仅用于隐私回归的敏感测试串"
        for private_mode in ("typed", "latched", "spoken"):
            with self.subTest(private_mode=private_mode):
                service, classifier, media, dida, _ = self.make_service()
                transcript = f"私密：{secret}" if private_mode == "spoken" else f"每天提醒我{secret}"
                incoming = self.incoming("private-voice", transcript, media, voice=True, seconds=1)
                if private_mode == "typed":
                    incoming = replace(incoming, text=f"私密：{secret}")
                elif private_mode == "latched":
                    service.handle(self.incoming("arm-private", "私密：下一条", media))
                result = service.handle(incoming)
                self.assertNotIn("我听到的是：", result.reply)
                self.assertNotIn(secret, result.reply)
                self.assertEqual(0, classifier.call_count)
                self.assertFalse(result.llm_called)
                self.assertEqual([], dida.tasks)
                if private_mode != "spoken":
                    self.assertEqual([], media.calls)

    def test_any_typed_caption_cannot_override_a_spoken_private_prefix(self):
        secret = "仅用于口述隐私回归的敏感测试串"
        for caption in ("待办：处理一下", "笔记：保存一下", "普通文字说明"):
            with self.subTest(caption=caption):
                service, classifier, media, dida, ledger = self.make_service()
                incoming = replace(
                    self.incoming("caption-private-voice", f"私密：{secret}", media, voice=True),
                    text=caption,
                )
                result = service.handle(incoming)
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                self.assertIn("请先发送", result.reply)
                self.assertNotIn("我听到的是：", result.reply)
                self.assertNotIn(secret, result.reply)
                self.assertEqual(0, classifier.call_count)
                self.assertFalse(result.llm_called)
                self.assertEqual([], dida.tasks)
                self.assertEqual((), result.results)
                self.assertIsNone(ledger.peek_pending_task(incoming.conversation_key, incoming.received_at))

    def test_private_prefix_in_any_voice_part_blocks_the_entire_message(self):
        secret = "仅用于第二段语音隐私回归的敏感测试串"
        for private_index in (0, 1):
            for caption in ("", "待办：处理这些语音"):
                with self.subTest(private_index=private_index, caption=caption):
                    service, classifier, media, dida, ledger = self.make_service()
                    parts = [DAILY_REQUEST, DAILY_REQUEST]
                    parts[private_index] = f"私密：{secret}"
                    combined = "\n\n".join(
                        f"[语音转写 {index}]\n{text}" for index, text in enumerate(parts, 1)
                    )
                    media.prepare = lambda _: PreparedMedia(
                        transcript_text=combined, transcript_parts=tuple(parts),
                    )
                    incoming = replace(
                        self.incoming("multi-private", "", media),
                        text=caption,
                        media_paths=("fake-audio/a.silk", "fake-audio/b.silk"),
                        media_types=("audio/silk", "audio/silk"),
                    )
                    result = service.handle(incoming)
                    self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                    self.assertIn("请先发送", result.reply)
                    self.assertNotIn("我听到的是：", result.reply)
                    self.assertNotIn(secret, result.reply)
                    self.assertEqual(0, classifier.call_count)
                    self.assertFalse(result.llm_called)
                    self.assertEqual([], dida.tasks)
                    self.assertEqual((), result.results)
                    self.assertIsNone(ledger.peek_pending_task(incoming.conversation_key, incoming.received_at))

    def test_a_quote_in_the_second_voice_part_is_not_a_private_prefix(self):
        service, classifier, media, dida, _ = self.make_service()
        parts = ("说明一下操作方式", "文档里面引用了“私密：示例内容”这个说法")
        combined = "\n\n".join(
            f"[语音转写 {index}]\n{text}" for index, text in enumerate(parts, 1)
        )
        media.prepare = lambda _: PreparedMedia(
            transcript_text=combined, transcript_parts=parts,
        )

        def ask_only(message, content, forced_kind, *args, **kwargs):
            classifier.calls.append((content, forced_kind))
            return IntentPlan(kind=IntentKind.CLARIFY, clarification="请说明需要处理什么。")

        classifier.classify = ask_only
        incoming = replace(
            self.incoming("multi-quote", "", media),
            media_paths=("fake-audio/a.silk", "fake-audio/b.silk"),
            media_types=("audio/silk", "audio/silk"),
        )
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertNotIn("请先发送", result.reply)
        self.assertNotIn("我听到的是：", result.reply)
        self.assertEqual(1, classifier.call_count)
        self.assertEqual(combined, classifier.calls[0][0])
        self.assertEqual([], dida.tasks)
        self.assertEqual((), result.results)


if __name__ == "__main__":
    unittest.main()
