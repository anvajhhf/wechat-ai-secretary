from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import test_voice_reminder_conversation as fixtures
from test_voice_reminder_conversation import SETTINGS, VoiceReminderConversationTests

from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.models import (
    IntentKind,
    IntentPlan,
    ReminderRecurrence,
    TaskDraft,
    TaskReference,
)
from wechat_secretary.reminders import (
    ReminderDeliveryUncertainError,
    ReminderQueue,
    ReminderScheduler,
)
from wechat_secretary.request_scope import is_explicit_reminder_candidate
from wechat_secretary.semantic_guard import validate_plan_semantics
from wechat_secretary.temporal import CLOCK_TOKEN_RE, resolve_time


NOW = datetime(2026, 9, 1, 10, 58, tzinfo=SETTINGS.tz)
TEXT = "每天早上8.30，晚上7点提醒我喝中药"
TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"


class DailyReminderTests(unittest.TestCase):
    def make_service(self):
        case = VoiceReminderConversationTests("run")
        service, classifier, media, dida, ledger = case.make_service()
        self.addCleanup(ledger.close)
        return case, service, classifier, media, dida, ledger

    def typed_message(self, case, media, message_id: str, text: str = TEXT):
        message = case.message(message_id, text, media, voice=False)
        return replace(message, received_at=NOW)

    def test_partner_screenshot_is_a_zero_token_daily_request(self):
        case, service, classifier, media, dida, ledger = self.make_service()
        message = self.typed_message(case, media, "partner-exact")

        result = service.handle(message)

        self.assertFalse(result.llm_called)
        self.assertEqual([], classifier.calls)
        self.assertEqual(1, len(dida.tasks))
        recurrence = dida.tasks[0].reminder_recurrence
        self.assertEqual("daily", recurrence.frequency)
        self.assertEqual(("08:30", "19:00"), recurrence.times)
        self.assertEqual("2026-09-01T19:00+08:00", dida.tasks[0].reminder_at)
        self.assertEqual(2, ledger.active_reminder_count("voice-task-1", message))
        self.assertIn("每天 08:30、19:00", result.reply)

    def test_single_daily_slot_is_supported_for_typed_text(self):
        case, service, classifier, media, dida, ledger = self.make_service()
        message = self.typed_message(
            case, media, "partner-single", "每天早上8.30提醒我喝中药",
        )

        result = service.handle(message)

        self.assertFalse(result.llm_called)
        self.assertEqual([], classifier.calls)
        self.assertEqual(("08:30",), dida.tasks[0].reminder_recurrence.times)
        self.assertEqual(1, ledger.active_reminder_count("voice-task-1", message))

    def test_complex_direct_wording_uses_one_task_only_model_fallback(self):
        case, service, classifier, media, dida, ledger = self.make_service()
        text = "从明天开始，每天早上8.30以及晚上7点，提醒我喝中药"
        message = self.typed_message(case, media, "partner-model-fallback", text)

        result = service.handle(message)

        self.assertTrue(result.llm_called)
        self.assertEqual([(text, IntentKind.TASK)], classifier.calls)
        self.assertEqual(1, len(dida.tasks))
        task = dida.tasks[0]
        self.assertEqual("喝中药", task.title)
        self.assertEqual("daily", task.reminder_recurrence.frequency)
        self.assertEqual(("08:30", "19:00"), task.reminder_recurrence.times)
        self.assertEqual("2026-09-02T08:30+08:00", task.reminder_at)
        self.assertEqual(2, ledger.active_reminder_count("voice-task-1", message))

    def test_model_fallback_does_not_accept_reports_or_status_questions(self):
        for text in (
            "系统会从明天开始每天早上8.30提醒我喝中药",
            "导师说从明天开始每天早上8.30提醒我喝中药",
            "为什么从明天开始每天早上8.30提醒我喝中药？",
            "从明天开始每天早上8.30提醒我喝中药了吗？",
            "“从明天开始每天早上8.30提醒我喝中药”",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_explicit_reminder_candidate(text))

    def test_voice_daily_request_still_requires_confirmation(self):
        case, service, classifier, media, dida, ledger = self.make_service()
        message = case.message("voice-daily", TEXT, media, voice=True)
        message = replace(message, received_at=NOW)

        result = service.handle(message)

        self.assertEqual(0, len(dida.tasks))
        self.assertEqual(0, ledger.active_reminder_count("voice-task-1", message))
        self.assertIn("每天", result.reply)

    def test_dot_clock_does_not_break_date_parsing(self):
        self.assertEqual("08:30", resolve_time("早上8.30"))
        self.assertEqual("08:30", resolve_time("每天8.30提醒我喝中药"))
        self.assertEqual("", resolve_time("2026.9.1"))
        self.assertIsNone(CLOCK_TOKEN_RE.search("2026.09.01"))
        self.assertIsNone(CLOCK_TOKEN_RE.search("买1.50升牛奶"))

    def test_replayed_message_does_not_duplicate_task_or_slots(self):
        case, service, _, media, dida, ledger = self.make_service()
        message = self.typed_message(case, media, "partner-replay")
        service.handle(message)

        replay = service.handle(message)

        self.assertTrue(replay.duplicate)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual(2, ledger.active_reminder_count("voice-task-1", message))

    def test_successful_delivery_rolls_only_that_slot_forward(self):
        case, service, _, media, _, ledger = self.make_service()
        message = self.typed_message(case, media, "partner-roll")
        service.handle(message)
        scheduler = ReminderScheduler(SETTINGS, ledger)
        sent = []

        stats = scheduler.poll_once(
            lambda record, content: sent.append((record, content)) or "outbound-1",
            NOW.replace(hour=19, minute=0),
        )

        self.assertEqual(1, stats.sent)
        self.assertEqual(1, len(sent))
        self.assertEqual(2, ledger.active_reminder_count("voice-task-1", message))
        self.assertEqual(
            "pending",
            ledger.reminder_status(
                "voice-task-1", NOW.replace(hour=19, minute=0) + timedelta(days=1),
            ),
        )

    def test_uncertain_delivery_does_not_stop_future_daily_slot(self):
        case, service, _, media, _, ledger = self.make_service()
        message = self.typed_message(case, media, "partner-uncertain")
        service.handle(message)
        scheduler = ReminderScheduler(SETTINGS, ledger)

        def uncertain(*_args):
            raise ReminderDeliveryUncertainError("delivery-timeout")

        stats = scheduler.poll_once(uncertain, NOW.replace(hour=19, minute=0))

        self.assertEqual(1, stats.uncertain)
        self.assertEqual(2, ledger.active_reminder_count("voice-task-1", message))

    def test_cancel_whole_series_stops_all_slots(self):
        case, service, _, media, _, ledger = self.make_service()
        source = self.typed_message(case, media, "partner-cancel")
        service.handle(source)
        first = self.typed_message(
            case, media, "partner-cancel-ask", "取消刚才那个提醒",
        )
        first = replace(first, received_at=NOW + timedelta(minutes=1))
        question = service.handle(first)
        self.assertIn("整个系列", question.reply)
        confirmation = self.typed_message(
            case, media, "partner-cancel-all", "整个系列",
        )
        confirmation = replace(
            confirmation, received_at=NOW + timedelta(minutes=2),
        )

        service.handle(confirmation)

        self.assertEqual(0, ledger.active_reminder_count("voice-task-1", source))

    def test_cancel_next_daily_occurrence_keeps_each_chain_alive(self):
        case, service, _, media, _, ledger = self.make_service()
        source = self.typed_message(case, media, "partner-cancel-next")
        service.handle(source)
        ask = self.typed_message(
            case, media, "partner-cancel-next-ask", "取消刚才那个提醒",
        )
        ask = replace(ask, received_at=NOW + timedelta(minutes=1))
        service.handle(ask)
        confirm = self.typed_message(
            case, media, "partner-cancel-next-confirm", "本次",
        )
        confirm = replace(confirm, received_at=NOW + timedelta(minutes=2))

        service.handle(confirm)

        self.assertEqual(2, ledger.active_reminder_count("voice-task-1", source))
        self.assertEqual(
            "pending",
            ledger.reminder_status(
                "voice-task-1", NOW.replace(hour=19, minute=0) + timedelta(days=1),
            ),
        )

    def test_daily_slots_survive_ledger_reopen(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_ROOT / f"daily-{uuid4().hex}.sqlite3"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        message = fixtures.VoiceReminderConversationTests.message(
            "persist-daily", TEXT, fixtures.TranscriptMedia(), voice=False,
        )
        message = replace(message, received_at=NOW)
        task = TaskDraft(
            "喝中药",
            reminder_at="2026-09-01T19:00+08:00",
            reminder_recurrence=ReminderRecurrence(
                frequency="daily", times=("08:30", "19:00"),
            ),
        )
        ref = TaskReference("persist-task", "喝中药")
        ledger = IdempotencyLedger(path)
        ReminderQueue(SETTINGS, ledger).schedule(task, ref, message)
        ledger.close()

        reopened = IdempotencyLedger(path)
        try:
            self.assertEqual(2, reopened.active_reminder_count("persist-task", message))
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
