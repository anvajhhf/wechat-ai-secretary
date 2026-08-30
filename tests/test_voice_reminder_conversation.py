from __future__ import annotations

import re
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from wechat_secretary.config import SecretarySettings
from wechat_secretary.hermes_plugin import _event_to_message
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.media import PreparedMedia
from wechat_secretary.models import (
    ActionResult,
    ExecutionStatus,
    IntentKind,
    IntentPlan,
    MessageEnvelope,
    TaskDraft,
    TaskReference,
)
from wechat_secretary.obsidian import ObsidianExecutor
from wechat_secretary.private_inbox import PrivateInboxExecutor
from wechat_secretary.reminders import ReminderQueue
from wechat_secretary.service import SecretaryService


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = SecretarySettings(
    project_root=ROOT,
    dry_run=True,
    account_id="voice-regression-account",
    allowed_users=frozenset({"voice-regression-user"}),
)
NOW = datetime(2026, 8, 30, 13, 25, tzinfo=SETTINGS.tz)
BODY = "让ChatGPT优化一下本地的生信相关技能，或者看看网上有没有更好的技能"
VAGUE_BODY = (
    "让ChatGPT查一下本地的生信技能有哪些，要不要优化，网上有没有更好用的"
)
WEIXIN_TRANSCRIPT_PREFIX = "[Voice transcription provided by Weixin]\n"


def compact_body(text: str) -> str:
    """Ignore title punctuation, but require every substantive source word."""

    return re.sub(r"[\s，,。；;！？!?：:、]+", "", text)


class RecordingClassifier:
    """Return an intentionally wrong draft; local grounding must recover it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, IntentKind | None]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def classify(
        self,
        incoming: MessageEnvelope,
        content: str,
        forced_kind: IntentKind | None,
        categories: object,
        links: object,
        *,
        deep_note: bool = False,
        image_inputs: object = (),
    ) -> IntentPlan:
        del incoming, categories, links, deep_note, image_inputs
        self.calls.append((content, forced_kind))
        return IntentPlan(
            kind=IntentKind.TASK,
            tasks=(
                TaskDraft(
                    "模型猜错的事项",
                    reminder_at="2026-08-31T09:00+08:00",
                ),
            ),
            confidence=0.99,
        )


class TranscriptMedia:
    """Fake the ASR boundary without reading audio or contacting a service."""

    def __init__(self) -> None:
        self.transcripts: dict[str, str] = {}
        self.calls: list[str] = []

    def prepare(self, incoming: MessageEnvelope) -> PreparedMedia:
        self.calls.append(incoming.message_id)
        return PreparedMedia(transcript_text=self.transcripts[incoming.message_id])


class RecordingDida:
    def __init__(self) -> None:
        self.tasks: list[TaskDraft] = []

    def create_task(
        self, task: TaskDraft, incoming: MessageEnvelope
    ) -> ActionResult:
        del incoming
        self.tasks.append(task)
        task_id = f"voice-task-{len(self.tasks)}"
        return ActionResult(
            action="task",
            status=ExecutionStatus.PLANNED,
            summary=task.title,
            destination="Inbox",
            external_id=task_id,
            task_refs=(TaskReference(task_id, task.title, "Inbox", "inbox"),),
        )


class VoiceReminderConversationTests(unittest.TestCase):
    def make_service(
        self,
    ) -> tuple[
        SecretaryService, RecordingClassifier, TranscriptMedia, RecordingDida, IdempotencyLedger
    ]:
        ledger = IdempotencyLedger(":memory:")
        self.addCleanup(ledger.close)
        classifier = RecordingClassifier()
        media = TranscriptMedia()
        dida = RecordingDida()
        service = SecretaryService(
            settings=SETTINGS,
            ledger=ledger,
            classifier=classifier,
            media=media,
            dida=dida,
            obsidian=ObsidianExecutor(SETTINGS),
            private_inbox=PrivateInboxExecutor(SETTINGS),
            reminders=ReminderQueue(SETTINGS, ledger),
        )
        return service, classifier, media, dida, ledger

    @staticmethod
    def message(
        message_id: str,
        text: str,
        media: TranscriptMedia,
        *,
        voice: bool = True,
        minutes_later: int = 0,
    ) -> MessageEnvelope:
        if voice:
            media.transcripts[message_id] = text
        return MessageEnvelope(
            platform="weixin",
            account_id=SETTINGS.account_id,
            user_id="voice-regression-user",
            chat_id="voice-regression-chat",
            chat_type="dm",
            message_id=message_id,
            text="" if voice else text,
            received_at=NOW + timedelta(minutes=minutes_later),
            media_paths=(f"fake-audio/{message_id}.silk",) if voice else (),
            media_types=("audio/silk",) if voice else (),
        )

    @staticmethod
    def weixin_transcript_event(
        message_id: str,
        transcript: str,
        *,
        message_type: str = "text",
        minutes_later: int = 0,
    ) -> SimpleNamespace:
        # _collect_media adds audio/* only after a successful download. Thus a
        # real Weixin transcript-only fallback is currently classified TEXT.
        return SimpleNamespace(
            source=SimpleNamespace(
                platform="weixin",
                user_id="voice-regression-user",
                chat_id="voice-regression-chat",
                chat_type="dm",
            ),
            message_id=message_id,
            message_type=SimpleNamespace(value=message_type),
            text=WEIXIN_TRANSCRIPT_PREFIX + transcript,
            timestamp=NOW + timedelta(minutes=minutes_later),
            media_urls=[],
            media_types=[],
            raw_message={
                "item_list": [{"type": 3, "voice_item": {"text": transcript}}]
            },
        )

    def assert_one_reminder(
        self,
        dida: RecordingDida,
        ledger: IdempotencyLedger,
        incoming: MessageEnvelope,
        expected_body: str,
        hour: int,
    ) -> None:
        self.assertEqual(1, len(dida.tasks))
        task = dida.tasks[0]
        self.assertEqual(compact_body(expected_body), compact_body(task.title))
        self.assertEqual(f"2026-08-30T{hour:02d}:00+08:00", task.reminder_at)
        self.assertEqual("", task.due_date)
        self.assertEqual("", task.due_time)
        self.assertEqual(1, ledger.active_reminder_count("voice-task-1", incoming))
        self.assertEqual(
            "pending",
            ledger.reminder_status(
                "voice-task-1", NOW.replace(hour=hour, minute=0)
            ),
        )

    def test_exact_afternoon_clock_and_embedded_question_create_reminder(self) -> None:
        for voice in (False, True):
            for clock, hour in (("三", 15), ("3", 15), ("四", 16), ("4", 16)):
                with self.subTest(voice=voice, clock=clock):
                    service, classifier, media, dida, ledger = self.make_service()
                    incoming = self.message(
                        f"exact-{voice}-{hour}-{clock}",
                        f"今天下午{clock}点的时候，提醒我{BODY}。",
                        media,
                        voice=voice,
                    )

                    result = service.handle(incoming)

                    self.assertEqual(ExecutionStatus.PLANNED, result.status)
                    self.assertEqual(["task", "reminder"], [r.action for r in result.results])
                    self.assertEqual([], classifier.calls)
                    self.assertNotIn("还差具体几点", result.reply)
                    self.assert_one_reminder(dida, ledger, incoming, BODY, hour)

    def test_vague_voice_clock_keeps_body_and_only_asks_for_schedule(self) -> None:
        service, classifier, media, dida, ledger = self.make_service()
        source = self.message(
            "vague-clock",
            f"4点多的时候，记得提醒我{VAGUE_BODY}。",
            media,
        )

        first = service.handle(source)

        self.assertEqual(ExecutionStatus.SKIPPED, first.status)
        self.assertEqual((), first.results)
        self.assertEqual([], dida.tasks)
        self.assertRegex(first.reply, r"几点|具体.*时间|精确.*时间|准确.*时间")
        self.assertNotRegex(first.reply, r"做什么|补充.*事项|创建或记录")
        claim = ledger.claim_pending_task(
            source.conversation_key, "vague-probe", NOW + timedelta(seconds=10)
        )
        self.assertIsNotNone(claim.pending)
        self.assertEqual(compact_body(VAGUE_BODY), compact_body(claim.pending.task.title))
        self.assertEqual("", claim.pending.reminder_time)
        self.assertTrue(ledger.release_pending_task(source.conversation_key, "vague-probe"))
        first_model_calls = len(classifier.calls)

        followup = self.message(
            "vague-clarified", "今天下午四点。", media, minutes_later=1
        )
        completed = service.handle(followup)
        duplicate = service.handle(followup)

        self.assertEqual(ExecutionStatus.PLANNED, completed.status)
        self.assertFalse(completed.llm_called)
        self.assertEqual(first_model_calls, len(classifier.calls))
        self.assertNotRegex(completed.reply, r"请补充|做什么|还差")
        self.assert_one_reminder(dida, ledger, followup, VAGUE_BODY, 16)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(2, len(media.calls))

    def test_spoken_date_clock_followup_resumes_without_second_model_call(self) -> None:
        for followup_text, hour in (
            ("今天下午三点", 15),
            ("今天下午三点。", 15),
            ("今天下午四点", 16),
            ("今天下午四点。", 16),
            ("今天下午4点。", 16),
        ):
            with self.subTest(followup=followup_text):
                service, classifier, media, dida, ledger = self.make_service()
                source = self.message("missing-clock", f"今天提醒我{BODY}。", media)
                first = service.handle(source)
                self.assertEqual(ExecutionStatus.SKIPPED, first.status)
                self.assertEqual([], dida.tasks)
                first_model_calls = len(classifier.calls)

                followup = self.message(
                    "complete-clock", followup_text, media, minutes_later=1
                )
                completed = service.handle(followup)

                self.assertEqual(ExecutionStatus.PLANNED, completed.status)
                self.assertFalse(completed.llm_called)
                self.assertEqual(first_model_calls, len(classifier.calls))
                self.assert_one_reminder(dida, ledger, followup, BODY, hour)

    def test_unrelated_spoken_time_question_does_not_consume_pending(self) -> None:
        service, classifier, media, dida, ledger = self.make_service()
        source = self.message("pending-source", f"今天提醒我{BODY}。", media)
        service.handle(source)

        unrelated = service.handle(
            self.message(
                "unrelated-question", "你下午四点有空吗？", media, minutes_later=1
            )
        )

        self.assertEqual(ExecutionStatus.SKIPPED, unrelated.status)
        self.assertEqual((), unrelated.results)
        self.assertEqual([], dida.tasks)
        claim = ledger.claim_pending_task(
            source.conversation_key, "question-probe", NOW + timedelta(minutes=2)
        )
        self.assertIsNotNone(claim.pending)
        self.assertEqual(compact_body(BODY), compact_body(claim.pending.task.title))
        self.assertEqual(source.message_id, claim.pending.source_message_id)
        self.assertTrue(ledger.release_pending_task(source.conversation_key, "question-probe"))
        model_calls_before_followup = len(classifier.calls)

        followup = self.message(
            "actual-followup", "今天下午四点。", media, minutes_later=3
        )
        completed = service.handle(followup)

        self.assertEqual(ExecutionStatus.PLANNED, completed.status)
        self.assertFalse(completed.llm_called)
        self.assertEqual(model_calls_before_followup, len(classifier.calls))
        self.assert_one_reminder(dida, ledger, followup, BODY, 16)

    def test_duplicate_voice_message_does_not_transcribe_or_create_again(self) -> None:
        service, classifier, media, dida, ledger = self.make_service()
        incoming = self.message(
            "duplicate-exact", f"今天下午四点的时候提醒我{BODY}。", media
        )

        first = service.handle(incoming)
        duplicate = service.handle(incoming)

        self.assertEqual(ExecutionStatus.PLANNED, first.status)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(ExecutionStatus.SKIPPED, duplicate.status)
        self.assertEqual(0, len(classifier.calls))
        self.assertEqual([incoming.message_id], media.calls)
        self.assert_one_reminder(dida, ledger, incoming, BODY, 16)

    def test_verified_weixin_transcription_wrapper_is_transport_metadata(self) -> None:
        for message_type in ("text", "voice"):
            with self.subTest(message_type=message_type):
                transcript = f"今天下午四点的时候提醒我{BODY}。"
                event = self.weixin_transcript_event(
                    "weixin-full-reminder", transcript, message_type=message_type
                )

                incoming = _event_to_message(event, SETTINGS)
                service, classifier, media, dida, ledger = self.make_service()
                result = service.handle(incoming)

                self.assertEqual(transcript, incoming.text)
                self.assertEqual((), incoming.media_paths)
                self.assertEqual([], media.calls)
                self.assertEqual(ExecutionStatus.PLANNED, result.status)
                self.assertEqual([], classifier.calls)
                self.assert_one_reminder(dida, ledger, incoming, BODY, 16)

    def test_verified_weixin_text_fallback_resumes_without_model_or_asr(self) -> None:
        for message_type in ("text", "voice"):
            with self.subTest(message_type=message_type):
                service, classifier, media, dida, ledger = self.make_service()
                source = self.message("raw-voice-source", f"今天提醒我{BODY}。", media)
                first = service.handle(source)
                self.assertEqual(ExecutionStatus.SKIPPED, first.status)
                first_model_calls = classifier.call_count

                event = self.weixin_transcript_event(
                    "transcript-only-followup",
                    "今天下午四点。",
                    message_type=message_type,
                    minutes_later=1,
                )
                followup = _event_to_message(event, SETTINGS)
                completed = service.handle(followup)
                duplicate = service.handle(followup)

                self.assertEqual(ExecutionStatus.PLANNED, completed.status)
                self.assertFalse(completed.llm_called)
                self.assertEqual(first_model_calls, classifier.call_count)
                self.assertEqual([source.message_id], media.calls)
                self.assert_one_reminder(dida, ledger, followup, BODY, 16)
                self.assertTrue(duplicate.duplicate)

    def test_unverified_weixin_transcription_wrapper_is_not_removed(self) -> None:
        transcript = "今天下午四点。"
        wrapped = WEIXIN_TRANSCRIPT_PREFIX + transcript
        cases = {
            "no-raw-message": None,
            "typed-text": {
                "item_list": [{"type": 1, "text_item": {"text": wrapped}}]
            },
            "typed-text-with-voice": {
                "item_list": [
                    {"type": 1, "text_item": {"text": wrapped}},
                    {"type": 3, "voice_item": {"text": transcript}},
                ]
            },
            "voice-with-raw-audio": {
                "item_list": [
                    {
                        "type": 3,
                        "voice_item": {
                            "text": transcript,
                            "media": {"encrypt_query_param": "fake-local-fixture"},
                        },
                    }
                ]
            },
            "different-transcript": {
                "item_list": [{"type": 3, "voice_item": {"text": "无关语音"}}]
            },
            "missing-transcript": {"item_list": [{"type": 3, "voice_item": {}}]},
        }
        for name, raw_message in cases.items():
            with self.subTest(case=name):
                event = self.weixin_transcript_event(name, transcript)
                event.raw_message = raw_message
                self.assertEqual(wrapped, _event_to_message(event, SETTINGS).text)

        event = self.weixin_transcript_event("batched-transcript", transcript)
        event.text += "\n这是另一条普通文字，不应被当作同一条语音"
        self.assertEqual(event.text, _event_to_message(event, SETTINGS).text)

        other_platform = self.weixin_transcript_event("other-platform", transcript)
        other_platform.source.platform = "telegram"
        self.assertEqual(wrapped, _event_to_message(other_platform, SETTINGS).text)


if __name__ == "__main__":
    unittest.main()
