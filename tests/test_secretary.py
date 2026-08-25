from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from wechat_secretary.classifier import HeuristicClassifier, HermesStructuredClassifier
from wechat_secretary.cli import (
    _dida_schema_errors,
    command_dry_run,
    command_inspect_dida_task,
    command_verify_dida_complete,
    command_verify_dida_create,
)
from wechat_secretary.config import SecretarySettings
from wechat_secretary.dida import DidaExecutor
from wechat_secretary.hermes_plugin import GatewayBridge, _json_tool_result
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.media import (
    LocalMediaPreprocessor,
    MediaPreparationError,
    PreparedImage,
    PreparedMedia,
)
from wechat_secretary.models import (
    ActionResult,
    ExecutionStatus,
    IntentKind,
    IntentPlan,
    MessageEnvelope,
    NoteDraft,
    TaskDraft,
    TaskQuery,
    TaskReference,
)
from wechat_secretary.obsidian import ObsidianExecutor
from wechat_secretary.private_inbox import PrivateInboxExecutor
from wechat_secretary.reminders import ReminderQueue, ReminderScheduler
from wechat_secretary.replies import add_dry_run_previews, format_results
from wechat_secretary.service import SecretaryService
from tools.mcp_oauth_manager import _normalize_root_issuer_trailing_slash
from tools.mcp_tool import (
    _operator_approved_tools,
    _operator_approved_tools_when_env,
    _record_tool_trust_metadata,
    _server_trust_levels,
    _tool_read_only_hints,
    _trust_gate_check,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.fromisoformat("2026-08-24T09:00:00+08:00")
TEST_TEMP = ROOT / "runtime" / "test-temp"
TEST_TEMP.mkdir(parents=True, exist_ok=True)


def test_directory(label: str) -> Path:
    path = TEST_TEMP / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True)
    return path


def settings(**changes: object) -> SecretarySettings:
    base = SecretarySettings(
        project_root=ROOT,
        dry_run=True,
        allowed_users=frozenset({"wx-user-1"}),
        account_id="dry-account",
        known_links=("年度目标", "AI个人秘书", "产品路线图", "领域A"),
        category_map={"工作": "project-work", "个人": "project-personal"},
        tag_map={"重要": "重要"},
    )
    return replace(base, **changes)


def message(message_id: str, text: str, when: datetime = NOW) -> MessageEnvelope:
    return MessageEnvelope(
        platform="weixin",
        account_id="dry-account",
        user_id="wx-user-1",
        chat_id="chat-1",
        chat_type="dm",
        message_id=message_id,
        text=text,
        received_at=when,
    )


def service_with(
    cfg: SecretarySettings | None = None,
    *,
    classifier: object | None = None,
    ledger: IdempotencyLedger | None = None,
    dida: object | None = None,
    obsidian: object | None = None,
    private_inbox: object | None = None,
    media: object | None = None,
) -> tuple[SecretaryService, object, IdempotencyLedger]:
    cfg = cfg or settings()
    ledger = ledger or IdempotencyLedger(":memory:")
    classifier = classifier or HeuristicClassifier(cfg)
    app = SecretaryService(
        settings=cfg,
        ledger=ledger,
        classifier=classifier,
        dida=dida or DidaExecutor(cfg),
        obsidian=obsidian or ObsidianExecutor(cfg),
        private_inbox=private_inbox or PrivateInboxExecutor(cfg),
        reminders=ReminderQueue(cfg, ledger),
        media=media,
    )
    return app, classifier, ledger


class RoutingTests(unittest.TestCase):
    class RejectingTaskClassifier:
        def __init__(self) -> None:
            self.call_count = 0

        def classify(self, *args: object, **kwargs: object) -> IntentPlan:
            del args, kwargs
            self.call_count += 1
            return IntentPlan(
                kind=IntentKind.CLARIFY,
                confidence=0.1,
                clarification="请补充内容",
            )

    def test_date_time_category_and_priority(self) -> None:
        app, _, _ = service_with()
        result = app.handle(message("m1", "待办：明天下午3点提交报告，高优先级，工作"))
        task = result.results[0]
        payload = json.loads(task.preview)["task"]
        self.assertEqual("2026-08-25T15:00:00+08:00", payload["dueDate"])
        self.assertFalse(payload["isAllDay"])
        self.assertEqual(5, payload["priority"])
        self.assertEqual("project-work", payload["projectId"])

    def test_explicit_plain_task_falls_back_to_inbox(self) -> None:
        classifier = self.RejectingTaskClassifier()
        app, _, _ = service_with(classifier=classifier)

        result = app.handle(
            message("explicit-plain-task", "待办：owner 任务完成验收")
        )

        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual("task", result.results[0].action)
        payload = json.loads(result.results[0].preview)["task"]
        self.assertEqual("owner 任务完成验收", payload["title"])
        self.assertEqual("inbox", payload["projectId"])
        self.assertNotIn("dueDate", payload)
        self.assertNotIn("请再补充", result.reply)
        self.assertEqual(1, classifier.call_count)

    def test_explicit_task_never_drops_unparsed_schedule(self) -> None:
        classifier = self.RejectingTaskClassifier()
        app, _, _ = service_with(classifier=classifier)

        result = app.handle(
            message("explicit-scheduled-task", "待办：明天15:00提交报告")
        )

        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertFalse(result.results)
        self.assertIn("请再补充", result.reply)
        self.assertEqual(1, classifier.call_count)

    def test_fuzzy_period_never_invents_clock_time(self) -> None:
        app, _, _ = service_with()
        result = app.handle(message("m2", "待办：明天下午提交报销单，工作"))
        payload = json.loads(result.results[0].preview)["task"]
        self.assertEqual("2026-08-25T00:00:00+08:00", payload["dueDate"])
        self.assertTrue(payload["isAllDay"])

    def test_note_links_are_existing_and_capped(self) -> None:
        app, _, _ = service_with()
        result = app.handle(
            message("m3", "笔记：AI个人秘书、年度目标、产品路线图和领域A彼此关联")
        )
        preview = result.results[0].preview
        self.assertLessEqual(preview.count("[["), 3)
        self.assertIn("[[AI个人秘书]]", preview)

    def test_explicit_note_falls_back_locally_when_model_returns_no_note(self) -> None:
        class RejectingNoteClassifier:
            def __init__(self) -> None:
                self.call_count = 0

            def classify(self, *args: object, **kwargs: object) -> IntentPlan:
                del args, kwargs
                self.call_count += 1
                return IntentPlan(
                    kind=IntentKind.CLARIFY,
                    confidence=0.1,
                    clarification="请补充内容",
                )

        classifier = RejectingNoteClassifier()
        app, _, _ = service_with(classifier=classifier)
        content = "owner 微信个人秘书已完成正式写入验收，记录于 2026-08-25。"
        result = app.handle(message("explicit-note-fallback", f"笔记：{content}"))

        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual("note", result.results[0].action)
        self.assertIn(content, result.results[0].preview)
        self.assertNotIn("请再补充", result.reply)
        self.assertEqual(1, classifier.call_count)

    def test_explicit_note_accepts_valid_low_confidence_model_note(self) -> None:
        class LowConfidenceNoteClassifier:
            def __init__(self) -> None:
                self.call_count = 0

            def classify(self, *args: object, **kwargs: object) -> IntentPlan:
                del args, kwargs
                self.call_count += 1
                return IntentPlan(
                    kind=IntentKind.NOTE,
                    notes=(NoteDraft("验收记录", "正式写入验收已完成。"),),
                    confidence=0.2,
                )

        classifier = LowConfidenceNoteClassifier()
        app, _, _ = service_with(classifier=classifier)
        result = app.handle(
            message("explicit-note-low-confidence", "笔记：正式写入验收已完成。")
        )

        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual("验收记录", result.results[0].summary)
        self.assertNotIn("请再补充", result.reply)
        self.assertEqual(1, classifier.call_count)

    def test_private_bypasses_classifier_and_does_not_echo(self) -> None:
        app, classifier, _ = service_with()
        secret = "PRIVATE-BODY-NEVER-ECHO"
        result = app.handle(message("m4", f"私密：{secret}"))
        self.assertEqual(0, classifier.call_count)
        self.assertNotIn(secret, result.reply)
        self.assertNotIn(secret, result.results[0].preview)

    def test_private_next_dry_run_is_explicitly_simulated(self) -> None:
        app, classifier, _ = service_with()
        result = app.handle(message("private-next-dry-run", "私密：下一条"))
        self.assertIn("Dry Run", result.reply)
        self.assertIn("只会模拟本地保存", result.reply)
        self.assertIn("不会实际写入", result.reply)
        self.assertEqual(0, classifier.call_count)

    def test_duplicate_message_id_does_not_repeat(self) -> None:
        app, _, _ = service_with()
        original = message("m5", "待办：明天提交报告")
        first = app.handle(original)
        second = app.handle(original)
        self.assertFalse(first.duplicate)
        self.assertTrue(second.duplicate)
        self.assertIn("不会重复", second.reply)

    def test_unknown_category_falls_back_to_inbox(self) -> None:
        app, _, _ = service_with()
        result = app.handle(message("m6", "待办：后天预订酒店，旅行"))
        self.assertEqual("Inbox", result.results[0].destination)

    def test_due_time_and_local_reminder_are_distinct(self) -> None:
        app, _, ledger = service_with()
        result = app.handle(message("m7", "待办：半小时后提醒我提交费用申请"))
        task_payload = json.loads(result.results[0].preview)["task"]
        self.assertNotIn("dueDate", task_payload)
        reminder = result.results[1]
        self.assertEqual("reminder", reminder.action)
        self.assertIn("2026-08-24T09:30:00+08:00", reminder.preview)
        self.assertEqual(
            "pending",
            ledger.reminder_status(
                result.results[0].external_id,
                datetime.fromisoformat("2026-08-24T09:30:00+08:00"),
            ),
        )


class MediaRoutingTests(unittest.TestCase):
    class CapturingClassifier:
        def __init__(self) -> None:
            self.call_count = 0
            self.content = ""
            self.forced_kind = None
            self.deep_note = False
            self.image_inputs: tuple[dict[str, object], ...] = ()

        def classify(
            self,
            incoming: MessageEnvelope,
            content: str,
            forced_kind: IntentKind | None,
            categories: object,
            links: object,
            *,
            deep_note: bool = False,
            image_inputs: tuple[dict[str, object], ...] = (),
        ) -> IntentPlan:
            del incoming, categories, links
            self.call_count += 1
            self.content = content
            self.forced_kind = forced_kind
            self.deep_note = deep_note
            self.image_inputs = tuple(image_inputs)
            if forced_kind is IntentKind.TASK:
                return IntentPlan(
                    kind=IntentKind.TASK,
                    tasks=(TaskDraft("媒体待办"),),
                    confidence=0.9,
                )
            return IntentPlan(
                kind=IntentKind.NOTE,
                notes=(NoteDraft("媒体笔记", content or "图片内容"),),
                confidence=0.9,
            )

    class StaticMedia:
        def __init__(self, prepared: PreparedMedia | None = None, error: str = "") -> None:
            self.prepared = prepared or PreparedMedia()
            self.error = error
            self.calls = 0

        def prepare(self, incoming: MessageEnvelope) -> PreparedMedia:
            del incoming
            self.calls += 1
            if self.error:
                raise MediaPreparationError(self.error)
            return self.prepared

    @staticmethod
    def media_message(message_id: str, text: str = "") -> MessageEnvelope:
        return replace(
            message(message_id, text),
            media_paths=("runtime/hermes-home/cache/images/fake.jpg",),
            media_types=("image/jpeg",),
        )

    def test_private_media_bypasses_every_preprocessor(self) -> None:
        media = self.StaticMedia(error="不应调用")
        app, classifier, _ = service_with(media=media)
        result = app.handle(self.media_message("private-media", "私密：附件"))
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(0, media.calls)
        self.assertEqual(0, classifier.call_count)

    def test_voice_transcript_can_force_task_without_second_model_call(self) -> None:
        classifier = self.CapturingClassifier()
        media = self.StaticMedia(PreparedMedia(transcript_text="待办：明天提交报告"))
        app, _, _ = service_with(classifier=classifier, media=media)
        incoming = replace(
            message("voice-task", ""),
            media_paths=("runtime/hermes-home/cache/audio/fake.silk",),
            media_types=("audio/silk",),
        )
        result = app.handle(incoming)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(IntentKind.TASK, classifier.forced_kind)
        self.assertEqual("明天提交报告", classifier.content)

    def test_spoken_private_prefix_never_reaches_deepseek(self) -> None:
        classifier = self.CapturingClassifier()
        media = self.StaticMedia(PreparedMedia(transcript_text="私密：不要上传"))
        app, _, _ = service_with(classifier=classifier, media=media)
        incoming = replace(
            message("spoken-private", ""),
            media_paths=("runtime/hermes-home/cache/audio/fake.silk",),
            media_types=("audio/silk",),
        )
        result = app.handle(incoming)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(0, classifier.call_count)
        self.assertNotIn("不要上传", result.reply)

    def test_image_is_forwarded_only_as_prepared_bytes(self) -> None:
        classifier = self.CapturingClassifier()
        image = PreparedImage(b"normalized", "image/jpeg", "wechat-image-1.jpg", "a" * 64)
        media = self.StaticMedia(PreparedMedia(images=(image,)))
        app, _, _ = service_with(classifier=classifier, media=media)
        result = app.handle(self.media_message("image-note"))
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(1, len(classifier.image_inputs))
        self.assertEqual(b"normalized", classifier.image_inputs[0]["data"])

    def test_media_failure_stops_before_classifier(self) -> None:
        classifier = self.CapturingClassifier()
        media = self.StaticMedia(error="图片无法安全解码。")
        app, _, _ = service_with(classifier=classifier, media=media)
        result = app.handle(self.media_message("bad-image"))
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertEqual(0, classifier.call_count)
        self.assertIn("图片无法安全解码", result.reply)

    def test_image_normalization_strips_metadata_and_resizes(self) -> None:
        from PIL import Image

        cache = test_directory("image-cache")
        source = cache / "incoming.jpg"
        picture = Image.new("RGB", (3000, 1200), "white")
        exif = Image.Exif()
        exif[0x010E] = "PRIVATE-META"
        picture.save(source, format="JPEG", exif=exif)
        cfg = settings(
            vision_enabled=True,
            media_cache_roots=(cache,),
            image_max_dimension=1024,
        )
        prepared = LocalMediaPreprocessor(cfg).prepare(
            replace(
                message("normalized-image", "笔记：截图"),
                media_paths=(str(source),),
                media_types=("image/jpeg",),
            )
        )
        self.assertEqual(1, len(prepared.images))
        self.assertNotIn(b"PRIVATE-META", prepared.images[0].data)
        with Image.open(io.BytesIO(prepared.images[0].data)) as normalized:
            self.assertLessEqual(max(normalized.size), 1024)

    def test_silk_transcription_uses_accessible_wav_and_cleans_it(self) -> None:
        cache = test_directory("audio-cache")
        source = cache / "incoming.silk"
        source.write_bytes(b"test-silk")
        project_root = test_directory("audio-project")
        cfg = settings(
            project_root=project_root,
            voice_asr_enabled=True,
            media_cache_roots=(cache,),
        )
        observed_paths: list[Path] = []

        def fake_decode(source_stream: object, pcm: object, sample_rate: int) -> None:
            self.assertEqual(24_000, sample_rate)
            self.assertEqual(b"test-silk", source_stream.read())
            pcm.write(b"\x00\x00" * 480)

        def fake_transcribe(audio_path: str, *, model: str) -> dict[str, object]:
            self.assertEqual("small", model)
            prepared = Path(audio_path)
            self.assertTrue(prepared.is_file())
            self.assertEqual(".wav", prepared.suffix)
            observed_paths.append(prepared)
            return {"success": True, "provider": "local", "transcript": "测试语音"}

        with patch("pysilk.decode", side_effect=fake_decode), patch(
            "tools.transcription_tools.transcribe_audio_local_fallback",
            side_effect=fake_transcribe,
        ):
            transcript, fingerprint = LocalMediaPreprocessor(cfg)._transcribe_audio(
                str(source)
            )

        self.assertEqual("测试语音", transcript)
        self.assertEqual(64, len(fingerprint))
        self.assertEqual(1, len(observed_paths))
        self.assertFalse(observed_paths[0].exists())
        self.assertEqual([], list(cfg.media_work_dir.glob("voice-*.wav")))

        with patch("pysilk.decode", side_effect=fake_decode), patch(
            "tools.transcription_tools.transcribe_audio_local_fallback",
            return_value={"success": False, "provider": "local"},
        ):
            with self.assertRaises(MediaPreparationError):
                LocalMediaPreprocessor(cfg)._transcribe_audio(str(source))
        self.assertEqual([], list(cfg.media_work_dir.glob("voice-*.wav")))


class ModelRoutingTests(unittest.TestCase):
    class FakeLlm:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def complete_structured(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            schema_name = str(kwargs.get("schema_name") or "")
            if "vision.extract" in schema_name:
                parsed = {
                    "description": "一张产品草图",
                    "visible_text": "新功能",
                    "confidence": 0.95,
                }
            elif "task" in schema_name:
                parsed = {
                    "tasks": [
                        {
                            "title": "提交报告",
                            "due_date": "",
                            "due_time": "",
                            "priority": "none",
                            "category": "",
                            "tags": [],
                            "description": "",
                            "reminder_at": "",
                        }
                    ],
                    "confidence": 0.9,
                    "clarification": "",
                }
            else:
                parsed = {
                    "notes": [
                        {
                            "title": "产品草图",
                            "body": "新功能",
                            "summary": "产品想法",
                            "tags": [],
                            "links": [],
                            "target_hint": "",
                        }
                    ],
                    "confidence": 0.9,
                    "clarification": "",
                }
            return SimpleNamespace(parsed=parsed)

    def classifier(self) -> tuple[HermesStructuredClassifier, "ModelRoutingTests.FakeLlm"]:
        fake = self.FakeLlm()
        return HermesStructuredClassifier(SimpleNamespace(llm=fake), settings()), fake

    def test_task_text_uses_flash_classifier_slot_and_compact_schema(self) -> None:
        classifier, fake = self.classifier()
        classifier.classify(message("model-task", ""), "提交报告", IntentKind.TASK, (), ())
        self.assertEqual("wechat_secretary_classifier", fake.calls[0]["task"])
        self.assertEqual("wechat.secretary.task.v1", fake.calls[0]["schema_name"])
        self.assertEqual(600, fake.calls[0]["max_tokens"])

    def test_image_uses_vision_slot_in_one_structured_call(self) -> None:
        classifier, fake = self.classifier()
        classifier.classify(
            message("model-image", ""),
            "",
            IntentKind.NOTE,
            (),
            (),
            image_inputs=({"type": "image", "data": b"jpeg", "mime_type": "image/jpeg"},),
        )
        self.assertEqual(1, len(fake.calls))
        self.assertEqual("wechat_secretary_vision", fake.calls[0]["task"])

    def test_note_prompt_requires_professional_objective_style(self) -> None:
        classifier, fake = self.classifier()
        classifier.classify(
            message("model-note-style", ""),
            "整理会议记录",
            IntentKind.NOTE,
            (),
            (),
        )
        instructions = str(fake.calls[0]["instructions"])
        self.assertEqual(1, len(fake.calls))
        self.assertIn("严谨、客观、专业", instructions)
        self.assertIn("保留主体归属、否定、条件、数字、时间和不确定程度", instructions)
        self.assertIn("不补充背景、因果、结论、建议或行动项", instructions)

        auto_classifier, auto_fake = self.classifier()
        auto_classifier.classify(
            message("model-auto-note-style", ""),
            "整理会议记录",
            None,
            (),
            (),
        )
        self.assertEqual(1, len(auto_fake.calls))
        self.assertIn("严谨、客观、专业", str(auto_fake.calls[0]["instructions"]))

    def test_deep_image_uses_vision_then_pro_without_forwarding_image_twice(self) -> None:
        classifier, fake = self.classifier()
        classifier.classify(
            message("deep-image", ""),
            "深入整理",
            IntentKind.NOTE,
            (),
            (),
            deep_note=True,
            image_inputs=({"type": "image", "data": b"jpeg", "mime_type": "image/jpeg"},),
        )
        self.assertEqual(2, len(fake.calls))
        self.assertEqual("wechat_secretary_vision", fake.calls[0]["task"])
        self.assertEqual("wechat_secretary_deep_note", fake.calls[1]["task"])
        self.assertFalse(
            any(block.get("type") == "image" for block in fake.calls[1]["input"])
        )


class LedgerPrivacyTests(unittest.TestCase):
    class FlakyPrivateInbox:
        def __init__(self) -> None:
            self.calls = 0

        def save(self, incoming: MessageEnvelope) -> ActionResult:
            del incoming
            self.calls += 1
            if self.calls == 1:
                return ActionResult(
                    "private",
                    ExecutionStatus.FAILED,
                    "私密内容",
                    error="本地磁盘暂不可用",
                )
            return ActionResult(
                "private",
                ExecutionStatus.SUCCEEDED,
                "私密内容",
                destination="2026-08-24.md",
            )

    def test_media_identity_is_part_of_content_collision_check(self) -> None:
        ledger = IdempotencyLedger(":memory:")
        first = replace(
            message("media-collision", "私密：附件"),
            media_paths=("cache/first.jpg",),
            media_types=("image",),
        )
        second = replace(first, media_paths=("cache/second.jpg",))
        self.assertTrue(ledger.claim(first).is_new)
        ledger.finish(first, ExecutionStatus.SUCCEEDED)
        collision = ledger.claim(second)
        self.assertFalse(collision.is_new)
        self.assertFalse(collision.content_matches)

    def test_operation_preview_and_private_sender_are_not_stored_plaintext(self) -> None:
        ledger = IdempotencyLedger(":memory:")
        incoming = message("privacy-ledger", "笔记：敏感正文")
        ledger.claim(incoming)
        ledger.claim_operation(incoming, "note_write:0", "note")
        ledger.finish_operation(
            incoming,
            "note_write:0",
            ActionResult(
                "note",
                ExecutionStatus.PLANNED,
                "笔记标题",
                preview="PREVIEW-MUST-NOT-BE-PERSISTED",
            ),
        )
        ledger.arm_private_latch(incoming.sender_key, incoming.message_id, NOW)
        preview = ledger._connection.execute(
            "SELECT preview FROM operations WHERE message_id = ?",
            (incoming.message_id,),
        ).fetchone()["preview"]
        sender_key = ledger._connection.execute(
            "SELECT sender_key FROM private_latches"
        ).fetchone()["sender_key"]
        self.assertEqual("", preview)
        self.assertNotEqual(incoming.sender_key, sender_key)

    def test_latched_private_retry_can_never_fall_through_to_classifier(self) -> None:
        private_inbox = self.FlakyPrivateInbox()
        app, classifier, _ = service_with(private_inbox=private_inbox)
        app.handle(message("private-arm", "私密：下一条"))
        incoming = message("private-body", "FAILED-PRIVATE-MUST-STAY-LOCAL")
        first = app.handle(incoming)
        second = app.handle(incoming)
        self.assertEqual(ExecutionStatus.FAILED, first.status)
        self.assertEqual(ExecutionStatus.SUCCEEDED, second.status)
        self.assertEqual(2, private_inbox.calls)
        self.assertEqual(0, classifier.call_count)


class PluginScopeTests(unittest.TestCase):
    def test_non_weixin_events_are_left_to_other_platforms(self) -> None:
        cfg = settings()
        fake_service = SimpleNamespace(
            settings=cfg,
            accepts=lambda incoming: False,
        )
        bridge = GatewayBridge(fake_service, SimpleNamespace())
        event = SimpleNamespace(source=SimpleNamespace(platform="telegram"))
        self.assertEqual({"action": "allow"}, bridge(event, SimpleNamespace()))

    def test_plugin_mapping_results_are_serialized_as_text(self) -> None:
        rendered = _json_tool_result({"list_projects": {"ok": True, "result": []}})
        self.assertIsInstance(rendered, str)
        self.assertIn('"list_projects"', rendered)

    def test_allowed_weixin_event_schedules_worker_without_name_error(self) -> None:
        cfg = settings()
        fake_service = SimpleNamespace(
            settings=cfg,
            accepts=lambda incoming: True,
        )
        scheduler = SimpleNamespace(attach=lambda sender: None)
        bridge = GatewayBridge(fake_service, scheduler)
        event = SimpleNamespace(
            source=SimpleNamespace(
                platform="weixin",
                user_id="wx-user-1",
                chat_id="chat-1",
                chat_type="dm",
            ),
            message_id="gateway-message-1",
            text="待办：明天交报告",
            timestamp=NOW,
        )
        thread_factory = Mock(
            return_value=SimpleNamespace(start=lambda: None)
        )

        async def invoke() -> dict[str, str]:
            with patch(
                "wechat_secretary.hermes_plugin.threading.Thread",
                thread_factory,
            ):
                return bridge(event, SimpleNamespace())

        outcome = asyncio.run(invoke())
        self.assertEqual("secretary-handled", outcome["reason"])
        self.assertTrue(
            thread_factory.call_args.kwargs["name"].startswith("secretary-")
        )

    def test_pre_dispatch_failure_schedules_one_clear_failure_reply(self) -> None:
        cfg = settings()
        bridge = GatewayBridge(
            SimpleNamespace(settings=cfg, accepts=lambda incoming: True),
            SimpleNamespace(),
        )
        event = SimpleNamespace(
            source=SimpleNamespace(platform="weixin", user_id="wx-user-1")
        )
        thread_factory = Mock(return_value=SimpleNamespace(start=lambda: None))

        async def invoke() -> dict[str, str]:
            with patch(
                "wechat_secretary.hermes_plugin._event_to_message",
                side_effect=RuntimeError("test failure"),
            ), patch(
                "wechat_secretary.hermes_plugin.threading.Thread",
                thread_factory,
            ):
                return bridge(event, SimpleNamespace())

        outcome = asyncio.run(invoke())
        self.assertEqual("secretary-fail-closed", outcome["reason"])
        self.assertEqual(1, thread_factory.call_count)
        args = thread_factory.call_args.kwargs["args"]
        self.assertIn("抱歉，这次没能处理成功", args[3])


class ProfileIsolationTests(unittest.TestCase):
    def test_profiles_use_independent_state_databases_and_idempotency(self) -> None:
        root = test_directory("profiles")
        owner = settings(project_root=root, profile_id="owner")
        partner = settings(project_root=root, profile_id="partner")
        self.assertNotEqual(owner.state_db_path, partner.state_db_path)

        owner_ledger = IdempotencyLedger(owner.state_db_path)
        partner_ledger = IdempotencyLedger(partner.state_db_path)
        try:
            incoming = message("same-message-id", "待办：提交报告")
            self.assertTrue(owner_ledger.claim(incoming).is_new)
            self.assertTrue(partner_ledger.claim(incoming).is_new)
            self.assertFalse(owner_ledger.claim(incoming).is_new)
            self.assertFalse(partner_ledger.claim(incoming).is_new)
            reminder_at = NOW + timedelta(hours=1)
            owner_ledger.enqueue_reminder(
                incoming,
                TaskReference("owner-task", "owner 提醒"),
                reminder_at,
            )
            self.assertEqual(
                "pending",
                owner_ledger.reminder_status("owner-task", reminder_at),
            )
            self.assertIsNone(
                partner_ledger.reminder_status("owner-task", reminder_at)
            )
        finally:
            owner_ledger.close()
            partner_ledger.close()

    def test_local_profile_configs_point_to_separate_data_roots(self) -> None:
        owner = SecretarySettings.from_file(ROOT / "config" / "secretary.toml", ROOT)
        partner = SecretarySettings.from_file(
            ROOT / "config" / "secretary.partner.toml", ROOT
        )
        self.assertEqual("owner", owner.profile_id)
        self.assertEqual("partner", partner.profile_id)
        self.assertTrue(owner.reminders_enabled)
        self.assertTrue(partner.reminders_enabled)
        self.assertNotEqual(owner.vault_path, partner.vault_path)
        self.assertNotEqual(owner.private_inbox_path, partner.private_inbox_path)
        self.assertTrue(all("hermes-home-partner" not in str(path) for path in owner.media_cache_roots))
        self.assertTrue(all("hermes-home-partner" in str(path) for path in partner.media_cache_roots))


class OAuthCompatibilityTests(unittest.TestCase):
    def test_only_allowlisted_https_root_slash_is_normalized(self) -> None:
        self.assertEqual(
            "https://dida365.com",
            _normalize_root_issuer_trailing_slash(
                "https://dida365.com/", "dida365.com"
            ),
        )
        unchanged = (
            "http://dida365.com/",
            "https://evil.example/",
            "https://dida365.com/oauth/",
            "https://dida365.com/?tenant=1",
        )
        for value in unchanged:
            with self.subTest(value=value):
                self.assertEqual(
                    value,
                    _normalize_root_issuer_trailing_slash(value, "dida365.com"),
                )


class WeixinReplyDeliveryTests(unittest.TestCase):
    def test_compact_multiline_reply_stays_in_one_bubble(self) -> None:
        from gateway.platforms.weixin import _split_text_for_weixin_delivery

        content = (
            "Dry Run｜已为你整理好模拟结果\n"
            "任务：提交报销材料｜Inbox\n"
            "日期：2026-08-27"
        )
        self.assertEqual(
            [content],
            _split_text_for_weixin_delivery(
                content,
                max_length=2_000,
                split_per_line=False,
            ),
        )

    def test_adapter_sends_multiline_reply_as_one_text_message(self) -> None:
        from gateway.config import PlatformConfig
        from gateway.platforms.weixin import WeixinAdapter

        content = (
            "Dry Run｜已为你整理好模拟结果\n"
            "任务：提交报销材料｜Inbox\n"
            "日期：2026-08-27"
        )
        sent_texts: list[str] = []

        async def fake_send_message(*args: object, **kwargs: object) -> dict[str, int]:
            del args
            sent_texts.append(str(kwargs["text"]))
            return {"ret": 0}

        with patch("gateway.platforms.weixin.ContextTokenStore") as token_store:
            token_store.return_value.get.return_value = None
            adapter = WeixinAdapter(
                PlatformConfig(
                    enabled=True,
                    token="test-token",
                    extra={"account_id": "test-account"},
                )
            )
        adapter._send_session = object()  # type: ignore[assignment]
        with patch(
            "gateway.platforms.weixin._send_message",
            new=fake_send_message,
        ):
            result = asyncio.run(adapter.send("chat", content))

        self.assertTrue(result.success)
        self.assertEqual([content], sent_texts)


class McpTrustCompatibilityTests(unittest.TestCase):
    def tearDown(self) -> None:
        for store in (
            _server_trust_levels,
            _tool_read_only_hints,
            _operator_approved_tools,
            _operator_approved_tools_when_env,
        ):
            store.pop("test-dida", None)

    def test_only_exact_discovered_operator_tools_bypass_prompt(self) -> None:
        tools = [
            SimpleNamespace(name="list_projects", annotations=None),
            SimpleNamespace(name="create_task", annotations=None),
        ]
        _record_tool_trust_metadata(
            "test-dida",
            {
                "trust": "untrusted",
                "operator_approved_tools": ["list_projects", "not_discovered"],
            },
            tools,
        )
        self.assertEqual(
            {"list_projects"}, _operator_approved_tools["test-dida"]
        )
        self.assertIsNone(_trust_gate_check("test-dida", "list_projects"))
        approval = Mock(return_value="deny")
        fake_approval_module = SimpleNamespace(
            request_elicitation_consent=approval
        )
        with patch.dict(
            sys.modules, {"tools.approval": fake_approval_module}
        ):
            blocked = _trust_gate_check("test-dida", "create_task")
        self.assertIsNotNone(blocked)
        approval.assert_called_once()

    def test_create_and_complete_require_independent_startup_flags(self) -> None:
        tools = [
            SimpleNamespace(name="create_task", annotations=None),
            SimpleNamespace(name="complete_task", annotations=None),
        ]
        _record_tool_trust_metadata(
            "test-dida",
            {
                "trust": "untrusted",
                "operator_approved_tools_when_env": {
                    "SECRETARY_DIDA_CREATES_APPROVED": ["create_task"],
                    "SECRETARY_DIDA_COMPLETIONS_APPROVED": ["complete_task"],
                },
            },
            tools,
        )
        with patch.dict(
            os.environ,
            {
                "SECRETARY_DIDA_CREATES_APPROVED": "1",
                "SECRETARY_DIDA_COMPLETIONS_APPROVED": "0",
            },
            clear=False,
        ):
            self.assertIsNone(_trust_gate_check("test-dida", "create_task"))
            approval = Mock(return_value="deny")
            fake_approval_module = SimpleNamespace(
                request_elicitation_consent=approval
            )
            with patch.dict(
                sys.modules, {"tools.approval": fake_approval_module}
            ):
                blocked = _trust_gate_check("test-dida", "complete_task")
            self.assertIsNotNone(blocked)
            approval.assert_called_once()

        with patch.dict(
            os.environ,
            {
                "SECRETARY_DIDA_CREATES_APPROVED": "0",
                "SECRETARY_DIDA_COMPLETIONS_APPROVED": "1",
            },
            clear=False,
        ):
            self.assertIsNone(_trust_gate_check("test-dida", "complete_task"))
        approval = Mock(return_value="deny")
        fake_approval_module = SimpleNamespace(
            request_elicitation_consent=approval
        )
        with patch.dict(
            os.environ,
            {
                "SECRETARY_DIDA_CREATES_APPROVED": "true",
                "SECRETARY_DIDA_COMPLETIONS_APPROVED": "0",
            },
            clear=False,
        ):
            with patch.dict(
                sys.modules, {"tools.approval": fake_approval_module}
            ):
                blocked = _trust_gate_check("test-dida", "create_task")
        self.assertIsNotNone(blocked)
        approval.assert_called_once()


class DidaSchemaValidationTests(unittest.TestCase):
    @staticmethod
    def valid_schemas() -> dict[str, object]:
        task_fields = {
            name: {}
            for name in (
                "title",
                "projectId",
                "content",
                "kind",
                "dueDate",
                "timeZone",
                "isAllDay",
                "priority",
                "tags",
            )
        }
        return {
            "create_task": {
                "parameters": {
                    "properties": {"task": {"$ref": "#/$defs/OpenTask"}},
                    "required": ["task"],
                    "$defs": {"OpenTask": {"properties": task_fields}},
                }
            },
            "complete_task": {
                "parameters": {
                    "properties": {"project_id": {}, "task_id": {}},
                    "required": ["project_id", "task_id"],
                }
            },
            "get_task_by_id": {
                "parameters": {
                    "properties": {"task_id": {}},
                    "required": ["task_id"],
                }
            },
            "search_task": {
                "parameters": {
                    "properties": {"query": {}},
                    "required": ["query"],
                }
            },
        }

    def test_live_schema_validator_rejects_empty_or_incomplete_definitions(self) -> None:
        errors = _dida_schema_errors(
            {
                "create_task": {"parameters": {}},
                "complete_task": {"parameters": {}},
                "get_task_by_id": {"parameters": {}},
            }
        )
        self.assertTrue(errors)

        incomplete = self.valid_schemas()
        incomplete["complete_task"]["parameters"]["required"] = ["task_id"]
        self.assertTrue(_dida_schema_errors(incomplete))

    def test_live_schema_validator_accepts_expected_contract(self) -> None:
        self.assertEqual((), _dida_schema_errors(self.valid_schemas()))


class ConfigSafetyTests(unittest.TestCase):
    def real_settings(self, **changes: object) -> SecretarySettings:
        root = test_directory("real-config")
        return settings(
            dry_run=False,
            vault_path=root / "vault",
            private_inbox_path=root / "private",
            obsidian_mapping_confirmed=True,
            dida_mapping_confirmed=True,
            dida_schema_confirmed=True,
            **changes,
        )

    def test_real_mode_doctor_requires_exact_create_approval(self) -> None:
        cfg = self.real_settings()
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_CREATES_APPROVED": "true"}, clear=False
        ):
            self.assertTrue(
                any("允许创建" in error for error in cfg.runtime_errors(strict=True))
            )
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_CREATES_APPROVED": "1"}, clear=False
        ):
            self.assertFalse(
                any("允许创建" in error for error in cfg.runtime_errors(strict=True))
            )

    def test_completion_approval_requires_confirmed_completion_schema(self) -> None:
        cfg = self.real_settings(dida_complete_schema_confirmed=False)
        with patch.dict(
            os.environ,
            {
                "SECRETARY_DIDA_CREATES_APPROVED": "1",
                "SECRETARY_DIDA_COMPLETIONS_APPROVED": "1",
            },
            clear=False,
        ):
            self.assertTrue(
                any("complete_task" in error for error in cfg.runtime_errors(strict=True))
            )


class DidaWriteContractTests(unittest.TestCase):
    def test_read_only_task_inspector_requires_a_bounded_title(self) -> None:
        self.assertEqual(
            2,
            command_inspect_dida_task(SimpleNamespace(title="")),
        )
        self.assertEqual(
            2,
            command_inspect_dida_task(SimpleNamespace(title="x" * 201)),
        )

    def test_dedicated_complete_probe_requires_explicit_confirmation(self) -> None:
        self.assertEqual(
            2,
            command_verify_dida_complete(
                SimpleNamespace(confirm_complete_test=False)
            ),
        )

    def test_dedicated_create_probe_requires_explicit_confirmation(self) -> None:
        self.assertEqual(
            2,
            command_verify_dida_create(
                SimpleNamespace(confirm_create_test=False)
            ),
        )

    def test_dedicated_probes_require_exact_process_approval(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SECRETARY_DIDA_CREATE_TEST_APPROVED": "true",
                "SECRETARY_DIDA_COMPLETION_TEST_APPROVED": "true",
            },
            clear=False,
        ):
            self.assertEqual(
                2,
                command_verify_dida_create(
                    SimpleNamespace(confirm_create_test=True)
                ),
            )
            self.assertEqual(
                2,
                command_verify_dida_complete(
                    SimpleNamespace(confirm_complete_test=True)
                ),
            )

    def test_dedicated_complete_probe_refuses_existing_lock(self) -> None:
        root = test_directory("dida-complete-lock")
        state_path = root / "dida-contract-test.json"
        lock_path = root / "dida-contract-test.complete.lock"
        lock_path.write_text("reserved\n", encoding="utf-8")
        runner = Mock(return_value=0)
        try:
            with (
                patch.dict(
                    os.environ,
                    {"SECRETARY_DIDA_COMPLETION_TEST_APPROVED": "1"},
                    clear=False,
                ),
                patch("wechat_secretary.cli.load_settings", return_value=settings()),
                patch(
                    "wechat_secretary.cli._dida_contract_state_path",
                    return_value=state_path,
                ),
                patch(
                    "wechat_secretary.cli._command_verify_dida_complete_locked",
                    runner,
                ),
            ):
                self.assertEqual(
                    1,
                    command_verify_dida_complete(
                        SimpleNamespace(confirm_complete_test=True)
                    ),
                )
        finally:
            lock_path.unlink(missing_ok=True)
        runner.assert_not_called()

    def test_dedicated_complete_probe_writes_once_and_verifies_exact_task(self) -> None:
        root = test_directory("dida-complete-success")
        state_path = root / "dida-contract-test.json"
        cfg = settings(
            profile_id="owner",
            dida_schema_confirmed=True,
            dida_server="dida365",
        )
        state_path.write_text(
            json.dumps(
                {
                    "profile_id": "owner",
                    "title": "专用核验任务",
                    "destination": "Inbox",
                    "project_id": "project-secret",
                    "task_id": "task-secret",
                    "status": "created_verified",
                    "error_type": "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        calls: list[tuple[str, dict]] = []

        class FakeSession:
            async def call_tool(self, tool: str, arguments: dict) -> object:
                calls.append((tool, arguments))
                if tool == "complete_task":
                    return SimpleNamespace(
                        is_error=False, structured_content={"accepted": True}
                    )
                completed = len(calls) > 1
                return SimpleNamespace(
                    is_error=False,
                    structured_content={
                        "id": "task-secret",
                        "title": "专用核验任务",
                        "projectId": "project-secret",
                        "status": 2 if completed else 0,
                    },
                )

        class FakeServer:
            session = FakeSession()

            async def shutdown(self) -> None:
                return None

        config_module = ModuleType("hermes_cli.mcp_config")
        config_module._get_mcp_servers = lambda: {"dida365": {}}
        config_module._resolve_mcp_server_config = lambda value: value
        mcp_module = ModuleType("tools.mcp_tool")

        async def connect(server_name: str, config: object) -> FakeServer:
            del server_name, config
            return FakeServer()

        mcp_module._connect_server = connect
        mcp_module._ensure_mcp_loop = lambda: None
        mcp_module._run_on_mcp_loop = lambda coro, timeout: asyncio.run(coro)
        mcp_module._stop_mcp_loop_if_idle = lambda: None

        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {"SECRETARY_DIDA_COMPLETION_TEST_APPROVED": "1"},
                clear=False,
            ),
            patch("wechat_secretary.cli.load_settings", return_value=cfg),
            patch(
                "wechat_secretary.cli._dida_contract_state_path",
                return_value=state_path,
            ),
            patch.dict(
                sys.modules,
                {
                    "hermes_cli.mcp_config": config_module,
                    "tools.mcp_tool": mcp_module,
                },
            ),
            redirect_stdout(output),
        ):
            result = command_verify_dida_complete(
                SimpleNamespace(confirm_complete_test=True)
            )

        self.assertEqual(0, result)
        self.assertEqual(
            [
                ("get_task_by_id", {"task_id": "task-secret"}),
                (
                    "complete_task",
                    {"project_id": "project-secret", "task_id": "task-secret"},
                ),
                ("get_task_by_id", {"task_id": "task-secret"}),
            ],
            calls,
        )
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual("completed_verified", saved["status"])
        self.assertNotIn("task-secret", output.getvalue())
        self.assertNotIn("project-secret", output.getvalue())
        self.assertFalse((root / "dida-contract-test.complete.lock").exists())

    def test_dedicated_complete_probe_never_retries_uncertain_write(self) -> None:
        root = test_directory("dida-complete-uncertain")
        state_path = root / "dida-contract-test.json"
        cfg = settings(
            profile_id="owner",
            dida_schema_confirmed=True,
            dida_server="dida365",
        )
        state_path.write_text(
            json.dumps(
                {
                    "profile_id": "owner",
                    "title": "专用核验任务",
                    "destination": "Inbox",
                    "project_id": "project-secret",
                    "task_id": "task-secret",
                    "status": "created_verified",
                    "error_type": "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        calls: list[str] = []

        class FakeSession:
            async def call_tool(self, tool: str, arguments: dict) -> object:
                del arguments
                calls.append(tool)
                if tool == "complete_task":
                    raise TimeoutError("simulated")
                return SimpleNamespace(
                    is_error=False,
                    structured_content={
                        "id": "task-secret",
                        "title": "专用核验任务",
                        "projectId": "project-secret",
                        "status": 0,
                    },
                )

        class FakeServer:
            session = FakeSession()

            async def shutdown(self) -> None:
                return None

        config_module = ModuleType("hermes_cli.mcp_config")
        config_module._get_mcp_servers = lambda: {"dida365": {}}
        config_module._resolve_mcp_server_config = lambda value: value
        mcp_module = ModuleType("tools.mcp_tool")

        async def connect(server_name: str, config: object) -> FakeServer:
            del server_name, config
            return FakeServer()

        mcp_module._connect_server = connect
        mcp_module._ensure_mcp_loop = lambda: None
        mcp_module._run_on_mcp_loop = lambda coro, timeout: asyncio.run(coro)
        mcp_module._stop_mcp_loop_if_idle = lambda: None

        with (
            patch.dict(
                os.environ,
                {"SECRETARY_DIDA_COMPLETION_TEST_APPROVED": "1"},
                clear=False,
            ),
            patch("wechat_secretary.cli.load_settings", return_value=cfg),
            patch(
                "wechat_secretary.cli._dida_contract_state_path",
                return_value=state_path,
            ),
            patch.dict(
                sys.modules,
                {
                    "hermes_cli.mcp_config": config_module,
                    "tools.mcp_tool": mcp_module,
                },
            ),
        ):
            self.assertEqual(
                1,
                command_verify_dida_complete(
                    SimpleNamespace(confirm_complete_test=True)
                ),
            )
            self.assertEqual(
                1,
                command_verify_dida_complete(
                    SimpleNamespace(confirm_complete_test=True)
                ),
            )

        self.assertEqual(["get_task_by_id", "complete_task"], calls)
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual("completion_uncertain", saved["status"])

    def test_create_requires_exact_startup_approval_without_calling_mcp(self) -> None:
        calls: list[str] = []

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, arguments, timeout
            calls.append(tool)
            return {"ok": True}

        cfg = settings(
            dry_run=False,
            dida_mapping_confirmed=True,
            dida_schema_confirmed=True,
            category_map={},
        )
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_CREATES_APPROVED": "true"}, clear=False
        ):
            result = DidaExecutor(cfg, caller).create_task(
                TaskDraft("提交报告"), message("create-no-approval", "待办：提交报告")
            )
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertFalse(calls)

    def test_create_uses_live_contract_and_requires_exact_readback(self) -> None:
        calls: list[tuple[str, dict]] = []

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, timeout
            calls.append((tool, arguments))
            if tool == "create_task":
                return {
                    "ok": True,
                    "result": "Task created successfully",
                    "structuredContent": {
                        "id": "task-created",
                        "title": "提交报告",
                        "projectId": "opaque-inbox-id",
                        "status": 0,
                    },
                }
            return {
                "ok": True,
                "result": "Task details loaded",
                "structuredContent": {
                    "id": "task-created",
                    "title": "提交报告",
                    "projectId": "opaque-inbox-id",
                    "status": 0,
                },
            }

        cfg = settings(
            dry_run=False,
            dida_mapping_confirmed=True,
            dida_schema_confirmed=True,
            category_map={},
        )
        draft = TaskDraft(
            "提交报告",
            due_date="2026-08-25",
            due_time="15:00",
            priority="high",
        )
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_CREATES_APPROVED": "1"}, clear=False
        ):
            result = DidaExecutor(cfg, caller).create_task(
                draft, message("create-contract", "待办：明天下午3点提交报告")
            )
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(["create_task", "get_task_by_id"], [name for name, _ in calls])
        self.assertEqual(
            {
                "task": {
                    "title": "提交报告",
                    "projectId": "inbox",
                    "dueDate": "2026-08-25T15:00:00+08:00",
                    "timeZone": "Asia/Shanghai",
                    "isAllDay": False,
                    "priority": 5,
                }
            },
            calls[0][1],
        )
        self.assertEqual({"task_id": "task-created"}, calls[1][1])

    def test_create_readback_mismatch_is_uncertain(self) -> None:
        calls: list[str] = []

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, arguments, timeout
            calls.append(tool)
            if tool == "create_task":
                return {
                    "ok": True,
                    "result": "Created the requested task with the correct title",
                    "structuredContent": {"id": "task-created"},
                }
            return {
                "ok": True,
                "result": "提交报告｜Inbox｜active",
                "structuredContent": {
                    "id": "task-created",
                    "title": "其他任务",
                    "projectId": "inbox",
                    "status": 0,
                },
            }

        cfg = settings(
            dry_run=False,
            dida_mapping_confirmed=True,
            dida_schema_confirmed=True,
            category_map={},
        )
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_CREATES_APPROVED": "1"}, clear=False
        ):
            result = DidaExecutor(cfg, caller).create_task(
                TaskDraft("提交报告"), message("create-mismatch", "待办：提交报告")
            )
        self.assertEqual(ExecutionStatus.UNCERTAIN, result.status)
        self.assertEqual(["create_task", "get_task_by_id"], calls)

    def test_query_and_search_use_structured_content_not_display_text(self) -> None:
        calls: list[tuple[str, dict]] = []

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, timeout
            calls.append((tool, arguments))
            return {
                "ok": True,
                "result": "No matching tasks were found",
                "structuredContent": {
                    "tasks": [
                        {
                            "id": "task-cd8",
                            "title": "分选CD8",
                            "projectId": "project-inbox",
                            "projectName": "Inbox",
                            "status": 0,
                        }
                    ]
                },
            }

        executor = DidaExecutor(settings(dry_run=False), caller)
        query = executor.query_tasks(TaskQuery(mode="search", keyword="分选CD8"))
        refs = executor.search_task_references("分选CD8")

        self.assertEqual(ExecutionStatus.SUCCEEDED, query.status)
        self.assertEqual("task-cd8", query.task_refs[0].task_id)
        self.assertEqual("project-inbox", query.task_refs[0].project_id)
        self.assertEqual("task-cd8", refs[0].task_id)
        self.assertEqual(
            [
                ("search_task", {"query": "分选CD8"}),
                ("search_task", {"query": "分选CD8"}),
            ],
            calls,
        )

    def test_exact_active_task_reference_requires_exact_readback(self) -> None:
        calls: list[tuple[str, dict]] = []

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, timeout
            calls.append((tool, arguments))
            if tool == "search_task":
                return {
                    "ok": True,
                    "structuredContent": {
                        "tasks": [
                            {
                                "id": "task-cd8",
                                "title": "分选cD8",
                                "projectId": "project-inbox",
                                "status": 0,
                            }
                        ]
                    },
                }
            return {
                "ok": True,
                "structuredContent": {
                    "id": "task-cd8",
                    "title": "分选cD8",
                    "projectId": "project-inbox",
                    "projectName": "Inbox",
                    "status": 0,
                },
            }

        refs = DidaExecutor(settings(dry_run=False), caller).exact_active_task_references(
            "分选cD8"
        )
        self.assertEqual(("task-cd8",), tuple(ref.task_id for ref in refs))
        self.assertEqual("project-inbox", refs[0].project_id)
        self.assertEqual(
            [
                ("search_task", {"query": "分选cD8"}),
                ("get_task_by_id", {"task_id": "task-cd8"}),
            ],
            calls,
        )

    def test_exact_active_task_reference_fails_closed_on_wrong_readback(self) -> None:
        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, arguments, timeout
            if tool == "search_task":
                return {
                    "ok": True,
                    "structuredContent": {
                        "id": "task-cd8",
                        "title": "分选cD8",
                        "status": 0,
                    },
                }
            return {
                "ok": True,
                "structuredContent": {
                    "id": "task-cd8",
                    "title": "其他任务",
                    "status": 0,
                },
            }

        executor = DidaExecutor(settings(dry_run=False), caller)
        with self.assertRaises(RuntimeError):
            executor.exact_active_task_references("分选cD8")


class CompletionTests(unittest.TestCase):
    def test_acknowledgements_and_negative_reply_never_complete(self) -> None:
        app, classifier, _ = service_with()
        for index, text in enumerate(("收到", "好的", "知道了", "开始做", "没做完")):
            result = app.handle(message(f"ack-{index}", text))
            self.assertIn("不会更改", result.reply)
        self.assertEqual(0, classifier.call_count)

    def test_explicit_recent_completion_uses_exact_task_id(self) -> None:
        app, classifier, _ = service_with()
        created = app.handle(message("create-1", "待办：明天提交报告"))
        task_id = created.results[0].external_id
        completed = app.handle(message("complete-1", "搞定了", NOW + timedelta(minutes=1)))
        self.assertEqual("complete", completed.results[0].action)
        self.assertEqual(task_id, completed.results[0].external_id)
        self.assertIn("完成任务", completed.reply)
        self.assertEqual(1, classifier.call_count)

    def test_ambiguous_recent_completion_requires_numbered_confirmation(self) -> None:
        app, classifier, ledger = service_with()
        refs = (
            TaskReference("task-1", "提交报告", "工作"),
            TaskReference("task-2", "提交预算", "工作"),
        )
        ledger.record_task_context(
            message("source", "").sender_key,
            refs,
            batch_id="reminder-batch",
            source_message_id="outbound-1",
            observed_at=NOW,
            ttl_seconds=3600,
            context_kind="reminder",
            reminder_at=NOW,
        )
        ask = app.handle(message("complete-many", "已完成", NOW + timedelta(minutes=1)))
        self.assertIn("可能的任务", ask.reply)
        self.assertEqual(0, classifier.call_count)
        selected = app.handle(message("complete-pick", "完成 2", NOW + timedelta(minutes=2)))
        self.assertEqual("task-2", selected.results[0].external_id)

    def test_single_fuzzy_title_match_still_requires_numbered_confirmation(self) -> None:
        class FuzzyDida:
            def __init__(self) -> None:
                self.complete_calls = 0

            def search_task_references(self, title: str) -> tuple[TaskReference, ...]:
                self.asserted_title = title
                return (
                    TaskReference(
                        "task-report",
                        "提交年度报告",
                        "工作",
                        "project-work",
                    ),
                )

            def complete_task(self, task: TaskReference) -> ActionResult:
                self.complete_calls += 1
                return ActionResult(
                    "complete",
                    ExecutionStatus.PLANNED,
                    task.title,
                    task.category,
                    external_id=task.task_id,
                    task_refs=(task,),
                )

        dida = FuzzyDida()
        app, _, _ = service_with(dida=dida)
        ask = app.handle(message("complete-fuzzy", "完成：报告"))
        self.assertIn("可能的任务", ask.reply)
        self.assertEqual(0, dida.complete_calls)

        confirmed = app.handle(
            message("complete-fuzzy-pick", "完成 1", NOW + timedelta(minutes=1))
        )
        self.assertEqual(1, dida.complete_calls)
        self.assertEqual("task-report", confirmed.results[0].external_id)

    def test_context_expiry_never_guesses(self) -> None:
        app, _, ledger = service_with()
        ledger.record_task_context(
            message("source", "").sender_key,
            (TaskReference("task-old", "旧任务", "工作"),),
            batch_id="old",
            source_message_id="outbound-old",
            observed_at=NOW - timedelta(days=2),
            ttl_seconds=60,
            context_kind="reminder",
        )
        result = app.handle(message("complete-old", "做完了", NOW))
        self.assertIn("已过期", result.reply)
        self.assertFalse(result.results)

    def test_batch_completion_is_refused(self) -> None:
        app, classifier, _ = service_with()
        result = app.handle(message("complete-batch", "这些都完成了"))
        self.assertIn("不会批量完成任务", result.reply)
        self.assertEqual(0, classifier.call_count)

    def test_complete_task_requires_successful_readback(self) -> None:
        calls: list[tuple[str, dict]] = []
        read_count = 0

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            nonlocal read_count
            del server, timeout
            calls.append((tool, arguments))
            if tool == "complete_task":
                return {
                    "ok": True,
                    "result": "Task completed",
                    "structuredContent": {"task_id": "t1"},
                }
            read_count += 1
            return {
                "ok": True,
                "result": "Human-readable task details",
                "structuredContent": {
                    "task_id": "t1",
                    "project_id": "project-work",
                    "title": "交报告",
                    "status": 0 if read_count == 1 else "completed",
                },
            }

        cfg = settings(dry_run=False, dida_complete_schema_confirmed=True)
        executor = DidaExecutor(cfg, caller)
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_COMPLETIONS_APPROVED": "1"}, clear=False
        ):
            result = executor.complete_task(
                TaskReference("t1", "交报告", "工作", "project-work")
            )
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(
            ["get_task_by_id", "complete_task", "get_task_by_id"],
            [name for name, _ in calls],
        )
        self.assertEqual({"task_id": "t1"}, calls[0][1])
        self.assertEqual(
            {"project_id": "project-work", "task_id": "t1"}, calls[1][1]
        )
        self.assertEqual({"task_id": "t1"}, calls[2][1])

    def test_complete_task_refuses_missing_project_id(self) -> None:
        calls: list[str] = []

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, tool, arguments, timeout
            calls.append("called")
            return {"ok": True}

        cfg = settings(dry_run=False, dida_complete_schema_confirmed=True)
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_COMPLETIONS_APPROVED": "1"}, clear=False
        ):
            result = DidaExecutor(cfg, caller).complete_task(
                TaskReference("t1", "交报告", "工作")
            )
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertFalse(calls)

    def test_complete_task_refuses_missing_task_id_or_completion_approval(self) -> None:
        calls: list[str] = []

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, arguments, timeout
            calls.append(tool)
            return {"ok": True}

        cfg = settings(dry_run=False, dida_complete_schema_confirmed=True)
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_COMPLETIONS_APPROVED": "1"}, clear=False
        ):
            missing = DidaExecutor(cfg, caller).complete_task(
                TaskReference("", "交报告", "工作", "project-work")
            )
        self.assertEqual(ExecutionStatus.FAILED, missing.status)
        self.assertFalse(calls)

        with patch.dict(
            os.environ, {"SECRETARY_DIDA_COMPLETIONS_APPROVED": "true"}, clear=False
        ):
            unapproved = DidaExecutor(cfg, caller).complete_task(
                TaskReference("t1", "交报告", "工作", "project-work")
            )
        self.assertEqual(ExecutionStatus.FAILED, unapproved.status)
        self.assertFalse(calls)

    def test_complete_readback_rejects_wrong_or_nested_completed_task(self) -> None:
        calls: list[str] = []
        read_count = 0

        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            nonlocal read_count
            del server, arguments, timeout
            calls.append(tool)
            if tool == "complete_task":
                return {"ok": True, "result": {"task_id": "t1"}}
            read_count += 1
            if read_count == 1:
                return {
                    "ok": True,
                    "result": {
                        "task_id": "t1",
                        "project_id": "project-work",
                        "title": "交报告",
                        "status": 0,
                    },
                }
            return {
                "ok": True,
                "result": {
                    "task_id": "t1",
                    "project_id": "project-work",
                    "title": "交报告",
                    "status": 0,
                    "items": [
                        {
                            "task_id": "other",
                            "project_id": "project-work",
                            "title": "子任务",
                            "status": "completed",
                        }
                    ],
                },
            }

        cfg = settings(dry_run=False, dida_complete_schema_confirmed=True)
        with patch.dict(
            os.environ, {"SECRETARY_DIDA_COMPLETIONS_APPROVED": "1"}, clear=False
        ):
            result = DidaExecutor(cfg, caller).complete_task(
                TaskReference("t1", "交报告", "工作", "project-work")
            )
        self.assertEqual(ExecutionStatus.UNCERTAIN, result.status)
        self.assertEqual(
            ["get_task_by_id", "complete_task", "get_task_by_id"], calls
        )


class DigestTests(unittest.TestCase):
    def test_morning_digest_is_concise_and_keeps_task_ids_for_context(self) -> None:
        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, arguments, timeout
            self.assertEqual("list_undone_tasks_by_time_query", tool)
            return {
                "ok": True,
                "result": "No tasks",
                "structuredContent": [
                    {"task_id": "today-1", "title": "提交报告", "category": "工作"},
                    {"task_id": "today-2", "title": "预约体检", "category": "个人"},
                ],
            }

        executor = DidaExecutor(settings(dry_run=False), caller)
        text, refs = executor.scheduled_digest("morning", NOW)
        self.assertIn("今日重点", text)
        self.assertIn("1. 提交报告｜工作", text)
        self.assertEqual(("today-1", "today-2"), tuple(ref.task_id for ref in refs))


class ReminderTests(unittest.TestCase):
    def test_named_reminder_binds_one_existing_task_without_model_or_create(self) -> None:
        cfg = settings(dry_run=False, reminders_enabled=True)
        finder = Mock(
            return_value=(
                TaskReference(
                    "task-cd8",
                    "分选cD8",
                    "Inbox",
                    "project-inbox",
                    "0",
                ),
            )
        )
        dida = SimpleNamespace(exact_active_task_references=finder)
        app, classifier, ledger = service_with(cfg, dida=dida)

        result = app.handle(
            message(
                "bind-existing-reminder",
                "补设提醒：2026-08-24 14:00｜分选cD8",
            )
        )

        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertIn("已为你设置好微信提醒", result.reply)
        self.assertEqual(0, classifier.call_count)
        finder.assert_called_once_with("分选cD8")
        self.assertEqual(
            "pending",
            ledger.reminder_status(
                "task-cd8", datetime.fromisoformat("2026-08-24T14:00:00+08:00")
            ),
        )

    def test_named_reminder_refuses_ambiguous_existing_tasks(self) -> None:
        cfg = settings(dry_run=False, reminders_enabled=True)
        dida = SimpleNamespace(
            exact_active_task_references=Mock(
                return_value=(
                    TaskReference("task-1", "分选cD8", "Inbox"),
                    TaskReference("task-2", "分选cD8", "Inbox"),
                )
            )
        )
        app, classifier, ledger = service_with(cfg, dida=dida)

        result = app.handle(
            message(
                "bind-ambiguous-reminder",
                "补设提醒：今天14:00｜分选cD8",
            )
        )

        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertIn("多个同名", result.reply)
        self.assertEqual(0, classifier.call_count)
        self.assertIsNone(
            ledger.reminder_status(
                "task-1", datetime.fromisoformat("2026-08-24T14:00:00+08:00")
            )
        )

    def test_named_reminder_respects_profile_reminder_gate(self) -> None:
        cfg = settings(dry_run=False, reminders_enabled=False)
        dida = SimpleNamespace(
            exact_active_task_references=Mock(
                return_value=(TaskReference("task-cd8", "分选cD8", "Inbox"),)
            )
        )
        app, classifier, ledger = service_with(cfg, dida=dida)

        result = app.handle(
            message(
                "bind-disabled-reminder",
                "补设提醒：今天14:00｜分选cD8",
            )
        )

        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertIn("尚未获准启用", result.reply)
        self.assertEqual(0, classifier.call_count)
        self.assertIsNone(
            ledger.reminder_status(
                "task-cd8", datetime.fromisoformat("2026-08-24T14:00:00+08:00")
            )
        )

    def test_relative_reminder_reschedules_queue_without_dida_update(self) -> None:
        app, _, ledger = service_with()
        created = app.handle(message("r-create", "待办：提交月报"))
        task_id = created.results[0].external_id
        adjusted = app.handle(
            message("r-adjust", "半小时后提醒", NOW + timedelta(minutes=1))
        )
        self.assertEqual("reminder", adjusted.results[0].action)
        self.assertEqual(
            "pending",
            ledger.reminder_status(
                task_id, datetime.fromisoformat("2026-08-24T09:31:00+08:00")
            ),
        )

    def test_scheduler_merges_old_reminders_and_marks_delivery(self) -> None:
        cfg = settings(reminder_overdue_merge_seconds=7200)
        ledger = IdempotencyLedger(":memory:")
        old_message = message("old-source", "", NOW - timedelta(hours=4))
        recent_message = message("new-source", "", NOW - timedelta(minutes=10))
        old_at = NOW - timedelta(hours=3)
        recent_at = NOW - timedelta(minutes=5)
        ledger.enqueue_reminder(old_message, TaskReference("old-1", "旧提醒一"), old_at)
        ledger.enqueue_reminder(old_message, TaskReference("old-2", "旧提醒二"), old_at)
        ledger.enqueue_reminder(recent_message, TaskReference("new-1", "新提醒"), recent_at)
        sent: list[str] = []

        def sender(record: object, content: str) -> str:
            del record
            sent.append(content)
            return f"out-{len(sent)}"

        stats = ReminderScheduler(cfg, ledger).poll_once(sender, NOW)
        self.assertEqual(3, stats.sent)
        self.assertEqual(1, stats.merged_messages)
        self.assertEqual(2, len(sent))
        self.assertIn("我现在补给你", sent[0])
        self.assertEqual("提醒你一下，别忘了新提醒。", sent[1])
        self.assertNotIn("Inbox", sent[1])
        self.assertEqual("sent", ledger.reminder_status("new-1", recent_at))

    def test_sent_idempotency_key_is_never_reactivated(self) -> None:
        ledger = IdempotencyLedger(":memory:")
        incoming = message("same-reminder", "半小时后提醒")
        task = TaskReference("same-task", "同一提醒")
        reminder_at = NOW + timedelta(minutes=30)
        created, row_id = ledger.enqueue_reminder(incoming, task, reminder_at)
        self.assertTrue(created)
        ledger.mark_reminders_sent((row_id,), reminder_at, "outbound-1")
        repeated, repeated_row_id = ledger.enqueue_reminder(
            incoming,
            task,
            reminder_at,
            replace_existing=True,
        )
        self.assertFalse(repeated)
        self.assertEqual(row_id, repeated_row_id)
        self.assertEqual("sent", ledger.reminder_status(task.task_id, reminder_at))


class PartialRetryTests(unittest.TestCase):
    class MixedClassifier:
        call_count = 0

        def classify(self, *args: object, **kwargs: object) -> IntentPlan:
            del args, kwargs
            self.call_count += 1
            return IntentPlan(
                kind=IntentKind.MIXED,
                tasks=(TaskDraft("任务A"),),
                notes=(NoteDraft("笔记A", "正文"),),
            )

    class FlakyDida:
        def __init__(self) -> None:
            self.calls = 0

        def create_task(self, task: TaskDraft, incoming: MessageEnvelope) -> ActionResult:
            del incoming
            self.calls += 1
            if self.calls == 1:
                return ActionResult("task", ExecutionStatus.FAILED, task.title, error="明确失败")
            ref = TaskReference("task-a", task.title, "Inbox")
            return ActionResult(
                "task",
                ExecutionStatus.SUCCEEDED,
                task.title,
                destination="Inbox",
                external_id=ref.task_id,
                task_refs=(ref,),
            )

    class CountingNotes:
        def __init__(self) -> None:
            self.calls = 0

        def available_links(self, text: str) -> tuple[str, ...]:
            del text
            return ()

        def save(self, note: NoteDraft, incoming: MessageEnvelope) -> ActionResult:
            del incoming
            self.calls += 1
            return ActionResult("note", ExecutionStatus.SUCCEEDED, note.title, "Inbox.md")

    def test_partial_retry_never_replays_failed_external_write(self) -> None:
        dida = self.FlakyDida()
        notes = self.CountingNotes()
        classifier = self.MixedClassifier()
        app, _, _ = service_with(classifier=classifier, dida=dida, obsidian=notes)
        incoming = message("partial-1", "混合消息")
        first = app.handle(incoming)
        second = app.handle(incoming)
        third = app.handle(incoming)
        self.assertEqual(ExecutionStatus.PARTIAL, first.status)
        self.assertEqual(ExecutionStatus.UNCERTAIN, second.status)
        self.assertTrue(third.duplicate)
        self.assertEqual(1, dida.calls)
        self.assertEqual(1, notes.calls)

    def test_partial_retry_preserves_created_task_project_id(self) -> None:
        class SuccessfulDida:
            def __init__(self) -> None:
                self.calls = 0

            def create_task(
                self, task: TaskDraft, incoming: MessageEnvelope
            ) -> ActionResult:
                del incoming
                self.calls += 1
                ref = TaskReference(
                    "task-preserved",
                    task.title,
                    "工作",
                    "project-work",
                    "0",
                )
                return ActionResult(
                    "task",
                    ExecutionStatus.SUCCEEDED,
                    task.title,
                    destination="工作",
                    external_id=ref.task_id,
                    task_refs=(ref,),
                )

        class FlakyNotes(self.CountingNotes):
            def save(
                self, note: NoteDraft, incoming: MessageEnvelope
            ) -> ActionResult:
                del incoming
                self.calls += 1
                if self.calls == 1:
                    return ActionResult(
                        "note", ExecutionStatus.FAILED, note.title, error="本地失败"
                    )
                return ActionResult(
                    "note", ExecutionStatus.SUCCEEDED, note.title, "Inbox.md"
                )

        dida = SuccessfulDida()
        notes = FlakyNotes()
        classifier = self.MixedClassifier()
        ledger = IdempotencyLedger(":memory:")
        app, _, _ = service_with(
            classifier=classifier,
            ledger=ledger,
            dida=dida,
            obsidian=notes,
        )
        incoming = message("partial-preserve", "混合消息")
        first = app.handle(incoming)
        second = app.handle(incoming)
        context = ledger.recent_task_context(incoming.sender_key, NOW)
        self.assertEqual(ExecutionStatus.PARTIAL, first.status)
        self.assertEqual(ExecutionStatus.SUCCEEDED, second.status)
        self.assertEqual(1, dida.calls)
        self.assertEqual(2, notes.calls)
        self.assertEqual("project-work", context.candidates[0].project_id)


class DryRunCliTests(unittest.TestCase):
    def test_identical_uncertain_reply_lines_are_sent_once(self) -> None:
        rendered = format_results(
            (
                ActionResult(
                    "task",
                    ExecutionStatus.UNCERTAIN,
                    "分选CD8",
                    error="滴答返回成功但没有 task_id",
                ),
                ActionResult(
                    "reminder",
                    ExecutionStatus.UNCERTAIN,
                    "分选CD8",
                    error="任务 task_id 未确认",
                ),
            ),
            dry_run=False,
        )
        self.assertEqual(1, rendered.count("这次的结果还需要确认"))
        self.assertEqual(1, rendered.count("没有自动重试"))

    def test_live_mode_replies_only_with_verified_results(self) -> None:
        rendered = format_results(
            (
                ActionResult(
                    "task",
                    ExecutionStatus.SUCCEEDED,
                    "提交报告",
                    destination="Inbox",
                ),
                ActionResult(
                    "note",
                    ExecutionStatus.SUCCEEDED,
                    "产品想法",
                    destination="Inbox/微信收件箱.md",
                ),
            ),
            dry_run=False,
        )
        self.assertEqual(
            "已为你创建好任务：提交报告｜Inbox\n"
            "已为你妥善保存笔记：Inbox/微信收件箱.md",
            rendered,
        )
        self.assertNotIn("准备", rendered)

    def test_live_preview_includes_actions_but_never_private_payload(self) -> None:
        rendered = add_dry_run_previews(
            "Dry Run｜已为你整理好模拟结果",
            (
                ActionResult(
                    "note",
                    ExecutionStatus.PLANNED,
                    "笔记",
                    preview=(
                        "## 笔记\n\n"
                        "> 摘要：一条简短摘要\n\n"
                        "OBSIDIAN-PREVIEW\n\n"
                        "标签：#想法\n\n"
                        "关联：[[年度目标]]"
                    ),
                ),
                ActionResult(
                    "private",
                    ExecutionStatus.PLANNED,
                    "私密内容",
                    preview="PRIVATE-MUST-NEVER-ECHO",
                ),
            ),
        )
        self.assertIn("摘要：一条简短摘要", rendered)
        self.assertIn("标签：#想法", rendered)
        self.assertIn("关联：[[年度目标]]", rendered)
        self.assertNotIn("OBSIDIAN-PREVIEW", rendered)
        self.assertNotIn("PRIVATE-MUST-NEVER-ECHO", rendered)

        private_reply = format_results(
            (
                ActionResult(
                    "private",
                    ExecutionStatus.PLANNED,
                    "私密内容",
                    preview="PRIVATE-MUST-NEVER-ECHO",
                ),
            ),
            dry_run=True,
        )
        self.assertIn("私密内容：模拟本地保存", private_reply)
        self.assertNotIn("PRIVATE-MUST-NEVER-ECHO", private_reply)

    def test_task_preview_is_human_readable_and_hides_tool_parameters(self) -> None:
        rendered = add_dry_run_previews(
            "Dry Run｜已为你整理好模拟结果\n任务：提交报告｜Inbox",
            (
                ActionResult(
                    "task",
                    ExecutionStatus.PLANNED,
                    "提交报告",
                    destination="Inbox",
                    preview=json.dumps(
                        {
                            "task": {
                                "title": "提交报告",
                                "projectId": "inbox",
                                "dueDate": "2026-08-25T15:00:00+08:00",
                                "timeZone": "Asia/Shanghai",
                                "isAllDay": False,
                                "priority": 5,
                            }
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
        )
        self.assertEqual(
            "Dry Run｜已为你整理好模拟结果\n任务：提交报告｜Inbox\n"
            "时间：2026-08-25 15:00｜优先级：高",
            rendered,
        )
        self.assertNotIn("projectId", rendered)
        self.assertNotIn("timeZone", rendered)
        self.assertNotIn("{", rendered)

        uncertain = format_results(
            (
                ActionResult(
                    "complete",
                    ExecutionStatus.UNCERTAIN,
                    "提交报告",
                    error='missing task_id in {"projectId":"inbox"}',
                ),
            ),
            dry_run=False,
        )
        self.assertIn("结果暂时无法确认", uncertain)
        self.assertIn("没有自动重试", uncertain)
        self.assertNotIn("未执行外部写入", uncertain)

    def test_mixed_dry_run_reply_stays_concise(self) -> None:
        task = ActionResult(
            "task",
            ExecutionStatus.PLANNED,
            "整理报销材料",
            destination="Inbox",
            preview=json.dumps(
                {
                    "task": {
                        "title": "整理报销材料",
                        "dueDate": "2026-08-28T10:00:00+08:00",
                        "isAllDay": False,
                    }
                },
                ensure_ascii=False,
            ),
        )
        note = ActionResult(
            "note",
            ExecutionStatus.PLANNED,
            "报销注意事项",
            destination="Inbox/微信收件箱.md",
            preview=(
                "## 报销注意事项\n\n"
                "> 摘要：报销前要先核对发票抬头\n\n"
                "标签：#报销"
            ),
        )
        rendered = add_dry_run_previews(
            format_results((task, note), dry_run=True),
            (task, note),
        )
        self.assertEqual(
            "Dry Run｜已为你整理好模拟结果\n"
            "任务：整理报销材料｜Inbox\n"
            "笔记：Inbox/微信收件箱.md\n"
            "时间：2026-08-28 10:00\n"
            "摘要：报销前要先核对发票抬头\n"
            "标签：#报销",
            rendered,
        )
        self.assertNotIn("整理报销材料：时间", rendered)
        self.assertNotIn("报销注意事项：摘要", rendered)

    def test_failure_reply_hides_internal_tool_fields(self) -> None:
        rendered = format_results(
            (
                ActionResult(
                    "complete",
                    ExecutionStatus.FAILED,
                    "提交报告",
                    error='missing task_id in {"projectId":"inbox"}',
                ),
            ),
            dry_run=True,
        )
        self.assertIn("内部核验未通过，结果暂时无法确认", rendered)
        self.assertNotIn("task_id", rendered)
        self.assertNotIn("projectId", rendered)
        self.assertNotIn("{", rendered)

    def test_fixture_run_never_echoes_private_body(self) -> None:
        args = type(
            "Args",
            (),
            {"fixtures": str(ROOT / "tests" / "fixtures" / "dry_run_messages.json")},
        )()
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = command_dry_run(args)
        rendered = output.getvalue()
        self.assertEqual(0, exit_code)
        self.assertNotIn("DRYRUN-PRIVATE-BODY-NEVER-ECHO", rendered)
        self.assertNotIn("DRYRUN-LATCHED-PRIVATE-BODY-NEVER-ECHO", rendered)
        self.assertIn("Dry Run 完成", rendered)


class ObsidianSafetyTests(unittest.TestCase):
    def test_append_preserves_existing_note_and_is_idempotent(self) -> None:
        vault = test_directory("append")
        target = vault / "Inbox" / "微信收件箱.md"
        target.parent.mkdir(parents=True)
        target.write_text("原有内容\n", encoding="utf-8")
        cfg = settings(
            dry_run=False,
            vault_path=vault,
            obsidian_mapping_confirmed=True,
        )
        executor = ObsidianExecutor(cfg)
        note = NoteDraft("新笔记", "新增正文")
        incoming = message("obs-1", "笔记：新增正文")
        first = executor.save(note, incoming)
        second = executor.save(note, incoming)
        body = target.read_text(encoding="utf-8")
        self.assertEqual(ExecutionStatus.SUCCEEDED, first.status)
        self.assertEqual(ExecutionStatus.SKIPPED, second.status)
        self.assertTrue(body.startswith("原有内容\n"))
        self.assertEqual(1, body.count("新增正文"))

    def test_path_traversal_is_blocked(self) -> None:
        parent = test_directory("traversal")
        vault = parent / "vault"
        vault.mkdir()
        cfg = settings(
            dry_run=False,
            vault_path=vault,
            obsidian_mapping_confirmed=True,
            folder_map={"evil": "../outside.md"},
        )
        executor = ObsidianExecutor(cfg)
        result = executor.save(
            NoteDraft("越界", "正文", target_hint="evil"),
            message("obs-2", "笔记：正文"),
        )
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertFalse((vault.parent / "outside.md").exists())


if __name__ == "__main__":
    unittest.main()
