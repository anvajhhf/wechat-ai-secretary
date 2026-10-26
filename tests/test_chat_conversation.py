"""Offline chat integration: conversational state never authorizes old actions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from wechat_secretary.chat import ChatStore
from wechat_secretary.classifier import HeuristicClassifier
from wechat_secretary.config import SecretarySettings
from wechat_secretary.hermes_plugin import _schedule_reply
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.media import PreparedImage, PreparedMedia
from wechat_secretary.models import (
    ActionResult, ExecutionStatus, IntentKind, IntentPlan, MessageEnvelope,
    NoteDraft, TaskReference,
)
from wechat_secretary.obsidian import ObsidianExecutor
from wechat_secretary.reminders import ReminderQueue
from wechat_secretary.service import SecretaryService
from wechat_secretary.web_reader import WebPage


NOW = datetime(2026, 10, 21, 9, 0, tzinfo=timezone(timedelta(hours=8)))
TEST_TEMP = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"


class FakeChat:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.answer = None

    @property
    def call_count(self):
        return len(self.calls)

    def respond(self, content, history, *, image_inputs=()):
        self.calls.append((content, deepcopy(list(history)), image_inputs))
        if self.failure is not None:
            raise self.failure
        return self.answer or f"这是第 {self.call_count} 次聊天答复。"


class FakeDida:
    def __init__(self):
        self.created = []
        self.completed = []
        self.queries = []
        self.references = []

    def create_task(self, task, message):
        self.created.append(task)
        reference = TaskReference(f"fake-task-{len(self.created)}", task.title, "Inbox", "fake-project")
        self.references.append(reference)
        return ActionResult(
            action="task", status=ExecutionStatus.SUCCEEDED, summary=task.title,
            external_id=reference.task_id, task_refs=(reference,),
        )

    def complete_task(self, reference):
        self.completed.append(reference)
        return ActionResult(action="complete", status=ExecutionStatus.SUCCEEDED, summary=reference.title)

    def search_task_references(self, title):
        return tuple(ref for ref in self.references if title in ref.title)

    def query_tasks(self, query):
        self.queries.append(query)
        return ActionResult(action="query", status=ExecutionStatus.SUCCEEDED, summary="合成任务列表", task_refs=tuple(self.references))


class FakePrivateInbox:
    def __init__(self):
        self.saved = []

    def save(self, message):
        self.saved.append(message)
        return ActionResult(action="private", status=ExecutionStatus.SUCCEEDED, summary="已保存合成私密消息")


class ChatServiceFixture(unittest.TestCase):
    def setUp(self):
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="secretary-chat-test-", dir=TEST_TEMP)
        # The cleanup target is a fresh child of this repository's test area.
        Path(temporary.name).resolve().relative_to(TEST_TEMP.resolve())
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = NOW
        self.message_count = 0
        self.instance_count = 0

    def make_service(self, *, profile="owner", enabled=True, media=None, classifier=None, **changes):
        self.instance_count += 1
        folder = self.root / f"{profile}-{self.instance_count}"
        vault = folder / "vault"
        vault.mkdir(parents=True)
        settings = SecretarySettings(
            project_root=folder, profile_id=profile, dry_run=False,
            account_id=f"synthetic-{profile}", allowed_users=frozenset({"test-user"}),
            reminders_enabled=True, local_only_explicit_reminders=True,
            vault_path=vault, obsidian_mapping_confirmed=True,
            chat_enabled=enabled, **changes,
        )
        ledger = IdempotencyLedger(":memory:")
        self.addCleanup(ledger.close)
        store = ChatStore(folder / "assistant.sqlite3", retention_days=settings.chat_retention_days, clock=lambda: self.now)
        self.addCleanup(store.close)
        chat = FakeChat()
        dida = FakeDida()
        private = FakePrivateInbox()
        service = SecretaryService(
            settings=settings, ledger=ledger,
            classifier=classifier or HeuristicClassifier(settings),
            dida=dida, obsidian=ObsidianExecutor(settings), private_inbox=private,
            reminders=ReminderQueue(settings, ledger), media=media,
            chat=chat, chat_store=store,
        )
        return service, chat, store, dida, private

    def message(self, service, text, *, chat_id="test-chat", message_id=None, **changes):
        self.message_count += 1
        incoming = MessageEnvelope(
            platform="weixin", account_id=service.settings.account_id,
            user_id="test-user", chat_id=chat_id, chat_type="dm",
            message_id=message_id or f"synthetic-message-{self.message_count}",
            text=text, received_at=self.now + timedelta(seconds=self.message_count),
        )
        return replace(incoming, **changes)

    def exchange(self, service, text, *, confirmed=True, **changes):
        incoming = self.message(service, text, **changes)
        result = service.handle(incoming)
        service.record_reply_delivery(incoming, result, confirmed=confirmed)
        return incoming, result

    @staticmethod
    def history(store, incoming):
        return list(store.history(incoming.conversation_key, max_rounds=10, max_chars=12000))

    @staticmethod
    def reminders(service):
        return [tuple(row) for row in service.ledger._connection.execute(
            "SELECT title, reminder_at, status FROM reminders ORDER BY id"
        ).fetchall()]


class ContinuousChatTests(ChatServiceFixture):
    def test_followup_uses_only_confirmed_user_assistant_pairs(self):
        service, chat, store, dida, _ = self.make_service()
        first, reply = self.exchange(service, "最近事情有点多，帮我理理思路")
        self.assertEqual(ExecutionStatus.SUCCEEDED, reply.status)
        self.assertTrue(reply.llm_called)
        self.assertEqual([], chat.calls[0][1])
        self.exchange(service, "刚才第二点再展开说说")
        self.assertEqual([
            {"role": "user", "content": first.text},
            {"role": "assistant", "content": reply.reply},
        ], chat.calls[1][1])
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))
        self.assertFalse((service.settings.vault_path / service.settings.default_note_path).exists())
        self.assertEqual(4, len(self.history(store, first)))

    def test_generated_but_unconfirmed_reply_is_not_conversation_history(self):
        service, chat, store, _, _ = self.make_service()
        incoming = self.message(service, "讨论一下周末休息方式")
        result = service.handle(incoming)
        self.assertEqual([], self.history(store, incoming))
        service.record_reply_delivery(incoming, result, confirmed=False)
        self.exchange(service, "还可以怎么放松")
        self.assertEqual([], chat.calls[1][1])

    def test_duplicate_inbound_and_repeated_delivery_receipt_do_not_duplicate_history(self):
        service, chat, store, _, _ = self.make_service()
        incoming, result = self.exchange(service, "谈谈如何安排休息")
        duplicate = service.handle(incoming)
        service.record_reply_delivery(incoming, duplicate, confirmed=True)
        service.record_reply_delivery(incoming, result, confirmed=True)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(1, chat.call_count)
        self.assertEqual(2, len(self.history(store, incoming)))

    def test_temporary_chat_neither_reads_nor_writes_regular_history(self):
        service, chat, store, _, _ = self.make_service()
        first, _ = self.exchange(service, "我们来讨论读书")
        before = self.history(store, first)
        self.exchange(service, "临时聊天：今天下午三点提醒我买牛奶")
        self.assertEqual("今天下午三点提醒我买牛奶", chat.calls[-1][0])
        self.assertEqual([], chat.calls[-1][1])
        self.assertEqual(before, self.history(store, first))
        self.assertEqual([], self.reminders(service))

    def test_explicit_chat_prefix_does_not_execute_action_words(self):
        service, chat, _, dida, _ = self.make_service()
        for text in ("聊天：明天下午三点提醒我交材料", "聊天：帮我记一下我今天的感受", "聊天：搞定了"):
            self.exchange(service, text)
        self.assertEqual(3, chat.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], dida.completed)
        self.assertEqual([], self.reminders(service))
        self.assertFalse((service.settings.vault_path / service.settings.default_note_path).exists())

    def test_new_topic_removes_old_context_without_calling_model(self):
        service, chat, _, _, _ = self.make_service()
        self.exchange(service, "我最近在学习摄影")
        self.exchange(service, "新话题")
        self.assertEqual(1, chat.call_count)
        self.exchange(service, "聊聊跑步")
        self.assertEqual([], chat.calls[-1][1])

    def test_clear_deletes_history_and_rejects_late_delivery_of_old_answer(self):
        service, chat, store, _, _ = self.make_service()
        first, _ = self.exchange(service, "讨论一下时间安排")
        delayed = self.message(service, "还有哪些角度")
        delayed_result = service.handle(delayed)
        self.exchange(service, "清空这段对话")
        service.record_reply_delivery(delayed, delayed_result, confirmed=True)
        self.assertEqual([], self.history(store, first))
        self.exchange(service, "我们重新开始聊")
        self.assertEqual([], chat.calls[-1][1])

    def test_service_respects_configured_round_limit(self):
        service, chat, _, _, _ = self.make_service(chat_max_rounds=2)
        for index in range(4):
            self.exchange(service, f"聊天：讨论问题 {index}")
        self.assertEqual(4, len(chat.calls[-1][1]))
        self.assertEqual("讨论问题 1", chat.calls[-1][1][0]["content"])

    def test_expired_conversation_is_not_sent_to_model(self):
        service, chat, store, _, _ = self.make_service()
        first, _ = self.exchange(service, "讨论旅行想法")
        self.now += timedelta(days=8)
        self.exchange(service, "我有个新想法")
        self.assertEqual([], chat.calls[-1][1])
        self.assertNotIn(first.text, repr(self.history(store, first)))

    def test_conversations_and_profiles_are_isolated(self):
        owner, owner_chat, _, _, _ = self.make_service(profile="owner")
        partner, partner_chat, _, _, _ = self.make_service(profile="partner")
        self.exchange(owner, "聊天：owner 独有话题", chat_id="shared-chat")
        self.exchange(partner, "聊天：partner 第一个话题", chat_id="shared-chat")
        self.exchange(owner, "聊天：另一个聊天窗口", chat_id="other-chat")
        self.assertEqual([], partner_chat.calls[0][1])
        self.assertEqual([], owner_chat.calls[-1][1])

    def test_chat_failure_does_not_claim_success_or_break_existing_operations(self):
        service, chat, store, _, _ = self.make_service()
        chat.failure = TimeoutError("synthetic backend timeout")
        failed, result = self.exchange(service, "最近想换一种学习方法")
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertEqual([], self.history(store, failed))
        reminder, _ = self.exchange(service, "今天下午三点提醒我买牛奶")
        self.exchange(service, "笔记：本周学习计划先从基础开始")
        self.assertEqual(1, len(self.reminders(service)))
        self.assertEqual("pending", self.reminders(service)[0][2])
        self.assertTrue((service.settings.vault_path / service.settings.default_note_path).exists())
        self.assertEqual(1, chat.call_count)
        self.assertEqual([], self.history(store, reminder))

    def test_chat_storage_failure_does_not_break_explicit_note(self):
        service, chat, store, _, _ = self.make_service()
        with patch.object(store, "history", side_effect=OSError("synthetic store failure")):
            _, failed = self.exchange(service, "聊聊阅读习惯")
            _, note = self.exchange(service, "笔记：阅读时记下三个问题")
        self.assertEqual(ExecutionStatus.FAILED, failed.status)
        self.assertEqual(0, chat.call_count)
        self.assertEqual(ExecutionStatus.SUCCEEDED, note.status)
        self.assertTrue((service.settings.vault_path / service.settings.default_note_path).exists())

    def test_unauthorized_and_missing_message_id_do_not_reach_chat(self):
        service, chat, _, _, _ = self.make_service()
        unauthorized = self.message(service, "普通聊天", user_id="not-allowed")
        missing = replace(self.message(service, "普通聊天"), message_id="")
        self.assertTrue(service.handle(unauthorized).suppressed)
        self.assertEqual(ExecutionStatus.FAILED, service.handle(missing).status)
        self.assertEqual(0, chat.call_count)

    def test_oversized_chat_is_rejected_without_applying_chat_limit_to_notes(self):
        service, chat, store, _, _ = self.make_service(chat_max_input_chars=20)
        incoming, result = self.exchange(service, "聊天：" + "合成聊天内容" * 10)
        self.assertEqual(0, chat.call_count)
        self.assertIn("20", result.reply)
        self.assertEqual([], self.history(store, incoming))
        _, note = self.exchange(service, "笔记：" + "合成笔记正文" * 10)
        self.assertEqual(ExecutionStatus.SUCCEEDED, note.status)
        self.assertTrue((service.settings.vault_path / service.settings.default_note_path).exists())


class ChatCompatibilityTests(ChatServiceFixture):
    def test_explicit_reminder_and_note_keep_original_execution_paths(self):
        service, chat, store, _, _ = self.make_service()
        reminder, result = self.exchange(service, "今天下午三点提醒我买牛奶")
        _, note = self.exchange(service, "笔记：项目讨论结论是先完成基础版本")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(ExecutionStatus.SUCCEEDED, note.status)
        self.assertEqual(0, chat.call_count)
        self.assertEqual([("买牛奶", "2026-10-21T15:00:00+08:00", "pending")], self.reminders(service))
        saved = (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8")
        self.assertIn("项目讨论结论是先完成基础版本", saved)
        self.assertEqual([], self.history(store, reminder))

    def test_incomplete_reminder_can_still_be_completed_without_topic_switch(self):
        service, chat, _, _, _ = self.make_service()
        self.exchange(service, "今天下午提醒我买牛奶")
        self.exchange(service, "三点")
        self.assertEqual(0, chat.call_count)
        self.assertEqual(1, len(self.reminders(service)))
        self.assertEqual("pending", self.reminders(service)[0][2])

    def test_chat_breaks_bare_completion_but_named_completion_still_works(self):
        service, _, _, dida, _ = self.make_service()
        self.exchange(service, "待办：提交报告")
        self.exchange(service, "聊聊怎么缓解紧张")
        self.exchange(service, "搞定了")
        self.assertEqual([], dida.completed)
        self.exchange(service, "完成：提交报告")
        self.assertEqual(["提交报告"], [ref.title for ref in dida.completed])

    def test_new_explicit_task_after_chat_establishes_fresh_completion_context(self):
        service, _, _, dida, _ = self.make_service()
        self.exchange(service, "待办：提交旧报告")
        self.exchange(service, "我们聊聊学习方式")
        self.exchange(service, "待办：提交新报告")
        self.exchange(service, "搞定了")
        self.assertEqual(["提交新报告"], [ref.title for ref in dida.completed])

    def test_chat_breaks_pending_time_clarification_without_creating_reminder(self):
        service, chat, _, dida, _ = self.make_service()
        first, _ = self.exchange(service, "今天下午提醒我买牛奶")
        self.assertIsNotNone(service.ledger.peek_pending_task(first.conversation_key, NOW + timedelta(seconds=30)))
        self.exchange(service, "最近有点累，聊聊休息吧")
        self.exchange(service, "下午三点")
        self.assertGreaterEqual(chat.call_count, 1)
        self.assertEqual([], self.reminders(service))
        self.assertEqual([], dida.created)

    def test_chat_breaks_implicit_reminder_cancellation_but_preserves_schedule(self):
        service, _, _, _, _ = self.make_service()
        self.exchange(service, "今天下午三点提醒我买牛奶")
        before = self.reminders(service)
        self.exchange(service, "我们聊聊阅读")
        self.exchange(service, "取消吧")
        self.assertEqual(before, self.reminders(service))

    def test_clear_chat_does_not_delete_existing_note_or_reminder(self):
        service, _, _, _, _ = self.make_service()
        self.exchange(service, "今天下午三点提醒我买牛奶")
        self.exchange(service, "笔记：保留这条项目结论")
        note_path = service.settings.vault_path / service.settings.default_note_path
        before_note = note_path.read_text(encoding="utf-8")
        before_reminder = self.reminders(service)
        self.exchange(service, "聊天：讨论其他事情")
        self.exchange(service, "清空这段对话")
        self.assertEqual(before_note, note_path.read_text(encoding="utf-8"))
        self.assertEqual(before_reminder, self.reminders(service))

    def test_note_prefix_takes_precedence_over_chat_control_words(self):
        service, chat, store, _, _ = self.make_service()
        first, _ = self.exchange(service, "我们聊聊日常安排")
        before = self.history(store, first)
        self.exchange(service, "笔记：新话题")
        self.exchange(service, "笔记：清空这段对话")
        self.assertEqual(before, self.history(store, first))
        self.assertEqual(1, chat.call_count)
        saved = (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8")
        self.assertIn("新话题", saved)
        self.assertIn("清空这段对话", saved)

    def test_private_text_and_private_next_never_enter_chat(self):
        service, chat, store, _, private = self.make_service()
        first, _ = self.exchange(service, "私密：临时聊天：合成私密正文")
        self.exchange(service, "私密：下一条")
        self.exchange(service, "聊聊另一条合成私密内容")
        self.assertEqual(2, len(private.saved))
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], self.history(store, first))

    def test_voice_chat_uses_local_transcript_and_spoken_private_stays_local(self):
        media = SimpleNamespace(prepare=Mock(return_value=PreparedMedia(transcript_text="我今天有点累，想聊聊")))
        service, chat, store, _, _ = self.make_service(media=media)
        self.exchange(service, "", media_paths=("synthetic.wav",), media_types=("audio/wav",))
        self.assertEqual("我今天有点累，想聊聊", chat.calls[0][0])
        media.prepare.return_value = PreparedMedia(transcript_text="私密：不应进入聊天的合成内容")
        incoming, _ = self.exchange(service, "", media_paths=("synthetic-private.wav",), media_types=("audio/wav",))
        self.assertEqual(1, chat.call_count)
        self.assertNotIn("不应进入聊天", repr(self.history(store, incoming)))

    def test_captioned_image_question_uses_only_prepared_image_in_chat(self):
        prepared = PreparedMedia(images=(PreparedImage(b"synthetic-image", "image/png", "prepared.png", "synthetic-hash"),))
        media = SimpleNamespace(prepare=Mock(return_value=prepared))
        service, chat, store, _, _ = self.make_service(media=media)
        incoming, _ = self.exchange(service, "图上有什么？", media_paths=("synthetic.png",), media_types=("image/png",))
        self.assertEqual(prepared.image_inputs, chat.calls[0][2])
        self.assertNotIn("synthetic-image", repr(self.history(store, incoming)))
        self.assertFalse((service.settings.vault_path / service.settings.default_note_path).exists())

    def test_image_without_caption_keeps_original_note_capture(self):
        prepared = PreparedMedia(images=(PreparedImage(b"synthetic-image", "image/png", "prepared.png", "synthetic-hash"),))
        media = SimpleNamespace(prepare=Mock(return_value=prepared))
        classifier = SimpleNamespace(call_count=0)
        def classify(*args, **kwargs):
            classifier.call_count += 1
            return IntentPlan(kind=IntentKind.NOTE, notes=(NoteDraft("图片记录", "图片中的合成内容"),))
        classifier.classify = classify
        service, chat, _, _, _ = self.make_service(media=media, classifier=classifier)
        _, result = self.exchange(service, "", media_paths=("synthetic.png",), media_types=("image/png",))
        self.assertEqual(0, chat.call_count)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertIn("图片中的合成内容", (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))

    def test_explicit_link_note_still_reads_and_saves_source(self):
        service, chat, _, _, _ = self.make_service()
        service.web = SimpleNamespace(read=Mock(return_value=WebPage(
            source_url="https://example.com/article", final_url="https://example.com/article",
            title="合成网页", text="合成文章正文。",
        )))
        _, result = self.exchange(service, "帮我记一下这个链接：https://example.com/article")
        self.assertEqual(0, chat.call_count)
        service.web.read.assert_called_once_with("https://example.com/article")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertIn("https://example.com/article", (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))

    def test_disabled_chat_keeps_existing_task_completion_behavior(self):
        service, chat, _, dida, _ = self.make_service(enabled=False)
        self.exchange(service, "待办：提交报告")
        self.exchange(service, "最近有点忙")
        self.exchange(service, "搞定了")
        self.assertEqual(0, chat.call_count)
        self.assertEqual(["提交报告"], [ref.title for ref in dida.completed])


class ChatNaturalDiscussionTests(ChatServiceFixture):
    def test_outer_writing_and_analysis_requests_do_not_execute_embedded_temporal_actions(self):
        for text in (
            "帮我写一段明天下午三点开会的通知",
            "帮我分析一下明天提交报告的利弊",
        ):
            with self.subTest(text=text):
                service, chat, _, dida, _ = self.make_service()
                _, result = self.exchange(service, text)
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertEqual(text, chat.calls[0][0])
                self.assertEqual(1, chat.call_count)
                self.assertEqual([], dida.created)
                self.assertEqual([], self.reminders(service))
                self.assertFalse((service.settings.vault_path / service.settings.default_note_path).exists())

    def test_writing_request_ends_old_missing_body_or_time_clarification(self):
        for incomplete in ("今天下午三点提醒我", "今天下午提醒我买牛奶"):
            with self.subTest(incomplete=incomplete):
                service, chat, _, dida, _ = self.make_service()
                first, _ = self.exchange(service, incomplete)
                self.assertIsNotNone(service.ledger.peek_pending_task(first.conversation_key, NOW + timedelta(seconds=30)))
                writing = "帮我写一段明天下午三点开会的通知"
                self.exchange(service, writing)
                self.assertEqual(writing, chat.calls[-1][0])
                self.assertIsNone(service.ledger.peek_pending_task(first.conversation_key, NOW + timedelta(seconds=30)))
                self.exchange(service, "下午三点")
                self.assertEqual([], self.reminders(service))
                self.assertEqual([], dida.created)

    def test_style_followups_continue_writing_with_delivered_history(self):
        service, chat, _, dida, _ = self.make_service()
        first, initial = self.exchange(service, "帮我写一段开场白")
        _, concise = self.exchange(service, "改成更简洁一点")
        self.exchange(service, "改成英文")
        self.assertEqual(3, chat.call_count)
        self.assertEqual([
            {"role": "user", "content": first.text},
            {"role": "assistant", "content": initial.reply},
        ], chat.calls[1][1])
        self.assertEqual("改成更简洁一点", chat.calls[1][0])
        self.assertEqual("改成英文", chat.calls[2][0])
        self.assertEqual({"role": "assistant", "content": concise.reply}, chat.calls[2][1][-1])
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))

    def test_emotional_statement_with_future_meeting_is_chat_not_task(self):
        service, chat, _, dida, _ = self.make_service()
        self.exchange(service, "明天开会有点紧张")
        self.assertEqual(1, chat.call_count)
        self.assertEqual("明天开会有点紧张", chat.calls[0][0])
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))

    def test_explicit_reminders_own_writing_and_analysis_words_in_their_body(self):
        for body in ("写会议通知", "分析报告"):
            with self.subTest(body=body):
                service, chat, _, _, _ = self.make_service()
                _, result = self.exchange(service, f"明天下午三点提醒我{body}")
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertEqual(0, chat.call_count)
                self.assertEqual([(body, "2026-10-22T15:00:00+08:00", "pending")], self.reminders(service))

    def test_explicit_note_owns_analysis_and_temporal_words_in_its_body(self):
        service, chat, _, dida, _ = self.make_service()
        body = "帮我分析一下明天提交报告的利弊"
        _, result = self.exchange(service, f"笔记：{body}")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))
        self.assertIn(body, (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))

    def test_bare_temporal_task_keeps_old_behavior_with_chat_on_or_off(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                service, chat, _, dida, _ = self.make_service(enabled=enabled)
                _, result = self.exchange(service, "明天提交报告")
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertEqual(0, chat.call_count)
                self.assertEqual(1, len(dida.created))
                self.assertEqual("2026-10-22", dida.created[0].due_date)

    def test_style_change_with_active_task_focus_does_not_modify_its_reminder(self):
        service, chat, _, _, _ = self.make_service()
        self.exchange(service, "帮我写一段开场白")
        self.exchange(service, "今天下午三点提醒我买牛奶")
        before = self.reminders(service)
        _, result = self.exchange(service, "改成英文")
        self.assertEqual(1, chat.call_count)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(before, self.reminders(service))

    def test_time_followup_after_writing_discussion_cannot_modify_old_reminder(self):
        service, chat, _, _, _ = self.make_service()
        self.exchange(service, "今天下午三点提醒我买牛奶")
        before = self.reminders(service)
        self.exchange(service, "帮我写一段开场白")
        self.exchange(service, "改成英文")
        _, result = self.exchange(service, "改成五点")
        self.assertEqual(2, chat.call_count)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(before, self.reminders(service))

    def test_existing_link_summary_and_analysis_requests_still_read_and_save_notes(self):
        for text in ("帮我总结一下 https://example.com/article", "分析一下 https://example.com/article"):
            with self.subTest(text=text):
                service, chat, _, dida, _ = self.make_service()
                service.web = SimpleNamespace(read=Mock(return_value=WebPage(
                    source_url="https://example.com/article", final_url="https://example.com/article",
                    title="合成网页", text="讨论后形成的合成文章正文。",
                )))
                _, result = self.exchange(service, text)
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertEqual(0, chat.call_count)
                self.assertEqual([], dida.created)
                service.web.read.assert_called_once_with("https://example.com/article")
                self.assertIn("https://example.com/article", (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))

    def test_explicit_chat_prefix_with_url_does_not_read_web_or_save_note(self):
        for prefix in ("聊天：", "临时聊天："):
            with self.subTest(prefix=prefix):
                service, chat, _, dida, _ = self.make_service()
                service.web = SimpleNamespace(read=Mock(side_effect=AssertionError("must not fetch for explicit chat")))
                content = "帮我总结一下 https://example.com/article"
                _, result = self.exchange(service, prefix + content)
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertEqual(content, chat.calls[0][0])
                self.assertEqual([], dida.created)
                service.web.read.assert_not_called()
                self.assertFalse((service.settings.vault_path / service.settings.default_note_path).exists())

    def test_named_completion_keeps_priority_over_emotional_words_in_task_title(self):
        service, chat, _, dida, _ = self.make_service()
        _, created = self.exchange(service, "待办：整理担心的问题")
        _, completed = self.exchange(service, "完成：整理担心的问题")
        self.assertEqual(ExecutionStatus.SUCCEEDED, created.status)
        self.assertEqual(ExecutionStatus.SUCCEEDED, completed.status)
        self.assertEqual(["整理担心的问题"], [ref.title for ref in dida.completed])
        self.assertEqual(0, chat.call_count)

    def test_existing_task_query_keeps_priority_over_emotional_comment(self):
        service, chat, _, dida, _ = self.make_service()
        _, result = self.exchange(service, "查一下今天有哪些任务，我担心遗漏")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(1, len(dida.queries))
        self.assertEqual("today", dida.queries[0].mode)
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))

    def test_named_reminder_bind_and_cancel_keep_emotional_words_as_task_title(self):
        service, chat, _, dida, _ = self.make_service()
        self.exchange(service, "待办：整理担心的问题")
        dida.exact_active_task_references = Mock(return_value=(dida.references[0],))
        _, bound = self.exchange(service, "补设提醒：今天下午五点｜整理担心的问题")
        self.assertEqual(ExecutionStatus.SUCCEEDED, bound.status)
        dida.exact_active_task_references.assert_called_once_with("整理担心的问题")
        self.assertEqual([("整理担心的问题", "2026-10-21T17:00:00+08:00", "pending")], self.reminders(service))
        _, cancelled = self.exchange(service, "取消整理担心的问题的提醒")
        self.assertEqual(ExecutionStatus.SUCCEEDED, cancelled.status)
        self.assertEqual([("整理担心的问题", "2026-10-21T17:00:00+08:00", "cancelled")], self.reminders(service))
        self.assertEqual(0, chat.call_count)


class ChatDeliveryResultTests(unittest.TestCase):
    def test_only_explicit_boolean_success_confirms_delivery(self):
        event = SimpleNamespace(source=SimpleNamespace(platform="weixin", chat_id="synthetic-chat"), message_id="synthetic-id")
        cases = (
            (SimpleNamespace(success=True), True), ({"success": True}, True),
            (SimpleNamespace(success=False), False), ({"success": False}, False),
            (None, False), ({}, False), (SimpleNamespace(success="true"), False),
            (SimpleNamespace(success=1), False), ({"success": 1}, False),
            ({"success": True, "error": "synthetic error"}, False),
        )
        for response, expected in cases:
            with self.subTest(response=response):
                loop = SimpleNamespace(is_closed=lambda: False)
                adapter = SimpleNamespace(send=Mock(return_value=object()))
                future = Mock()
                future.result.return_value = response
                with patch("wechat_secretary.hermes_plugin.asyncio.run_coroutine_threadsafe", return_value=future):
                    result = _schedule_reply(loop, SimpleNamespace(adapters={"weixin": adapter}), event, "合成回复")
                self.assertIs(expected, result)

    def test_timeout_or_missing_route_does_not_confirm_delivery(self):
        event = SimpleNamespace(source=SimpleNamespace(platform="weixin", chat_id="synthetic-chat"), message_id="synthetic-id")
        loop = SimpleNamespace(is_closed=lambda: False)
        self.assertIs(False, _schedule_reply(loop, SimpleNamespace(adapters={}), event, "合成回复"))
        future = Mock()
        future.result.side_effect = TimeoutError("synthetic delivery timeout")
        adapter = SimpleNamespace(send=Mock(return_value=object()))
        with patch("wechat_secretary.hermes_plugin.asyncio.run_coroutine_threadsafe", return_value=future):
            self.assertIs(False, _schedule_reply(loop, SimpleNamespace(adapters={"weixin": adapter}), event, "合成回复"))


if __name__ == "__main__":
    unittest.main()
