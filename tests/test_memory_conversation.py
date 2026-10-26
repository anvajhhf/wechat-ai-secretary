"""Explicit personal memory must not capture task, note, or private payloads."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_chat_lifecycle import ChatLifecycleFixture
from wechat_secretary.memory import MemoryCommand, MemoryStore
from wechat_secretary.media import PreparedMedia
from wechat_secretary.models import ExecutionStatus
from wechat_secretary.service import SecretaryService


class MemoryRecordingChat:
    def __init__(self):
        self.calls = []
        self.answer = "合成聊天答复。"
        self.before_answer = None

    @property
    def call_count(self):
        return len(self.calls)

    def respond(self, content, history, *, image_inputs=(), memories=()):
        self.calls.append({
            "content": content, "history": deepcopy(list(history)),
            "images": image_inputs, "memories": tuple(memories),
        })
        if self.before_answer is not None:
            self.before_answer()
        return self.answer


class MemoryServiceFixture(ChatLifecycleFixture):
    def make_memory_service(self, *, profile="owner", enabled=True, media=None, **changes):
        old, _, chat_store, dida, private = self.make_service(
            profile=profile, media=media, memory_enabled=enabled, **changes,
        )
        memory_store = MemoryStore(
            Path(chat_store._path), guard_path=old.settings.memory_guard_path,
            clock=lambda: self.now, max_entries=old.settings.memory_max_entries,
            max_entry_chars=old.settings.memory_max_entry_chars,
        )
        self.addCleanup(memory_store.close)
        chat = MemoryRecordingChat()
        service = SecretaryService(
            settings=old.settings, ledger=old.ledger, classifier=old.classifier,
            dida=dida, obsidian=old.obsidian, private_inbox=private,
            reminders=old.reminders, media=old.media, web=old.web,
            chat=chat, chat_store=chat_store, memory_store=memory_store,
        )
        return service, chat, memory_store, chat_store, dida, private

    def seed(self, service, memory_store, text="我喜欢简洁的回答", *, incoming=None, **options):
        incoming = incoming or self.message(service, "synthetic memory source")
        result = memory_store.apply(
            incoming.conversation_key, incoming.message_id,
            MemoryCommand(action="remember", content=text, **options),
        )
        self.assertIn(result.status, {"added", "duplicate"})
        return incoming, result.entry

    def entries(self, memory_store, incoming):
        return tuple(memory_store.list_entries(incoming.conversation_key))

    def memory_contents(self, chat, index=-1):
        return tuple(entry.content for entry in chat.calls[index]["memories"])


class ExplicitMemoryConversationTests(MemoryServiceFixture):
    def test_explicit_remember_and_list_are_local_and_read_back_persisted_fact(self):
        service, chat, memory_store, chat_store, dida, _ = self.make_memory_service()
        incoming, result = self.exchange(service, "记住：以后回答尽量简洁")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        entries = self.entries(memory_store, incoming)
        self.assertEqual(["以后回答尽量简洁"], [entry.content for entry in entries])
        self.assertEqual(incoming.message_id, entries[0].source_message_id)
        _, listed = self.exchange(service, "你记住了我哪些事情？")
        self.assertEqual(ExecutionStatus.SUCCEEDED, listed.status)
        self.assertIn(entries[0].memory_id, listed.reply)
        self.assertIn(entries[0].content, listed.reply)
        self.assertEqual(0, chat.call_count)
        self.assertEqual(0, service.classifier.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))
        self.assertEqual([], self.history(chat_store, incoming))

    def test_plain_statements_and_quoted_memory_commands_do_not_save_or_forget(self):
        cases = (
            "我更喜欢简洁一点的回答",
            "忘记吃药了",
            "帮我解释‘记住：以后回答简洁’这句话",
            "朋友说：忘记关于研究方向的那条记忆",
        )
        for text in cases:
            with self.subTest(text=text):
                service, _, memory_store, _, _, _ = self.make_memory_service()
                source, entry = self.seed(service, memory_store)
                self.exchange(service, text)
                self.assertEqual((entry,), self.entries(memory_store, source))

    def test_remember_payload_does_not_execute_embedded_action_words(self):
        service, chat, memory_store, _, dida, _ = self.make_memory_service()
        content = "我的报告项目叫明天下午三点开会"
        incoming, result = self.exchange(service, "记住：" + content)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual([content], [entry.content for entry in self.entries(memory_store, incoming)])
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))

    def test_existing_note_and_task_prefixes_own_memory_words(self):
        service, chat, memory_store, _, dida, _ = self.make_memory_service()
        note, note_result = self.exchange(service, "笔记：记住：这只是笔记里的示例")
        _, task_result = self.exchange(service, "待办：查看你记住了我哪些事情")
        self.assertEqual(ExecutionStatus.SUCCEEDED, note_result.status)
        self.assertEqual(ExecutionStatus.SUCCEEDED, task_result.status)
        self.assertIn("记住：这只是笔记里的示例", (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))
        self.assertEqual(1, len(dida.created))
        self.assertEqual((), self.entries(memory_store, note))
        self.assertEqual(0, chat.call_count)

    def test_old_remember_task_phrase_and_reminder_payload_keep_task_flow(self):
        for text in ("记着买血压计", "明天下午三点提醒我记住实验步骤"):
            with self.subTest(text=text):
                service, chat, memory_store, _, dida, _ = self.make_memory_service()
                incoming, result = self.exchange(service, text)
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertTrue(dida.created or self.reminders(service))
                self.assertEqual((), self.entries(memory_store, incoming))
                self.assertEqual(0, chat.call_count)

    def test_bare_remember_without_colon_retains_existing_route_instead_of_saving_memory(self):
        observations = []
        for enabled in (False, True):
            service, chat, memory_store, _, dida, _ = self.make_memory_service(enabled=enabled)
            incoming, result = self.exchange(service, "记住明天提交报告")
            self.assertEqual((), self.entries(memory_store, incoming))
            observations.append((result.status, len(dida.created), len(self.reminders(service)), chat.call_count, service.classifier.call_count))
        self.assertEqual(observations[0], observations[1])

    def test_private_text_latch_and_spoken_private_never_read_or_write_memory(self):
        media = SimpleNamespace(prepare=Mock(return_value=PreparedMedia(transcript_text="私密：记住：合成私密事实")))
        service, chat, memory_store, _, _, private = self.make_memory_service(media=media)
        with patch.object(memory_store, "apply", side_effect=AssertionError("private memory write")), patch.object(memory_store, "snapshot", side_effect=AssertionError("private memory read")):
            self.exchange(service, "私密：记住：合成私密事实")
            self.exchange(service, "私密：下一条")
            self.exchange(service, "记住：下一条合成私密事实")
            self.exchange(service, "", media_paths=("synthetic-private.wav",), media_types=("audio/wav",))
        self.assertEqual(2, len(private.saved))
        self.assertEqual(0, chat.call_count)

    def test_explicit_chat_owns_memory_command_text_without_saving_it(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        incoming, _ = self.exchange(service, "聊天：记住：这是聊天中的示例")
        self.assertEqual("记住：这是聊天中的示例", chat.calls[0]["content"])
        self.assertEqual((), self.entries(memory_store, incoming))

    def test_temporary_chat_neither_reads_existing_memories_nor_writes_commands(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, entry = self.seed(service, memory_store)
        with patch.object(memory_store, "snapshot", side_effect=AssertionError("temporary memory read")), patch.object(memory_store, "apply", side_effect=AssertionError("temporary memory write")):
            self.exchange(service, "临时聊天：记住：这条不应保存")
        self.assertEqual(1, chat.call_count)
        self.assertEqual([], chat.calls[0]["history"])
        self.assertEqual((), self.memory_contents(chat))
        self.assertEqual((entry,), self.entries(memory_store, source))
        self.assertEqual([], self.history(chat_store, source))

    def test_memory_management_does_not_consume_old_task_drafts(self):
        for draft in ("今天下午三点提醒我", "今天下午提醒我买牛奶"):
            with self.subTest(draft=draft):
                service, chat, memory_store, _, dida, _ = self.make_memory_service()
                source, _ = self.exchange(service, draft)
                self.assertIsNotNone(service.ledger.peek_pending_task(source.conversation_key, self.now + timedelta(minutes=1)))
                self.exchange(service, "记住：回答使用中文")
                self.assertIsNone(service.ledger.peek_pending_task(source.conversation_key, self.now + timedelta(minutes=1)))
                self.exchange(service, "下午三点")
                self.assertEqual(1, len(self.entries(memory_store, source)))
                self.assertEqual([], dida.created)
                self.assertEqual([], self.reminders(service))

    def test_new_topic_and_clear_chat_preserve_long_term_memory(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, entry = self.seed(service, memory_store, kind="preference")
        self.exchange(service, "你好")
        for control in ("新话题", "清空这段对话"):
            self.exchange(service, control)
            self.assertEqual((entry,), self.entries(memory_store, source))
            self.assertEqual([], self.history(chat_store, source))
            self.exchange(service, "我们继续聊")
            self.assertIn(entry.content, self.memory_contents(chat))

    def test_memory_is_isolated_by_profile_user_and_chat(self):
        owner, chat, memory_store, _, _, _ = self.make_memory_service(profile="owner")
        partner, partner_chat, _, _, _, _ = self.make_memory_service(profile="partner")
        source, entry = self.seed(owner, memory_store)
        owner.settings = replace(owner.settings, allowed_users=frozenset({"test-user", "second-user"}))
        for changes in ({"chat_id": "another-chat"}, {"user_id": "second-user"}):
            self.exchange(owner, "你好", **changes)
            self.assertEqual((), self.memory_contents(chat))
        self.exchange(partner, "你好")
        self.assertEqual((), self.memory_contents(partner_chat))
        self.assertEqual((entry,), self.entries(memory_store, source))

    def test_duplicate_remember_message_creates_only_one_fact(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        incoming, original = self.exchange(service, "记住：沟通时先说结论")
        revision = memory_store.current_revision(incoming.conversation_key)
        duplicate = service.handle(incoming)
        service.record_reply_delivery(incoming, duplicate, confirmed=True)
        self.assertEqual(ExecutionStatus.SUCCEEDED, original.status)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(1, len(self.entries(memory_store, incoming)))
        self.assertEqual(revision, memory_store.current_revision(incoming.conversation_key))
        self.assertEqual(0, chat.call_count)

    def test_unconfirmed_memory_receipt_does_not_undo_committed_fact(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        incoming, result = self.exchange(service, "记住：偏好先给简短答案", confirmed=False)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(1, len(self.entries(memory_store, incoming)))
        _, listing = self.exchange(service, "你记住了我哪些事情？")
        self.assertIn("偏好先给简短答案", listing.reply)
        self.assertEqual(0, chat.call_count)

    def test_empty_memory_command_does_not_claim_saved_fact(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        incoming, result = self.exchange(service, "记住：")
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual((), self.entries(memory_store, incoming))
        self.assertEqual(0, chat.call_count)

    def test_correction_by_id_and_natural_forget_do_not_call_model(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        incoming, _ = self.exchange(service, "记住 研究方向：合成旧方向")
        original = self.entries(memory_store, incoming)[0]
        _, corrected = self.exchange(service, f"更正记忆 {original.memory_id}：合成新方向")
        entries = self.entries(memory_store, incoming)
        self.assertEqual(ExecutionStatus.SUCCEEDED, corrected.status)
        self.assertEqual(1, len(entries))
        self.assertEqual(original.memory_id, entries[0].memory_id)
        self.assertEqual("合成新方向", entries[0].content)
        self.assertGreater(entries[0].version, original.version)
        _, forgotten = self.exchange(service, "忘记关于研究方向的那条记忆")
        self.assertEqual(ExecutionStatus.SUCCEEDED, forgotten.status)
        self.assertEqual((), self.entries(memory_store, incoming))
        self.assertEqual(0, chat.call_count)

    def test_another_user_cannot_correct_or_forget_fact_using_its_id(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        source, original = self.seed(service, memory_store)
        service.settings = replace(service.settings, allowed_users=frozenset({"test-user", "second-user"}))
        for text in (f"更正记忆 {original.memory_id}：越界替换", f"忘记记忆 {original.memory_id}"):
            _, result = self.exchange(service, text, user_id="second-user")
            self.assertIn("没有找到", result.reply)
            self.assertEqual((original,), self.entries(memory_store, source))
        self.assertEqual(0, chat.call_count)

    def test_forget_clears_derived_chat_but_preserves_existing_note_and_reminder(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        self.exchange(service, "笔记：合成实验原始记录")
        self.exchange(service, "明天下午三点提醒我整理实验记录")
        reminder_before = self.reminders(service)
        note_path = service.settings.vault_path / service.settings.default_note_path
        note_before = note_path.read_text(encoding="utf-8")
        source, entry = self.seed(service, memory_store, "我的合成代号是蓝竹")
        chat.answer = "蓝竹，建议先整理资料。"
        self.exchange(service, "聊天：按我的代号给一个建议")
        self.assertIn(entry.content, self.memory_contents(chat))
        chat.answer = "延续蓝竹的安排。"
        self.exchange(service, "聊天：再详细一点")
        self.assertTrue(self.history(chat_store, source))
        self.exchange(service, f"忘记记忆 {entry.memory_id}")
        self.assertEqual([], self.history(chat_store, source))
        chat.answer = "从当前问题开始。"
        self.exchange(service, "聊天：继续给我建议")
        self.assertEqual([], chat.calls[-1]["history"])
        self.assertEqual((), self.memory_contents(chat))
        self.assertNotIn("蓝竹", repr(chat.calls[-1]))
        self.assertEqual(reminder_before, self.reminders(service))
        self.assertEqual(note_before, note_path.read_text(encoding="utf-8"))

    def test_correction_invalidates_old_derived_history_before_next_prompt(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, original = self.seed(service, memory_store, "合成称呼是青灯")
        chat.answer = "青灯，你可以先休息。"
        self.exchange(service, "聊天：聊聊我的称呼")
        self.assertIn(original.content, self.memory_contents(chat))
        self.exchange(service, f"更正记忆 {original.memory_id}：合成称呼是白帆")
        self.assertEqual([], self.history(chat_store, source))
        self.exchange(service, "聊天：从当前称呼聊起")
        self.assertEqual([], chat.calls[-1]["history"])
        self.assertEqual(("合成称呼是白帆",), self.memory_contents(chat))
        self.assertNotIn("青灯", repr(chat.calls[-1]))

    def test_disabled_memory_allows_listing_and_forgetting_but_not_saving(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        source, entry = self.seed(service, memory_store)
        service.settings = replace(service.settings, memory_enabled=False)
        _, refused = self.exchange(service, "记住：这条不能新增")
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, refused.status)
        self.assertEqual((entry,), self.entries(memory_store, source))
        _, listing = self.exchange(service, "查看记忆")
        self.assertIn(entry.memory_id, listing.reply)
        self.exchange(service, f"忘记记忆 {entry.memory_id}")
        self.assertEqual((), self.entries(memory_store, source))
        self.assertEqual(0, chat.call_count)

    def test_image_caption_memory_command_is_not_saved_as_a_personal_fact(self):
        media = SimpleNamespace(prepare=Mock(return_value=PreparedMedia(images=("synthetic-image",))))
        service, chat, memory_store, _, _, _ = self.make_memory_service(media=media)
        source, result = self.exchange(service, "记住：这张图里的信息", media_paths=("synthetic.png",), media_types=("image/png",))
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual((), self.entries(memory_store, source))
        self.assertEqual(0, chat.call_count)
