"""Offline memory failures and delayed replies must not revive old facts."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import os
from unittest.mock import Mock, patch

from test_memory_conversation import MemoryServiceFixture
from wechat_secretary import hermes_plugin
from wechat_secretary.config import SecretarySettings
from wechat_secretary.hermes_plugin import FailClosedBridge, GatewayBridge, GatewayReadyBridge
from wechat_secretary.memory import MemoryCommand
from wechat_secretary.models import ExecutionStatus, TaskReference


class MemoryReplyLifecycleTests(MemoryServiceFixture):
    def forget(self, memory_store, incoming, entry):
        return memory_store.apply(
            incoming.conversation_key, f"synthetic-forget-{self.message_count}",
            MemoryCommand(action="forget", target=entry.memory_id),
        )

    def test_forget_during_generation_prevents_old_reply_and_late_history(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, entry = self.seed(service, memory_store, "合成昵称是红杉")
        chat.answer = "红杉，这是根据旧资料生成的回复。"
        chat.before_answer = lambda: self.forget(memory_store, source, entry)
        incoming = self.message(service, "聊天：说说我的昵称")
        old_reply = service.handle(incoming)
        self.assertTrue(old_reply.memory_guarded)
        self.assertIn(entry.content, self.memory_contents(chat))
        delivery = service.prepare_reply_delivery(incoming, old_reply)
        self.assertEqual(ExecutionStatus.SKIPPED, delivery.status)
        self.assertNotIn("红杉", delivery.reply)
        self.assertEqual(1, chat.call_count)
        service.record_reply_delivery(incoming, old_reply, confirmed=True)
        self.assertEqual([], self.history(chat_store, source))

    def test_correction_after_generation_before_delivery_replaces_old_answer(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, entry = self.seed(service, memory_store, "合成方向是旧主题")
        chat.answer = "旧主题值得继续。"
        incoming = self.message(service, "聊天：接着聊我的方向")
        old_reply = service.handle(incoming)
        outcome = memory_store.apply(
            source.conversation_key, "synthetic-correction",
            MemoryCommand(action="correct", target=entry.memory_id, content="合成方向是新主题"),
        )
        self.assertEqual("updated", outcome.status)
        delivery = service.prepare_reply_delivery(incoming, old_reply)
        self.assertIsNot(delivery, old_reply)
        self.assertNotIn("旧主题", delivery.reply)
        service.record_reply_delivery(incoming, old_reply, confirmed=True)
        self.assertEqual([], self.history(chat_store, source))
        self.exchange(service, "聊天：从当前方向聊起")
        self.assertEqual([], chat.calls[-1]["history"])
        self.assertEqual(("合成方向是新主题",), self.memory_contents(chat))

    def test_late_receipt_of_forgotten_reply_cannot_replace_fresh_conversation(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, entry = self.seed(service, memory_store, "合成旧偏好是绿色")
        chat.answer = "按绿色的偏好回答。"
        incoming = self.message(service, "聊天：聊聊我的偏好")
        old_reply = service.handle(incoming)
        self.forget(memory_store, source, entry)
        chat.answer = "这是新的聊天回答。"
        self.exchange(service, "聊天：现在说点新的")
        fresh_history = self.history(chat_store, source)
        self.assertTrue(fresh_history)
        service.record_reply_delivery(incoming, old_reply, confirmed=True)
        self.assertEqual(fresh_history, self.history(chat_store, source))
        self.assertNotIn("绿色", repr(fresh_history))

    def test_expiry_at_send_validation_clears_derived_history_and_blocks_old_answer(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, _ = self.seed(service, memory_store, "合成短期项目是紫桥", expires_at=self.now + timedelta(minutes=1))
        chat.answer = "继续紫桥项目。"
        self.exchange(service, "聊天：给项目一点建议")
        incoming = self.message(service, "聊天：再延续这个建议")
        old_reply = service.handle(incoming)
        self.now += timedelta(minutes=2)
        delivery = service.prepare_reply_delivery(incoming, old_reply)
        self.assertEqual(ExecutionStatus.SKIPPED, delivery.status)
        self.assertNotIn("紫桥", delivery.reply)
        service.record_reply_delivery(incoming, old_reply, confirmed=True)
        self.assertEqual((), self.entries(memory_store, source))
        self.assertEqual([], self.history(chat_store, source))

    def test_revision_read_failure_never_sends_or_records_unverified_answer(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, _ = self.seed(service, memory_store, "合成代号是深海")
        chat.answer = "深海，我建议休息。"
        incoming = self.message(service, "聊天：给代号一个解释")
        result = service.handle(incoming)
        with patch.object(memory_store, "is_current", side_effect=OSError("synthetic-guard-fault")):
            with self.assertLogs("wechat_secretary.service", level="WARNING"):
                delivery = service.prepare_reply_delivery(incoming, result)
                service.record_reply_delivery(incoming, result, confirmed=True)
        self.assertEqual(ExecutionStatus.SKIPPED, delivery.status)
        self.assertNotIn("深海", delivery.reply)
        self.assertEqual([], self.history(chat_store, source))

    def test_guard_read_failure_after_delivery_still_clears_new_reminder_focus(self):
        service, _, memory_store, chat_store, dida, _ = self.make_memory_service()
        source, _ = self.seed(service, memory_store)
        incoming = self.message(service, "聊天：聊聊今天")
        result = service.handle(incoming)
        # While adapter I/O is in flight, a reminder worker publishes context.
        observed_at = datetime.now(service.settings.tz)
        reference = TaskReference("synthetic-interleaved-task", "合成送达提醒", "Inbox", "synthetic-project")
        dida.references.append(reference)
        service.ledger.record_task_context(
            incoming.conversation_key, (reference,), batch_id="synthetic-reminder-batch",
            source_message_id="synthetic-reminder", observed_at=observed_at,
            ttl_seconds=3600, context_kind="reminder-delivery",
        )
        self.assertEqual((reference,), service.ledger.recent_task_context(incoming.conversation_key, observed_at).candidates)
        with patch.object(memory_store, "is_current", side_effect=OSError("synthetic-receipt-guard-read")):
            with self.assertLogs("wechat_secretary.service", level="WARNING"):
                service.record_reply_delivery(incoming, result, confirmed=True)
        self.assertEqual([], self.history(chat_store, source))
        _, followup = self.exchange(service, "搞定了", received_at=datetime.now(service.settings.tz) + timedelta(seconds=1))
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, followup.status)
        self.assertEqual([], dida.completed)

    def test_unconfirmed_personalized_reply_is_not_available_as_followup_history(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service()
        source, _ = self.seed(service, memory_store)
        self.exchange(service, "聊天：先聊聊", confirmed=False)
        self.assertEqual([], self.history(chat_store, source))
        self.exchange(service, "聊天：再聊聊")
        self.assertEqual([], chat.calls[-1]["history"])
        self.assertEqual(2, len(self.history(chat_store, source)))

    def test_memory_management_reply_is_also_revalidated_after_later_forget(self):
        service, chat, memory_store, _, _, _ = self.make_memory_service()
        incoming = self.message(service, "记住：合成保密代号是风铃")
        saved = service.handle(incoming)
        entry = self.entries(memory_store, incoming)[0]
        self.forget(memory_store, incoming, entry)
        delivery = service.prepare_reply_delivery(incoming, saved)
        self.assertEqual(ExecutionStatus.SKIPPED, delivery.status)
        self.assertNotIn("风铃", delivery.reply)
        self.assertEqual(0, chat.call_count)


class MemoryFailureIsolationTests(MemoryServiceFixture):
    def test_snapshot_failure_uses_current_input_only_and_keeps_note_reminder_available(self):
        service, chat, memory_store, chat_store, dida, _ = self.make_memory_service()
        source, _ = self.seed(service, memory_store, "合成旧信息是松塔")
        chat.answer = "按松塔的旧信息回复。"
        self.exchange(service, "聊天：第一轮")
        old_history = self.history(chat_store, source)
        chat.answer = "只基于当前消息回答。"
        with patch.object(memory_store, "snapshot", side_effect=OSError("synthetic-memory-read")):
            with self.assertLogs("wechat_secretary.service", level="WARNING"):
                _, result = self.exchange(service, "聊天：我当前想聊散步")
            self.exchange(service, "笔记：合成故障期间记录")
            self.exchange(service, "明天下午三点提醒我整理资料")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertIn("只根据这条消息", result.reply)
        self.assertEqual([], chat.calls[-1]["history"])
        self.assertEqual((), self.memory_contents(chat))
        self.assertEqual("我当前想聊散步", chat.calls[-1]["content"])
        self.assertEqual(old_history, self.history(chat_store, source))
        self.assertEqual(1, len(self.reminders(service)))
        self.assertEqual([], dida.created)
        self.assertIn("合成故障期间记录", (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))

    def test_unavailable_initialization_flag_discards_history_even_when_memory_disabled(self):
        service, chat, _, chat_store, _, _ = self.make_memory_service()
        incoming = self.message(service, "聊天：这轮只看当前输入")
        chat_store.record_sent(incoming.conversation_key, chat_store.get_generation(incoming.conversation_key), "合成旧问句", "合成旧答案")
        service.settings = replace(service.settings, memory_enabled=False)
        service.memory_store = None
        service.memory_unavailable = True
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual([], chat.calls[0]["history"])
        self.assertEqual((), self.memory_contents(chat))
        self.assertEqual("", result.chat_user_text)

    def test_missing_memory_module_degrades_chat_without_claiming_save_success(self):
        service, chat, _, _, _, _ = self.make_memory_service()
        service.memory_store = None
        _, saved = self.exchange(service, "记住：不能确认的事实")
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, saved.status)
        with self.assertLogs("wechat_secretary.service", level="WARNING"):
            _, result = self.exchange(service, "聊天：聊聊今天")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual([], chat.calls[-1]["history"])
        self.assertEqual((), self.memory_contents(chat))

    def test_memory_write_failure_does_not_claim_success_or_break_existing_tasks(self):
        service, chat, memory_store, _, dida, _ = self.make_memory_service()
        incoming = self.message(service, "记住：失败不能冒充保存")
        with patch.object(memory_store, "apply", side_effect=OSError("synthetic-memory-write")):
            with self.assertLogs("wechat_secretary.service", level="WARNING"):
                result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertIn("没有确认", result.reply)
        self.assertEqual((), self.entries(memory_store, incoming))
        self.assertEqual(0, chat.call_count)
        _, task = self.exchange(service, "待办：完成合成回归检查")
        self.assertEqual(ExecutionStatus.SUCCEEDED, task.status)
        self.assertEqual(1, len(dida.created))

    def test_disabled_guard_failure_degrades_to_current_input_only(self):
        service, chat, memory_store, chat_store, _, _ = self.make_memory_service(enabled=False)
        incoming = self.message(service, "聊天：当前的问题")
        chat_store.record_sent(incoming.conversation_key, chat_store.get_generation(incoming.conversation_key), "合成旧问句", "合成旧答案")
        with patch.object(memory_store, "current_revision", side_effect=OSError("synthetic-control-read")):
            result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual([], chat.calls[-1]["history"])
        self.assertEqual((), self.memory_contents(chat))
        self.assertEqual("", result.chat_user_text)


class MemoryConfigurationAndRegistrationTests(MemoryServiceFixture):
    def load_settings(self, text):
        path = self.root / "synthetic-secretary.toml"
        path.write_text(text, encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            return SecretarySettings.from_file(path, self.root)

    def test_memory_defaults_disabled_with_bounded_storage_and_context(self):
        settings = self.load_settings('[secretary]\nprofile_id = "owner"\n')
        self.assertFalse(settings.memory_enabled)
        self.assertEqual((100, 500, 6, 2000), (
            settings.memory_max_entries, settings.memory_max_entry_chars,
            settings.memory_context_max_items, settings.memory_context_max_chars,
        ))

    def test_memory_options_do_not_change_note_reminder_or_chat_settings(self):
        settings = self.load_settings(
            '[memory]\nenabled = true\nmax_entries = 12\nmax_entry_chars = 200\n'
            'context_max_items = 3\ncontext_max_chars = 700\n'
            '[chat]\nenabled = true\nmax_rounds = 4\n'
            '[reminders]\nenabled = true\npoll_seconds = 25\n'
            '[obsidian]\ndefault_note_path = "Inbox/Existing.md"\n'
        )
        self.assertTrue(settings.memory_enabled)
        self.assertEqual((12, 200, 3, 700), (
            settings.memory_max_entries, settings.memory_max_entry_chars,
            settings.memory_context_max_items, settings.memory_context_max_chars,
        ))
        self.assertTrue(settings.chat_enabled)
        self.assertEqual(4, settings.chat_max_rounds)
        self.assertTrue(settings.reminders_enabled)
        self.assertEqual(25, settings.reminder_poll_seconds)
        self.assertEqual("Inbox/Existing.md", settings.default_note_path)

    def test_guard_file_is_outside_backup_database_directory_and_profile_scoped(self):
        owner = self.load_settings('[secretary]\nprofile_id = "owner"\n')
        partner = replace(owner, profile_id="partner")
        self.assertNotEqual(owner.memory_guard_path, partner.memory_guard_path)
        self.assertNotEqual(owner.memory_guard_path.parent, owner.assistant_db_path.parent)
        owner.memory_guard_path.relative_to(self.root)

    def test_memory_initialization_failure_keeps_existing_gateway_and_reminder_scheduler(self):
        settings = SecretarySettings(
            project_root=self.root, dry_run=True, allowed_users=frozenset({"test-user"}),
            chat_enabled=True, memory_enabled=True,
        )
        ctx = Mock()
        mocks = {
            "load_settings": Mock(return_value=settings),
            "IdempotencyLedger": Mock(), "ChatStore": Mock(),
            "MemoryStore": Mock(side_effect=OSError("synthetic-memory-init")),
            "HermesChatResponder": Mock(), "DidaExecutor": Mock(),
            "ObsidianExecutor": Mock(), "PrivateInboxExecutor": Mock(),
            "LocalMediaPreprocessor": Mock(), "SafeWebReader": Mock(),
            "ReminderScheduler": Mock(), "_start_chat_retention": Mock(),
        }
        with patch.dict(os.environ, {}, clear=True), patch.multiple(hermes_plugin, **mocks):
            with self.assertLogs("wechat_secretary.hermes_plugin", level="ERROR"):
                hermes_plugin.register(ctx)
        hooks = {call.args[0]: call.args[1] for call in ctx.register_hook.call_args_list}
        self.assertIsInstance(hooks["pre_gateway_dispatch"], GatewayBridge)
        self.assertNotIsInstance(hooks["pre_gateway_dispatch"], FailClosedBridge)
        self.assertIsInstance(hooks["gateway_ready"], GatewayReadyBridge)
        service = hooks["pre_gateway_dispatch"].service
        self.assertTrue(service.memory_unavailable)
        self.assertIsNone(service.memory_store)
        self.assertIs(service.chat_store, mocks["ChatStore"].return_value)
        self.assertIs(service.chat, mocks["HermesChatResponder"].return_value)
        mocks["ReminderScheduler"].assert_called_once()
        mocks["DidaExecutor"].assert_called_once()
        mocks["ObsidianExecutor"].assert_called_once()

    def test_disabled_memory_still_checks_existing_state_when_guard_is_missing(self):
        old, chat, chat_store, dida, private = self.make_service(memory_enabled=False)
        settings = replace(old.settings, dry_run=True)
        self.assertFalse(settings.memory_enabled)
        self.assertFalse(settings.memory_guard_path.exists())
        incoming = self.message(old, "你好，我们聊聊散步")
        chat_store.record_sent(
            incoming.conversation_key, chat_store.get_generation(incoming.conversation_key),
            "合成旧记忆问句", "合成旧记忆派生回答", memory_context=True,
        )
        ctx = Mock()
        mocks = {
            "load_settings": Mock(return_value=settings),
            "IdempotencyLedger": Mock(return_value=old.ledger),
            "ChatStore": Mock(return_value=chat_store),
            # The storage-level probe has separate SQLite tests. This test
            # fixes its positive result at the registration branch boundary.
            "has_memory_state": Mock(return_value=True),
            "MemoryStore": Mock(side_effect=OSError("synthetic-missing-guard")),
            "HermesChatResponder": Mock(return_value=chat),
            "DidaExecutor": Mock(return_value=dida),
            "ObsidianExecutor": Mock(return_value=old.obsidian),
            "PrivateInboxExecutor": Mock(return_value=private),
            "LocalMediaPreprocessor": Mock(return_value=old.media),
            "SafeWebReader": Mock(return_value=old.web),
            "ReminderScheduler": Mock(), "_start_chat_retention": Mock(),
        }
        with patch.dict(os.environ, {}, clear=True), patch.multiple(hermes_plugin, **mocks):
            with self.assertLogs("wechat_secretary.hermes_plugin", level="ERROR"):
                hermes_plugin.register(ctx)
        hooks = {call.args[0]: call.args[1] for call in ctx.register_hook.call_args_list}
        service = hooks["pre_gateway_dispatch"].service
        mocks["has_memory_state"].assert_called_once_with(settings.assistant_db_path)
        mocks["MemoryStore"].assert_called_once_with(
            settings.assistant_db_path, guard_path=settings.memory_guard_path,
            max_entries=settings.memory_max_entries, max_entry_chars=settings.memory_max_entry_chars,
        )
        self.assertTrue(service.memory_unavailable)
        self.assertIsNone(service.memory_store)
        with patch.object(chat_store, "history", wraps=chat_store.history) as read_history:
            result = service.handle(incoming)
            service.record_reply_delivery(incoming, result, confirmed=True)
            read_history.assert_not_called()
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual([], chat.calls[-1][1])
        self.assertEqual("", result.chat_user_text)
        self.assertIn("只根据这条消息", result.reply)
        self.assertEqual([], dida.created)
