"""Offline configuration, initialization and delivery-bookkeeping boundaries."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import os
import shutil
from unittest.mock import Mock, patch
from uuid import uuid4

from test_chat_conversation import ChatServiceFixture, NOW, TEST_TEMP
from wechat_secretary.config import SecretarySettings
from wechat_secretary.hermes_plugin import FailClosedBridge, GatewayBridge, GatewayReadyBridge
from wechat_secretary import hermes_plugin
from wechat_secretary.models import ExecutionStatus, TaskReference


class ChatLifecycleFixture(ChatServiceFixture):
    def setUp(self):
        # Python 3.13 TemporaryDirectory can deny the Windows restricted token
        # strict path resolution; use an owned directory with inherited ACLs.
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        self.root = (TEST_TEMP / f"chat-lifecycle-{uuid4().hex}").resolve()
        self.root.mkdir()
        self.now = NOW
        self.message_count = 0
        self.instance_count = 0
        self.addCleanup(self.clean_fixture)

    def clean_fixture(self):
        if self.root.parent != TEST_TEMP.resolve() or not self.root.name.startswith("chat-lifecycle-"):
            raise AssertionError("refusing to clean outside the owned test fixture")
        shutil.rmtree(self.root)


class ChatConfigurationTests(ChatLifecycleFixture):
    def load(self, text):
        path = self.root / "secretary.toml"
        path.write_text(text, encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            return SecretarySettings.from_file(path, self.root)

    def test_existing_configuration_keeps_chat_disabled_and_bounded_defaults(self):
        settings = self.load('[secretary]\nprofile_id = "owner"\n')
        self.assertFalse(settings.chat_enabled)
        self.assertEqual(7, settings.chat_retention_days)
        self.assertEqual(10, settings.chat_max_rounds)
        self.assertEqual(12000, settings.chat_context_max_chars)
        self.assertEqual(6000, settings.chat_max_input_chars)
        self.assertEqual(4000, settings.chat_max_reply_chars)
        self.assertEqual(1200, settings.chat_max_tokens)

    def test_chat_can_be_enabled_without_changing_reminder_or_note_settings(self):
        settings = self.load(
            '[secretary]\nprofile_id = "partner"\n'
            '[chat]\nenabled = true\nretention_days = 3\nmax_rounds = 4\n'
            '[reminders]\nenabled = true\npoll_seconds = 25\n'
            '[obsidian]\ndefault_note_path = "Inbox/Existing.md"\n'
        )
        self.assertTrue(settings.chat_enabled)
        self.assertEqual((3, 4), (settings.chat_retention_days, settings.chat_max_rounds))
        self.assertTrue(settings.reminders_enabled)
        self.assertEqual(25, settings.reminder_poll_seconds)
        self.assertEqual("Inbox/Existing.md", settings.default_note_path)

    def test_chat_numeric_options_are_bounded(self):
        settings = self.load(
            '[chat]\nretention_days = -1\nmax_rounds = 999\n'
            'context_max_chars = 1\nmax_input_chars = 999999\n'
            'max_reply_chars = 1\nmax_tokens = 99999\n'
        )
        self.assertEqual(1, settings.chat_retention_days)
        self.assertEqual(20, settings.chat_max_rounds)
        self.assertEqual(4000, settings.chat_context_max_chars)
        self.assertEqual(12000, settings.chat_max_input_chars)
        self.assertEqual(500, settings.chat_max_reply_chars)
        self.assertEqual(3000, settings.chat_max_tokens)

    def test_assistant_database_is_distinct_and_profile_scoped(self):
        owner = self.load('[secretary]\nprofile_id = "owner"\n')
        partner = replace(owner, profile_id="partner")
        self.assertEqual(self.root / "runtime" / "state" / "owner" / "assistant.sqlite3", owner.assistant_db_path)
        self.assertEqual(self.root / "runtime" / "state" / "partner" / "assistant.sqlite3", partner.assistant_db_path)
        self.assertNotEqual(owner.assistant_db_path, owner.state_db_path)
        self.assertNotEqual(owner.assistant_db_path, partner.assistant_db_path)


class ChatRegistrationTests(ChatLifecycleFixture):
    def settings(self, *, enabled=True):
        return SecretarySettings(
            project_root=self.root, dry_run=True, allowed_users=frozenset({"test-user"}),
            chat_enabled=enabled,
        )

    def register(self, settings, *, store_failure=None, retention_failure=None):
        ctx = Mock()
        mocks = {
            "load_settings": Mock(return_value=settings),
            "IdempotencyLedger": Mock(),
            "ChatStore": Mock(side_effect=store_failure),
            "HermesChatResponder": Mock(),
            "DidaExecutor": Mock(),
            "ObsidianExecutor": Mock(),
            "PrivateInboxExecutor": Mock(),
            "LocalMediaPreprocessor": Mock(),
            "SafeWebReader": Mock(),
            "ReminderScheduler": Mock(),
            "_start_chat_retention": Mock(side_effect=retention_failure),
        }
        with patch.dict(os.environ, {}, clear=True), patch.multiple(hermes_plugin, **mocks):
            hermes_plugin.register(ctx)
        hooks = {call.args[0]: call.args[1] for call in ctx.register_hook.call_args_list}
        return ctx, mocks, hooks

    def assert_regular_hooks(self, hooks):
        self.assertIsInstance(hooks["pre_gateway_dispatch"], GatewayBridge)
        self.assertNotIsInstance(hooks["pre_gateway_dispatch"], FailClosedBridge)
        self.assertIsInstance(hooks["gateway_ready"], GatewayReadyBridge)

    def test_enabled_chat_registers_existing_features_and_new_responder(self):
        settings = self.settings()
        ctx, mocks, hooks = self.register(settings)
        self.assert_regular_hooks(hooks)
        mocks["ChatStore"].assert_called_once_with(settings.assistant_db_path, settings.chat_retention_days)
        mocks["HermesChatResponder"].assert_called_once_with(ctx, settings)
        mocks["_start_chat_retention"].assert_called_once_with(mocks["ChatStore"].return_value)
        mocks["ReminderScheduler"].assert_called_once_with(settings, mocks["IdempotencyLedger"].return_value)
        names = {call.kwargs["name"] for call in ctx.register_tool.call_args_list}
        self.assertEqual({"secretary_morning_digest", "secretary_evening_review", "secretary_dida_taxonomy"}, names)

    def test_chat_database_failure_does_not_disable_existing_secretary(self):
        with self.assertLogs("wechat_secretary.hermes_plugin", level="ERROR"):
            _, mocks, hooks = self.register(self.settings(), store_failure=OSError("synthetic-chat-db"))
        self.assert_regular_hooks(hooks)
        service = hooks["pre_gateway_dispatch"].service
        self.assertIsNone(service.chat)
        self.assertIsNone(service.chat_store)
        mocks["ReminderScheduler"].assert_called_once()
        mocks["DidaExecutor"].assert_called_once()
        mocks["ObsidianExecutor"].assert_called_once()
        mocks["HermesChatResponder"].assert_not_called()
        mocks["_start_chat_retention"].assert_not_called()

    def test_retention_thread_failure_is_isolated_from_gateway_registration(self):
        with self.assertLogs("wechat_secretary.hermes_plugin", level="WARNING"):
            ctx, mocks, hooks = self.register(self.settings(), retention_failure=RuntimeError("synthetic-thread"))
        self.assert_regular_hooks(hooks)
        self.assertIs(hooks["pre_gateway_dispatch"].service.chat_store, mocks["ChatStore"].return_value)
        self.assertEqual(3, ctx.register_tool.call_count)
        self.assertEqual(2, ctx.register_hook.call_count)

    def test_disabled_chat_without_database_creates_no_chat_storage_or_worker(self):
        _, mocks, hooks = self.register(self.settings(enabled=False))
        self.assert_regular_hooks(hooks)
        mocks["ChatStore"].assert_not_called()
        mocks["HermesChatResponder"].assert_not_called()
        mocks["_start_chat_retention"].assert_not_called()

    def test_disabled_chat_still_opens_existing_database_for_retention(self):
        settings = self.settings(enabled=False)
        settings.assistant_db_path.parent.mkdir(parents=True)
        # A synthetic presence marker only; the constructor is mocked.
        settings.assistant_db_path.touch()
        _, mocks, hooks = self.register(settings)
        self.assert_regular_hooks(hooks)
        mocks["ChatStore"].assert_called_once_with(settings.assistant_db_path, settings.chat_retention_days)
        mocks["_start_chat_retention"].assert_called_once_with(mocks["ChatStore"].return_value)
        mocks["HermesChatResponder"].assert_not_called()
        self.assertIsNone(hooks["pre_gateway_dispatch"].service.chat)

    def test_invalid_profile_fails_before_any_database_or_scheduler_is_opened(self):
        settings = replace(self.settings(), profile_id="../outside")
        with self.assertLogs("wechat_secretary.hermes_plugin", level="ERROR"):
            _, mocks, hooks = self.register(settings)
        self.assertIsInstance(hooks["pre_gateway_dispatch"], FailClosedBridge)
        mocks["IdempotencyLedger"].assert_not_called()
        mocks["ChatStore"].assert_not_called()
        mocks["ReminderScheduler"].assert_not_called()


class ChatDeliveryLifecycleTests(ChatLifecycleFixture):
    def seed_context(self, service, message):
        now = datetime.now(service.settings.tz)
        reference = TaskReference("synthetic-task", "合成任务", "Inbox", "synthetic-project")
        service.ledger.record_task_context(
            message.conversation_key, (reference,), batch_id="synthetic-batch",
            source_message_id="synthetic-source", observed_at=now,
            ttl_seconds=3600, context_kind="reminder",
        )
        self.assertEqual((reference,), service.ledger.recent_task_context(message.conversation_key, now + timedelta(seconds=1)).candidates)
        return now, reference

    def test_generation_lookup_failure_still_invalidates_task_focus_without_saving_text(self):
        service, _, store, _, _ = self.make_service()
        incoming = self.message(service, "我们聊聊周末安排")
        result = service.handle(incoming)
        now, _ = self.seed_context(service, incoming)
        with patch.object(store, "get_generation", side_effect=OSError("synthetic-store-read")), patch.object(store, "record_sent") as record:
            with self.assertLogs("wechat_secretary.service", level="WARNING"):
                service.record_reply_delivery(incoming, result, confirmed=True)
            record.assert_not_called()
        self.assertEqual((), service.ledger.recent_task_context(incoming.conversation_key, now + timedelta(seconds=1)).candidates)
        self.assertEqual([], self.history(store, incoming))

    def test_known_old_generation_receipt_does_not_invalidate_new_topic_focus(self):
        service, _, store, _, _ = self.make_service()
        incoming = self.message(service, "讨论一下休息方式")
        result = service.handle(incoming)
        store.reset(incoming.conversation_key, clear=True)
        now, reference = self.seed_context(service, incoming)
        service.record_reply_delivery(incoming, result, confirmed=True)
        self.assertEqual((reference,), service.ledger.recent_task_context(incoming.conversation_key, now + timedelta(seconds=1)).candidates)
        self.assertEqual([], self.history(store, incoming))

    def test_disabled_chat_can_clear_saved_conversation_without_calling_model(self):
        service, chat, store, _, _ = self.make_service(enabled=False)
        incoming = self.message(service, "清空这段对话")
        store.record_sent(incoming.conversation_key, store.get_generation(incoming.conversation_key), "旧聊天", "旧回答")
        result = service.handle(incoming)
        service.record_reply_delivery(incoming, result, confirmed=True)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertIn("已清空", result.reply)
        self.assertEqual([], self.history(store, incoming))
        self.assertEqual(0, chat.call_count)

    def test_disabled_chat_ordinary_message_neither_reads_nor_calls_chat(self):
        service, chat, store, dida, _ = self.make_service(enabled=False)
        with patch.object(store, "history") as history:
            self.exchange(service, "我今天心情不错，想聊聊读书")
            history.assert_not_called()
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], self.reminders(service))

    def test_model_failure_is_terminal_for_same_id_but_new_message_can_retry(self):
        service, chat, store, dida, _ = self.make_service()
        incoming = self.message(service, "聊天：讨论一下休息方式")
        chat.failure = RuntimeError("synthetic-model-failure")
        with self.assertLogs("wechat_secretary.service", level="WARNING"):
            failed = service.handle(incoming)
        self.assertEqual(ExecutionStatus.FAILED, failed.status)
        self.assertEqual(1, chat.call_count)
        chat.failure = None
        duplicate = service.handle(incoming)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(1, chat.call_count)
        self.assertEqual([], self.history(store, incoming))
        _, retried = self.exchange(service, incoming.text)
        self.assertEqual(ExecutionStatus.SUCCEEDED, retried.status)
        self.assertEqual(2, chat.call_count)
        self.assertEqual([], dida.created)

    def test_history_failure_cannot_automatically_resume_generation_for_same_id(self):
        service, chat, store, _, _ = self.make_service()
        incoming = self.message(service, "聊天：继续讨论读书安排")
        with patch.object(store, "history", side_effect=OSError("synthetic-history-failure")):
            with self.assertLogs("wechat_secretary.service", level="WARNING"):
                failed = service.handle(incoming)
        self.assertEqual(ExecutionStatus.FAILED, failed.status)
        self.assertEqual(0, chat.call_count)
        duplicate = service.handle(incoming)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(0, chat.call_count)
        _, retried = self.exchange(service, incoming.text)
        self.assertEqual(ExecutionStatus.SUCCEEDED, retried.status)
        self.assertEqual(1, chat.call_count)

    def test_non_chat_failed_message_keeps_existing_retry_contract(self):
        service, _, _, _, _ = self.make_service()
        incoming = self.message(service, "笔记：合成记录")
        self.assertTrue(service.ledger.claim(incoming).is_new)
        service.ledger.finish(incoming, ExecutionStatus.FAILED, error_code="synthetic-note-write-failure")
        retried = service.ledger.claim(incoming)
        self.assertTrue(retried.is_new)
        self.assertTrue(retried.retrying_failed_operations)

    def pending_recovery(self, service):
        incoming = self.message(service, "聊天：讨论一下周末安排")
        result = service.handle(incoming)
        now, reference = self.seed_context(service, incoming)
        service.ledger.enqueue_reminders(incoming, reference, (now + timedelta(hours=1),))
        with patch.object(service.ledger, "reset_conversation_focus", side_effect=OSError("synthetic-focus-write")):
            with self.assertLogs("wechat_secretary.service", level="WARNING"):
                service.record_reply_delivery(incoming, result, confirmed=True)
        self.assertIn(incoming.conversation_key, service._focus_recovery)
        return incoming, now, reference

    def test_failed_delivery_focus_reset_recovers_before_next_completion(self):
        service, _, store, dida, _ = self.make_service()
        incoming, now, reference = self.pending_recovery(service)
        before = service.ledger.reminder_snapshot(reference.task_id, incoming)
        next_message = self.message(service, "搞定了", received_at=now + timedelta(seconds=2))
        result = service.handle(next_message)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertNotIn(incoming.conversation_key, service._focus_recovery)
        self.assertEqual([], dida.completed)
        self.assertEqual(before, service.ledger.reminder_snapshot(reference.task_id, incoming))
        self.assertEqual([], self.history(store, incoming))

    def test_persistent_focus_failure_blocks_only_affected_nonprivate_conversation(self):
        service, chat, _, dida, private = self.make_service()
        incoming, now, reference = self.pending_recovery(service)
        before = service.ledger.reminder_snapshot(reference.task_id, incoming)
        original_reset = service.ledger.reset_conversation_focus

        def failed_only_here(key, reset_at):
            if key == incoming.conversation_key:
                raise OSError("synthetic-persistent-focus-write")
            return original_reset(key, reset_at)

        with patch.object(service.ledger, "reset_conversation_focus", side_effect=failed_only_here):
            blocked = self.message(service, "搞定了", received_at=now + timedelta(seconds=2))
            failure = service.handle(blocked)
            self.assertEqual(ExecutionStatus.FAILED, failure.status)
            self.assertIn("没有执行", failure.reply)
            blocked_note = service.handle(self.message(service, "笔记：这个会话暂时不能保存"))
            self.assertEqual(ExecutionStatus.FAILED, blocked_note.status)
            private_result = service.handle(self.message(service, "私密：这段仍只走私密保存"))
            self.assertEqual(ExecutionStatus.SUCCEEDED, private_result.status)
            self.assertEqual(1, len(private.saved))
            _, other_result = self.exchange(service, "聊天：另一个会话正常讨论", chat_id="unaffected-chat")
            self.assertEqual(ExecutionStatus.SUCCEEDED, other_result.status)
            _, other_note = self.exchange(service, "笔记：另一会话独立记录", chat_id="unaffected-chat")
            self.assertEqual(ExecutionStatus.SUCCEEDED, other_note.status)
        self.assertEqual([], dida.completed)
        self.assertEqual(before, service.ledger.reminder_snapshot(reference.task_id, incoming))
        self.assertEqual(2, chat.call_count)
        self.assertIn(incoming.conversation_key, service._focus_recovery)
        note = (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8")
        self.assertIn("另一会话独立记录", note)
        self.assertNotIn("这个会话暂时不能保存", note)
