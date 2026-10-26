"""Offline result-context and source-validation races for note retrieval."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from test_notes_conversation import NotesServiceFixture
from wechat_secretary.models import ExecutionStatus, TaskReference
from wechat_secretary.notes_search import NotesSnapshot
from wechat_secretary.service import SecretaryService


class NotesContextLifecycleTests(NotesServiceFixture):
    def test_unconfirmed_search_result_never_activates_followup_numbers(self):
        service, _, answer, _, _, _, _, _, _ = self.make_notes_service()
        self.search(service, confirmed=False)
        self.exchange(service, "笔记问答：总结这些方案")
        self.assertEqual(0, answer.call_count)

    def test_late_old_search_receipt_cannot_replace_new_result_context(self):
        service, index, answer, _, _, _, _, _, _ = self.make_notes_service()
        first_snapshot = NotesSnapshot(index.revision, (index.hits[0],))
        second_snapshot = NotesSnapshot(index.revision, (index.hits[1],))
        with patch.object(index, "search", side_effect=(first_snapshot, second_snapshot)):
            first = self.message(service, "搜索笔记：第一组")
            old_result = service.handle(first)
            _, new_result = self.search(service)
        self.assertNotEqual(old_result.notes_context_id, new_result.notes_context_id)
        service.record_reply_delivery(first, old_result, confirmed=True)
        self.exchange(service, "笔记问答：当前结果讲了什么")
        self.assertEqual((index.hits[1],), answer.calls[-1][1])

    def test_topic_and_non_notes_operations_clear_result_focus(self):
        operations = (
            "新话题", "清空这段对话", "聊天：聊聊散步", "临时聊天：聊聊散步",
            "记住偏好：回答尽量简洁", "待办：整理实验资料",
            "笔记：合成独立笔记", "私密：合成私密内容",
        )
        for operation in operations:
            with self.subTest(operation=operation):
                service, _, answer, _, _, _, _, _, _ = self.make_notes_service(memory_enabled=True)
                self.search(service)
                self.exchange(service, operation)
                self.exchange(service, "笔记问答：总结第1条")
                self.assertEqual(0, answer.call_count)

    def test_result_ttl_uses_clock_not_message_timestamp(self):
        service, _, answer, _, _, _, _, _, _ = self.make_notes_service(notes_context_ttl_seconds=10)
        with patch("wechat_secretary.notes_service.time.monotonic", return_value=100.0):
            self.search(service)
        with patch("wechat_secretary.notes_service.time.monotonic", return_value=111.0):
            # An old receipt timestamp does not extend the live result lease.
            self.exchange(service, "笔记问答：总结第1条", received_at=self.now)
        self.assertEqual(0, answer.call_count)

    def test_service_restart_does_not_restore_old_result_numbers(self):
        old, index, answer, chat, memory, chat_store, dida, private, notify = self.make_notes_service()
        self.search(old)
        restarted = SecretaryService(
            settings=old.settings, ledger=old.ledger, classifier=old.classifier,
            dida=dida, obsidian=old.obsidian, private_inbox=private,
            reminders=old.reminders, media=old.media, web=old.web,
            chat=chat, chat_store=chat_store, memory_store=memory,
            notes_index=index, notes_answer=answer, notes_notify=notify,
        )
        self.exchange(restarted, "笔记问答：总结第1条")
        self.assertEqual(0, answer.call_count)

    def test_deleted_changed_or_excluded_source_invalidates_result_before_answering(self):
        for mutation in ("delete", "change", "exclude"):
            with self.subTest(mutation=mutation):
                service, index, answer, _, _, _, _, _, _ = self.make_notes_service()
                self.search(service)
                if mutation == "delete":
                    index.change_sources(())
                elif mutation == "change":
                    index.change_sources((replace(index.hits[0], excerpt="合成已修改正文", version="new-version"),))
                else:
                    index.exclude_source(index.hits[0].source_id)
                self.exchange(service, "笔记问答：总结这些方案")
                self.assertEqual(0, answer.call_count)

    def test_source_change_during_generation_blocks_answer_at_delivery(self):
        service, index, answer, _, _, chat_store, _, _, _ = self.make_notes_service()
        source, _ = self.search(service)
        answer.answer = "合成旧片段派生的回答。[1]"
        answer.before_answer = lambda: index.change_sources(())
        incoming = self.message(service, "笔记问答：比较这些方案")
        result = service.handle(incoming)
        delivery = service.prepare_reply_delivery(incoming, result)
        self.assertNotIn("合成旧片段派生", delivery.reply)
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, delivery.status)
        service.record_reply_delivery(incoming, result, confirmed=True)
        self.assertEqual([], self.history(chat_store, source))

    def test_changed_source_after_generation_is_not_sent_or_used_by_late_receipt(self):
        service, index, answer, _, _, chat_store, _, _, _ = self.make_notes_service()
        source, _ = self.search(service)
        answer.answer = "合成旧结论。[1]"
        incoming = self.message(service, "笔记问答：比较这些方案")
        old_result = service.handle(incoming)
        index.change_sources(())
        delivery = service.prepare_reply_delivery(incoming, old_result)
        self.assertNotIn("合成旧结论", delivery.reply)
        service.record_reply_delivery(incoming, old_result, confirmed=True)
        self.exchange(service, "笔记问答：继续刚才的比较")
        self.assertEqual(1, answer.call_count)
        self.assertEqual([], self.history(chat_store, source))

    def test_memory_change_after_note_answer_generation_invalidates_its_context(self):
        service, _, answer, _, _, _, _, _, _ = self.make_notes_service(memory_enabled=True)
        self.search(service)
        answer.answer = "合成旧问答。[1]"
        incoming = self.message(service, "笔记问答：总结一下")
        result = service.handle(incoming)
        self.exchange(service, "记住偏好：回答先给结论")
        delivery = service.prepare_reply_delivery(incoming, result)
        self.assertNotIn("合成旧问答", delivery.reply)
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, delivery.status)

    def test_source_validation_fault_after_confirmed_send_clears_anonymous_reminder_focus(self):
        service, index, _, _, _, chat_store, dida, _, _ = self.make_notes_service()
        self.search(service)
        incoming = self.message(service, "笔记问答：总结这些方案")
        result = service.handle(incoming)
        observed_at = datetime.now(service.settings.tz)
        reference = TaskReference("synthetic-interleaved-note-task", "合成提醒", "Inbox", "synthetic-project")
        dida.references.append(reference)
        service.ledger.record_task_context(
            incoming.conversation_key, (reference,), batch_id="synthetic-note-race",
            source_message_id="synthetic-reminder", observed_at=observed_at,
            ttl_seconds=3600, context_kind="reminder-delivery",
        )
        index.validation_error = OSError("synthetic-note-control-read")
        service.record_reply_delivery(incoming, result, confirmed=True)
        self.exchange(service, "搞定了", received_at=datetime.now(service.settings.tz) + timedelta(seconds=1))
        self.assertEqual([], dida.completed)
        self.assertEqual([], self.history(chat_store, incoming))


class NotesFailureIsolationTests(NotesServiceFixture):
    def test_failed_answer_is_terminal_for_same_message_but_new_id_can_retry(self):
        service, _, answer, _, _, _, _, _, _ = self.make_notes_service()
        self.search(service)
        incoming = self.message(service, "笔记问答：总结这些方案")
        answer.error = RuntimeError("synthetic-answer-failure")
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertEqual(1, answer.call_count)
        duplicate = service.handle(incoming)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(1, answer.call_count)
        answer.error = None
        # Failure retires the result focus. An explicit new search and new
        # question are permitted, without replaying the original message.
        self.search(service)
        _, retried = self.exchange(service, "笔记问答：总结这些方案")
        self.assertEqual(ExecutionStatus.SUCCEEDED, retried.status)
        self.assertEqual(2, answer.call_count)

    def test_failed_search_is_terminal_for_same_message_but_new_id_can_retry(self):
        service, index, answer, chat, _, _, _, _, _ = self.make_notes_service()
        incoming = self.message(service, "搜索笔记：实验方案")
        index.search_error = OSError("synthetic-index-read")
        result = service.handle(incoming)
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertEqual(1, len(index.search_calls))
        duplicate = service.handle(incoming)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(1, len(index.search_calls))
        index.search_error = None
        _, retried = self.search(service)
        self.assertEqual(ExecutionStatus.SUCCEEDED, retried.status)
        self.assertEqual(2, len(index.search_calls))
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)

    def test_disabled_search_never_reads_index_or_calls_a_model(self):
        service, index, answer, chat, _, _, _, _, _ = self.make_notes_service(enabled=False)
        with patch.object(index, "search", side_effect=AssertionError("disabled index read")):
            self.search(service)
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)

    def test_search_failure_preserves_note_save_and_reminder_and_never_falls_back_to_chat(self):
        service, index, answer, chat, _, _, _, _, notify = self.make_notes_service()
        index.search_error = OSError("synthetic-index-read")
        _, result = self.search(service)
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)
        _, note = self.exchange(service, "笔记：故障期间的合成记录")
        _, reminder = self.exchange(service, "明天下午三点提醒我整理实验数据")
        self.assertEqual(ExecutionStatus.SUCCEEDED, note.status)
        self.assertEqual(ExecutionStatus.SUCCEEDED, reminder.status)
        self.assertEqual(1, len(self.reminders(service)))
        notify.assert_called_once_with()

    def test_answer_failure_does_not_call_general_chat_or_store_note_content(self):
        service, _, answer, chat, _, chat_store, _, _, _ = self.make_notes_service()
        source, _ = self.search(service)
        answer.error = RuntimeError("synthetic-answer-failure")
        _, result = self.exchange(service, "笔记问答：总结这些方案")
        self.assertNotEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(1, answer.call_count)
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], self.history(chat_store, source))

    def test_index_refresh_notification_failure_cannot_turn_successful_note_into_failure(self):
        service, _, _, _, _, _, _, _, notify = self.make_notes_service()
        notify.side_effect = OSError("synthetic-refresh-notify")
        _, result = self.exchange(service, "笔记：已保存的合成笔记")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        notify.assert_called_once_with()
        body = (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8")
        self.assertIn("已保存的合成笔记", body)

    def test_failed_or_dry_run_note_does_not_notify_index_as_a_successful_write(self):
        for failed in (True, False):
            with self.subTest(failed=failed):
                service, _, _, _, _, _, _, _, notify = self.make_notes_service()
                if failed:
                    service.obsidian.save = Mock(side_effect=OSError("synthetic-note-save"))
                else:
                    service.settings = replace(service.settings, dry_run=True)
                    service.obsidian.settings = service.settings
                self.exchange(service, "笔记：未实际写入的合成记录")
                notify.assert_not_called()
