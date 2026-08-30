"""Adversarial lifecycle regressions; no real services or persisted user state."""
from __future__ import annotations

import unittest
import os
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import test_voice_reminder_conversation as fixtures
from wechat_secretary.models import ActionResult, ExecutionStatus, TaskReference
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.dida import DidaExecutor
from wechat_secretary.reminders import (
    ReminderDeliveryPreSendError, ReminderDeliveryUncertainError, ReminderScheduler,
)


NOW = fixtures.NOW


class LifecycleAuditRound2Tests(unittest.TestCase):
    make_service = fixtures.VoiceReminderConversationTests.make_service
    message = staticmethod(fixtures.VoiceReminderConversationTests.message)

    @staticmethod
    def record_completion(dida):
        completed = []

        def complete(task):
            completed.append(task.task_id)
            return ActionResult("complete", ExecutionStatus.PLANNED, task.title)

        dida.complete_task = complete
        return completed

    def test_recent_completion_cannot_cross_chat(self):
        service, _, media, dida, ledger = self.make_service()
        source = self.message("audit-create", "今天下午三点提醒我买牛奶", media, voice=False)
        service.handle(source)
        completed = self.record_completion(dida)
        other = replace(
            self.message("audit-complete-other", "搞定了", media, voice=False, minutes_later=1),
            chat_id="another-private-chat",
        )

        result = service.handle(other)

        self.assertEqual([], completed)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(1, ledger.active_reminder_count("voice-task-1", source))

    def test_numbered_completion_cannot_cross_chat(self):
        service, _, media, dida, ledger = self.make_service()
        source = self.message("audit-choice", "完成：报告", media, voice=False)
        refs = (TaskReference("report-a", "报告A"), TaskReference("report-b", "报告B"))
        dida.search_task_references = lambda title: refs
        service.handle(source)
        completed = self.record_completion(dida)
        other = replace(
            self.message("audit-choice-other", "完成 1", media, voice=False, minutes_later=1),
            chat_id="another-private-chat",
        )

        result = service.handle(other)

        self.assertEqual([], completed)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_legacy_sender_only_context_never_enables_recent_completion(self):
        service, _, media, dida, ledger = self.make_service()
        source = self.message("audit-legacy", "", media, voice=False)
        ledger.record_task_context(
            source.sender_key, (TaskReference("legacy-task", "旧会话未标明的任务"),),
            batch_id="legacy", source_message_id="legacy", observed_at=NOW,
            ttl_seconds=3600, context_kind="task-create",
        )
        completed = self.record_completion(dida)

        service.handle(self.message("audit-no-legacy-fallback", "搞定了", media, voice=False))

        self.assertEqual([], completed)

    def test_numbered_completion_works_in_original_chat_after_other_chat_attempt(self):
        service, _, media, dida, _ = self.make_service()
        refs = (TaskReference("report-a", "报告A"), TaskReference("report-b", "报告B"))
        dida.search_task_references = lambda title: refs
        service.handle(self.message("audit-choice-local", "完成：报告", media, voice=False))
        completed = self.record_completion(dida)
        other = replace(
            self.message("audit-choice-wrong-chat", "完成 1", media, voice=False, minutes_later=1),
            chat_id="another-private-chat",
        )
        service.handle(other)

        result = service.handle(self.message("audit-choice-right-chat", "完成 2", media, voice=False, minutes_later=2))

        self.assertEqual(["report-b"], completed)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)

    def test_selection_older_than_current_list_does_not_complete_any_task(self):
        service, _, media, dida, _ = self.make_service()
        refs = (TaskReference("report-a", "报告A"), TaskReference("report-b", "报告B"))
        dida.search_task_references = lambda title: refs
        service.handle(self.message("audit-newer-choice", "完成：报告", media, voice=False, minutes_later=2))
        completed = self.record_completion(dida)

        result = service.handle(self.message("audit-stale-choice", "完成 1", media, voice=False, minutes_later=1))

        self.assertEqual([], completed)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_older_list_request_cannot_replace_newer_completion_choices(self):
        service, _, media, dida, ledger = self.make_service()
        report = TaskReference("report-a", "报告A")
        budget = TaskReference("budget-a", "预算A")
        dida.search_task_references = lambda title: (report,) if title == "报告" else (budget,)
        newer = self.message("audit-budget-choice", "完成：预算", media, voice=False, minutes_later=2)
        service.handle(newer)
        completed = self.record_completion(dida)
        older = self.message("audit-old-report-choice", "完成：报告", media, voice=False, minutes_later=1)

        result = service.handle(older)

        self.assertIn("早于", result.reply)
        self.assertEqual((budget,), ledger.pending_completion(newer.conversation_key, NOW + timedelta(minutes=3)))
        service.handle(self.message("audit-final-budget-choice", "完成 1", media, voice=False, minutes_later=3))
        self.assertEqual(["budget-a"], completed)

    def test_older_adjustment_cannot_target_a_task_created_after_it(self):
        service, _, media, dida, ledger = self.make_service()
        first = self.message("audit-old-task", "今天下午三点提醒我买牛奶", media, voice=False)
        second = self.message("audit-new-task", "今天下午四点提醒我买咖啡", media, voice=False, minutes_later=2)
        service.handle(first)
        service.handle(second)
        before_first = ledger.reminder_snapshot("voice-task-1", first)
        before_second = ledger.reminder_snapshot("voice-task-2", second)
        late = self.message("audit-late-update", "改成五点", media, voice=False, minutes_later=1)

        result = service.handle(late)

        self.assertEqual(before_first, ledger.reminder_snapshot("voice-task-1", first))
        self.assertEqual(before_second, ledger.reminder_snapshot("voice-task-2", second))
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(2, len(dida.tasks))

    def test_older_relative_adjustment_or_completion_cannot_target_future_task(self):
        for control in ("半小时后提醒我", "搞定了"):
            with self.subTest(control=control):
                service, _, media, dida, ledger = self.make_service()
                source = self.message("audit-future-create", "今天下午四点提醒我买咖啡", media, voice=False, minutes_later=2)
                service.handle(source)
                before = ledger.reminder_snapshot("voice-task-1", source)
                completed = self.record_completion(dida)

                result = service.handle(self.message("audit-future-control", control, media, voice=False, minutes_later=1))

                self.assertEqual([], completed)
                self.assertEqual(before, ledger.reminder_snapshot("voice-task-1", source))
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_stale_named_completion_cannot_bypass_local_time_guard_via_remote_search(self):
        for has_prior_creation in (False, True):
            with self.subTest(has_prior_creation=has_prior_creation):
                service, _, media, dida, ledger = self.make_service()
                source = self.message("audit-named-watermark-create", "今天下午三点提醒我买牛奶", media, voice=False, minutes_later=0 if has_prior_creation else 2)
                service.handle(source)
                if has_prior_creation:
                    service.handle(self.message("audit-named-watermark-update", "改成四点", media, voice=False, minutes_later=2))
                calls = []
                ref = TaskReference("voice-task-1", "买牛奶")
                dida.search_task_references = lambda title: calls.append(title) or (ref,)
                completed = self.record_completion(dida)
                before = ledger.reminder_snapshot(ref.task_id, source)

                result = service.handle(self.message("audit-named-watermark-complete", "完成：买牛奶", media, voice=False, minutes_later=1))

                self.assertEqual([], completed)
                self.assertEqual([], calls)
                self.assertEqual(before, ledger.reminder_snapshot(ref.task_id, source))
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_unknown_named_completion_still_allows_exact_remote_lookup(self):
        service, _, media, dida, _ = self.make_service()
        service.handle(self.message("audit-unrelated-newer-task", "今天下午三点提醒我买牛奶", media, voice=False, minutes_later=2))
        calls = []
        ref = TaskReference("remote-report", "远程报告")
        dida.search_task_references = lambda title: calls.append(title) or (ref,)
        completed = self.record_completion(dida)

        result = service.handle(self.message("audit-unknown-remote-complete", "完成：远程报告", media, voice=False, minutes_later=1))

        self.assertEqual(["远程报告"], calls)
        self.assertEqual([ref.task_id], completed)
        self.assertEqual(ExecutionStatus.PLANNED, result.status)

    def test_unverified_creation_fields_stay_uncertain_without_retry_or_reminder(self):
        service, _, media, _, ledger = self.make_service()
        calls = []
        payload = {}

        def caller(server, tool, arguments, timeout):
            calls.append(tool)
            if tool == "create_task":
                payload.update(arguments["task"])
            # Return the correct identity but deliberately omit requested
            # priority. An ID alone is not evidence the requested task is ready.
            return {"ok": True, "structuredContent": {
                "id": "contract-task", "title": payload["title"],
                "projectId": payload["projectId"], "status": 0,
            }}

        settings = replace(fixtures.SETTINGS, dry_run=False, dida_mapping_confirmed=True, dida_schema_confirmed=True)
        service.dida = DidaExecutor(settings, caller)
        incoming = self.message("audit-unverified-fields", "今天下午三点提醒我提交报告，高优先级", media, voice=False)
        with patch.dict(os.environ, {"SECRETARY_DIDA_CREATES_APPROVED": "1"}):
            first = service.handle(incoming)
            replay = service.handle(incoming)

        self.assertEqual(5, payload["priority"])
        self.assertEqual(ExecutionStatus.UNCERTAIN, first.status)
        self.assertEqual(ExecutionStatus.UNCERTAIN, replay.status)
        self.assertTrue(replay.duplicate)
        self.assertEqual(["create_task", "get_task_by_id"], calls)
        self.assertEqual((), ledger.reminder_snapshot("contract-task", incoming))
        self.assertEqual((), ledger.recent_task_context(incoming.conversation_key, incoming.received_at).candidates)

    def test_private_protection_is_hashed_persistent_and_generation_guarded(self):
        _, _, media, _, ledger = self.make_service()
        source = self.message("audit-private-generation", "", media, voice=False)
        self.assertEqual("", ledger.get_private_protection(source.sender_key))
        ledger.set_private_protection(source.sender_key, "generation-one")
        ledger._initialize()
        self.assertEqual("generation-one", ledger.get_private_protection(source.sender_key))
        self.assertEqual("", ledger.get_private_protection("different-sender"))
        ledger.set_private_protection(source.sender_key, "generation-two")
        self.assertFalse(ledger.clear_private_protection(source.sender_key, "generation-one"))
        self.assertFalse(ledger.clear_private_protection(source.sender_key, ""))
        self.assertEqual("generation-two", ledger.get_private_protection(source.sender_key))
        row = ledger._connection.execute("SELECT sender_hash, token FROM private_ingress_protection").fetchone()
        self.assertEqual(IdempotencyLedger._hash(source.sender_key), row["sender_hash"])
        self.assertNotIn(source.user_id, row["sender_hash"])
        self.assertTrue(ledger.clear_private_protection(source.sender_key, "generation-two"))
        self.assertEqual("", ledger.get_private_protection(source.sender_key))

    def test_older_cancel_cannot_undo_a_newer_successful_update(self):
        service, _, media, _, ledger = self.make_service()
        source = self.message("audit-watermark-create", "今天下午三点提醒我买牛奶", media, voice=False)
        service.handle(source)
        service.handle(self.message("audit-watermark-update", "改成四点", media, voice=False, minutes_later=2))
        before = ledger.reminder_snapshot("voice-task-1", source)

        result = service.handle(self.message("audit-watermark-stale-cancel", "刚才那个不要了", media, voice=False, minutes_later=1))

        self.assertEqual(before, ledger.reminder_snapshot("voice-task-1", source))
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_successful_reminder_controls_advance_the_context_watermark(self):
        pairs = (
            ("半小时后提醒我", "刚才那个不要了"),
            ("再提醒三次，每隔20分钟", "取消整个系列提醒"),
            ("刚才那个不要了", "半小时后提醒我"),
            ("改成四点", "取消买牛奶的提醒"),
        )
        for current_control, stale_control in pairs:
            with self.subTest(current=current_control, stale=stale_control):
                service, _, media, _, ledger = self.make_service()
                source = self.message("audit-control-create", "今天下午三点提醒我买牛奶", media, voice=False)
                service.handle(source)
                newer = self.message("audit-control-current", current_control, media, voice=False, minutes_later=2)
                self.assertIn(service.handle(newer).status, (ExecutionStatus.PLANNED, ExecutionStatus.SUCCEEDED))
                before = ledger.reminder_snapshot("voice-task-1", source)

                result = service.handle(self.message("audit-control-stale", stale_control, media, voice=False, minutes_later=1))

                self.assertEqual(before, ledger.reminder_snapshot("voice-task-1", source))
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_cancel_context_refresh_does_not_reactivate_cancelled_reminder(self):
        service, _, media, _, ledger = self.make_service()
        source = self.message("audit-cancel-create", "今天下午三点提醒我买牛奶", media, voice=False)
        service.handle(source)
        cancelled = self.message("audit-cancel-current", "刚才那个不要了", media, voice=False, minutes_later=1)
        service.handle(cancelled)
        self.assertEqual(0, ledger.active_reminder_count("voice-task-1", source))
        self.assertEqual(("voice-task-1",), tuple(ref.task_id for ref in ledger.recent_task_context(source.conversation_key, cancelled.received_at).candidates))
        before = ledger.reminder_snapshot("voice-task-1", source)

        for index, command in enumerate(("改成四点", "再提醒三次，每隔20分钟"), 2):
            service.handle(self.message(f"audit-cancel-no-reactivate-{index}", command, media, voice=False, minutes_later=index))
            self.assertEqual(before, ledger.reminder_snapshot("voice-task-1", source))

    def test_control_watermark_cannot_resurrect_a_concurrently_completed_task(self):
        service, _, media, _, ledger = self.make_service()
        source = self.message("audit-completed-create", "今天下午三点提醒我买牛奶", media, voice=False)
        service.handle(source)
        record_context = ledger.record_task_context

        def complete_before_record(*args, **kwargs):
            ledger.mark_task_completed(source.sender_key, "voice-task-1", conversation_key=source.conversation_key)
            return record_context(*args, **kwargs)

        ledger.record_task_context = complete_before_record
        changed = self.message("audit-completed-update", "改成四点", media, voice=False, minutes_later=1)
        service.handle(changed)

        self.assertEqual(0, ledger.active_reminder_count("voice-task-1", source))
        self.assertEqual((), ledger.recent_task_context(source.conversation_key, changed.received_at).candidates)

    def test_legacy_confirmation_table_migrates_without_reusing_unsafe_rows(self):
        _, _, media, _, migrated = self.make_service()
        source = self.message("audit-migration", "", media, voice=False)
        # Rebuild only this isolated in-memory fixture's table with the old
        # schema, then run the same initializer used when opening old databases.
        migrated._connection.execute("DROP TABLE pending_completion")
        migrated._connection.execute("""CREATE TABLE pending_completion (
            sender_hash TEXT NOT NULL, ordinal INTEGER NOT NULL,
            task_id TEXT NOT NULL, title TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '', project_id TEXT NOT NULL DEFAULT '',
            task_status TEXT NOT NULL DEFAULT '', source_message_id TEXT NOT NULL,
            expires_at TEXT NOT NULL, PRIMARY KEY (sender_hash, ordinal))""")
        migrated._connection.execute(
            "INSERT INTO pending_completion VALUES (?, 1, 'old-task', '旧任务', '', '', '', 'old-source', ?)",
            (IdempotencyLedger._hash(source.conversation_key), (NOW + timedelta(hours=1)).isoformat()),
        )
        migrated._initialize()
        self.assertEqual((), migrated.pending_completion(source.conversation_key, NOW))
        replacement = TaskReference("new-task", "新任务")
        self.assertTrue(migrated.set_pending_completion(
            source.conversation_key, (replacement,), "new-source",
            NOW + timedelta(minutes=5), observed_at=NOW,
        ))
        self.assertEqual((replacement,), migrated.pending_completion(source.conversation_key, NOW))

    def test_every_overdue_item_marked_sent_has_its_title_delivered(self):
        _, _, media, _, ledger = self.make_service()
        source = self.message("audit-overdue", "", media, voice=False)
        refs = tuple(TaskReference(f"audit-task-{i}", f"审核事项编号{i:02d}终") for i in range(11))
        due_at = NOW - timedelta(seconds=fixtures.SETTINGS.reminder_overdue_merge_seconds + 1)
        for ref in refs:
            ledger.enqueue_reminder(source, ref, due_at)
        delivered = []

        result = ReminderScheduler(fixtures.SETTINGS, ledger).poll_once(
            lambda record, text: delivered.append(text) or f"audit-delivery-{len(delivered)}",
            NOW,
        )

        self.assertEqual(11, result.sent)
        self.assertEqual(2, result.merged_messages)
        self.assertEqual(2, len(delivered))
        for ref in refs:
            with self.subTest(task=ref.task_id):
                self.assertEqual("sent", ledger.reminder_status(ref.task_id, due_at))
                self.assertTrue(any(ref.title in text for text in delivered))

    def test_overdue_chunks_retry_only_the_chunk_proven_not_sent(self):
        _, _, media, _, ledger = self.make_service()
        source = self.message("audit-overdue-failure", "", media, voice=False)
        refs = tuple(TaskReference(f"audit-retry-task-{i}", f"分组事项{i:02d}终") for i in range(11))
        due_at = NOW - timedelta(seconds=fixtures.SETTINGS.reminder_overdue_merge_seconds + 1)
        for ref in refs:
            ledger.enqueue_reminder(source, ref, due_at)
        calls = []

        def first_attempt(record, text):
            calls.append(text)
            if len(calls) == 2:
                raise ReminderDeliveryPreSendError()
            return "audit-first-chunk"

        scheduler = ReminderScheduler(fixtures.SETTINGS, ledger)
        first = scheduler.poll_once(first_attempt, NOW)
        self.assertEqual((10, 1, 0, 2), (first.sent, first.failed, first.uncertain, first.merged_messages))
        retry_texts = []
        second = scheduler.poll_once(
            lambda record, text: retry_texts.append(text) or "audit-retry-chunk",
            NOW + timedelta(seconds=fixtures.SETTINGS.reminder_retry_seconds + 1),
        )

        self.assertEqual(1, second.claimed)
        self.assertEqual(1, second.sent)
        self.assertEqual(1, len(retry_texts))
        self.assertIn(refs[-1].title, retry_texts[0])
        self.assertNotIn(refs[0].title, retry_texts[0])

    def test_overdue_chunk_cancelled_after_claim_is_not_sent_or_counted_as_a_message(self):
        _, _, media, _, ledger = self.make_service()
        source = self.message("audit-group-cancel", "", media, voice=False)
        refs = tuple(TaskReference(f"audit-group-cancel-{i}", f"待补发事项{i:02d}终") for i in range(11))
        due_at = NOW - timedelta(seconds=fixtures.SETTINGS.reminder_overdue_merge_seconds + 1)
        for ref in refs:
            ledger.enqueue_reminder(source, ref, due_at)
        delivered = []

        def sender(record, text):
            delivered.append(text)
            ledger.cancel_reminders(
                refs[-1].task_id, source, scope="all",
                expected_snapshot=ledger.reminder_snapshot(refs[-1].task_id, source),
            )
            return "audit-cancelled-next-chunk"

        result = ReminderScheduler(fixtures.SETTINGS, ledger).poll_once(sender, NOW)

        self.assertEqual((11, 10, 0, 0, 1), (result.claimed, result.sent, result.failed, result.uncertain, result.merged_messages))
        self.assertEqual(1, len(delivered))
        self.assertNotIn(refs[-1].title, delivered[0])
        self.assertEqual("cancelled", ledger.reminder_status(refs[-1].task_id, due_at))

    def test_uncertain_overdue_chunk_never_retries_and_does_not_hide_next_chunk(self):
        _, _, media, _, ledger = self.make_service()
        source = self.message("audit-group-uncertain", "", media, voice=False)
        refs = tuple(TaskReference(f"audit-group-uncertain-{i}", f"发送不确定事项{i:02d}终") for i in range(11))
        due_at = NOW - timedelta(seconds=fixtures.SETTINGS.reminder_overdue_merge_seconds + 1)
        for ref in refs:
            ledger.enqueue_reminder(source, ref, due_at)
        attempted = []

        def sender(record, text):
            attempted.append(text)
            if len(attempted) == 1:
                raise ReminderDeliveryUncertainError("delivery-timeout")
            return "audit-last-chunk-sent"

        scheduler = ReminderScheduler(fixtures.SETTINGS, ledger)
        result = scheduler.poll_once(sender, NOW)

        self.assertEqual((1, 0, 10, 2), (result.sent, result.failed, result.uncertain, result.merged_messages))
        self.assertIn(refs[-1].title, attempted[-1])
        self.assertEqual("uncertain", ledger.reminder_status(refs[0].task_id, due_at))
        self.assertEqual("sent", ledger.reminder_status(refs[-1].task_id, due_at))
        self.assertEqual(0, scheduler.poll_once(sender, NOW + timedelta(hours=1)).claimed)
        self.assertEqual(2, len(attempted))


if __name__ == "__main__":
    unittest.main()
