from __future__ import annotations

import concurrent.futures
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from wechat_secretary.config import SecretarySettings
from wechat_secretary.hermes_plugin import _reminder_sender
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.models import MessageEnvelope, TaskReference
from wechat_secretary.reminders import (
    ReminderDeliveryPreSendError,
    ReminderDeliveryUncertainError,
    ReminderScheduler,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.fromisoformat("2026-08-24T09:00:00+08:00")


def settings() -> SecretarySettings:
    return SecretarySettings(
        project_root=ROOT,
        dry_run=False,
        reminders_enabled=True,
        allowed_users=frozenset({"wx-user-1"}),
        account_id="owner-account",
        reminder_retry_seconds=60,
    )


def message(message_id: str) -> MessageEnvelope:
    return MessageEnvelope(
        platform="weixin",
        account_id="owner-account",
        user_id="wx-user-1",
        chat_id="chat-1",
        chat_type="dm",
        message_id=message_id,
        text="",
        received_at=NOW,
    )


class ReminderDeliveryUncertaintyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = IdempotencyLedger(":memory:")
        self.scheduler = ReminderScheduler(settings(), self.ledger)

    def tearDown(self) -> None:
        self.ledger.close()

    def _enqueue(self, task_id: str, reminder_at: datetime = NOW) -> int:
        created, row_id = self.ledger.enqueue_reminder(
            message(f"source-{task_id}"),
            TaskReference(task_id, f"提醒 {task_id}"),
            reminder_at,
        )
        self.assertTrue(created)
        return row_id

    def test_unclassified_sender_exception_is_terminal_uncertain(self) -> None:
        self._enqueue("unknown-exception")
        calls = 0

        def sender(record: object, content: str) -> str:
            nonlocal calls
            del record, content
            calls += 1
            raise TimeoutError("provider-secret-must-not-be-persisted-or-retried")

        stats = self.scheduler.poll_once(sender, NOW)

        self.assertEqual(1, stats.claimed)
        self.assertEqual(0, stats.failed)
        self.assertEqual(1, stats.uncertain)
        self.assertEqual(
            "uncertain", self.ledger.reminder_status("unknown-exception", NOW)
        )

        later = self.scheduler.poll_once(sender, NOW + timedelta(hours=1))
        self.assertEqual(0, later.claimed)
        self.assertEqual(1, calls)

    def test_explicit_pre_send_failure_is_the_only_retryable_path(self) -> None:
        self._enqueue("pre-send")
        calls = 0

        def sender(record: object, content: str) -> str:
            nonlocal calls
            del record, content
            calls += 1
            if calls == 1:
                raise ReminderDeliveryPreSendError("route-unavailable")
            return "outbound-2"

        first = self.scheduler.poll_once(sender, NOW)
        self.assertEqual(1, first.failed)
        self.assertEqual(0, first.uncertain)
        self.assertEqual("failed", self.ledger.reminder_status("pre-send", NOW))

        too_soon = self.scheduler.poll_once(sender, NOW + timedelta(seconds=59))
        self.assertEqual(0, too_soon.claimed)
        retried = self.scheduler.poll_once(sender, NOW + timedelta(seconds=60))
        self.assertEqual(1, retried.sent)
        self.assertEqual(2, calls)
        self.assertEqual("sent", self.ledger.reminder_status("pre-send", NOW))

    def test_stale_delivering_claim_becomes_uncertain_not_retryable(self) -> None:
        self._enqueue("stale-claim")
        claimed = self.ledger.claim_due_reminders(NOW)
        self.assertEqual(1, len(claimed))

        reclaimed = self.ledger.claim_due_reminders(
            NOW + timedelta(seconds=601), stale_seconds=600
        )

        self.assertEqual((), reclaimed)
        self.assertEqual("uncertain", self.ledger.reminder_status("stale-claim", NOW))
        self.ledger.mark_reminders_sent(
            (claimed[0].row_id,), NOW + timedelta(seconds=602), "late-message-id"
        )
        self.assertEqual("uncertain", self.ledger.reminder_status("stale-claim", NOW))

    def test_uncertain_result_does_not_overwrite_concurrent_cancellation(self) -> None:
        row_id = self._enqueue("cancel-race")
        self.assertEqual(1, len(self.ledger.claim_due_reminders(NOW)))
        self.ledger.mark_task_completed(
            "weixin:owner-account:wx-user-1", "cancel-race"
        )

        self.ledger.mark_reminders_uncertain((row_id,), "delivery-timeout")

        self.assertEqual("cancelled", self.ledger.reminder_status("cancel-race", NOW))


class HermesReminderSenderClassificationTests(unittest.TestCase):
    @staticmethod
    def _record() -> SimpleNamespace:
        return SimpleNamespace(platform="weixin", chat_id="chat-1")

    def test_disconnected_preflight_is_retryable_and_does_not_call_send(self) -> None:
        adapter = SimpleNamespace(is_connected=False, send=Mock())
        gateway = SimpleNamespace(adapters={"weixin": adapter})
        loop = Mock()
        loop.is_closed.return_value = False

        with self.assertRaises(ReminderDeliveryPreSendError) as raised:
            _reminder_sender(loop, gateway, self._record(), "提醒内容")

        self.assertEqual("transport-not-ready", raised.exception.reason_code)
        adapter.send.assert_not_called()

    def test_future_timeout_is_uncertain_and_error_is_sanitized(self) -> None:
        class Adapter:
            is_connected = True

            async def send(self, **kwargs: object) -> object:
                del kwargs
                return SimpleNamespace(success=True, message_id="late-id")

        class TimedOutFuture:
            @staticmethod
            def result(timeout: int) -> object:
                if timeout != 30:
                    raise AssertionError("unexpected timeout")
                raise concurrent.futures.TimeoutError("provider-token=secret")

        def submit(awaitable: object, loop: object) -> TimedOutFuture:
            del loop
            awaitable.close()
            return TimedOutFuture()

        loop = Mock()
        loop.is_closed.return_value = False
        gateway = SimpleNamespace(adapters={"weixin": Adapter()})
        with patch(
            "wechat_secretary.hermes_plugin.asyncio.run_coroutine_threadsafe",
            side_effect=submit,
        ):
            with self.assertRaises(ReminderDeliveryUncertainError) as raised:
                _reminder_sender(loop, gateway, self._record(), "提醒内容")

        self.assertEqual("delivery-timeout", raised.exception.reason_code)
        self.assertNotIn("secret", str(raised.exception).casefold())

    def test_adapter_failure_result_is_uncertain_without_leaking_detail(self) -> None:
        class Adapter:
            is_connected = True

            async def send(self, **kwargs: object) -> object:
                del kwargs
                return None

        class FailedFuture:
            @staticmethod
            def result(timeout: int) -> object:
                del timeout
                return SimpleNamespace(
                    success=False,
                    error="access_token=provider-secret",
                    message_id=None,
                )

        def submit(awaitable: object, loop: object) -> FailedFuture:
            del loop
            awaitable.close()
            return FailedFuture()

        loop = Mock()
        loop.is_closed.return_value = False
        gateway = SimpleNamespace(adapters={"weixin": Adapter()})
        with patch(
            "wechat_secretary.hermes_plugin.asyncio.run_coroutine_threadsafe",
            side_effect=submit,
        ):
            with self.assertRaises(ReminderDeliveryUncertainError) as raised:
                _reminder_sender(loop, gateway, self._record(), "提醒内容")

        self.assertEqual("adapter-reported-failure", raised.exception.reason_code)
        self.assertNotIn("provider-secret", str(raised.exception))

    def test_submission_failure_is_known_pre_send_and_sanitized(self) -> None:
        class Adapter:
            is_connected = True

            async def send(self, **kwargs: object) -> object:
                del kwargs
                return None

        loop = Mock()
        loop.is_closed.return_value = False
        gateway = SimpleNamespace(adapters={"weixin": Adapter()})
        with patch(
            "wechat_secretary.hermes_plugin.asyncio.run_coroutine_threadsafe",
            side_effect=RuntimeError("internal credential=secret"),
        ):
            with self.assertRaises(ReminderDeliveryPreSendError) as raised:
                _reminder_sender(loop, gateway, self._record(), "提醒内容")

        self.assertEqual("dispatch-submit-failed", raised.exception.reason_code)
        self.assertNotIn("secret", str(raised.exception).casefold())


if __name__ == "__main__":
    unittest.main()
