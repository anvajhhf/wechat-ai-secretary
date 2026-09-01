"""Journal migration tests use owned fixtures only, never runtime/state databases."""
from __future__ import annotations

import sqlite3
import subprocess
import sys
import shutil
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.models import ExecutionStatus, MessageEnvelope, TaskReference


TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"
NOW = datetime(2035, 1, 1, 9, tzinfo=timezone(timedelta(hours=8)))


class SQLiteJournalSafetyTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        # Python 3.13's Windows TemporaryDirectory ACL can exclude the sandbox
        # token; inherit this workspace test directory's permissions instead.
        self.root = (TEST_ROOT / f"sqlite-journal-{uuid4().hex}").resolve()
        self.assertTrue(self.root.is_relative_to(TEST_ROOT.resolve()))
        self.root.mkdir()
        self.addCleanup(self.cleanup_fixture)

    def cleanup_fixture(self):
        if self.root.parent != TEST_ROOT.resolve() or not self.root.name.startswith("sqlite-journal-"):
            raise AssertionError("refusing to clean outside the owned test fixture")
        shutil.rmtree(self.root)

    def ledger(self, path):
        ledger = IdempotencyLedger(path)
        self.addCleanup(ledger.close)
        return ledger

    def seed(self, path):
        ledger = self.ledger(path)
        message = MessageEnvelope(
            platform="weixin", account_id="fixture-account", user_id="fixture-user",
            chat_id="fixture-chat", chat_type="dm", message_id="fixture-message",
            text="fixture", received_at=NOW,
        )
        task = TaskReference("fixture-task", "虚拟提醒")
        ledger.claim(message)
        ledger.finish(message, ExecutionStatus.SUCCEEDED)
        _, row_ids = ledger.enqueue_reminders(message, task, (NOW, NOW + timedelta(days=1)))
        ledger.mark_reminders_sent((row_ids[0],), NOW, "fixture-delivered")
        ledger.set_private_protection(message.sender_key, "fixture-generation")
        ledger.set_pending_completion(
            message.conversation_key, (task,), message.message_id,
            NOW + timedelta(minutes=5), observed_at=NOW,
        )
        snapshot = ledger.reminder_snapshot(task.task_id, message)
        ledger.close()
        return message, task, snapshot

    def test_disk_and_memory_use_verified_safe_mode_and_extra_sync(self):
        for path, expected in ((self.root / "fresh.sqlite3", "delete"), (":memory:", "memory")):
            with self.subTest(path=str(path)):
                ledger = self.ledger(path)
                self.assertEqual(expected, ledger._connection.execute("PRAGMA journal_mode").fetchone()[0])
                self.assertEqual(3, ledger._connection.execute("PRAGMA synchronous").fetchone()[0])
                self.assertEqual(1, ledger._connection.execute("PRAGMA foreign_keys").fetchone()[0])

    def test_quiesced_wal_migration_preserves_reminders_messages_and_privacy(self):
        path = self.root / "legacy-wal.sqlite3"
        message, task, expected = self.seed(path)
        legacy = sqlite3.connect(path, isolation_level=None)
        try:
            self.assertEqual("wal", legacy.execute("PRAGMA journal_mode=WAL").fetchone()[0])
        finally:
            legacy.close()

        migrated = self.ledger(path)

        self.assertEqual("delete", migrated._connection.execute("PRAGMA journal_mode").fetchone()[0])
        self.assertEqual("ok", migrated._connection.execute("PRAGMA integrity_check").fetchone()[0])
        self.assertEqual(expected, migrated.reminder_snapshot(task.task_id, message))
        self.assertEqual(ExecutionStatus.SUCCEEDED.value, migrated.state_for(message))
        self.assertEqual("fixture-generation", migrated.get_private_protection(message.sender_key))
        self.assertEqual((task,), migrated.pending_completion(message.conversation_key, NOW))

    def test_legacy_reminder_table_gains_daily_columns_without_changing_rows(self):
        path = self.root / "legacy-reminders.sqlite3"
        legacy = sqlite3.connect(path, isolation_level=None)
        try:
            legacy.execute(
                """
                CREATE TABLE reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    platform TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    reminder_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL DEFAULT '',
                    delivered_at TEXT NOT NULL DEFAULT '',
                    delivered_message_id TEXT NOT NULL DEFAULT '',
                    last_error_code TEXT NOT NULL DEFAULT '',
                    UNIQUE(task_id, reminder_at)
                )
                """
            )
            legacy.execute(
                """
                INSERT INTO reminders(
                    task_id, title, platform, account_id, user_id, chat_id,
                    source_message_id, reminder_at, status, next_attempt_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-task", "旧提醒", "weixin", "legacy-account",
                    "legacy-user", "legacy-chat", "legacy-message",
                    NOW.isoformat(), "pending", NOW.isoformat(),
                ),
            )
        finally:
            legacy.close()

        migrated = self.ledger(path)
        row = migrated._connection.execute(
            """
            SELECT task_id, status, recurrence_frequency,
                   recurrence_interval, recurrence_slot, recurrence_active
            FROM reminders WHERE task_id = 'legacy-task'
            """
        ).fetchone()

        self.assertEqual(
            ("legacy-task", "pending", "", 0, "", 0), tuple(row),
        )

    def test_committed_wal_left_by_interrupted_writer_is_recovered_before_conversion(self):
        path = self.root / "interrupted-wal.sqlite3"
        message, task, _ = self.seed(path)
        program = """
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
assert connection.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
connection.execute('PRAGMA wal_autocheckpoint=0')
connection.execute("UPDATE private_ingress_protection SET token='committed-generation'")
connection.execute("UPDATE reminders SET status='cancelled' WHERE status='pending'")
connection.execute("UPDATE messages SET state='uncertain'")
os._exit(0)
"""
        subprocess.run([sys.executable, "-c", program, str(path)], check=True, timeout=10, capture_output=True)
        self.assertTrue(Path(str(path) + "-wal").is_file())

        migrated = self.ledger(path)

        self.assertEqual("delete", migrated._connection.execute("PRAGMA journal_mode").fetchone()[0])
        self.assertEqual("ok", migrated._connection.execute("PRAGMA integrity_check").fetchone()[0])
        self.assertEqual("committed-generation", migrated.get_private_protection(message.sender_key))
        self.assertEqual("uncertain", migrated.state_for(message))
        self.assertEqual(["sent", "cancelled"], [row[1] for row in migrated.reminder_snapshot(task.task_id, message)])
        self.assertEqual(0, migrated.active_reminder_count(task.task_id, message))

    def test_wal_read_or_write_lock_refuses_transition_and_closes_failed_connection(self):
        real_connect = sqlite3.connect
        for transaction in ("BEGIN", "BEGIN IMMEDIATE"):
            with self.subTest(transaction=transaction):
                path = self.root / ("reader.sqlite3" if transaction == "BEGIN" else "writer.sqlite3")
                owner = real_connect(path, isolation_level=None)
                self.addCleanup(owner.close)
                owner.execute("PRAGMA journal_mode=WAL")
                owner.execute("CREATE TABLE fixture_payload(value TEXT NOT NULL)")
                owner.execute("INSERT INTO fixture_payload VALUES ('preserved')")
                owner.execute(transaction)
                owner.execute("SELECT * FROM fixture_payload").fetchall()
                attempted = []

                def fast_connect(*args, **kwargs):
                    kwargs["timeout"] = 0
                    connection = real_connect(*args, **kwargs)
                    attempted.append(connection)
                    return connection

                with patch("wechat_secretary.ledger.sqlite3.connect", side_effect=fast_connect):
                    with self.assertRaises(sqlite3.OperationalError):
                        IdempotencyLedger(path)
                self.assertEqual(1, len(attempted))
                with self.assertRaises(sqlite3.ProgrammingError):
                    attempted[0].execute("SELECT 1")
                self.assertEqual("wal", owner.execute("PRAGMA journal_mode").fetchone()[0])
                self.assertEqual("preserved", owner.execute("SELECT value FROM fixture_payload").fetchone()[0])
                owner.rollback()
                owner.close()
                migrated = self.ledger(path)
                self.assertEqual("delete", migrated._connection.execute("PRAGMA journal_mode").fetchone()[0])
                self.assertEqual("preserved", migrated._connection.execute("SELECT value FROM fixture_payload").fetchone()[0])

    def test_mode_or_sync_refusal_fails_before_schema_initialization(self):
        for mode, sync in (("wal", 3), ("memory", 3), (None, 3), ("delete", 2), ("delete", None)):
            with self.subTest(mode=mode, sync=sync):
                connection = Mock()

                def execute(sql):
                    cursor = Mock()
                    value = mode if sql == "PRAGMA journal_mode=DELETE" else sync
                    cursor.fetchone.return_value = (value,) if value is not None else None
                    return cursor

                connection.execute.side_effect = execute
                with patch("wechat_secretary.ledger.sqlite3.connect", return_value=connection):
                    with self.assertRaises(RuntimeError):
                        IdempotencyLedger(self.root / "refused.sqlite3")
                connection.executescript.assert_not_called()
                self.assertTrue(connection.close.called)

    def test_initialization_failure_closes_connection_even_while_instance_is_retained(self):
        retained = []

        def fail(ledger):
            retained.append(ledger)
            raise RuntimeError("fixture initialization failed")

        with patch.object(IdempotencyLedger, "_initialize", fail):
            with self.assertRaisesRegex(RuntimeError, "fixture initialization failed"):
                IdempotencyLedger(self.root / "initialization-failed.sqlite3")
        self.addCleanup(retained[0].close)
        with self.assertRaises(sqlite3.ProgrammingError):
            retained[0]._connection.execute("SELECT 1")

    def test_close_failure_does_not_mask_original_initialization_error(self):
        connection = Mock()
        connection.close.side_effect = RuntimeError("fixture close failure")
        with patch("wechat_secretary.ledger.sqlite3.connect", return_value=connection):
            with patch.object(IdempotencyLedger, "_initialize", side_effect=ValueError("original initialization failure")):
                with self.assertRaisesRegex(ValueError, "original initialization failure"):
                    IdempotencyLedger(self.root / "close-failed.sqlite3")

    def test_two_delete_connections_can_alternate_writes_without_lost_state(self):
        path = self.root / "shared-delete.sqlite3"
        first = self.ledger(path)
        second = self.ledger(path)
        first.set_private_protection("fixture-sender", "generation-one")
        self.assertEqual("generation-one", second.get_private_protection("fixture-sender"))
        second.set_private_protection("fixture-sender", "generation-two")
        self.assertFalse(first.clear_private_protection("fixture-sender", "generation-one"))
        self.assertTrue(first.clear_private_protection("fixture-sender", "generation-two"))
        self.assertEqual("", second.get_private_protection("fixture-sender"))
        for connection in (first._connection, second._connection):
            self.assertEqual("delete", connection.execute("PRAGMA journal_mode").fetchone()[0])
            self.assertEqual("ok", connection.execute("PRAGMA integrity_check").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
