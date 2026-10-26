"""Synthetic-only tests for explicit facts, deletion and restore boundaries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

from wechat_secretary.chat import ChatStore
from wechat_secretary.memory import (
    MemoryCommand, MemoryStore, MemoryUnavailable, has_memory_state, parse_memory_command,
)


TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"
NOW = datetime(2035, 1, 1, 9, tzinfo=timezone(timedelta(hours=8)))
KEY = "weixin:fixture-account:fixture-user:fixture-chat"


class MemoryCommandTests(unittest.TestCase):
    def test_explicit_save_types_and_names(self):
        cases = (
            ("记住：我喜欢茶", "fact", "", "我喜欢茶"),
            ("请记住偏好：回答简洁", "preference", "", "回答简洁"),
            ("记住背景：我住在上海", "background", "", "我住在上海"),
            ("记住目标：学习游泳", "goal", "", "学习游泳"),
            ("记住项目：在整理实验方案", "project", "", "在整理实验方案"),
            ("记住 研究方向：免疫学", "fact", "研究方向", "免疫学"),
            ("记住偏好 回复风格：简短", "preference", "回复风格", "简短"),
        )
        for text, kind, name, content in cases:
            with self.subTest(text=text):
                command = parse_memory_command(text, NOW)
                self.assertEqual("remember", command.action)
                self.assertEqual((kind, name, content, ""), (command.kind, command.name, command.content, command.error))

    def test_date_expiry_is_end_of_named_local_day(self):
        command = parse_memory_command("记住项目[有效至2035-01-02]：本周整理数据", NOW)
        self.assertEqual(datetime(2035, 1, 3, tzinfo=NOW.tzinfo), command.expires_at)
        self.assertFalse(command.error)

    def test_invalid_memory_syntax_returns_error_not_chat(self):
        for text in ("记住：", "记住[有效至昨日]：内容", "记住[有效至2034-12-31]：内容", "更正记忆：新内容", "忘记记忆", "更正记忆 M-0123456789："):
            with self.subTest(text=text):
                command = parse_memory_command(text, NOW)
                self.assertIsNotNone(command)
                self.assertTrue(command.error)

    def test_list_correction_and_forget(self):
        self.assertEqual("list", parse_memory_command("你记住了我哪些事情？", NOW).action)
        self.assertEqual("方向", parse_memory_command("查看记忆 方向", NOW).target)
        correction = parse_memory_command("更正记忆 M-0123456789：新内容", NOW)
        self.assertEqual(("correct", "M-0123456789", "新内容"), (correction.action, correction.target, correction.content))
        self.assertEqual("M-0123456789", parse_memory_command("忘记记忆 M-0123456789", NOW).target)
        self.assertEqual("研究方向", parse_memory_command("忘记关于研究方向的那条记忆。", NOW).target)

    def test_non_commands_and_old_task_syntax_are_not_claimed(self):
        for text in ("忘记吃药了", "忘了吧", "帮我记住明天下午提醒我开会", "我记住了", "他说记住：这个道理", "“记住：喜欢茶”", "笔记：记住这个方法", "聊天：记住：回答简洁", "私密：记住我的地址"):
            with self.subTest(text=text):
                self.assertIsNone(parse_memory_command(text, NOW))


class MemoryFixture(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = (TEST_ROOT / f"memory-store-{uuid4().hex}").resolve()
        self.root.mkdir()
        self.now = NOW
        self.counter = 0
        self.addCleanup(self.clean_fixture)

    def clean_fixture(self):
        if self.root.parent != TEST_ROOT.resolve() or not self.root.name.startswith("memory-store-"):
            raise AssertionError("refusing to clean outside owned memory fixture")
        shutil.rmtree(self.root)

    def paths(self, profile="owner"):
        return self.root / "state" / profile / "assistant.sqlite3", self.root / "control" / profile / "guard.json"

    def memory(self, profile="owner", **kwargs):
        path, guard = self.paths(profile)
        store = MemoryStore(path, guard_path=guard, clock=lambda: self.now, **kwargs)
        self.addCleanup(store.close)
        return store

    def chat(self, profile="owner"):
        path, _ = self.paths(profile)
        store = ChatStore(path, clock=lambda: self.now)
        self.addCleanup(store.close)
        return store

    def apply(self, store, command, *, key=KEY, message_id=None):
        self.counter += 1
        if isinstance(command, str):
            command = parse_memory_command(command, self.now)
        return store.apply(key, message_id or f"synthetic-message-{self.counter}", command)


class MemoryPersistenceTests(MemoryFixture):
    def test_explicit_fact_has_stable_identity_provenance_and_version_after_restart(self):
        store = self.memory()
        added = self.apply(store, "记住背景：我的研究方向是免疫学", message_id="original-source")
        entry = added.entry
        self.assertEqual("added", added.status)
        self.assertEqual(1, added.revision)
        self.assertTrue(added.cleared_chat)
        self.assertRegex(entry.memory_id, r"^M-[0-9a-f]{10}$")
        self.assertEqual((1, "active", "original-source", "background"), (entry.version, entry.status, entry.source_message_id, entry.kind))
        self.assertEqual("研究方向", entry.name)
        store.close()
        reopened = self.memory()
        self.assertEqual((entry,), reopened.list_entries(KEY))
        self.assertEqual(1, reopened.current_revision(KEY))

    def test_named_and_conservative_natural_keys_refuse_conflicting_new_facts(self):
        store = self.memory()
        added = self.apply(store, "记住：我的研究方向是免疫学")
        conflict = self.apply(store, "记住：我的研究方向是神经科学")
        self.assertEqual("conflict", conflict.status)
        self.assertEqual((added.entry,), conflict.candidates)
        self.assertEqual(1, store.current_revision(KEY))
        named = self.apply(store, "记住 当前项目：原项目")
        conflict = self.apply(store, "记住 当前项目：新项目")
        self.assertEqual("conflict", conflict.status)
        self.assertEqual((named.entry,), conflict.candidates)

    def test_explicit_correction_replaces_text_but_keeps_identity_and_updates_sources(self):
        store = self.memory()
        first = self.apply(store, "记住 研究方向：old-sensitive-fact", message_id="first-source")
        updated = self.apply(store, f"更正记忆 {first.entry.memory_id}：new-current-fact", message_id="updated-source")
        self.assertEqual(("updated", first.entry.memory_id, 2, "updated-source"), (updated.status, updated.entry.memory_id, updated.entry.version, updated.entry.source_message_id))
        self.assertEqual("new-current-fact", updated.entry.content)
        self.assertEqual(2, updated.revision)
        self.assertEqual([("first-source", 1), ("updated-source", 0)], [tuple(row) for row in store._connection.execute(
            "SELECT source_message_id,revoked FROM memory_sources ORDER BY source_message_id"
        )])
        self.assertNotIn(b"old-sensitive-fact", self.paths()[0].read_bytes())

    def test_forget_erases_text_and_name_but_retains_source_control(self):
        store = self.memory()
        first = self.apply(store, "记住 private-name-marker：private-fact-marker")
        forgotten = self.apply(store, f"忘记记忆 {first.entry.memory_id}")
        self.assertEqual("forgotten", forgotten.status)
        self.assertEqual(("", "", "forgotten", 2), (forgotten.entry.content, forgotten.entry.name, forgotten.entry.status, forgotten.entry.version))
        self.assertEqual((), store.list_entries(KEY))
        self.assertEqual((forgotten.entry,), store.list_entries(KEY, include_inactive=True))
        self.assertEqual(1, store._connection.execute("SELECT revoked FROM memory_sources").fetchone()[0])
        data = self.paths()[0].read_bytes()
        self.assertNotIn(b"private-fact-marker", data)
        self.assertNotIn(b"private-name-marker", data)

    def test_replayed_message_never_returns_forgotten_content_or_recreates_fact(self):
        store = self.memory()
        command = parse_memory_command("记住：sensitive-original-value", self.now)
        added = self.apply(store, command, message_id="stable-id")
        self.apply(store, f"忘记记忆 {added.entry.memory_id}")
        duplicate = self.apply(store, command, message_id="stable-id")
        self.assertEqual("duplicate", duplicate.status)
        self.assertTrue(duplicate.duplicate)
        self.assertIsNone(duplicate.entry)
        self.assertEqual((), duplicate.candidates)
        self.assertEqual((), store.list_entries(KEY))

    def test_same_id_different_request_is_rejected(self):
        store = self.memory()
        self.apply(store, "记住：喜欢茶", message_id="same")
        invalid = self.apply(store, "记住：喜欢咖啡", message_id="same")
        self.assertEqual("invalid", invalid.status)
        self.assertEqual("喜欢茶", store.list_entries(KEY)[0].content)

    def test_exact_duplicate_new_request_does_not_bump_revision(self):
        store = self.memory()
        first = self.apply(store, "记住：我喜欢茶")
        duplicate = self.apply(store, "记住：我喜欢茶。")
        self.assertEqual("duplicate", duplicate.status)
        self.assertEqual(first.entry.memory_id, duplicate.entry.memory_id)
        self.assertEqual(first.revision, duplicate.revision)
        self.assertFalse(duplicate.cleared_chat)

    def test_ambiguous_keyword_requires_choice_and_unknown_id_does_not_modify(self):
        store = self.memory()
        self.apply(store, "记住：项目甲使用Python")
        self.apply(store, "记住：项目乙使用R")
        ambiguous = self.apply(store, "忘记关于项目的那条记忆")
        self.assertEqual("ambiguous", ambiguous.status)
        self.assertEqual(2, len(ambiguous.candidates))
        missing = self.apply(store, "忘记记忆 M-0000000000")
        self.assertEqual("not_found", missing.status)
        self.assertEqual(2, len(store.list_entries(KEY)))

    def test_profiles_and_conversations_are_independent(self):
        owner = self.memory("owner")
        partner = self.memory("partner")
        self.apply(owner, "记住偏好：owner-fact", message_id="shared-id")
        self.apply(partner, "记住偏好：partner-fact", message_id="shared-id")
        self.apply(owner, "记住偏好：other-chat-fact", key=KEY + "-other", message_id="shared-id")
        self.assertEqual("owner-fact", owner.snapshot(KEY).entries[0].content)
        self.assertEqual("partner-fact", partner.snapshot(KEY).entries[0].content)
        self.assertEqual("other-chat-fact", owner.snapshot(KEY + "-other").entries[0].content)
        self.assertEqual(1, owner.current_revision(KEY))
        self.assertEqual(1, owner.current_revision(KEY + "-other"))

    def test_capacity_limit_and_invalid_body_do_not_create_facts(self):
        store = self.memory(max_entries=1, max_entry_chars=10)
        self.assertEqual("invalid", self.apply(store, MemoryCommand("remember", content="x" * 11)).status)
        first = self.apply(store, "记住：第一条")
        self.assertEqual("invalid", self.apply(store, "记住：第二条").status)
        self.apply(store, f"忘记记忆 {first.entry.memory_id}")
        self.assertEqual("added", self.apply(store, "记住：第二条").status)

    def test_listing_is_current_revision_and_can_filter(self):
        store = self.memory()
        first = self.apply(store, "记住 研究方向：免疫学")
        self.apply(store, "记住偏好：回答简洁")
        listing = self.apply(store, "查看记忆 方向")
        self.assertEqual("list", listing.status)
        self.assertEqual((first.entry,), listing.candidates)
        self.assertEqual(store.current_revision(KEY), listing.revision)


class MemoryContextTests(MemoryFixture):
    def test_snapshot_includes_preferences_and_only_query_related_facts(self):
        store = self.memory()
        self.apply(store, "记住：以后回答尽量简洁")
        self.apply(store, "记住背景：我的研究方向是免疫学")
        self.apply(store, "记住项目：项目使用Python工具")
        self.apply(store, "记住背景：我的家乡是青岛")
        snapshot = store.snapshot(KEY, "聊聊研究方向和 Python")
        contents = {entry.content for entry in snapshot.entries}
        self.assertEqual({"以后回答尽量简洁", "我的研究方向是免疫学", "项目使用Python工具"}, contents)
        self.assertEqual(("以后回答尽量简洁",), tuple(entry.content for entry in store.snapshot(KEY, "周末如何休息").entries))

    def test_snapshot_never_dumps_unrelated_facts_when_query_has_no_match(self):
        store = self.memory()
        self.apply(store, "记住背景：我的研究方向是免疫学")
        self.assertEqual((), store.snapshot(KEY).entries)
        self.assertEqual((), store.snapshot(KEY, "你好").entries)

    def test_snapshot_obeys_six_entry_and_character_budgets(self):
        store = self.memory()
        for index in range(9):
            self.apply(store, MemoryCommand("remember", content=f"偏好内容{index}" + "文字" * 120, kind="preference"))
        snapshot = store.snapshot(KEY, max_items=99, max_chars=99999)
        self.assertLessEqual(len(snapshot.entries), 6)
        self.assertLessEqual(sum(len(entry.content) + len(entry.name) + len(entry.memory_id) + len(entry.kind) + 64 for entry in snapshot.entries), 2000)
        self.assertEqual((), store.snapshot(KEY, max_chars=5).entries)

    def test_fact_mutation_atomically_clears_all_chat_topics_and_late_receipts(self):
        chat = self.chat()
        first_generation = chat.get_generation(KEY)
        chat.record_sent(KEY, first_generation, "old-topic", "old-answer")
        chat.reset(KEY)
        second_generation = chat.get_generation(KEY)
        chat.record_sent(KEY, second_generation, "current-topic", "current-answer")
        other = KEY + "-other"
        chat.record_sent(other, chat.get_generation(other), "other-chat", "other-answer")
        store = self.memory()
        self.apply(store, "记住偏好：回答简洁")
        self.assertEqual([], chat.history(KEY, 10, 1000))
        self.assertFalse(chat.record_sent(KEY, second_generation, "late", "late-answer"))
        self.assertEqual(2, len(chat.history(other, 10, 1000)))
        rows = chat._connection.execute("SELECT COUNT(*) FROM chat_turns WHERE conversation_hash=?", (chat._key(KEY),)).fetchone()[0]
        self.assertEqual(0, rows)

    def test_expiration_during_generation_invalidates_revision_and_clears_history(self):
        chat = self.chat()
        store = self.memory()
        expires = self.now + timedelta(seconds=5)
        added = self.apply(store, MemoryCommand("remember", content="短期项目内容", kind="project", expires_at=expires))
        snapshot = store.snapshot(KEY, "项目")
        generation = chat.get_generation(KEY)
        chat.record_sent(KEY, generation, "旧问题", "依据短期项目的旧答案")
        self.now = expires
        self.assertFalse(store.is_current(KEY, snapshot.revision))
        self.assertEqual((), store.list_entries(KEY))
        self.assertEqual([], chat.history(KEY, 10, 1000))
        self.assertFalse(chat.record_sent(KEY, generation, "late", "late-answer"))
        tombstone = store.list_entries(KEY, include_inactive=True)[0]
        self.assertEqual((added.entry.memory_id, "expired", ""), (tombstone.memory_id, tombstone.status, tombstone.content))

    def test_correction_preserves_original_expiry_unless_explicitly_replaced(self):
        store = self.memory()
        first = self.apply(store, "记住项目[有效至2035-01-02]：旧计划")
        changed = self.apply(store, f"更正记忆 {first.entry.memory_id}：新计划")
        self.assertEqual(first.entry.expires_at, changed.entry.expires_at)
        extended = self.apply(store, f"更正记忆 {first.entry.memory_id}[有效至2035-01-03]：新计划")
        self.assertNotEqual(first.entry.expires_at, extended.entry.expires_at)

    def test_send_boundary_lock_is_reentrant_and_serializes_mutations(self):
        store = self.memory()
        started, completed = threading.Event(), threading.Event()
        errors = []
        def writer():
            started.set()
            try:
                store.apply(KEY, "concurrent", MemoryCommand("remember", content="并发记忆"))
            except Exception as exc:
                errors.append(exc)
            finally:
                completed.set()
        with store.delivery_guard():
            self.assertTrue(store.is_current(KEY, 0))
            thread = threading.Thread(target=writer)
            thread.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(completed.wait(0.03))
        thread.join(2)
        self.assertTrue(completed.is_set())
        self.assertEqual([], errors)
        self.assertEqual(1, store.current_revision(KEY))


class MemoryRestoreTests(MemoryFixture):
    def test_readonly_state_detection_does_not_create_a_missing_database(self):
        path, _ = self.paths()
        self.assertFalse(has_memory_state(path))
        self.assertFalse(path.exists())

    def test_readonly_state_detection_distinguishes_chat_only_and_memory_database(self):
        self.chat()
        path, _ = self.paths()
        before = path.read_bytes()
        self.assertFalse(has_memory_state(path))
        self.assertEqual(before, path.read_bytes())
        self.memory()
        self.assertTrue(has_memory_state(path))

    def test_corrupt_or_unreadable_database_is_treated_as_possible_memory_state(self):
        path = self.root / "corrupt.sqlite3"
        path.write_bytes(b"synthetic corrupt database")
        self.assertTrue(has_memory_state(path))
        with patch("wechat_secretary.memory.sqlite3.connect", side_effect=OSError("synthetic-access-failure")):
            self.assertTrue(has_memory_state(path))

    def test_guard_contains_only_identity_revision_and_control(self):
        store = self.memory()
        self.apply(store, "记住 secret-name：secret-content")
        data = self.paths()[1].read_text(encoding="utf-8")
        self.assertNotIn("secret-name", data)
        self.assertNotIn("secret-content", data)
        self.assertEqual({"schema", "database_id", "revision", "pending"}, set(json.loads(data)))

    def test_missing_guard_for_existing_memory_fails_closed(self):
        store = self.memory()
        self.apply(store, "记住偏好：敏感内容")
        store.close()
        self.paths()[1].unlink()
        with self.assertRaises(MemoryUnavailable):
            self.memory()

    def test_guard_removal_during_runtime_is_detected_before_recall(self):
        store = self.memory()
        self.apply(store, "记住偏好：敏感内容")
        self.paths()[1].unlink()
        for operation in (lambda: store.snapshot(KEY), lambda: store.list_entries(KEY), lambda: store.is_current(KEY, 1)):
            with self.assertRaises(MemoryUnavailable):
                operation()

    def test_restoring_older_database_cannot_revive_forgotten_fact(self):
        store = self.memory()
        first = self.apply(store, "记住偏好：old-backup-sensitive-value")
        backup = self.root / "older.sqlite3"
        destination = sqlite3.connect(backup)
        try:
            store._connection.backup(destination)
        finally:
            destination.close()
        self.apply(store, f"忘记记忆 {first.entry.memory_id}")
        store.close()
        shutil.copyfile(backup, self.paths()[0])
        with self.assertRaises(MemoryUnavailable):
            self.memory()

    def test_existing_guard_rejects_missing_database_instead_of_reinitializing(self):
        store = self.memory()
        self.apply(store, "记住：原内容")
        store.close()
        self.paths()[0].unlink()
        with self.assertRaises(MemoryUnavailable):
            self.memory()

    def test_foreign_guard_is_not_accepted(self):
        store = self.memory()
        self.apply(store, "记住：原内容")
        guard_path = self.paths()[1]
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
        guard["database_id"] = "another-database"
        guard_path.write_text(json.dumps(guard), encoding="utf-8")
        with self.assertRaises(MemoryUnavailable):
            store.snapshot(KEY, "原内容")

    def test_failed_prepared_write_leaves_guard_pending_and_never_recalls_old_context(self):
        chat = self.chat()
        store = self.memory()
        first = self.apply(store, "记住偏好：原内容")
        generation = chat.get_generation(KEY)
        chat.record_sent(KEY, generation, "old question", "old answer")
        original_write = store._write_guard
        def interrupted(database_id, revision, pending=None):
            original_write(database_id, revision, pending)
            if pending is not None:
                raise OSError("synthetic-interruption-before-db-commit")
        with patch.object(store, "_write_guard", side_effect=interrupted):
            with self.assertRaises(MemoryUnavailable):
                self.apply(store, f"忘记记忆 {first.entry.memory_id}")
        self.assertIsNotNone(json.loads(self.paths()[1].read_text(encoding="utf-8"))["pending"])
        with self.assertRaises(MemoryUnavailable):
            store.snapshot(KEY)
        store.close()
        with self.assertRaises(MemoryUnavailable):
            self.memory()

    def test_committed_pending_revision_can_finalize_without_replaying_mutation(self):
        store = self.memory()
        original_write = store._write_guard
        def interrupted(database_id, revision, pending=None):
            if pending is None and revision > 0:
                raise OSError("synthetic-interruption-after-db-commit")
            original_write(database_id, revision, pending)
        command = MemoryCommand("remember", content="已提交事实", kind="preference")
        with patch.object(store, "_write_guard", side_effect=interrupted):
            with self.assertRaises(MemoryUnavailable):
                self.apply(store, command, message_id="committed-id")
        snapshot = store.snapshot(KEY)
        self.assertEqual("已提交事实", snapshot.entries[0].content)
        self.assertEqual(1, snapshot.revision)
        self.assertIsNone(json.loads(self.paths()[1].read_text(encoding="utf-8"))["pending"])
        duplicate = self.apply(store, command, message_id="committed-id")
        self.assertTrue(duplicate.duplicate)
        self.assertIsNone(duplicate.entry)
        self.assertEqual(1, len(store.list_entries(KEY)))

    def test_guard_write_failure_rolls_back_fact_and_chat_together(self):
        chat = self.chat()
        store = self.memory()
        first = self.apply(store, "记住偏好：旧事实")
        generation = chat.get_generation(KEY)
        chat.record_sent(KEY, generation, "问题", "旧回答")
        with patch.object(store, "_write_guard", side_effect=OSError("synthetic-control-unwritable")):
            with self.assertRaises(MemoryUnavailable):
                self.apply(store, f"更正记忆 {first.entry.memory_id}：新事实")
        self.assertEqual("旧事实", store.list_entries(KEY)[0].content)
        self.assertEqual(generation, chat.get_generation(KEY))
        self.assertEqual(2, len(chat.history(KEY, 10, 1000)))
        self.assertEqual(1, store.current_revision(KEY))

    def test_second_connection_reads_revision_and_sources_committed_by_first(self):
        first = self.memory()
        second = self.memory()
        added = self.apply(first, "记住偏好：回答简洁")
        self.assertEqual((added.entry,), second.list_entries(KEY))
        self.apply(second, f"忘记记忆 {added.entry.memory_id}")
        self.assertFalse(first.is_current(KEY, added.revision))
        self.assertEqual((), first.list_entries(KEY))


if __name__ == "__main__":
    unittest.main()
