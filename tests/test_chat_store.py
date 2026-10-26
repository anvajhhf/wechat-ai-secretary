from __future__ import annotations

import shutil
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from wechat_secretary.chat import ChatStore, HermesChatResponder


TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"
NOW = datetime(2035, 1, 1, 9, tzinfo=timezone.utc)
KEY = "weixin:account:user:chat"


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = (TEST_ROOT / f"chat-store-{uuid4().hex}").resolve()
        self.root.mkdir()
        self.now = NOW
        self.addCleanup(self.clean_fixture)

    def clean_fixture(self):
        if self.root.parent != TEST_ROOT.resolve() or not self.root.name.startswith("chat-store-"):
            raise AssertionError("refusing to clean outside the owned test fixture")
        shutil.rmtree(self.root)

    def store(self, name=":memory:", **kwargs):
        path = name if name == ":memory:" else self.root / name
        store = ChatStore(path, clock=lambda: self.now, **kwargs)
        self.addCleanup(store.close)
        return store

    @staticmethod
    def add(store, user="问题", assistant="回答", *, key=KEY, message_id=""):
        return store.record_sent(
            key, store.get_generation(key), user, assistant, message_id=message_id,
        )

    def test_text_survives_restart_with_current_generation(self):
        store = self.store("history.sqlite3")
        self.add(store, "我在写报告", "先列出主要结论", message_id="first")
        generation = store.get_generation(KEY)
        store.close()
        reopened = self.store("history.sqlite3")
        self.assertEqual(generation, reopened.get_generation(KEY))
        self.assertEqual([
            {"role": "user", "content": "我在写报告"},
            {"role": "assistant", "content": "先列出主要结论"},
        ], reopened.history(KEY, 10, 12000))

    def test_history_limits_recent_complete_rounds(self):
        store = self.store()
        for number in range(12):
            self.add(store, f"user-{number}", f"reply-{number}")
        history = store.history(KEY, 10, 12000)
        self.assertEqual(20, len(history))
        self.assertEqual("user-2", history[0]["content"])
        self.assertEqual("reply-11", history[-1]["content"])

    def test_character_budget_never_includes_partial_round(self):
        store = self.store()
        self.add(store, "123", "456")
        self.add(store, "ab", "cd")
        self.assertEqual([
            {"role": "user", "content": "ab"},
            {"role": "assistant", "content": "cd"},
        ], store.history(KEY, 10, 8))
        self.assertEqual([], store.history(KEY, 10, 3))

    def test_large_latest_round_does_not_expose_detached_older_context(self):
        store = self.store()
        self.add(store, "a", "b")
        self.add(store, "long" * 30, "answer")
        self.assertEqual([], store.history(KEY, 10, 50))

    def test_retention_uses_actual_clock_and_expires_at_boundary(self):
        store = self.store()
        self.add(store, "old", "reply")
        self.now += timedelta(days=7) - timedelta(seconds=1)
        self.assertEqual(2, len(store.history(KEY, 10, 100)))
        self.now += timedelta(seconds=1)
        self.assertEqual([], store.history(KEY, 10, 100))
        self.assertEqual(0, store._connection.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0])

    def test_reopening_database_purges_expired_text(self):
        store = self.store("expired.sqlite3")
        self.add(store)
        store.close()
        self.now += timedelta(days=8)
        reopened = self.store("expired.sqlite3")
        self.assertEqual(0, reopened._connection.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0])

    def test_purge_can_run_without_inbound_message(self):
        store = self.store()
        self.add(store)
        self.now += timedelta(days=8)
        self.assertEqual(1, store.purge())

    def test_new_topic_excludes_prior_turns_but_retains_until_expiry(self):
        store = self.store()
        self.add(store, "old-topic", "old-answer")
        old = store.get_generation(KEY)
        store.reset(KEY)
        self.assertNotEqual(old, store.get_generation(KEY))
        self.assertEqual([], store.history(KEY, 10, 1000))
        self.assertEqual(1, store._connection.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0])
        self.assertFalse(store.record_sent(KEY, old, "late", "late-reply"))

    def test_clear_removes_all_topics_and_late_receipt_cannot_restore_them(self):
        store = self.store("clear.sqlite3")
        self.add(store, "sensitive-previous-topic", "previous-answer")
        store.reset(KEY)
        self.add(store, "sensitive-current-topic", "current-answer")
        generation = store.get_generation(KEY)
        store.reset(KEY, clear=True)
        self.assertFalse(store.record_sent(KEY, generation, "late", "late-answer"))
        store.close()
        reopened = self.store("clear.sqlite3")
        self.assertEqual([], reopened.history(KEY, 10, 1000))
        self.assertEqual(0, reopened._connection.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0])
        disk = (self.root / "clear.sqlite3").read_bytes()
        self.assertNotIn(b"sensitive-previous-topic", disk)
        self.assertNotIn(b"sensitive-current-topic", disk)

    def test_another_connection_clear_invalidates_old_generation(self):
        first = self.store("shared.sqlite3")
        old = first.get_generation(KEY)
        second = self.store("shared.sqlite3")
        second.reset(KEY, clear=True)
        self.assertFalse(first.record_sent(KEY, old, "late", "reply"))

    def test_duplicate_delivery_receipt_does_not_duplicate_or_replace_turn(self):
        store = self.store()
        self.assertTrue(self.add(store, "original", "answer", message_id="same-message"))
        self.assertFalse(self.add(store, "changed", "changed-answer", message_id="same-message"))
        history = store.history(KEY, 10, 100)
        self.assertEqual(2, len(history))
        self.assertEqual("original", history[0]["content"])

    def test_profiles_and_conversations_are_isolated(self):
        owner = self.store("owner.sqlite3")
        partner = self.store("partner.sqlite3")
        self.add(owner, "owner-text", "owner-reply")
        self.add(partner, "partner-text", "partner-reply")
        self.add(owner, "other-chat", "other-reply", key="weixin:account:user:other")
        self.add(owner, "other-user", "other-reply", key="weixin:account:other:chat")
        self.add(owner, "other-account", "other-reply", key="weixin:other:user:chat")
        self.assertEqual("owner-text", owner.history(KEY, 10, 1000)[0]["content"])
        self.assertEqual("partner-text", partner.history(KEY, 10, 1000)[0]["content"])
        owner.reset(KEY, clear=True)
        self.assertEqual(2, len(owner.history("weixin:account:user:other", 10, 1000)))
        self.assertEqual(2, len(partner.history(KEY, 10, 1000)))

    def test_database_uses_safe_journal_and_secure_delete(self):
        for filename, mode in ((":memory:", "memory"), ("safe.sqlite3", "delete")):
            with self.subTest(filename=filename):
                store = self.store(filename)
                self.assertEqual(mode, store._connection.execute("PRAGMA journal_mode").fetchone()[0])
                self.assertEqual(3, store._connection.execute("PRAGMA synchronous").fetchone()[0])
                self.assertEqual(1, store._connection.execute("PRAGMA secure_delete").fetchone()[0])

    def test_clock_must_be_timezone_aware(self):
        with self.assertRaises(ValueError):
            ChatStore(":memory:", clock=lambda: datetime(2035, 1, 1))


class HermesChatResponderTests(unittest.TestCase):
    def responder(self, text="正常回答", **changes):
        defaults = dict(
            chat_max_input_chars=6000, chat_context_max_chars=12000,
            chat_max_rounds=10, chat_max_reply_chars=4000,
            chat_max_tokens=1200, image_max_files=4,
        )
        defaults.update(changes)
        settings = SimpleNamespace(**defaults)
        llm = Mock()
        llm.complete.return_value = SimpleNamespace(text=text, usage=SimpleNamespace(total_tokens=75))
        return HermesChatResponder(SimpleNamespace(llm=llm), settings), llm

    def test_plain_chat_is_one_tool_free_completion_with_ordered_history(self):
        responder, llm = self.responder()
        history = [{"role": "user", "content": "第一个问题"}, {"role": "assistant", "content": "第一个回答"}]
        self.assertEqual("正常回答", responder.respond("请展开", history))
        self.assertEqual(1, responder.call_count)
        llm.complete.assert_called_once()
        args, kwargs = llm.complete.call_args
        self.assertEqual(["system", "user", "assistant", "user"], [message["role"] for message in args[0]])
        self.assertEqual("请展开", args[0][-1]["content"])
        self.assertEqual("wechat_secretary_chat", kwargs["task"])
        self.assertEqual(1200, kwargs["max_tokens"])
        self.assertNotIn("tools", kwargs)

    def test_responder_reapplies_total_character_and_round_budgets(self):
        responder, llm = self.responder(chat_context_max_chars=900, chat_max_rounds=2)
        history = [item for number in range(8) for item in (
            {"role": "user", "content": f"u{number}" + "x" * 60},
            {"role": "assistant", "content": f"a{number}" + "x" * 60},
        )]
        responder.respond("新问题", history)
        messages = llm.complete.call_args.args[0]
        self.assertLessEqual(sum(len(message["content"]) for message in messages), 900)
        self.assertLessEqual(len(messages), 6)
        if len(messages) > 2:
            self.assertTrue(messages[-2]["content"].startswith("a7"))

    def test_oversized_input_is_rejected_before_model_call(self):
        responder, llm = self.responder(chat_max_input_chars=5)
        with self.assertRaises(ValueError):
            responder.respond("这是超过限制的内容", [])
        llm.complete.assert_not_called()
        self.assertEqual(0, responder.call_count)

    def test_historical_system_or_tool_role_is_rejected(self):
        responder, llm = self.responder()
        for role in ("system", "tool"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                responder.respond("继续", [{"role": role, "content": "bad"}, {"role": "assistant", "content": "x"}])
        llm.complete.assert_not_called()

    def test_incomplete_history_is_rejected(self):
        responder, llm = self.responder()
        with self.assertRaises(ValueError):
            responder.respond("继续", [{"role": "user", "content": "unfinished"}])
        llm.complete.assert_not_called()

    def test_prepared_images_use_vision_slot_and_data_urls_only(self):
        responder, llm = self.responder()
        responder.respond("图里有什么？", [], image_inputs=({
            "type": "image", "data": b"sanitized", "mime_type": "image/png", "file_name": "private-path.png",
        },))
        args, kwargs = llm.complete.call_args
        self.assertEqual("wechat_secretary_vision", kwargs["task"])
        image = args[0][-1]["content"][1]
        self.assertEqual("data:image/png;base64,c2FuaXRpemVk", image["image_url"]["url"])
        self.assertNotIn("private-path", str(args))

    def test_unprepared_or_remote_image_is_rejected(self):
        responder, llm = self.responder()
        with self.assertRaises(ValueError):
            responder.respond("图片", [], image_inputs=({"type": "image", "url": "https://example.invalid/image.png"},))
        llm.complete.assert_not_called()

    def test_empty_model_result_is_not_a_success(self):
        responder, _ = self.responder(text=" ")
        with self.assertRaises(ValueError):
            responder.respond("问题", [])

    def test_output_is_bounded_with_truncation_notice(self):
        responder, _ = self.responder(text="很长" * 100, chat_max_reply_chars=60)
        answer = responder.respond("问题", [])
        self.assertLessEqual(len(answer), 60)
        self.assertIn("已截取", answer)

    def test_logs_contain_metrics_but_not_question_or_answer(self):
        responder, _ = self.responder(text="sensitive-response")
        with self.assertLogs("wechat_secretary.chat", level="INFO") as captured:
            responder.respond("sensitive-question", [])
        output = "\n".join(captured.output)
        self.assertIn("tokens=75", output)
        self.assertNotIn("sensitive-question", output)
        self.assertNotIn("sensitive-response", output)

    def test_failure_log_does_not_include_exception_payload(self):
        responder, llm = self.responder()
        llm.complete.side_effect = ValueError("secret-provider-payload")
        with self.assertLogs("wechat_secretary.chat", level="WARNING") as captured:
            with self.assertRaises(ValueError):
                responder.respond("private-question", [])
        output = "\n".join(captured.output)
        self.assertIn("error=ValueError", output)
        self.assertNotIn("secret-provider-payload", output)
        self.assertNotIn("private-question", output)


if __name__ == "__main__":
    unittest.main()
