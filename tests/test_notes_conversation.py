"""Synthetic service tests: retrieved notes never authorize unrelated actions."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_memory_conversation import MemoryServiceFixture
from wechat_secretary.media import PreparedMedia
from wechat_secretary.models import ExecutionStatus
from wechat_secretary.notes_search import NotesExclusionResult, NotesHit, NotesSnapshot
from wechat_secretary.service import SecretaryService
from wechat_secretary.web_reader import WebPage


class ScriptedNotesIndex:
    """Script source changes; the real index has its own filesystem tests."""

    def __init__(self, hits):
        self.hits = tuple(hits)
        self.revision = 1
        self.search_calls = []
        self.validations = []
        self.exclusions = []
        self.rebuild_calls = 0
        self.scan_calls = 0
        self.search_error = None
        self.validation_error = None

    def search(self, query, limit=5):
        self.search_calls.append((query, limit))
        if self.search_error:
            raise self.search_error
        return NotesSnapshot(self.revision, self.hits[:limit])

    def validate(self, snapshot):
        self.validations.append(snapshot)
        if self.validation_error:
            raise self.validation_error
        return snapshot.revision == self.revision and all(hit in self.hits for hit in snapshot.hits)

    def change_sources(self, hits=()):
        self.hits = tuple(hits)
        self.revision += 1

    def current_revision(self):
        return self.revision

    def exclude_source(self, source_id):
        self.exclusions.append(source_id)
        matched = any(hit.source_id == source_id for hit in self.hits)
        if matched:
            self.change_sources(hit for hit in self.hits if hit.source_id != source_id)
        return NotesExclusionResult("excluded" if matched else "not_found", self.revision, source_id)

    def scan(self, force=False):
        self.scan_calls += 1
        return SimpleNamespace(scanned=2, indexed=2, removed=0, excluded=0, revision=self.revision, incomplete=False)

    def rebuild(self):
        self.rebuild_calls += 1
        self.revision += 1
        return SimpleNamespace(scanned=2, indexed=2, removed=0, excluded=0, revision=self.revision, incomplete=False)

    def delivery_guard(self):
        return nullcontext()

    def close(self):
        pass


class RecordingNotesAnswer:
    def __init__(self):
        self.calls = []
        self.answer = "合成方案的侧重点不同。[1]"
        self.before_answer = None
        self.error = None

    @property
    def call_count(self):
        return len(self.calls)

    def respond(self, question, sources):
        self.calls.append((question, tuple(sources)))
        if self.before_answer is not None:
            self.before_answer()
        if self.error is not None:
            raise self.error
        return self.answer


class NotesServiceFixture(MemoryServiceFixture):
    @staticmethod
    def sources():
        return tuple(
            NotesHit(
                source_id=f"N-{index:010x}", chunk_id=f"chunk-{index}",
                relative_path=f"Inbox/合成方案{index}.md", title=f"合成实验方案{index}",
                recorded_at="2026-09-15T09:00:00+08:00" if index == 1 else "",
                tags=("实验",), excerpt=f"合成实验方案{index}：对照组与观察指标。",
                line_start=1, line_end=5, version=f"synthetic-version-{index}",
                source_ref=f"synthetic-source-{index}",
            )
            for index in (1, 2)
        )

    def make_notes_service(self, *, profile="owner", enabled=True, memory_enabled=False, media=None, **changes):
        old, chat, memory, chat_store, dida, private = self.make_memory_service(
            profile=profile, enabled=memory_enabled, media=media,
            notes_search_enabled=enabled, notes_allowed_paths=("Inbox",), **changes,
        )
        index = ScriptedNotesIndex(self.sources())
        answer = RecordingNotesAnswer()
        notify = Mock()
        service = SecretaryService(
            settings=old.settings, ledger=old.ledger, classifier=old.classifier,
            dida=dida, obsidian=old.obsidian, private_inbox=private,
            reminders=old.reminders, media=old.media, web=old.web,
            chat=chat, chat_store=chat_store, memory_store=memory,
            notes_index=index, notes_answer=answer, notes_notify=notify,
        )
        return service, index, answer, chat, memory, chat_store, dida, private, notify

    def search(self, service, *, confirmed=True, **changes):
        return self.exchange(service, "搜索笔记：实验方案", confirmed=confirmed, **changes)


class NotesConversationTests(NotesServiceFixture):
    def test_explicit_search_is_local_with_numbered_sources_and_no_chat_history(self):
        service, index, answer, chat, _, chat_store, dida, _, _ = self.make_notes_service()
        incoming, result = self.search(service)
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertTrue(result.notes_guarded)
        self.assertTrue(result.notes_activate)
        self.assertTrue(result.notes_context_id)
        self.assertEqual(1, len(index.search_calls))
        for hit in index.hits:
            self.assertIn(hit.title, result.reply)
            self.assertIn(hit.relative_path, result.reply)
        self.assertEqual(0, service.classifier.call_count)
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], dida.queries)
        self.assertEqual([], self.history(chat_store, incoming))

    def test_natural_search_maps_last_month_and_tags_without_model(self):
        service, index, answer, chat, _, _, _, _, _ = self.make_notes_service()
        self.exchange(service, "找一下上个月记的实验方案")
        query = index.search_calls[-1][0]
        self.assertIn("实验方案", query.text)
        self.assertEqual(date(2026, 9, 1), query.date_from)
        self.assertEqual(date(2026, 9, 30), query.date_to)
        self.exchange(service, "查找笔记：实验 #对照")
        self.assertIn("对照", index.search_calls[-1][0].tags)
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)

    def test_followup_answer_uses_current_results_only_and_leaves_general_history_empty(self):
        service, index, answer, chat, _, chat_store, dida, _, _ = self.make_notes_service()
        source, _ = self.search(service)
        _, result = self.exchange(service, "这些方案有什么区别")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(1, answer.call_count)
        self.assertEqual(index.hits, answer.calls[0][1])
        self.assertIn("[1]", result.reply)
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], dida.created)
        self.assertEqual([], self.history(chat_store, source))

    def test_numbered_followup_uses_only_requested_current_source(self):
        service, index, answer, _, _, _, _, _, _ = self.make_notes_service()
        self.search(service)
        self.exchange(service, "第2条再详细说说")
        self.assertEqual(1, answer.call_count)
        self.assertEqual((index.hits[1],), answer.calls[-1][1])

    def test_missing_or_out_of_range_sources_never_call_answer_model(self):
        service, index, answer, chat, _, _, _, _, _ = self.make_notes_service()
        self.exchange(service, "笔记问答：这些方案有什么区别")
        self.assertEqual(0, answer.call_count)
        self.search(service)
        self.exchange(service, "第9条再详细说说")
        self.assertEqual(0, answer.call_count)
        index.change_sources(())
        self.search(service)
        self.exchange(service, "笔记问答：总结一下")
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)

    def test_search_with_no_matches_does_not_activate_a_result_set(self):
        service, index, answer, _, _, _, _, _, _ = self.make_notes_service()
        index.change_sources(())
        _, result = self.search(service)
        self.assertFalse(result.notes_guarded)
        self.assertFalse(result.notes_activate)
        self.exchange(service, "笔记问答：总结一下")
        self.assertEqual(0, answer.call_count)

    def test_explicit_prefixes_retain_ownership_of_search_words(self):
        cases = (
            ("待办：搜索笔记中的实验方案", "task"),
            ("笔记：搜索笔记：这只是语法示例", "note"),
            ("聊天：搜索笔记：这只是一个例子", "chat"),
            ("临时聊天：搜索笔记：这只是一个例子", "chat"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                service, index, answer, chat, _, _, dida, _, _ = self.make_notes_service()
                _, result = self.exchange(service, text)
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertEqual([], index.search_calls)
                self.assertEqual(0, answer.call_count)
                self.assertEqual(1 if expected == "chat" else 0, chat.call_count)
                self.assertEqual(1 if expected == "task" else 0, len(dida.created))
                if expected == "note":
                    self.assertIn("这只是语法示例", (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))

    def test_private_messages_and_spoken_private_never_read_notes(self):
        media = SimpleNamespace(prepare=Mock(return_value=PreparedMedia(transcript_text="私密：搜索笔记：合成秘密")))
        service, index, answer, chat, _, _, _, private, _ = self.make_notes_service(media=media)
        with patch.object(index, "search", side_effect=AssertionError("private search")), patch.object(index, "validate", side_effect=AssertionError("private source read")):
            self.exchange(service, "私密：搜索笔记：合成秘密")
            self.exchange(service, "私密：下一条")
            self.exchange(service, "搜索笔记：下一条私密查询")
            self.exchange(service, "", media_paths=("synthetic.wav",), media_types=("audio/wav",))
        self.assertEqual(2, len(private.saved))
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)

    def test_quoted_search_does_not_execute_and_old_task_query_still_works(self):
        service, index, answer, _, _, _, dida, _, _ = self.make_notes_service()
        self.exchange(service, "帮我解释‘搜索笔记：实验’这句话")
        self.assertEqual([], index.search_calls)
        self.exchange(service, "查询今天有哪些任务")
        self.assertEqual(1, len(dida.queries))
        self.assertEqual(0, answer.call_count)

    def test_search_ends_pending_task_and_anonymous_completion_never_uses_result_numbers(self):
        service, _, answer, _, _, _, dida, _, _ = self.make_notes_service()
        incoming, _ = self.exchange(service, "今天下午三点提醒我")
        self.assertIsNotNone(service.ledger.peek_pending_task(incoming.conversation_key, self.now + timedelta(minutes=1)))
        self.search(service)
        self.assertIsNone(service.ledger.peek_pending_task(incoming.conversation_key, self.now + timedelta(minutes=1)))
        for text in ("2", "就这个", "搞定了"):
            self.exchange(service, text)
        self.assertEqual([], dida.created)
        self.assertEqual([], dida.completed)
        self.assertEqual([], self.reminders(service))
        self.assertEqual(0, answer.call_count)

    def test_exclude_number_uses_current_source_without_deleting_note_or_task(self):
        service, index, answer, _, _, _, dida, _, _ = self.make_notes_service()
        source = index.hits[1]
        note_path = service.settings.vault_path / source.relative_path
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("合成原始笔记，排除后保留。", encoding="utf-8")
        self.search(service)
        self.exchange(service, "排除笔记 2")
        self.assertEqual([source.source_id], index.exclusions)
        self.assertEqual("合成原始笔记，排除后保留。", note_path.read_text(encoding="utf-8"))
        self.assertEqual([], dida.completed)
        self.assertEqual(0, answer.call_count)

    def test_local_rebuild_does_not_use_model_or_mutate_existing_operations(self):
        service, index, answer, chat, _, _, dida, _, _ = self.make_notes_service()
        self.exchange(service, "重建笔记索引")
        self.assertEqual(1, index.rebuild_calls)
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, chat.call_count)
        self.assertEqual([], dida.created)

    def test_result_context_is_isolated_by_user_chat_and_profile(self):
        service, _, answer, _, _, _, _, _, _ = self.make_notes_service(profile="owner")
        partner, _, partner_answer, _, _, _, _, _, _ = self.make_notes_service(profile="partner")
        service.settings = replace(service.settings, allowed_users=frozenset({"test-user", "second-user"}))
        self.search(service)
        for changes in ({"user_id": "second-user"}, {"chat_id": "second-chat"}):
            self.exchange(service, "笔记问答：总结第1条", **changes)
        self.exchange(partner, "笔记问答：总结第1条")
        self.assertEqual(0, answer.call_count)
        self.assertEqual(0, partner_answer.call_count)

    def test_unsupported_answer_citations_are_not_delivered_as_grounded_answers(self):
        for response in ("没有任何出处的合成回答", "错误引用。[99]"):
            with self.subTest(response=response):
                service, _, answer, _, _, chat_store, _, _, _ = self.make_notes_service()
                incoming, _ = self.search(service)
                answer.answer = response
                _, result = self.exchange(service, "笔记问答：比较这些方案")
                self.assertNotEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertNotIn(response, result.reply)
                self.assertEqual([], self.history(chat_store, incoming))

    def test_empty_overlong_or_invalid_date_search_never_reads_index_or_calls_model(self):
        for text in ("搜索笔记：", "搜索笔记：" + "字" * 201, "搜索笔记：2026-02-31 实验"):
            with self.subTest(text=text[:50]):
                service, index, answer, chat, _, _, _, _, _ = self.make_notes_service()
                self.exchange(service, text)
                self.assertEqual([], index.search_calls)
                self.assertEqual(0, answer.call_count)
                self.assertEqual(0, chat.call_count)

    def test_natural_comparison_without_note_focus_remains_normal_chat(self):
        service, index, answer, chat, _, _, _, _, _ = self.make_notes_service()
        _, result = self.exchange(service, "这些方案有什么区别")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual(1, chat.call_count)
        self.assertEqual([], index.search_calls)
        self.assertEqual(0, answer.call_count)

    def test_existing_link_summary_keeps_note_save_and_refresh_after_search(self):
        for text in ("帮我总结一下 https://example.com/article", "分析一下 https://example.com/article"):
            with self.subTest(text=text):
                service, index, answer, chat, _, _, dida, _, notify = self.make_notes_service()
                self.search(service)
                service.web = SimpleNamespace(read=Mock(return_value=WebPage(
                    source_url="https://example.com/article", final_url="https://example.com/article",
                    title="合成网页", text="合成公开资料正文。",
                )))
                _, result = self.exchange(service, text)
                self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
                self.assertEqual(1, len(index.search_calls))
                self.assertEqual(0, answer.call_count)
                self.assertEqual(0, chat.call_count)
                self.assertEqual([], dida.created)
                service.web.read.assert_called_once_with("https://example.com/article")
                notify.assert_called_once_with()
                self.assertIn("https://example.com/article", (service.settings.vault_path / service.settings.default_note_path).read_text(encoding="utf-8"))
