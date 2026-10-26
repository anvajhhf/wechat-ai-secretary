"""Short-lived note focus; never copied into generic chat or memory."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from uuid import uuid4

from .models import HandlingResult, ExecutionStatus
from .ledger import CHAT_FAILURE_PREFIX
from .notes_answer import validate_answer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Focus:
    context_id: str
    expires: float
    snapshot: object | None = None


class NotesServiceMixin:
    def _init_notes(self, index, answer, notify):
        self.notes_index, self.notes_answer, self.notes_notify = index, answer, notify
        self._notes_lock = threading.RLock()
        self._notes_contexts: dict[str, _Focus] = {}

    def _clear_notes_focus(self, key):
        with self._notes_lock:
            self._notes_contexts.pop(key, None)

    def _notes_focus(self, key):
        with self._notes_lock:
            focus = self._notes_contexts.get(key)
            if focus and time.monotonic() >= focus.expires:
                self._notes_contexts.pop(key, None)
                return None
            return focus

    def _notes_is_current(self, message, result):
        with self._notes_lock:
            focus = self._notes_focus(message.conversation_key)
            return bool(self.settings.notes_search_enabled and focus
                        and focus.context_id == result.notes_context_id
                        and result.notes_snapshot is not None and self.notes_index is not None
                        and self.notes_index.validate(result.notes_snapshot))

    def _notes_record_delivery(self, message, result):
        with self._notes_lock:
            if result.notes_activate and self._notes_is_current(message, result):
                focus = self._notes_contexts[message.conversation_key]
                self._notes_contexts[message.conversation_key] = replace(focus, snapshot=result.notes_snapshot)

    @staticmethod
    def _notes_sources(hits, *, excerpts=False):
        lines = []
        clean = lambda value, size: " ".join(str(value or "").split())[:size]
        for number, hit in enumerate(hits, 1):
            lines.append(f"[{number}] {clean(hit.title, 120)}｜{clean(hit.recorded_at or '日期未知', 40)}\n"
                         f"{clean(hit.relative_path, 300)} · 行{hit.line_start}–{hit.line_end} · {hit.source_id}")
            if excerpts:
                lines.append(clean(hit.excerpt, 180) + ("…" if len(hit.excerpt) > 180 else ""))
        return "\n".join(lines)

    def _notes_finish(self, message, text, **kwargs):
        self.ledger.finish(message, ExecutionStatus.SUCCEEDED)
        return HandlingResult(status=ExecutionStatus.SUCCEEDED, reply=text, chat_reset_focus=True, **kwargs)

    def _handle_notes(self, message, command, *, images=False):
        called = False
        key = message.conversation_key
        if command.action == "search":
            self._clear_notes_focus(key)
        try:
            self.ledger.reset_conversation_focus(key, datetime.now(self.settings.tz))
            if images:
                return self._notes_finish(message, "笔记检索请单独发送文字；图片不会作为搜索条件。")
            if command.error:
                return self._notes_finish(message, command.error)
            if self.notes_index is None:
                return self._notes_finish(message, "笔记检索当前不可用；聊天、长期记忆、提醒和笔记保存仍可使用。")
            if not self.settings.notes_search_enabled and command.action != "exclude":
                return self._notes_finish(message, "笔记检索当前未开启；现有功能仍可使用。")
            if command.action == "search":
                context_id = uuid4().hex
                with self._notes_lock:
                    # Drop expired contexts without retaining excerpts indefinitely.
                    now = time.monotonic()
                    self._notes_contexts = {k: f for k, f in self._notes_contexts.items() if f.expires > now}
                    if len(self._notes_contexts) >= 1000:
                        self._notes_contexts.clear()
                    self._notes_contexts[key] = _Focus(context_id, now + self.settings.notes_context_ttl_seconds)
                generation = ""
                if self.chat_store is not None:
                    self.chat_store.reset(key, clear=False)
                    generation = self.chat_store.get_generation(key)
                snapshot = self.notes_index.search(command.query, limit=5)
                if not snapshot.hits:
                    return self._notes_finish(message, "在允许检索的笔记中没有找到匹配结果。日期不明的记录不会进入日期筛选结果。可缩短关键词再试。")
                return self._notes_finish(message,
                    "找到这些笔记片段（最多5条）：\n" + self._notes_sources(snapshot.hits, excerpts=True)
                    + "\n可发送“笔记问答：这些方案有什么区别”或“第2条再详细说说”。结果10分钟内有效；“排除笔记 2”会排除该条所在整个文件，不删除原笔记。",
                    notes_guarded=True, notes_snapshot=snapshot, notes_context_id=context_id,
                    notes_activate=True, chat_generation=generation)
            if command.action == "rebuild":
                if self.settings.dry_run:
                    return self._notes_finish(message, "Dry Run｜没有重建索引。")
                self._clear_notes_focus(key)
                self.notes_index.rebuild()
                return self._notes_finish(message, "已重建允许范围内的笔记索引；原笔记与持久排除记录保留。请重新搜索。")
            focus = self._notes_focus(key)
            if command.action == "exclude":
                source_id = command.text
                if command.number is not None:
                    if not focus or focus.snapshot is None or not self.notes_index.validate(focus.snapshot):
                        return self._notes_finish(message, "搜索结果已失效，请重新搜索后再按编号排除，或使用已知来源ID。")
                    if not 1 <= command.number <= len(focus.snapshot.hits):
                        return self._notes_finish(message, "这个编号不在当前笔记结果中，没有排除任何来源。")
                    source_id = focus.snapshot.hits[command.number - 1].source_id
                if self.settings.dry_run:
                    return self._notes_finish(message, "Dry Run｜没有修改笔记排除记录。")
                outcome = self.notes_index.exclude_source(source_id)
                self._clear_notes_focus(key)
                if outcome.status == "not_found":
                    return self._notes_finish(message, "没有找到这个来源ID，没有改动。")
                return self._notes_finish(message, "已排除该来源整个文件的检索，包含同文件其他笔记；重建索引也不会恢复它。原笔记保留，已发微信消息不会撤回。")
            if not focus or focus.snapshot is None:
                return self._notes_finish(message, "请先发送“搜索笔记：关键词”，再根据收到的结果提问。")
            if not self.notes_index.validate(focus.snapshot):
                self._clear_notes_focus(key)
                return self._notes_finish(message, "笔记来源已变化，请重新搜索后再提问。")
            snapshot = focus.snapshot
            if command.number is not None:
                if not 1 <= command.number <= len(snapshot.hits):
                    return self._notes_finish(message, "这个编号不在当前笔记结果中，请使用结果中的编号。")
                snapshot = replace(snapshot, hits=(snapshot.hits[command.number - 1],))
            if not command.text or len(command.text) > 1000:
                return self._notes_finish(message, "请在“笔记问答：”后写下不超过1000字的问题。")
            if self.notes_answer is None:
                return self._notes_finish(message, "笔记问答暂不可用，仍可使用本地搜索。")
            called = True
            reply = validate_answer(self.notes_answer.respond(command.text, snapshot.hits), len(snapshot.hits))
            result = self._notes_finish(message, reply + "\n\n来源：\n" + self._notes_sources(snapshot.hits),
                                       llm_called=True, notes_guarded=True, notes_snapshot=snapshot,
                                       notes_context_id=focus.context_id)
            return self.prepare_reply_delivery(message, result)
        except Exception as exc:
            logger.warning("Secretary notes request unavailable: %s", type(exc).__name__)
            self._clear_notes_focus(key)
            self.ledger.finish(message, ExecutionStatus.FAILED, error_code=CHAT_FAILURE_PREFIX + "notes-" + type(exc).__name__)
            return HandlingResult(status=ExecutionStatus.FAILED, llm_called=called, chat_reset_focus=True,
                                  reply="这次笔记检索或问答未完成，没有使用过期结果。聊天、长期记忆、提醒和笔记保存仍可使用。请稍后重新搜索。")
