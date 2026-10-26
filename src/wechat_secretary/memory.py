"""Explicit personal facts, deterministic commands and rollback detection.

The external guard contains no fact text and must not be restored together
with assistant.sqlite3. A detected rollback disables access rather than
silently reviving forgotten facts. Restoring both files cannot be detected
without an independently retained authority; that is outside this boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, time as wall_time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Callable, Iterator, TypeVar
from uuid import uuid4


class MemoryUnavailable(RuntimeError):
    """Memory integrity or availability is unverified; do not use old context."""


def has_memory_state(path: Path | str) -> bool:
    """Read-only detection; uncertainty must trigger normal guard validation."""
    candidate = Path(path)
    try:
        candidate.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    connection = None
    try:
        connection = sqlite3.connect(candidate.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_meta'"
        ).fetchone() is not None
    except Exception:
        return True
    finally:
        if connection is not None:
            connection.close()


@dataclass(frozen=True)
class MemoryCommand:
    action: str
    content: str = ""
    target: str = ""
    kind: str = "fact"
    name: str = ""
    expires_at: datetime | None = None
    error: str = ""


@dataclass(frozen=True)
class MemoryEntry:
    memory_id: str
    content: str
    kind: str
    name: str
    source_message_id: str
    created_at: str
    updated_at: str
    expires_at: str
    version: int
    status: str


@dataclass(frozen=True)
class MemoryResult:
    status: str
    revision: int
    entry: MemoryEntry | None = None
    candidates: tuple[MemoryEntry, ...] = ()
    duplicate: bool = False
    cleared_chat: bool = False
    error: str = ""


@dataclass(frozen=True)
class MemorySnapshot:
    revision: int
    entries: tuple[MemoryEntry, ...] = ()


_KINDS = {"fact", "preference", "background", "goal", "project"}
_TYPE_WORDS = {"偏好": "preference", "背景": "background", "目标": "goal", "项目": "project"}
_MEMORY_ID = re.compile(r"M-[0-9a-f]{10}", re.IGNORECASE)
_NATURAL_KEY = re.compile(r"^我的(研究方向|名字|称呼|职业|所在地|工作单位|时区)(?:现在)?是")
_T = TypeVar("_T")


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip("。.!！").casefold()


def _date_option(option: str, now: datetime) -> tuple[datetime | None, str]:
    if not option:
        return None, ""
    match = re.fullmatch(r"[\[【]有效至\s*(\d{4}-\d{2}-\d{2})[\]】]", option)
    if match is None or now.tzinfo is None or now.utcoffset() is None:
        return None, "有效期请使用 [有效至2026-11-01] 这样的明确日期"
    try:
        day = datetime.strptime(match[1], "%Y-%m-%d").date()
        # The named calendar day is included. The following midnight is the
        # exclusive expiry boundary in the caller's configured time zone.
        expires = datetime.combine(day + timedelta(days=1), wall_time.min, now.tzinfo)
    except (ValueError, OverflowError):
        return None, "有效期日期无效"
    if expires <= now:
        return None, "有效期必须是今天或未来的日期"
    return expires, ""


def parse_memory_command(text: str, now: datetime) -> MemoryCommand | None:
    """Parse only outer, explicit management commands; never infer from chat."""
    text = (text or "").strip()
    listing = text.rstrip("。.!！?？ ")
    if listing in {"我的记忆", "查看记忆", "列出记忆", "你记住了我哪些事情", "你记住了什么"}:
        return MemoryCommand("list")
    match = re.fullmatch(r"(?:查看记忆|查找记忆)\s+(.+)", listing, flags=re.DOTALL)
    if match:
        return MemoryCommand("list", target=match[1].strip())

    # A colon is required: "记住明天提醒我..." remains the pre-existing
    # task grammar, not an accidental personal-memory command.
    remember = re.fullmatch(
        r"(?:请|帮我)?记住(?P<kind>偏好|背景|目标|项目)?"
        r"(?P<name>\s+[^：:\[【\r\n]+?)?\s*"
        r"(?P<expiry>[\[【][^\]】\r\n]*[\]】])?\s*[：:]\s*(?P<body>.*)",
        text, flags=re.DOTALL,
    )
    if remember:
        content = remember["body"].strip()
        name = (remember["name"] or "").strip()
        expires, error = _date_option(remember["expiry"] or "", now)
        kind = _TYPE_WORDS.get(remember["kind"] or "", "fact")
        if not content:
            error = "请在记住：后面写出要保存的内容"
        if len(name) > 60:
            error = "记忆名称最多 60 字"
        return MemoryCommand("remember", content=content, kind=kind, name=name, expires_at=expires, error=error)

    correction = re.fullmatch(
        r"(?:请|帮我)?更正记忆\s*(?P<target>[^：:\[【\r\n]*?)\s*"
        r"(?P<expiry>[\[【][^\]】\r\n]*[\]】])?\s*[：:]\s*(?P<body>.*)",
        text, flags=re.DOTALL,
    )
    if correction:
        expires, error = _date_option(correction["expiry"] or "", now)
        target, content = correction["target"].strip(), correction["body"].strip()
        if not target or not content:
            error = "请使用 更正记忆 M-编号：新内容，或写出明确的记忆名称"
        return MemoryCommand("correct", content=content, target=target, expires_at=expires, error=error)

    forget = re.fullmatch(r"(?:请|帮我)?忘记记忆(?:\s+|[：:])(.+)", listing, flags=re.DOTALL)
    if forget is None:
        forget = re.fullmatch(r"(?:请|帮我)?忘记关于(.+?)的那条记忆", listing, flags=re.DOTALL)
    if forget:
        return MemoryCommand("forget", target=forget[1].strip())
    if re.match(r"^(?:请|帮我)?(?:更正记忆|忘记记忆)(?:\s|[：:]|$)", text):
        return MemoryCommand("invalid", error="请写出记忆编号或明确名称；更正时还需提供新内容")
    # An explicitly delimited but malformed memory option is not ordinary
    # chat and must not silently fall through to another action.
    if re.match(r"^(?:请|帮我)?记住(?:偏好|背景|目标|项目)?(?:\s|[\[【：:])", text) and re.search(r"[：:]", text):
        return MemoryCommand("invalid", error="记忆命令格式不完整；可用 记住：内容")
    return None


class MemoryStore:
    """Explicit facts and source controls in the profile's assistant database.

    Every fact mutation rotates the corresponding chat generation and erases
    that conversation's retained text in the same SQLite transaction. This
    deliberately trades chat continuity for reliable correction/forgetting.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        guard_path: Path | str | None = None,
        clock: Callable[[], datetime] | None = None,
        max_entries: int = 100,
        max_entry_chars: int = 500,
    ):
        if not 1 <= max_entries <= 100 or not 1 <= max_entry_chars <= 2000:
            raise ValueError("记忆数量或正文长度限制无效")
        self._path = str(path)
        self._guard_path = Path(guard_path) if guard_path is not None else None
        if self._path != ":memory:" and self._guard_path is None:
            raise ValueError("磁盘记忆库必须配置独立的恢复控制文件")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._max_entries = max_entries
        self._max_entry_chars = max_entry_chars
        self._lock = threading.RLock()
        self._closed = False
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None, timeout=5,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            row = self._connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            expected = "memory" if self._path == ":memory:" else "delete"
            if not row or str(row[0]).casefold() != expected:
                raise MemoryUnavailable("memory-journal-unavailable")
            self._connection.execute("PRAGMA synchronous=EXTRA")
            if self._connection.execute("PRAGMA synchronous").fetchone()[0] != 3:
                raise MemoryUnavailable("memory-durability-unavailable")
            self._connection.execute("PRAGMA secure_delete=ON")
            self._initialize()
            self.expire()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    @staticmethod
    def _key(value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("记忆会话标识为空")
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise MemoryUnavailable("memory-clock-invalid")
        return now.astimezone(timezone.utc)

    def _write_guard(self, database_id: str, revision: int, pending: int | None = None) -> None:
        if self._guard_path is None:
            return
        self._guard_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": 1, "database_id": database_id, "revision": revision, "pending": pending}
        temporary = self._guard_path.with_name(f".{self._guard_path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._guard_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _validate_guard_locked(self) -> sqlite3.Row:
        meta = self._connection.execute("SELECT database_id, revision FROM memory_meta WHERE singleton = 1").fetchone()
        if meta is None:
            raise MemoryUnavailable("memory-metadata-missing")
        if self._guard_path is None:
            return meta
        try:
            if self._guard_path.stat().st_size > 4096:
                raise ValueError("oversize")
            guard = json.loads(self._guard_path.read_text(encoding="utf-8"))
        except Exception:
            raise MemoryUnavailable("memory-restore-guard-missing-or-invalid") from None
        if (
            not isinstance(guard, dict)
            or guard.get("schema") != 1
            or guard.get("database_id") != meta["database_id"]
            or type(guard.get("revision")) is not int
            or guard["revision"] < 0
        ):
            raise MemoryUnavailable("memory-restore-guard-mismatch")
        revision = int(meta["revision"])
        pending = guard.get("pending")
        if pending is not None:
            if type(pending) is not int or pending != guard["revision"] + 1 or revision != pending:
                raise MemoryUnavailable("memory-interrupted-write-unverified")
            # The complete SQLite transaction reached the prepared revision;
            # finish only the text-free control acknowledgement after a crash.
            self._write_guard(str(meta["database_id"]), revision)
        elif guard["revision"] != revision:
            raise MemoryUnavailable("memory-restore-revision-mismatch")
        return meta

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                statements = (
                    """CREATE TABLE IF NOT EXISTS memory_meta (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        database_id TEXT NOT NULL, revision INTEGER NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS memory_scopes (
                        conversation_hash TEXT PRIMARY KEY, revision INTEGER NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS personal_memories (
                        memory_id TEXT PRIMARY KEY, conversation_hash TEXT NOT NULL,
                        content TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
                        normalized_key TEXT NOT NULL, content_key TEXT NOT NULL,
                        source_message_id TEXT NOT NULL, created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL, expires_at TEXT NOT NULL DEFAULT '',
                        expires_timestamp REAL, version INTEGER NOT NULL, status TEXT NOT NULL)""",
                    """CREATE INDEX IF NOT EXISTS personal_memories_scope
                        ON personal_memories(conversation_hash, status)""",
                    """CREATE TABLE IF NOT EXISTS memory_operations (
                        conversation_hash TEXT NOT NULL, message_id TEXT NOT NULL,
                        request_hash TEXT NOT NULL, result_status TEXT NOT NULL,
                        memory_id TEXT NOT NULL, PRIMARY KEY(conversation_hash, message_id))""",
                    """CREATE TABLE IF NOT EXISTS memory_sources (
                        conversation_hash TEXT NOT NULL, memory_id TEXT NOT NULL,
                        source_message_id TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY(conversation_hash, memory_id, source_message_id))""",
                )
                for statement in statements:
                    self._connection.execute(statement)
                meta = self._connection.execute("SELECT database_id, revision FROM memory_meta WHERE singleton = 1").fetchone()
                if meta is None:
                    if self._guard_path is not None and self._guard_path.exists():
                        raise MemoryUnavailable("memory-database-missing-with-existing-guard")
                    database_id = uuid4().hex
                    self._write_guard(database_id, 0)
                    self._connection.execute("INSERT INTO memory_meta VALUES (1, ?, 0)", (database_id,))
                else:
                    self._validate_guard_locked()
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def _revision_locked(self, key: str, changed: set[str]) -> int:
        row = self._connection.execute(
            "SELECT revision FROM memory_scopes WHERE conversation_hash = ?", (key,),
        ).fetchone()
        return (int(row[0]) if row else 0) + int(key in changed)

    def _erase_entry_locked(self, row: sqlite3.Row, status: str, now: datetime) -> None:
        self._connection.execute(
            """UPDATE personal_memories SET content='', name='', normalized_key='', content_key='',
               status=?, updated_at=?, version=version+1 WHERE memory_id=?""",
            (status, now.isoformat(), row["memory_id"]),
        )
        self._connection.execute(
            "UPDATE memory_sources SET revoked=1 WHERE conversation_hash=? AND memory_id=?",
            (row["conversation_hash"], row["memory_id"]),
        )

    def _expire_locked(self, now: datetime, changed: set[str]) -> int:
        expired = self._connection.execute(
            "SELECT * FROM personal_memories WHERE status='active' AND expires_timestamp <= ?", (now.timestamp(),),
        ).fetchall()
        for row in expired:
            self._erase_entry_locked(row, "expired", now)
            changed.add(str(row["conversation_hash"]))
        return len(expired)

    def _reset_chat_locked(self, key: str) -> None:
        tables = {str(row[0]) for row in self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chat_sessions','chat_turns')"
        )}
        if "chat_sessions" in tables:
            self._connection.execute(
                """INSERT INTO chat_sessions(conversation_hash,generation) VALUES (?,?)
                   ON CONFLICT(conversation_hash) DO UPDATE SET generation=excluded.generation""",
                (key, uuid4().hex),
            )
        if "chat_turns" in tables:
            self._connection.execute("DELETE FROM chat_turns WHERE conversation_hash=?", (key,))

    def _run(self, operation: Callable[[datetime, set[str], int], _T]) -> _T:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                meta = self._validate_guard_locked()
                now = self._now()
                changed: set[str] = set()
                expired = self._expire_locked(now, changed)
                result = operation(now, changed, expired)
                revision = int(meta["revision"])
                if changed:
                    for key in changed:
                        self._connection.execute(
                            """INSERT INTO memory_scopes VALUES (?,1) ON CONFLICT(conversation_hash)
                               DO UPDATE SET revision=memory_scopes.revision+1""", (key,),
                        )
                        self._reset_chat_locked(key)
                    self._connection.execute("UPDATE memory_meta SET revision=revision+1 WHERE singleton=1")
                    # Persist intent before committing facts. If the following
                    # transaction fails, leave pending in place and fail closed.
                    self._write_guard(str(meta["database_id"]), revision, revision + 1)
                self._connection.execute("COMMIT")
                if changed:
                    self._write_guard(str(meta["database_id"]), revision + 1)
                return result
            except Exception as exc:
                if self._connection.in_transaction:
                    try:
                        self._connection.execute("ROLLBACK")
                    except Exception:
                        pass
                if isinstance(exc, MemoryUnavailable):
                    raise
                raise MemoryUnavailable("memory-operation-unavailable") from None

    @staticmethod
    def _entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(**{name: row[name] for name in MemoryEntry.__dataclass_fields__})

    def _active_locked(self, key: str) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM personal_memories WHERE conversation_hash=? AND status='active' ORDER BY updated_at DESC, memory_id",
            (key,),
        ).fetchall()

    @staticmethod
    def _targets(rows: list[sqlite3.Row], target: str) -> list[sqlite3.Row]:
        wanted = _normalized(target)
        if not wanted:
            return []
        if _MEMORY_ID.fullmatch(wanted):
            return [row for row in rows if str(row["memory_id"]).casefold() == wanted]
        exact = [row for row in rows if wanted in {_normalized(str(row["name"])), str(row["content_key"])}]
        return exact or [row for row in rows if wanted in _normalized(str(row["name"]) + " " + str(row["content"]))]

    def apply(self, conversation_key: str, message_id: str, command: MemoryCommand) -> MemoryResult:
        key = self._key(conversation_key)
        if not message_id:
            raise ValueError("记忆操作需要稳定消息标识")
        request = {name: getattr(command, name) for name in MemoryCommand.__dataclass_fields__}
        request["expires_at"] = command.expires_at.isoformat() if command.expires_at is not None else None
        fingerprint = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

        def operation(now: datetime, changed: set[str], expired: int) -> MemoryResult:
            del expired
            previous = self._connection.execute(
                "SELECT request_hash FROM memory_operations WHERE conversation_hash=? AND message_id=?", (key, message_id),
            ).fetchone()
            if previous is not None:
                return MemoryResult(
                    "duplicate" if previous[0] == fingerprint else "invalid", self._revision_locked(key, changed),
                    duplicate=True, cleared_chat=key in changed,
                    error="" if previous[0] == fingerprint else "相同消息标识对应不同内容，未执行",
                )
            rows = self._active_locked(key)
            entry: MemoryEntry | None = None
            candidates: tuple[MemoryEntry, ...] = ()
            status, error = "invalid", command.error
            content = command.content.strip()
            if not error and command.action in {"remember", "correct"}:
                if not content or len(content) > self._max_entry_chars:
                    error = f"记忆正文需要 1 到 {self._max_entry_chars} 字"
                elif command.kind not in _KINDS or len(command.name.strip()) > 60:
                    error = "记忆类型或名称无效"
                elif command.expires_at is not None and (
                    command.expires_at.tzinfo is None or command.expires_at.utcoffset() is None
                    or command.expires_at <= now
                ):
                    error = "有效期必须是包含时区的未来时间"
            if not error and command.action == "list":
                selected = rows if not command.target else self._targets(rows, command.target)
                candidates = tuple(self._entry(row) for row in selected)
                status = "list"
            elif not error and command.action == "remember":
                name = command.name.strip()
                kind = command.kind
                natural = _NATURAL_KEY.match(content)
                if not name and natural:
                    name = natural[1]
                    if kind == "fact":
                        kind = "background"
                if not name and re.match(r"^(?:以后)?(?:请)?(?:回答|回复)", content):
                    name = "回复风格"
                    if kind == "fact":
                        kind = "preference"
                normalized_key = _normalized(name)
                same_key = [row for row in rows if normalized_key and row["normalized_key"] == normalized_key]
                exact = [row for row in rows if row["content_key"] == _normalized(content)]
                if same_key and any(row["content_key"] != _normalized(content) for row in same_key):
                    status, candidates = "conflict", tuple(self._entry(row) for row in same_key)
                elif exact:
                    status, entry = "duplicate", self._entry(exact[0])
                elif len(rows) >= self._max_entries:
                    status, error = "invalid", "记忆数量已达上限，请先更正或删除不再需要的条目"
                else:
                    memory_id = "M-" + uuid4().hex[:10]
                    expires = command.expires_at.astimezone(timezone.utc) if command.expires_at is not None else None
                    self._connection.execute(
                        """INSERT INTO personal_memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,'active')""",
                        (memory_id, key, content, kind, name, normalized_key, _normalized(content), message_id,
                         now.isoformat(), now.isoformat(), expires.isoformat() if expires else "", expires.timestamp() if expires else None),
                    )
                    self._connection.execute("INSERT INTO memory_sources VALUES (?,?,?,0)", (key, memory_id, message_id))
                    entry = self._entry(self._connection.execute("SELECT * FROM personal_memories WHERE memory_id=?", (memory_id,)).fetchone())
                    changed.add(key)
                    status = "added"
            elif not error and command.action in {"correct", "forget"}:
                matches = self._targets(rows, command.target)
                if not command.target.strip():
                    status, error = "invalid", "请写出明确的记忆编号或名称"
                elif not matches:
                    status = "not_found"
                elif len(matches) != 1:
                    status, candidates = "ambiguous", tuple(self._entry(row) for row in matches)
                else:
                    old = matches[0]
                    if command.action == "forget":
                        self._erase_entry_locked(old, "forgotten", now)
                        status = "forgotten"
                    else:
                        expires = command.expires_at.astimezone(timezone.utc) if command.expires_at is not None else None
                        self._connection.execute(
                            """UPDATE personal_memories SET content=?,content_key=?,source_message_id=?,updated_at=?,
                               expires_at=?,expires_timestamp=?,version=version+1 WHERE memory_id=?""",
                            (content, _normalized(content), message_id, now.isoformat(),
                             expires.isoformat() if expires else old["expires_at"],
                             expires.timestamp() if expires else old["expires_timestamp"], old["memory_id"]),
                        )
                        self._connection.execute("UPDATE memory_sources SET revoked=1 WHERE conversation_hash=? AND memory_id=?", (key, old["memory_id"]))
                        self._connection.execute("INSERT INTO memory_sources VALUES (?,?,?,0)", (key, old["memory_id"], message_id))
                        status = "updated"
                    entry = self._entry(self._connection.execute("SELECT * FROM personal_memories WHERE memory_id=?", (old["memory_id"],)).fetchone())
                    changed.add(key)
            elif not error:
                error = "不支持的记忆操作"
            self._connection.execute(
                "INSERT INTO memory_operations VALUES (?,?,?,?,?)",
                (key, message_id, fingerprint, status, entry.memory_id if entry else ""),
            )
            return MemoryResult(status, self._revision_locked(key, changed), entry, candidates,
                                duplicate=status == "duplicate", cleared_chat=key in changed, error=error)
        return self._run(operation)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        ignored = {"什么", "哪些", "记得", "我的", "之前", "关于", "那个", "现在", "怎么", "一下", "帮我", "请问", "可以", "事情", "告诉", "我们", "你们"}
        tokens = set(re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", text.casefold()))
        for phrase in re.findall(r"[\u3400-\u9fff]+", text):
            tokens.update(phrase[index:index + 2] for index in range(len(phrase) - 1))
        return tokens - ignored

    def snapshot(self, conversation_key: str, query: str = "", max_items: int = 6, max_chars: int = 2000) -> MemorySnapshot:
        key = self._key(conversation_key)
        def operation(now: datetime, changed: set[str], expired: int) -> MemorySnapshot:
            del now, expired
            rows = self._active_locked(key)
            query_tokens = self._tokens(query[:6000])
            preferences = [row for row in rows if row["kind"] == "preference"]
            relevant = []
            for row in rows:
                if row["kind"] == "preference":
                    continue
                score = len(query_tokens & self._tokens(str(row["name"]) + " " + str(row["content"])))
                if score:
                    relevant.append((score, row))
            relevant.sort(key=lambda item: item[0], reverse=True)
            ordered = preferences[:3] + [row for _, row in relevant] + preferences[3:]
            entries: list[MemoryEntry] = []
            used = 0
            for row in ordered:
                entry = self._entry(row)
                cost = len(entry.content) + len(entry.name) + len(entry.memory_id) + len(entry.kind) + 64
                if len(entries) >= min(6, max(0, max_items)):
                    break
                if used + cost > min(2000, max(0, max_chars)):
                    continue
                entries.append(entry)
                used += cost
            return MemorySnapshot(self._revision_locked(key, changed), tuple(entries))
        return self._run(operation)

    def list_entries(self, conversation_key: str, include_inactive: bool = False) -> tuple[MemoryEntry, ...]:
        key = self._key(conversation_key)
        def operation(now: datetime, changed: set[str], expired: int) -> tuple[MemoryEntry, ...]:
            del now, changed, expired
            rows = self._connection.execute(
                "SELECT * FROM personal_memories WHERE conversation_hash=? "
                + ("" if include_inactive else "AND status='active' ") + "ORDER BY updated_at DESC, memory_id", (key,),
            ).fetchall()
            return tuple(self._entry(row) for row in rows)
        return self._run(operation)

    def current_revision(self, conversation_key: str) -> int:
        key = self._key(conversation_key)
        return self._run(lambda now, changed, expired: self._revision_locked(key, changed))

    def is_current(self, conversation_key: str, revision: int) -> bool:
        return self.current_revision(conversation_key) == revision

    def expire(self) -> int:
        return self._run(lambda now, changed, expired: expired)

    @contextmanager
    def delivery_guard(self) -> Iterator[None]:
        """Serialize the short final send-boundary check with local mutations."""
        with self._lock:
            yield

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True
