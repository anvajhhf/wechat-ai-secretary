"""Bounded local Markdown retrieval with independently retained exclusions.

The control file is authoritative and must not be restored with the rebuildable
index. Restoring both to an older state, or removing both, cannot be detected by
this local authority. Unknown, independently edited copies are not identifiable.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import stat
import threading
from typing import Callable, Iterator
import unicodedata
from uuid import uuid4


class NotesUnavailable(RuntimeError):
    """Source safety or retrieval completeness could not be established."""


@dataclass(frozen=True)
class NotesQuery:
    text: str = ""
    tags: tuple[str, ...] = ()
    date_from: date | None = None
    date_to: date | None = None


@dataclass(frozen=True)
class NotesHit:
    source_id: str
    chunk_id: str
    relative_path: str
    title: str
    recorded_at: str
    tags: tuple[str, ...]
    excerpt: str
    line_start: int
    line_end: int
    version: str
    source_ref: str = ""


@dataclass(frozen=True)
class NotesSnapshot:
    revision: int
    hits: tuple[NotesHit, ...] = ()


@dataclass(frozen=True)
class ScanResult:
    scanned: int
    indexed: int
    removed: int
    excluded: int
    revision: int
    incomplete: bool = False


@dataclass(frozen=True)
class NotesExclusionResult:
    status: str
    revision: int
    source_id: str


@dataclass(frozen=True)
class _Source:
    relative_path: str
    source_id: str
    path_hash: str
    identity_hash: str
    version: str
    size: int
    text: str


class _RejectedSource(Exception):
    pass


_REF = re.compile(r"<!--\s*wechat-ai-secretary:ref=([0-9a-f]{16})\s*-->", re.I)
_SOURCE_LINE = re.compile(r"^>\s*来源：微信｜时间：([^｜\n]+)｜引用：([0-9a-f]{16})\s*$", re.M | re.I)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_HEX = re.compile(r"[0-9a-f]{64}")
_PRIVATE = re.compile(r"wechat-ai-secretary:private-ref=", re.I)


def _hash(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _normal(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _tokens(value: str) -> set[str]:
    value = _normal(value)
    output = set(re.findall(r"[a-z0-9]+", value))
    for word in re.findall(r"[\u3400-\u9fff]+", value):
        output.update(word[index:index + 2] for index in range(max(1, len(word) - 1)))
    return output


def has_notes_state(path: Path | str) -> bool:
    """Read-only detection; uncertain state must trigger integrity checking."""
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
        return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='notes_meta'").fetchone() is not None
    except Exception:
        return True
    finally:
        if connection is not None:
            connection.close()


def _frontmatter(lines: list[str]) -> tuple[dict[str, str], int]:
    if not lines or lines[0].strip() != "---":
        return {}, 0
    for end in range(1, min(len(lines), 100)):
        if lines[end].strip() == "---":
            values = {}
            for line in lines[1:end]:
                match = re.fullmatch(r"([A-Za-z_]+):\s*(.*?)\s*", line)
                if match:
                    values[match[1].casefold()] = match[2].strip("\"'")
            return values, end + 1
    return {}, 0


def _record_date(value: str) -> str:
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value).isoformat()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed.astimezone(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    except (ValueError, TypeError):
        pass
    return ""


def _private_text(text: str) -> bool:
    if _PRIVATE.search(text):
        return True
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    seen = set()
    for line in lines[1:100]:
        if line.strip() == "---":
            return False
        match = re.match(r"\s*(private|privacy)\s*:\s*(.*?)\s*$", line, re.I)
        if not match:
            continue
        key, value = match[1].casefold(), match[2]
        if key in seen:
            return True
        seen.add(key)
        parsed = re.fullmatch(r'''(?:"([A-Za-z0-9]+)"|'([A-Za-z0-9]+)'|([A-Za-z0-9]+))\s*(?:#.*)?''', value)
        if not parsed:
            return True
        value = next(item for item in parsed.groups() if item is not None).casefold()
        public = {"false", "no", "0"} if key == "private" else {"public"}
        if value not in public:
            return True
    # Do not interpret an unterminated or over-budget metadata region as
    # ordinary prose: a private flag may exist beyond the parsing budget.
    return True


def _chunks(source: _Source) -> tuple[NotesHit, ...]:
    lines = source.text.splitlines()
    metadata, first = _frontmatter(lines)
    default_date = _record_date(metadata.get("date", "") or metadata.get("created", "") or metadata.get("created_at", ""))
    default_title = metadata.get("title", "")[:200] or Path(source.relative_path).stem[:200]
    default_tags = re.findall(r"[^\s,\[\]'\"]+", metadata.get("tags", ""))
    boundaries = {first, len(lines)}
    for index in range(first, len(lines)):
        if re.match(r"^##\s+", lines[index]):
            boundaries.add(index)
        if _REF.search(lines[index]):
            boundaries.add(index + 1)
    points = sorted(boundaries)
    hits = []
    for begin, end in zip(points, points[1:]):
        segment = "\n".join(lines[begin:end])
        refs = _REF.findall(segment)
        source_ref = refs[-1].lower() if refs else ""
        recorded_at = default_date
        for match in _SOURCE_LINE.finditer(segment):
            if source_ref and match[2].lower() == source_ref:
                recorded_at = _record_date(match[1].strip()) or recorded_at
        title = next((match[1][:200] for line in lines[begin:end] if (match := _HEADING.match(line))), default_title)
        tags = tuple(sorted({_normal(tag).lstrip("#")[:60] for tag in default_tags + re.findall(r"(?<!\w)#([\w\u3400-\u9fff/-]+)", segment) if tag})[:30])
        # Location refers to the original Markdown lines, including when a long
        # single line is divided into several bounded blocks.
        pieces: list[tuple[str, int]] = []
        for index in range(begin, end):
            line = lines[index]
            if _REF.search(line) or _SOURCE_LINE.fullmatch(line):
                continue
            if not line.strip():
                pieces.append(("", index + 1))
                continue
            pieces.extend((line[offset:offset + 1500], index + 1) for offset in range(0, len(line), 1500))
        block = ""
        start_line = end_line = 0
        ordinal = 0

        def emit() -> None:
            nonlocal block, ordinal
            excerpt = block.strip()
            if excerpt:
                chunk_id = "N-" + _hash(f"{source.source_id}|{source_ref}|{begin}|{ordinal}")[:16]
                hits.append(NotesHit(source.source_id, chunk_id, source.relative_path, title, recorded_at, tags, excerpt, start_line, end_line, source.version, source_ref))
                ordinal += 1
            block = ""

        for piece, line_number in pieces:
            if block and len(block) + len(piece) + 1 > 1500:
                emit()
            if not block:
                start_line = line_number
                block = piece
            else:
                block += "\n" + piece
            end_line = line_number
        emit()
    return tuple(hits)


class NotesIndex:
    """One configured profile's index; callers cannot select arbitrary paths."""

    def __init__(self, path: Path | str, *, vault_path: Path | str, allowed_paths: tuple[str, ...] | list[str],
                 private_paths: tuple[Path | str, ...] = (), control_path: Path | str,
                 clock: Callable[[], datetime] | None = None, max_file_bytes: int = 2_000_000,
                 max_files: int = 2000, max_total_bytes: int = 20_000_000):
        self.path = Path(path)
        self.vault_path = Path(os.path.abspath(vault_path))
        self.control_path = Path(control_path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._closed = False
        self._connection = None
        self._control: dict = {}
        self._high_water = -1
        self.max_file_bytes = int(max_file_bytes)
        self.max_files = int(max_files)
        self.max_total_bytes = int(max_total_bytes)
        if min(self.max_file_bytes, self.max_files, self.max_total_bytes) < 1:
            raise ValueError("notes scan limits must be positive")
        allowed = []
        for value in allowed_paths:
            raw = str(value).replace("\\", "/")
            candidate = Path(raw)
            if not raw or candidate.is_absolute() or PureWindowsPath(raw).drive or any(part in {"", ".", ".."} or part.startswith(".") for part in raw.split("/")):
                raise ValueError("notes allowlist must contain visible relative paths")
            allowed.append(candidate.as_posix())
        if not allowed:
            raise ValueError("notes allowlist cannot be empty")
        self.allowed_paths = tuple(sorted(set(allowed)))
        self.private_paths = tuple(Path(os.path.abspath(value)) for value in private_paths)
        self._config = _hash(json.dumps({"vault": os.path.normcase(str(self.vault_path)), "allowed": [os.path.normcase(value) for value in self.allowed_paths], "private": sorted(os.path.normcase(str(value)) for value in self.private_paths)}, sort_keys=True))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path, timeout=5, isolation_level=None, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            for pragma in ("PRAGMA journal_mode=DELETE", "PRAGMA synchronous=EXTRA", "PRAGMA secure_delete=ON", "PRAGMA foreign_keys=ON"):
                self._connection.execute(pragma)
            had_meta = self._connection.execute("SELECT 1 FROM sqlite_master WHERE name='notes_meta'").fetchone() is not None
            if self.control_path.exists():
                self._control = self._read_control()
            elif had_meta:
                raise NotesUnavailable("notes exclusion control is missing")
            else:
                self._control = {"schema": 1, "control_id": uuid4().hex, "revision": 0, "config": self._config, "excluded": []}
                self._write_control(self._control)
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS notes_meta(singleton INTEGER PRIMARY KEY CHECK(singleton=1),control_id TEXT NOT NULL,revision INTEGER NOT NULL,config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS note_files(source_id TEXT PRIMARY KEY,relative_path TEXT NOT NULL UNIQUE,path_hash TEXT NOT NULL,identity_hash TEXT NOT NULL,version TEXT NOT NULL,size INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS note_chunks(chunk_id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES note_files(source_id) ON DELETE CASCADE,title TEXT NOT NULL,recorded_at TEXT NOT NULL,tags TEXT NOT NULL,excerpt TEXT NOT NULL,line_start INTEGER NOT NULL,line_end INTEGER NOT NULL,source_ref TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS note_chunks_source ON note_chunks(source_id);
            """)
            meta = self._connection.execute("SELECT * FROM notes_meta WHERE singleton=1").fetchone()
            if meta and (meta["control_id"] != self._control["control_id"] or meta["revision"] > self._control["revision"]):
                raise NotesUnavailable("notes exclusion control cannot be verified")
            if self._control["config"] != self._config:
                updated = dict(self._control, config=self._config, revision=self._control["revision"] + 1)
                self._write_control(updated)
                self._control = updated
            self._high_water = self._control["revision"]
            if meta and meta["config"] != self._config:
                self._connection.execute("DELETE FROM note_files")
            self._save_meta()
        except Exception as exc:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if isinstance(exc, (NotesUnavailable, ValueError)):
                raise
            raise NotesUnavailable("notes index initialization failed") from None

    def _read_control(self) -> dict:
        try:
            if self.control_path.stat().st_size > 5_000_000:
                raise ValueError
            control = json.loads(self.control_path.read_text(encoding="utf-8"))
            if control.get("schema") != 1 or not re.fullmatch(r"[0-9a-f]{32}", control.get("control_id", "")) or type(control.get("revision")) is not int or control["revision"] < 0 or not _HEX.fullmatch(control.get("config", "")) or not isinstance(control.get("excluded"), list):
                raise ValueError
            for item in control["excluded"]:
                if not re.fullmatch(r"S-[0-9a-f]{20}", item.get("source_id", "")):
                    raise ValueError
                for key in ("paths", "identities", "versions"):
                    if not isinstance(item.get(key), list) or not item[key] or any(not isinstance(value, str) or not _HEX.fullmatch(value) for value in item[key]):
                        raise ValueError
            return control
        except Exception:
            raise NotesUnavailable("notes exclusion control cannot be verified") from None

    def _write_control(self, value: dict) -> None:
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.control_path.with_name(f".{self.control_path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.control_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _save_meta(self) -> None:
        self._connection.execute("INSERT OR REPLACE INTO notes_meta VALUES(1,?,?,?)", (self._control["control_id"], self._control["revision"], self._config))

    def _check_control(self) -> None:
        if self._closed:
            raise NotesUnavailable("notes index is closed")
        current = self._read_control()
        meta = self._connection.execute("SELECT * FROM notes_meta WHERE singleton=1").fetchone()
        if current["control_id"] != self._control["control_id"] or current["config"] != self._config or current["revision"] < self._high_water or not meta or current["revision"] < meta["revision"]:
            raise NotesUnavailable("notes exclusion control cannot be verified")
        self._control = current
        self._high_water = current["revision"]

    def _salted(self, value: str) -> str:
        return _hash(self._control["control_id"] + "|" + value)

    def _within(self, candidate: Path, root: Path) -> bool:
        try:
            return os.path.commonpath((os.path.normcase(str(candidate)), os.path.normcase(str(root)))) == os.path.normcase(str(root))
        except ValueError:
            return False

    def _allowed(self, relative: str) -> bool:
        candidate = Path(relative)
        if candidate.is_absolute() or PureWindowsPath(relative).drive or any(part in {".", ".."} or part.startswith(".") or part.casefold() == "privateinbox" for part in candidate.parts):
            return False
        path = self.vault_path / candidate
        if not self._within(path, self.vault_path) or any(self._within(path, private) or self._within(path, Path(os.path.realpath(private))) for private in self.private_paths):
            return False
        normalized = os.path.normcase(candidate.as_posix()).replace("\\", "/")
        return any(normalized == os.path.normcase(item).replace("\\", "/") or (not item.lower().endswith(".md") and normalized.startswith(os.path.normcase(item).replace("\\", "/").rstrip("/") + "/")) for item in self.allowed_paths)

    def _safe_stat(self, path: Path, *, directory: bool = False):
        try:
            relative = path.relative_to(self.vault_path)
            # A link above the configured vault is still a linked directory.
            # Check it before any content read, even when the vault itself is
            # an ordinary directory reached through that alias.
            for ancestor in reversed(self.vault_path.parents):
                ancestor_info = ancestor.lstat()
                if stat.S_ISLNK(ancestor_info.st_mode) or getattr(ancestor_info, "st_file_attributes", 0) & 0x400:
                    raise _RejectedSource
            current = self.vault_path
            for component in (None, *relative.parts):
                if component is not None:
                    if component.startswith(".") or component.casefold() == "privateinbox":
                        raise _RejectedSource
                    current = current / component
                info = current.lstat()
                attributes = getattr(info, "st_file_attributes", 0)
                if stat.S_ISLNK(info.st_mode) or attributes & (0x400 | 0x2):
                    raise _RejectedSource
                if current != path and not stat.S_ISDIR(info.st_mode):
                    raise _RejectedSource
            if directory:
                if not stat.S_ISDIR(info.st_mode):
                    raise _RejectedSource
            elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_ino <= 0:
                raise _RejectedSource
            return info
        except (FileNotFoundError, NotADirectoryError, ValueError):
            raise _RejectedSource from None
        except OSError:
            raise NotesUnavailable("notes source state is unavailable") from None

    def _read_source(self, relative: str) -> _Source:
        if not self._allowed(relative) or not relative.lower().endswith(".md"):
            raise _RejectedSource
        path = self.vault_path / relative
        before = self._safe_stat(path)
        if before.st_size > self.max_file_bytes:
            raise NotesUnavailable("notes file exceeds scan limit")
        try:
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise NotesUnavailable("notes source changed during reading")
                data = handle.read(self.max_file_bytes + 1)
                after = os.fstat(handle.fileno())
            current = self._safe_stat(path)
            if len(data) > self.max_file_bytes:
                raise NotesUnavailable("notes file exceeds scan limit")
            stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            # Windows Python can expose creation time via lstat.st_ctime and
            # change time via fstat.st_ctime. Compare ctime only within the
            # same API; compare identity, size and mtime across APIs.
            if (stamp(opened) != stamp(after) or opened.st_ctime_ns != after.st_ctime_ns
                    or stamp(before) != stamp(current) or before.st_ctime_ns != current.st_ctime_ns
                    or stamp(after) != stamp(current) or len(data) != after.st_size):
                raise NotesUnavailable("notes source changed during reading")
            text = data.decode("utf-8-sig")
        except (FileNotFoundError, NotADirectoryError):
            raise _RejectedSource from None
        except (OSError, UnicodeError):
            raise NotesUnavailable("notes source cannot be read") from None
        if _private_text(text):
            raise _RejectedSource
        identity = self._salted(f"identity:{current.st_dev}:{current.st_ino}")
        return _Source(relative, "S-" + identity[:20], self._salted("path:" + os.path.normcase(relative)), identity, _hash(data), len(data), text)

    def _matching_exclusion(self, source: _Source | sqlite3.Row) -> dict | None:
        get = (lambda key: getattr(source, key)) if isinstance(source, _Source) else (lambda key: source[key])
        for item in self._control["excluded"]:
            if get("source_id") == item["source_id"] or get("path_hash") in item["paths"] or get("identity_hash") in item["identities"] or get("version") in item["versions"]:
                return item
        return None

    def _track_exclusion(self, source: _Source, item: dict) -> None:
        values = {"paths": source.path_hash, "identities": source.identity_hash, "versions": source.version}
        if all(value in item[key] for key, value in values.items()):
            return
        updated = json.loads(json.dumps(self._control))
        target = next(entry for entry in updated["excluded"] if entry["source_id"] == item["source_id"])
        for key, value in values.items():
            if value not in target[key]:
                target[key].append(value)
        updated["revision"] += 1
        self._write_control(updated)
        self._control = updated
        self._high_water = updated["revision"]

    def _candidates(self) -> list[str]:
        candidates: set[str] = set()
        visited = 0
        for allowed in self.allowed_paths:
            initial = self.vault_path / allowed
            if allowed.lower().endswith(".md"):
                candidates.add(allowed)
                continue
            try:
                self._safe_stat(initial, directory=True)
            except _RejectedSource:
                continue
            pending = [initial]
            while pending:
                parent = pending.pop()
                try:
                    self._safe_stat(parent, directory=True)
                    with os.scandir(parent) as entries:
                        for entry in entries:
                            visited += 1
                            if visited > self.max_files * 20:
                                raise NotesUnavailable("notes directory scan exceeds limit")
                            path = Path(entry.path)
                            relative = path.relative_to(self.vault_path).as_posix()
                            if not self._allowed(relative):
                                continue
                            try:
                                if entry.is_dir(follow_symlinks=False):
                                    self._safe_stat(path, directory=True)
                                    pending.append(path)
                                elif relative.lower().endswith(".md"):
                                    candidates.add(relative)
                            except (_RejectedSource, FileNotFoundError, NotADirectoryError):
                                continue
                            if len(candidates) > self.max_files:
                                raise NotesUnavailable("notes file scan exceeds limit")
                except (_RejectedSource, FileNotFoundError, NotADirectoryError):
                    continue
                except OSError:
                    raise NotesUnavailable("notes directory cannot be read") from None
        if len(candidates) > self.max_files:
            raise NotesUnavailable("notes file scan exceeds limit")
        return sorted(candidates)

    def scan(self, force: bool = False) -> ScanResult:
        with self._lock:
            try:
                self._check_control()
                candidates = self._candidates()
                sources = []
                total_bytes = excluded = 0
                for relative in candidates:
                    try:
                        source = self._read_source(relative)
                    except _RejectedSource:
                        continue
                    total_bytes += source.size
                    if total_bytes > self.max_total_bytes:
                        raise NotesUnavailable("notes byte scan exceeds limit")
                    exclusion = self._matching_exclusion(source)
                    if exclusion:
                        self._track_exclusion(source, exclusion)
                        excluded += 1
                    else:
                        sources.append(source)
                # An excluded file encountered later may teach us a new
                # version hash shared by an earlier file. Filter the whole
                # batch against the resulting authority before storing it.
                visible_sources = []
                for source in sources:
                    exclusion = self._matching_exclusion(source)
                    if exclusion:
                        self._track_exclusion(source, exclusion)
                        excluded += 1
                    else:
                        visible_sources.append(source)
                sources = visible_sources
                self._check_control()
                before = {row["relative_path"]: row for row in self._connection.execute("SELECT * FROM note_files")}
                visible = {source.relative_path for source in sources}
                removed = sum(relative not in visible for relative in before)
                indexed = 0
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    for relative in before.keys() - visible:
                        self._connection.execute("DELETE FROM note_files WHERE relative_path=?", (relative,))
                    for source in sources:
                        previous = before.get(source.relative_path)
                        if not force and previous and previous["source_id"] == source.source_id and previous["version"] == source.version:
                            continue
                        self._connection.execute("DELETE FROM note_files WHERE relative_path=? OR source_id=?", (source.relative_path, source.source_id))
                        self._connection.execute("INSERT INTO note_files VALUES(?,?,?,?,?,?)", (source.source_id, source.relative_path, source.path_hash, source.identity_hash, source.version, source.size))
                        for hit in _chunks(source):
                            self._connection.execute("INSERT INTO note_chunks VALUES(?,?,?,?,?,?,?,?,?)", (hit.chunk_id, hit.source_id, hit.title, hit.recorded_at, json.dumps(hit.tags, ensure_ascii=False), hit.excerpt, hit.line_start, hit.line_end, hit.source_ref))
                        indexed += 1
                    self._save_meta()
                    self._connection.execute("COMMIT")
                except Exception:
                    self._connection.execute("ROLLBACK")
                    raise
                return ScanResult(len(candidates), indexed, removed, excluded, self._control["revision"])
            except NotesUnavailable:
                raise
            except Exception:
                raise NotesUnavailable("notes scan failed") from None

    def rebuild(self) -> ScanResult:
        with self._lock:
            try:
                self._check_control()
                updated = dict(self._control, revision=self._control["revision"] + 1)
                self._write_control(updated)
                self._control = updated
                self._high_water = updated["revision"]
                self._save_meta()
                return self.scan(force=True)
            except NotesUnavailable:
                raise
            except Exception:
                raise NotesUnavailable("notes rebuild failed") from None

    def search(self, query: NotesQuery, limit: int = 5) -> NotesSnapshot:
        if not isinstance(query, NotesQuery) or len(query.text) > 1000 or len(query.tags) > 20 or any(len(tag) > 60 for tag in query.tags):
            raise ValueError("invalid notes query")
        if query.date_from and query.date_to and query.date_from > query.date_to:
            raise ValueError("notes date range is reversed")
        with self._lock:
            self.scan()
            wanted = _tokens(query.text)
            phrase = _normal(query.text)
            tags = {_normal(tag).lstrip("#") for tag in query.tags}
            ranked = []
            try:
                for row in self._connection.execute("SELECT note_chunks.*,note_files.relative_path,note_files.version FROM note_chunks JOIN note_files USING(source_id)"):
                    hit_tags = tuple(json.loads(row["tags"]))
                    if tags and not tags.issubset(set(hit_tags)):
                        continue
                    recorded = date.fromisoformat(row["recorded_at"][:10]) if row["recorded_at"] else None
                    if (query.date_from or query.date_to) and (recorded is None or (query.date_from and recorded < query.date_from) or (query.date_to and recorded > query.date_to)):
                        continue
                    title, body, tag_text = _normal(row["title"]), _normal(row["excerpt"]), " ".join(hit_tags)
                    score = sum(8 * (token in title) + 3 * (token in tag_text) + 2 * (token in body) for token in wanted)
                    if wanted and score == 0:
                        continue
                    if phrase:
                        score += 20 * (phrase in title) + 8 * (phrase in body)
                    hit = NotesHit(row["source_id"], row["chunk_id"], row["relative_path"], row["title"], row["recorded_at"], hit_tags, row["excerpt"], row["line_start"], row["line_end"], row["version"], row["source_ref"])
                    ranked.append((score, row["recorded_at"], hit))
                ranked.sort(key=lambda item: (-item[0], -int(item[1][:10].replace("-", "") or "0"), item[2].relative_path, item[2].line_start))
                snapshot = NotesSnapshot(self._control["revision"], tuple(item[2] for item in ranked[:max(0, min(5, int(limit)))]))
                if not self.validate(snapshot):
                    raise NotesUnavailable("notes sources changed during retrieval")
                return snapshot
            except NotesUnavailable:
                raise
            except Exception:
                raise NotesUnavailable("notes search failed") from None

    def validate(self, snapshot: NotesSnapshot) -> bool:
        with self._lock:
            try:
                self._check_control()
                if snapshot.revision != self._control["revision"]:
                    return False
                sources: dict[str, _Source] = {}
                for hit in snapshot.hits:
                    if hit.relative_path not in sources:
                        try:
                            sources[hit.relative_path] = self._read_source(hit.relative_path)
                        except _RejectedSource:
                            return False
                    source = sources[hit.relative_path]
                    exclusion = self._matching_exclusion(source)
                    if exclusion:
                        self._track_exclusion(source, exclusion)
                        return False
                    if source.source_id != hit.source_id or source.version != hit.version:
                        return False
                self._check_control()
                return snapshot.revision == self._control["revision"]
            except NotesUnavailable:
                raise
            except Exception:
                raise NotesUnavailable("notes source verification failed") from None

    def exclude_source(self, source_id: str) -> NotesExclusionResult:
        with self._lock:
            try:
                self._check_control()
                if any(item["source_id"] == source_id for item in self._control["excluded"]):
                    return NotesExclusionResult("already_excluded", self._control["revision"], source_id)
                row = self._connection.execute("SELECT * FROM note_files WHERE source_id=?", (source_id,)).fetchone()
                if row is None:
                    return NotesExclusionResult("not_found", self._control["revision"], source_id)
                exclusion = self._matching_exclusion(row)
                if exclusion is None:
                    updated = json.loads(json.dumps(self._control))
                    updated["excluded"].append({"source_id": source_id, "paths": [row["path_hash"]], "identities": [row["identity_hash"]], "versions": [row["version"]]})
                    updated["revision"] += 1
                    # Persist authority first; an index failure cannot undo it.
                    self._write_control(updated)
                    self._control = updated
                    self._high_water = updated["revision"]
                for candidate in self._connection.execute("SELECT * FROM note_files").fetchall():
                    if self._matching_exclusion(candidate):
                        self._connection.execute("DELETE FROM note_files WHERE source_id=?", (candidate["source_id"],))
                self._save_meta()
                return NotesExclusionResult("excluded", self._control["revision"], source_id)
            except NotesUnavailable:
                raise
            except Exception:
                raise NotesUnavailable("notes exclusion failed") from None

    def current_revision(self) -> int:
        with self._lock:
            self._check_control()
            return self._control["revision"]

    @contextmanager
    def delivery_guard(self) -> Iterator[None]:
        with self._lock:
            yield

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._connection is not None:
                    self._connection.close()
