from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .temporal import RELATIVE_TOKEN_RE, resolve_datetime, resolve_relative_time


class CompletionKind(StrEnum):
    NONE = "none"
    ACKNOWLEDGE = "acknowledge"
    RECENT = "recent"
    NAMED = "named"
    SELECT = "select"
    BATCH_REFUSED = "batch_refused"


@dataclass(frozen=True)
class CompletionDecision:
    kind: CompletionKind
    title: str = ""
    selection: int = 0


@dataclass(frozen=True)
class NamedReminderDecision:
    title: str
    reminder_at: datetime


_TRAILING_PUNCTUATION = re.compile(r"[。！!，,\s]+$")
_ACKNOWLEDGEMENTS = frozenset({"收到", "好的", "知道了", "开始做", "没做完"})
_RECENT_COMPLETIONS = frozenset({"已完成", "做完了", "搞定了"})
_BATCH_WORDS = ("全部", "全都", "都完成", "这些", "所有")


def parse_completion(text: str) -> CompletionDecision:
    """Recognize only the deliberately small, deterministic completion grammar."""

    compact = _TRAILING_PUNCTUATION.sub("", re.sub(r"\s+", " ", text or "").strip())
    if compact in _ACKNOWLEDGEMENTS:
        return CompletionDecision(CompletionKind.ACKNOWLEDGE)

    if any(word in compact for word in _BATCH_WORDS) and any(
        word in compact for word in ("完成", "做完", "搞定")
    ):
        return CompletionDecision(CompletionKind.BATCH_REFUSED)

    if compact in _RECENT_COMPLETIONS:
        return CompletionDecision(CompletionKind.RECENT)

    named = re.fullmatch(r"完成\s*[：:]\s*(.+)", compact)
    if named:
        title = named.group(1).strip()
        if title:
            return CompletionDecision(CompletionKind.NAMED, title=title[:300])

    selected = re.fullmatch(r"(?:确认)?完成\s+([1-9]\d?)", compact)
    if selected:
        return CompletionDecision(CompletionKind.SELECT, selection=int(selected.group(1)))

    return CompletionDecision(CompletionKind.NONE)


def parse_relative_reminder(text: str, now: datetime) -> datetime | None:
    """Parse only a standalone adjustment such as “半小时后提醒”."""

    compact = _TRAILING_PUNCTUATION.sub("", re.sub(r"\s+", "", text or "").strip())
    matched = re.fullmatch(r"(.+?)(?:再)?提醒(?:我)?", compact)
    if not matched:
        return None
    schedule = matched[1]
    if not RELATIVE_TOKEN_RE.fullmatch(schedule):
        return None
    return resolve_relative_time(schedule, now)


def parse_named_reminder(text: str, now: datetime) -> NamedReminderDecision | None:
    """Parse an explicit local-only binding for one existing task."""

    if now.tzinfo is None:
        return None
    compact = re.sub(r"\s+", " ", text or "").strip()
    matched = re.fullmatch(
        r"补设提醒\s*[：:]\s*(.+?)\s*[｜|]\s*(.+)", compact
    )
    if not matched:
        return None
    when_text = matched.group(1).strip()
    title = _TRAILING_PUNCTUATION.sub("", matched.group(2).strip())[:300]
    if not title:
        return None

    reminder_at = resolve_datetime(when_text, now)
    if reminder_at is None:
        return None
    return NamedReminderDecision(title=title, reminder_at=reminder_at)
