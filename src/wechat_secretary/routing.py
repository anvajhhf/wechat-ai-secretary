from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .models import IntentKind
from .request_scope import (
    INDEPENDENT_REQUEST_CONNECTOR_RE,
    REMINDER_REQUEST_RE,
    has_negated_reminder,
    mask_quoted_text,
    note_request_match,
    outer_reminder_is_question,
    outer_reminder_match,
)


class RouteSource(StrEnum):
    """Where a routing hint came from; hints never authorize execution."""

    NONE = "none"
    EXPLICIT = "explicit"
    NATURAL = "natural"


@dataclass(frozen=True)
class NormalizedText:
    original: str
    text: str
    changes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteHint:
    """A conservative classification hint, not an executable intent plan."""

    kind: IntentKind | None
    source: RouteSource = RouteSource.NONE
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    normalized_text: str = ""

    @property
    def explicit(self) -> bool:
        return self.source is RouteSource.EXPLICIT

    @property
    def natural(self) -> bool:
        return self.source is RouteSource.NATURAL


_B2M_ASR = re.compile(
    r"(?<![A-Za-z0-9])B2\s*[,，]\s*M(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_NEXT_WEEK = re.compile(r"下个(?:星期|礼拜)")
_EVERY_WEEK = re.compile(r"每个(?:星期|礼拜)")

_DATE_OR_TIME_SIGNAL = re.compile(
    r"(?:今天|今早|今晨|今晚|今夜|明天|明早|明晨|明晚|明夜|后天|本周|这周|下周|下个(?:星期|礼拜)|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日|"
    r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"(?:凌晨|早上|上午|中午|下午|傍晚|晚上)?\s*\d{1,2}\s*(?:[:：点时])|"
    r"(?:半|\d{1,3})\s*(?:分钟|小时)后|"
    r"每(?:天|日|周|月)|每个(?:星期|礼拜)|连续\s*\d+\s*(?:天|周|个月)|"
    r"(?:共|总共)\s*\d+\s*次)"
)
_ACTION_SIGNAL = re.compile(
    r"(?:(?:要|得|需要|去)(?:买|购买|办|办理|提交|联系|发送|准备|整理|预约|缴费|续费|回复|完成|"
    r"开会|参加|复查|看医生|吃药|接人|接孩子)|"
    r"买|购买|办|办理|提交|联系|发送|准备|整理|预约|缴费|续费|取快递|打电话|回电话|"
    r"开会|参加|复查|看医生|吃药|接人|接孩子|带伞)"
)
_QUESTION_SIGNAL = re.compile(
    r"(?:怎么|如何|什么|哪些|为何|为什么|是否|有没有|要不要|会不会|是不是|吗|么|\?|？)"
)

_QUERY_SIGNAL = re.compile(
    r"(?:(?:查询|查一下|列出|看看|找一下).{0,12}(?:任务|待办|笔记)|"
    r"(?:任务|待办|笔记).{0,12}(?:有哪些|有什么|多少|在哪))"
)
_QUERY_REQUEST_PREFIX = re.compile(
    r"^(?:\s|[，,：:]|请(?!问)|麻烦|劳驾|拜托|帮(?:一下)?我|你|"
    r"我(?:想|要)(?:知道)?|(?:能否|可否|能不能|可不可以|可以|能)|"
    r"今天|今日|明天|明日|后天|本周|这周|下周|未来七天|未来7天|"
    r"(?:我|今天|明天|本周|这周|下周)的|所有|全部|未完成|待完成|已完成)*$"
)
_QUERY_META_DESCRIPTION = re.compile(
    r"(?:查询|检索|搜索)(?:任务|待办|笔记)(?:的)?(?:接口|API|代码|语法|原理|教程|方法)|"
    r"(?:任务|待办|笔记)(?:查询|检索|搜索)(?:的)?(?:接口|API|代码|语法|原理|教程|方法)|"
    r"(?:如何|怎么)(?:查(?:询)?|检索|搜索|使用)|是什么(?:意思)?|是什么意思"
)
_REMINDER_STATUS_DESCRIPTION = re.compile(
    r"(?:(?:已经|早已|刚才|刚刚|刚).{0,40}(?:提醒|通知|叫)(?:过)?我|"
    r"(?:^|[，,。；;！？!?])\s*[^，,。；;！？!?]{0,30}?"
    r"(?:(?<!开)会(?!议|后|前|中|上)|将|负责|答应)"
    r".{0,40}(?:提醒|通知|叫)(?:过)?我)"
)
_LEADING_ACTION_REQUEST = re.compile(
    r"^\s*(?:请|麻烦|帮我)?\s*(?:记得|记着|记住|别忘了|安排(?!得)(?:一下)?)"
)
_EXPLICIT_TASK_CREATION = re.compile(
    r"^\s*(?:请|麻烦|帮我)?\s*(?:创建|新建)\s*(?:一个|一条|个|条)?\s*"
    r"(?:任务|待办)(?!队列|系统|模型|框架|接口|数据结构|表结构)"
)
_STATUS_DESCRIPTION = re.compile(
    r"(?:(?:已经|早已|刚刚|刚才|刚)"
    r".{0,40}(?:安排|买|购买|提交|联系|发送|准备|整理|预约|缴费|续费|回复|完成|开会)|"
    r"(?:安排|准备|整理|处理|完成)(?:得(?:很好|不错|可以)|好?了|完了))"
)
_TEMPORAL_ACTION = re.compile(
    r"(?:今天|今早|今晨|今晚|今夜|明天|明早|明晨|明晚|明夜|后天|本周|这周|下周|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日|"
    r"\d{1,2}\s*(?:[:：点时]))"
    r".{0,30}(?:买|购买|办|办理|提交|联系|发送|准备|整理|预约|缴费|续费|回复|完成|打电话|"
    r"回电话|开会|参加|复查|看医生|吃药|接人|接孩子)"
)
_ACTION_THEN_TEMPORAL = re.compile(
    r"(?:买|购买|办|办理|提交|联系|发送|准备|整理|预约|缴费|续费|回复|完成|打电话|回电话|"
    r"开会|参加|复查|看医生|吃药|接人|接孩子)"
    r".{0,30}(?:今天|今早|今晨|今晚|今夜|明天|明早|明晨|明晚|明夜|后天|本周|这周|下周|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}\s*(?:[:：点时]))"
)
_SUBJECT_TIME_ACTION_STATEMENT = re.compile(
    r"(?:^|[，,。；;！？!?])\s*"
    r"(?!(?:请(?!问)|麻烦|劳驾|帮我|记得|别忘了|不要忘记|安排(?:一下)?|让|"
    r"(?:你\s*)?(?:能否|可否|能不能|可不可以|能|可以)))"
    r"(?!(?:今天|明天|后天|本周|这周|下周|每(?:周|星期|礼拜)[一二三四五六日天]|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日|20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}))"
    r"[^，,。；;！？!?\s]{1,16}?\s*"
    r"(?:今天|明天|后天|本周|这周|下周|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日|"
    r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})"
    r".{0,40}(?:提醒|通知|叫|买|购买|提交|联系|发送|准备|整理|预约|缴费|续费|"
    r"回复|完成|打电话|回电话|开会|参加|复查|看医生|吃药|接人|接孩子)"
)
_CLAUSE_CONNECTOR_PREFIX = re.compile(
    r"((?:^|[，,。；;！？!?])\s*)(?:另外|同时|并且|还有|顺便)\s*"
)


def _unquoted_text(text: str) -> str:
    """Mask quoted content without changing offsets into the original text."""

    return mask_quoted_text(text)


def _outer_reminder_match(text: str) -> re.Match[str] | None:
    return outer_reminder_match(text)


def _outer_reminder_is_question(text: str, match: re.Match[str]) -> bool:
    return outer_reminder_is_question(text, match)


def _outer_query_match(text: str) -> re.Match[str] | None:
    """A read request, not a quoted/reported mention of looking up tasks."""

    candidate = _unquoted_text(text)
    match = _QUERY_SIGNAL.search(candidate)
    if (
        match is None
        or candidate[:match.start()] != text[:match.start()]
        or not _QUERY_REQUEST_PREFIX.fullmatch(candidate[:match.start()])
        or _QUERY_META_DESCRIPTION.search(candidate)
    ):
        return None
    return match


def has_task_or_time_signal(text: str) -> bool:
    """Return true only for a reasonably strong task or temporal signal."""

    candidate = _unquoted_text(text or "").strip()
    if not candidate:
        return False
    if has_negated_reminder(text):
        return False
    if _DATE_OR_TIME_SIGNAL.search(candidate) or REMINDER_REQUEST_RE.search(candidate):
        return True
    if _QUESTION_SIGNAL.search(candidate):
        return False
    return bool(_ACTION_SIGNAL.search(candidate))


def is_non_action_task_utterance(text: str) -> bool:
    """Identify questions or subject-led statements that must not authorize writes."""

    candidate = _unquoted_text(text or "").strip()
    if not candidate:
        return False
    if has_negated_reminder(text):
        return True

    outer_reminder = _outer_reminder_match(text)
    if outer_reminder is not None:
        # Once an explicit outer reminder command is identified, questions and
        # reported statements inside its payload describe the future task.
        return _outer_reminder_is_question(text, outer_reminder)

    raw_reminder_request = bool(REMINDER_REQUEST_RE.search(candidate))
    if raw_reminder_request:
        # Without a recognized outer command, a mention of a reminder could
        # be forwarded text, a report, or a question. It is not write consent.
        return True

    # Discourse connectors do not become grammatical subjects.  Strip only a
    # connector at a clause boundary, then rerun the subject test: “另外明天提交”
    # stays an elliptical command, while “另外导师明天提交” remains a statement.
    subject_candidate = _CLAUSE_CONNECTOR_PREFIX.sub(r"\1", candidate)
    return bool(_SUBJECT_TIME_ACTION_STATEMENT.search(subject_candidate))


def normalize_routing_text(text: str, *, speech: bool = False) -> NormalizedText:
    """Apply a deliberately small set of meaning-preserving normalizations."""

    original = text or ""
    normalized = original
    changes: list[str] = []

    updated = _NEXT_WEEK.sub("下周", normalized)
    if updated != normalized:
        normalized = updated
        changes.append("next-week")

    updated = _EVERY_WEEK.sub("每周", normalized)
    if updated != normalized:
        normalized = updated
        changes.append("weekly")

    updated = _B2M_ASR.sub("B2M", normalized)
    if updated != normalized:
        normalized = updated
        changes.append("term-b2m")

    if speech:
        spoken = re.match(r"^(\s*)代办(.*)$", normalized, re.DOTALL)
        if spoken is not None:
            remainder = spoken.group(2)
            task_content = re.sub(r"^[\s：:,，]+", "", remainder)
            if task_content and has_task_or_time_signal(task_content):
                normalized = f"{spoken.group(1)}待办：{task_content}"
                changes.append("spoken-task-prefix")

    return NormalizedText(original=original, text=normalized, changes=tuple(changes))


def detect_route_hint(
    text: str,
    *,
    explicit_kind: IntentKind | None = None,
    speech: bool = False,
) -> RouteHint:
    """Infer a routing hint without creating, saving, querying, or completing anything."""

    normalized = normalize_routing_text(text, speech=speech)
    candidate = normalized.text.strip()
    if explicit_kind is not None:
        return RouteHint(
            kind=explicit_kind,
            source=RouteSource.EXPLICIT,
            confidence=1.0,
            evidence=("explicit-prefix",),
            normalized_text=normalized.text,
        )
    if not candidate:
        return RouteHint(kind=None, normalized_text=normalized.text)

    unquoted_candidate = _unquoted_text(candidate)
    note_match = note_request_match(candidate)
    if note_match is not None:
        # The note owns its body, including lookup/reminder/cancellation words.
        # Only a separate, explicitly connected task clause creates MIXED.
        body = candidate[note_match.end():]
        for connector in INDEPENDENT_REQUEST_CONNECTOR_RE.finditer(_unquoted_text(body)):
            tail = body[connector.end():]
            if detect_route_hint(tail).kind is IntentKind.TASK:
                return RouteHint(
                    kind=IntentKind.MIXED,
                    source=RouteSource.NATURAL,
                    confidence=0.9,
                    evidence=("note-request", "task-request", "mixed-connector"),
                    normalized_text=normalized.text,
                )
        return RouteHint(
            kind=IntentKind.NOTE,
            source=RouteSource.NATURAL,
            confidence=0.95,
            evidence=("note-request",),
            normalized_text=normalized.text,
        )

    query_match = _outer_query_match(candidate)
    reminder_match = REMINDER_REQUEST_RE.search(unquoted_candidate)
    outer_reminder = _outer_reminder_match(candidate)
    scoped_reminder_request = bool(
        outer_reminder is not None
        and not _outer_reminder_is_question(candidate, outer_reminder)
        and not has_negated_reminder(candidate)
    )
    # A lookup after a reminder marker belongs to its payload, even when the
    # outer utterance is a rejected question/negation about that reminder.
    if query_match is not None and not scoped_reminder_request and (
        reminder_match is None or query_match.start() < reminder_match.start()
    ):
        return RouteHint(
            kind=IntentKind.QUERY,
            source=RouteSource.NATURAL,
            confidence=0.95,
            evidence=("query-request",),
            normalized_text=normalized.text,
        )

    raw_reminder_request = bool(
        outer_reminder
        and not has_negated_reminder(candidate)
    )
    question_like = bool(_QUESTION_SIGNAL.search(unquoted_candidate))
    reminder_status_description = bool(_REMINDER_STATUS_DESCRIPTION.search(unquoted_candidate))
    status_description = bool(
        _STATUS_DESCRIPTION.search(unquoted_candidate) or reminder_status_description
    )
    blocked_task_statement = is_non_action_task_utterance(candidate)
    reminder_request = bool(
        raw_reminder_request
        and not blocked_task_statement
    )
    leading_action_request = bool(
        _LEADING_ACTION_REQUEST.search(unquoted_candidate) and has_task_or_time_signal(candidate)
    )
    explicit_task_creation = bool(_EXPLICIT_TASK_CREATION.search(unquoted_candidate))
    inferred_task_request = bool(
        explicit_task_creation
        or leading_action_request
        or _TEMPORAL_ACTION.search(unquoted_candidate)
        or _ACTION_THEN_TEMPORAL.search(unquoted_candidate)
    )
    task_request = bool(
        reminder_request
        or (
            inferred_task_request
            and not question_like
            and not status_description
            and not blocked_task_statement
        )
    )

    if task_request:
        evidence = "reminder-request" if reminder_request else "task-request"
        return RouteHint(
            kind=IntentKind.TASK,
            source=RouteSource.NATURAL,
            confidence=0.95,
            evidence=(evidence,),
            normalized_text=normalized.text,
        )

    return RouteHint(kind=None, normalized_text=normalized.text)
