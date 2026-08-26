from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .models import IntentKind


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
    r"(?:今天|明天|后天|本周|这周|下周|下个(?:星期|礼拜)|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日|"
    r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"(?:凌晨|早上|上午|中午|下午|傍晚|晚上)?\s*\d{1,2}\s*(?:[:：点时])|"
    r"(?:半|\d{1,3})\s*(?:分钟|小时)后|"
    r"每(?:天|日|周|月)|每个(?:星期|礼拜)|连续\s*\d+\s*(?:天|周|个月)|"
    r"(?:共|总共)\s*\d+\s*次)"
)
_ACTION_SIGNAL = re.compile(
    r"(?:(?:要|得|需要|去)(?:买|购买|提交|联系|发送|准备|整理|预约|缴费|续费|回复|完成|"
    r"开会|参加|复查|看医生|吃药|接人|接孩子)|"
    r"买|购买|提交|联系|发送|准备|整理|预约|缴费|续费|取快递|打电话|回电话|"
    r"开会|参加|复查|看医生|吃药|接人|接孩子|带伞)"
)
_QUESTION_SIGNAL = re.compile(r"(?:怎么|如何|什么|哪些|为何|为什么|是否|吗|么|\?|？)")

_QUERY_SIGNAL = re.compile(
    r"(?:(?:查询|查一下|列出|看看|找一下).{0,12}(?:任务|待办|笔记)|"
    r"(?:任务|待办|笔记).{0,12}(?:有哪些|有什么|多少|在哪))"
)
_NOTE_REQUEST = re.compile(
    r"^\s*(?:请|麻烦|帮我)?\s*(?:"
    r"记一下|记下来|记录一下|记录下来|保存一下|保存这(?:段|条|个)?|"
    r"存成(?:一篇|一个)?笔记|整理成(?:一篇|一个)?笔记|做个笔记)"
)
_NOTE_STATEMENT = re.compile(
    r"^\s*(?:这是|以下是)?\s*(?:实验|会议|项目)?(?:结论|纪要|记录)\s*[：:,，]"
)
_REMINDER_REQUEST = re.compile(r"(?:(?:提醒|通知)我|到时候(?:提醒|通知|叫)我)")
_REMINDER_STOP_INTENT = re.compile(
    r"(?:(?:不要|不用|无需|不必|不需要|请勿|别)"
    r"\s*(?:再)?\s*(?:提醒|通知|叫)(?:我)?|"
    r"(?:取消|撤销|停止|关闭|删除|移除)"
    r"[^，,。；;！？!?]{0,20}?(?:提醒|通知|叫)(?:我)?)"
)
_REMINDER_META_QUESTION = re.compile(
    r"(?:为什么|为何|是否|是不是|会不会|你会|是否会|什么时候|何时|几点|怎么|如何)"
)
_POLITE_REMINDER_QUESTION = re.compile(
    r"(?:^\s*(?:(?:请|麻烦|劳驾)(?!问)|(?:你\s*)?(?:能否|可否|能不能|可不可以|能|可以))"
    r".{0,80}(?:提醒我|到时候(?:提醒|叫)我)|"
    r"(?:提醒我|到时候(?:提醒|叫)我).{0,80}(?:可以|行|好|没问题)(?:吗|么)?\s*[?？]?\s*$)"
)
_EXPLICIT_REQUEST_LEAD = re.compile(
    r"(?:^|[，,。；;！？!?])\s*(?:请(?!问)|麻烦|劳驾|帮我|记得|别忘了|不要忘记|"
    r"安排(?:一下)?|让|(?:你\s*)?(?:能否|可否|能不能|可不可以|能|可以))"
)
_REMINDER_STATUS_DESCRIPTION = re.compile(
    r"(?:(?:已经|早已|刚才|刚刚|刚).{0,40}(?:提醒|通知|叫)(?:过)?我|"
    r"(?:^|[，,。；;！？!?])\s*[^，,。；;！？!?]{0,30}?"
    r"(?:(?<!开)会(?!议|后|前|中|上)|将|负责|答应)"
    r".{0,40}(?:提醒|通知|叫)(?:过)?我)"
)
_LEADING_ACTION_REQUEST = re.compile(
    r"^\s*(?:请|麻烦|帮我)?\s*(?:记得|别忘了|安排(?!得)(?:一下)?)"
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
    r"(?:今天|明天|后天|本周|这周|下周|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日|"
    r"\d{1,2}\s*(?:[:：点时]))"
    r".{0,30}(?:买|购买|提交|联系|发送|准备|整理|预约|缴费|续费|回复|完成|打电话|"
    r"回电话|开会|参加|复查|看医生|吃药|接人|接孩子)"
)
_ACTION_THEN_TEMPORAL = re.compile(
    r"(?:买|购买|提交|联系|发送|准备|整理|预约|缴费|续费|回复|完成|打电话|回电话|"
    r"开会|参加|复查|看医生|吃药|接人|接孩子)"
    r".{0,30}(?:今天|明天|后天|本周|这周|下周|"
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
_MIXED_CONNECTOR = re.compile(r"(?:另外|同时|并且|还有|顺便)")
_CLAUSE_CONNECTOR_PREFIX = re.compile(
    r"((?:^|[，,。；;！？!?])\s*)(?:另外|同时|并且|还有|顺便)\s*"
)


def has_task_or_time_signal(text: str) -> bool:
    """Return true only for a reasonably strong task or temporal signal."""

    candidate = (text or "").strip()
    if not candidate:
        return False
    if _REMINDER_STOP_INTENT.search(candidate):
        return False
    if _DATE_OR_TIME_SIGNAL.search(candidate) or _REMINDER_REQUEST.search(candidate):
        return True
    if _QUESTION_SIGNAL.search(candidate):
        return False
    return bool(_ACTION_SIGNAL.search(candidate))


def is_non_action_task_utterance(text: str) -> bool:
    """Identify questions or subject-led statements that must not authorize writes."""

    candidate = (text or "").strip()
    if not candidate:
        return False
    if _REMINDER_STOP_INTENT.search(candidate):
        return True

    raw_reminder_request = bool(
        _REMINDER_REQUEST.search(candidate)
    )
    if raw_reminder_request:
        question_like = bool(_QUESTION_SIGNAL.search(candidate))
        polite_question = bool(_POLITE_REMINDER_QUESTION.search(candidate))
        if question_like and (
            _REMINDER_META_QUESTION.search(candidate) or not polite_question
        ):
            return True
        if (
            _REMINDER_STATUS_DESCRIPTION.search(candidate)
            and not _EXPLICIT_REQUEST_LEAD.search(candidate)
        ):
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

    if _QUERY_SIGNAL.search(candidate):
        return RouteHint(
            kind=IntentKind.QUERY,
            source=RouteSource.NATURAL,
            confidence=0.95,
            evidence=("query-request",),
            normalized_text=normalized.text,
        )

    note_request = bool(_NOTE_REQUEST.search(candidate) or _NOTE_STATEMENT.search(candidate))
    raw_reminder_request = bool(
        _REMINDER_REQUEST.search(candidate)
        and not _REMINDER_STOP_INTENT.search(candidate)
    )
    question_like = bool(_QUESTION_SIGNAL.search(candidate))
    reminder_status_description = bool(_REMINDER_STATUS_DESCRIPTION.search(candidate))
    status_description = bool(
        _STATUS_DESCRIPTION.search(candidate) or reminder_status_description
    )
    blocked_task_statement = is_non_action_task_utterance(candidate)
    reminder_request = bool(
        raw_reminder_request
        and not blocked_task_statement
    )
    leading_action_request = bool(
        _LEADING_ACTION_REQUEST.search(candidate) and has_task_or_time_signal(candidate)
    )
    explicit_task_creation = bool(_EXPLICIT_TASK_CREATION.search(candidate))
    inferred_task_request = bool(
        explicit_task_creation
        or leading_action_request
        or _TEMPORAL_ACTION.search(candidate)
        or _ACTION_THEN_TEMPORAL.search(candidate)
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

    if note_request and task_request and _MIXED_CONNECTOR.search(candidate):
        return RouteHint(
            kind=IntentKind.MIXED,
            source=RouteSource.NATURAL,
            confidence=0.9,
            evidence=("note-request", "task-request", "mixed-connector"),
            normalized_text=normalized.text,
        )

    # An explicit note-taking request owns its clause. This prevents words such
    # as “提交” inside “记录一下周五要提交的材料” from forcing a task route.
    if note_request and not reminder_request:
        return RouteHint(
            kind=IntentKind.NOTE,
            source=RouteSource.NATURAL,
            confidence=0.95,
            evidence=("note-request",),
            normalized_text=normalized.text,
        )

    if task_request and not note_request:
        evidence = "reminder-request" if reminder_request else "task-request"
        return RouteHint(
            kind=IntentKind.TASK,
            source=RouteSource.NATURAL,
            confidence=0.95,
            evidence=(evidence,),
            normalized_text=normalized.text,
        )

    return RouteHint(kind=None, normalized_text=normalized.text)
