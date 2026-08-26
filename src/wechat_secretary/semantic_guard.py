from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .classifier import _resolve_date, _resolve_reminder, _resolve_time
from .models import (
    ClarificationReason,
    IntentKind,
    IntentPlan,
    NoteDraft,
    PendingTaskClarification,
    ReminderRecurrence,
    TaskDraft,
)


_WEEKDAY_ISO = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7}
_WEEKDAY_TEXT = {value: key for key, value in _WEEKDAY_ISO.items() if key != "天"}
_REMINDER_RE = re.compile(
    r"(?:到时候\s*)?(?:提醒|通知|叫)\s*我|别忘(?:了)?"
)
_NEGATED_REMINDER_RE = re.compile(
    r"(?:(?:不要|不用|无需|不必|不需要|别|请勿)\s*(?:再)?\s*|"
    r"(?:取消|撤销|停止|关闭|删除|移除)\s*(?:再)?[^，,。；;！？!?\n]{0,30}?)"
    r"(?:提醒|通知|叫)(?:我)?"
)
_UNSUPPORTED_RECURRENCE_RE = re.compile(
    r"每天|每日|每月|每个月|每季度|每年|隔周|每两周|无限|一直提醒|长期提醒"
)
_WEEKLY_RE = re.compile(r"每(?:周|星期|礼拜)\s*([一二三四五六日天])")
_BARE_WEEKLY_RE = re.compile(
    r"每(?:周|星期|礼拜)(?=\s*(?:(?:都|固定)\s*)?"
    r"(?:凌晨|早上|上午|中午|下午|傍晚|晚上|\d|提醒|通知|叫|[，,]))"
)
_DUE_RE = re.compile(r"截止|到期|之前|最晚|日程|安排在")
_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9._+-]{2,}\b)[A-Za-z][A-Za-z0-9._+-]*\d[A-Za-z0-9._+-]*", re.I)
_TITLE_ACTION_RE = re.compile(
    r"((?:购买|买|提交|联系|发送|准备|预约|缴费|续费|取快递|打电话|回电话|回复|完成)"
    r"[^，,。；;！？!?]{1,100})"
)
_GROUNDING_PUNCT_RE = re.compile(r"[\s，,。；;！？!?：:、（）()\[\]【】\"“”'‘’]+")
_STRONG_MULTI_TASK_SEPARATOR_RE = re.compile(
    r"(?:[；;\n]|(?:另外|还有(?:一项|一个)?|第二(?:项|个任务)|下一项))"
)
_TASK_REQUEST_PREFIX_RE = re.compile(
    r"^\s*(?:(?:请|麻烦|劳驾|帮我|你能|能否|可否|能不能|可不可以|可以)\s*)?"
    r"(?:(?:创建|新建)\s*(?:一个|一条|个|条)?\s*(?:任务|待办)\s*|"
    r"(?:记得|别忘了|安排(?:一下)?)\s*)"
)
_TASK_DATE_CONTROL_RE = re.compile(
    r"(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}\s*月\s*\d{1,2}\s*日|"
    r"今天|今日|明天|明日|后天|"
    r"(?:(?:本周|这周|下周|每周|每星期|每礼拜|周|星期|礼拜)\s*[一二三四五六日天])|"
    r"每(?:周|星期|礼拜)(?:\s*(?:都|固定))?)"
)
_TASK_TIME_CONTROL_RE = re.compile(
    r"(?:(?:凌晨|早上|上午|中午|下午|傍晚|晚上)\s*)?"
    r"(?:[01]?\d|2[0-3])\s*(?:[:：点时])\s*(?:[0-5]?\d)?\s*(?:分)?"
)
_TASK_REPEAT_CONTROL_RE = re.compile(
    r"(?:连续\s*[0-9一二两三四五六七八九十]{1,3}\s*(?:周|次)|"
    r"(?:共|总共|一共)\s*[0-9一二两三四五六七八九十]{1,3}\s*次|"
    r"(?:半|\d{1,3})\s*(?:分钟|小时)后)"
)
_NOTE_REQUEST_PREFIX_RE = re.compile(
    r"^\s*(?:请|麻烦|帮我)?\s*(?:"
    r"记一下|记下来|记录一下|记录下来|保存一下|保存这(?:段|条|个)?|"
    r"存成(?:一篇|一个)?笔记|整理成(?:一篇|一个)?笔记|做个笔记)"
    r"\s*[：:,，]?\s*"
)
_PAIRED_QUOTES = (("《", "》"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))
_SAME_QUOTES = ('"', "'")


@dataclass(frozen=True)
class TaskSemanticSignals:
    requests_reminder: bool = False
    negated_reminder: bool = False
    requested_date: str = ""
    requested_time: str = ""
    relative_reminder_at: str = ""
    recurrence_requested: bool = False
    recurrence_frequency: str = ""
    recurrence_weekday: int = 0
    repeat_count: int | None = None
    recurrence_start_explicit: bool = False
    explicit_due: bool = False
    identifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardDecision:
    plan: IntentPlan
    reason: ClarificationReason = ClarificationReason.NONE
    question: str = ""
    pending: PendingTaskClarification | None = None

    @property
    def ready(self) -> bool:
        return self.reason is ClarificationReason.NONE and not self.question


def _quoted_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for opening, closing in _PAIRED_QUOTES:
        cursor = 0
        while cursor < len(text):
            start = text.find(opening, cursor)
            if start < 0:
                break
            end = text.find(closing, start + len(opening))
            if end < 0:
                ranges.append((start, len(text)))
                break
            ranges.append((start, end + len(closing)))
            cursor = end + len(closing)
    for quote in _SAME_QUOTES:
        cursor = 0
        while cursor < len(text):
            start = text.find(quote, cursor)
            if start < 0:
                break
            end = text.find(quote, start + 1)
            if end < 0:
                ranges.append((start, len(text)))
                break
            ranges.append((start, end + 1))
            cursor = end + 1
    return tuple(ranges)


def _first_unquoted_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    quoted = _quoted_ranges(text)
    for match in pattern.finditer(text):
        if not any(start <= match.start() and match.end() <= end for start, end in quoted):
            return match
    return None


def _reminder_marker(text: str) -> re.Match[str] | None:
    return _first_unquoted_match(_REMINDER_RE, text)


def _chinese_number(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if raw in digits:
        return digits[raw]
    if raw == "十":
        return 10
    if "十" in raw and len(raw) <= 3:
        left, right = raw.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _repeat_count(text: str) -> int | None:
    number = r"([0-9]{1,3}|[一二两三四五六七八九十]{1,3})"
    patterns = (
        rf"(?:共|总共|一共)\s*{number}\s*次",
        rf"连续(?:提醒我?)?\s*{number}\s*(?:周|次)",
        rf"提醒我?\s*{number}\s*次",
        rf"{number}\s*次\s*[，,、 ]*(?:共|总共)",
    )
    for index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text):
            value = _chinese_number(match.group(1))
            # “每周二提醒我一次”描述的是每次触发，不等于系列总次数。
            if index == 2 and value == 1:
                continue
            return value
    week_count = re.search(
        rf"连续\s*{number}\s*周", text
    )
    return _chinese_number(week_count.group(1)) if week_count else None


def extract_task_semantics(text: str, now: datetime) -> TaskSemanticSignals:
    requested_date = _resolve_date(text, now)
    requested_time = _resolve_time(text)
    negated_reminder = bool(_first_unquoted_match(_NEGATED_REMINDER_RE, text))
    requests_reminder = bool(_reminder_marker(text)) and not negated_reminder
    weekly = _WEEKLY_RE.search(text)
    bare_weekly = bool(_BARE_WEEKLY_RE.search(text))
    unsupported = bool(_UNSUPPORTED_RECURRENCE_RE.search(text))
    recurrence_requested = bool(
        weekly or bare_weekly or unsupported or re.search(r"重复提醒|循环提醒", text)
    )
    recurrence_frequency = (
        "unsupported" if unsupported else "weekly" if weekly or bare_weekly else ""
    )
    weekday = _WEEKDAY_ISO[weekly.group(1)] if weekly else 0
    recurrence_start_explicit = bool(
        re.search(
            r"(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}月\d{1,2}日|"
            r"今天|明天|后天|(?:本周|这周|下周)[一二三四五六日天])",
            text,
        )
    )
    relative = _resolve_reminder(text, now, requested_date, requested_time)
    if requests_reminder and not relative:
        relative_match = re.search(r"(半|\d{1,3})\s*(分钟|小时)后", text)
        if relative_match:
            amount = 0.5 if relative_match.group(1) == "半" else int(relative_match.group(1))
            delta = (
                timedelta(hours=amount)
                if relative_match.group(2) == "小时"
                else timedelta(minutes=amount)
            )
            relative = (now + delta).isoformat(timespec="minutes")
    identifiers = tuple(dict.fromkeys(match.group(0) for match in _IDENTIFIER_RE.finditer(text)))
    return TaskSemanticSignals(
        requests_reminder=requests_reminder,
        negated_reminder=negated_reminder,
        requested_date=requested_date,
        requested_time=requested_time,
        relative_reminder_at=relative,
        recurrence_requested=recurrence_requested,
        recurrence_frequency=recurrence_frequency,
        recurrence_weekday=weekday,
        repeat_count=_repeat_count(text),
        recurrence_start_explicit=recurrence_start_explicit,
        explicit_due=bool(_DUE_RE.search(text)),
        identifiers=identifiers,
    )


def _clarify(
    plan: IntentPlan,
    reason: ClarificationReason,
    question: str,
    task: TaskDraft | None = None,
    *,
    reminder_date: str = "",
    reminder_time: str = "",
) -> GuardDecision:
    pending = None
    if task is not None:
        pending = PendingTaskClarification(
            reason=reason,
            task=task,
            reminder_date=reminder_date,
            reminder_time=reminder_time,
        )
    return GuardDecision(
        plan=replace(plan, kind=IntentKind.CLARIFY, clarification_reason=reason),
        reason=reason,
        question=question,
        pending=pending,
    )


def _first_weekday(now: datetime, weekday: int, clock: str) -> datetime:
    hour, minute = (int(part) for part in clock.split(":"))
    delta = (weekday - now.isoweekday()) % 7
    candidate = (now + timedelta(days=delta)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def _deterministic_task_title(text: str) -> str:
    """Extract reminder payload without deleting date-like words inside it."""

    source = re.sub(
        r"^\s*(?:(?:请|麻烦|劳驾|帮我|你能|能否|可否|能不能|可不可以|可以)\s*(?:在)?\s*)",
        "",
        text.strip(),
        count=1,
    )
    source = _TASK_REQUEST_PREFIX_RE.sub("", source, count=1)
    marker = _reminder_marker(source)
    prefix = source[: marker.start()] if marker else source
    suffix = source[marker.end() :] if marker else ""

    def strip_leading_schedule(value: str) -> str:
        candidate = value.strip(" ，,。；;！？!?：:、")
        while candidate:
            updated = re.sub(r"^(?:在|从|到时候)\s*", "", candidate, count=1)
            matched = False
            for pattern in (
                _TASK_DATE_CONTROL_RE,
                _TASK_TIME_CONTROL_RE,
                _TASK_REPEAT_CONTROL_RE,
            ):
                found = pattern.match(updated)
                if found:
                    updated = updated[found.end() :]
                    matched = True
                    break
            updated = re.sub(r"^(?:开始)?\s*[，,、；;]?\s*", "", updated)
            if not matched and updated == candidate:
                break
            candidate = updated
        return candidate

    candidate = strip_leading_schedule(suffix) if marker else ""
    candidate = re.sub(
        r"[，,、；;]?\s*(?:(?:共|总共|一共)\s*"
        r"[0-9一二两三四五六七八九十]{1,3}\s*次|"
        r"连续(?:提醒(?:我)?)?\s*[0-9一二两三四五六七八九十]{1,3}\s*(?:周|次))\s*$",
        "",
        candidate,
    )
    candidate = re.sub(
        r"(?:可以吗|好吗|行吗|没问题吗|可以|好吧|行吧|吗|么)\s*[?？]?$",
        "",
        candidate,
    )
    if marker and len(_compact_grounding_text(candidate)) < 2:
        candidate = prefix.strip(" ，,。；;！？!?：:、")
        while candidate:
            updated = candidate
            for pattern in (
                _TASK_TIME_CONTROL_RE,
                _TASK_DATE_CONTROL_RE,
                _TASK_REPEAT_CONTROL_RE,
            ):
                found = re.search(rf"(?:{pattern.pattern})\s*$", updated)
                if found:
                    updated = updated[: found.start()].rstrip(" ，,、；;")
                    break
            if updated == candidate:
                break
            candidate = updated
    candidate = re.sub(r"[\s，,。；;！？!?：:、]+", " ", candidate)
    return candidate.strip(" 吧呢啊呀")[:300]


def _deterministic_plain_task_title(text: str) -> str:
    candidate = re.sub(
        r"^\s*(?:(?:请|麻烦|劳驾|帮我)\s*)",
        "",
        text.strip(),
        count=1,
    )
    candidate = _TASK_REQUEST_PREFIX_RE.sub("", candidate, count=1)
    while candidate:
        updated = re.sub(r"^(?:在|从|到时候)\s*", "", candidate, count=1)
        matched = False
        for pattern in (
            _TASK_DATE_CONTROL_RE,
            _TASK_TIME_CONTROL_RE,
            _TASK_REPEAT_CONTROL_RE,
        ):
            found = pattern.match(updated)
            if found:
                updated = updated[found.end() :]
                matched = True
                break
        updated = re.sub(r"^(?:开始)?\s*[，,、；;]?\s*", "", updated)
        if not matched and updated == candidate:
            break
        candidate = updated
    candidate = re.sub(r"[\s，,。；;！？!?：:、]+", " ", candidate)
    return candidate.strip(" 吧呢啊呀")[:300]


def _source_priority(text: str) -> str:
    if re.search(r"高优先级|非常重要|紧急", text):
        return "high"
    if re.search(r"中优先级|(?<!非常)重要", text):
        return "medium"
    if re.search(r"低优先级|不急", text):
        return "low"
    return "none"


def _sanitize_task_metadata(text: str, task: TaskDraft) -> TaskDraft:
    source = _compact_grounding_text(text)
    category = (
        task.category
        if task.category and _compact_grounding_text(task.category) in source
        else ""
    )
    return replace(
        task,
        priority=_source_priority(text),
        category=category,
        tags=(),
        description="",
    )


def _note_source_body(text: str) -> str:
    body = _NOTE_REQUEST_PREFIX_RE.sub("", text.strip(), count=1)
    body = re.split(r"(?:，|,|；|;)\s*(?:另外|同时|并且|还有|顺便)", body, maxsplit=1)[0]
    return body.strip().lstrip("，,：:")


def _grounded_note_title(candidate: str, body: str) -> str:
    compact_candidate = _compact_grounding_text(candidate)
    compact_body = _compact_grounding_text(body)
    compact_core = re.sub(r"(?:记录|笔记|纪要)$", "", compact_candidate)
    if compact_candidate and (
        compact_candidate in compact_body
        or (len(compact_core) >= 2 and compact_core in compact_body)
    ):
        return candidate.strip()[:200]
    first_clause = re.split(r"[。！？；.!?;\n]", body, maxsplit=1)[0].strip()
    return (first_clause[:80] or "微信笔记")


def _ground_plain_note(text: str, plan: IntentPlan) -> GuardDecision:
    body = _note_source_body(text)
    if not body:
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "我听懂你想记录内容，但原话里还没有可保存的正文。请再说具体一点。",
        )
    template = plan.notes[0] if plan.notes else NoteDraft("", "")
    title = _grounded_note_title(template.title, body)
    links = tuple(link for link in template.links if link and link in text)
    note = NoteDraft(
        title=title,
        body=body,
        summary=re.sub(r"\s+", " ", body).strip()[:200],
        tags=(),
        links=links,
        target_hint=(
            template.target_hint
            if template.target_hint and template.target_hint in text
            else ""
        ),
    )
    return GuardDecision(
        replace(
            plan,
            kind=IntentKind.NOTE,
            tasks=(),
            notes=(note,),
            query=None,
            confidence=max(plan.confidence, 0.85),
            clarification="",
            clarification_reason=ClarificationReason.NONE,
        )
    )


def _with_tasks(plan: IntentPlan, tasks: tuple[TaskDraft, ...]) -> IntentPlan:
    return replace(
        plan,
        kind=IntentKind.TASK if not plan.notes else IntentKind.MIXED,
        tasks=tasks,
        notes=plan.notes,
        query=None,
        confidence=max(plan.confidence, 0.85),
        clarification="",
        clarification_reason=ClarificationReason.NONE,
    )


def _with_task(plan: IntentPlan, task: TaskDraft) -> IntentPlan:
    return _with_tasks(plan, (task, *plan.tasks[1:]))


def _compact_grounding_text(value: str) -> str:
    return _GROUNDING_PUNCT_RE.sub("", value).casefold()


def _validate_multiple_tasks(
    text: str,
    plan: IntentPlan,
    signals: TaskSemanticSignals,
) -> GuardDecision:
    """Allow only directly grounded, timeless batches; otherwise fail closed.

    A single date/reminder expression cannot be assigned safely across multiple
    model-produced tasks.  Timeless batches are accepted only when the user used
    a strong item boundary and every distinct title occurs literally in the
    source.  Model-invented scheduling fields are then removed from every item.
    """

    if plan.notes:
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "这条同时被拆成了多个任务和笔记，我无法逐项确认你的意图。请分开发送。",
        )
    if (
        signals.requests_reminder
        or signals.recurrence_requested
        or signals.requested_date
        or signals.requested_time
        or signals.relative_reminder_at
        or signals.explicit_due
    ):
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "这句话被拆成了多个任务，但时间或提醒不能可靠地对应到每一项。为避免批量建错，请每项单独发送。",
        )
    if not _STRONG_MULTI_TASK_SEPARATOR_RE.search(text):
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "我不能确认这句话是一个整体任务，还是多个任务。请用分号明确分项，或每项单独发送。",
        )

    source = _compact_grounding_text(text)
    titles = tuple(_compact_grounding_text(task.title) for task in plan.tasks)
    if (
        any(not title or title not in source for title in titles)
        or len(set(titles)) != len(titles)
    ):
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "我不能从原话逐项确认拆出的多个任务。为避免批量建错，请把每项任务分开发送。",
        )

    spans = sorted((source.index(title), source.index(title) + len(title)) for title in titles)
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "模型拆出的多个任务在原话中相互重叠，我不能安全批量创建。请每项单独发送。",
        )

    searchable = " ".join(task.title for task in plan.tasks).casefold()
    missing_identifiers = [
        item for item in signals.identifiers if item.casefold() not in searchable
    ]
    if missing_identifiers:
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            f"我没有可靠保留关键标识“{'、'.join(missing_identifiers)}”，为避免批量建错，请确认后重发。",
        )

    sanitized = tuple(
        replace(
            _sanitize_task_metadata(text, task),
            due_date="",
            due_time="",
            reminder_at="",
            reminder_recurrence=None,
        )
        for task in plan.tasks
    )
    return GuardDecision(_with_tasks(plan, sanitized))


def validate_plan_semantics(
    text: str,
    plan: IntentPlan,
    now: datetime,
    *,
    expected_kind: IntentKind | None = None,
    allow_enriched_note: bool = False,
    allow_explicit_task_fallback: bool = False,
) -> GuardDecision:
    signals = extract_task_semantics(text, now)

    if signals.negated_reminder:
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "我理解你想停止提醒，但不会把这句话当成新任务。请发送“完成：任务名”来结束对应任务和后续提醒。",
        )

    # The model may extract structure, but it does not authorize a write.  A
    # task/note/mixed write must first have been positively routed by the local
    # prefix or natural-language detector.  This keeps questions and ordinary
    # status statements read-only even if the model misclassifies them.
    has_write_plan = bool(
        plan.tasks
        or plan.notes
        or plan.kind in {IntentKind.TASK, IntentKind.NOTE, IntentKind.MIXED}
    )
    if expected_kind is None and has_write_plan:
        return _clarify(
            plan,
            ClarificationReason.AMBIGUOUS_INTENT,
            "我没有识别到明确的创建或记录请求，因此没有写入。若要执行，请直接说“明天下午3点提醒我回电话”或“帮我记一下……”。",
        )

    if expected_kind is IntentKind.QUERY:
        if (
            plan.kind is IntentKind.QUERY
            and plan.query is not None
            and not plan.tasks
            and not plan.notes
        ):
            return GuardDecision(
                replace(plan, kind=IntentKind.QUERY, tasks=(), notes=())
            )
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "我识别到你想查询，但这次返回的不是可靠的纯查询结构。为避免误写入，我没有执行，请重试。",
        )

    if expected_kind is IntentKind.NOTE:
        if allow_enriched_note and plan.notes:
            return GuardDecision(
                replace(plan, kind=IntentKind.NOTE, tasks=(), query=None)
            )
        return _ground_plain_note(text, plan)

    if expected_kind is IntentKind.MIXED and (not plan.tasks or not plan.notes):
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "我识别到你同时想建任务和记笔记，但这次没有可靠提取完整。请把两项分开发给我。",
        )
    if expected_kind is IntentKind.MIXED:
        grounded_note = _ground_plain_note(text, plan)
        if not grounded_note.ready:
            return grounded_note
        plan = replace(plan, notes=grounded_note.plan.notes, query=None)

    task_expected = (
        expected_kind in {IntentKind.TASK, IntentKind.MIXED}
        or signals.requests_reminder
        or signals.recurrence_requested
    )
    if not task_expected and plan.kind not in {IntentKind.TASK, IntentKind.MIXED}:
        return GuardDecision(plan)
    if not plan.tasks:
        question = (
            "我听懂你想设置提醒，但还没识别出具体要做什么。请补充提醒事项。"
            if signals.requests_reminder
            else "我还没识别出具体任务内容，请再补充并说清楚要做什么。"
        )
        return _clarify(plan, ClarificationReason.MISSING_TASK_BODY, question)
    if signals.recurrence_requested and len(plan.tasks) != 1:
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            "重复提醒一次只能绑定一个任务，请把这些事项分开发给我。",
        )
    if len(plan.tasks) > 1:
        return _validate_multiple_tasks(text, plan, signals)

    task = plan.tasks[0]
    if signals.requests_reminder or signals.recurrence_requested:
        title = _deterministic_task_title(text)
        if len(_compact_grounding_text(title)) < 2:
            return _clarify(
                plan,
                ClarificationReason.MISSING_TASK_BODY,
                "我听懂你想设置提醒，但原话里还没有可靠的提醒事项。请补充具体要做什么。",
            )
        task = replace(task, title=title)
    elif (
        not _compact_grounding_text(task.title)
        or _compact_grounding_text(task.title) not in _compact_grounding_text(text)
    ):
        fallback_title = (
            _deterministic_plain_task_title(text)
            if allow_explicit_task_fallback
            else ""
        )
        if len(_compact_grounding_text(fallback_title)) < 2:
            return _clarify(
                plan,
                ClarificationReason.SEMANTIC_MISMATCH,
                "我无法从原话确认提取出的任务标题。为避免建错任务，请重新说明要做什么。",
            )
        task = replace(task, title=fallback_title)
    task = _sanitize_task_metadata(text, task)
    searchable = task.title.casefold()
    missing_identifiers = [item for item in signals.identifiers if item.casefold() not in searchable]
    if missing_identifiers:
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH,
            f"我没有可靠保留关键标识“{'、'.join(missing_identifiers)}”，为避免建错任务，请确认后重发。",
        )

    if signals.recurrence_requested:
        if not signals.requests_reminder:
            return _clarify(
                plan,
                ClarificationReason.UNSUPPORTED_RECURRENCE,
                "我识别到了重复任务，但当前只支持明确说出的有限微信提醒。若你需要提醒，请说“每周二上午9点提醒我……，共3次”。",
            )
        if signals.recurrence_frequency != "weekly":
            return _clarify(
                plan,
                ClarificationReason.UNSUPPORTED_RECURRENCE,
                "目前只支持有明确结束次数的每周提醒。请改成“每周二上午9点，共3次”这类说法。",
            )
        recurrence = ReminderRecurrence(
            frequency="weekly",
            interval=1,
            weekday=signals.recurrence_weekday,
            count=signals.repeat_count or 0,
        )
        pending_task = replace(task, reminder_recurrence=recurrence, reminder_at="")
        recurrence_start_date = (
            signals.requested_date if signals.recurrence_start_explicit else ""
        )
        if not signals.recurrence_weekday:
            return _clarify(
                plan,
                ClarificationReason.MISSING_RECURRENCE_DETAILS,
                "我已识别到重复提醒，还差星期几。请回复例如“每周二”。",
                pending_task,
                reminder_date=recurrence_start_date,
                reminder_time=signals.requested_time,
            )
        if signals.repeat_count is None:
            return _clarify(
                plan,
                ClarificationReason.MISSING_RECURRENCE_COUNT,
                "我已识别到每周提醒，还差总次数（2—52次），例如“共3次”。",
                pending_task,
                reminder_date=recurrence_start_date,
                reminder_time=signals.requested_time,
            )
        if not 2 <= signals.repeat_count <= 52:
            return _clarify(
                plan,
                ClarificationReason.UNSUPPORTED_RECURRENCE,
                "每周提醒的总次数需要在2到52次之间，请换一个次数。",
                pending_task,
                reminder_date=recurrence_start_date,
                reminder_time=signals.requested_time,
            )
        if not signals.requested_time:
            return _clarify(
                plan,
                ClarificationReason.MISSING_REMINDER_TIME,
                "我已识别到每周提醒和总次数，还差具体几点，例如“上午9点”。",
                pending_task,
                reminder_date=recurrence_start_date,
            )
        first = _first_weekday(now, signals.recurrence_weekday, signals.requested_time)
        if recurrence_start_date:
            requested_first = datetime.fromisoformat(
                f"{recurrence_start_date}T{signals.requested_time}:00"
            ).replace(tzinfo=now.tzinfo)
            if requested_first.isoweekday() != signals.recurrence_weekday:
                return _clarify(
                    plan,
                    ClarificationReason.SEMANTIC_MISMATCH,
                    "开始日期与“每周几”不一致，请确认后重新发送。",
                )
            if requested_first <= now:
                return _clarify(
                    plan,
                    ClarificationReason.SEMANTIC_MISMATCH,
                    "重复提醒的开始时间已经过去，请给我一个新的开始日期和时间。",
                )
            first = requested_first
        task = replace(
            task,
            due_date=task.due_date if signals.explicit_due else "",
            due_time=task.due_time if signals.explicit_due else "",
            reminder_at=first.isoformat(timespec="minutes"),
            reminder_recurrence=recurrence,
        )
        return GuardDecision(_with_task(plan, task))

    if signals.requests_reminder:
        reminder_at = signals.relative_reminder_at
        if not reminder_at:
            if not signals.requested_date and not signals.requested_time:
                return _clarify(
                    plan,
                    ClarificationReason.MISSING_REMINDER_DATE_TIME,
                    "我已识别到提醒事项，还差日期和具体时间。",
                    task,
                )
            if not signals.requested_date:
                return _clarify(
                    plan,
                    ClarificationReason.MISSING_REMINDER_DATE,
                    "我已识别到提醒时间，还差哪一天。",
                    task,
                    reminder_time=signals.requested_time,
                )
            if not signals.requested_time:
                return _clarify(
                    plan,
                    ClarificationReason.MISSING_REMINDER_TIME,
                    "我已识别到提醒日期，还差具体几点，例如“上午9点”。",
                    task,
                    reminder_date=signals.requested_date,
                )
            reminder_at = datetime.fromisoformat(
                f"{signals.requested_date}T{signals.requested_time}:00"
            ).replace(tzinfo=now.tzinfo).isoformat(timespec="minutes")
        parsed = datetime.fromisoformat(reminder_at).astimezone(now.tzinfo)
        if parsed <= now:
            return _clarify(
                plan,
                ClarificationReason.SEMANTIC_MISMATCH,
                "这个提醒时间已经过去了，请给我一个新的日期和时间。",
                task,
            )
        task = replace(
            task,
            due_date=task.due_date if signals.explicit_due else "",
            due_time=task.due_time if signals.explicit_due else "",
            reminder_at=parsed.isoformat(timespec="minutes"),
            reminder_recurrence=None,
        )
        return GuardDecision(_with_task(plan, task))

    if signals.requested_date:
        task = replace(
            task,
            due_date=signals.requested_date,
            due_time=signals.requested_time,
            reminder_at="",
            reminder_recurrence=None,
        )
        return GuardDecision(_with_task(plan, task))
    task = replace(
        task,
        due_date="",
        due_time="",
        reminder_at="",
        reminder_recurrence=None,
    )
    return GuardDecision(_with_task(plan, task))


def looks_like_pending_followup(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if compact in {"取消", "算了", "不用了", "不设置了"}:
        return True
    if len(compact) > 40 or re.search(
        r"怎么|如何|什么|为什么|为何|是否|有空|天气|怎么样|吗|么|[?？]",
        compact,
    ):
        return False

    candidate = re.sub(
        r"^(?:请)?(?:好的|好|可以|确认|就|定在|改成|设为|设置为|时间是|日期是)+",
        "",
        compact,
    )
    field_patterns = (
        r"每(?:周|星期|礼拜)(?:[一二三四五六日天])?",
        r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}月\d{1,2}日|"
        r"今天|今日|明天|明日|后天|(?:本周|这周|下周)[一二三四五六日天]|"
        r"(?:周|星期|礼拜)[一二三四五六日天]",
        r"(?:(?:凌晨|早上|上午|中午|下午|傍晚|晚上)(?:(?:[01]?\d|2[0-3])(?:[:：点时](?:[0-5]?\d)?)?)?|"
        r"(?:[01]?\d|2[0-3])(?:[:：点时](?:[0-5]?\d)?))",
        r"(?:(?:共|总共|一共|连续)?[0-9一二两三四五六七八九十]{1,3}(?:次|周))",
    )
    found_field = False
    for pattern in field_patterns:
        candidate, count = re.subn(pattern, "", candidate)
        found_field = found_field or count > 0
    candidate = re.sub(r"(?:就行|即可|确认|吧)$", "", candidate)
    candidate = re.sub(r"[，,、；;和及+]|从|开始", "", candidate)
    return found_field and not candidate


def _pending_weekday(text: str) -> int:
    match = re.search(r"(?:每)?(?:周|星期|礼拜)\s*([一二三四五六日天])", text)
    return _WEEKDAY_ISO[match.group(1)] if match else 0


def _pending_repeat_count(text: str) -> int | None:
    parsed = _repeat_count(text)
    if parsed is not None:
        return parsed
    match = re.fullmatch(
        r"\s*(?:共|总共|一共|连续)?\s*"
        r"([0-9]{1,3}|[一二两三四五六七八九十]{1,3})\s*(?:次|周)\s*",
        text,
    )
    return _chinese_number(match.group(1)) if match else None


def _pending_explicit_start_date(text: str, resolved: str) -> str:
    if not resolved:
        return ""
    if re.search(
        r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}月\d{1,2}日|"
        r"今天|今日|明天|明日|后天|(?:本周|这周|下周)[一二三四五六日天]",
        text,
    ):
        return resolved
    return ""


def resume_pending_task(
    pending: PendingTaskClarification,
    text: str,
    now: datetime,
) -> GuardDecision:
    signals = extract_task_semantics(text, now)
    task = pending.task
    recurrence = task.reminder_recurrence
    if recurrence is not None:
        reminder_date = (
            _pending_explicit_start_date(text, signals.requested_date)
            or pending.reminder_date
        )
        reminder_time = signals.requested_time or pending.reminder_time
        weekday = signals.recurrence_weekday or _pending_weekday(text) or recurrence.weekday
        supplied_count = _pending_repeat_count(text)
        count = supplied_count if supplied_count is not None else recurrence.count
        recurrence = replace(recurrence, weekday=weekday, count=count)
        task = replace(task, reminder_recurrence=recurrence)
        weekday_text = _WEEKDAY_TEXT.get(weekday, "")
        start = f"{reminder_date} 开始，" if reminder_date else ""
        canonical = (
            f"{start}每周{weekday_text}{reminder_time}提醒我{task.title}，共{count}次"
        )
    else:
        reminder_date = signals.requested_date or pending.reminder_date
        reminder_time = signals.requested_time or pending.reminder_time
        canonical = f"{reminder_date} {reminder_time}提醒我{task.title}"
    plan = IntentPlan(kind=IntentKind.TASK, tasks=(task,), confidence=1.0)
    return validate_plan_semantics(canonical, plan, now, expected_kind=IntentKind.TASK)
