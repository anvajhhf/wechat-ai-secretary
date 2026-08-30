from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .temporal import (
    CLOCK_TOKEN_RE,
    DATE_TOKEN_RE,
    PERIOD_TOKEN_RE,
    RELATIVE_TOKEN_RE,
    resolve_date as _resolve_date,
    resolve_relative_time,
    resolve_time as _resolve_time,
)
from .request_scope import (
    INDEPENDENT_REQUEST_CONNECTOR_RE,
    has_multiple_reminder_requests,
    has_negated_reminder,
    mask_quoted_text,
    note_source_body,
    outer_reminder_match,
    reminder_marker,
)
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
_UNSUPPORTED_RECURRENCE_RE = re.compile(
    r"每天|每日|每月|每个月|每季度|每年|隔周|每两周|无限|一直提醒|长期提醒"
)
_WEEKLY_RE = re.compile(r"每(?:周|星期|礼拜)\s*([一二三四五六日天])")
_BARE_WEEKLY_RE = re.compile(
    r"每(?:周|星期|礼拜)(?=\s*(?:(?:都|固定)\s*)?"
    r"(?:凌晨|早上|上午|中午|下午|傍晚|晚上|\d|提醒|通知|叫|[，,]))"
)
_COMPLEX_WEEKLY_RE = re.compile(
    r"(?:每)?(?:周|星期|礼拜)\s*[一二三四五六日天]"
    r"\s*(?:和|及|、|到|至|[-—~～])\s*"
    r"(?:(?:每)?(?:周|星期|礼拜))?\s*[一二三四五六日天]|"
    r"(?:工作日|每个工作日)|"
    r"每(?:周|星期|礼拜)[^，,。；;！？!?]{0,12}"
    r"每(?:周|星期|礼拜)"
)
_DUE_RE = re.compile(r"截止|到期|之前|最晚|日程|安排在")
_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9._+-]{2,}\b)[A-Za-z][A-Za-z0-9._+-]*\d[A-Za-z0-9._+-]*", re.I)
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
    rf"(?:{_COMPLEX_WEEKLY_RE.pattern}|"
    rf"{_UNSUPPORTED_RECURRENCE_RE.pattern}|"
    rf"每(?:周|星期|礼拜)(?:\s*[一二三四五六日天]|\s*(?:都|固定))?|"
    rf"{DATE_TOKEN_RE.pattern})"
)
_TASK_TIME_CONTROL_RE = CLOCK_TOKEN_RE
_TASK_PERIOD_CONTROL_RE = PERIOD_TOKEN_RE
_TASK_REPEAT_CONTROL_RE = re.compile(
    r"(?:连续\s*[0-9一二两三四五六七八九十]{1,3}\s*(?:周|次)|"
    r"(?:共|总共|一共)\s*[0-9一二两三四五六七八九十]{1,3}\s*次|"
    rf"{RELATIVE_TOKEN_RE.pattern})"
)


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
    date_supplied: bool = False
    clock_supplied: bool = False
    invalid_date: bool = False
    conflicting_schedule: bool = False
    date_conflict: bool = False
    clock_conflict: bool = False
    reminder_period: str = ""
    complex_recurrence: bool = False


@dataclass(frozen=True)
class GuardDecision:
    plan: IntentPlan
    reason: ClarificationReason = ClarificationReason.NONE
    question: str = ""
    pending: PendingTaskClarification | None = None

    @property
    def ready(self) -> bool:
        return self.reason is ClarificationReason.NONE and not self.question


def _reminder_marker(text: str) -> re.Match[str] | None:
    return reminder_marker(text)


def _split_leading_schedule(value: str) -> tuple[str, str]:
    """Separate a leading schedule from payload without scanning its body."""
    candidate = value.strip(" \t\r\n，,。.；;！？!?：:、")
    fragments: list[str] = []
    while candidate:
        updated = re.sub(r"^(?:在|从|到时候)\s*", "", candidate, count=1)
        for pattern in (
            _TASK_DATE_CONTROL_RE, _TASK_TIME_CONTROL_RE,
            _TASK_REPEAT_CONTROL_RE, _TASK_PERIOD_CONTROL_RE,
        ):
            found = pattern.match(updated)
            if found:
                # These are noun compounds, not clock expressions. In other
                # positions the body is never searched for scheduling tokens.
                if pattern is _TASK_TIME_CONTROL_RE and re.match(
                    r"水|估计|透视", updated[found.end():]
                ) and not found["period"]:
                    return candidate, "，".join(fragments)
                fragments.append(found.group())
                candidate = re.sub(
                    r"^\s*(?:的时候|时候)?\s*(?:开始)?\s*"
                    r"(?:(?:[，,。.、；;！!和及]|或者|或)\s*)*",
                    "", updated[found.end():],
                )
                break
        else:
            break
    return candidate, "，".join(fragments)


def _split_trailing_schedule(value: str) -> tuple[str, str]:
    candidate = value.strip(" \t\r\n，,。.；;！？!?：:、")
    candidate = re.sub(
        r"(?:(?:请你|请|务必|一定要|一定|记得|帮我|麻烦|劳驾)\s*)+$", "", candidate
    ).rstrip(" ，,。.、；;")
    fragments: list[str] = []
    while candidate:
        for pattern in (
            _TASK_TIME_CONTROL_RE, _TASK_DATE_CONTROL_RE,
            _TASK_REPEAT_CONTROL_RE, _TASK_PERIOD_CONTROL_RE,
        ):
            found = re.search(
                rf"(?:{pattern.pattern})\s*(?:的时候|时候)?\s*(?:开始)?\s*$", candidate
            )
            if found:
                fragments.insert(0, found.group())
                candidate = re.sub(
                    r"(?:(?:[\s，,。.、；;！!和及]|或者|或))+$", "", candidate[:found.start()]
                )
                break
        else:
            break
    return candidate, "，".join(fragments)


def _split_schedule_suffix(value: str) -> tuple[str, str]:
    """Extract only independently delimited, entirely temporal tail clauses.

    Body sentences such as '查看明天下午的会议安排' and quoted times stay
    untouched. A final '下午四点半' is a schedule refinement, not task content.
    """
    candidate = value.rstrip(" \t\r\n。.!！")
    fragments: list[str] = []
    while candidate:
        boundaries = list(re.finditer(r"[，,。;；、\n]|(?<!\d)\.|\.(?!\d)", mask_quoted_text(candidate)))
        if not boundaries:
            break
        boundary = boundaries[-1]
        tail = candidate[boundary.end():].strip()
        if mask_quoted_text(tail) != tail:
            break
        remainder, schedule = _split_leading_schedule(tail)
        if not schedule or remainder.strip(" \t\r\n。.!！，,；;、"):
            break
        fragments.insert(0, schedule)
        candidate = candidate[:boundary.start()].rstrip()
    return candidate, "，".join(fragments)


def _reminder_body_source(text: str) -> str:
    marker = _reminder_marker(text)
    if marker is None:
        return text
    body, _ = _split_leading_schedule(text[marker.end():])
    body, _ = _split_schedule_suffix(body)
    if len(_compact_grounding_text(body)) < 2:
        body, _ = _split_trailing_schedule(text[:marker.start()])
    return body


def has_compound_reminder_body(text: str) -> bool:
    """Time punctuation is not a separator between independent task bodies."""
    return bool(_STRONG_MULTI_TASK_SEPARATOR_RE.search(_reminder_body_source(text)))


def _reminder_schedule_source(text: str) -> str:
    marker = _reminder_marker(text)
    if marker:
        _, before = _split_trailing_schedule(text[:marker.start()])
        body, after = _split_leading_schedule(text[marker.end():])
        _, tail = _split_schedule_suffix(body)
        return f"{before} 提醒我 {after}，{tail}"
    # Quoted text never supplies scheduling fields on its own.
    return mask_quoted_text(text)


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


def _period_for_clock(clock: str) -> str:
    hour = int(clock.split(":", 1)[0])
    if hour < 6:
        return "凌晨"
    if hour < 12:
        return "上午"
    if hour == 12:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


def _reminder_iso(at: datetime) -> str:
    precision = "microseconds" if at.microsecond else "seconds" if at.second else "minutes"
    return at.isoformat(timespec=precision)


def extract_task_semantics(
    text: str, now: datetime, *, default_period: str = "",
) -> TaskSemanticSignals:
    schedule = _reminder_schedule_source(text)
    requested_date = _resolve_date(schedule, now)
    date_tokens = [
        found.group() for found in _TASK_DATE_CONTROL_RE.finditer(schedule)
        if not found.group().startswith("每")
        and not _COMPLEX_WEEKLY_RE.fullmatch(found.group())
        and not _UNSUPPORTED_RECURRENCE_RE.fullmatch(found.group())
    ]
    parsed_dates = [_resolve_date(token, now) for token in date_tokens]
    if parsed_dates and all(parsed_dates) and len(set(parsed_dates)) == 1:
        requested_date = parsed_dates[0]
    invalid_date = any(not parsed for parsed in parsed_dates)
    clock_tokens = list(CLOCK_TOKEN_RE.finditer(schedule))
    periods = tuple(dict.fromkeys(found.group() for found in PERIOD_TOKEN_RE.finditer(schedule)))
    inherited_period = default_period if PERIOD_TOKEN_RE.fullmatch(default_period) else ""
    clock_period = periods[0] if len(periods) == 1 else inherited_period
    clock_values = [
        "" if re.match(r"[.：:]|秒|刻|分|点|小时|时间|时长", schedule[found.end():]) else
        _resolve_time(found.group(), default_period=clock_period)
        for found in clock_tokens
    ]
    requested_time = clock_values[0] if clock_values and all(clock_values) and len(set(clock_values)) == 1 else ""
    date_conflict = len(set(parsed_dates)) > 1
    clock_conflict = (len(clock_tokens) > 1 and (not all(clock_values) or len(set(clock_values)) > 1)) or len(periods) > 1
    conflicting_schedule = date_conflict or clock_conflict
    if invalid_date or date_conflict:
        requested_date = ""
    if clock_conflict:
        requested_time = ""
    negated_reminder = has_negated_reminder(text)
    requests_reminder = bool(_reminder_marker(text)) and not negated_reminder
    # Recurrence controls belong to the outer schedule, not a book title or
    # arbitrary words such as "检查每天备份的日志" in the reminder body.
    weekly = _WEEKLY_RE.search(schedule)
    bare_weekly = bool(_BARE_WEEKLY_RE.search(schedule))
    marker = _reminder_marker(text)
    recurrence_scope = mask_quoted_text(text[:marker.start()]) if marker else schedule
    complex_recurrence = bool(_COMPLEX_WEEKLY_RE.search(schedule) or _COMPLEX_WEEKLY_RE.search(recurrence_scope))
    unsupported = bool(_UNSUPPORTED_RECURRENCE_RE.search(schedule) or _UNSUPPORTED_RECURRENCE_RE.search(recurrence_scope))
    repeat_count = _repeat_count(text)
    recurrence_requested = bool(
        weekly or bare_weekly or unsupported or complex_recurrence
        or re.search(r"重复提醒|循环提醒", recurrence_scope)
        or (repeat_count is not None and repeat_count != 1)
    )
    recurrence_frequency = (
        "unsupported" if unsupported or complex_recurrence else "weekly" if weekly or bare_weekly else ""
    )
    weekday = _WEEKDAY_ISO[weekly.group(1)] if weekly and not complex_recurrence else 0
    recurrence_start_explicit = bool(
        re.search(
            r"(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}月\d{1,2}日|"
            r"今天|明天|后天|(?:本周|这周|下周)[一二三四五六日天])",
            schedule,
        )
    )
    relative_at = resolve_relative_time(schedule, now)
    relative = _reminder_iso(relative_at) if relative_at else ""
    if relative and (date_tokens or clock_tokens):
        conflicting_schedule = True
        date_conflict = clock_conflict = True
        requested_date = requested_time = relative = ""
    identifiers = tuple(dict.fromkeys(match.group(0) for match in _IDENTIFIER_RE.finditer(text)))
    explicit_24h = any(
        found["colon"] or (value and (int(value[:2]) == 0 or int(value[:2]) >= 13))
        for found, value in zip(clock_tokens, clock_values)
    )
    reminder_period = (
        periods[0] if len(periods) == 1 else "ambiguous" if len(periods) > 1 else
        _period_for_clock(requested_time) if requested_time and explicit_24h else
        inherited_period if requested_time and inherited_period else
        "unknown" if requested_time and clock_tokens else ""
    )
    if default_period == "ambiguous" and not periods and not explicit_24h:
        requested_time, reminder_period = "", "ambiguous"
    # Infer only the current day's still-future, unambiguous time. A bare 4:30
    # expressed as '四点半' is NOT evidence that the user meant early morning.
    if (requests_reminder and not recurrence_requested and not date_tokens
        and requested_time and reminder_period not in {"", "unknown", "ambiguous"}
        and not conflicting_schedule and not relative):
        today_at = datetime.fromisoformat(f"{now.date().isoformat()}T{requested_time}").replace(tzinfo=now.tzinfo)
        if today_at > now:
            requested_date = now.date().isoformat()
    return TaskSemanticSignals(
        requests_reminder=requests_reminder,
        negated_reminder=negated_reminder,
        requested_date=requested_date,
        requested_time=requested_time,
        relative_reminder_at=relative,
        recurrence_requested=recurrence_requested,
        recurrence_frequency=recurrence_frequency,
        recurrence_weekday=weekday,
        repeat_count=repeat_count,
        recurrence_start_explicit=recurrence_start_explicit,
        explicit_due=bool(_DUE_RE.search(text)),
        identifiers=identifiers,
        date_supplied=bool(date_tokens),
        clock_supplied=bool(clock_tokens or _TASK_PERIOD_CONTROL_RE.search(schedule)),
        invalid_date=invalid_date,
        conflicting_schedule=conflicting_schedule,
        date_conflict=date_conflict,
        clock_conflict=clock_conflict,
        reminder_period=reminder_period,
        complex_recurrence=complex_recurrence,
    )


def _clarify(
    plan: IntentPlan,
    reason: ClarificationReason,
    question: str,
    task: TaskDraft | None = None,
    *,
    reminder_date: str = "",
    reminder_time: str = "",
    reminder_period: str = "",
) -> GuardDecision:
    pending = None
    if task is not None:
        pending = PendingTaskClarification(
            reason=reason,
            task=task,
            reminder_date=reminder_date,
            reminder_time=reminder_time,
            reminder_period=reminder_period,
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


def _substantive_task_title(title: str) -> bool:
    compact = _compact_grounding_text(title)
    if len(compact) < 2:
        return False
    return not bool(re.fullmatch(
        r"(?:(?:再|还要|继续|一共|总共|共|连续)?"
        r"[0-9零一二两三四五六七八九十百]{1,4}(?:次|周)|"
        r"(?:刚才|刚刚|上面|前面|上一个)?(?:那个|这个|那件事|这件事|那条|这条|这个任务|那个任务)|"
        r"记得|提醒|一下|的时候|到时候)",
        compact,
    ))


def _missing_body_decision(
    plan: IntentPlan, signals: TaskSemanticSignals, question: str,
) -> GuardDecision:
    # The empty title is deliberately persisted only as a clarification draft;
    # it is never an executable TaskDraft and model-invented titles are dropped.
    recurrence = None
    if signals.recurrence_requested:
        recurrence = ReminderRecurrence(
            frequency=signals.recurrence_frequency,
            weekday=signals.recurrence_weekday,
            count=signals.repeat_count or 0,
        )
    date, clock = signals.requested_date, signals.requested_time
    if signals.relative_reminder_at:
        at = datetime.fromisoformat(signals.relative_reminder_at)
        date, clock = at.date().isoformat(), at.strftime("%H:%M")
    if recurrence and not signals.recurrence_start_explicit:
        date = ""
    return _clarify(
        plan, ClarificationReason.MISSING_TASK_BODY, question,
        TaskDraft("", reminder_at=signals.relative_reminder_at, reminder_recurrence=recurrence),
        reminder_date=date, reminder_time=clock,
        reminder_period=signals.reminder_period,
    )


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

    candidate = _split_leading_schedule(suffix)[0] if marker else ""
    candidate, _ = _split_schedule_suffix(candidate)
    candidate = re.sub(r"^一下\s*", "", candidate)
    candidate = re.sub(r"^一次(?=.{2,})", "", candidate)
    candidate = re.sub(
        r"[，,、；;]?\s*(?:(?:共|总共|一共)\s*"
        r"[0-9一二两三四五六七八九十]{1,3}\s*次|"
        r"连续(?:提醒(?:我)?)?\s*[0-9一二两三四五六七八九十]{1,3}\s*(?:周|次))\s*$",
        "",
        candidate,
    )
    if marker and len(_compact_grounding_text(candidate)) < 2:
        candidate = _split_trailing_schedule(prefix)[0]
    candidate = _strip_outer_request_politeness(candidate, text)
    candidate = re.sub(r"[\s，,。；;！？!?：:、]+", " ", candidate)
    return (candidate.strip() if _is_question_task_body(candidate) else candidate.strip(" 吧呢啊呀"))[:300]


def _is_question_task_body(text: str) -> bool:
    return bool(re.match(
        r"^\s*(?:问(?:一下|问)?|询问|咨询|请教|确认|"
        r"向[^，,。；;！？!?]{1,20}(?:询问|请教|确认)|"
        r"找[^，,。；;！？!?]{1,20}问)",
        text,
    ))


def _strip_outer_request_politeness(body: str, source: str) -> str:
    # An independently separated courtesy clause belongs to the request; a
    # question asked *by the future task* is part of its title, including 吗/呢.
    courtesy = r"(?:可以吗|好吗|行吗|没问题吗|好吧|行吧)"
    body = re.sub(rf"[，,；;]\s*{courtesy}\s*[?？]?$", "", body)
    if _is_question_task_body(body):
        return body
    body = re.sub(rf"{courtesy}\s*[?？]?$", "", body)
    marker = _reminder_marker(source)
    outer = source[:marker.start()] if marker else ""
    if re.search(r"请|麻烦|劳驾|帮我|你能|能否|可否|能不能|可不可以|可以", outer):
        body = re.sub(r"[吗么]\s*[?？]?$", "", body)
    return body


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
        if matched:
            updated = re.sub(r"^\s*(?:的时候|时候)?\s*(?:开始)?", "", updated)
        updated = re.sub(r"^\s*[，,、；;]?\s*", "", updated)
        if not matched and updated == candidate:
            break
        candidate = updated
    candidate = re.sub(r"[\s，,。；;！？!?：:、]+", " ", candidate)
    return candidate.strip(" 吧呢啊呀")[:300]


def _source_priority(text: str) -> str:
    # Only a standalone priority instruction controls metadata. Mentioning an
    # important protein, an emergency procedure, or quoted instructions does
    # not raise a task's priority, even if the model says otherwise.
    level = r"(?P<level>高|中|低|无|默认|普通)"
    lead = r"(?:(?:请|麻烦|帮我)\s*)?(?:(?:把|将)\s*)?(?:(?:这个|这项|该)?(?:任务|待办)(?:的)?\s*)?"
    setting = r"(?:设为|设置为|设成|改为|改成|调整为|标为|标记为|是|为|[：:])?"
    label_first = re.compile(rf"{lead}优先级\s*{setting}\s*{level}(?:优先级)?")
    level_first = re.compile(rf"{lead}{setting}\s*{level}优先级")
    levels: set[str] = set()
    rejected: set[str] = set()
    for clause in re.split(r"[，,。；;！？!?\n]", mask_quoted_text(text)):
        clause = clause.strip()
        matched = label_first.fullmatch(clause) or level_first.fullmatch(clause)
        if matched:
            levels.add(matched.group("level"))
            continue
        negated = re.fullmatch(
            rf"{lead}(?:优先级\s*)?(?:不要|不用|不是|别|无需)\s*"
            rf"{setting}\s*(?:优先级\s*)?{level}(?:优先级)?",
            clause,
        )
        if negated:
            rejected.add(negated.group("level"))
    if len(levels) != 1 or levels & rejected:
        return "none"
    return {"高": "high", "中": "medium", "低": "low"}.get(next(iter(levels)), "none")


def _source_category(text: str, candidate: str) -> str:
    if not candidate:
        return ""
    # A category name inside a title, quote or negative instruction is not
    # permission to move the task into that category.
    lead = r"(?:(?:请|麻烦|帮我)\s*)?(?:(?:把|将)\s*)?(?:(?:这个|这项|该)?(?:任务|待办)(?:的)?\s*)?"
    setting = r"(?:分类|清单)\s*(?:设为|设置为|是|为|[：:])\s*|(?:归入|归到|归类到|放入|放到)\s*"
    pattern = re.compile(rf"{lead}(?:{setting})(?P<category>.+?)(?:清单|分类)?")
    categories: set[str] = set()
    rejected: set[str] = set()
    expected = _compact_grounding_text(candidate)
    for clause in re.split(r"[，,。；;！？!?\n]", mask_quoted_text(text)):
        clause = clause.strip()
        # Retain the established comma-separated shorthand '…, 高优先级，工作'.
        # Only an entire unquoted clause can be a shorthand category label.
        if _compact_grounding_text(clause) == expected:
            categories.add(expected)
            continue
        matched = pattern.fullmatch(clause)
        if matched:
            categories.add(_compact_grounding_text(matched.group("category")))
            continue
        negated = re.fullmatch(
            rf"{lead}(?:(?:分类|清单)\s*)?(?:不要|不用|不是|别|无需)\s*"
            rf"(?:{setting})?(?P<category>.+?)(?:清单|分类)?", clause,
        )
        if negated:
            rejected.add(_compact_grounding_text(negated.group("category")))
    return candidate if categories == {expected} and expected not in rejected else ""


def _sanitize_task_metadata(text: str, task: TaskDraft) -> TaskDraft:
    return replace(
        task,
        priority=_source_priority(text),
        category=_source_category(text, task.category),
        tags=(),
        description="",
    )


def _note_source_body(text: str, *, mixed: bool = False) -> str:
    body = note_source_body(text)
    if mixed:
        body = re.split(
            r"(?:，|,|；|;)\s*(?:另外|同时|并且|还有|顺便)", body, maxsplit=1,
        )[0]
    return body.strip()


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


def _ground_plain_note(text: str, plan: IntentPlan, *, mixed: bool = False) -> GuardDecision:
    body = _note_source_body(text, mixed=mixed)
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

    if plan.kind is IntentKind.QUERY and expected_kind is not IntentKind.QUERY:
        return _clarify(plan, ClarificationReason.AMBIGUOUS_INTENT,
                        "我没有识别到当前的任务查询请求，因此没有读取任务。请说明要查询的范围或关键词。")

    if expected_kind is IntentKind.QUERY:
        if (
            plan.kind is IntentKind.CLARIFY
            and not plan.tasks
            and not plan.notes
            and plan.query is None
            and plan.clarification.strip()
        ):
            return _clarify(
                plan,
                plan.clarification_reason
                if plan.clarification_reason is not ClarificationReason.NONE
                else ClarificationReason.AMBIGUOUS_INTENT,
                plan.clarification.strip(),
            )
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
        grounded_note = _ground_plain_note(text, plan, mixed=True)
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
    reminder_scope = text
    if expected_kind is IntentKind.MIXED:
        for connector in INDEPENDENT_REQUEST_CONNECTOR_RE.finditer(mask_quoted_text(text)):
            tail = text[connector.end():]
            if outer_reminder_match(tail) is not None:
                reminder_scope = tail
                break
    if has_multiple_reminder_requests(reminder_scope):
        return _clarify(
            plan, ClarificationReason.SEMANTIC_MISMATCH,
            "这句话包含多个独立提醒，日期和时间需要分别绑定。为避免合并或漏掉其中一项，请把每个提醒分开发送。",
        )
    if (
        not plan.tasks
        and expected_kind is IntentKind.TASK
        and signals.requests_reminder
        and not has_compound_reminder_body(text)
    ):
        # A model clarification must not discard an explicit, source-grounded
        # reminder body. Rebuild only the locally authorized single task; the
        # same date/time validation below still decides whether it can execute.
        title = _deterministic_task_title(text)
        if _substantive_task_title(title):
            plan = replace(
                plan,
                kind=IntentKind.TASK,
                tasks=(TaskDraft(title=title),),
                notes=(),
                query=None,
                clarification="",
                clarification_reason=ClarificationReason.NONE,
            )
    if not plan.tasks:
        question = (
            "我听懂你想设置提醒，但还没识别出具体要做什么。请补充提醒事项。"
            if signals.requests_reminder
            else "我还没识别出具体任务内容，请再补充并说清楚要做什么。"
        )
        if signals.requests_reminder:
            return _missing_body_decision(plan, signals, question)
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
        if not _substantive_task_title(title):
            return _missing_body_decision(
                plan, signals,
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

    if signals.conflicting_schedule or signals.invalid_date:
        if signals.recurrence_requested:
            task = replace(task, reminder_recurrence=ReminderRecurrence(
                weekday=signals.recurrence_weekday, count=signals.repeat_count or 0,
            ))
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH
            if signals.conflicting_schedule else ClarificationReason.MISSING_REMINDER_DATE,
            "我记住了提醒事项，但识别到了多个日期或时刻。请确认一个日期和具体时间。"
            if signals.conflicting_schedule
            else "我记住了提醒事项，但这个日期无效。请补充一个有效日期。",
            task,
            reminder_date=signals.requested_date,
            reminder_time=signals.requested_time,
            reminder_period=signals.reminder_period,
        )

    if signals.recurrence_requested:
        if not signals.requests_reminder:
            return _clarify(
                plan,
                ClarificationReason.UNSUPPORTED_RECURRENCE,
                "我识别到了重复任务，但当前只支持明确说出的有限微信提醒。若你需要提醒，请说“每周二上午9点提醒我……，共3次”。",
            )
        if signals.recurrence_frequency != "weekly":
            pending_task = replace(task, reminder_at="", reminder_recurrence=ReminderRecurrence(
                frequency=signals.recurrence_frequency,
                weekday=0,
                count=signals.repeat_count or 0,
            ))
            return _clarify(
                plan,
                ClarificationReason.UNSUPPORTED_RECURRENCE,
                "目前还不支持每周多个星期几或星期范围，我没有截取其中一天。事项已记住，请改成单个星期几，或把各天分开发送。"
                if signals.complex_recurrence else
                "我记住了事项和次数，但还需要明确频率。目前支持单个星期几的有限提醒，例如“每周二上午9点，共3次”。",
                pending_task,
                reminder_date=signals.requested_date if signals.recurrence_start_explicit else "",
                reminder_time=signals.requested_time,
                reminder_period=signals.reminder_period,
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
                reminder_period=signals.reminder_period,
            )
        if signals.repeat_count is None:
            return _clarify(
                plan,
                ClarificationReason.MISSING_RECURRENCE_COUNT,
                "我已识别到每周提醒，还差总次数（2—52次），例如“共3次”。",
                pending_task,
                reminder_date=recurrence_start_date,
                reminder_time=signals.requested_time,
                reminder_period=signals.reminder_period,
            )
        if not 2 <= signals.repeat_count <= 52:
            return _clarify(
                plan,
                ClarificationReason.UNSUPPORTED_RECURRENCE,
                "每周提醒的总次数需要在2到52次之间，请换一个次数。",
                pending_task,
                reminder_date=recurrence_start_date,
                reminder_time=signals.requested_time,
                reminder_period=signals.reminder_period,
            )
        if signals.requested_time and signals.reminder_period == "unknown":
            return _clarify(plan, ClarificationReason.MISSING_REMINDER_TIME,
                            "事项和重复次数已保留。这个钟点是上午还是下午？",
                            pending_task, reminder_date=recurrence_start_date,
                            reminder_time=signals.requested_time, reminder_period="unknown")
        if not signals.requested_time:
            return _clarify(
                plan,
                ClarificationReason.MISSING_REMINDER_TIME,
                "我已识别到每周提醒和总次数，还差具体几点，例如“上午9点”。",
                pending_task,
                reminder_date=recurrence_start_date,
                reminder_period=signals.reminder_period,
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
                    pending_task,
                    reminder_date=recurrence_start_date,
                    reminder_time=signals.requested_time,
                    reminder_period=signals.reminder_period,
                )
            if requested_first <= now:
                return _clarify(
                    plan,
                    ClarificationReason.SEMANTIC_MISMATCH,
                    "重复提醒的开始时间已经过去，请给我一个新的开始日期和时间。",
                    pending_task,
                    reminder_date=recurrence_start_date,
                    reminder_time=signals.requested_time,
                    reminder_period=signals.reminder_period,
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
        if signals.requested_time and signals.reminder_period == "unknown":
            return _clarify(plan, ClarificationReason.MISSING_REMINDER_TIME,
                            "事项和钟点已记住。你说的是上午还是下午？",
                            task, reminder_date=signals.requested_date,
                            reminder_time=signals.requested_time, reminder_period="unknown")
        reminder_at = signals.relative_reminder_at
        if not reminder_at:
            if not signals.requested_date and not signals.requested_time:
                return _clarify(
                    plan,
                    ClarificationReason.MISSING_REMINDER_DATE_TIME,
                    "我已识别到提醒事项，还差日期和具体时间。",
                    task,
                    reminder_period=signals.reminder_period,
                )
            if not signals.requested_date:
                return _clarify(
                    plan,
                    ClarificationReason.MISSING_REMINDER_DATE,
                    "我已识别到提醒时间，还差哪一天。",
                    task,
                    reminder_time=signals.requested_time,
                    reminder_period=signals.reminder_period,
                )
            if not signals.requested_time:
                return _clarify(
                    plan,
                    ClarificationReason.MISSING_REMINDER_TIME,
                    "我已识别到提醒日期，还差具体几点，例如“上午9点”。",
                    task,
                    reminder_date=signals.requested_date,
                    reminder_period=signals.reminder_period,
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
                reminder_date=parsed.date().isoformat(),
                reminder_time=parsed.strftime("%H:%M"),
                reminder_period=signals.reminder_period,
            )
        task = replace(
            task,
            due_date=task.due_date if signals.explicit_due else "",
            due_time=task.due_time if signals.explicit_due else "",
            reminder_at=_reminder_iso(parsed),
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


def is_pending_cancellation(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).rstrip("。.!！")
    return compact in {
        "取消", "算了", "不用了", "不设置了", "取消刚才那个", "刚才那个不要了",
        "刚才那个取消", "刚才那条不要了", "这个提醒取消", "取消这个提醒",
    }


def looks_like_pending_followup(text: str) -> bool:
    return _only_pending_fields(_corrected_pending_fields(text))


def _only_pending_fields(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).rstrip("。.!！")
    if is_pending_cancellation(compact):
        return True
    if len(compact) > 40 or re.search(
        r"怎么|如何|什么|为什么|为何|是否|有空|天气|怎么样|吗|么|[?？]",
        compact,
    ):
        return False

    candidate = re.sub(
        r"^(?:请)?(?:好的|好|可以|确认|就|定在|改成|换成|设为|设置为|时间是|日期是)+",
        "",
        compact,
    )
    candidate = re.sub(r"^(?:再|还要|还|继续)(?:提醒|通知)(?:我)?", "", candidate)
    field_patterns = (
        _COMPLEX_WEEKLY_RE.pattern,
        RELATIVE_TOKEN_RE.pattern,
        r"每(?:周|星期|礼拜)(?:[一二三四五六日天])?",
        DATE_TOKEN_RE.pattern,
        CLOCK_TOKEN_RE.pattern,
        _TASK_PERIOD_CONTROL_RE.pattern,
        r"(?:(?:共|总共|一共|连续)?[0-9一二两三四五六七八九十]{1,3}(?:次|周))",
    )
    found_field = False
    for pattern in field_patterns:
        candidate, count = re.subn(pattern, "", candidate)
        found_field = found_field or count > 0
    candidate = re.sub(r"(?:就行|即可|确认|吧)$", "", candidate)
    candidate = re.sub(r"[，,、；;和及+]|或者|或|从|开始|的时候|时候", "", candidate)
    return found_field and not candidate


def _corrected_pending_fields(text: str) -> str:
    """Discard an explicitly rejected field, never a clause inside task text."""
    candidate = text.strip().rstrip("。.!！")
    corrected = re.fullmatch(
        r"(?:不对[，,]?\s*)?不是\s*(.+?)\s*[，,]?\s*(?:而是|是|改成|换成)\s*(.+)",
        candidate,
    )
    if corrected and all(_only_pending_fields(part) for part in corrected.groups()):
        return corrected.group(2)
    return text


def looks_like_pending_body(text: str) -> bool:
    """A narrow body-only continuation, used only for an empty reminder draft."""
    candidate = text.strip().strip(" ，,。.!！")
    candidate = re.sub(r"^(?:事项是|事情是|要做的是)\s*[：:]?\s*", "", candidate)
    if not _substantive_task_title(candidate) or len(candidate) > 300:
        return False
    if looks_like_pending_followup(candidate) or is_pending_cancellation(candidate):
        return False
    if has_negated_reminder(candidate) or _reminder_marker(candidate):
        return False
    if re.search(
        r"[?？]|(?:吗|么|呢)$|^(?:你|您|为什么|怎么|如何|什么|是否|有没有|能不能|可以吗|不用|不要|取消|算了|不对|不是|改成|换成)|"
        r"^(?:待办|任务|笔记|私密|完成|查询|补设提醒|设置提醒)\s*[：:]|"
        r"^(?:帮我|请)?(?:记一下|记下来|记录一下|保存|新建|创建|查任务|查待办)",
        candidate,
    ):
        return False
    # A new schedule/request must be routed on its own, not smuggled into the
    # empty slot as a task title. Date-like words inside a noun are left intact.
    body, schedule = _split_leading_schedule(candidate)
    if schedule or not body:
        return False
    if re.fullmatch(r"(?:好的?|嗯|哦|知道了|谢谢|随便|都行|那个|这个|几点|哪天|几次|下午|上午)", candidate):
        return False
    return bool(re.match(
        r"(?:购买|买|卖|喝|吃|服用|拿|取|还|归还|借|给|让|去|回|打|开|关|发|送|"
        r"寄|收|查|检查|看|阅读|读|学习|复习|写|改|修改|优化|测试|修复|更新|安装|"
        r"卸载|下载|备份|清理|整理|提交|联系|准备|预约|缴费|付款|续费|完成|处理|"
        r"确认|核对|打印|报销|签到|参加|接|接送|跑步|运动|锻炼|洗|浇|喂|做|办理)",
        candidate,
    ))


def _pending_weekday(text: str) -> int:
    match = re.search(r"(?:每)?(?:周|星期|礼拜)\s*([一二三四五六日天])", text)
    return _WEEKDAY_ISO[match.group(1)] if match else 0


def _pending_repeat_count(text: str) -> int | None:
    text = text.strip().rstrip("。.!！")
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
    text = _corrected_pending_fields(text)
    task = pending.task
    is_body_reply = not task.title and looks_like_pending_body(text)
    if not is_body_reply and not looks_like_pending_followup(text):
        return GuardDecision(
            plan=IntentPlan(kind=IntentKind.CLARIFY, clarification_reason=pending.reason),
            reason=pending.reason,
            question="原来的提醒信息仍已保留；请补充具体事项。" if not task.title else "原来的提醒事项仍已保留，请只补充需要的日期、时间或次数。",
            pending=pending,
        )
    if is_body_reply:
        title = re.sub(r"^(?:事项是|事情是|要做的是)\s*[：:]?\s*", "", text.strip())
        task = replace(task, title=title.rstrip("。.!！"))
        signals = TaskSemanticSignals()
    else:
        signals = extract_task_semantics(text, now, default_period=pending.reminder_period)
    period = signals.reminder_period or pending.reminder_period
    if signals.clock_conflict and not signals.reminder_period:
        period = ""
    # An approximate/invalid replacement is still a supplied clock, not an
    # omission: never silently reuse a previously precise time in that case.
    reminder_time = (
        signals.requested_time
        if signals.clock_supplied
        else signals.requested_time or pending.reminder_time
    )
    reminder_date = signals.requested_date if signals.date_supplied else pending.reminder_date
    if (pending.reminder_period == "unknown" and pending.reminder_time
        and signals.reminder_period and PERIOD_TOKEN_RE.fullmatch(signals.reminder_period)
        and not CLOCK_TOKEN_RE.search(text) and not signals.clock_conflict):
        hour, minute = map(int, pending.reminder_time.split(":"))
        reminder_time = _resolve_time(f"{signals.reminder_period}{hour}点{minute}分")
    if signals.relative_reminder_at:
        relative = datetime.fromisoformat(signals.relative_reminder_at)
        reminder_date, reminder_time = relative.date().isoformat(), relative.strftime("%H:%M")
    plan = IntentPlan(kind=IntentKind.TASK, tasks=(task,), confidence=1.0)
    anchored_at = (
        signals.relative_reminder_at
        or (pending.task.reminder_at if is_body_reply and not pending.task.reminder_recurrence else "")
    )
    if anchored_at and task.title and task.reminder_recurrence is None:
        anchored = datetime.fromisoformat(anchored_at).astimezone(now.tzinfo)
        if anchored > now:
            return GuardDecision(_with_task(plan, replace(task, reminder_at=_reminder_iso(anchored))))
        return _clarify(
            plan, ClarificationReason.SEMANTIC_MISMATCH,
            "原来指定的提醒时间已经过去，事项已保留，请重新给我日期和时间。",
            task, reminder_date=anchored.date().isoformat(),
            reminder_time=anchored.strftime("%H:%M"), reminder_period=_period_for_clock(anchored.strftime("%H:%M")),
        )
    if signals.conflicting_schedule or signals.invalid_date:
        return _clarify(
            plan,
            ClarificationReason.SEMANTIC_MISMATCH
            if signals.conflicting_schedule else ClarificationReason.MISSING_REMINDER_DATE,
            "事项已保留，请只确认一个日期和具体时间。"
            if signals.conflicting_schedule
            else "事项已保留，但这个日期无效，请重新补充日期。",
            task,
            reminder_date=reminder_date,
            reminder_time=reminder_time,
            reminder_period=period,
        )
    recurrence = task.reminder_recurrence
    supplied_count = None if is_body_reply else _pending_repeat_count(text)
    if recurrence is None and (signals.recurrence_requested or supplied_count is not None):
        recurrence = ReminderRecurrence(
            frequency=signals.recurrence_frequency,
            weekday=signals.recurrence_weekday,
            count=supplied_count if supplied_count is not None else signals.repeat_count or 0,
        )
    if recurrence is not None:
        reminder_date = (
            _pending_explicit_start_date(text, signals.requested_date)
            if signals.date_supplied else pending.reminder_date
        )
        weekday = signals.recurrence_weekday or (0 if is_body_reply else _pending_weekday(text)) or recurrence.weekday
        count = supplied_count if supplied_count is not None else recurrence.count
        frequency = recurrence.frequency
        if signals.complex_recurrence:
            frequency, weekday = "unsupported", 0
        elif signals.recurrence_frequency:
            frequency = signals.recurrence_frequency
        elif not is_body_reply and _pending_weekday(text):
            frequency = "weekly"
        recurrence = replace(recurrence, frequency=frequency, weekday=weekday, count=count)
        task = replace(task, reminder_recurrence=recurrence)
        if frequency != "weekly":
            return _clarify(
                plan, ClarificationReason.MISSING_TASK_BODY if not task.title else ClarificationReason.UNSUPPORTED_RECURRENCE,
                "提醒次数和时间已保留，请先补充具体要提醒什么。" if not task.title else
                "事项和其它信息已保留，目前不支持多个星期几或星期范围。请确认单个星期几，例如“每周二”。"
                if signals.complex_recurrence else
                "事项和次数已保留，请明确提醒频率。目前支持单个星期几，例如“每周二”，不会把次数当成新事项。",
                task, reminder_date=reminder_date, reminder_time=reminder_time,
                reminder_period=period,
            )
        weekday_text = _WEEKDAY_TEXT.get(weekday, "")
        start = f"{reminder_date} 开始，" if reminder_date else ""
        clock = _pending_clock_text(reminder_time, period)
        canonical = (
            f"{start}每周{weekday_text} {clock}提醒我{task.title}，共{count}次"
        )
    else:
        clock = _pending_clock_text(reminder_time, period)
        canonical = f"{reminder_date} {clock}提醒我{task.title}"
    plan = IntentPlan(kind=IntentKind.TASK, tasks=(task,), confidence=1.0)
    decision = validate_plan_semantics(canonical, plan, now, expected_kind=IntentKind.TASK)
    if decision.pending is not None:
        decision = replace(decision, pending=replace(
            decision.pending,
            reminder_period=decision.pending.reminder_period or period,
            source_message_id=pending.source_message_id,
            last_received_at=pending.last_received_at,
        ))
    return decision


def _pending_clock_text(clock: str, period: str) -> str:
    if clock and period == "unknown":
        hour, minute = map(int, clock.split(":"))
        return f"{hour}点{minute}分"
    return clock or (period if period not in {"unknown", "ambiguous"} else "")
