"""Shared, conservative parsing of Chinese reminder schedules.

The token regexes deliberately include invalid/vague candidates. Callers must
first isolate schedule text from the requested task's body; these helpers do not
decide whether a sentence authorizes creating or changing a reminder.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta


_NUMBER = r"(?:\d+|[零〇○一二两三四五六七八九十百千万亿]+)"
_NUMBER_CHARS = r"\d零〇○一二两三四五六七八九十百千万亿"
_APPROXIMATION = r"(?:大约|大概|大致|约莫|差不多|将近|接近|约)(?:\s*在)?"
_PERIOD = r"凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里|夜间|午夜|半夜|深夜"
_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_DAY_OFFSETS = {"今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3}

PERIOD_TOKEN_RE = re.compile(rf"(?:{_PERIOD})(?!茶)")
DATE_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z\d])\d{{4,}}[-/.]\d+[-/.]\d+(?!\d)|"
    rf"(?<![{_NUMBER_CHARS}])(?:{_NUMBER}年)?{_NUMBER}月{_NUMBER}(?:日|号)|"
    r"大*后天|明天|明日|今天|今日|"
    r"(?:下下|上上|下|上|本|这|这个)?(?:周|星期|礼拜)\s*[一二三四五六日天]"
)

# Match the entire candidate, never just the valid-looking suffix of 25:00 or
# 15:70. Consumers also use its span for clarification and title extraction.
CLOCK_TOKEN_RE = re.compile(
    rf"(?:(?<![A-Za-z{_NUMBER_CHARS}.])|"
    r"(?<=周[一二三四五六日天])|(?<=星期[一二三四五六日天])|(?<=礼拜[一二三四五六日天]))"
    rf"(?<![{_NUMBER_CHARS}][:：])"
    r"(?!(?<=周)[一二三四五六日天])"
    r"(?!(?<=星期)[一二三四五六日天])"
    r"(?!(?<=礼拜)[一二三四五六日天])"
    rf"(?P<approx_before>{_APPROXIMATION}\s*)?"
    rf"(?P<period>{_PERIOD})?\s*"
    rf"(?P<approx_middle>{_APPROXIMATION}\s*)?"
    rf"(?P<hour>{_NUMBER})\s*"
    r"(?:"
    rf"(?P<colon>[:：])\s*(?P<colon_minute>{_NUMBER})?(?:\s*分)?"
    rf"(?P<colon_seconds>[:：]\s*{_NUMBER}?)?"
    r"|"
    r"(?P<unit>点|时)"
    r"(?:"
    rf"\s*(?P<fraction>半|{_NUMBER}刻)"
    r"|"
    rf"\s*(?P<minute>{_NUMBER})(?P<minute_approx>多|来)?(?:\s*分)?"
    r")?"
    rf"(?P<seconds>\s*{_NUMBER}秒)?"
    r")"
    r"(?:\s*(?P<clock_suffix>钟|整))?"
    r"(?P<approx_after>多(?:一点|钟)?|左右|前后|上下|许|出头|来钟|过一点)?"
    rf"(?![{_NUMBER_CHARS}])"
)

RELATIVE_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z{_NUMBER_CHARS}.负-])"
    rf"(?:{_APPROXIMATION}\s*)?"
    rf"(?:半|{_NUMBER})(?:多|来)?\s*(?:个\s*)?(?:半\s*)?(?:分钟|小时)"
    rf"(?:\s*(?:半|{_NUMBER})(?:多|来)?\s*(?:分钟|小时))*"
    r"(?:半)?(?:左右|前后)?\s*(?:以)?后"
)


def _clock_number(value: str) -> int | None:
    """Parse a complete conventional Chinese/Arabic number below one hundred."""
    if value.isdecimal():
        return int(value) if len(value) <= 2 else None
    digits = {char: number for number, char in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "○": 0, "两": 2})
    if len(value) == 1:
        return 10 if value == "十" else digits.get(value)
    if len(value) == 2 and value[0] in "零〇○" and value[1] in digits:
        return digits[value[1]]
    match = re.fullmatch(r"([一二三四五六七八九])?十([一二三四五六七八九])?", value)
    if match:
        return (digits[match[1]] if match[1] else 1) * 10 + (
            digits[match[2]] if match[2] else 0
        )
    return None


def resolve_date(text: str, now: datetime) -> str:
    """Resolve one explicit date; a missing, invalid or repeated date is empty.

    下周 means the next Monday-Sunday calendar week, not the nearest weekday
    plus another seven days. Month/day expressions keep the current year; past
    dates are for the caller to clarify, never silently moved to another year.
    """
    matches = list(DATE_TOKEN_RE.finditer(text))
    if len(matches) != 1:
        return ""
    token = re.sub(r"\s+", "", matches[0].group())
    exact = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", token)
    md = re.fullmatch(rf"(?:({_NUMBER})年)?({_NUMBER})月({_NUMBER})(?:日|号)", token)
    try:
        if exact:
            return datetime(int(exact[1]), int(exact[2]), int(exact[3])).date().isoformat()
        if md:
            year = now.year
            if md[1]:
                raw_year = md[1]
                if raw_year.isdecimal() and len(raw_year) == 4:
                    year = int(raw_year)
                elif re.fullmatch(r"[零〇○一二三四五六七八九]{4}", raw_year):
                    year = int("".join(str(_clock_number(char)) for char in raw_year))
                else:
                    return ""
            month, day = _clock_number(md[2]), _clock_number(md[3])
            if month is None or day is None:
                return ""
            return datetime(year, month, day).date().isoformat()
        if token in _DAY_OFFSETS:
            return (now.date() + timedelta(days=_DAY_OFFSETS[token])).isoformat()
        weekday = re.fullmatch(
            r"(下下|上上|下|上|本|这|这个)?(?:周|星期|礼拜)([一二三四五六日天])", token
        )
        if weekday:
            qualifier, target = weekday[1] or "", _WEEKDAY[weekday[2]]
            if qualifier:
                week_offset = {"下下": 14, "下": 7, "上上": -14, "上": -7,
                               "本": 0, "这": 0, "这个": 0}[qualifier]
                delta = target - now.weekday() + week_offset
            else:
                delta = (target - now.weekday()) % 7 or 7
            return (now.date() + timedelta(days=delta)).isoformat()
    except (ValueError, OverflowError):
        return ""
    return ""


def resolve_time(text: str, *, default_period: str = "") -> str:
    """Resolve one exact clock, optionally inheriting a previously stated period.

    Bare clocks retain their 24-hour interpretation for compatibility. A caller
    deciding that AM/PM is missing should ask instead of inventing a period.
    Evening/midnight twelve requires a date-boundary clarification; it must not
    become noon or be silently moved to the following day.
    """
    matches = list(CLOCK_TOKEN_RE.finditer(text))
    if len(matches) != 1:
        return ""
    match = matches[0]
    if any(match[name] for name in (
        "approx_before", "approx_middle", "approx_after", "minute_approx",
        "colon_seconds", "seconds",
    )):
        return ""
    if re.match(r"[.：:]|秒|刻|分|点|小时|时间|时长", text[match.end():]):
        return ""
    if match["unit"] == "时" and re.match(r"间|长|速|效|区|序", text[match.end():]):
        return ""
    hour = _clock_number(match["hour"])
    if hour is None or not 0 <= hour <= 23:
        return ""
    raw_minute = match["colon_minute"] if match["colon"] else match["minute"]
    if match["colon"] and raw_minute is None:
        return ""
    fraction = match["fraction"]
    if fraction:
        minute = {"半": 30, "一刻": 15, "1刻": 15, "三刻": 45, "3刻": 45}.get(fraction)
    else:
        minute = _clock_number(raw_minute) if raw_minute else 0
    if minute is None or not 0 <= minute <= 59:
        return ""
    periods = {candidate.group() for candidate in PERIOD_TOKEN_RE.finditer(text)}
    if len(periods) > 1:
        return ""
    # A colon clock is already a 24-hour representation: changing an afternoon
    # reminder to "09:00" means 09:00, not an inherited 21:00. Inheritance is for
    # conversational "三点" / "四点半" only; an explicit period always applies.
    period = match["period"] or next(iter(periods), "") or (
        default_period if not match["colon"] else ""
    )
    if period and not PERIOD_TOKEN_RE.fullmatch(period):
        return ""
    if period in {"晚上", "夜里", "夜间", "傍晚", "午夜", "半夜", "深夜"} and hour in {0, 12}:
        return ""
    if period in {"早上", "早晨", "上午"} and hour >= 12:
        return ""
    if period in {"午夜", "半夜", "深夜"}:
        return ""
    if period == "凌晨":
        if hour == 12:
            hour = 0
        elif hour > 11:
            return ""
    if period in {"下午", "傍晚", "晚上", "夜里", "夜间"}:
        if hour == 0:
            return ""
        if hour < 12:
            hour += 12
    if period == "中午":
        if hour == 1:
            hour = 13
        elif hour not in {11, 12, 13}:
            return ""
    return f"{hour:02d}:{minute:02d}"


def resolve_relative_time(text: str, now: datetime) -> datetime | None:
    """Resolve a single exact positive duration from the supplied reference time."""
    matches = list(RELATIVE_TOKEN_RE.finditer(text))
    if len(matches) != 1 or DATE_TOKEN_RE.search(text) or CLOCK_TOKEN_RE.search(text):
        return None
    match = re.fullmatch(
        rf"(半|{_NUMBER})\s*(?:个\s*)?(分钟|小时)\s*(?:以)?后", matches[0].group()
    )
    if not match:
        return None
    raw_amount = match[1]
    if raw_amount == "半":
        amount = 0.5
    elif raw_amount.isdecimal() and len(raw_amount) <= 3:
        amount = int(raw_amount)
    else:
        amount = _clock_number(raw_amount)
    if amount is None or not 0 < amount <= 999:
        return None
    delta = timedelta(hours=amount) if match[2] == "小时" else timedelta(minutes=amount)
    try:
        return now + delta
    except OverflowError:
        return None


def resolve_datetime(text: str, now: datetime) -> datetime | None:
    """Resolve a whole standalone schedule, with no ignored content or defaults."""
    # The ISO separator is syntax, not an alphabetic prefix of a clock token.
    schedule = re.sub(r"(?<=\d)T(?=\d)", " ", text.strip())
    relative = resolve_relative_time(schedule, now)
    if relative is not None and RELATIVE_TOKEN_RE.fullmatch(schedule):
        return relative
    if RELATIVE_TOKEN_RE.search(schedule):
        return None
    date = resolve_date(schedule, now)
    clock = resolve_time(schedule)
    if not date or not clock:
        return None
    spans = [match.span() for pattern in (DATE_TOKEN_RE, CLOCK_TOKEN_RE)
             for match in pattern.finditer(schedule)]
    remaining = list(schedule)
    for start, end in spans:
        remaining[start:end] = " " * (end - start)
    if not re.fullmatch(r"\s*(?:在\s*)?(?:的时候\s*)?", "".join(remaining)):
        return None
    try:
        return datetime.fromisoformat(f"{date}T{clock}:00").replace(tzinfo=now.tzinfo)
    except ValueError:
        return None
