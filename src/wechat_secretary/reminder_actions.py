"""Bounded, side-effect-free controls for an already identified reminder.

Only outer commands match; words inside a task, note or quotation cannot select
an operation. Target IDs are resolved locally by the service, never from text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class ReminderAction:
    kind: str
    value: str = ""
    count: int = 0
    scope: str = ""
    title: str = ""


def small_number(value: str) -> int:
    if value.isdecimal():
        return int(value)
    if not re.fullmatch(r"[一二两三四五六七八九]|[一二两三四五六七八九]?十[一二三四五六七八九]?", value):
        return 0
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits["两"] = 2
    if "十" in value and value.count("十") == 1:
        first, last = value.split("十")
        return (digits.get(first, 1) * 10) + digits.get(last, 0)
    return digits.get(value, 0)


def parse_reminder_action(text: str) -> ReminderAction | None:
    text = re.sub(r"\s+", "", text).rstrip("。.!！")
    if re.search(r"[?？吗么“”\"「」『』]", text):
        return None
    text = re.sub(r"^(?:请|帮我|麻烦)", "", text)
    correction = re.fullmatch(r"不是(?:今天|明天|后天)[，,]是((?:今天|明天|后天).*)", text)
    if correction:
        return ReminderAction("update", value=correction.group(1))
    recent = r"(?:刚才|刚刚|上一个|最近)(?:的)?(?:那个|那条|这个)?"
    if re.fullmatch(rf"(?:{recent}|这个|这条)(?:提醒)?(?:不要了|不用了|取消了)", text):
        return ReminderAction("cancel")
    cancel = re.fullmatch(
        rf"(?:取消|撤销|停止|关闭)(?P<target>{recent}|这个|这条|本次|这次|全部|整个系列|所有后续)?(?:的)?提醒(?P<tail>整个系列|全部|本次|这次)?",
        text,
    )
    if cancel:
        target = (cancel.group("target") or "") + (cancel.group("tail") or "")
        scope = "all" if re.search(r"全部|所有|整个", target) else "next" if re.search(r"本次|这次", target) else ""
        return ReminderAction("cancel", scope=scope)
    named = re.fullmatch(r"(?:取消|停止)([^，,。；;]{1,100}?)(?:的)提醒", text)
    if named:
        return ReminderAction("cancel", title=named.group(1))
    update = re.fullmatch(
        rf"(?:把)?(?:(?:{recent}|这个|这条)(?:提醒)?(?:的时间)?)?(?:改成|改到|改为|调整到|延后到|推迟到)(.+)", text
    )
    if update:
        return ReminderAction("update", value=update.group(1))
    append = re.fullmatch(r"(?:再|追加|额外|还要|继续)(?:给我)?(?:提醒(?:我)?|加)?([0-9一二两三四五六七八九十]{1,3})次(?:[，,]?(.+))?", text)
    if append:
        return ReminderAction("append", value=append.group(2) or "", count=small_number(append.group(1)))
    return None


def repeat_interval(text: str) -> timedelta | None:
    text = re.sub(r"\s+", "", text).strip("，,。.!！")
    if text in {"每天", "每天这个时间", "每天一次", "每隔一天"}:
        return timedelta(days=1)
    if text in {"每周", "每周这个时间", "每周一次", "每隔一周"}:
        return timedelta(weeks=1)
    match = re.fullmatch(r"(?:每隔|间隔|隔|每)([0-9一二两三四五六七八九十]{1,3})(分钟|小时|天|周)(?:一次)?", text)
    if not match:
        return None
    amount = small_number(match.group(1))
    if not 1 <= amount <= 52:
        return None
    minutes = {"分钟": 1, "小时": 60, "天": 1440, "周": 10080}[match.group(2)] * amount
    return timedelta(minutes=minutes)
