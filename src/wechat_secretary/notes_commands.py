"""Anchored, local-only commands. Quoted examples are never commands."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date

from .notes_search import NotesQuery


@dataclass(frozen=True)
class NotesCommand:
    action: str
    query: NotesQuery | None = None
    text: str = ""
    number: int | None = None
    error: str = ""


def parse_notes_command(content: str, now: datetime, *, has_focus: bool = False) -> NotesCommand | None:
    text = content.strip().rstrip("。！! ")
    if text == "重建笔记索引":
        return NotesCommand("rebuild")
    exclude = re.fullmatch(r"排除笔记\s+([^\s]+)", text)
    if exclude:
        return NotesCommand("exclude", text=exclude[1], number=int(exclude[1]) if exclude[1].isdigit() else None)
    answer = re.fullmatch(r"笔记问答\s*[：:]\s*(.*)", text, re.DOTALL)
    if answer:
        return NotesCommand("answer", text=answer[1].strip())
    if has_focus and re.fullmatch(r"(?:这[些几]?(?:条|个)?(?:笔记|方案|记录|结果).{0,60}(?:区别|不同|总结|比较|对比|共同点|详细说说)|(?:总结|比较|对比)(?:一下)?这[些几].{0,40})", text):
        return NotesCommand("answer", text=text)
    followup = re.fullmatch(r"第\s*(\d+)\s*条\s*(再详细说说|详细说说|讲讲|总结一下|内容是什么)", text)
    if has_focus and followup:
        return NotesCommand("answer", text=text, number=int(followup[1]))
    search = re.fullmatch(r"(?:请|帮我)?(?:搜索|检索|查找|找一下|找找)笔记\s*[：:]?\s*(.*)", text, re.DOTALL)
    if not search:
        search = re.fullmatch(r"(?:请|帮我)?(?:找一下|查找|搜索|找找)\s*((?:上个月|本月|这个月|今天|昨天|上周)?记的.+)", text, re.DOTALL)
    if not search:
        return None
    query = search[1].strip()
    if len(query) > 200:
        return NotesCommand("search", error="搜索条件请控制在200字以内。")
    today = now.date()
    start = end = None
    for phrase in ("上个月", "这个月", "本月", "今天", "昨天", "上周"):
        if phrase not in query:
            continue
        query = query.replace(phrase, " ", 1)
        if phrase == "上个月":
            end = today.replace(day=1) - timedelta(days=1)
            start = end.replace(day=1)
        elif phrase in {"本月", "这个月"}:
            start, end = today.replace(day=1), today
        elif phrase == "上周":
            end = today - timedelta(days=today.weekday() + 1)
            start = end - timedelta(days=6)
        else:
            start = end = today - timedelta(days=int(phrase == "昨天"))
        break
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", query)
    try:
        if len(dates) > 2:
            raise ValueError()
        if dates:
            start, end = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
            if start > end:
                raise ValueError()
            query = re.sub(r"\d{4}-\d{2}-\d{2}(?:\s*(?:至|到|~)\s*)?", " ", query)
    except ValueError:
        return NotesCommand("search", error="日期请用有效的 YYYY-MM-DD，日期范围按先后顺序填写。")
    tags = tuple(dict.fromkeys(re.findall(r"(?:#|标签\s*[：:])([\w\-\u4e00-\u9fff]+)", query)))
    query = re.sub(r"(?:#|标签\s*[：:])[\w\-\u4e00-\u9fff]+", " ", query)
    query = re.sub(r"^\s*记的", "", query).strip()
    if not query and not tags and start is None:
        return NotesCommand("search", error="请给出关键词、标签或日期，例如“搜索笔记：实验方案 #实验”。")
    return NotesCommand("search", query=NotesQuery(text=query, tags=tags, date_from=start, date_to=end))
