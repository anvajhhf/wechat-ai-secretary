"""Read-only arrangement queries, using recorded task/reminder facts only."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time
import re
from typing import Any

from .request_scope import mask_quoted_text


@dataclass(frozen=True)
class ScheduleQuery:
    mode: str = "range"
    label: str = "未来7天（含今天）"
    start: datetime | None = None
    end: datetime | None = None  # exclusive, Shanghai calendar boundaries
    error: str = ""


@dataclass(frozen=True)
class ScheduleItem:
    source: str  # dida / local
    task_id: str
    project_id: str
    title: str
    at: datetime | None
    time_kind: str  # deadline / start / reminder / undated
    all_day: bool = False
    status: str = ""
    projected: bool = False


@dataclass(frozen=True)
class ScheduleSource:
    items: tuple[ScheduleItem, ...] = ()
    available: bool = True
    incomplete: bool = False
    problem: str = ""


_TIME = r"今天|今日|明天|明日|后天|本周|这周|本星期|这个星期|下周|下星期|未来[7七]天|接下来[7七]天|最近|近期|这几天"
_TOKENS = re.compile(
    rf"(?:{_TIME}|\d{{4}}-\d{{2}}-\d{{2}}|请|麻烦|劳驾|拜托|帮我|给我|帮忙|"
    r"查询|查一下|查查|查|查看|看看|看一下|列出|列一下|汇总|整理一下|告诉我|说说|一下|"
    r"我|你|的|都|是|还有|还|有什么|有哪些|有没有|什么|哪些|有多少|有|需要|要|该|"
    r"没做完|未完成|没完成|待完成|已过期|过期|逾期|全部|所有|"
    r"待办|任务|安排|日程|提醒|事情|事项|事|做|处理|清单|列表|汇总|"
    r"吗|呢|么|了|到|至|~|[\s，,。.!！?？：:、；;])+"
)
_OBJECT = re.compile(r"待办|任务|安排|日程|提醒|(?:要|该|需要).*?(?:做|处理)|没做完|没完成|事情|事")
_READ = re.compile(r"查|看看|看一下|列|汇总|整理一下|告诉我|说说|有什么|有哪些|有没有|有多少|什么|哪些")


def parse_schedule_query(content: str, now: datetime) -> ScheduleQuery | None:
    """A bounded vocabulary recognizes read intent without a classifier call."""
    text = content.strip()
    explicit = re.fullmatch(r"(?:安排查询|查安排)\s*[：:]\s*(.*)", text, re.S)
    body = explicit[1].strip() if explicit else text
    if len(text) > 300 or mask_quoted_text(body) != body:
        return ScheduleQuery(error="请单独说明要查的时间范围，例如“本周有什么安排”。") if explicit else None
    if not body or not _TOKENS.fullmatch(body):
        return ScheduleQuery(error="可查询今天、明天、本周、下周、未来7天、全部未完成或逾期事项。") if explicit else None
    has_time = re.search(_TIME + r"|\d{4}-\d{2}-\d{2}", body)
    if not explicit and not has_time and not re.search(r"待办|任务|安排|日程|提醒|要做|该做|没做完|没完成|逾期|过期", body):
        return None
    if not explicit and (not _OBJECT.search(body) or not (_READ.search(body) or has_time or re.search(r"全部|所有|逾期|过期|未完成|没做完|没完成", body))):
        return None
    if not explicit and not _READ.search(body) and re.search(r"提醒我|给我安排|帮我安排", body):
        return None
    if not explicit and not _READ.search(body):
        short = re.sub(r"[\s。！!？?]", "", body)
        shorthand = rf"(?:我的?)?(?:{_TIME}|\d{{4}}-\d{{2}}-\d{{2}})(?:的)?(?:(?:未完成|待完成|逾期|过期)(?:的)?)?(?:待办|任务|安排|日程|提醒|要做的(?:事|事情)?)(?:清单|列表|汇总)?"
        pending = r"(?:(?:全部|所有)(?:未完成|待完成)?|未完成|待完成|逾期|过期|已过期)(?:的)?(?:待办|任务|安排|提醒|事项|事情)(?:清单|列表|汇总)?"
        if not re.fullmatch(shorthand, short) and not re.fullmatch(pending, short):
            return None
    # Pure completion/delivery-state questions remain in their existing routes.
    if re.search(r"(?:提醒|任务).*(?:完成了吗|发送了吗)", body):
        return None
    midnight = datetime.combine(now.date(), time(), tzinfo=now.tzinfo)
    modes = set(re.findall(r"逾期|已过期|过期", body))
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", body)
    selected = set()
    mappings = {"今天": "today", "今日": "today", "明天": "tomorrow", "明日": "tomorrow", "后天": "later",
                "本周": "week", "这周": "week", "本星期": "week", "这个星期": "week", "下周": "nextweek", "下星期": "nextweek",
                "未来7天": "seven", "未来七天": "seven", "接下来7天": "seven", "接下来七天": "seven"}
    for word, mode in mappings.items():
        if word in body:
            selected.add(mode)
    if modes and (selected or dates):
        return ScheduleQuery(error="请分别查询“逾期事项”或具体时间范围，避免混淆截止日期。")
    if modes:
        return ScheduleQuery(mode="overdue", label="已到期事项")
    if len(selected) > 1 or (selected and dates):
        return ScheduleQuery(error="这条消息包含多个时间范围，请明确一个，例如“本周”或“未来7天”。")
    if dates:
        try:
            if len(dates) > 2 or (len(dates) == 2 and not re.search(r"\d{4}-\d{2}-\d{2}\s*(?:至|到|~)\s*\d{4}-\d{2}-\d{2}", body)):
                raise ValueError()
            start = datetime.combine(datetime.fromisoformat(dates[0]).date(), time(), tzinfo=now.tzinfo)
            end = datetime.combine(datetime.fromisoformat(dates[-1]).date(), time(), tzinfo=now.tzinfo) + timedelta(days=1)
            if not timedelta(days=1) <= end - start <= timedelta(days=31):
                raise ValueError()
            return ScheduleQuery(label="指定日期", start=start, end=end)
        except ValueError:
            return ScheduleQuery(error="日期请用有效的 YYYY-MM-DD；范围按先后顺序填写，最多31天。")
    if not selected and not has_time and re.search(r"未完成|没做完|没完成|全部|所有|待完成", body):
        return ScheduleQuery(mode="all", label="全部未完成待办与本地提醒状态")
    mode = next(iter(selected), "seven")
    monday = midnight - timedelta(days=midnight.weekday())
    start, days, label = {
        "today": (midnight, 1, "今天"), "tomorrow": (midnight + timedelta(days=1), 1, "明天"),
        "later": (midnight + timedelta(days=2), 1, "后天"), "week": (monday, 7, "本周"),
        "nextweek": (monday + timedelta(days=7), 7, "下周"), "seven": (midnight, 7, "未来7天（含今天）"),
    }[mode]
    return ScheduleQuery(label=label, start=start, end=start + timedelta(days=days))


def _date(value: Any, tz) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("invalid-date")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("date-without-timezone")
    return parsed.astimezone(tz)


def _clean(value: Any, limit=160) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid-text")
    text = " ".join(value.split())
    if not text:
        raise ValueError("empty-text")
    return text[:limit] + ("…" if len(text) > limit else "")


def _identifier(value: Any) -> str:
    # Never truncate or normalize IDs into accidental cross-task matches.
    if not isinstance(value, str) or not value or len(value) > 300 or value.strip() != value:
        raise ValueError("invalid-identity")
    return value


def _task_array(payload):
    """Accept documented containers only, never recursively treat projects/items as tasks."""
    for _ in range(4):
        if isinstance(payload, list):
            return payload, False
        if not isinstance(payload, dict):
            break
        if payload.get("success") is False or payload.get("error"):
            raise ValueError("provider-error")
        incomplete = any(payload.get(k) not in (None, False, "", 0) for k in ("nextCursor", "next_cursor", "hasMore", "has_more", "truncated"))
        if incomplete:
            raise ValueError("paginated-result")
        children = [payload[k] for k in ("result", "data", "tasks") if k in payload]
        if len(children) != 1:
            break
        child = children[0]
        if isinstance(child, list) and isinstance(payload.get("total"), int) and payload["total"] > len(child):
            raise ValueError("incomplete-result")
        payload = child
    raise ValueError("unrecognized-task-result")


def parse_dida_schedule(envelope: dict, tz, limit=2000) -> ScheduleSource:
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        return ScheduleSource(available=False, problem="滴答待办暂时未查成功")
    try:
        rows, _ = _task_array(envelope.get("result"))
    except (ValueError, TypeError):
        return ScheduleSource(available=False, problem="滴答返回的列表完整性暂时无法确认")
    incomplete = len(rows) > limit
    items: dict[tuple[str, str], ScheduleItem] = {}
    conflicts = set()
    for row in rows[:limit]:
        try:
            if not isinstance(row, dict):
                raise ValueError()
            if row.get("completed") is True or row.get("completedTime") or str(row.get("status")).casefold() in {"2", "completed", "done"}:
                continue
            kind = str(row.get("kind") or "TASK").upper()
            if kind == "NOTE":
                continue
            # The live Dida API uses TEXT for ordinary tasks although the MCP
            # filter description calls that kind TASK. NOTE remains excluded.
            if kind not in {"TASK", "TEXT", "CHECKLIST"}:
                raise ValueError("unknown-task-kind")
            if row.get("status") != 0 or type(row.get("status")) is not int:
                raise ValueError("unknown-task-status")
            task_id = _identifier(row.get("id") or row.get("task_id"))
            project_id = _identifier(row.get("projectId") or row.get("project_id"))
            title = _clean(row.get("title"))
            due = _date(row.get("dueDate"), tz)
            start = _date(row.get("startDate"), tz)
            all_day = row.get("isAllDay")
            if (due or start) and type(all_day) is not bool:
                raise ValueError("unknown-all-day-state")
            item = ScheduleItem("dida", task_id, project_id, title, due or start,
                                "deadline" if due else "start" if start else "undated", bool(all_day), "undone")
            key = (project_id, task_id)
            if key in conflicts:
                continue
            if key in items and items[key] != item:
                conflicts.add(key)
                del items[key]
                raise ValueError("conflicting-task-record")
            items[key] = item
        except (ValueError, TypeError, OverflowError):
            incomplete = True
    return ScheduleSource(tuple(items.values()), incomplete=incomplete,
                          problem="滴答部分记录未能完整解析或超出读取上限" if incomplete else "")


def _in_range(item: ScheduleItem, query: ScheduleQuery, now: datetime) -> bool:
    if query.mode == "all":
        return item.source != "local" or item.status != "sent"
    if query.mode == "overdue":
        if item.at is None or item.time_kind in {"start", "undated"} or item.status == "sent":
            return False
        boundary = datetime.combine(item.at.date() + timedelta(days=1), time(), tzinfo=item.at.tzinfo) if item.all_day else item.at
        return boundary < now if not item.all_day else boundary <= now
    return item.at is not None and query.start <= item.at < query.end


def local_schedule(rows, query: ScheduleQuery, now: datetime, limit=2000) -> ScheduleSource:
    items = []
    incomplete = len(rows) > limit
    for row in rows[:limit]:
        try:
            status = row["status"]
            if status in {"cancelled", "rescheduled"}:
                continue
            if status not in {"pending", "failed", "delivering", "sending", "sent", "uncertain"}:
                raise ValueError("unknown-reminder-state")
            at = _date(row["reminder_at"], now.tzinfo)
            if at is None:
                raise ValueError()
            item = ScheduleItem("local", str(row["task_id"]), str(row["project_id"]), _clean(row["title"]), at, "reminder", status=status)
            if _in_range(item, query, now):
                items.append(item)
            # Daily queues store only the next occurrence. Future occurrences
            # are explicitly marked projections and never inserted into the queue.
            if (query.mode == "range" and row.get("recurrence_active") == 1
                    and row.get("recurrence_frequency") == "daily" and row.get("recurrence_interval") == 1
                    and status in {"pending", "failed", "delivering", "sending"}):
                candidate = at + timedelta(days=1)
                floor = max(query.start, now)
                if candidate < floor:
                    candidate += timedelta(days=max(0, (floor.date() - candidate.date()).days))
                    if candidate < floor:
                        candidate += timedelta(days=1)
                while candidate < query.end:
                    items.append(ScheduleItem("local", item.task_id, item.project_id, item.title, candidate, "reminder", status="pending", projected=True))
                    candidate += timedelta(days=1)
                    if len(items) > 2000:
                        raise ValueError("too-many-occurrences")
        except (ValueError, TypeError, KeyError, OverflowError):
            incomplete = True
    unique = {}
    for item in items[:2000]:
        key = (item.project_id, item.task_id, item.at)
        if key not in unique or not item.projected:
            unique[key] = item
    return ScheduleSource(tuple(unique.values()), incomplete=incomplete,
                          problem="本地部分提醒记录未能完整读取或超出上限" if incomplete else "")


def format_schedule(query: ScheduleQuery, now: datetime, local: ScheduleSource, dida: ScheduleSource, limit=12) -> str:
    heading = query.label
    if query.start:
        heading += f"｜{query.start:%Y-%m-%d} 至 {query.end - timedelta(days=1):%Y-%m-%d}"
    lines = [heading + "（上海时间）"]
    for source, result in (("本地提醒", local), ("滴答待办", dida)):
        if not result.available or result.incomplete:
            lines.append(f"{source}：{result.problem or '查询暂不可用'}，以下结果不完整。")
    remote = {(i.project_id, i.task_id): i for i in dida.items}
    groups = {}
    for item in dida.items:
        if _in_range(item, query, now):
            groups[("dida", item.project_id, item.task_id)] = [item]
    for item in local.items:
        key = (item.project_id, item.task_id)
        if key in remote:
            group = groups.setdefault(("dida", *key), [remote[key]])
            group.append(item)
        else:
            groups.setdefault(("local", *key), []).append(item)
    # Never merge titles alone: two different tasks may share the same title.
    def key(group):
        selected = [i.at for i in group if _in_range(i, query, now) and i.at]
        return min(selected) if selected else datetime.max.replace(tzinfo=now.tzinfo)
    ordered = sorted(groups.values(), key=lambda group: (key(group), group[0].title))
    if ordered:
        lines.append(f"{'当前已查到' if not local.available or not dida.available or local.incomplete or dida.incomplete else '共'} {len(ordered)} 项；同一任务与提醒按关联记录合并。")
    if not ordered:
        lines.append("在已成功查询的记录中，没有匹配这次范围的事项。" if not local.available or not dida.available or local.incomplete or dida.incomplete
                     else "已记录的本地提醒和滴答未完成待办中，没有匹配这次范围的事项。")
    labels = {"pending": "待提醒", "failed": "发送失败待重试", "delivering": "正在处理发送", "sending": "正在发送",
              "sent": "已发送", "uncertain": "送达未确认"}
    for group in ordered[:limit]:
        task = next((i for i in group if i.source == "dida"), None)
        lines.append(f"• {(task or group[0]).title}")
        if task:
            if task.at is None:
                detail = "未设日期"
            else:
                value = task.at.strftime("%m-%d" if task.all_day else "%m-%d %H:%M")
                detail = ("截止 " if task.time_kind == "deadline" else "开始 ") + value + ("（全天）" if task.all_day else "")
                if _in_range(task, ScheduleQuery(mode="overdue"), now):
                    detail += "｜已逾期"
            lines.append("  滴答：" + detail + "｜未完成")
        reminders = sorted((i for i in group if i.source == "local"), key=lambda i: i.at)
        for reminder in reminders[:4]:
            state = "按每日规则预计，届时仍需发送" if reminder.projected else labels[reminder.status]
            if not reminder.projected and reminder.status in {"pending", "failed"} and reminder.at < now:
                state += "（已到提醒时间）"
            lines.append(f"  微信提醒 {reminder.at:%m-%d %H:%M}｜{state}")
        if len(reminders) > 4:
            lines.append(f"  同一事项另有 {len(reminders) - 4} 次提醒未展开")
    if len(ordered) > limit:
        lines.append(f"另有 {len(ordered) - limit} 项未展开，可缩小到今天或明天查询。")
    undated = sum(i.at is None and ("dida", i.project_id, i.task_id) not in groups for i in dida.items)
    if query.mode == "range" and undated:
        lines.append(f"另有 {undated} 项滴答未完成待办没有日期，未归入本时间段；可问“全部未完成待办”。")
    lines.append("提醒发送状态不代表事情已完成；仅查询已有记录，没有修改任务或提醒。")
    return "\n".join(lines)
