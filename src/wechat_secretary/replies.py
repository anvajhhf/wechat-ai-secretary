from __future__ import annotations

import json
from datetime import datetime

from .models import ActionResult, ExecutionStatus


_ERROR_LIMIT = 160
_INTERNAL_ERROR_MARKERS = (
    "authorization",
    "api_key",
    "api-key",
    "apikey",
    "bearer ",
    "secret",
    "token=",
    "sk-",
    "projectid",
    "timezone",
    "task_id",
    "duedate",
    "isallday",
    "{",
    "}",
)


def _success_line(result: ActionResult, dry_run: bool) -> str:
    if dry_run or result.status is ExecutionStatus.PLANNED:
        if result.action == "task":
            return f"任务：{result.summary}｜{result.destination or 'Inbox'}"
        if result.action == "note":
            return f"笔记：{result.destination}"
        if result.action == "private":
            return "私密内容：模拟本地保存"
        if result.action == "query":
            return result.summary
        if result.action == "complete":
            return f"完成任务：{result.summary}｜{result.destination or 'Inbox'}"
        if result.action == "reminder":
            return f"微信提醒：{result.summary}"
        return result.summary

    if result.action == "task":
        return f"已为你创建好任务：{result.summary}｜{result.destination or 'Inbox'}"
    if result.action == "note":
        saved = f"已为你妥善保存笔记：{result.destination}"
        details = _note_preview_lines(result.preview)
        return "\n".join((saved, *details)) if details else saved
    if result.action == "private":
        return "已为你在本地保存私密内容"
    if result.action == "query":
        return result.summary
    if result.action == "complete":
        return f"已为你完成任务：{result.summary}｜{result.destination or 'Inbox'}"
    if result.action == "reminder":
        return f"已为你设置好微信提醒：{result.summary}"
    return result.summary


def format_results(results: tuple[ActionResult, ...], dry_run: bool) -> str:
    lines: list[str] = []
    for result in results:
        if result.status in {ExecutionStatus.PLANNED, ExecutionStatus.SUCCEEDED}:
            line = _success_line(result, dry_run)
        elif result.status is ExecutionStatus.SKIPPED:
            line = f"这条已经处理过了，我没有重复执行：{result.summary}"
        elif result.status is ExecutionStatus.UNCERTAIN:
            line = (
                f"这次的结果还需要确认：{result.summary}｜{_safe_error(result.error)}。"
                "为了避免重复，我没有自动重试。"
            )
        else:
            line = f"抱歉，这次没能处理成功：{result.summary}｜{_safe_error(result.error)}"
        if line not in lines:
            lines.append(line)
    if dry_run and lines:
        return "Dry Run｜已为你整理好模拟结果\n" + "\n".join(lines)
    return "\n".join(lines)


def format_failure(reason: str | None) -> str:
    return f"抱歉，这次没能处理成功：{_safe_error(reason)}"


def add_dry_run_previews(
    reply: str,
    results: tuple[ActionResult, ...],
    *,
    max_chars: int = 1200,
) -> str:
    """Add short human previews; never send tool parameters or private payloads."""

    previewable = [
        result
        for result in results
        if result.status is ExecutionStatus.PLANNED
        and result.action in {"task", "note"}
        and result.preview
    ]
    details: list[str] = []
    for result in previewable:
        if result.action == "task":
            lines = _task_preview_lines(result.preview)
        else:
            lines = _note_preview_lines(result.preview)
        details.extend(lines)

    if not details:
        return reply

    rendered = "\n".join(details)
    if len(rendered) > max_chars:
        rendered = rendered[: max(0, max_chars - 8)].rstrip() + "…（已截断）"
    return reply + "\n" + rendered


def _task_preview_lines(preview: str) -> list[str]:
    """Turn the internal Dida payload into one user-facing Chinese line."""

    try:
        payload = json.loads(preview)
    except (TypeError, ValueError):
        return []
    task = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task, dict):
        return []

    parts: list[str] = []
    raw_due = task.get("dueDate")
    if isinstance(raw_due, str) and raw_due:
        try:
            due = datetime.fromisoformat(raw_due.replace("Z", "+00:00"))
            if task.get("isAllDay"):
                parts.append(f"日期：{due:%Y-%m-%d}")
            else:
                parts.append(f"时间：{due:%Y-%m-%d %H:%M}")
        except ValueError:
            # An invalid internal value must not leak into the Weixin reply.
            pass

    priority = {5: "高", 3: "中", 1: "低"}.get(task.get("priority"))
    if priority:
        parts.append(f"优先级：{priority}")
    return ["｜".join(parts)] if parts else []


def _note_preview_lines(preview: str) -> list[str]:
    """Extract compact note fields instead of echoing generated Markdown."""

    summary = ""
    tags = ""
    links = ""
    body_lines: list[str] = []
    for raw_line in preview.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("## "):
            continue
        if line.startswith("> 摘要："):
            summary = line.removeprefix("> 摘要：").strip()
        elif line.startswith("标签："):
            tags = line.removeprefix("标签：").strip()
        elif line.startswith("关联："):
            links = line.removeprefix("关联：").strip()
        elif line.startswith("> 来源：") or line.startswith("<!--"):
            continue
        else:
            body_lines.append(line)

    lines: list[str] = []
    if summary:
        lines.append(f"摘要：{_compact(summary, 120)}")
    elif body_lines:
        lines.append(f"内容：{_compact(' '.join(body_lines), 120)}")
    if tags:
        lines.append(f"标签：{_compact(tags, 100)}")
    if links:
        lines.append(f"关联：{_compact(links, 100)}")
    return lines


def _compact(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _safe_error(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "未知错误"
    lowered = raw.casefold()
    if any(marker in lowered for marker in _INTERNAL_ERROR_MARKERS):
        return "内部核验未通过，结果暂时无法确认"
    return _compact(raw, _ERROR_LIMIT)
