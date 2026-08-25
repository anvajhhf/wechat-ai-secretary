from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime
from typing import Any, Callable

from .config import SecretarySettings
from .models import (
    ActionResult,
    ExecutionStatus,
    MessageEnvelope,
    TaskDraft,
    TaskQuery,
    TaskReference,
)


McpCaller = Callable[[str, str, dict[str, Any], float], dict[str, Any]]


READ_TOOLS = frozenset(
    {
        "list_projects",
        "list_project_groups",
        "list_columns",
        "list_tags",
        "search_task",
        "get_task_by_id",
        "get_project_by_id",
        "get_project_with_undone_tasks",
        "get_task_in_project",
        "list_undone_tasks_by_time_query",
        "list_undone_tasks_by_date",
        "list_completed_tasks_by_date",
        "filter_tasks",
    }
)
WRITE_TOOLS = frozenset({"create_task", "complete_task"})
ALLOWED_TOOLS = READ_TOOLS | WRITE_TOOLS
PRIORITY_VALUES = {"low": 1, "medium": 3, "high": 5}
CREATE_APPROVAL_ENV = "SECRETARY_DIDA_CREATES_APPROVED"
COMPLETE_APPROVAL_ENV = "SECRETARY_DIDA_COMPLETIONS_APPROVED"


def _operator_approved(env_name: str) -> bool:
    return os.getenv(env_name, "").strip() == "1"


def _extract_id(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("id", "taskId", "task_id"):
            if value.get(key):
                return str(value[key])
        for child in value.values():
            found = _extract_id(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _extract_id(child)
            if found:
                return found
    return ""


def _extract_task_references(value: Any) -> tuple[TaskReference, ...]:
    found: dict[str, TaskReference] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            task_id = next(
                (str(node[key]) for key in ("task_id", "taskId", "id") if node.get(key)),
                "",
            )
            title = next(
                (
                    str(node[key]).strip()
                    for key in ("title", "taskTitle", "name")
                    if node.get(key)
                ),
                "",
            )
            if task_id and title:
                project_id = next(
                    (
                        str(node[key])
                        for key in ("project_id", "projectId")
                        if node.get(key)
                    ),
                    "",
                )
                category = str(node.get("projectName") or node.get("category") or "")
                status = str(node.get("status") or "")
                found.setdefault(
                    task_id,
                    TaskReference(task_id, title, category, project_id, status),
                )
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return tuple(found.values())


def _find_exact_task_node(value: Any, task_id: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        node_task_id = next(
            (str(value[key]) for key in ("task_id", "taskId", "id") if value.get(key)),
            "",
        )
        if node_task_id == task_id:
            return value
        for child in value.values():
            found = _find_exact_task_node(child, task_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_exact_task_node(child, task_id)
            if found is not None:
                return found
    return None


def _node_is_completed(value: dict[str, Any]) -> bool:
    """Accept completion evidence only from the task object itself."""

    if value.get("completed") is True:
        return True
    if value.get("completedTime") or value.get("completed_at"):
        return True
    status = str(value.get("status") or "").casefold()
    return status in {"2", "completed", "complete", "done"}


def _is_exact_completed_task(value: Any, task_id: str, project_id: str) -> bool:
    """Match the requested task and project before accepting completed state."""

    if isinstance(value, dict):
        node_task_id = next(
            (str(value[key]) for key in ("task_id", "taskId", "id") if value.get(key)),
            "",
        )
        if node_task_id == task_id:
            node_project_id = next(
                (
                    str(value[key])
                    for key in ("project_id", "projectId")
                    if value.get(key)
                ),
                "",
            )
            return node_project_id == project_id and _node_is_completed(value)
        return any(
            _is_exact_completed_task(child, task_id, project_id)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(
            _is_exact_completed_task(child, task_id, project_id) for child in value
        )
    return False


def _format_task_list(
    heading: str,
    refs: tuple[TaskReference, ...],
    *,
    empty_text: str,
    limit: int = 8,
) -> str:
    if not refs:
        return empty_text
    lines = [heading]
    for index, ref in enumerate(refs[:limit], start=1):
        lines.append(f"{index}. {ref.title}｜{ref.category or 'Inbox'}")
    if len(refs) > limit:
        lines.append(f"另有 {len(refs) - limit} 项未展开")
    return "\n".join(lines)


class DidaExecutor:
    def __init__(self, settings: SecretarySettings, caller: McpCaller | None = None):
        self.settings = settings
        self._caller = caller

    def _call(self, tool: str, arguments: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
        if tool not in ALLOWED_TOOLS:
            raise PermissionError(f"滴答工具 {tool} 不在第一版允许列表中")
        if self._caller is None:
            raise RuntimeError("滴答 MCP 尚未连接")
        envelope = self._caller(self.settings.dida_server, tool, arguments, timeout)
        if not isinstance(envelope, dict):
            raise TypeError("滴答 MCP 返回的不是结果对象")
        # Hermes exposes human-readable content as ``result`` and the
        # machine payload separately when an MCP response contains both.
        # Identity and status checks must use the structured payload only.
        structured = envelope.get("structuredContent")
        if structured is None:
            structured = envelope.get("structured_content")
        if structured is not None:
            envelope = dict(envelope)
            envelope["result"] = structured
        return envelope

    def taxonomy(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for tool in ("list_projects", "list_project_groups", "list_tags"):
            output[tool] = self._call(tool, {})
        return output

    def _create_arguments(self, task: TaskDraft) -> tuple[dict[str, Any], str]:
        project_id = (
            self.settings.category_map.get(task.category, "")
            if task.category
            else ""
        ) or "inbox"
        destination = task.category if project_id != "inbox" else "Inbox"
        payload: dict[str, Any] = {
            "title": task.title,
            "projectId": project_id,
        }
        if task.description:
            payload["content"] = task.description
            payload["kind"] = "TEXT"
        if task.due_date:
            clock = task.due_time or "00:00"
            local_due = datetime.fromisoformat(
                f"{task.due_date}T{clock}:00"
            ).replace(tzinfo=self.settings.tz)
            payload.update(
                {
                    "dueDate": local_due.isoformat(),
                    "timeZone": self.settings.timezone_name,
                    "isAllDay": not bool(task.due_time),
                }
            )
        if task.priority in PRIORITY_VALUES:
            payload["priority"] = PRIORITY_VALUES[task.priority]
        mapped_tags = [
            self.settings.tag_map[tag]
            for tag in task.tags
            if tag in self.settings.tag_map
        ]
        if mapped_tags:
            payload["tags"] = mapped_tags
        return {"task": payload}, destination

    def create_task(self, task: TaskDraft, message: MessageEnvelope) -> ActionResult:
        arguments, category_label = self._create_arguments(task)
        destination = category_label
        if self.settings.dry_run:
            external_id = "dry-" + hashlib.sha256(
                f"{message.message_id}\0{task.title}".encode("utf-8")
            ).hexdigest()[:16]
            reference = TaskReference(
                task_id=external_id,
                title=task.title,
                category=destination,
                project_id=str(arguments["task"]["projectId"]),
            )
            return ActionResult(
                action="task",
                status=ExecutionStatus.PLANNED,
                summary=task.title,
                destination=destination,
                external_id=external_id,
                preview=json.dumps(arguments, ensure_ascii=False, indent=2),
                task_refs=(reference,),
            )
        if not self.settings.dida_mapping_confirmed:
            return ActionResult(
                action="task",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=destination,
                error="滴答清单分类映射尚未确认",
            )
        if not self.settings.dida_schema_confirmed:
            return ActionResult(
                action="task",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=destination,
                error="滴答 create_task 尚未通过专用测试任务创建并回读确认",
            )
        if not _operator_approved(CREATE_APPROVAL_ENV):
            return ActionResult(
                action="task",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=destination,
                error="本次前台启动未确认允许创建滴答任务",
            )
        try:
            envelope = self._call("create_task", arguments, timeout=30)
        except Exception as exc:
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                error=f"滴答调用中断，未自动重试：{type(exc).__name__}",
            )
        if envelope.get("ok") is not True:
            return ActionResult(
                action="task",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=destination,
                error="滴答明确返回失败",
            )
        external_id = _extract_id(envelope.get("result"))
        if not external_id:
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                error="滴答返回成功但没有 task_id，未记录为已创建",
            )
        try:
            verification = self._call("get_task_by_id", {"task_id": external_id}, timeout=15)
            if verification.get("ok") is not True:
                return ActionResult(
                    action="task",
                    status=ExecutionStatus.UNCERTAIN,
                    summary=task.title,
                    destination=destination,
                    external_id=external_id,
                    error="任务已返回 ID，但回读核验失败",
                )
        except Exception:
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                external_id=external_id,
                error="任务可能已创建，但回读核验中断",
            )
        verified_node = _find_exact_task_node(verification.get("result"), external_id)
        expected_project_id = str(arguments["task"]["projectId"])
        verified_title = str((verified_node or {}).get("title") or "").strip()
        verified_project_id = str(
            (verified_node or {}).get("projectId")
            or (verified_node or {}).get("project_id")
            or ""
        )
        project_matches = verified_project_id == expected_project_id or (
            expected_project_id == "inbox" and bool(verified_project_id)
        )
        if (
            verified_node is None
            or verified_title.casefold() != task.title.strip().casefold()
            or not project_matches
            or _node_is_completed(verified_node)
        ):
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                external_id=external_id,
                error="任务已返回 ID，但回读内容无法与该任务精确对应",
            )
        reference = TaskReference(
            task_id=external_id,
            title=verified_title,
            category=str(verified_node.get("projectName") or destination),
            project_id=verified_project_id,
            status=str(verified_node.get("status") or ""),
        )
        return ActionResult(
            action="task",
            status=ExecutionStatus.SUCCEEDED,
            summary=task.title,
            destination=destination,
            external_id=external_id,
            task_refs=(reference,),
        )

    def query_tasks(self, query: TaskQuery) -> ActionResult:
        if self.settings.dry_run:
            label = query.keyword if query.mode == "search" else query.mode
            return ActionResult(
                action="query",
                status=ExecutionStatus.PLANNED,
                summary=f"查询任务：{label}",
                destination="滴答清单",
            )
        try:
            if query.mode == "search":
                envelope = self._call("search_task", {"query": query.keyword})
            else:
                envelope = self._call(
                    "list_undone_tasks_by_time_query",
                    {"time_query": query.mode},
                )
        except Exception as exc:
            return ActionResult(
                action="query",
                status=ExecutionStatus.FAILED,
                summary="查询任务",
                error=f"滴答查询失败：{type(exc).__name__}",
            )
        if envelope.get("ok") is not True:
            return ActionResult(
                action="query",
                status=ExecutionStatus.FAILED,
                summary="查询任务",
                error="滴答明确返回查询失败",
            )
        refs = _extract_task_references(envelope.get("result"))
        label = query.keyword if query.mode == "search" else query.mode
        rendered = _format_task_list(
            f"查询结果（{label}）：",
            refs,
            empty_text="没有找到匹配的未完成任务。",
        )
        return ActionResult(
            action="query",
            status=ExecutionStatus.SUCCEEDED,
            summary=rendered[:1200],
            destination="滴答清单",
            task_refs=refs,
        )

    def search_task_references(self, title: str) -> tuple[TaskReference, ...]:
        if self.settings.dry_run:
            return ()
        envelope = self._call("search_task", {"query": title}, timeout=20)
        if not envelope.get("ok"):
            return ()
        return _extract_task_references(envelope.get("result"))

    def exact_active_task_references(self, title: str) -> tuple[TaskReference, ...]:
        """Read back exact active tasks before a local-only reminder binding."""

        if self.settings.dry_run:
            return ()
        wanted = title.strip().casefold()
        if not wanted:
            return ()
        search = self._call("search_task", {"query": title}, timeout=20)
        if search.get("ok") is not True:
            raise RuntimeError("dida search failed")
        exact = tuple(
            ref
            for ref in _extract_task_references(search.get("result"))
            if ref.title.strip().casefold() == wanted
        )
        verified: list[TaskReference] = []
        for ref in exact:
            detail = self._call(
                "get_task_by_id", {"task_id": ref.task_id}, timeout=15
            )
            if detail.get("ok") is not True:
                raise RuntimeError("dida task readback failed")
            node = _find_exact_task_node(detail.get("result"), ref.task_id)
            verified_title = str((node or {}).get("title") or "").strip()
            if node is None or verified_title.casefold() != wanted:
                raise RuntimeError("dida task readback mismatch")
            if _node_is_completed(node):
                continue
            verified.append(
                TaskReference(
                    task_id=ref.task_id,
                    title=verified_title,
                    category=str(
                        node.get("projectName")
                        or ref.category
                        or "Inbox"
                    ),
                    project_id=str(
                        node.get("projectId")
                        or node.get("project_id")
                        or ref.project_id
                        or ""
                    ),
                    status=str(node.get("status") or ref.status or ""),
                )
            )
        return tuple(verified)

    def complete_task(self, task: TaskReference) -> ActionResult:
        arguments: dict[str, Any] = {
            "project_id": task.project_id,
            "task_id": task.task_id,
        }
        if self.settings.dry_run:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.PLANNED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                preview=json.dumps(arguments, ensure_ascii=False, indent=2),
                task_refs=(task,),
            )
        if not task.task_id:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                error="缺少任务 ID，未执行完成操作",
            )
        if not task.project_id:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="缺少任务所属清单 ID，未执行完成操作",
            )
        if not self.settings.dida_complete_schema_confirmed:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="滴答 complete_task 参数结构尚未实测确认",
            )
        if not _operator_approved(COMPLETE_APPROVAL_ENV):
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="本次前台启动未单独确认允许完成滴答任务",
            )
        try:
            current = self._call(
                "get_task_by_id", {"task_id": task.task_id}, timeout=15
            )
        except Exception:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="完成前无法回读并核对目标任务，未执行完成操作",
            )
        current_node = _find_exact_task_node(current.get("result"), task.task_id)
        current_project_id = str(
            (current_node or {}).get("projectId")
            or (current_node or {}).get("project_id")
            or ""
        )
        current_title = str((current_node or {}).get("title") or "").strip()
        if (
            current.get("ok") is not True
            or current_project_id != task.project_id
            or current_title.casefold() != task.title.strip().casefold()
        ):
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="完成前无法精确核对任务与所属清单，未执行完成操作",
            )
        if current_node is not None and _node_is_completed(current_node):
            return ActionResult(
                action="complete",
                status=ExecutionStatus.SKIPPED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="任务已经是完成状态，本次未重复执行",
                task_refs=(task,),
            )
        try:
            envelope = self._call("complete_task", arguments, timeout=30)
        except Exception as exc:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error=f"完成调用中断，未自动重试：{type(exc).__name__}",
            )
        if envelope.get("ok") is not True:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="滴答明确返回完成失败",
            )
        try:
            verification = self._call(
                "get_task_by_id", {"task_id": task.task_id}, timeout=15
            )
        except Exception:
            return ActionResult(
                action="complete",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="完成调用已返回，但回读核验中断",
            )
        if verification.get("ok") is not True or not _is_exact_completed_task(
            verification.get("result"), task.task_id, task.project_id
        ):
            return ActionResult(
                action="complete",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="回读未确认任务为已完成状态",
            )
        completed_ref = TaskReference(
            task_id=task.task_id,
            title=task.title,
            category=task.category,
            project_id=task.project_id,
            status="completed",
        )
        return ActionResult(
            action="complete",
            status=ExecutionStatus.SUCCEEDED,
            summary=task.title,
            destination=task.category or "Inbox",
            external_id=task.task_id,
            task_refs=(completed_ref,),
        )

    def scheduled_digest(
        self, job_name: str, now: datetime
    ) -> tuple[str, tuple[TaskReference, ...]]:
        if self.settings.dry_run:
            if job_name == "morning":
                return "Dry Run｜仅模拟 08:00 今日重点，不会实际查询或发送。", ()
            return "Dry Run｜仅模拟 22:00 简短复盘，不会实际查询或发送。", ()
        if job_name == "morning":
            result = self.query_tasks(TaskQuery(mode="today"))
            text = (
                _format_task_list(
                    "今日重点：",
                    result.task_refs,
                    empty_text="今天暂无未完成任务。",
                    limit=5,
                )
                if result.successful
                else "抱歉，今日重点暂时没能生成：滴答查询没有成功。"
            )
            return text, result.task_refs
        today = now.astimezone(self.settings.tz).date().isoformat()
        try:
            completed = self._call(
                "filter_tasks",
                {"start_date": today, "end_date": today, "status": "completed"},
            )
            undone = self._call(
                "list_undone_tasks_by_date",
                {"start_date": today, "end_date": today},
            )
        except Exception:
            return "抱歉，晚间复盘暂时没能生成：滴答查询没有成功。", ()
        if not completed.get("ok") or not undone.get("ok"):
            return "抱歉，晚间复盘暂时没能生成：滴答查询没有成功。", ()
        completed_refs = _extract_task_references(completed.get("result"))
        undone_refs = _extract_task_references(undone.get("result"))
        lines = ["今日简短复盘：", f"已完成 {len(completed_refs)} 项"]
        lines.extend(f"- {ref.title}" for ref in completed_refs[:5])
        lines.append(f"仍待处理 {len(undone_refs)} 项")
        lines.extend(f"- {ref.title}" for ref in undone_refs[:5])
        return "\n".join(lines), undone_refs
