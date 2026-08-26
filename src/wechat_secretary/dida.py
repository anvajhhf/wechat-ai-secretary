from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime
from threading import Lock
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

_CONNECTION_BEFORE_SEND_MARKERS = (
    "unreachable",
    "not connected",
    "transport down",
    "transport is down",
    "failed initial connection",
    "parked",
    "connection refused",
    "econnrefused",
    "no route to host",
    "尚未连接",
    "未连接",
    "无法连接",
    "初始连接失败",
)
_UNCERTAIN_DELIVERY_MARKERS = (
    "timeout",
    "timed out",
    "time out",
    "deadline exceeded",
    "mcp call failed",
    "connection reset",
    "connection closed",
    "connection lost",
    "session terminated",
    "broken pipe",
    "unexpected eof",
    "end of file",
    "cancelled",
    "canceled",
    "interrupted",
)
_BUSINESS_REJECTION_MARKERS = (
    "rejected",
    "invalid",
    "validation",
    "bad request",
    "unauthorized",
    "forbidden",
    "permission denied",
    "not permitted",
    "not allowed",
    "authentication failed",
    "not found",
    "duplicate",
    "rate limit",
    "too many requests",
    "quota",
    "拒绝",
    "参数错误",
    "参数无效",
    "校验失败",
    "验证失败",
    "未授权",
    "无权限",
    "认证失败",
    "鉴权失败",
    "不存在",
    "重复",
    "限流",
    "配额",
)


def _diagnostic_text(value: Any) -> str:
    """Return bounded, case-folded diagnostic text for failure classification."""

    if isinstance(value, BaseException):
        parts: list[str] = []
        current: BaseException | None = value
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            parts.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        return " | ".join(parts).casefold()[:4000]
    if isinstance(value, str):
        return value.casefold()[:4000]
    try:
        return json.dumps(value, ensure_ascii=False, default=str).casefold()[:4000]
    except (TypeError, ValueError):
        return str(value).casefold()[:4000]


def _failure_kind(value: Any, *, raised: bool = False) -> str:
    """Classify whether a failed write was sent before the failure surfaced."""

    text = _diagnostic_text(value)
    # These messages are emitted by Hermes before it dispatches tools/call.
    # Check them before timeout because an initial connection failure may name
    # TimeoutError while still proving that no write request was sent.
    if any(marker in text for marker in _CONNECTION_BEFORE_SEND_MARKERS):
        return "connection_before_send"
    if isinstance(value, PermissionError):
        return "local_rejection"
    if isinstance(value, TimeoutError) or any(
        marker in text for marker in _UNCERTAIN_DELIVERY_MARKERS
    ):
        return "uncertain"
    if not raised and any(marker in text for marker in _BUSINESS_REJECTION_MARKERS):
        return "business_rejection"
    # An arbitrary exception does not establish whether the transport failed
    # before or after dispatch.  In contrast, after the transport markers above
    # have been excluded, Hermes' normal ok:false envelope represents a tool
    # response and is therefore an explicit business rejection.
    return "uncertain" if raised else "business_rejection"


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
    show_category: bool = True,
) -> str:
    if not refs:
        return empty_text
    lines = [heading]
    for index, ref in enumerate(refs[:limit], start=1):
        suffix = f"｜{ref.category or 'Inbox'}" if show_category else ""
        lines.append(f"{index}. {ref.title}{suffix}")
    if len(refs) > limit:
        lines.append(f"另有 {len(refs) - limit} 项未展开")
    return "\n".join(lines)


class DidaExecutor:
    def __init__(self, settings: SecretarySettings, caller: McpCaller | None = None):
        self.settings = settings
        self._caller = caller
        self._health_lock = Lock()
        self._health: dict[str, str] = {
            "status": "unknown",
            "summary": "尚无滴答调用记录",
            "updated_at": "",
        }

    def _record_health(self, status: str, summary: str) -> None:
        snapshot = {
            "status": status,
            "summary": summary,
            "updated_at": datetime.now(tz=self.settings.tz).isoformat(),
        }
        with self._health_lock:
            self._health = snapshot

    def health_summary(self) -> dict[str, str]:
        """Return the latest lightweight Dida connectivity/outcome snapshot."""

        with self._health_lock:
            return dict(self._health)

    def _call(self, tool: str, arguments: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
        if tool not in ALLOWED_TOOLS:
            raise PermissionError(f"滴答工具 {tool} 不在第一版允许列表中")
        if self._caller is None:
            self._record_health("connection_fault", "滴答 MCP 尚未连接")
            raise RuntimeError("滴答 MCP 尚未连接")
        try:
            envelope = self._caller(self.settings.dida_server, tool, arguments, timeout)
        except Exception as exc:
            kind = _failure_kind(exc, raised=True)
            if kind == "connection_before_send":
                self._record_health("connection_fault", "滴答连接故障，请求未发出")
            elif kind == "local_rejection":
                self._record_health("connection_fault", "滴答本地调用配置不可用")
            else:
                self._record_health("result_uncertain", "滴答调用结果不确定")
            raise
        if not isinstance(envelope, dict):
            self._record_health("result_uncertain", "滴答返回格式异常，调用结果不确定")
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
        if envelope.get("ok") is True:
            self._record_health("recent_success", "最近一次滴答调用成功")
        elif envelope.get("ok") is False:
            kind = _failure_kind(envelope.get("error"))
            if kind == "connection_before_send":
                self._record_health("connection_fault", "滴答连接故障，请求未发出")
            elif kind == "uncertain":
                self._record_health("result_uncertain", "滴答调用结果不确定")
            else:
                self._record_health("recent_success", "滴答已响应，但请求被拒绝")
        else:
            self._record_health("result_uncertain", "滴答返回缺少状态，调用结果不确定")
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
            kind = _failure_kind(exc, raised=True)
            if kind == "connection_before_send":
                return ActionResult(
                    action="task",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination=destination,
                    error="滴答连接不可用，创建请求未发出，任务未创建。",
                )
            if kind == "local_rejection":
                return ActionResult(
                    action="task",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination=destination,
                    error="滴答创建请求在本地被阻止，任务未创建。",
                )
            self._record_health("result_uncertain", "滴答创建结果不确定")
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                error="无法确认滴答是否已创建该任务。请不要重试，以免重复创建。",
            )
        if envelope.get("ok") is not True:
            kind = _failure_kind(envelope.get("error"))
            if kind == "connection_before_send":
                return ActionResult(
                    action="task",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination=destination,
                    error="滴答连接不可用，创建请求未发出，任务未创建。",
                )
            if kind == "uncertain":
                self._record_health("result_uncertain", "滴答创建结果不确定")
                return ActionResult(
                    action="task",
                    status=ExecutionStatus.UNCERTAIN,
                    summary=task.title,
                    destination=destination,
                    error="无法确认滴答是否已创建该任务。请不要重试，以免重复创建。",
                )
            return ActionResult(
                action="task",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=destination,
                error="滴答明确拒绝创建请求，任务未创建。",
            )
        external_id = _extract_id(envelope.get("result"))
        if not external_id:
            self._record_health("result_uncertain", "滴答创建结果不确定：成功响应缺少任务 ID")
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                error=(
                    "滴答返回成功但没有 task_id，无法确认任务是否已创建。"
                    "请不要重试，以免重复创建。"
                ),
            )
        try:
            verification = self._call("get_task_by_id", {"task_id": external_id}, timeout=15)
            if verification.get("ok") is not True:
                self._record_health("result_uncertain", "滴答创建结果不确定：回读核验失败")
                return ActionResult(
                    action="task",
                    status=ExecutionStatus.UNCERTAIN,
                    summary=task.title,
                    destination=destination,
                    external_id=external_id,
                    error=(
                        "任务已返回 ID，但回读核验失败，无法确认最终状态。"
                        "请不要重试，以免重复创建。"
                    ),
                )
        except Exception:
            self._record_health("result_uncertain", "滴答创建结果不确定：回读核验中断")
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                external_id=external_id,
                error=(
                    "任务可能已创建，但回读核验中断。"
                    "请不要重试，以免重复创建。"
                ),
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
            self._record_health("result_uncertain", "滴答创建结果不确定：回读内容不匹配")
            return ActionResult(
                action="task",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=destination,
                external_id=external_id,
                error=(
                    "任务已返回 ID，但回读内容无法与该任务精确对应。"
                    "请不要重试，以免重复创建。"
                ),
            )
        reference = TaskReference(
            task_id=external_id,
            title=verified_title,
            category=str(verified_node.get("projectName") or destination),
            project_id=verified_project_id,
            status=str(verified_node.get("status") or ""),
        )
        self._record_health("recent_success", "最近一次滴答创建已成功回读确认")
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
            kind = _failure_kind(exc, raised=True)
            if kind == "connection_before_send":
                return ActionResult(
                    action="complete",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination=task.category or "Inbox",
                    external_id=task.task_id,
                    error="滴答连接不可用，完成请求未发出，任务未改动。",
                )
            if kind == "local_rejection":
                return ActionResult(
                    action="complete",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination=task.category or "Inbox",
                    external_id=task.task_id,
                    error="滴答完成请求在本地被阻止，任务未改动。",
                )
            self._record_health("result_uncertain", "滴答完成结果不确定")
            return ActionResult(
                action="complete",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="无法确认滴答任务是否已完成。请不要重试，以免重复操作。",
            )
        if envelope.get("ok") is not True:
            kind = _failure_kind(envelope.get("error"))
            if kind == "connection_before_send":
                return ActionResult(
                    action="complete",
                    status=ExecutionStatus.FAILED,
                    summary=task.title,
                    destination=task.category or "Inbox",
                    external_id=task.task_id,
                    error="滴答连接不可用，完成请求未发出，任务未改动。",
                )
            if kind == "uncertain":
                self._record_health("result_uncertain", "滴答完成结果不确定")
                return ActionResult(
                    action="complete",
                    status=ExecutionStatus.UNCERTAIN,
                    summary=task.title,
                    destination=task.category or "Inbox",
                    external_id=task.task_id,
                    error="无法确认滴答任务是否已完成。请不要重试，以免重复操作。",
                )
            return ActionResult(
                action="complete",
                status=ExecutionStatus.FAILED,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error="滴答明确拒绝完成请求，任务未改动。",
            )
        try:
            verification = self._call(
                "get_task_by_id", {"task_id": task.task_id}, timeout=15
            )
        except Exception:
            self._record_health("result_uncertain", "滴答完成结果不确定：回读核验中断")
            return ActionResult(
                action="complete",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error=(
                    "完成调用已返回，但回读核验中断。"
                    "请不要重试，以免重复操作。"
                ),
            )
        if verification.get("ok") is not True or not _is_exact_completed_task(
            verification.get("result"), task.task_id, task.project_id
        ):
            self._record_health("result_uncertain", "滴答完成结果不确定：回读未确认完成")
            return ActionResult(
                action="complete",
                status=ExecutionStatus.UNCERTAIN,
                summary=task.title,
                destination=task.category or "Inbox",
                external_id=task.task_id,
                error=(
                    "回读未确认任务为已完成状态。"
                    "请不要重试，以免重复操作。"
                ),
            )
        completed_ref = TaskReference(
            task_id=task.task_id,
            title=task.title,
            category=task.category,
            project_id=task.project_id,
            status="completed",
        )
        self._record_health("recent_success", "最近一次滴答完成操作已成功回读确认")
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
                    show_category=False,
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
