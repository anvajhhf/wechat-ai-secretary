from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .classifier import HeuristicClassifier
from .config import SecretarySettings, load_settings
from .dida import (
    DidaExecutor,
    _extract_id,
    _extract_task_references,
    _find_exact_task_node,
    _is_exact_completed_task,
    _node_is_completed,
)
from .ledger import IdempotencyLedger
from .models import MessageEnvelope, TaskDraft
from .obsidian import ObsidianExecutor
from .private_inbox import PrivateInboxExecutor
from .service import SecretaryService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREATE_TEST_APPROVAL_ENV = "SECRETARY_DIDA_CREATE_TEST_APPROVED"
COMPLETE_TEST_APPROVAL_ENV = "SECRETARY_DIDA_COMPLETION_TEST_APPROVED"


def _dry_run_settings() -> SecretarySettings:
    return SecretarySettings(
        project_root=PROJECT_ROOT,
        profile_id="dry-run",
        dry_run=True,
        allowed_users=frozenset({"wx-user-1"}),
        account_id="dry-run-account",
        known_links=("年度目标", "AI个人秘书", "产品路线图"),
        category_map={"工作": "project-work", "个人": "project-personal"},
        tag_map={"重要": "重要"},
    )


def _build_dry_service() -> tuple[SecretaryService, HeuristicClassifier]:
    settings = _dry_run_settings()
    classifier = HeuristicClassifier(settings)
    service = SecretaryService(
        settings=settings,
        ledger=IdempotencyLedger(":memory:"),
        classifier=classifier,
        dida=DidaExecutor(settings),
        obsidian=ObsidianExecutor(settings),
        private_inbox=PrivateInboxExecutor(settings),
    )
    return service, classifier


def command_dry_run(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixtures).resolve()
    entries = json.loads(fixture_path.read_text(encoding="utf-8"))
    service, classifier = _build_dry_service()
    for index, item in enumerate(entries, start=1):
        timestamp = datetime.fromisoformat(item["received_at"])
        message = MessageEnvelope(
            platform="weixin",
            account_id="dry-run-account",
            user_id=item.get("user_id", "wx-user-1"),
            chat_id="dry-run-chat",
            chat_type=item.get("chat_type", "dm"),
            message_id=item["message_id"],
            text=item.get("text", ""),
            received_at=timestamp,
            media_paths=tuple(item.get("media_paths", [])),
            media_types=tuple(item.get("media_types", [])),
        )
        result = service.handle(message)
        print(f"[{index}] {item.get('name', message.message_id)}")
        print(f"输入：{item.get('display_text', message.text)}")
        print(result.reply or "（已按策略静默忽略）")
        for action in result.results:
            if action.preview:
                preview = action.preview.replace("\n", "\n    ")
                print(f"  预览[{action.action}]：\n    {preview}")
        print()
    print(f"非私密分类调用次数：{classifier.call_count}")
    print("Dry Run 完成：未连接微信、DeepSeek、滴答或真实 Vault。")
    return 0


def _safe_path_state(path: Path | None) -> str:
    if path is None:
        return "未配置"
    return f"已配置，{'存在' if path.exists() else '不存在'}"


def command_doctor(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(PROJECT_ROOT)
        load_error = ""
    except Exception as exc:
        settings = None
        load_error = type(exc).__name__
    print("微信 AI 个人秘书状态检查")
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"配置文件：{'可读取' if settings else '不可读取（' + load_error + '）'}")
    print(f"DeepSeek Key：{'已在环境中设置' if os.getenv('DEEPSEEK_API_KEY') else '未设置'}")
    local_hermes = PROJECT_ROOT / ".venv" / "Scripts" / "hermes.exe"
    hermes_found = bool(shutil.which("hermes") or local_hermes.is_file())
    print(f"Hermes 命令：{'已找到' if hermes_found else '未找到'}")
    plugin_entry = (
        PROJECT_ROOT
        / ".hermes"
        / "plugins"
        / "wechat-secretary"
        / "plugin.yaml"
    ).is_file()
    print(f"项目插件入口：{'已找到' if plugin_entry else '未找到'}")
    if settings is None:
        return 1
    print(f"运行档案：{settings.profile_id}")
    print(f"运行模式：{'Dry Run' if settings.dry_run else '真实写入'}")
    print(f"时区：{settings.timezone_name}")
    print(f"微信 allowlist：{len(settings.allowed_users)} 个账号")
    print(f"Obsidian Vault：{_safe_path_state(settings.vault_path)}")
    print(f"私密收件箱：{_safe_path_state(settings.private_inbox_path)}")
    print(f"滴答分类映射：{'已确认' if settings.dida_mapping_confirmed else '未确认'}")
    print(f"滴答创建结构：{'已确认' if settings.dida_schema_confirmed else '未确认'}")
    print(
        f"滴答完成结构：{'已确认' if settings.dida_complete_schema_confirmed else '未确认'}"
    )
    print(f"Obsidian 映射：{'已确认' if settings.obsidian_mapping_confirmed else '未确认'}")
    print(f"本地微信提醒：{'已启用' if settings.reminders_enabled else '未启用'}")
    print(f"公开链接笔记：{'已启用' if settings.web_enabled else '未启用'}")
    pillow_found = importlib.util.find_spec("PIL") is not None
    whisper_found = importlib.util.find_spec("faster_whisper") is not None
    silk_found = importlib.util.find_spec("pysilk") is not None
    print(
        f"图片安全预处理：{'已就绪' if pillow_found and settings.vision_enabled else '未就绪'}"
    )
    print(
        f"本地语音转写：{'已就绪' if whisper_found and silk_found and settings.voice_asr_enabled else '未就绪'}"
    )
    model_home_value = os.getenv("HF_HOME", "").strip()
    whisper_cache_ready = False
    if model_home_value:
        snapshots = (
            Path(model_home_value)
            / "hub"
            / f"models--Systran--faster-whisper-{settings.asr_model}"
            / "snapshots"
        )
        if snapshots.is_dir():
            for snapshot in snapshots.iterdir():
                model_file = snapshot / "model.bin"
                if (
                    snapshot.is_dir()
                    and (snapshot / "config.json").is_file()
                    and (snapshot / "tokenizer.json").is_file()
                    and model_file.is_file()
                    and model_file.stat().st_size > 10 * 1024 * 1024
                ):
                    whisper_cache_ready = True
                    break
    print(
        f"Whisper {settings.asr_model} 模型："
        f"{'已完整缓存' if whisper_cache_ready else '尚未完整缓存'}"
    )
    errors = settings.runtime_errors(
        strict=args.strict,
        require_write_approval=args.strict,
    )
    if args.strict and not os.getenv("DEEPSEEK_API_KEY"):
        errors.append("DeepSeek Key 未配置；请仅在本机授权向导中设置")
    if args.strict and not hermes_found:
        errors.append("项目隔离环境中未找到 Hermes")
    if args.strict and not plugin_entry:
        errors.append("Hermes 项目插件入口缺失")
    if args.strict and settings.vision_enabled and not pillow_found:
        errors.append("图片安全预处理依赖 Pillow 未安装")
    if args.strict and settings.voice_asr_enabled and not whisper_found:
        errors.append("本地语音依赖 faster-whisper 未安装")
    if args.strict and settings.voice_asr_enabled and not silk_found:
        errors.append("微信 .silk 解码依赖 silk-python 未安装")
    if args.strict and settings.voice_asr_enabled and not whisper_cache_ready:
        errors.append(f"Whisper {settings.asr_model} 模型尚未完整缓存")
    for error in errors:
        print(f"阻止项：{error}")
    return 1 if errors else 0


def command_inspect_vault(args: argparse.Namespace) -> int:
    settings = load_settings(PROJECT_ROOT)
    executor = ObsidianExecutor(settings)
    rows = executor.structure_summary(max_depth=2, max_entries=args.limit)
    if not rows:
        print("Vault 未配置、路径不存在或没有可展示的目录。")
        return 1
    print("Vault 结构（最多两层，只统计 Markdown 文件数量）：")
    for row in rows:
        print(f"- {row}")
    return 0


def _plain_mcp_result(result: object) -> dict[str, object]:
    is_error = bool(
        getattr(result, "is_error", getattr(result, "isError", False))
    )
    structured = getattr(
        result,
        "structured_content",
        getattr(result, "structuredContent", None),
    )
    if structured is not None:
        value: object = structured
    else:
        text_value = "\n".join(
            str(block.text)
            for block in (getattr(result, "content", None) or [])
            if getattr(block, "text", None)
        )
        try:
            value = json.loads(text_value)
        except (json.JSONDecodeError, TypeError):
            value = text_value
    return {"ok": not is_error, "result": value}


def command_inspect_dida(args: argparse.Namespace) -> int:
    """Read the three taxonomy endpoints without starting an agent chat."""
    del args
    settings = load_settings(PROJECT_ROOT)
    from hermes_cli.mcp_config import _get_mcp_servers, _resolve_mcp_server_config
    from tools.mcp_tool import (
        _connect_server,
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _stop_mcp_loop_if_idle,
    )

    servers = _get_mcp_servers()
    if settings.dida_server not in servers:
        print("滴答 MCP 未配置。")
        return 1
    config = _resolve_mcp_server_config(servers[settings.dida_server])
    selected_names = ("list_projects", "list_project_groups", "list_tags")

    async def inspect() -> dict[str, object]:
        server = await asyncio.wait_for(
            _connect_server(settings.dida_server, config), timeout=30
        )
        try:
            output: dict[str, object] = {}
            for tool_name in selected_names:
                result = await asyncio.wait_for(
                    server.session.call_tool(tool_name, arguments={}), timeout=30
                )
                output[tool_name] = _plain_mcp_result(result)
            return output
        finally:
            await server.shutdown()

    _ensure_mcp_loop()
    try:
        taxonomy = _run_on_mcp_loop(inspect(), timeout=110)
    finally:
        _stop_mcp_loop_if_idle()
    print(json.dumps(taxonomy, ensure_ascii=False, indent=2, default=str))
    return 0 if all(item.get("ok") for item in taxonomy.values()) else 1


def command_inspect_dida_schema(args: argparse.Namespace) -> int:
    """Discover selected Dida schemas without invoking any Dida tool."""
    del args
    settings = load_settings(PROJECT_ROOT)
    from hermes_cli.mcp_config import _get_mcp_servers, _resolve_mcp_server_config
    from tools.mcp_tool import (
        _connect_server,
        _convert_mcp_schema,
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _stop_mcp_loop_if_idle,
    )

    servers = _get_mcp_servers()
    if settings.dida_server not in servers:
        print("滴答 MCP 未配置。")
        return 1
    config = _resolve_mcp_server_config(servers[settings.dida_server])
    selected_names = {
        "create_task",
        "complete_task",
        "get_task_by_id",
        "search_task",
    }

    async def probe() -> dict[str, object]:
        server = await asyncio.wait_for(
            _connect_server(settings.dida_server, config), timeout=30
        )
        try:
            return {
                tool.name: _convert_mcp_schema(settings.dida_server, tool)
                for tool in server._tools
                if tool.name in selected_names
            }
        finally:
            await server.shutdown()

    _ensure_mcp_loop()
    try:
        schemas = _run_on_mcp_loop(probe(), timeout=40)
    finally:
        _stop_mcp_loop_if_idle()
    print(json.dumps(schemas, ensure_ascii=False, indent=2, default=str))
    errors = _dida_schema_errors(schemas)
    if errors:
        for error in errors:
            print(f"结构核验失败：{error}")
        return 1
    print("结构核验通过：仅完成工具发现，未调用任何任务工具。")
    return 0


def _dida_schema_errors(schemas: object) -> tuple[str, ...]:
    """Validate the live tool schemas required by the local Dida adapter."""

    if not isinstance(schemas, dict):
        return ("工具定义不是对象",)
    expected: dict[str, tuple[set[str], set[str]]] = {
        "create_task": ({"task"}, {"task"}),
        "complete_task": ({"project_id", "task_id"}, {"project_id", "task_id"}),
        "get_task_by_id": ({"task_id"}, {"task_id"}),
        "search_task": ({"query"}, {"query"}),
    }
    errors: list[str] = []
    for tool_name, (required_fields, property_fields) in expected.items():
        schema = schemas.get(tool_name)
        parameters = schema.get("parameters") if isinstance(schema, dict) else None
        if not isinstance(parameters, dict) or not parameters:
            errors.append(f"{tool_name} 缺少非空 parameters")
            continue
        required = set(parameters.get("required") or ())
        properties = parameters.get("properties")
        property_names = set(properties) if isinstance(properties, dict) else set()
        if not required_fields.issubset(required):
            errors.append(f"{tool_name} 缺少必填字段 {sorted(required_fields - required)}")
        if not property_fields.issubset(property_names):
            errors.append(f"{tool_name} 缺少参数字段 {sorted(property_fields - property_names)}")
    create_schema = schemas.get("create_task")
    create_parameters = (
        create_schema.get("parameters") if isinstance(create_schema, dict) else None
    )
    definitions = (
        create_parameters.get("$defs") if isinstance(create_parameters, dict) else None
    )
    open_task = definitions.get("OpenTask") if isinstance(definitions, dict) else None
    task_properties = open_task.get("properties") if isinstance(open_task, dict) else None
    expected_task_fields = {
        "title",
        "projectId",
        "content",
        "kind",
        "dueDate",
        "timeZone",
        "isAllDay",
        "priority",
        "tags",
    }
    actual_task_fields = set(task_properties) if isinstance(task_properties, dict) else set()
    if not expected_task_fields.issubset(actual_task_fields):
        errors.append(
            "create_task.OpenTask 缺少字段 "
            f"{sorted(expected_task_fields - actual_task_fields)}"
        )
    return tuple(errors)


def _dida_contract_state_path(settings: SecretarySettings) -> Path:
    home_text = os.getenv("HERMES_HOME", "").strip()
    if not home_text:
        raise RuntimeError("HERMES_HOME 未设置")
    home = Path(home_text).resolve(strict=True)
    project = settings.project_root.resolve(strict=True)
    if not home.is_relative_to(project):
        raise RuntimeError("HERMES_HOME 不在本项目目录内")
    return home / "state" / "dida-contract-test.json"


def _save_dida_contract_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(state, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    # Windows' fsync requires a writable descriptor even when no bytes change.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _run_contract_probe_with_lock(
    state_path: Path, operation: str, callback: Callable[[], int]
) -> int:
    lock_path = state_path.with_name(f"{state_path.stem}.{operation}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        print("已拒绝：本档案已有同类核验在进行或曾中断；不会重复写入。")
        return 1
    completed_normally = False
    try:
        os.write(lock_fd, b"reserved\n")
        os.fsync(lock_fd)
        result = callback()
        completed_normally = True
        return int(result)
    finally:
        os.close(lock_fd)
        if completed_normally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _command_verify_dida_create_locked(args: argparse.Namespace) -> int:
    """Create exactly one dedicated task and verify it by ID without retrying."""

    if not args.confirm_create_test:
        print("已拒绝：缺少专用测试任务创建确认。")
        return 2
    settings = load_settings(PROJECT_ROOT)
    if not settings.dida_mapping_confirmed:
        print("已拒绝：滴答 Inbox 映射尚未确认。")
        return 1
    try:
        state_path = _dida_contract_state_path(settings)
    except Exception as exc:
        print(f"本地核验状态路径不可用：{type(exc).__name__}")
        return 1
    if state_path.exists():
        print("已拒绝：本档案已有创建核验记录，不会重复创建测试任务。")
        return 1

    from hermes_cli.mcp_config import _get_mcp_servers, _resolve_mcp_server_config
    from tools.mcp_tool import (
        _connect_server,
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _stop_mcp_loop_if_idle,
    )

    servers = _get_mcp_servers()
    if settings.dida_server not in servers:
        print("滴答 MCP 未配置；未发送创建请求。")
        return 1
    config = _resolve_mcp_server_config(servers[settings.dida_server])
    title = f"微信AI秘书结构核验（测试）{datetime.now(settings.tz):%Y%m%d-%H%M%S}"
    arguments, destination = DidaExecutor(settings)._create_arguments(TaskDraft(title))
    expected_project_id = str(arguments["task"]["projectId"])

    async def verify() -> dict[str, object]:
        server = await asyncio.wait_for(
            _connect_server(settings.dida_server, config), timeout=30
        )
        try:
            try:
                created_raw = await asyncio.wait_for(
                    server.session.call_tool("create_task", arguments=arguments),
                    timeout=30,
                )
            except Exception as exc:
                return {
                    "status": "create_uncertain",
                    "error_type": type(exc).__name__,
                }
            created = _plain_mcp_result(created_raw)
            task_id = _extract_id(created.get("result"))
            if created.get("ok") is not True or not task_id:
                return {
                    "status": "create_uncertain",
                    "task_id": task_id,
                    "error_type": "missing-confirmed-task-id",
                }
            try:
                read_raw = await asyncio.wait_for(
                    server.session.call_tool(
                        "get_task_by_id", arguments={"task_id": task_id}
                    ),
                    timeout=15,
                )
            except Exception as exc:
                return {
                    "status": "readback_uncertain",
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                }
            readback = _plain_mcp_result(read_raw)
            node = _find_exact_task_node(readback.get("result"), task_id)
            read_title = str((node or {}).get("title") or "").strip()
            read_project_id = str(
                (node or {}).get("projectId")
                or (node or {}).get("project_id")
                or ""
            )
            project_matches = read_project_id == expected_project_id or (
                expected_project_id == "inbox" and bool(read_project_id)
            )
            verified = (
                readback.get("ok") is True
                and node is not None
                and read_title.casefold() == title.casefold()
                and project_matches
                and not _node_is_completed(node)
            )
            return {
                "status": "created_verified" if verified else "readback_uncertain",
                "task_id": task_id,
                "project_id": read_project_id,
                "error_type": "" if verified else "readback-mismatch",
            }
        finally:
            await server.shutdown()

    _ensure_mcp_loop()
    try:
        try:
            outcome = _run_on_mcp_loop(verify(), timeout=85)
        except Exception as exc:
            print(f"连接滴答失败：{type(exc).__name__}；未自动重试。")
            return 1
    finally:
        _stop_mcp_loop_if_idle()

    state = {
        "profile_id": settings.profile_id,
        "title": title,
        "destination": destination,
        "project_id": str(outcome.get("project_id") or expected_project_id),
        "task_id": str(outcome.get("task_id") or ""),
        "status": str(outcome.get("status") or "unknown"),
        "observed_at": datetime.now(settings.tz).isoformat(timespec="seconds"),
        "error_type": str(outcome.get("error_type") or ""),
    }
    _save_dida_contract_state(state_path, state)
    if state["status"] != "created_verified":
        print("创建请求已发出，但结果未能精确核验；已记录状态且不会自动重试。")
        return 1
    print(f"创建并回读核验通过：{title}｜{destination}")
    return 0


def command_verify_dida_create(args: argparse.Namespace) -> int:
    if not args.confirm_create_test:
        print("已拒绝：缺少专用测试任务创建确认。")
        return 2
    if os.getenv(CREATE_TEST_APPROVAL_ENV, "").strip() != "1":
        print("已拒绝：当前进程没有专用创建核验授权。")
        return 2
    settings = load_settings(PROJECT_ROOT)
    try:
        state_path = _dida_contract_state_path(settings)
    except Exception as exc:
        print(f"本地核验状态路径不可用：{type(exc).__name__}")
        return 1
    return _run_contract_probe_with_lock(
        state_path,
        "create",
        lambda: _command_verify_dida_create_locked(args),
    )


def _command_verify_dida_complete_locked(args: argparse.Namespace) -> int:
    """Complete only the recorded contract-test task, once, with exact readbacks."""

    if not args.confirm_complete_test:
        print("已拒绝：缺少专用测试任务完成确认。")
        return 2
    settings = load_settings(PROJECT_ROOT)
    if not settings.dida_schema_confirmed:
        print("已拒绝：本档案尚未通过创建结构核验。")
        return 1
    try:
        state_path = _dida_contract_state_path(settings)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"无法读取本地核验状态：{type(exc).__name__}")
        return 1

    title = str(state.get("title") or "").strip()
    task_id = str(state.get("task_id") or "").strip()
    project_id = str(state.get("project_id") or "").strip()
    destination = str(state.get("destination") or "Inbox").strip() or "Inbox"
    if str(state.get("profile_id") or "") != settings.profile_id:
        print("已拒绝：核验记录不属于当前档案。")
        return 1
    if str(state.get("status") or "") != "created_verified":
        print("已拒绝：专用核验任务不是可完成状态，不会重复执行。")
        return 1
    if not title or not task_id or not project_id:
        print("已拒绝：本地核验记录不完整；未调用滴答。")
        return 1

    from hermes_cli.mcp_config import _get_mcp_servers, _resolve_mcp_server_config
    from tools.mcp_tool import (
        _connect_server,
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _stop_mcp_loop_if_idle,
    )

    servers = _get_mcp_servers()
    if settings.dida_server not in servers:
        print("滴答 MCP 未配置；未发送完成请求。")
        return 1
    config = _resolve_mcp_server_config(servers[settings.dida_server])

    def mark_pending() -> None:
        state["status"] = "completion_pending"
        state["completion_started_at"] = datetime.now(settings.tz).isoformat(
            timespec="seconds"
        )
        state["error_type"] = ""
        _save_dida_contract_state(state_path, state)

    async def verify() -> dict[str, str]:
        server = await asyncio.wait_for(
            _connect_server(settings.dida_server, config), timeout=30
        )
        try:
            try:
                before_raw = await asyncio.wait_for(
                    server.session.call_tool(
                        "get_task_by_id", arguments={"task_id": task_id}
                    ),
                    timeout=15,
                )
            except Exception as exc:
                return {"status": "precheck_failed", "error_type": type(exc).__name__}

            before = _plain_mcp_result(before_raw)
            node = _find_exact_task_node(before.get("result"), task_id)
            before_title = str((node or {}).get("title") or "").strip()
            before_project_id = str(
                (node or {}).get("projectId")
                or (node or {}).get("project_id")
                or ""
            )
            if (
                before.get("ok") is not True
                or node is None
                or before_title.casefold() != title.casefold()
                or before_project_id != project_id
            ):
                return {
                    "status": "precheck_failed",
                    "error_type": "precheck-mismatch",
                }
            if _node_is_completed(node):
                return {
                    "status": "already_completed",
                    "error_type": "already-completed-before-test",
                }

            # Persist the no-retry barrier before the sole remote write request.
            mark_pending()
            try:
                completed_raw = await asyncio.wait_for(
                    server.session.call_tool(
                        "complete_task",
                        arguments={"project_id": project_id, "task_id": task_id},
                    ),
                    timeout=30,
                )
            except Exception as exc:
                return {
                    "status": "completion_uncertain",
                    "error_type": type(exc).__name__,
                }
            completed = _plain_mcp_result(completed_raw)
            if completed.get("ok") is not True:
                return {
                    "status": "completion_uncertain",
                    "error_type": "complete-call-not-confirmed",
                }
            try:
                after_raw = await asyncio.wait_for(
                    server.session.call_tool(
                        "get_task_by_id", arguments={"task_id": task_id}
                    ),
                    timeout=15,
                )
            except Exception as exc:
                return {
                    "status": "completion_uncertain",
                    "error_type": type(exc).__name__,
                }
            after = _plain_mcp_result(after_raw)
            verified = after.get("ok") is True and _is_exact_completed_task(
                after.get("result"), task_id, project_id
            )
            return {
                "status": "completed_verified" if verified else "completion_uncertain",
                "error_type": "" if verified else "completion-readback-mismatch",
            }
        finally:
            try:
                await server.shutdown()
            except Exception:
                pass

    _ensure_mcp_loop()
    try:
        try:
            outcome = _run_on_mcp_loop(verify(), timeout=90)
        except Exception as exc:
            if str(state.get("status") or "") == "completion_pending":
                state["status"] = "completion_uncertain"
                state["error_type"] = type(exc).__name__
                state["completion_observed_at"] = datetime.now(settings.tz).isoformat(
                    timespec="seconds"
                )
                _save_dida_contract_state(state_path, state)
                print("完成请求可能已发出，但结果未能精确核验；不会自动重试。")
            else:
                print(f"完成前连接或回读失败：{type(exc).__name__}；未修改任务。")
            return 1
    finally:
        _stop_mcp_loop_if_idle()

    outcome_status = str(outcome.get("status") or "")
    if outcome_status == "precheck_failed":
        print("完成前未能精确核对专用任务；未发送完成请求。")
        return 1
    if outcome_status == "already_completed":
        print("专用任务在本次核验前已完成；未重复执行，完成结构仍不算实测通过。")
        return 1

    state["status"] = outcome_status or "completion_uncertain"
    state["error_type"] = str(outcome.get("error_type") or "")
    state["completion_observed_at"] = datetime.now(settings.tz).isoformat(
        timespec="seconds"
    )
    if state["status"] == "completed_verified":
        state["completion_verified_at"] = state["completion_observed_at"]
    _save_dida_contract_state(state_path, state)
    if state["status"] != "completed_verified":
        print("完成请求已发出，但结果未能精确核验；已记录状态且不会自动重试。")
        return 1
    print(f"完成并回读核验通过：{title}｜{destination}")
    return 0


def command_verify_dida_complete(args: argparse.Namespace) -> int:
    if not args.confirm_complete_test:
        print("已拒绝：缺少专用测试任务完成确认。")
        return 2
    if os.getenv(COMPLETE_TEST_APPROVAL_ENV, "").strip() != "1":
        print("已拒绝：当前进程没有专用完成核验授权。")
        return 2
    settings = load_settings(PROJECT_ROOT)
    try:
        state_path = _dida_contract_state_path(settings)
    except Exception as exc:
        print(f"本地核验状态路径不可用：{type(exc).__name__}")
        return 1
    return _run_contract_probe_with_lock(
        state_path,
        "complete",
        lambda: _command_verify_dida_complete_locked(args),
    )


def _find_project_node(value: object, project_id: str) -> dict[str, object] | None:
    if isinstance(value, dict):
        node_id = next(
            (
                str(value[key])
                for key in ("project_id", "projectId", "id")
                if value.get(key)
            ),
            "",
        )
        if node_id == project_id:
            return value
        for child in value.values():
            found = _find_project_node(child, project_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_project_node(child, project_id)
            if found is not None:
                return found
    return None


def command_inspect_dida_contract(args: argparse.Namespace) -> int:
    """Read back the test task and reconcile only the local verification record."""

    del args
    settings = load_settings(PROJECT_ROOT)
    try:
        state_path = _dida_contract_state_path(settings)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"无法读取本地核验状态：{type(exc).__name__}")
        return 1
    task_id = str(state.get("task_id") or "")
    expected_title = str(state.get("title") or "")
    if not task_id or not expected_title:
        print("本地核验状态缺少任务引用；不会调用滴答。")
        return 1

    from hermes_cli.mcp_config import _get_mcp_servers, _resolve_mcp_server_config
    from tools.mcp_tool import (
        _connect_server,
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _stop_mcp_loop_if_idle,
    )

    servers = _get_mcp_servers()
    if settings.dida_server not in servers:
        print("滴答 MCP 未配置。")
        return 1
    config = _resolve_mcp_server_config(servers[settings.dida_server])

    async def inspect() -> dict[str, object]:
        server = await asyncio.wait_for(
            _connect_server(settings.dida_server, config), timeout=30
        )
        try:
            task_raw = await asyncio.wait_for(
                server.session.call_tool(
                    "get_task_by_id", arguments={"task_id": task_id}
                ),
                timeout=15,
            )
            task_plain = _plain_mcp_result(task_raw)
            projects_raw = await asyncio.wait_for(
                server.session.call_tool("list_projects", arguments={}), timeout=20
            )
            return {
                "task": task_plain,
                "projects": _plain_mcp_result(projects_raw),
            }
        finally:
            await server.shutdown()

    _ensure_mcp_loop()
    try:
        try:
            result = _run_on_mcp_loop(inspect(), timeout=75)
        except Exception as exc:
            print(f"只读回读失败：{type(exc).__name__}")
            return 1
    finally:
        _stop_mcp_loop_if_idle()

    task_result = result.get("task") if isinstance(result, dict) else None
    project_result = result.get("projects") if isinstance(result, dict) else None
    task_payload = task_result.get("result") if isinstance(task_result, dict) else None
    node = _find_exact_task_node(task_payload, task_id)
    returned_project_id = str(
        (node or {}).get("projectId") or (node or {}).get("project_id") or ""
    )
    project_payload = (
        project_result.get("result") if isinstance(project_result, dict) else None
    )
    project_node = _find_project_node(project_payload, returned_project_id)
    project_name = str(
        (project_node or {}).get("name")
        or (project_node or {}).get("title")
        or (project_node or {}).get("projectName")
        or ""
    )
    task_project_name = str((node or {}).get("projectName") or "")
    project_is_default_inbox = (
        str(state.get("destination") or "").strip().casefold() == "inbox"
        and bool(returned_project_id)
        and project_node is None
    )
    project_matches = (
        project_name.strip().casefold() == "inbox"
        or task_project_name.strip().casefold() == "inbox"
        or project_is_default_inbox
    )
    checks = {
        "task_read_ok": isinstance(task_result, dict)
        and task_result.get("ok") is True,
        "task_id_match": node is not None,
        "title_match": str((node or {}).get("title") or "").strip().casefold()
        == expected_title.strip().casefold(),
        "task_is_active": node is not None and not _node_is_completed(node),
        "project_read_ok": isinstance(project_result, dict)
        and project_result.get("ok") is True,
        "project_matches_inbox": project_matches,
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    verified = all(checks.values())
    if verified:
        state["status"] = "created_verified"
        state["error_type"] = ""
        state["verified_at"] = datetime.now(settings.tz).isoformat(timespec="seconds")
        _save_dida_contract_state(state_path, state)
    return 0 if verified else 1


def command_inspect_dida_task(args: argparse.Namespace) -> int:
    """Search a task title read-only and report counts without exposing IDs."""

    title = str(args.title or "").strip()
    if not title or len(title) > 200:
        print("已拒绝：请提供 1 到 200 个字符的任务标题。")
        return 2
    settings = load_settings(PROJECT_ROOT)
    from hermes_cli.mcp_config import _get_mcp_servers, _resolve_mcp_server_config
    from tools.mcp_tool import (
        _connect_server,
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _stop_mcp_loop_if_idle,
    )

    servers = _get_mcp_servers()
    if settings.dida_server not in servers:
        print("滴答 MCP 未配置。")
        return 1
    config = _resolve_mcp_server_config(servers[settings.dida_server])

    async def inspect() -> dict[str, object]:
        server = await asyncio.wait_for(
            _connect_server(settings.dida_server, config), timeout=30
        )
        try:
            raw = await asyncio.wait_for(
                server.session.call_tool(
                    "search_task", arguments={"query": title}
                ),
                timeout=20,
            )
            search = _plain_mcp_result(raw)
            refs = _extract_task_references(search.get("result"))
            exact = tuple(
                ref
                for ref in refs
                if ref.title.strip().casefold() == title.casefold()
            )
            detail: dict[str, object] | None = None
            if search.get("ok") is True and len(exact) == 1:
                detail_raw = await asyncio.wait_for(
                    server.session.call_tool(
                        "get_task_by_id", arguments={"task_id": exact[0].task_id}
                    ),
                    timeout=15,
                )
                detail = _plain_mcp_result(detail_raw)
            return {"search": search, "detail": detail}
        finally:
            try:
                await server.shutdown()
            except Exception:
                pass

    _ensure_mcp_loop()
    try:
        try:
            outcome = _run_on_mcp_loop(inspect(), timeout=80)
        except Exception as exc:
            print(f"只读任务查询失败：{type(exc).__name__}")
            return 1
    finally:
        _stop_mcp_loop_if_idle()

    envelope = outcome.get("search") if isinstance(outcome, dict) else None
    detail = outcome.get("detail") if isinstance(outcome, dict) else None
    if not isinstance(envelope, dict):
        print("只读任务查询返回结构无效。")
        return 1
    refs = _extract_task_references(envelope.get("result"))
    exact = tuple(ref for ref in refs if ref.title.strip().casefold() == title.casefold())
    completed_values = {"2", "completed", "complete", "done"}
    active = tuple(ref for ref in exact if ref.status.casefold() not in completed_values)
    report: dict[str, object] = {
        "search_ok": envelope.get("ok") is True,
        "exact_matches": len(exact),
        "exact_active_matches": len(active),
        "exact_completed_matches": len(exact) - len(active),
    }
    if len(exact) == 1 and isinstance(detail, dict) and detail.get("ok") is True:
        node = _find_exact_task_node(detail.get("result"), exact[0].task_id)
        if node is not None:
            due_raw = str(node.get("dueDate") or node.get("due_date") or "")
            if due_raw:
                try:
                    due = datetime.fromisoformat(due_raw.replace("Z", "+00:00"))
                    if due.tzinfo is not None:
                        due = due.astimezone(settings.tz)
                    report["due"] = due.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    report["due"] = "无法解析"
            report["is_all_day"] = bool(
                node.get("isAllDay") or node.get("is_all_day")
            )
            report["detail_matches_exact_task"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if envelope.get("ok") is True else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="微信 AI 个人秘书本地工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser("dry-run", help="运行离线测试消息，不执行真实写入")
    dry.add_argument(
        "--fixtures",
        default=str(PROJECT_ROOT / "tests" / "fixtures" / "dry_run_messages.json"),
    )
    dry.set_defaults(func=command_dry_run)

    doctor = subparsers.add_parser("doctor", help="检查配置状态，不显示秘密值")
    doctor.add_argument("--strict", action="store_true")
    doctor.set_defaults(func=command_doctor)

    vault = subparsers.add_parser("inspect-vault", help="只读查看已指定 Vault 的两层结构")
    vault.add_argument("--limit", type=int, default=200)
    vault.set_defaults(func=command_inspect_vault)

    dida = subparsers.add_parser(
        "inspect-dida",
        help="只读列出滴答现有清单、文件夹与标签，不启动聊天",
    )
    dida.set_defaults(func=command_inspect_dida)

    dida_schema = subparsers.add_parser(
        "inspect-dida-schema",
        help="只读发现滴答创建、完成与回读工具结构，不调用工具",
    )
    dida_schema.set_defaults(func=command_inspect_dida_schema)

    dida_create = subparsers.add_parser(
        "verify-dida-create",
        help="创建一条专用测试任务并回读核验；绝不自动重试",
    )
    dida_create.add_argument("--confirm-create-test", action="store_true")
    dida_create.set_defaults(func=command_verify_dida_create)

    dida_complete = subparsers.add_parser(
        "verify-dida-complete",
        help="只完成已记录的专用测试任务并回读核验；绝不自动重试",
    )
    dida_complete.add_argument("--confirm-complete-test", action="store_true")
    dida_complete.set_defaults(func=command_verify_dida_complete)

    dida_contract = subparsers.add_parser(
        "inspect-dida-contract",
        help="只读核对已记录的专用测试任务，不显示任务标识",
    )
    dida_contract.set_defaults(func=command_inspect_dida_contract)

    dida_task = subparsers.add_parser(
        "inspect-dida-task",
        help="按精确标题只读查询滴答任务；不显示任务标识",
    )
    dida_task.add_argument("--title", required=True)
    dida_task.set_defaults(func=command_inspect_dida_task)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
