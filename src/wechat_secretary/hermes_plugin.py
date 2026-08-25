from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .classifier import HermesStructuredClassifier
from .config import SecretarySettings, load_settings
from .dida import DidaExecutor
from .ledger import IdempotencyLedger
from .media import LocalMediaPreprocessor
from .models import MessageEnvelope
from .obsidian import ObsidianExecutor
from .path_security import source_reference
from .private_inbox import PrivateInboxExecutor
from .reminders import ReminderScheduler
from .replies import add_dry_run_previews
from .service import SecretaryService


logger = logging.getLogger(__name__)


def _json_tool_result(value: Any) -> str:
    """Hermes plugin tools must return text, not a raw Python mapping."""
    return json.dumps(value, ensure_ascii=False, default=str)


def _platform_name(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _find_adapter(gateway: Any, platform: str) -> Any | None:
    adapters = getattr(gateway, "adapters", {}) or {}
    if platform in adapters:
        return adapters[platform]
    for key, adapter in adapters.items():
        if _platform_name(key) == platform:
            return adapter
    return None


def _event_to_message(event: Any, settings: SecretarySettings) -> MessageEnvelope:
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", ""))
    user_id = str(
        getattr(source, "user_id_alt", None)
        or getattr(source, "user_id", None)
        or getattr(event, "user_id", None)
        or ""
    )
    timestamp = getattr(event, "timestamp", None) or datetime.now(settings.tz)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=settings.tz)
    media_paths = tuple(str(item) for item in (getattr(event, "media_urls", None) or []))
    media_types = tuple(str(item) for item in (getattr(event, "media_types", None) or []))
    return MessageEnvelope(
        platform=platform,
        account_id=settings.account_id or "weixin-default",
        user_id=user_id,
        chat_id=str(getattr(source, "chat_id", "") or ""),
        chat_type=str(getattr(source, "chat_type", "") or ""),
        message_id=str(getattr(event, "message_id", "") or ""),
        text=str(getattr(event, "text", "") or ""),
        received_at=timestamp,
        media_paths=media_paths,
        media_types=media_types,
    )


def _schedule_reply(loop: asyncio.AbstractEventLoop, gateway: Any, event: Any, content: str) -> None:
    if not content:
        return
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", ""))
    adapter = _find_adapter(gateway, platform)
    chat_id = str(getattr(source, "chat_id", "") or "")
    if adapter is None or not chat_id or loop.is_closed():
        logger.warning("Secretary reply route unavailable for platform=%s", platform)
        return
    metadata = dict(getattr(event, "metadata", None) or {})
    thread_id = getattr(source, "thread_id", None)
    if thread_id is not None:
        metadata.setdefault("thread_id", thread_id)
    future = asyncio.run_coroutine_threadsafe(
        adapter.send(
            chat_id=chat_id,
            content=content,
            reply_to=str(getattr(event, "message_id", "") or "") or None,
            metadata=metadata or None,
        ),
        loop,
    )
    try:
        result = future.result(timeout=30)
        if hasattr(result, "success") and not bool(result.success):
            logger.warning("Secretary reply delivery returned unsuccessful status")
        if isinstance(result, dict) and result.get("error"):
            logger.warning("Secretary reply delivery returned an error status")
    except Exception as exc:
        logger.warning("Secretary reply delivery failed: %s", type(exc).__name__)


def _reminder_sender(
    loop: asyncio.AbstractEventLoop, gateway: Any, record: Any, content: str
) -> str:
    adapter = _find_adapter(gateway, record.platform)
    if adapter is None or not record.chat_id or loop.is_closed():
        raise RuntimeError("weixin reminder route unavailable")
    future = asyncio.run_coroutine_threadsafe(
        adapter.send(
            chat_id=record.chat_id,
            content=content,
            reply_to=None,
            metadata=None,
        ),
        loop,
    )
    result = future.result(timeout=30)
    if hasattr(result, "success") and not bool(result.success):
        raise RuntimeError("weixin reminder delivery was not successful")
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError("weixin reminder delivery returned an error status")
    if getattr(result, "message_id", None):
        return str(result.message_id)
    if isinstance(result, dict):
        for key in ("message_id", "messageId", "id"):
            if result.get(key):
                return str(result[key])
    return ""


class FailClosedBridge:
    def __init__(self, reason_code: str, allowed_users: frozenset[str]):
        self.reason_code = reason_code
        self.allowed_users = allowed_users

    def __call__(self, event: Any, gateway: Any, **kwargs: Any) -> dict[str, str]:
        del kwargs
        source = getattr(event, "source", None)
        platform = _platform_name(getattr(source, "platform", ""))
        if platform != "weixin":
            return {"action": "allow"}
        user_id = str(
            getattr(source, "user_id_alt", None)
            or getattr(source, "user_id", None)
            or getattr(event, "user_id", None)
            or ""
        )
        chat_type = str(getattr(source, "chat_type", "") or "")
        if chat_type not in {"dm", "private"} or user_id not in self.allowed_users:
            return {"action": "skip", "reason": "secretary-policy"}
        try:
            loop = asyncio.get_running_loop()
            thread = threading.Thread(
                target=_schedule_reply,
                args=(loop, gateway, event, "安全组件还没有准备好，这条消息暂时没有处理。请先在本机运行状态检查。"),
                daemon=True,
                name="secretary-deny-reply",
            )
            thread.start()
        except BaseException:
            pass
        logger.error("Secretary fail-closed: %s", self.reason_code)
        return {"action": "skip", "reason": "secretary-not-ready"}


class GatewayBridge:
    def __init__(self, service: SecretaryService, scheduler: ReminderScheduler):
        self.service = service
        self.scheduler = scheduler
        self._slots = threading.BoundedSemaphore(service.settings.worker_limit)

    def __call__(self, event: Any, gateway: Any, **kwargs: Any) -> dict[str, str]:
        del kwargs
        try:
            source = getattr(event, "source", None)
            platform = _platform_name(getattr(source, "platform", ""))
            if platform != "weixin":
                return {"action": "allow"}
            message = _event_to_message(event, self.service.settings)
            if not self.service.accepts(message):
                return {"action": "skip", "reason": "secretary-policy"}
            ref = source_reference(message)
            loop = asyncio.get_running_loop()
            self.scheduler.attach(
                lambda record, content: _reminder_sender(
                    loop, gateway, record, content
                )
            )
            if not self._slots.acquire(blocking=False):
                threading.Thread(
                    target=_schedule_reply,
                    args=(loop, gateway, event, "现在消息有点多，请稍后再试。这条消息没有执行。"),
                    daemon=True,
                    name="secretary-busy-reply",
                ).start()
                return {"action": "skip", "reason": "secretary-busy"}

            def worker() -> None:
                try:
                    result = self.service.handle(message)
                    reply = result.reply
                    if self.service.settings.dry_run:
                        reply = add_dry_run_previews(reply, result.results)
                    _schedule_reply(loop, gateway, event, reply)
                    logger.info(
                        "Secretary handled ref=%s status=%s actions=%d",
                        ref,
                        result.status.value,
                        len(result.results),
                    )
                except BaseException as exc:
                    logger.error(
                        "Secretary worker failed ref=%s error=%s",
                        ref,
                        type(exc).__name__,
                    )
                    _schedule_reply(
                        loop,
                        gateway,
                        event,
                        "抱歉，这次没能处理成功：本地处理组件异常，没有确认任何外部写入。",
                    )
                finally:
                    self._slots.release()

            threading.Thread(
                target=worker,
                daemon=True,
                name=f"secretary-{ref}",
            ).start()
            return {"action": "skip", "reason": "secretary-handled"}
        except BaseException as exc:
            logger.error("Secretary pre-dispatch failed closed: %s", type(exc).__name__)
            try:
                loop = asyncio.get_running_loop()
                threading.Thread(
                    target=_schedule_reply,
                    args=(
                        loop,
                        gateway,
                        event,
                        "抱歉，这次没能处理成功：本地安全检查异常，这条消息没有执行。",
                    ),
                    daemon=True,
                    name="secretary-fail-closed-reply",
                ).start()
            except BaseException:
                pass
            return {"action": "skip", "reason": "secretary-fail-closed"}


def _tool_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }


def register(ctx: Any) -> None:
    """Hermes plugin registration. Configuration failures still install a deny-all hook."""
    ctx.register_auxiliary_task(
        "wechat_secretary_classifier",
        display_name="微信秘书分类器",
        description="使用 DeepSeek 对单条非私密微信消息进行结构化分类",
        defaults={
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "none",
            "timeout": 60,
        },
    )
    ctx.register_auxiliary_task(
        "wechat_secretary_vision",
        display_name="微信秘书图片理解",
        description="使用 DeepSeek 低成本视觉模型理解单条非私密微信图片",
        defaults={
            "provider": "deepseek",
            "model": "deepseek-v4-flash-vision-exp",
            "reasoning_effort": "none",
            "timeout": 60,
        },
    )
    ctx.register_auxiliary_task(
        "wechat_secretary_deep_note",
        display_name="微信秘书深度笔记",
        description="使用 DeepSeek Pro 整理显式指定的深度笔记",
        defaults={
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "reasoning_effort": "none",
            "timeout": 60,
        },
    )
    settings: SecretarySettings | None = None
    try:
        settings = load_settings()
        config_errors = settings.runtime_errors(strict=True)
        if config_errors:
            raise ValueError("; ".join(config_errors))
        ledger = IdempotencyLedger(settings.state_db_path)
        dida = DidaExecutor(
            settings,
            caller=lambda server, tool, arguments, timeout: ctx.call_mcp(
                server, tool, arguments, timeout=timeout
            ),
        )
        service = SecretaryService(
            settings=settings,
            ledger=ledger,
            classifier=HermesStructuredClassifier(ctx, settings),
            dida=dida,
            obsidian=ObsidianExecutor(settings),
            private_inbox=PrivateInboxExecutor(settings),
            media=LocalMediaPreprocessor(settings),
        )
        scheduler = ReminderScheduler(settings, ledger)
        bridge = GatewayBridge(service, scheduler)
        ctx.register_hook("pre_gateway_dispatch", bridge)

        ctx.register_tool(
            name="secretary_morning_digest",
            toolset="wechat_secretary_digest",
            schema=_tool_schema(
                "secretary_morning_digest",
                "只读查询滴答清单并生成今日重点；不会创建、更新或删除任务。",
            ),
            handler=lambda params, **kwargs: service.run_scheduled_digest("morning"),
        )
        ctx.register_tool(
            name="secretary_evening_review",
            toolset="wechat_secretary_digest",
            schema=_tool_schema(
                "secretary_evening_review",
                "只读查询滴答清单并生成今日简短复盘；不会创建、更新或删除任务。",
            ),
            handler=lambda params, **kwargs: service.run_scheduled_digest("evening"),
        )
        ctx.register_tool(
            name="secretary_dida_taxonomy",
            toolset="wechat_secretary_setup",
            schema=_tool_schema(
                "secretary_dida_taxonomy",
                "只读列出滴答现有清单、文件夹与标签，用于人工确认分类映射。",
            ),
            handler=lambda params, **kwargs: _json_tool_result(dida.taxonomy()),
        )
        logger.info("Wechat secretary plugin ready (dry_run=%s)", settings.dry_run)
    except BaseException as exc:
        logger.error("Wechat secretary initialization failed: %s", type(exc).__name__)
        allowed_users = settings.allowed_users if settings is not None else frozenset(
            item.strip()
            for item in os.getenv("WEIXIN_ALLOWED_USERS", "").split(",")
            if item.strip()
        )
        ctx.register_hook(
            "pre_gateway_dispatch",
            FailClosedBridge(type(exc).__name__, allowed_users),
        )
