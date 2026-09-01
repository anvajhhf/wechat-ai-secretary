from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import logging
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .classifier import HermesStructuredClassifier
from .config import SecretarySettings, load_settings
from .dida import DidaExecutor
from .ledger import IdempotencyLedger
from .media import LocalMediaPreprocessor
from .models import ExecutionStatus, HandlingResult, MessageEnvelope
from .obsidian import ObsidianExecutor
from .path_security import source_reference
from .prefixes import parse_prefix
from .private_inbox import PrivateInboxExecutor
from .reminders import (
    ReminderDeliveryPreSendError,
    ReminderDeliveryUncertainError,
    ReminderScheduler,
)
from .replies import add_dry_run_previews
from .service import SecretaryService
from .web_reader import SafeWebReader


logger = logging.getLogger(__name__)
_PRIVATE_PROTECTION_REPLY = "上一条私密开关还没有成功处理。为保护隐私，这条内容没有处理；请稍后重发“私密：下一条”，收到确认后再发送内容。"


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


def _event_text(event: Any, platform: str) -> str:
    """Trust native payload provenance, never typed wrapper lookalikes."""

    text = str(getattr(event, "text", "") or "")
    raw_message = getattr(event, "raw_message", None)
    if platform != "weixin" or not isinstance(raw_message, dict):
        return text
    items = raw_message.get("item_list")
    if not isinstance(items, list) or not items:
        return text
    text_items = [item for item in items if isinstance(item, dict) and item.get("type") == 1]
    for native_item in text_items:
        native = native_item.get("text_item") or {}
        native_text = native.get("text") if isinstance(native, dict) else None
        if isinstance(native_text, str) and parse_prefix(native_text).private:
            decision = parse_prefix(native_text)
            if len(text_items) != 1:
                raise ValueError("ambiguous-private-envelope")
            if isinstance(native_item.get("ref_msg"), dict):
                # The adapter prepends reference context before the user's own
                # message. Only native private requests bypass that wrapper;
                # other referenced text remains non-authoritative data.
                from gateway.platforms.weixin import _extract_text

                if text == _extract_text(items):
                    return native_text if decision.arm_private_next else f"私密：\n{text}"
                raise ValueError("mismatched-private-envelope")
            if text != native_text:
                raise ValueError("mismatched-private-envelope")
    if len(items) != 1:
        return text
    item = items[0]
    if not isinstance(item, dict):
        return text
    if item.get("type") != 3:
        return text
    voice = item.get("voice_item")
    if not isinstance(voice, dict):
        return text
    media = voice.get("media")
    if media is not None and media != {}:
        return text
    transcript = voice.get("text")
    if not isinstance(transcript, str) or not transcript:
        return text

    # With no downloaded audio, the current Weixin adapter labels its STT
    # fallback as TEXT, not VOICE. Verify the raw item and the entire text so
    # typed lookalikes or debounced/combined messages are never rewritten.
    wrapper = "[Voice transcription provided by Weixin]\n"
    return transcript if text == wrapper + transcript else text


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
        text=_event_text(event, platform),
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
    send_awaitable = None
    try:
        metadata = dict(getattr(event, "metadata", None) or {})
        thread_id = getattr(source, "thread_id", None)
        if thread_id is not None:
            metadata.setdefault("thread_id", thread_id)
        send_awaitable = adapter.send(
            chat_id=chat_id,
            content=content,
            reply_to=str(getattr(event, "message_id", "") or "") or None,
            metadata=metadata or None,
        )
        try:
            future = asyncio.run_coroutine_threadsafe(send_awaitable, loop)
        except Exception:
            if inspect.iscoroutine(send_awaitable):
                send_awaitable.close()
            raise
        result = future.result(timeout=30)
        if hasattr(result, "success") and not bool(result.success):
            logger.warning("Secretary reply delivery returned unsuccessful status")
        if isinstance(result, dict) and (result.get("success") is False or result.get("error")):
            logger.warning("Secretary reply delivery returned an error status")
    except Exception as exc:
        logger.warning("Secretary reply delivery failed: %s", type(exc).__name__)


def _reminder_sender(
    loop: asyncio.AbstractEventLoop, gateway: Any, record: Any, content: str
) -> str:
    adapter = _find_adapter(gateway, record.platform)
    if adapter is None or not record.chat_id:
        raise ReminderDeliveryPreSendError("route-unavailable")
    if loop.is_closed():
        raise ReminderDeliveryPreSendError("dispatch-loop-closed")
    if getattr(adapter, "is_connected", None) is False:
        raise ReminderDeliveryPreSendError("transport-not-ready")

    try:
        send_awaitable = adapter.send(
            chat_id=record.chat_id,
            content=content,
            reply_to=None,
            metadata=None,
        )
    except Exception:
        # A non-standard synchronous adapter may already have performed side
        # effects before raising, so absence of a returned awaitable is not
        # sufficient proof that no message was sent.
        raise ReminderDeliveryUncertainError("delivery-exception") from None
    try:
        future = asyncio.run_coroutine_threadsafe(send_awaitable, loop)
    except Exception:
        # The awaitable was never submitted to the event loop, so this is the
        # one post-construction failure for which retry remains safe.
        if inspect.iscoroutine(send_awaitable):
            send_awaitable.close()
        raise ReminderDeliveryPreSendError("dispatch-submit-failed") from None

    try:
        result = future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        # Do not cancel: the platform operation may still finish.  Either way,
        # its outcome cannot be proven here and an automatic retry is unsafe.
        raise ReminderDeliveryUncertainError("delivery-timeout") from None
    except concurrent.futures.CancelledError:
        raise ReminderDeliveryUncertainError("delivery-cancelled") from None
    except Exception:
        raise ReminderDeliveryUncertainError("delivery-exception") from None
    if hasattr(result, "success") and not bool(result.success):
        raise ReminderDeliveryUncertainError("adapter-reported-failure")
    if isinstance(result, dict) and (
        result.get("success") is False or result.get("error")
    ):
        raise ReminderDeliveryUncertainError("adapter-reported-failure")
    if getattr(result, "message_id", None):
        return str(result.message_id)
    if isinstance(result, dict):
        for key in ("message_id", "messageId", "id"):
            if result.get(key):
                return str(result[key])
    return ""


def _attach_reminder_scheduler(
    scheduler: ReminderScheduler,
    loop: asyncio.AbstractEventLoop | None,
    gateway: Any,
) -> bool:
    """Bind the live Weixin sender without making gateway readiness fragile."""

    if loop is None or loop.is_closed() or _find_adapter(gateway, "weixin") is None:
        return False
    scheduler.attach(
        lambda record, content: _reminder_sender(loop, gateway, record, content)
    )
    return True


class GatewayReadyBridge:
    """Attach reminder delivery as soon as Hermes exposes ready adapters."""

    def __init__(self, scheduler: ReminderScheduler):
        self.scheduler = scheduler

    def __call__(
        self,
        gateway: Any,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        ready_loop = loop or getattr(gateway, "_gateway_loop", None)
        try:
            if _attach_reminder_scheduler(self.scheduler, ready_loop, gateway):
                logger.info("Wechat secretary reminder scheduler attached at gateway ready")
        except Exception as exc:
            logger.warning(
                "Wechat secretary gateway-ready reminder attach failed: %s",
                type(exc).__name__,
            )


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
        self._queue_lock = threading.Lock()
        self._queues: dict[str, deque[tuple[Any, Any, Any, MessageEnvelope, str]]] = {}
        # Bounded by the configured allowlist, not by arbitrary chat IDs. A
        # rejected private-next command must not expose the following message.
        self._privacy_blocked: set[str] = set()
        self._privacy_tokens: dict[str, str] = {}
        self._unpersisted_private: set[str] = set()
        self._privacy_lock = threading.RLock()

    def _private_protection_token(self, sender_key: str) -> str:
        with self._privacy_lock:
            ledger = getattr(self.service, "ledger", None)
            if sender_key in self._unpersisted_private:
                token = self._privacy_tokens[sender_key]
                setter = getattr(ledger, "set_private_protection", None)
                try:
                    if callable(setter):
                        setter(sender_key, token)
                    self._unpersisted_private.discard(sender_key)
                except Exception:
                    # Failure reporting itself must never escape the gateway
                    # hook and fall through to the generic model dispatcher.
                    return token
            getter = getattr(ledger, "get_private_protection", None)
            token = getter(sender_key) if callable(getter) else self._privacy_tokens.get(sender_key, "")
            if token:
                self._privacy_blocked.add(sender_key)
            else:
                self._privacy_blocked.discard(sender_key)
            return token

    def _block_private(self, sender_key: str) -> None:
        with self._privacy_lock:
            token = uuid4().hex
            self._privacy_tokens[sender_key] = token
            self._privacy_blocked.add(sender_key)
            ledger = getattr(self.service, "ledger", None)
            setter = getattr(ledger, "set_private_protection", None)
            try:
                if callable(setter):
                    setter(sender_key, token)
                self._unpersisted_private.discard(sender_key)
            except Exception as exc:
                self._unpersisted_private.add(sender_key)
                logger.error("Secretary private protection persistence failed: %s", type(exc).__name__)

    def _clear_private_protection(self, sender_key: str, expected_token: str) -> None:
        with self._privacy_lock:
            if self._private_protection_token(sender_key) != expected_token:
                return
            ledger = getattr(self.service, "ledger", None)
            clearer = getattr(ledger, "clear_private_protection", None)
            if callable(clearer) and not clearer(sender_key, expected_token):
                return
            self._privacy_tokens.pop(sender_key, None)
            self._privacy_blocked.discard(sender_key)

    def _drain_conversation(self, conversation_key: str) -> None:
        """Keep private latches and clarifications ahead of their follow-ups."""
        while True:
            with self._queue_lock:
                queue = self._queues[conversation_key]
                if not queue:
                    del self._queues[conversation_key]
                    return
                loop, gateway, event, message, protection_token = queue.popleft()
            ref = source_reference(message)
            try:
                private_decision = parse_prefix(message.text)
                blocked = bool(self._private_protection_token(message.sender_key))
                if blocked and not private_decision.private:
                    result = HandlingResult(status=ExecutionStatus.SKIPPED, reply=_PRIVATE_PROTECTION_REPLY)
                else:
                    result = self.service.handle(message)
                if private_decision.arm_private_next:
                    # For this exact prefix the service's only successful,
                    # nonduplicate path has persisted the private latch. A
                    # command queued BEFORE a newer failure cannot clear it.
                    if result.status is ExecutionStatus.SUCCEEDED and not result.duplicate:
                        self._clear_private_protection(message.sender_key, protection_token)
                    elif not result.duplicate:
                        self._block_private(message.sender_key)
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
                if parse_prefix(message.text).arm_private_next:
                    self._block_private(message.sender_key)
                logger.error(
                    "Secretary worker failed ref=%s error=%s", ref, type(exc).__name__
                )
                _schedule_reply(
                    loop,
                    gateway,
                    event,
                    "抱歉，这次没能处理成功：本地处理组件异常，没有确认任何外部写入。",
                )
            finally:
                self._slots.release()

    def __call__(self, event: Any, gateway: Any, **kwargs: Any) -> dict[str, str]:
        del kwargs
        private_next = False
        privacy_sender = ""
        try:
            source = getattr(event, "source", None)
            platform = _platform_name(getattr(source, "platform", ""))
            if platform != "weixin":
                return {"action": "allow"}
            user_id = str(getattr(source, "user_id_alt", None) or getattr(source, "user_id", None) or getattr(event, "user_id", None) or "")
            if user_id in self.service.settings.allowed_users:
                privacy_sender = f"{platform}:{self.service.settings.account_id or 'weixin-default'}:{user_id}"
                private_next = parse_prefix(str(getattr(event, "text", "") or "")).arm_private_next
                raw = getattr(event, "raw_message", None)
                for item in (raw.get("item_list", []) if isinstance(raw, dict) else []):
                    if isinstance(item, dict) and item.get("type") == 1:
                        native = item.get("text_item") or {}
                        if isinstance(native, dict):
                            private_next |= parse_prefix(str(native.get("text") or "")).arm_private_next
            message = _event_to_message(event, self.service.settings)
            if not self.service.accepts(message):
                return {"action": "skip", "reason": "secretary-policy"}
            ref = source_reference(message)
            loop = asyncio.get_running_loop()
            private_decision = parse_prefix(message.text)
            private_next |= private_decision.arm_private_next
            protection_token = self._private_protection_token(message.sender_key)
            privacy_blocked = bool(protection_token)
            if privacy_blocked and not private_decision.private:
                threading.Thread(
                    target=_schedule_reply,
                    args=(loop, gateway, event, _PRIVATE_PROTECTION_REPLY),
                    daemon=True,
                    name="secretary-private-protection",
                ).start()
                return {"action": "skip", "reason": "secretary-private-protection"}
            # Fallback for older Hermes runtimes and adapters that reconnect
            # after the one-shot gateway_ready hook.
            _attach_reminder_scheduler(self.scheduler, loop, gateway)
            if not self._slots.acquire(blocking=False):
                if private_next:
                    self._block_private(message.sender_key)
                threading.Thread(
                    target=_schedule_reply,
                    args=(loop, gateway, event, "现在消息有点多，请稍后再试。这条消息没有执行。"),
                    daemon=True,
                    name="secretary-busy-reply",
                ).start()
                return {"action": "skip", "reason": "secretary-busy"}

            with self._queue_lock:
                queue = self._queues.get(message.conversation_key)
                if queue is None:
                    queue = deque()
                    self._queues[message.conversation_key] = queue
                    queue.append((loop, gateway, event, message, protection_token))
                    try:
                        threading.Thread(
                            target=lambda: self._drain_conversation(message.conversation_key),
                            daemon=True,
                            name=f"secretary-{ref}",
                        ).start()
                    except BaseException:
                        del self._queues[message.conversation_key]
                        self._slots.release()
                        raise
                else:
                    queue.append((loop, gateway, event, message, protection_token))
            return {"action": "skip", "reason": "secretary-handled"}
        except BaseException as exc:
            if private_next and privacy_sender:
                self._block_private(privacy_sender)
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
            web=SafeWebReader(settings),
        )
        scheduler = ReminderScheduler(settings, ledger)
        bridge = GatewayBridge(service, scheduler)
        ctx.register_hook("gateway_ready", GatewayReadyBridge(scheduler))
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
        logger.info(
            "Wechat secretary plugin ready (dry_run=%s, asr_backend=%s, asr_threads=%s)",
            settings.dry_run, settings.asr_backend, settings.asr_threads,
        )
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
