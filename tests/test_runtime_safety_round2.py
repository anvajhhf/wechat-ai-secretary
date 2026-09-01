from __future__ import annotations

import asyncio
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.weixin import WeixinAdapter, _extract_text

from wechat_secretary.config import SecretarySettings
from wechat_secretary.hermes_gateway_entry import main as gateway_entry_main
from wechat_secretary.hermes_plugin import GatewayBridge, _event_to_message, _schedule_reply
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.models import ActionResult, ExecutionStatus, HandlingResult, IntentKind, IntentPlan
from wechat_secretary.obsidian import ObsidianExecutor
from wechat_secretary.service import SecretaryService


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = SecretarySettings(
    project_root=ROOT,
    allowed_users=frozenset({"audit-user"}),
    account_id="audit-account",
)
NOW = datetime(2026, 8, 30, 15, 29, tzinfo=SETTINGS.tz)


def event(message_id: str, text: str, *, chat_id: str = "audit-chat") -> SimpleNamespace:
    return SimpleNamespace(
        source=SimpleNamespace(
            platform="weixin", user_id="audit-user", chat_id=chat_id, chat_type="dm"
        ),
        message_id=message_id,
        text=text,
        timestamp=NOW,
        media_urls=[],
        media_types=[],
    )


class WeixinIngressTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self) -> tuple[WeixinAdapter, list[object]]:
        adapter = object.__new__(WeixinAdapter)
        adapter._poll_session = object()
        adapter._account_id = "audit-bot"
        adapter._dedup = MessageDeduplicator(ttl_seconds=300)
        adapter._group_policy = "disabled"
        adapter._token_store = Mock()
        adapter._is_dm_intake_allowed = lambda sender: True
        adapter._maybe_fetch_typing_ticket = AsyncMock()
        adapter._text_batch_key = lambda incoming: "audit-chat"
        adapter._pending_text_batches = {}
        adapter._pending_text_batch_tasks = {}
        adapter._text_batch_delay_seconds = 0.01
        adapter._text_batch_split_delay_seconds = 0.01
        adapter.platform = SimpleNamespace(value="weixin")
        adapter.build_source = lambda **kwargs: SimpleNamespace(platform=SimpleNamespace(value="weixin"), **kwargs)
        seen: list[object] = []
        async def collect(item: dict, paths: list, types: list) -> None:
            if item.get("type") == 3:
                paths.append("fake-audio.silk")
                types.append("audio/silk")
        async def handle(incoming: object) -> None:
            seen.append(incoming)
        adapter._collect_media = collect
        adapter.handle_message = handle
        return adapter, seen

    @staticmethod
    def incoming(message_id: str, text: str = "", *, voice: bool = False) -> dict:
        item = {"type": 3, "voice_item": {"media": {"fake": True}}} if voice else {
            "type": 1, "text_item": {"text": text}
        }
        return {"message_id": message_id, "from_user_id": "audit-user", "item_list": [item]}

    async def flush(self, adapter: WeixinAdapter) -> None:
        tasks = tuple(adapter._pending_text_batch_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_distinct_message_ids_with_same_text_are_not_silently_dropped(self) -> None:
        adapter, seen = self.adapter()
        with patch.dict(os.environ, {"WECHAT_SECRETARY_STRICT_INGRESS": "1"}):
            await adapter._process_message(self.incoming("first", "今天"))
            await self.flush(adapter)
            await adapter._process_message(self.incoming("second", "今天"))
            await self.flush(adapter)
        self.assertEqual(["first", "second"], [incoming.message_id for incoming in seen])

    async def test_same_stable_message_id_is_still_deduplicated(self) -> None:
        adapter, seen = self.adapter()
        with patch.dict(os.environ, {"WECHAT_SECRETARY_STRICT_INGRESS": "1"}):
            await adapter._process_message(self.incoming("same", "今天"))
            await adapter._process_message(self.incoming("same", "今天"))
            await self.flush(adapter)
        self.assertEqual(1, len(seen))

    async def test_private_latch_text_is_dispatched_before_following_voice(self) -> None:
        adapter, seen = self.adapter()
        with patch.dict(os.environ, {"WECHAT_SECRETARY_STRICT_INGRESS": "1"}):
            await adapter._process_message(self.incoming("private-latch", "私密：下一条"))
            await adapter._process_message(self.incoming("voice", voice=True))
            await self.flush(adapter)
        self.assertEqual(["private-latch", "voice"], [incoming.message_id for incoming in seen])

    async def test_rapid_texts_keep_separate_message_identity(self) -> None:
        adapter, seen = self.adapter()
        with patch.dict(os.environ, {"WECHAT_SECRETARY_STRICT_INGRESS": "1"}):
            await adapter._process_message(self.incoming("private-latch", "私密：下一条"))
            await adapter._process_message(self.incoming("secret", "私密内容"))
            await self.flush(adapter)
        self.assertEqual(["私密：下一条", "私密内容"], [incoming.text for incoming in seen])

    async def test_generic_hermes_mode_keeps_original_batching(self) -> None:
        adapter, seen = self.adapter()
        with patch.dict(os.environ, {"WECHAT_SECRETARY_STRICT_INGRESS": ""}):
            await adapter._process_message(self.incoming("first", "第一条"))
            await adapter._process_message(self.incoming("second", "第二条"))
            await self.flush(adapter)
        self.assertEqual(["第一条\n第二条"], [incoming.text for incoming in seen])

    async def test_media_download_does_not_allow_following_text_to_overtake(self) -> None:
        adapter, seen = self.adapter()
        collect = adapter._collect_media
        async def slow_media(item: dict, paths: list, types: list) -> None:
            if item.get("type") == 3:
                await asyncio.sleep(0.01)
            await collect(item, paths, types)
        adapter._collect_media = slow_media
        with patch.dict(os.environ, {"WECHAT_SECRETARY_STRICT_INGRESS": "1"}):
            await asyncio.gather(
                adapter._process_message_safe(self.incoming("voice", voice=True)),
                adapter._process_message_safe(self.incoming("followup", "今天")),
            )
        self.assertEqual(["voice", "followup"], [incoming.message_id for incoming in seen])

    async def test_failed_download_releases_ingress_lock_for_followup(self) -> None:
        adapter, seen = self.adapter()
        collect = adapter._collect_media
        async def failed_media(item: dict, paths: list, types: list) -> None:
            if item.get("type") == 3:
                await asyncio.sleep(0)
                raise ValueError("isolated download failure")
            await collect(item, paths, types)
        adapter._collect_media = failed_media
        with patch.dict(os.environ, {"WECHAT_SECRETARY_STRICT_INGRESS": "1"}), self.assertLogs("gateway.platforms.weixin", level="ERROR"):
            await asyncio.gather(
                adapter._process_message_safe(self.incoming("voice", voice=True)),
                adapter._process_message_safe(self.incoming("followup", "今天")),
            )
        self.assertEqual(["followup"], [incoming.message_id for incoming in seen])
        self.assertEqual(0, len(adapter._secretary_ingress_locks))


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_project_gateway_refuses_to_run_without_ingress_patch_support(self) -> None:
        argv = ["entry", "hermes_cli.main", f"HERMES_HOME={ROOT}", "gateway", "run"]
        fake_main = SimpleNamespace(main=Mock(return_value=0))
        for support in (SimpleNamespace(), SimpleNamespace(WECHAT_SECRETARY_STRICT_INGRESS_SUPPORTED=False)):
            with self.subTest(support=support), patch.dict(os.environ, {"HERMES_HOME": str(ROOT), "HERMES_ENABLE_PROJECT_PLUGINS": "true"}), patch.object(sys, "argv", argv), patch.dict(sys.modules, {"gateway.platforms.weixin": support, "hermes_cli.main": fake_main}), patch("wechat_secretary.hermes_gateway_entry.os.chdir"):
                self.assertEqual(2, gateway_entry_main())
                fake_main.main.assert_not_called()

    def test_installer_includes_ingress_compatibility_patch(self) -> None:
        installer = (ROOT / "scripts" / "install-local.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("hermes-weixin-secretary-ingress.patch", installer)
        self.assertTrue((ROOT / "patches" / "hermes-weixin-secretary-ingress.patch").is_file())

    def test_gateway_oauth_patch_is_installed_and_process_wide(self) -> None:
        from tools.mcp_oauth import (
            OAuthNonInteractiveError,
            _is_interactive,
            _make_redirect_handler,
            force_interactive_oauth,
        )

        installer = (ROOT / "scripts" / "install-local.ps1").read_text(
            encoding="utf-8-sig"
        )
        patch_path = ROOT / "patches" / "hermes-gateway-no-interactive-oauth.patch"
        self.assertIn(patch_path.name, installer)
        self.assertTrue(patch_path.is_file())
        with patch.dict(
            os.environ,
            {"HERMES_MCP_OAUTH_INTERACTIVE": "0"},
            clear=False,
        ), force_interactive_oauth():
            self.assertFalse(_is_interactive())
            redirect = _make_redirect_handler(49152)
            with patch("tools.mcp_oauth.webbrowser.open") as browser_open:
                with self.assertRaises(OAuthNonInteractiveError):
                    asyncio.run(redirect("https://example.invalid/authorize"))
                browser_open.assert_not_called()


class GatewayOrderingTests(unittest.TestCase):
    def test_one_worker_per_conversation_preserves_submission_order(self) -> None:
        handled: list[str] = []
        queued: list[dict] = []
        def handle(incoming: object) -> HandlingResult:
            handled.append(incoming.message_id)
            return HandlingResult(status=ExecutionStatus.SUCCEEDED, reply="")
        service = SimpleNamespace(settings=SETTINGS, accepts=lambda incoming: True, handle=handle)
        bridge = GatewayBridge(service, Mock())
        def thread(**kwargs: object) -> object:
            queued.append(kwargs)
            return SimpleNamespace(start=lambda: None)
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                gateway = SimpleNamespace(adapters={"weixin": object()})
                bridge(event("private-latch", "私密：下一条"), gateway)
                bridge(event("voice", "私密后续"), gateway)
        asyncio.run(submit())
        self.assertEqual(1, len(queued))
        queued[0]["target"]()
        self.assertEqual(["private-latch", "voice"], handled)

    def test_different_conversations_can_run_independently(self) -> None:
        queued: list[dict] = []
        service = SimpleNamespace(settings=SETTINGS, accepts=lambda incoming: True)
        bridge = GatewayBridge(service, Mock())
        def thread(**kwargs: object) -> object:
            queued.append(kwargs)
            return SimpleNamespace(start=lambda: None)
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                gateway = SimpleNamespace(adapters={"weixin": object()})
                bridge(event("a", "第一条", chat_id="a"), gateway)
                bridge(event("b", "第二条", chat_id="b"), gateway)
        asyncio.run(submit())
        self.assertEqual(2, len(queued))

    def test_synchronous_reply_adapter_failure_does_not_escape_worker(self) -> None:
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        gateway = SimpleNamespace(adapters={"weixin": SimpleNamespace(send=Mock(side_effect=RuntimeError("local failure")))})
        _schedule_reply(loop, gateway, event("reply", "测试"), "回执")

    def test_failed_message_releases_slot_and_next_message_runs(self) -> None:
        handled: list[str] = []
        queued: list[dict] = []
        def handle(incoming: object) -> HandlingResult:
            handled.append(incoming.message_id)
            if len(handled) == 1:
                raise ValueError("isolated failure")
            return HandlingResult(status=ExecutionStatus.SUCCEEDED, reply="")
        service = SimpleNamespace(settings=SETTINGS, accepts=lambda incoming: True, handle=handle)
        bridge = GatewayBridge(service, Mock())
        def thread(**kwargs: object) -> object:
            queued.append(kwargs)
            return SimpleNamespace(start=lambda: None)
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                gateway = SimpleNamespace(adapters={"weixin": object()})
                bridge(event("failed", "第一条"), gateway)
                bridge(event("following", "第二条"), gateway)
        asyncio.run(submit())
        queued[0]["target"]()
        self.assertEqual(["failed", "following"], handled)
        self.assertEqual({}, bridge._queues)
        self.assertEqual(SETTINGS.worker_limit, bridge._slots._value)


class NativePrivateBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = IdempotencyLedger(":memory:")
        self.addCleanup(self.ledger.close)
        self.classifier = SimpleNamespace(
            call_count=0,
            classify=Mock(return_value=IntentPlan(kind=IntentKind.CLARIFY, confidence=0)),
        )
        self.media = SimpleNamespace(prepare=Mock(side_effect=AssertionError("private media must not be decoded")))
        self.private = SimpleNamespace(save=Mock(return_value=ActionResult("private", ExecutionStatus.PLANNED, "私密内容")))
        self.service = SecretaryService(
            settings=SETTINGS,
            ledger=self.ledger,
            classifier=self.classifier,
            dida=Mock(),
            obsidian=ObsidianExecutor(SETTINGS),
            private_inbox=self.private,
            media=self.media,
        )

    @staticmethod
    def quoted_event(message_id: str, own_text: str, quoted: str = "普通引用内容") -> SimpleNamespace:
        items = [{"type": 1, "text_item": {"text": own_text}, "ref_msg": {
            "title": "引用原文", "message_item": {"type": 1, "text_item": {"text": quoted}}
        }}]
        incoming = event(message_id, _extract_text(items))
        incoming.raw_message = {"item_list": items}
        return incoming

    def test_native_quoted_private_message_keeps_full_reference_without_model(self) -> None:
        incoming = self.quoted_event("private", "私密：仅本地保存")
        incoming.media_urls = ["fake-private.silk"]
        incoming.media_types = ["audio/silk"]
        self.service.handle(_event_to_message(incoming, SETTINGS))
        self.private.save.assert_called_once()
        saved = self.private.save.call_args.args[0]
        self.assertIn(incoming.text, saved.text)
        self.classifier.classify.assert_not_called()
        self.media.prepare.assert_not_called()

    def test_native_quoted_private_next_arms_before_following_media(self) -> None:
        self.service.handle(_event_to_message(self.quoted_event("latch", "私密：下一条"), SETTINGS))
        following = event("audio", "")
        following.media_urls = ["fake-private.silk"]
        following.media_types = ["audio/silk"]
        self.service.handle(_event_to_message(following, SETTINGS))
        self.private.save.assert_called_once()
        self.classifier.classify.assert_not_called()
        self.media.prepare.assert_not_called()

    def test_private_only_in_quoted_material_does_not_activate_latch(self) -> None:
        incoming = self.quoted_event("reference-only", "看看原文", "私密：下一条")
        normalized = _event_to_message(incoming, SETTINGS)
        self.assertEqual(incoming.text, normalized.text)
        self.service.handle(normalized)
        self.private.save.assert_not_called()
        self.assertFalse(self.ledger.consume_private_latch(normalized.sender_key, NOW))

    def test_typed_or_mismatched_reference_wrapper_is_not_stripped(self) -> None:
        incoming = event("typed", "[引用: 原文]\n私密：下一条")
        self.assertEqual(incoming.text, _event_to_message(incoming, SETTINGS).text)
        incoming = self.quoted_event("mismatch", "私密：下一条")
        incoming.text += "\nadditional actual message"
        with self.assertRaises(ValueError):
            _event_to_message(incoming, SETTINGS)

    def test_native_quoted_private_with_additional_media_stays_private(self) -> None:
        incoming = self.quoted_event("compound", "私密：仅本地保存")
        incoming.raw_message["item_list"].append({"type": 2, "image_item": {}})
        incoming.text = _extract_text(incoming.raw_message["item_list"])
        incoming.media_urls = ["fake-private.jpg"]
        incoming.media_types = ["image/jpeg"]
        self.service.handle(_event_to_message(incoming, SETTINGS))
        self.private.save.assert_called_once()
        self.classifier.classify.assert_not_called()
        self.media.prepare.assert_not_called()

    def test_ambiguous_native_private_multiple_text_items_fail_closed(self) -> None:
        incoming = self.quoted_event("compound-text", "私密：下一条")
        incoming.raw_message["item_list"].append({"type": 1, "text_item": {"text": "其他文本"}})
        with self.assertRaises(ValueError):
            _event_to_message(incoming, SETTINGS)
        self.classifier.classify.assert_not_called()

    def test_busy_private_next_blocks_body_until_fresh_command_is_processed(self) -> None:
        bridge = GatewayBridge(self.service, Mock())
        threads: list[dict] = []
        def thread(**kwargs: object) -> object:
            threads.append(kwargs)
            return SimpleNamespace(start=lambda: None)
        gateway = SimpleNamespace(adapters={"weixin": object()})
        for _ in range(SETTINGS.worker_limit):
            bridge._slots.acquire()
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                self.assertEqual("secretary-busy", bridge(event("busy-private", "私密：下一条"), gateway)["reason"])
                bridge._slots.release()
                self.assertEqual("secretary-private-protection", bridge(event("secret", "不应发给模型"), gateway)["reason"])
                self.assertEqual("secretary-handled", bridge(event("new-private", "私密：下一条"), gateway)["reason"])
        asyncio.run(submit())
        worker = next(item for item in threads if item["name"].startswith("secretary-") and item["name"] not in {"secretary-busy-reply", "secretary-private-protection"})
        worker["target"]()
        self.assertEqual(set(), bridge._privacy_blocked)
        async def followup() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                bridge(event("safe-followup", "这是私密正文"), gateway)
        asyncio.run(followup())
        threads[-1]["target"]()
        self.private.save.assert_called_once()
        self.classifier.classify.assert_not_called()
        self.media.prepare.assert_not_called()

    def test_private_next_worker_start_failure_blocks_following_body(self) -> None:
        bridge = GatewayBridge(self.service, Mock())
        gateway = SimpleNamespace(adapters={"weixin": object()})
        starts = 0
        def start() -> None:
            nonlocal starts
            starts += 1
            if starts == 1:
                raise RuntimeError("isolated thread failure")
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", return_value=SimpleNamespace(start=start)):
                self.assertEqual("secretary-fail-closed", bridge(event("failed-private", "私密：下一条"), gateway)["reason"])
                self.assertEqual("secretary-private-protection", bridge(event("secret", "不应发给模型"), gateway)["reason"])
        asyncio.run(submit())
        self.assertEqual(SETTINGS.worker_limit, bridge._slots._value)
        self.classifier.classify.assert_not_called()
        self.media.prepare.assert_not_called()

    def test_duplicate_old_private_next_preserves_barrier_without_creating_one(self) -> None:
        incoming = _event_to_message(event("old-private", "私密：下一条"), SETTINGS)
        self.service.handle(incoming)
        self.service.handle(_event_to_message(event("consume", "私密正文"), SETTINGS))
        for was_blocked in (False, True):
            bridge = GatewayBridge(self.service, Mock())
            if was_blocked:
                bridge._block_private(incoming.sender_key)
            queued: list[dict] = []
            def thread(**kwargs: object) -> object:
                queued.append(kwargs)
                return SimpleNamespace(start=lambda: None)
            async def submit() -> None:
                with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                    bridge(event("old-private", "私密：下一条"), SimpleNamespace(adapters={"weixin": object()}))
            asyncio.run(submit())
            queued[0]["target"]()
            self.assertEqual(was_blocked, incoming.sender_key in bridge._privacy_blocked)

    def test_private_latch_storage_failure_blocks_already_queued_body(self) -> None:
        bridge = GatewayBridge(self.service, Mock())
        queued: list[dict] = []
        def thread(**kwargs: object) -> object:
            queued.append(kwargs)
            return SimpleNamespace(start=lambda: None)
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                gateway = SimpleNamespace(adapters={"weixin": object()})
                bridge(event("arm-fails", "私密：下一条"), gateway)
                bridge(event("queued-body", "不应发送给模型"), gateway)
        asyncio.run(submit())
        with patch.object(self.ledger, "arm_private_latch", side_effect=RuntimeError("isolated storage failure")):
            queued[0]["target"]()
        self.classifier.classify.assert_not_called()
        self.media.prepare.assert_not_called()
        self.assertEqual(SETTINGS.worker_limit, bridge._slots._value)

    def test_older_private_success_cannot_clear_a_newer_busy_private_failure(self) -> None:
        bridge = GatewayBridge(self.service, Mock())
        queued: list[dict] = []
        def thread(**kwargs: object) -> object:
            queued.append(kwargs)
            return SimpleNamespace(start=lambda: None)
        gateway = SimpleNamespace(adapters={"weixin": object()})
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                bridge(event("older-private", "私密：下一条"), gateway)
                for _ in range(SETTINGS.worker_limit - 1):
                    bridge._slots.acquire()
                self.assertEqual("secretary-busy", bridge(event("newer-busy-private", "私密：下一条"), gateway)["reason"])
        asyncio.run(submit())
        queued[0]["target"]()
        async def followup() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", side_effect=thread):
                self.assertEqual("secretary-private-protection", bridge(event("private-body", "仍应保护"), gateway)["reason"])
        asyncio.run(followup())
        self.classifier.classify.assert_not_called()

    def test_private_failure_protection_survives_bridge_restart(self) -> None:
        bridge = GatewayBridge(self.service, Mock())
        for _ in range(SETTINGS.worker_limit):
            bridge._slots.acquire()
        async def submit() -> None:
            with patch("wechat_secretary.hermes_plugin.threading.Thread", return_value=SimpleNamespace(start=lambda: None)):
                gateway = SimpleNamespace(adapters={"weixin": object()})
                bridge(event("busy-private", "私密：下一条"), gateway)
                restarted = GatewayBridge(self.service, Mock())
                self.assertEqual("secretary-private-protection", restarted(event("secret-after-restart", "不应发送模型"), gateway)["reason"])
        asyncio.run(submit())
        self.classifier.classify.assert_not_called()

    def test_protection_storage_error_cannot_escape_hook_or_send_body_to_model(self) -> None:
        bridge = GatewayBridge(self.service, Mock())
        for _ in range(SETTINGS.worker_limit):
            bridge._slots.acquire()
        async def submit() -> None:
            with patch.object(self.ledger, "set_private_protection", side_effect=RuntimeError("isolated persistence failure")), patch("wechat_secretary.hermes_plugin.threading.Thread", return_value=SimpleNamespace(start=lambda: None)):
                gateway = SimpleNamespace(adapters={"weixin": object()})
                self.assertEqual("secretary-busy", bridge(event("busy-private", "私密：下一条"), gateway)["reason"])
                bridge._slots.release()
                self.assertEqual("secretary-private-protection", bridge(event("private-body", "不应进入模型"), gateway)["reason"])
        asyncio.run(submit())
        # Once storage recovers, the retained protection is persisted before
        # any later body can be admitted.
        incoming = _event_to_message(event("sender", ""), SETTINGS)
        token = bridge._private_protection_token(incoming.sender_key)
        self.assertTrue(token)
        self.assertEqual(token, self.ledger.get_private_protection(incoming.sender_key))
        self.classifier.classify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
