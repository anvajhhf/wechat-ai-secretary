from __future__ import annotations

import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from wechat_secretary.config import SecretarySettings
from wechat_secretary.dida import DidaExecutor
from wechat_secretary.models import (
    ActionResult,
    ExecutionStatus,
    MessageEnvelope,
    TaskDraft,
)


ROOT = Path(__file__).resolve().parents[1]


class DidaFailureClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SecretarySettings(
            project_root=ROOT,
            dry_run=False,
            dida_mapping_confirmed=True,
            dida_schema_confirmed=True,
        )
        self.message = MessageEnvelope(
            platform="weixin",
            account_id="test-account",
            user_id="test-user",
            chat_id="test-chat",
            chat_type="dm",
            message_id="test-message",
            text="待办：提交报告",
            received_at=datetime.fromisoformat("2026-08-26T16:43:00+08:00"),
        )

    def create(self, caller) -> tuple[DidaExecutor, ActionResult]:
        executor = DidaExecutor(self.settings, caller)
        with patch.dict(
            os.environ,
            {"SECRETARY_DIDA_CREATES_APPROVED": "1"},
            clear=False,
        ):
            result = executor.create_task(TaskDraft("提交报告"), self.message)
        return executor, result

    def test_parked_timeout_is_failed_and_explicitly_not_created(self) -> None:
        diagnostic = (
            "MCP server 'dida365' failed initial connection after 3 attempts, "
            "parked: TimeoutError: private-detail"
        )
        executor, result = self.create(
            lambda server, tool, arguments, timeout: {
                "ok": False,
                "error": diagnostic,
            }
        )

        self.assertIs(ExecutionStatus.FAILED, result.status)
        self.assertIn("请求未发出", result.error)
        self.assertIn("任务未创建", result.error)
        self.assertNotIn("private-detail", result.error)
        self.assertEqual("connection_fault", executor.health_summary()["status"])

    def test_plain_timeout_is_uncertain_and_warns_against_retry(self) -> None:
        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, tool, arguments, timeout
            raise TimeoutError("private-timeout-detail")

        executor, result = self.create(caller)

        self.assertIs(ExecutionStatus.UNCERTAIN, result.status)
        self.assertIn("不要重试", result.error)
        self.assertIn("重复创建", result.error)
        self.assertNotIn("private-timeout-detail", result.error)
        self.assertNotIn("TimeoutError", result.error)
        self.assertEqual("result_uncertain", executor.health_summary()["status"])

    def test_explicit_business_rejections_are_failed(self) -> None:
        for diagnostic in (
            "request rejected: private-rejection-detail",
            "invalid projectId: private-validation-detail",
        ):
            with self.subTest(diagnostic=diagnostic):
                executor, result = self.create(
                    lambda server, tool, arguments, timeout, error=diagnostic: {
                        "ok": False,
                        "error": error,
                    }
                )

                self.assertIs(ExecutionStatus.FAILED, result.status)
                self.assertIn("明确拒绝创建请求", result.error)
                self.assertIn("任务未创建", result.error)
                self.assertNotIn("private-", result.error)
                self.assertEqual(
                    "recent_success", executor.health_summary()["status"]
                )

    def test_successful_create_updates_health(self) -> None:
        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, arguments, timeout
            if tool == "create_task":
                return {
                    "ok": True,
                    "result": "created",
                    "structuredContent": {"id": "task-1"},
                }
            return {
                "ok": True,
                "result": "loaded",
                "structuredContent": {
                    "id": "task-1",
                    "title": "提交报告",
                    "projectId": "inbox-project-id",
                    "status": 0,
                },
            }

        executor, result = self.create(caller)
        health = executor.health_summary()

        self.assertIs(ExecutionStatus.SUCCEEDED, result.status)
        self.assertEqual("recent_success", health["status"])
        self.assertIn("成功回读确认", health["summary"])
        self.assertTrue(health["updated_at"])

    def test_unknown_exception_is_sanitized_and_uncertain(self) -> None:
        def caller(server: str, tool: str, arguments: dict, timeout: float) -> dict:
            del server, tool, arguments, timeout
            raise RuntimeError("SECRET_INTERNAL_ENDPOINT_AND_TOKEN")

        _, result = self.create(caller)

        self.assertIs(ExecutionStatus.UNCERTAIN, result.status)
        self.assertIn("不要重试", result.error)
        self.assertNotIn("RuntimeError", result.error)
        self.assertNotIn("SECRET_INTERNAL_ENDPOINT_AND_TOKEN", result.error)


if __name__ == "__main__":
    unittest.main()
