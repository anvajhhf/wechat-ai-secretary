from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from uuid import uuid4

from wechat_secretary.config import SecretarySettings
from wechat_secretary.dida import _failure_kind
from wechat_secretary.ledger import IdempotencyLedger
from wechat_secretary.models import (
    ActionResult,
    ClarificationReason,
    ExecutionStatus,
    IntentKind,
    IntentPlan,
    MessageEnvelope,
    NoteDraft,
    PendingTaskClarification,
    ReminderRecurrence,
    TaskDraft,
    TaskReference,
)
from wechat_secretary.obsidian import ObsidianExecutor
from wechat_secretary.prefixes import parse_prefix
from wechat_secretary.private_inbox import PrivateInboxExecutor
from wechat_secretary.reminders import ReminderQueue
from wechat_secretary.routing import (
    RouteSource,
    detect_route_hint,
    is_non_action_task_utterance,
    normalize_routing_text,
)
from wechat_secretary.semantic_guard import looks_like_pending_followup
from wechat_secretary.service import SecretaryService


ROOT = Path(__file__).resolve().parents[1]
TZ = SecretarySettings(project_root=ROOT).tz
NOW = datetime(2026, 8, 24, 9, 0, tzinfo=TZ)


def make_settings(**changes: object) -> SecretarySettings:
    base = SecretarySettings(
        project_root=ROOT,
        dry_run=True,
        allowed_users=frozenset({"wx-user-1"}),
        account_id="dry-account",
        category_map={"工作": "project-work"},
    )
    return replace(base, **changes)


def make_message(
    message_id: str,
    text: str,
    when: datetime = NOW,
    *,
    user_id: str = "wx-user-1",
    chat_id: str = "chat-1",
) -> MessageEnvelope:
    return MessageEnvelope(
        platform="weixin",
        account_id="dry-account",
        user_id=user_id,
        chat_id=chat_id,
        chat_type="dm",
        message_id=message_id,
        text=text,
        received_at=when,
    )


class StaticClassifier:
    def __init__(self, plan: IntentPlan | Callable[[str], IntentPlan]):
        self.plan = plan
        self.call_count = 0
        self.forced_kinds: list[IntentKind | None] = []

    def classify(
        self,
        incoming: MessageEnvelope,
        content: str,
        forced_kind: IntentKind | None,
        categories: object,
        links: object,
        *,
        deep_note: bool = False,
        image_inputs: object = (),
    ) -> IntentPlan:
        del incoming, categories, links, deep_note, image_inputs
        self.call_count += 1
        self.forced_kinds.append(forced_kind)
        return self.plan(content) if callable(self.plan) else self.plan


class RecordingDida:
    def __init__(self, status: ExecutionStatus = ExecutionStatus.PLANNED):
        self.status = status
        self.create_calls: list[TaskDraft] = []

    def create_task(
        self, task: TaskDraft, incoming: MessageEnvelope
    ) -> ActionResult:
        del incoming
        self.create_calls.append(task)
        if self.status not in {ExecutionStatus.PLANNED, ExecutionStatus.SUCCEEDED}:
            return ActionResult(
                action="task",
                status=self.status,
                summary=task.title,
                destination="Inbox",
                error="synthetic create failure",
            )
        task_id = f"task-{len(self.create_calls)}"
        reference = TaskReference(task_id, task.title, "Inbox", "inbox")
        return ActionResult(
            action="task",
            status=self.status,
            summary=task.title,
            destination="Inbox",
            external_id=task_id,
            task_refs=(reference,),
        )

    @staticmethod
    def health_summary() -> dict[str, str]:
        return {}


def make_service(
    classifier: StaticClassifier,
    *,
    dida: RecordingDida | None = None,
    ledger: IdempotencyLedger | None = None,
    cfg: SecretarySettings | None = None,
) -> tuple[SecretaryService, RecordingDida, IdempotencyLedger]:
    cfg = cfg or make_settings()
    ledger = ledger or IdempotencyLedger(":memory:")
    dida = dida or RecordingDida()
    service = SecretaryService(
        settings=cfg,
        ledger=ledger,
        classifier=classifier,
        dida=dida,
        obsidian=ObsidianExecutor(cfg),
        private_inbox=PrivateInboxExecutor(cfg),
        reminders=ReminderQueue(cfg, ledger),
    )
    return service, dida, ledger


class NormalizationAndPrefixTests(unittest.TestCase):
    def test_normalizes_only_bounded_asr_and_calendar_variants(self) -> None:
        normalized = normalize_routing_text(
            "下个礼拜二提醒我买B2，M抗体", speech=True
        )
        self.assertEqual("下周二提醒我买B2M抗体", normalized.text)
        self.assertEqual(("next-week", "term-b2m"), normalized.changes)

        spoken = normalize_routing_text(
            "代办下周二提醒我买B2，M抗体", speech=True
        )
        self.assertEqual("待办：下周二提醒我买B2M抗体", spoken.text)
        self.assertIn("spoken-task-prefix", spoken.changes)

        literal = normalize_routing_text("代办营业执照", speech=True)
        self.assertEqual("代办营业执照", literal.text)
        self.assertNotIn("spoken-task-prefix", literal.changes)

    def test_prefixes_accept_voice_punctuation_but_keep_word_boundaries(self) -> None:
        for text in ("待办，明天提交报告", "待办 明天提交报告", "任务：明天提交报告"):
            with self.subTest(text=text):
                self.assertIs(parse_prefix(text).forced_kind, IntentKind.TASK)

        self.assertIs(parse_prefix("笔记，记录实验结论").forced_kind, IntentKind.NOTE)
        self.assertIsNone(parse_prefix("待办事项很多").forced_kind)
        self.assertIsNone(parse_prefix("笔记本电脑到了").forced_kind)
        self.assertFalse(parse_prefix("私密，下一条").private)
        self.assertIsNone(parse_prefix("代办营业执照", speech=True).forced_kind)
        self.assertIs(
            parse_prefix("代办下周二提醒我买抗体", speech=True).forced_kind,
            IntentKind.TASK,
        )

    def test_natural_route_hints_are_conservative(self) -> None:
        task = detect_route_hint("明天下午3点提醒我回电话")
        note = detect_route_hint("帮我记一下，B2M适合流式实验")
        statement = detect_route_hint("B2M抗体今天到了")
        negated = detect_route_hint("不用提醒我B2M抗体到了")

        self.assertIs(task.kind, IntentKind.TASK)
        self.assertIs(task.source, RouteSource.NATURAL)
        self.assertIs(note.kind, IntentKind.NOTE)
        self.assertIs(note.source, RouteSource.NATURAL)
        self.assertIsNone(statement.kind)
        self.assertIsNone(negated.kind)

    def test_questions_statuses_and_technical_terms_do_not_force_task(self) -> None:
        for text in (
            "记得我们第一次见面吗？",
            "记得我们第一次见面",
            "别忘了我们的约定",
            "安排得很好",
            "帮我安排得很好",
            "创建一个任务队列需要什么数据结构？",
            "明天开会需要准备什么？",
            "我已经安排好明天开会了",
            "为什么明天3点提醒我买牛奶？",
            "你会在明天3点提醒我买牛奶吗？",
            "系统会在明天3点提醒我买牛奶",
            "谢谢你刚才提醒我买牛奶",
            "导师会在明天3点提醒我交报告",
            "老板明天3点会提醒我交报告",
            "别担心，妈妈明天3点会提醒我买牛奶",
            "导师明天提交报告",
            "公司明天提交财报",
            "他下周联系客户",
            "我明天提交报告",
            "无需再提醒我明天3点买牛奶",
            "不必再提醒我买牛奶",
            "不需要通知我明天3点买牛奶",
            "请勿再提醒我买牛奶",
            "取消明天3点提醒我买牛奶",
            "撤销明天3点通知我交报告",
            "停止再提醒我买牛奶",
            "关闭明天3点提醒我买牛奶",
            "删除明天3点提醒我买牛奶",
            "移除明天3点提醒我买牛奶",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_route_hint(text).kind)

        for text in (
            "记得整理实验材料",
            "安排下周二开会",
            "请创建一个任务，整理实验报告",
            "明天下午3点提醒我回电话，可以吗？",
            "你能在明天3点提醒我买牛奶吗？",
            "麻烦明天3点提醒我买牛奶，好吗？",
            "明天提交报告",
            "请导师明天提交报告",
            "记得明天提交报告",
            "不要忘记明天3点提醒我买牛奶",
        ):
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)

    def test_pending_followup_accepts_only_pure_scheduling_fields(self) -> None:
        for text in ("每周二", "每星期二", "每周二上午9点", "上午9点", "共3次"):
            with self.subTest(text=text):
                self.assertTrue(looks_like_pending_followup(text))
        for text in ("你上午9点有空吗", "上午9点会议怎么样", "周二天气怎么样"):
            with self.subTest(text=text):
                self.assertFalse(looks_like_pending_followup(text))

    def test_mixed_connectors_are_not_mistaken_for_third_party_subjects(self) -> None:
        for connector in ("另外", "同时", "并且", "还有", "顺便"):
            with self.subTest(connector=connector, form="elliptical-command"):
                text = f"帮我记一下正文，{connector}明天提交任务A"
                self.assertFalse(is_non_action_task_utterance(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.MIXED)

            with self.subTest(connector=connector, form="third-party-statement"):
                text = f"帮我记一下正文，{connector}导师明天提交任务A"
                self.assertTrue(is_non_action_task_utterance(text))
                self.assertIs(detect_route_hint(text).kind, IntentKind.NOTE)


class NaturalServiceRoutingTests(unittest.TestCase):
    def test_non_commands_are_not_passed_as_forced_task_intents(self) -> None:
        for index, text in enumerate(
            (
                "记得我们第一次见面吗？",
                "安排得很好",
                "创建一个任务队列需要什么数据结构？",
                "为什么明天3点提醒我买牛奶？",
                "你会在明天3点提醒我买牛奶吗？",
                "系统会在明天3点提醒我买牛奶",
                "谢谢你刚才提醒我买牛奶",
            )
        ):
            with self.subTest(text=text):
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.CLARIFY,
                        confidence=0.2,
                        clarification="请说明想让我做什么。",
                        clarification_reason=ClarificationReason.AMBIGUOUS_INTENT,
                    )
                )
                service, dida, ledger = make_service(classifier)
                self.addCleanup(ledger.close)

                result = service.handle(make_message(f"non-command-{index}", text))

                self.assertIsNone(classifier.forced_kinds[0])
                self.assertEqual(0, len(dida.create_calls))
                self.assertEqual((), result.results)

    def test_adversarial_task_plan_cannot_turn_statements_into_writes(self) -> None:
        cases = (
            ("导师会在明天3点提醒我交报告", "交报告", True),
            ("老板明天3点会提醒我交报告", "交报告", True),
            ("别担心，妈妈明天3点会提醒我买牛奶", "买牛奶", True),
            ("导师明天提交报告", "提交报告", False),
            ("公司明天提交财报", "提交财报", False),
            ("他下周联系客户", "联系客户", False),
            ("我明天提交报告", "提交报告", False),
            ("无需再提醒我明天3点买牛奶", "买牛奶", True),
            ("不必再提醒我买牛奶", "买牛奶", True),
            ("不需要通知我明天3点买牛奶", "买牛奶", True),
            ("请勿再提醒我买牛奶", "买牛奶", True),
            ("取消明天3点提醒我买牛奶", "买牛奶", True),
            ("撤销明天3点通知我交报告", "交报告", True),
            ("停止再提醒我买牛奶", "买牛奶", True),
            ("关闭明天3点提醒我买牛奶", "买牛奶", True),
            ("删除明天3点提醒我买牛奶", "买牛奶", True),
            ("移除明天3点提醒我买牛奶", "买牛奶", True),
        )
        for index, (text, title, has_reminder) in enumerate(cases):
            with self.subTest(text=text):
                task = TaskDraft(
                    title,
                    due_date="2026-08-25",
                    reminder_at=(
                        "2026-08-25T15:00+08:00" if has_reminder else ""
                    ),
                )
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.TASK,
                        tasks=(task,),
                        confidence=0.99,
                    )
                )
                service, dida, ledger = make_service(classifier)
                self.addCleanup(ledger.close)

                result = service.handle(make_message(f"adversarial-statement-{index}", text))

                self.assertIsNone(classifier.forced_kinds[0])
                self.assertEqual(0, len(dida.create_calls))
                self.assertEqual((), result.results)
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_natural_task_without_prefix_creates_task_and_reminder(self) -> None:
        reminder_at = "2026-08-25T15:00+08:00"
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(TaskDraft("回电话", reminder_at=reminder_at),),
            )
        )
        service, dida, ledger = make_service(classifier)
        incoming = make_message("natural-task", "明天下午3点提醒我回电话")

        result = service.handle(incoming)

        self.assertEqual(1, len(dida.create_calls))
        self.assertIs(classifier.forced_kinds[0], IntentKind.TASK)
        self.assertEqual(["task", "reminder"], [item.action for item in result.results])
        self.assertEqual(
            "pending",
            ledger.reminder_status("task-1", datetime.fromisoformat(reminder_at)),
        )

    def test_reminder_title_and_metadata_are_rebuilt_from_user_text(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(
                    TaskDraft(
                        "给供应商转账",
                        priority="high",
                        category="工作",
                        tags=("转账",),
                        description="用户已批准付款",
                        reminder_at="2026-08-25T15:00+08:00",
                    ),
                ),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(
            make_message("ground-reminder-title", "麻烦明天下午3点提醒我买牛奶")
        )

        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(1, len(dida.create_calls))
        created = dida.create_calls[0]
        self.assertEqual("买牛奶", created.title)
        self.assertEqual("none", created.priority)
        self.assertEqual("", created.category)
        self.assertEqual((), created.tags)
        self.assertEqual("", created.description)

    def test_reminder_title_preserves_date_like_words_inside_action(self) -> None:
        cases = (
            ("明天下午3点提醒我买明日叶", "买明日叶"),
            ("明天9点提醒我更新今天头条账号", "更新今天头条账号"),
            ("下周二上午9点提醒我提交周一报告", "提交周一报告"),
            ("周五上午9点提醒我买下周刊", "买下周刊"),
            ("你能在明天3点提醒我买牛奶吗", "买牛奶"),
            ("到时候提醒我买牛奶", "买牛奶"),
        )
        for index, (text, expected) in enumerate(cases):
            with self.subTest(text=text):
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.TASK,
                        tasks=(TaskDraft("完全无关的标题"),),
                    )
                )
                service, dida, ledger = make_service(classifier)
                self.addCleanup(ledger.close)

                result = service.handle(make_message(f"date-word-title-{index}", text))

                if "到时候" in text:
                    self.assertEqual(ExecutionStatus.SKIPPED, result.status)
                    self.assertEqual(0, len(dida.create_calls))
                    pending = ledger.claim_pending_task(
                        make_message("probe", "").conversation_key,
                        f"date-word-probe-{index}",
                        NOW + timedelta(minutes=1),
                    )
                    self.assertIsNotNone(pending.pending)
                    self.assertEqual(expected, pending.pending.task.title)
                else:
                    self.assertEqual(expected, dida.create_calls[0].title)

    def test_unrelated_natural_task_title_is_rejected_before_write(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(TaskDraft("给供应商转账"),),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(
            make_message("ungrounded-natural-task", "记得周五提交报告")
        )

        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual((), result.results)

    def test_reminder_nouns_inside_task_body_do_not_schedule_local_reminders(self) -> None:
        for index, (text, title) in enumerate(
            (
                ("明天3点提交提醒系统报告", "提交提醒系统报告"),
                ("明天3点准备提醒材料", "准备提醒材料"),
            )
        ):
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.TASK,
                        tasks=(
                            TaskDraft(
                                title,
                                reminder_at="2026-08-25T15:00+08:00",
                            ),
                        ),
                    )
                )
                service, dida, ledger = make_service(classifier)
                self.addCleanup(ledger.close)

                result = service.handle(
                    make_message(f"reminder-noun-{index}", text)
                )

                self.assertEqual(1, len(dida.create_calls))
                created = dida.create_calls[0]
                self.assertEqual(title, created.title)
                self.assertEqual("2026-08-25", created.due_date)
                self.assertEqual("03:00", created.due_time)
                self.assertEqual("", created.reminder_at)
                self.assertEqual(("task",), tuple(item.action for item in result.results))

    def test_quoted_reminder_words_stay_inside_task_title(self) -> None:
        cases = (
            ("明天3点提交《别忘了》观后感", "提交《别忘了》观后感"),
            ("明天3点提交“通知我”功能设计", "提交“通知我”功能设计"),
            ('明天3点提交"提醒我"功能设计', '提交"提醒我"功能设计'),
            ("明天3点提交'通知我'交互稿", "提交'通知我'交互稿"),
        )
        for index, (text, title) in enumerate(cases):
            with self.subTest(text=text):
                self.assertIs(detect_route_hint(text).kind, IntentKind.TASK)
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.TASK,
                        tasks=(
                            TaskDraft(
                                title,
                                reminder_at="2026-08-25T15:00+08:00",
                            ),
                        ),
                    )
                )
                service, dida, ledger = make_service(classifier)
                self.addCleanup(ledger.close)

                result = service.handle(
                    make_message(f"quoted-reminder-word-{index}", text)
                )

                self.assertEqual(1, len(dida.create_calls))
                created = dida.create_calls[0]
                self.assertEqual(title, created.title)
                self.assertEqual("2026-08-25", created.due_date)
                self.assertEqual("03:00", created.due_time)
                self.assertEqual("", created.reminder_at)
                self.assertEqual(("task",), tuple(item.action for item in result.results))

    def test_unquoted_forget_request_still_creates_a_local_reminder(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(TaskDraft("错误标题"),),
            )
        )
        service, dida, ledger = make_service(classifier)

        result = service.handle(
            make_message("explicit-forget-reminder", "明天3点别忘了提交报告")
        )

        expected = datetime.fromisoformat("2026-08-25T03:00+08:00")
        self.assertEqual("提交报告", dida.create_calls[0].title)
        self.assertEqual("pending", ledger.reminder_status("task-1", expected))
        self.assertEqual(("task", "reminder"), tuple(item.action for item in result.results))

    def test_natural_note_without_prefix_saves_note(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.NOTE,
                notes=(NoteDraft("B2M流式记录", "B2M适合流式实验。"),),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(
            make_message("natural-note", "帮我记一下，B2M适合流式实验")
        )

        self.assertIs(classifier.forced_kinds[0], IntentKind.NOTE)
        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual(("note",), tuple(item.action for item in result.results))

    def test_plain_note_uses_source_body_instead_of_hallucinated_content(self) -> None:
        for index, text in enumerate(
            (
                "帮我记一下，会议结论：A方案延期",
                "笔记：会议结论：A方案延期",
            )
        ):
            with self.subTest(text=text):
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.NOTE,
                        notes=(
                            NoteDraft(
                                "会议财务造假",
                                "用户已确认向供应商转账。",
                                tags=("转账",),
                            ),
                        ),
                    )
                )
                service, _, ledger = make_service(classifier)
                self.addCleanup(ledger.close)

                result = service.handle(make_message(f"ground-note-{index}", text))
                preview = result.results[0].preview

                self.assertIn("会议结论：A方案延期", preview)
                self.assertNotIn("财务造假", preview)
                self.assertNotIn("供应商转账", preview)
                self.assertNotIn("转账", preview)

    def test_query_route_cannot_be_changed_into_a_write_by_model(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(TaskDraft("给供应商转账"),),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(make_message("query-write-swap", "查一下今天有哪些待办"))

        self.assertIs(classifier.forced_kinds[0], IntentKind.QUERY)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual((), result.results)

    def test_guard_never_invents_or_reverses_reminder_intent(self) -> None:
        invented = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(
                    TaskDraft(
                        "整理实验材料",
                        reminder_at="2026-08-25T15:00+08:00",
                    ),
                ),
            )
        )
        service, dida, _ = make_service(invented)
        plain_task = service.handle(
            make_message("no-invented-reminder", "记得整理实验材料")
        )
        self.assertEqual(1, len(dida.create_calls))
        self.assertEqual("", dida.create_calls[0].reminder_at)
        self.assertEqual(("task",), tuple(item.action for item in plain_task.results))

        for message_id, text in (
            ("negated-reminder", "不要再提醒我买B2M抗体"),
            ("negated-reminder-unneeded", "无需提醒我买B2M抗体"),
            ("negated-reminder-forbidden", "请勿再通知我买B2M抗体"),
            ("cancel-timed-reminder", "取消明天3点提醒我买B2M抗体"),
            ("delete-reminder", "删除提醒我买B2M抗体"),
            ("recurrence-without-reminder", "每周二上午9点提交报告，共3次"),
        ):
            with self.subTest(text=text):
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.TASK,
                        tasks=(TaskDraft("买B2M抗体"),),
                    )
                )
                guarded, guarded_dida, _ = make_service(classifier)
                result = guarded.handle(make_message(message_id, text))
                self.assertEqual(0, len(guarded_dida.create_calls))
                self.assertEqual((), result.results)

    def test_multi_task_plan_rejects_any_item_not_grounded_in_source(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(
                    TaskDraft("提交报告"),
                    TaskDraft(
                        "给供应商转账",
                        due_date="2026-08-25",
                        reminder_at="2026-08-25T15:00+08:00",
                    ),
                ),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(
            make_message("ungrounded-second-task", "待办：提交报告；整理资料")
        )

        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual((), result.results)
        self.assertIn("逐项确认", result.reply)

    def test_grounded_timeless_batch_sanitizes_every_task_before_write(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(
                    TaskDraft(
                        "买B2M抗体",
                        due_date="2026-08-25",
                        due_time="15:00",
                        reminder_at="2026-08-25T15:00+08:00",
                    ),
                    TaskDraft(
                        "提交报告",
                        due_date="2026-08-26",
                        due_time="09:00",
                        reminder_at="2026-08-26T09:00+08:00",
                        reminder_recurrence=ReminderRecurrence(
                            frequency="weekly",
                            interval=1,
                            weekday=3,
                            count=3,
                        ),
                    ),
                ),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(
            make_message("grounded-task-batch", "待办：买B2M抗体；提交报告")
        )

        self.assertEqual(2, len(dida.create_calls))
        self.assertEqual(("买B2M抗体", "提交报告"), tuple(task.title for task in dida.create_calls))
        for task in dida.create_calls:
            self.assertEqual("", task.due_date)
            self.assertEqual("", task.due_time)
            self.assertEqual("", task.reminder_at)
            self.assertIsNone(task.reminder_recurrence)
        self.assertEqual(("task", "task"), tuple(item.action for item in result.results))

    def test_timed_multi_task_sentence_fails_closed_before_any_write(self) -> None:
        reminder_at = "2026-08-25T15:00+08:00"
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(
                    TaskDraft("提交报告", reminder_at=reminder_at),
                    TaskDraft("买B2M抗体", reminder_at=reminder_at),
                ),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(
            make_message(
                "timed-multi-task",
                "明天下午3点提醒我提交报告；另外买B2M抗体",
            )
        )

        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual((), result.results)
        self.assertIn("每项单独发送", result.reply)

    def test_plain_statement_is_not_saved_as_a_note_by_default(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.CLARIFY,
                confidence=0.1,
                clarification="请说明想让我做什么。",
                clarification_reason=ClarificationReason.AMBIGUOUS_INTENT,
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(make_message("plain-statement", "B2M抗体今天到了"))

        self.assertIsNone(classifier.forced_kinds[0])
        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual((), result.results)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_user_omitting_reminder_time_must_not_write(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(TaskDraft("回电话"),),
            )
        )
        service, dida, _ = make_service(classifier)

        result = service.handle(
            make_message("user-missing-time", "明天提醒我回电话")
        )

        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual((), result.results)
        self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_local_guard_rebuilds_exact_user_time_instead_of_model_guess(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(
                    TaskDraft(
                        "回电话",
                        reminder_at="2026-08-25T16:00+08:00",
                    ),
                ),
            )
        )
        service, dida, ledger = make_service(classifier)

        service.handle(
            make_message("correct-model-time", "明天下午3点提醒我回电话")
        )

        expected = datetime.fromisoformat("2026-08-25T15:00+08:00")
        guessed = datetime.fromisoformat("2026-08-25T16:00+08:00")
        self.assertEqual(1, len(dida.create_calls))
        self.assertEqual(
            expected,
            datetime.fromisoformat(dida.create_calls[0].reminder_at),
        )
        self.assertEqual("pending", ledger.reminder_status("task-1", expected))
        self.assertIsNone(ledger.reminder_status("task-1", guessed))

    def test_missing_time_followup_creates_exactly_once(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(TaskDraft("买B2M抗体"),),
            )
        )
        service, dida, ledger = make_service(classifier)
        first = make_message("missing-time", "下周二提醒我买B2M抗体")

        clarification = service.handle(first)
        self.assertEqual(ExecutionStatus.SKIPPED, clarification.status)
        self.assertIn("几点", clarification.reply)
        self.assertEqual(0, len(dida.create_calls))

        followup = make_message("supply-time", "上午9点", NOW + timedelta(minutes=1))
        created = service.handle(followup)
        duplicate = service.handle(followup)

        expected = datetime.fromisoformat("2026-09-01T09:00+08:00")
        self.assertEqual(1, len(dida.create_calls))
        self.assertEqual("2026-09-01T09:00+08:00", dida.create_calls[0].reminder_at)
        self.assertEqual("pending", ledger.reminder_status("task-1", expected))
        self.assertTrue(duplicate.duplicate)
        self.assertIn("reminder", tuple(item.action for item in created.results))


class PendingTaskIsolationTests(unittest.TestCase):
    @staticmethod
    def _pending() -> PendingTaskClarification:
        return PendingTaskClarification(
            reason=ClarificationReason.MISSING_REMINDER_TIME,
            task=TaskDraft("买B2M抗体"),
            reminder_date="2026-09-01",
            source_message_id="source",
        )

    def test_cancel_clears_pending_without_creating(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(kind=IntentKind.TASK, tasks=(TaskDraft("买B2M抗体"),))
        )
        service, dida, ledger = make_service(classifier)
        first = make_message("cancel-source", "下周二提醒我买B2M抗体")
        service.handle(first)

        cancelled = service.handle(
            make_message("cancel-followup", "取消", NOW + timedelta(minutes=1))
        )
        remaining = ledger.claim_pending_task(
            first.conversation_key, "probe", NOW + timedelta(minutes=2)
        )

        self.assertEqual(0, len(dida.create_calls))
        self.assertIn("已取消", cancelled.reply)
        self.assertIsNone(remaining.pending)
        self.assertEqual("none", remaining.state)

    def test_expired_pending_is_deleted(self) -> None:
        ledger = IdempotencyLedger(":memory:")
        self.addCleanup(ledger.close)
        incoming = make_message("expired-source", "")
        ledger.set_pending_task(
            incoming.conversation_key,
            self._pending(),
            NOW - timedelta(seconds=1),
        )

        expired = ledger.claim_pending_task(incoming.conversation_key, "reply", NOW)
        missing = ledger.claim_pending_task(incoming.conversation_key, "again", NOW)

        self.assertEqual("expired", expired.state)
        self.assertIsNone(expired.pending)
        self.assertEqual("none", missing.state)

    def test_pending_context_is_isolated_by_chat(self) -> None:
        def classify(content: str) -> IntentPlan:
            if "提醒我" in content:
                return IntentPlan(
                    kind=IntentKind.TASK,
                    tasks=(TaskDraft("买B2M抗体"),),
                )
            return IntentPlan(
                kind=IntentKind.CLARIFY,
                confidence=0.1,
                clarification="请说明想让我做什么。",
                clarification_reason=ClarificationReason.AMBIGUOUS_INTENT,
            )

        classifier = StaticClassifier(classify)
        service, dida, ledger = make_service(classifier)
        source = make_message("chat-source", "下周二提醒我买B2M抗体", chat_id="chat-a")
        service.handle(source)

        other_chat = service.handle(
            make_message(
                "chat-b-reply",
                "上午9点",
                NOW + timedelta(minutes=1),
                chat_id="chat-b",
            )
        )
        original = ledger.claim_pending_task(
            source.conversation_key, "chat-a-probe", NOW + timedelta(minutes=2)
        )

        self.assertEqual(0, len(dida.create_calls))
        self.assertEqual((), other_chat.results)
        self.assertIsNotNone(original.pending)
        self.assertEqual("claimed", original.state)

    def test_unrelated_time_questions_do_not_consume_pending_task(self) -> None:
        questions = (
            "你上午9点有空吗",
            "上午9点会议怎么样",
            "为什么是上午9点",
            "周二天气怎么样",
        )
        for index, question in enumerate(questions):
            with self.subTest(question=question):
                def classify(content: str) -> IntentPlan:
                    if "提醒我" in content:
                        return IntentPlan(
                            kind=IntentKind.TASK,
                            tasks=(TaskDraft("买B2M抗体"),),
                        )
                    return IntentPlan(
                        kind=IntentKind.CLARIFY,
                        confidence=0.1,
                        clarification="请说明想让我做什么。",
                        clarification_reason=ClarificationReason.AMBIGUOUS_INTENT,
                    )

                service, dida, ledger = make_service(StaticClassifier(classify))
                self.addCleanup(ledger.close)
                source = make_message(
                    f"question-source-{index}", "下周二提醒我买B2M抗体"
                )
                service.handle(source)

                result = service.handle(
                    make_message(
                        f"unrelated-question-{index}",
                        question,
                        NOW + timedelta(minutes=1),
                    )
                )
                pending = ledger.claim_pending_task(
                    source.conversation_key,
                    f"question-probe-{index}",
                    NOW + timedelta(minutes=2),
                )

                self.assertEqual(0, len(dida.create_calls))
                self.assertEqual((), result.results)
                self.assertIsNotNone(pending.pending)
                self.assertEqual("买B2M抗体", pending.pending.task.title)

    def test_only_one_concurrent_claim_gets_the_pending_draft(self) -> None:
        db_path = ROOT / f".natural-routing-{uuid4().hex}.sqlite3"
        first = IdempotencyLedger(db_path)
        second = IdempotencyLedger(db_path)
        try:
            conversation_key = make_message("concurrent-source", "").conversation_key
            first.set_pending_task(
                conversation_key,
                self._pending(),
                NOW + timedelta(minutes=10),
            )
            barrier = threading.Barrier(2)

            def claim(ledger: IdempotencyLedger, message_id: str) -> tuple[str, object]:
                barrier.wait(timeout=5)
                return message_id, ledger.claim_pending_task(
                    conversation_key, message_id, NOW
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = tuple(
                    pool.map(
                        lambda item: claim(*item),
                        ((first, "reply-a"), (second, "reply-b")),
                    )
                )

            owners = [
                message_id
                for message_id, outcome in outcomes
                if outcome.pending is not None
            ]
            self.assertEqual(1, len(owners))
            self.assertTrue(first.release_pending_task(conversation_key, owners[0]))
        finally:
            first.close()
            second.close()
            for suffix in ("", "-shm", "-wal"):
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)


class RecurringReminderTests(unittest.TestCase):
    @staticmethod
    def _weekly_plan(count: int = 3) -> IntentPlan:
        first = "2026-08-25T09:00+08:00"
        return IntentPlan(
            kind=IntentKind.TASK,
            tasks=(
                TaskDraft(
                    "买B2M抗体",
                    reminder_at=first,
                    reminder_recurrence=ReminderRecurrence(
                        frequency="weekly",
                        interval=1,
                        weekday=2,
                        count=count,
                    ),
                ),
            ),
        )

    def test_weekly_tuesday_nine_three_times_expands_locally(self) -> None:
        classifier = StaticClassifier(self._weekly_plan())
        service, dida, ledger = make_service(classifier)
        incoming = make_message(
            "weekly-three",
            "每周二上午9点提醒我买B2M抗体，共3次",
        )

        result = service.handle(incoming)

        expected = tuple(
            datetime.fromisoformat(value)
            for value in (
                "2026-08-25T09:00+08:00",
                "2026-09-01T09:00+08:00",
                "2026-09-08T09:00+08:00",
            )
        )
        self.assertEqual(1, len(dida.create_calls))
        self.assertEqual(3, ledger.active_reminder_count("task-1", incoming))
        self.assertTrue(
            all(ledger.reminder_status("task-1", item) == "pending" for item in expected)
        )
        self.assertEqual(["task", "reminder"], [item.action for item in result.results])

    def test_bare_weekly_prompt_keeps_time_count_and_accepts_plain_weekday(self) -> None:
        for index, wording in enumerate(("每周", "每周都")):
            with self.subTest(wording=wording):
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.TASK,
                        tasks=(TaskDraft("无关的模型标题"),),
                    )
                )
                service, dida, ledger = make_service(classifier)
                self.addCleanup(ledger.close)
                source = make_message(
                    f"bare-weekly-{index}",
                    f"{wording}上午9点提醒我买牛奶，共3次",
                )

                first = service.handle(source)
                completed = service.handle(
                    make_message(
                        f"bare-weekly-weekday-{index}",
                        "周二",
                        NOW + timedelta(minutes=1),
                    )
                )

                self.assertEqual(ExecutionStatus.SKIPPED, first.status)
                self.assertIn("星期几", first.reply)
                self.assertEqual(1, classifier.call_count)
                self.assertEqual(1, len(dida.create_calls))
                self.assertEqual("买牛奶", dida.create_calls[0].title)
                self.assertEqual(3, ledger.active_reminder_count("task-1", source))
                self.assertEqual(["task", "reminder"], [item.action for item in completed.results])

    def test_zero_count_followup_never_reuses_previous_valid_count(self) -> None:
        classifier = StaticClassifier(
            IntentPlan(
                kind=IntentKind.TASK,
                tasks=(TaskDraft("模型标题"),),
            )
        )
        service, dida, _ = make_service(classifier)
        service.handle(
            make_message(
                "weekly-count-source",
                "每周二提醒我买牛奶，共3次",
            )
        )

        result = service.handle(
            make_message(
                "weekly-zero-count",
                "上午9点，共0次",
                NOW + timedelta(minutes=1),
            )
        )

        self.assertEqual(ExecutionStatus.SKIPPED, result.status)
        self.assertEqual(0, len(dida.create_calls))
        self.assertIn("2到52", result.reply)

    def test_missing_count_and_unbounded_repeat_make_zero_writes(self) -> None:
        cases = (
            ("missing-count", "每周二上午9点提醒我买B2M抗体"),
            ("unbounded", "每周二上午9点一直提醒我买B2M抗体"),
        )
        for message_id, text in cases:
            with self.subTest(text=text):
                classifier = StaticClassifier(
                    IntentPlan(
                        kind=IntentKind.TASK,
                        tasks=(TaskDraft("买B2M抗体"),),
                    )
                )
                service, dida, ledger = make_service(classifier)
                self.addCleanup(ledger.close)

                result = service.handle(make_message(message_id, text))

                self.assertEqual(0, len(dida.create_calls))
                self.assertEqual((), result.results)
                self.assertEqual(ExecutionStatus.SKIPPED, result.status)

    def test_relative_reschedule_refuses_a_multi_occurrence_series(self) -> None:
        classifier = StaticClassifier(self._weekly_plan())
        service, dida, ledger = make_service(classifier)
        source = make_message(
            "weekly-for-reschedule",
            "每周二上午9点提醒我买B2M抗体，共3次",
        )
        service.handle(source)

        adjusted = service.handle(
            make_message(
                "weekly-relative-adjust",
                "半小时后提醒",
                NOW + timedelta(minutes=1),
            )
        )

        self.assertEqual(ExecutionStatus.SKIPPED, adjusted.status)
        self.assertIn("重复提醒", adjusted.reply)
        self.assertEqual(1, len(dida.create_calls))
        self.assertEqual(3, ledger.active_reminder_count("task-1", source))
        self.assertIsNone(
            ledger.reminder_status("task-1", NOW + timedelta(minutes=31))
        )


class DidaFailureClassificationTests(unittest.TestCase):
    def test_write_failure_kinds_separate_pre_send_uncertain_and_rejection(self) -> None:
        self.assertEqual(
            "connection_before_send",
            _failure_kind(
                RuntimeError(
                    "dida365 failed initial connection after 3 attempts, "
                    "parked: TimeoutError"
                ),
                raised=True,
            ),
        )
        self.assertEqual(
            "uncertain",
            _failure_kind(TimeoutError("read timed out"), raised=True),
        )
        self.assertEqual(
            "business_rejection",
            _failure_kind({"message": "validation failed: invalid task"}),
        )


if __name__ == "__main__":
    unittest.main()
