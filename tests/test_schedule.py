"""Offline contracts and integration for read-only arrangement summaries."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from copy import deepcopy
import os
import unittest
from unittest.mock import Mock, patch

from test_notes_conversation import NotesServiceFixture
from wechat_secretary.config import SecretarySettings
from wechat_secretary.dida import DidaExecutor
from wechat_secretary.media import PreparedMedia
from wechat_secretary.models import ExecutionStatus, ReminderRecurrence, TaskReference
from wechat_secretary.schedule import (ScheduleItem, ScheduleQuery, ScheduleSource, format_schedule,
                                      local_schedule, parse_dida_schedule, parse_schedule_query)

TZ = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 26, 12, tzinfo=TZ)


def task(id="t1", project="p1", title="合成待办", due="2026-10-27T07:00:00Z", **changes):
    value = {"id": id, "projectId": project, "title": title, "status": 0, "dueDate": due, "isAllDay": False, "kind": "TASK"}
    value.update(changes)
    return value


def row(id="r1", title="合成提醒", at="2026-10-27T08:30:00+08:00", status="pending", **changes):
    value = {"task_id": id, "project_id": "local", "title": title, "reminder_at": at, "status": status,
             "recurrence_frequency": "", "recurrence_interval": 0, "recurrence_active": 0}
    value.update(changes)
    return value


class ScheduleParsingTests(unittest.TestCase):
    def test_screenshot_question_is_this_calendar_week(self):
        q = parse_schedule_query("最近有什么要提醒我的事情吗？本周要做的。", NOW)
        self.assertEqual(datetime(2026, 10, 26, tzinfo=TZ), q.start)
        self.assertEqual(datetime(2026, 11, 2, tzinfo=TZ), q.end)

    def test_colloquial_read_phrases(self):
        for text in ("今天有什么事", "这周有哪些提醒", "帮我看看本周要做的事情", "本周安排", "本周未完成的任务", "查一下明天待办", "最近有什么要提醒我的事情吗？", "安排查询：下周", "未来七天有什么安排"):
            with self.subTest(text=text):
                result = parse_schedule_query(text, NOW)
                self.assertIsNotNone(result)
                self.assertFalse(result.error)

    def test_calendar_week_differs_from_next_seven_days(self):
        sunday = NOW + timedelta(days=6)
        week = parse_schedule_query("本周安排", sunday)
        seven = parse_schedule_query("未来7天安排", sunday)
        self.assertEqual(6, (seven.start - week.start).days)
        self.assertEqual(7, (seven.end - seven.start).days)

    def test_year_and_month_boundary_and_tomorrow(self):
        now = datetime(2026, 12, 31, 23, 59, tzinfo=TZ)
        tomorrow = parse_schedule_query("明天有哪些任务", now)
        self.assertEqual("2027-01-01", tomorrow.start.date().isoformat())
        self.assertEqual("2027-01-02", tomorrow.end.date().isoformat())

    def test_all_and_overdue_have_explicit_modes(self):
        self.assertEqual("all", parse_schedule_query("有没有没做完的", NOW).mode)
        self.assertEqual("all", parse_schedule_query("全部未完成待办", NOW).mode)
        self.assertEqual("overdue", parse_schedule_query("查一下逾期任务", NOW).mode)

    def test_explicit_dates_and_invalid_or_conflicting_ranges(self):
        q = parse_schedule_query("查安排：2026-10-26至2026-10-28", NOW)
        self.assertEqual(3, (q.end - q.start).days)
        for text in ("今天明天有哪些安排", "本周逾期任务", "安排查询：2026-13-01", "安排查询：2026-10-29至2026-10-26", "安排查询：2026-01-01至2026-12-31", "安排查询：", "安排查询：乱猜时间"):
            with self.subTest(text=text):
                self.assertTrue(parse_schedule_query(text, NOW).error)

    def test_quoted_discussion_negated_and_write_requests_are_not_reads(self):
        for text in ("他说‘本周有什么安排’", "“本周有哪些任务”", "为什么任务查询失败", "不要查询本周待办", "明天提醒我喝水", "提醒我明天", "明天提醒我", "帮我记一下本周的安排", "本周有哪些任务，顺便明天提醒我买牛奶", "安排得不错", "怎么设计查询待办的接口"):
            with self.subTest(text=text):
                self.assertIsNone(parse_schedule_query(text, NOW))

    def test_plain_statements_are_not_queries(self):
        for text in ("今天我有事", "我今天有任务", "我明天还有事情", "今天的事情做了", "本周我有安排", "什么事"):
            self.assertIsNone(parse_schedule_query(text, NOW), text)
        self.assertEqual("all", parse_schedule_query("全部待办", NOW).mode)


class ScheduleRemoteTests(unittest.TestCase):
    def parse(self, payload):
        return parse_dida_schedule({"ok": True, "result": payload}, TZ)

    def test_verified_nested_result_and_timezone(self):
        result = self.parse({"result": [task()]})
        self.assertTrue(result.available)
        self.assertFalse(result.incomplete)
        self.assertEqual(15, result.items[0].at.hour)

    def test_completed_and_notes_excluded_even_when_server_returns_them(self):
        result = self.parse([task(status=2), task(kind="NOTE"), task(completedTime="yes"), task(id="active")])
        self.assertEqual(["active"], [i.task_id for i in result.items])

    def test_live_dida_text_kind_is_an_ordinary_task(self):
        result = self.parse({"result": [task(kind="TEXT")]})
        self.assertFalse(result.incomplete)
        self.assertEqual(1, len(result.items))

    def test_undated_and_start_only_are_not_invented_deadlines(self):
        result = self.parse([task(due=None), task(id="start", due=None, startDate="2026-10-27T08:00:00+08:00")])
        self.assertEqual(["undated", "start"], [i.time_kind for i in result.items])

    def test_bad_time_status_or_identity_is_partial_not_empty_success(self):
        for value in (task(due="bad"), task(due="2026-10-27T08:00:00"), task(status=99), task(status=False), task(id=""), task(project=""), task(isAllDay=None)):
            with self.subTest(value=value):
                result = self.parse([value])
                self.assertTrue(result.incomplete)
                self.assertEqual((), result.items)

    def test_unrecognized_or_paginated_payload_is_not_a_successful_empty_list(self):
        for value in ("No tasks", None, {}, {"error": "unauthorized"}, {"result": [], "hasMore": True}, {"tasks": [], "total": 10}, {"project": {"id": "p", "name": "project"}}):
            with self.subTest(value=value):
                self.assertFalse(self.parse(value).available)

    def test_legitimate_empty_list_is_complete(self):
        self.assertEqual(ScheduleSource(), self.parse({"result": []}))

    def test_duplicate_ids_collapse_and_over_limit_is_visible(self):
        self.assertEqual(1, len(self.parse([task(), task()]).items))
        result = parse_dida_schedule({"ok": True, "result": [task(id=str(i)) for i in range(4)]}, TZ, limit=3)
        self.assertTrue(result.incomplete)
        self.assertEqual(3, len(result.items))

    def test_conflicting_duplicates_are_removed_instead_of_choosing_a_date(self):
        result = self.parse([task(), task(due="2026-11-10T08:00:00Z"), task()])
        self.assertTrue(result.incomplete)
        self.assertEqual((), result.items)

    def test_failure_envelope_cannot_masquerade_as_empty_data(self):
        self.assertFalse(self.parse({"result": [], "success": False}).available)
        self.assertFalse(self.parse({"result": [], "error": "upstream-failure"}).available)

    def test_invalid_ids_are_not_truncated_into_merge_candidates(self):
        result = self.parse([task(id="a" * 301), task(project=" p1")])
        self.assertTrue(result.incomplete)
        self.assertEqual((), result.items)

    def test_exact_remote_contract_one_read_no_write_or_model(self):
        calls = []
        def caller(server, tool, args, timeout):
            calls.append((tool, args, timeout))
            return {"ok": True, "structuredContent": {"result": [task()]}, "result": "untrusted prose"}
        settings = SecretarySettings(project_root=__import__('pathlib').Path('.'), dry_run=False)
        result = DidaExecutor(settings, caller).schedule_tasks()
        self.assertEqual(1, len(result.items))
        self.assertEqual([("filter_tasks", {"filter": {"status": [0]}, "client_timezone": "Asia/Shanghai"}, 20)], calls)

    def test_remote_connection_failure_and_dry_run_are_explicit(self):
        settings = SecretarySettings(project_root=__import__('pathlib').Path('.'), dry_run=False)
        self.assertFalse(DidaExecutor(settings, Mock(side_effect=TimeoutError())).schedule_tasks().available)
        caller = Mock()
        self.assertFalse(DidaExecutor(replace(settings, dry_run=True), caller).schedule_tasks().available)
        caller.assert_not_called()


class ScheduleMergeTests(unittest.TestCase):
    def test_daily_future_projection_is_read_only_and_explicit(self):
        q = parse_schedule_query("本周安排", NOW)
        rows = [row(recurrence_frequency="daily", recurrence_interval=1, recurrence_active=1)]
        before = deepcopy(rows)
        result = local_schedule(rows, q, NOW)
        self.assertEqual(6, len(result.items))
        self.assertFalse(result.items[0].projected)
        self.assertTrue(all(i.projected for i in result.items[1:]))
        self.assertEqual(before, rows)
        self.assertIn("按每日规则预计", format_schedule(q, NOW, result, ScheduleSource()))

    def test_inactive_series_and_cancelled_rows_do_not_project(self):
        q = parse_schedule_query("本周安排", NOW)
        result = local_schedule([row(status="cancelled", recurrence_active=1, recurrence_frequency="daily", recurrence_interval=1), row(status="rescheduled"), row(id="once")], q, NOW)
        self.assertEqual(1, len(result.items))

    def test_real_occurrence_wins_over_same_projected_day(self):
        q = parse_schedule_query("本周安排", NOW)
        rows = [row(recurrence_frequency="daily", recurrence_interval=1, recurrence_active=1), row(at="2026-10-28T08:30:00+08:00")]
        result = local_schedule(rows, q, NOW)
        day = [i for i in result.items if i.at.day == 28]
        self.assertEqual(1, len(day))
        self.assertFalse(day[0].projected)

    def test_local_invalid_date_and_limit_are_disclosed(self):
        q = parse_schedule_query("本周安排", NOW)
        self.assertTrue(local_schedule([row(at="invalid")], q, NOW).incomplete)
        self.assertTrue(local_schedule([row(id="one"), row(id="two")], q, NOW, limit=1).incomplete)

    def test_display_limit_is_explicit(self):
        q = parse_schedule_query("本周安排", NOW)
        remote = parse_dida_schedule({"ok": True, "result": [task(id=str(i)) for i in range(15)]}, TZ)
        text = format_schedule(q, NOW, ScheduleSource(), remote)
        self.assertEqual(12, text.count("• "))
        self.assertIn("另有 3 项未展开", text)

    def test_exact_identity_merges_but_same_title_different_ids_does_not(self):
        q = parse_schedule_query("本周安排", NOW)
        remote = parse_dida_schedule({"ok": True, "result": [task()]}, TZ)
        local = local_schedule([row(id="t1", project_id="p1", title="合成待办"), row(id="t2", title="合成待办")], q, NOW)
        text = format_schedule(q, NOW, local, remote)
        self.assertEqual(2, text.count("• 合成待办"))
        self.assertIn("截止 10-27 15:00", text)
        self.assertIn("微信提醒 10-27 08:30", text)

    def test_reminder_inside_window_keeps_linked_outside_deadline_labeled(self):
        q = parse_schedule_query("本周安排", NOW)
        remote = parse_dida_schedule({"ok": True, "result": [task(due="2026-11-10T08:00:00+08:00")]}, TZ)
        local = local_schedule([row(id="t1", project_id="p1")], q, NOW)
        text = format_schedule(q, NOW, local, remote)
        self.assertIn("截止 11-10", text)
        self.assertEqual(1, text.count("• "))

    def test_sent_uncertain_failed_labels_never_assert_completion(self):
        q = parse_schedule_query("今天安排", NOW)
        rows = [row(id=status, at="2026-10-26T08:00:00+08:00", status=status) for status in ("sent", "uncertain", "failed")]
        text = format_schedule(q, NOW, local_schedule(rows, q, NOW), ScheduleSource())
        self.assertIn("已发送", text)
        self.assertIn("送达未确认", text)
        self.assertIn("发送失败待重试", text)
        self.assertIn("不代表事情已完成", text)

    def test_all_day_overdue_only_after_local_day_end(self):
        remote = parse_dida_schedule({"ok": True, "result": [task(due="2026-10-25T16:00:00Z", isAllDay=True)]}, TZ)
        q = ScheduleQuery(mode="overdue", label="逾期")
        self.assertNotIn("• ", format_schedule(q, NOW, ScheduleSource(), remote))
        self.assertIn("• ", format_schedule(q, NOW + timedelta(days=1), ScheduleSource(), remote))

    def test_partial_source_never_reports_no_arrangements(self):
        q = parse_schedule_query("今天安排", NOW)
        text = format_schedule(q, NOW, ScheduleSource(), ScheduleSource(available=False, problem="连接失败"))
        self.assertIn("结果不完整", text)
        self.assertIn("已成功查询", text)
        self.assertNotIn("没有安排", text)

    def test_start_boundary_inclusive_end_exclusive_and_undated_disclosure(self):
        q = parse_schedule_query("今天安排", NOW)
        remote = parse_dida_schedule({"ok": True, "result": [task(id="start", due=q.start.isoformat()), task(id="end", due=q.end.isoformat()), task(id="none", due=None)]}, TZ)
        text = format_schedule(q, NOW, ScheduleSource(), remote)
        self.assertEqual(1, text.count("• "))
        self.assertIn("1 项滴答未完成待办没有日期", text)

    def test_linked_undated_task_is_not_counted_again_as_unlisted(self):
        q = parse_schedule_query("本周安排", NOW)
        remote = parse_dida_schedule({"ok": True, "result": [task(due=None)]}, TZ)
        local = local_schedule([row(id="t1", project_id="p1")], q, NOW)
        text = format_schedule(q, NOW, local, remote)
        self.assertIn("未设日期", text)
        self.assertNotIn("另有 1 项", text)


class ScheduleServiceTests(NotesServiceFixture):
    def make(self, **kwargs):
        self.now = NOW
        service, index, answer, chat, memory, store, dida, private, notify = self.make_notes_service(schedule_query_enabled=True, memory_enabled=True, **kwargs)
        dida.schedule_tasks = Mock(return_value=parse_dida_schedule({"ok": True, "result": [task()]}, TZ))
        return service, index, answer, chat, memory, store, dida, private, notify

    def seed_reminder(self, service, *, incoming=None, id="local", status="pending"):
        incoming = incoming or self.message(service, "synthetic")
        service.ledger.enqueue_reminders(incoming, TaskReference(id, "合成微信事项", project_id="local"), (NOW + timedelta(days=1),))
        with service.ledger._lock:
            service.ledger._connection.execute("UPDATE reminders SET status=? WHERE task_id=?", (status, id))
        return incoming

    def test_combined_summary_reads_both_and_preserves_all_business_data(self):
        service, index, answer, chat, memory, store, dida, _, _ = self.make()
        incoming = self.seed_reminder(service)
        before = service.ledger.reminder_snapshot("local", incoming)
        _, result = self.exchange(service, "最近有什么要提醒我的事情吗？本周要做的。")
        self.assertEqual(ExecutionStatus.SUCCEEDED, result.status)
        self.assertIn("合成微信事项", result.reply)
        self.assertIn("合成待办", result.reply)
        self.assertFalse(result.llm_called)
        self.assertEqual(0, chat.call_count + answer.call_count + service.classifier.call_count)
        self.assertEqual([], index.search_calls)
        self.assertEqual(before, service.ledger.reminder_snapshot("local", incoming))
        self.assertEqual([], store.history(incoming.conversation_key, 10, 12000))
        self.assertEqual((), memory.list_entries(incoming.conversation_key))
        self.assertEqual([], dida.created + dida.completed)

    def test_local_route_scope_rejects_other_user_account_chat_and_profile(self):
        service, *_ = self.make()
        incoming = self.seed_reminder(service)
        self.assertEqual(1, len(service.ledger.schedule_reminders(incoming)))
        for field in ("platform", "account_id", "user_id", "chat_id"):
            self.assertEqual((), service.ledger.schedule_reminders(replace(incoming, **{field: "different"})))
        other, *_ = self.make(profile="partner")
        self.assertEqual((), other.ledger.schedule_reminders(incoming))

    def test_dida_failure_preserves_local_results_and_is_terminal_per_message(self):
        service, _, _, chat, _, _, dida, _, _ = self.make()
        self.seed_reminder(service)
        dida.schedule_tasks.side_effect = OSError("synthetic")
        message = self.message(service, "本周安排")
        result = service.handle(message)
        self.assertEqual(ExecutionStatus.PARTIAL, result.status)
        self.assertIn("合成微信事项", result.reply)
        self.assertTrue(service.handle(message).duplicate)
        self.assertEqual(1, dida.schedule_tasks.call_count)
        self.assertEqual(0, chat.call_count)

    def test_local_failure_still_shows_dida_and_both_failures_are_explicit(self):
        service, _, _, _, _, _, dida, _, _ = self.make()
        with patch.object(service.ledger, "schedule_reminders", side_effect=OSError("synthetic")):
            _, result = self.exchange(service, "本周安排")
            self.assertEqual(ExecutionStatus.PARTIAL, result.status)
            self.assertIn("合成待办", result.reply)
            dida.schedule_tasks.side_effect = TimeoutError()
            _, result = self.exchange(service, "本周安排")
            self.assertEqual(ExecutionStatus.FAILED, result.status)
            self.assertIn("结果不完整", result.reply)

    def test_private_chat_temporary_note_memory_and_reminder_keep_priority(self):
        for text in ("私密：本周有什么安排", "聊天：本周有什么安排", "临时聊天：本周有什么安排", "笔记：本周有什么安排", "记住偏好：喜欢周一看安排", "明天早上八点提醒我喝水", "待办：查询本周安排"):
            with self.subTest(text=text):
                service, _, _, _, _, _, dida, _, _ = self.make()
                self.exchange(service, text)
                dida.schedule_tasks.assert_not_called()

    def test_spoken_query_uses_existing_asr_but_no_classifier_or_chat(self):
        media = Mock()
        media.prepare.return_value = PreparedMedia(transcript_text="最近有什么要提醒我的事情吗？本周要做的。")
        service, _, _, chat, _, _, dida, _, _ = self.make(media=media)
        _, result = self.exchange(service, "", media_paths=("synthetic.wav",), media_types=("audio",))
        self.assertIn("合成待办", result.reply)
        dida.schedule_tasks.assert_called_once()
        self.assertEqual(0, chat.call_count + service.classifier.call_count)

    def test_query_clears_old_numbered_task_and_note_focus(self):
        service, _, answer, _, _, _, dida, _, _ = self.make()
        self.exchange(service, "待办：旧的合成任务")
        self.search(service)
        self.exchange(service, "本周安排")
        self.exchange(service, "第二条完成了")
        self.exchange(service, "笔记问答：总结这些方案")
        self.assertEqual([], dida.completed)
        self.assertEqual(0, answer.call_count)

    def test_invalid_range_and_disabled_dry_run_do_not_query_sources(self):
        service, _, _, _, _, _, dida, _, _ = self.make()
        for text in ("安排查询：", "今天明天有哪些安排", "安排查询：2026-13-01"):
            self.exchange(service, text)
        dida.schedule_tasks.assert_not_called()
        service.settings = replace(service.settings, schedule_query_enabled=False)
        self.exchange(service, "安排查询：本周")
        dida.schedule_tasks.assert_not_called()
        service.settings = replace(service.settings, schedule_query_enabled=True, dry_run=True)
        self.exchange(service, "本周安排")
        dida.schedule_tasks.assert_not_called()

    def test_summaries_do_not_contaminate_later_chat_or_memory(self):
        service, _, _, chat, _, _, _, _, _ = self.make()
        self.exchange(service, "本周安排")
        self.exchange(service, "聊天：聊一下散步")
        self.assertNotIn("合成待办", repr(chat.calls))

    def test_private_latch_and_unauthorized_sender_do_not_read_schedules(self):
        service, _, _, _, _, _, dida, private, _ = self.make()
        self.exchange(service, "私密：下一条")
        self.exchange(service, "本周安排")
        self.assertEqual(1, len(private.saved))
        self.exchange(service, "本周安排", user_id="unknown")
        dida.schedule_tasks.assert_not_called()

    def test_query_failure_does_not_break_next_note_reminder_or_memory_command(self):
        service, _, _, _, memory, _, dida, _, notify = self.make()
        dida.schedule_tasks.side_effect = TimeoutError()
        self.exchange(service, "本周安排")
        _, note = self.exchange(service, "笔记：合成查询回归笔记")
        self.assertEqual(ExecutionStatus.SUCCEEDED, note.status)
        notify.assert_called_once()
        incoming, reminder = self.exchange(service, "明天早上八点提醒我喝水")
        self.assertEqual(ExecutionStatus.SUCCEEDED, reminder.status)
        self.assertTrue(service.ledger.schedule_reminders(incoming))
        self.exchange(service, "记住偏好：回答尽量简洁")
        self.assertTrue(memory.list_entries(incoming.conversation_key))

    def test_disabled_natural_task_query_keeps_existing_path(self):
        service, _, _, _, _, _, dida, _, _ = self.make()
        service.settings = replace(service.settings, schedule_query_enabled=False)
        self.exchange(service, "查一下今天待办")
        self.assertEqual(1, len(dida.queries))
        dida.schedule_tasks.assert_not_called()

    def test_config_option_does_not_change_existing_features(self):
        path = self.root / "schedule-test.toml"
        path.write_text('[schedule_query]\nenabled=true\n[chat]\nenabled=true\n[memory]\nenabled=true\n[notes_search]\nenabled=true\nallowed_paths=["Inbox"]\n[reminders]\nenabled=true\n', encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            settings = SecretarySettings.from_file(path, self.root)
        self.assertTrue(settings.schedule_query_enabled and settings.chat_enabled and settings.memory_enabled and settings.notes_search_enabled and settings.reminders_enabled)
        self.assertFalse(SecretarySettings(project_root=self.root).schedule_query_enabled)

    def test_image_query_does_not_read_arrangement_sources(self):
        media = Mock()
        media.prepare.return_value = PreparedMedia(images=(object(),))
        service, _, _, _, _, _, dida, _, _ = self.make(media=media)
        _, result = self.exchange(service, "本周安排", media_paths=("synthetic.png",), media_types=("image",))
        self.assertIn("图片", result.reply)
        dida.schedule_tasks.assert_not_called()


if __name__ == '__main__':
    unittest.main()
