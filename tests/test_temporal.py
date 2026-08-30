from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from wechat_secretary.classifier import _resolve_date, _resolve_reminder, _resolve_time
from wechat_secretary.completion import parse_named_reminder, parse_relative_reminder
from wechat_secretary.temporal import (
    CLOCK_TOKEN_RE,
    DATE_TOKEN_RE,
    PERIOD_TOKEN_RE,
    RELATIVE_TOKEN_RE,
    resolve_date,
    resolve_datetime,
    resolve_relative_time,
    resolve_time,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 30, 13, 25, tzinfo=TZ)


class TemporalTests(unittest.TestCase):
    def test_next_week_uses_the_next_calendar_week_from_every_current_weekday(self):
        for day in range(24, 31):
            with self.subTest(day=day):
                now = NOW.replace(day=day)
                for text in ("下周二", "下星期二", "下礼拜二"):
                    self.assertEqual(resolve_date(text, now), "2026-09-01")
                    self.assertEqual(_resolve_date(text, now), "2026-09-01")
        self.assertEqual(resolve_date("下周一", NOW.replace(month=12, day=31)), "2027-01-04")

    def test_this_week_does_not_silently_shift_a_past_day_to_next_week(self):
        self.assertEqual(resolve_date("本周二", NOW), "2026-08-25")
        self.assertEqual(resolve_date("这星期二", NOW), "2026-08-25")

    def test_relative_days_and_month_day_variants(self):
        for text, expected in {
            "今天": "2026-08-30", "明天": "2026-08-31", "明日": "2026-08-31",
            "后天": "2026-09-01", "大后天": "2026-09-02", "9月1号": "2026-09-01",
            "9月1日": "2026-09-01", "九月一号": "2026-09-01",
            "十二月三十一日": "2026-12-31", "2026-09-01": "2026-09-01",
            "2026/9/1": "2026-09-01", "2026.9.1": "2026-09-01",
            "2030年9月1号": "2030-09-01", "二〇三〇年九月一日": "2030-09-01",
        }.items():
            with self.subTest(text=text):
                self.assertEqual(resolve_date(text, NOW), expected)
                self.assertEqual(_resolve_date(text, NOW), expected)
        self.assertEqual(resolve_date("大后天", NOW.replace(month=12, day=30)), "2027-01-02")
        self.assertEqual(resolve_date("9月1号", NOW.replace(month=10)), "2026-09-01")

    def test_dates_reject_invalid_or_conflicting_candidates(self):
        for text in (
            "2月30号", "13月1日", "2026-02-29", "2026-13-01", "2026-9-999",
            "20260-09-01", "三十三月一号", "大大后天", "今天或者明天",
            "明天和明天", "2月30号明天", "下周二或下周四", "下午三点", "每周",
            "20300年9月1号", "二千零三十年9月1号",
        ):
            with self.subTest(text=text):
                self.assertEqual(resolve_date(text, NOW), "")

    def test_shared_tokens_preserve_complete_candidates(self):
        for pattern, text in (
            (DATE_TOKEN_RE, "大后天"), (DATE_TOKEN_RE, "2026-99-99"),
            (DATE_TOKEN_RE, "九月一号"), (CLOCK_TOKEN_RE, "晚上十二点"),
            (CLOCK_TOKEN_RE, "下午四点多"), (CLOCK_TOKEN_RE, "25:70"),
            (PERIOD_TOKEN_RE, "下午"), (RELATIVE_TOKEN_RE, "二十分钟后"),
            (RELATIVE_TOKEN_RE, "二十多分钟后"),
            (RELATIVE_TOKEN_RE, "一小时三十分钟后"),
            (RELATIVE_TOKEN_RE, "一个半小时后"),
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(pattern.fullmatch(text))

    def test_a_bare_clock_can_inherit_an_explicit_previous_period(self):
        self.assertEqual(resolve_time("三点", default_period="下午"), "15:00")
        self.assertEqual(_resolve_time("四点半", default_period="下午"), "16:30")
        self.assertEqual(resolve_time("上午九点", default_period="下午"), "09:00")
        self.assertEqual(resolve_time("15:00", default_period="下午"), "15:00")
        self.assertEqual(resolve_time("09:00", default_period="下午"), "09:00")
        self.assertEqual(resolve_time("3:00", default_period="下午"), "03:00")
        self.assertEqual(resolve_time("下午3:00", default_period="上午"), "15:00")
        self.assertEqual(resolve_time("三点"), "03:00")
        self.assertEqual(resolve_time("下午"), "")
        self.assertEqual(resolve_time("三点", default_period="不确定"), "")

    def test_midnight_does_not_become_noon_or_get_a_guessed_day(self):
        for text in (
            "晚上十二点", "夜里12点", "夜间十二点半", "傍晚12点", "午夜十二点",
            "半夜12点", "深夜12点", "晚上零点", "上午十二点", "早上12点",
            "下午零点", "中午零点", "上午15点", "凌晨23点",
        ):
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text), "")
                self.assertIsNone(resolve_datetime(f"明天{text}", NOW))
        self.assertEqual(resolve_time("中午十二点"), "12:00")
        self.assertEqual(resolve_time("凌晨十二点"), "00:00")
        self.assertEqual(resolve_time("凌晨零点"), "00:00")
        self.assertEqual(resolve_time("00:00"), "00:00")

    def test_multiple_clocks_or_periods_do_not_pick_the_first(self):
        for text in ("下午三点或者四点", "15:00和16:00", "下午三点上午", "上午下午3点"):
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text), "")

    def test_chinese_relative_duration_is_shared_by_create_and_adjustment(self):
        for text, delta in {
            "二十分钟后": timedelta(minutes=20),
            "两小时后": timedelta(hours=2),
            "两个小时以后": timedelta(hours=2),
            "半小时后": timedelta(minutes=30),
            "半个小时后": timedelta(minutes=30),
            "半分钟后": timedelta(seconds=30),
            "三十五分钟后": timedelta(minutes=35),
            "120分钟后": timedelta(minutes=120),
        }.items():
            with self.subTest(text=text):
                expected = NOW + delta
                self.assertEqual(resolve_relative_time(text, NOW), expected)
                self.assertEqual(parse_relative_reminder(f"{text}再提醒我", NOW), expected)
                self.assertEqual(
                    datetime.fromisoformat(_resolve_reminder(f"{text}提醒我喝水", NOW, "", "")),
                    expected,
                )

    def test_relative_duration_uses_current_turn_reference_across_midnight(self):
        later = NOW.replace(hour=23, minute=50)
        expected = datetime(2026, 8, 31, 0, 10, tzinfo=TZ)
        self.assertEqual(resolve_relative_time("二十分钟后", later), expected)
        self.assertEqual(parse_relative_reminder("二十分钟后提醒", later), expected)

    def test_relative_duration_does_not_take_a_substring_or_choose_between_options(self):
        for text in (
            "0分钟后", "零小时后", "-20分钟后", "负二十分钟后", "1.5小时后",
            "1000分钟后", "一千分钟后", "大约二十分钟后", "二十多分钟后",
            "半小时左右后", "20分钟后或30分钟后", "明天二十分钟后",
            "三点二十分钟后", "一个半小时后", "一小时三十分钟后", "两小时半后",
        ):
            with self.subTest(text=text):
                self.assertIsNone(resolve_relative_time(text, NOW))
                self.assertIsNone(parse_relative_reminder(f"{text}提醒", NOW))
        for text in ("请问二十分钟后提醒是什么意思", "二十分钟后提醒我喝水", "不要半小时后提醒"):
            self.assertIsNone(parse_relative_reminder(text, NOW))

    def test_named_binding_accepts_the_same_unambiguous_schedule_forms(self):
        for schedule, expected in {
            "明天下午三点": "2026-08-31T15:00:00+08:00",
            "大后天下午四点整": "2026-09-02T16:00:00+08:00",
            "下周二上午九点半": "2026-09-01T09:30:00+08:00",
            "9月1号15:00": "2026-09-01T15:00:00+08:00",
            "九月一号下午三点": "2026-09-01T15:00:00+08:00",
            "2026-09-01T15:00": "2026-09-01T15:00:00+08:00",
            "2026/9/1 15:00": "2026-09-01T15:00:00+08:00",
            "二十分钟后": "2026-08-30T13:45:00+08:00",
            "明天凌晨零点": "2026-08-31T00:00:00+08:00",
        }.items():
            with self.subTest(schedule=schedule):
                result = parse_named_reminder(f"补设提醒：{schedule}｜提交报告", NOW)
                self.assertIsNotNone(result)
                self.assertEqual(result.title, "提交报告")
                self.assertEqual(result.reminder_at.isoformat(), expected)

    def test_named_binding_does_not_ignore_unconsumed_schedule_content(self):
        for schedule in (
            "明天下午", "下午三点", "明天晚上十二点", "明天下午四点多",
            "明天25:00", "明天15:70", "今天或明天下午三点", "明天三点或四点",
            "2026-02-30 15:00", "每周二下午三点", "不是明天下午三点",
            "《明天下午三点》", "明天下午三点和提交报告", "明天二十分钟后",
        ):
            with self.subTest(schedule=schedule):
                self.assertIsNone(parse_named_reminder(f"补设提醒：{schedule}｜提交报告", NOW))
        self.assertIsNone(parse_named_reminder("补设提醒：明天15:00｜提交报告", NOW.replace(tzinfo=None)))


if __name__ == "__main__":
    unittest.main()
