from __future__ import annotations

import unittest

from wechat_secretary.classifier import CLOCK_TOKEN_RE, _resolve_time


class ChineseClockParserTests(unittest.TestCase):
    def test_exact_chinese_and_arabic_clocks(self):
        cases = {
            "今天下午三点的时候，提醒我让ChatGPT优化本地的生信技能": "15:00",
            "今天下午四点的时候提醒我让ChatGPT优化本地技能": "16:00",
            "今天下午4点的时候提醒我": "16:00",
            "三点半": "03:30",
            "下午三点半": "15:30",
            "凌晨零点": "00:00",
            "下午两点": "14:00",
            "中午十二点": "12:00",
            "二十三点": "23:00",
            "二十三点五十九分": "23:59",
            "下午三点零五分": "15:05",
            "下午三点〇五分": "15:05",
            "下午三点五分": "15:05",
            "下午三点二十分": "15:20",
            "下午三点一刻": "15:15",
            "下午三点三刻": "15:45",
            "下午3点1刻": "15:15",
            "下午3点3刻": "15:45",
            "15:00": "15:00",
            "15：05": "15:05",
            "15:5": "15:05",
            "十五：五十九": "15:59",
            "3点05分": "03:05",
            "3点05": "03:05",
            "09时30分": "09:30",
            "上午9点钟": "09:00",
            "早上9点整": "09:00",
            "晚上8点": "20:00",
            "下午 3 点 30 分": "15:30",
            "今天下午３点提醒我": "15:00",
            "每周二09:00提醒我买牛奶，共3次": "09:00",
            "每周二下午9点提醒我买牛奶，共3次": "21:00",
            "星期二下午三点": "15:00",
            "每礼拜三三点半": "03:30",
            "时间：下午三点": "15:00",
            "时间:15:00": "15:00",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_resolve_time(text), expected)

    def test_vague_clocks_are_not_promoted_to_exact(self):
        for text in (
            "4点多", "四点多的时候", "三点左右", "三点半左右", "大约四点",
            "大概下午四点", "下午大约四点", "约四点", "差不多四点", "四点前后",
            "四点上下", "四点出头", "四点来钟", "四点过一点", "三点二十多分",
            "三点二十来分", "下午4点钟左右", "大约在下午四点", "下午大约在四点",
        ):
            with self.subTest(text=text):
                self.assertEqual(_resolve_time(text), "")

    def test_invalid_clocks_are_not_partially_matched(self):
        for text in (
            "25:00", "15:70", "24点", "125:00", "15:700", "二十五点",
            "二十四点", "三点六十分", "23:60", "123点", "123点45分",
            "15:", "15：", "15:00:30", "三点二十分三十秒", "三点五刻",
            "3.5点", "三点六十", "3点5.5分", "15:00.5", "三百点",
            "15:70，随后16:00", "24点后是1点", "二三点", "10点600分", "3:4点",
        ):
            with self.subTest(text=text):
                self.assertEqual(_resolve_time(text), "")

    def test_non_clock_text_is_not_a_clock(self):
        for text in (
            "3小时", "三小时以后", "15分钟", "三十分钟后提醒我", "三天后",
            "今天下午", "提醒我优化生信技能", "B2点位", "3时间段",
        ):
            with self.subTest(text=text):
                self.assertEqual(_resolve_time(text), "")

    def test_candidate_spans_include_vague_and_invalid_values(self):
        for token in (
            "下午三点半", "下午四点多", "大约四点", "三点左右", "25:00",
            "15:70", "24点", "二十四点", "三点六十分", "三点二十多分",
        ):
            with self.subTest(token=token):
                text = f"今天{token}的时候提醒我买牛奶"
                match = CLOCK_TOKEN_RE.search(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.group(), token)


if __name__ == "__main__":
    unittest.main()
