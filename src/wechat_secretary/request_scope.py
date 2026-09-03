from __future__ import annotations

import re

from .temporal import CLOCK_TOKEN_RE, DATE_TOKEN_RE, PERIOD_TOKEN_RE, RELATIVE_TOKEN_RE


_QUOTED_TEXT = re.compile(
    r"《[^》]*(?:》|$)|“[^”]*(?:”|$)|‘[^’]*(?:’|$)|"
    r"「[^」]*(?:」|$)|『[^』]*(?:』|$)|\"[^\"]*(?:\"|$)|'[^']*(?:'|$)"
)
_POLITE_PREFIX = r"(?:(?:请(?!问)|麻烦|劳驾|拜托)(?:你)?|帮(?:一下)?我)"
_NOTE_VERB = (
    r"(?:记(?:录)?(?:一下|下来|下)|记个笔记|记(?:一条|一个|一篇)笔记|"
    r"保存(?:一下|下来|这(?:段|条|个)?)|存下来|"
    r"存成(?:一篇|一个|一条)?笔记|整理成(?:一篇|一个|一条)?笔记|做个笔记)"
)
_NOTE_REQUEST = re.compile(
    rf"^\s*(?:{_POLITE_PREFIX}\s*)*"
    rf"(?:{_NOTE_VERB}|把这(?:段|条|个)?(?:内容|话|想法|结论|记录)?\s*{_NOTE_VERB})"
    r"(?!来|的|了|过|着)"
    r"\s*[：:,，]?\s*"
)
_NOTE_STATEMENT = re.compile(
    r"^\s*(?:这是|以下是)?\s*(?:实验|会议|项目)?(?:结论|纪要|记录)\s*[：:,，]\s*"
)
_NOTE_META_SUFFIX = re.compile(
    r"^\s*(?:了吗|了么|了没有|过吗|过么|是什么(?:意思)?|是什么意思|怎么用|如何使用)"
    r"\s*[?？。.]?\s*$"
)

# A match retains offsets into the original utterance. The action's payload
# starts after the whole marker, including the optional polite “一下”.
REMINDER_REQUEST_RE = re.compile(
    r"(?:到时候\s*)?(?:提醒|通知|叫)\s*(?:一下\s*)?我(?:\s*一下)?"
)
_MEMORY_CUE_RE = re.compile(r"别忘(?:了)?")
_OUTER_REMINDER_PREFIX = re.compile(
    # ASR punctuation does not change a pure scheduling prefix's intent. Keep
    # the entire prefix on this whitelist: never skip preceding prose/quotes.
    # Question marks intentionally remain outside this grammar.
    rf"^(?:\s|[，,、：:。.;；!！]|{_POLITE_PREFIX}|"
    r"记得|别忘了|不要忘记|务必|一定|"
    r"我想(?:请|让)你|(?:你\s*)?(?:能否|可否|能不能|可不可以|可以|能)|"
    r"在|到|于|的时候|时候|时|都|固定|"
    r"今天|今日|今早|今晨|今晚|今夜|明天|明日|明早|明晨|明晚|明夜|后天|大后天|"
    r"(?:本|这|下|每)(?:个)?(?:周|星期|礼拜)[一二三四五六日天]?|"
    r"(?:周|星期|礼拜)[一二三四五六日天]|每(?:天|日|月)|"
    r"(?:每(?:个)?)?(?:工作日|周末)|隔周|每两周|每(?:季度|年)|每个月|"
    r"(?:到|至|和|及|、)\s*(?:(?:周|星期|礼拜))?[一二三四五六日天]|"
    r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]|"
    r"凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里|夜间|"
    r"(?:半|\d{1,3})\s*(?:分钟|小时)后)*$"
)
_REMINDER_BODY_META_QUESTION = re.compile(
    r"^\s*[：:,，]?\s*(?:什么|啥|(?:是)?谁|哪些|哪个|为什么|为何|怎么|如何|"
    r"什么时候|何时|几点|是否|是不是|会不会|这个功能|这项功能|是什么意思)"
)
_REMINDER_STATUS_QUESTION = re.compile(
    r"(?:了吗|了么|了没有|过吗|过么|是真的吗|是什么意思|是什么情况|怎么回事)"
    r"\s*[?？。.]?\s*$|"
    r"(?:这件事|这个提醒|这条提醒|这项提醒).{0,40}"
    r"(?:为什么|为何|怎么|如何|有没有|是否|成功|失败|取消|删除)"
)
_REMINDER_TRAILING_QUESTION = re.compile(r"[吗么]\s*[?？。.]?\s*$")
_EXPLICIT_QUESTION_PAYLOAD = re.compile(
    r"^\s*[：:,，]?\s*(?:问(?:一下|问)?|询问|"
    r"确认(?!\s*(?:了吗|了么|了没有|过吗|过么))|向[^，,。；;！？!?]{1,12}询问)"
)
_POLITE_REQUEST_LEAD = re.compile(
    rf"^\s*(?:{_POLITE_PREFIX}|(?:你\s*)?(?:能否|可否|能不能|可不可以|能|可以))"
)
_POLITE_REQUEST_END = re.compile(
    r"(?:可以|行|好|没问题)(?:吗|么)?\s*[?？]?\s*$"
)
_MODEL_FALLBACK_DIRECT_PREFIX = re.compile(
    rf"^\s*(?:(?:{_POLITE_PREFIX}|记得|别忘(?:了)?|不要忘记|务必|一定|"
    r"我想(?:请|让)你|(?:你\s*)?(?:能否|可否|能不能|可不可以|可以|能))"
    r"\s*[，,、：:。.;；!！]*)*$"
)
_MODEL_FALLBACK_TEMPORAL_PREFIX = re.compile(
    rf"^\s*(?:(?:{_POLITE_PREFIX}|记得|别忘(?:了)?|不要忘记|务必|一定|"
    r"我想(?:请|让)你|(?:你\s*)?(?:能否|可否|能不能|可不可以|可以|能))"
    r"\s*)*(?:从|自|在|到|于|等到|到了)?\s*"
    r"(?:今天|今日|今早|今晨|今晚|今夜|明天|明日|明早|明晨|明晚|明夜|后天|大后天|每天|每日|每逢|每个|"
    r"本周|这周|下周|下个(?:周|星期|礼拜)|(?:周|星期|礼拜)[一二三四五六日天]|"
    r"工作日|周末|凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里|夜间|"
    r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}\s*月\s*\d{1,2}\s*[日号]|"
    r"\d{1,2}\s*(?:[:：.]\s*\d{1,2}|点|时)|(?:半|\d{1,3})\s*(?:分钟|小时)后)"
)
_MODEL_FALLBACK_REPORTED_PREFIX = re.compile(
    r"(?:^|[，,。；;：:\s])(?:转发|示例|例如|原话|他说|她说|他们说|她们说|"
    r"系统|助手|机器人|导师|老师|同事|朋友|家人)"
    r"(?:会|将|说|表示|答应|负责|已经|刚才|刚刚|此前|曾经)?"
)
_MODEL_FALLBACK_QUESTION_PREFIX = re.compile(
    r"(?:为什么|为何|怎么|如何|是否|是不是|会不会|有没有|什么时候|何时|几点|请问|我想知道)"
)
_REMINDER_STOP_INTENT = re.compile(
    r"(?:(?:不要|不用|无需|不必|不需要|请勿|别)"
    r"(?!\s*忘)[^，,。；;！？!?\n]{0,30}?(?:提醒|通知|叫)(?:\s*一下)?(?:\s*我)?|"
    r"(?:取消|撤销|停止|关闭|删除|移除)"
    r"[^，,。；;！？!?\n]{0,20}?(?:提醒|通知|叫)(?:\s*一下)?(?:\s*我)?)"
)
_RETRACTION_CLAUSE = re.compile(
    r"[，,。；;！？!?\n]\s*(?:(?:算了|还是|现在|请|麻烦|我)[，,\s]*)*"
)
_DIRECT_RETRACTION = re.compile(
    r"(?:不用|不要|不必|无需|不需要|别)(?:再)?(?:提醒|通知|叫)(?:一下)?(?:我)?(?:了|啦)?"
    r"\s*[。.!！]?\s*$"
)
_CALL_ACTION = re.compile(
    r"^(?:去|来|起床|出发|开会|参加|买|购买|提交|联系|发送|准备|整理|"
    r"预约|缴费|续费|回复|完成|取|打|回|看|吃|接|带|查|问|确认|取消)"
)
_TRAILING_REMINDER_END = re.compile(
    r"^\s*[，,]?\s*(?:吧|呢|啊|呀|可以吗|好吗|行吗|没问题吗)?\s*[。.!！?？]?\s*$"
)
INDEPENDENT_REQUEST_CONNECTOR_RE = re.compile(
    r"[，,。；;\n]\s*(?:另外|同时|并且|还有|顺便)\s*"
)


def mask_quoted_text(text: str) -> str:
    """Mask titles/quotes without changing original character offsets."""

    return _QUOTED_TEXT.sub(lambda match: " " * len(match.group()), text or "")


def note_request_match(text: str) -> re.Match[str] | None:
    """Find an outer note-saving request, never an instruction inside a quote."""

    candidate = mask_quoted_text(text)
    match = _NOTE_REQUEST.match(text) or _NOTE_STATEMENT.match(text)
    if match is None or _NOTE_META_SUFFIX.fullmatch(candidate[match.end():]):
        return None
    return match


def note_source_body(text: str) -> str:
    match = note_request_match(text)
    # “会议结论：” is data framing, not a command prefix to delete. Only
    # remove the actual save verb (“记下来：”), retaining factual headings.
    if match is not None and _NOTE_REQUEST.match(text) is None:
        match = None
    return (text[match.end():] if match else text).strip(" \t\r\n，,：:")


def reminder_marker(text: str) -> re.Match[str] | None:
    """Find a lexical reminder marker outside quoted or note-owned content.

    This locates scheduling fields; it is not authorization to create a task.
    Bare “别忘了” is retained for the existing explicit/elliptical task path.
    """

    candidate = mask_quoted_text(text)
    note = note_request_match(text)
    if note is not None:
        for connector in INDEPENDENT_REQUEST_CONNECTOR_RE.finditer(candidate, note.end()):
            tail = text[connector.end():]
            outer = outer_reminder_match(tail)
            if outer is not None and not outer_reminder_is_question(tail, outer):
                return REMINDER_REQUEST_RE.match(candidate, connector.end() + outer.start())
        return None
    return REMINDER_REQUEST_RE.search(candidate) or _MEMORY_CUE_RE.search(candidate)


def is_explicit_reminder_candidate(text: str) -> bool:
    """Allow one constrained model fallback for a direct reminder request.

    The strict parser remains the zero-token path.  This fallback covers
    schedule wording that the parser does not yet understand (for example
    ``从明天开始每天……``), but still requires an unquoted first-person
    reminder marker and rejects reports, status questions and retractions.
    It authorizes interpretation only; semantic grounding still decides
    whether any task can be written.
    """

    if not text or note_request_match(text) is not None:
        return False
    candidate = mask_quoted_text(text)
    marker = REMINDER_REQUEST_RE.search(candidate)
    if marker is None or has_negated_reminder(text):
        return False

    prefix = candidate[:marker.start()]
    payload = candidate[marker.end():]
    if (
        _MODEL_FALLBACK_REPORTED_PREFIX.search(prefix)
        or _MODEL_FALLBACK_QUESTION_PREFIX.search(prefix)
        or _REMINDER_BODY_META_QUESTION.search(payload)
        or _REMINDER_STATUS_QUESTION.search(payload)
    ):
        return False
    if (
        _REMINDER_TRAILING_QUESTION.search(payload)
        and not _POLITE_REQUEST_LEAD.search(candidate)
        and not _POLITE_REQUEST_END.search(payload)
    ):
        return False

    return bool(
        _MODEL_FALLBACK_DIRECT_PREFIX.fullmatch(prefix)
        or _MODEL_FALLBACK_TEMPORAL_PREFIX.match(prefix)
    )


def _schedule_prefix_matches(prefix: str) -> bool:
    # Shared tokens consume whole Chinese/Arabic and vague candidates. This
    # identifies the request's scope, not whether its time is valid or exact.
    remainder = RELATIVE_TOKEN_RE.sub(" ", prefix)
    remainder = CLOCK_TOKEN_RE.sub(" ", remainder)
    return bool(_OUTER_REMINDER_PREFIX.fullmatch(remainder))


def _front_loaded_reminder_prefix(text: str, match: re.Match[str]) -> bool:
    """Recognize an action followed by a separately addressed timed reminder.

    Do not accept arbitrary prose before a reminder mention: require an action
    lead, a clause boundary, a schedule-only suffix and no new payload after the
    marker. Quotes, reported subjects and status questions remain non-actions.
    """

    if not _TRAILING_REMINDER_END.fullmatch(text[match.end():]):
        return False
    prefix = text[:match.start()]
    boundaries = list(re.finditer(r"[，,。;；!！\n]|(?<!\d)\.|\.(?!\d)", prefix))
    if not boundaries:
        return False
    boundary = boundaries[-1]
    body = prefix[:boundary.start()].strip()
    schedule = prefix[boundary.end():]
    # Multiple standalone action sentences must not collapse into one task.
    # This expanded ASR-punctuation tolerance only joins one body to its time.
    if re.search(r"[。;；!！\n]|(?<!\d)\.|\.(?!\d)", body):
        return False
    if not _schedule_prefix_matches(schedule) or not any(
        pattern.search(schedule)
        for pattern in (CLOCK_TOKEN_RE, DATE_TOKEN_RE, PERIOD_TOKEN_RE, RELATIVE_TOKEN_RE)
    ):
        return False
    body = re.sub(rf"^(?:{_POLITE_PREFIX}\s*)*", "", body)
    action = _CALL_ACTION.match(body)
    if action is None or body[action.end():].startswith(("了", "过", "着")):
        return False
    if not _EXPLICIT_QUESTION_PAYLOAD.search(body) and (
        _REMINDER_STATUS_QUESTION.search(body)
        or _REMINDER_TRAILING_QUESTION.search(body)
    ):
        return False
    return True


def outer_reminder_match(text: str) -> re.Match[str] | None:
    """Locate the outer reminder command, with leading or front-loaded payload."""

    if note_request_match(text) is not None:
        return None
    candidate = mask_quoted_text(text)
    match = REMINDER_REQUEST_RE.search(candidate)
    if match is None or candidate[:match.start()] != text[:match.start()]:
        return None
    if not _schedule_prefix_matches(candidate[:match.start()]) and not (
        _front_loaded_reminder_prefix(candidate, match)
    ):
        return None
    # “叫我小王” names the speaker; a schedule or action distinguishes a
    # reminder. “明天下午三点叫我一下” remains a request with a missing body.
    if "叫" in match.group():
        body = candidate[match.end():].lstrip(" ，,：:")
        schedule = candidate[:match.start()]
        has_schedule = bool(
            CLOCK_TOKEN_RE.search(schedule)
            or re.search(r"今天|明天|后天|今早|今晨|今晚|今夜|明早|明晨|明晚|明夜|周|星期|礼拜|月|日|早上|上午|中午|下午|晚上|后", schedule)
            or "到时候" in match.group()
        )
        if not _CALL_ACTION.match(body) and not has_schedule and not CLOCK_TOKEN_RE.search(body):
            return None
    return match


def has_multiple_reminder_requests(text: str) -> bool:
    """Detect independently addressed reminder clauses, not marker word count.

    Note-owned content is never an instruction. A MIXED caller may supply only
    its independently introduced task portion. Original slices retain quote
    evidence when checking each clause's outer scope.
    """

    if note_request_match(text) is not None:
        return False
    candidate = mask_quoted_text(text)
    count = 0
    for span in re.finditer(r"[^，,。；;\n]+", candidate):
        clause = text[span.start():span.end()]
        clause = re.sub(r"^\s*(?:另外|同时|并且|还有|顺便)\s*", "", clause)
        match = outer_reminder_match(clause)
        if (
            match is not None
            and not outer_reminder_is_question(clause, match)
            and not has_negated_reminder(clause)
        ):
            count += 1
            if count > 1:
                return True
    return False


def outer_reminder_is_question(text: str, match: re.Match[str]) -> bool:
    candidate = mask_quoted_text(text)
    payload = candidate[match.end():]
    if _REMINDER_BODY_META_QUESTION.search(payload):
        return True
    # “问导师材料交了吗” is the future question, whereas “查天气了吗”
    # without an asking verb is an enquiry about an existing reminder.
    if _EXPLICIT_QUESTION_PAYLOAD.search(payload):
        return False
    if _REMINDER_STATUS_QUESTION.search(payload):
        return True
    return bool(
        _REMINDER_TRAILING_QUESTION.search(payload)
        and not _POLITE_REQUEST_LEAD.search(candidate)
        and not _POLITE_REQUEST_END.search(payload)
    )


def has_negated_reminder(text: str) -> bool:
    """Apply negation to the outer request, not to a future task/note body."""

    if note_request_match(text) is not None:
        return False
    candidate = mask_quoted_text(text)
    outer = outer_reminder_match(text)
    if outer is None:
        return bool(_REMINDER_STOP_INTENT.search(candidate))
    payload = candidate[outer.end():].lstrip(" \t，,：:")
    if _DIRECT_RETRACTION.fullmatch(payload):
        return True
    # A new, directly addressed cancellation clause retracts the request.
    # Do not scan the first payload clause: “提醒我取消会议提醒” creates a
    # future cancellation task, and “告诉导师不用提醒我” is reported content.
    for clause in _RETRACTION_CLAUSE.finditer(payload):
        if _REMINDER_STOP_INTENT.match(payload, clause.end()):
            return True
    return False
