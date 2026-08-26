from __future__ import annotations

import re
from dataclasses import dataclass

from .models import IntentKind
from .routing import normalize_routing_text


@dataclass(frozen=True)
class PrefixDecision:
    forced_kind: IntentKind | None
    content: str
    private: bool = False
    arm_private_next: bool = False
    deep_note: bool = False


_PREFIXES = (
    ("待办", IntentKind.TASK),
    ("任务", IntentKind.TASK),
    ("深度笔记", IntentKind.NOTE),
    ("笔记", IntentKind.NOTE),
)


def parse_prefix(text: str, *, speech: bool = False) -> PrefixDecision:
    original = text or ""
    stripped = original.lstrip()

    # Private content keeps its original bytes and deliberately retains the
    # stricter colon-only grammar.
    for separator in ("：", ":"):
        token = f"私密{separator}"
        if not stripped.startswith(token):
            continue
        content = stripped[len(token) :].lstrip()
        if content.strip() == "下一条":
            return PrefixDecision(
                forced_kind=IntentKind.PRIVATE,
                content="",
                private=True,
                arm_private_next=True,
            )
        return PrefixDecision(
            forced_kind=IntentKind.PRIVATE,
            content=original,
            private=True,
        )

    normalized = normalize_routing_text(original, speech=speech).text
    stripped = normalized.lstrip()
    for label, kind in _PREFIXES:
        if not stripped.startswith(label):
            continue
        remainder = stripped[len(label) :]
        separated = re.match(r"^(?:[：:,，]\s*|\s+)(.*)$", remainder, re.DOTALL)
        if separated is None:
            continue
        return PrefixDecision(
            forced_kind=kind,
            content=separated.group(1).lstrip(),
            deep_note=label == "深度笔记",
        )
    return PrefixDecision(forced_kind=None, content=normalized.strip())
