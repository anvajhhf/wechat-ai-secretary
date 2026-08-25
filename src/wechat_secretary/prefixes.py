from __future__ import annotations

from dataclasses import dataclass

from .models import IntentKind


@dataclass(frozen=True)
class PrefixDecision:
    forced_kind: IntentKind | None
    content: str
    private: bool = False
    arm_private_next: bool = False
    deep_note: bool = False


_PREFIXES = (
    ("待办", IntentKind.TASK),
    ("深度笔记", IntentKind.NOTE),
    ("笔记", IntentKind.NOTE),
    ("私密", IntentKind.PRIVATE),
)


def parse_prefix(text: str) -> PrefixDecision:
    original = text or ""
    stripped = original.lstrip()
    for label, kind in _PREFIXES:
        for separator in ("：", ":"):
            token = f"{label}{separator}"
            if not stripped.startswith(token):
                continue
            content = stripped[len(token) :].lstrip()
            if kind is IntentKind.PRIVATE:
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
            return PrefixDecision(
                forced_kind=kind,
                content=content,
                deep_note=label == "深度笔记",
            )
    return PrefixDecision(forced_kind=None, content=original.strip())
