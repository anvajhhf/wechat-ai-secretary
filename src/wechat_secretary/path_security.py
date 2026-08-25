from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from .models import MessageEnvelope


def source_reference(message: MessageEnvelope) -> str:
    material = "\x1f".join(message.identity_key)
    return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:16]


def safe_filename(value: str, fallback: str = "微信笔记", limit: int = 80) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return (cleaned[:limit].strip(" .") or fallback)


def resolve_within(root: Path, relative: str | Path) -> Path:
    root_resolved = root.resolve(strict=False)
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("只允许 Vault 内相对路径")
    candidate = (root_resolved / relative_path).resolve(strict=False)
    root_norm = os.path.normcase(str(root_resolved))
    candidate_norm = os.path.normcase(str(candidate))
    try:
        common = os.path.commonpath([root_norm, candidate_norm])
    except ValueError as exc:
        raise ValueError("目标路径不在允许目录内") from exc
    if common != root_norm:
        raise ValueError("目标路径越过了允许目录边界")
    return candidate


def is_within_any(path: Path, roots: tuple[Path, ...]) -> bool:
    candidate = os.path.normcase(str(path.resolve(strict=False)))
    for root in roots:
        root_norm = os.path.normcase(str(root.resolve(strict=False)))
        try:
            if os.path.commonpath([root_norm, candidate]) == root_norm:
                return True
        except ValueError:
            continue
    return False

