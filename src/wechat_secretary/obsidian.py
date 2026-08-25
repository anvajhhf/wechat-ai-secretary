from __future__ import annotations

import os
import threading
from pathlib import Path

from .config import SecretarySettings
from .models import ActionResult, ExecutionStatus, MessageEnvelope, NoteDraft
from .path_security import resolve_within, safe_filename, source_reference


class ObsidianExecutor:
    def __init__(self, settings: SecretarySettings):
        self.settings = settings
        self._write_lock = threading.RLock()

    def available_links(self, text: str = "", limit: int = 5000) -> tuple[str, ...]:
        names: list[str] = list(dict.fromkeys(self.settings.known_links))
        root = self.settings.vault_path
        if root is None or not root.is_dir():
            return tuple(item for item in names if not text or item in text)

        count = 0
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [item for item in dirs if not item.startswith(".")]
            for filename in files:
                if not filename.lower().endswith(".md"):
                    continue
                stem = Path(filename).stem.strip()
                if stem and stem not in names:
                    names.append(stem)
                count += 1
                if count >= limit:
                    break
            if count >= limit:
                break
        if text:
            names = [item for item in names if item in text]
        return tuple(names)

    def _target_relative(self, note: NoteDraft) -> Path:
        mapped = self.settings.folder_map.get(note.target_hint, "") if note.target_hint else ""
        configured = mapped or self.settings.default_note_path
        target = Path(configured)
        if target.suffix.lower() == ".md":
            return target
        return target / f"{safe_filename(note.title)}.md"

    def _render(self, note: NoteDraft, message: MessageEnvelope) -> tuple[str, str]:
        ref = source_reference(message)
        timestamp = message.received_at.astimezone(self.settings.tz).isoformat(timespec="seconds")
        tags = " ".join(
            f"#{safe_filename(tag, fallback='tag', limit=40).replace(' ', '_')}"
            for tag in note.tags
            if tag.strip()
        )
        links = " ".join(f"[[{link}]]" for link in note.links[: self.settings.max_links])
        lines = [f"## {note.title}", ""]
        if note.summary:
            lines.extend([f"> 摘要：{note.summary}", ""])
        lines.append(note.body)
        if tags:
            lines.extend(["", f"标签：{tags}"])
        if links:
            lines.extend(["", f"关联：{links}"])
        lines.extend(
            [
                "",
                f"> 来源：微信｜时间：{timestamp}｜引用：{ref}",
                f"<!-- wechat-ai-secretary:ref={ref} -->",
                "",
            ]
        )
        return "\n".join(lines), ref

    @staticmethod
    def _contains_marker(path: Path, marker: str) -> bool:
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return any(marker in line for line in handle)

    def save(self, note: NoteDraft, message: MessageEnvelope) -> ActionResult:
        relative = self._target_relative(note)
        rendered, ref = self._render(note, message)
        destination = relative.as_posix()
        if self.settings.dry_run:
            return ActionResult(
                action="note",
                status=ExecutionStatus.PLANNED,
                summary=note.title,
                destination=destination,
                preview=rendered,
            )
        if not self.settings.obsidian_mapping_confirmed:
            return ActionResult(
                action="note",
                status=ExecutionStatus.FAILED,
                summary=note.title,
                destination=destination,
                error="Obsidian 分类映射尚未确认",
            )
        root = self.settings.vault_path
        if root is None or not root.is_dir():
            return ActionResult(
                action="note",
                status=ExecutionStatus.FAILED,
                summary=note.title,
                destination=destination,
                error="Obsidian Vault 路径无效",
            )
        try:
            target = resolve_within(root, relative)
            marker = f"<!-- wechat-ai-secretary:ref={ref} -->"
            with self._write_lock:
                if self._contains_marker(target, marker):
                    return ActionResult(
                        action="note",
                        status=ExecutionStatus.SKIPPED,
                        summary=note.title,
                        destination=destination,
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                existed = target.exists()
                mode = "a" if existed else "x"
                with target.open(mode, encoding="utf-8", newline="\n") as handle:
                    if existed and target.stat().st_size:
                        handle.write("\n")
                    handle.write(rendered)
            return ActionResult(
                action="note",
                status=ExecutionStatus.SUCCEEDED,
                summary=note.title,
                destination=destination,
                external_id=ref,
                # Keep the rendered note only in this in-memory result so the
                # verified reply can describe what was saved. The ledger
                # deliberately discards previews instead of persisting them.
                preview=rendered,
            )
        except Exception as exc:
            return ActionResult(
                action="note",
                status=ExecutionStatus.FAILED,
                summary=note.title,
                destination=destination,
                error=f"本地写入失败：{type(exc).__name__}",
            )

    def structure_summary(self, max_depth: int = 2, max_entries: int = 500) -> list[str]:
        root = self.settings.vault_path
        if root is None or not root.is_dir():
            return []
        output: list[str] = []
        root_depth = len(root.resolve().parts)
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.resolve().parts) - root_depth
            dirs[:] = [item for item in dirs if not item.startswith(".")]
            if depth >= max_depth:
                dirs[:] = []
            relative = current_path.relative_to(root).as_posix()
            md_count = sum(1 for name in files if name.lower().endswith(".md"))
            output.append(f"{relative or '.'}｜Markdown {md_count} 个")
            if len(output) >= max_entries:
                break
        return output
