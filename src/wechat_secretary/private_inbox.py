from __future__ import annotations

import shutil
import threading
from pathlib import Path

from .config import SecretarySettings
from .models import ActionResult, ExecutionStatus, MessageEnvelope
from .path_security import is_within_any, safe_filename, source_reference


class PrivateInboxExecutor:
    def __init__(self, settings: SecretarySettings):
        self.settings = settings
        self._lock = threading.RLock()

    def _render(self, message: MessageEnvelope, copied_names: list[str]) -> str:
        ref = source_reference(message)
        timestamp = message.received_at.astimezone(self.settings.tz).isoformat(timespec="seconds")
        lines = [f"## {timestamp}", "", message.text]
        if copied_names:
            lines.extend(["", "本地附件：" + "、".join(copied_names)])
        lines.extend(
            [
                "",
                f"> 来源：微信｜时间：{timestamp}｜引用：{ref}",
                f"<!-- wechat-ai-secretary:private-ref={ref} -->",
                "",
            ]
        )
        return "\n".join(lines)

    def save(self, message: MessageEnvelope) -> ActionResult:
        local_date = message.received_at.astimezone(self.settings.tz).date().isoformat()
        destination = f"{local_date}.md"
        if self.settings.dry_run:
            media_count = len(message.media_paths)
            summary = "私密文本" if not media_count else f"私密内容（含 {media_count} 个附件）"
            return ActionResult(
                action="private",
                status=ExecutionStatus.PLANNED,
                summary=summary,
                destination=destination,
                preview="[私密正文不在 Dry Run 输出中回显；真实模式将原样本地保存]",
            )
        root = self.settings.private_inbox_path
        if root is None:
            return ActionResult(
                action="private",
                status=ExecutionStatus.FAILED,
                summary="私密内容",
                error="私密收件箱路径尚未配置",
            )

        copied: list[str] = []
        try:
            with self._lock:
                root.mkdir(parents=True, exist_ok=True)
                ref = source_reference(message)
                if message.media_paths:
                    attachments = root / "attachments" / local_date
                    attachments.mkdir(parents=True, exist_ok=True)
                    for index, raw_path in enumerate(message.media_paths, start=1):
                        source = Path(raw_path).resolve(strict=True)
                        if not is_within_any(source, self.settings.media_cache_roots):
                            raise PermissionError("媒体文件不在允许的 Hermes 缓存目录内")
                        target_name = f"{ref}-{index}-{safe_filename(source.name, 'attachment.bin')}"
                        target = attachments / target_name
                        if not target.exists():
                            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                                shutil.copyfileobj(source_handle, target_handle)
                        copied.append((Path("attachments") / local_date / target_name).as_posix())

                target_note = root / destination
                marker = f"<!-- wechat-ai-secretary:private-ref={ref} -->"
                if target_note.exists():
                    with target_note.open("r", encoding="utf-8", errors="replace") as handle:
                        if any(marker in line for line in handle):
                            return ActionResult(
                                action="private",
                                status=ExecutionStatus.SKIPPED,
                                summary="私密内容",
                                destination=destination,
                            )
                existed = target_note.exists()
                with target_note.open("a" if existed else "x", encoding="utf-8", newline="\n") as handle:
                    if existed and target_note.stat().st_size:
                        handle.write("\n")
                    handle.write(self._render(message, copied))
            return ActionResult(
                action="private",
                status=ExecutionStatus.SUCCEEDED,
                summary="私密内容",
                destination=destination,
                external_id=ref,
            )
        except Exception as exc:
            return ActionResult(
                action="private",
                status=ExecutionStatus.FAILED,
                summary="私密内容",
                destination=destination,
                error=f"私密本地保存失败：{type(exc).__name__}",
            )
