from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, replace
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_path(value: object, project_root: Path) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    expanded = Path(os.path.expandvars(raw)).expanduser()
    if not expanded.is_absolute():
        expanded = project_root / expanded
    return expanded.resolve(strict=False)


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


@dataclass(frozen=True)
class SecretarySettings:
    project_root: Path
    profile_id: str = "owner"
    dry_run: bool = True
    timezone_name: str = "Asia/Shanghai"
    max_actions_per_message: int = 3
    worker_limit: int = 4
    allowed_users: frozenset[str] = field(default_factory=frozenset)
    account_id: str = ""
    dm_only: bool = True
    private_next_ttl_seconds: int = 120
    completion_context_ttl_seconds: int = 86400
    completion_confirmation_ttl_seconds: int = 300
    reminders_enabled: bool = False
    reminder_poll_seconds: int = 15
    reminder_overdue_merge_seconds: int = 7200
    reminder_retry_seconds: int = 60
    vault_path: Path | None = None
    private_inbox_path: Path | None = None
    default_note_path: str = "Inbox/微信收件箱.md"
    max_links: int = 3
    obsidian_mapping_confirmed: bool = False
    known_links: tuple[str, ...] = ()
    folder_map: dict[str, str] = field(default_factory=dict)
    media_cache_roots: tuple[Path, ...] = ()
    vision_enabled: bool = False
    voice_asr_enabled: bool = False
    image_max_files: int = 4
    image_max_bytes: int = 16 * 1024 * 1024
    image_max_dimension: int = 2048
    audio_max_bytes: int = 25 * 1024 * 1024
    media_text_max_chars: int = 12000
    asr_model: str = "small"
    asr_language: str = "zh"
    dida_server: str = "dida365"
    dida_mapping_confirmed: bool = False
    dida_schema_confirmed: bool = False
    dida_complete_schema_confirmed: bool = False
    category_map: dict[str, str] = field(default_factory=dict)
    tag_map: dict[str, str] = field(default_factory=dict)

    @property
    def tz(self) -> tzinfo:
        try:
            return ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            if self.timezone_name == "Asia/Shanghai":
                return timezone(timedelta(hours=8), name="Asia/Shanghai")
            raise

    @property
    def state_db_path(self) -> Path:
        return (
            self.project_root
            / "runtime"
            / "state"
            / self.profile_id
            / "secretary.sqlite3"
        )

    @property
    def media_work_dir(self) -> Path:
        return (
            self.project_root
            / "runtime"
            / "media-work"
            / self.profile_id
        )

    def with_environment(self) -> "SecretarySettings":
        allowed_env = os.getenv("WEIXIN_ALLOWED_USERS", "").strip()
        allowed = self.allowed_users
        if allowed_env:
            allowed = frozenset(part.strip() for part in allowed_env.split(",") if part.strip())
        dry_run = self.dry_run
        if "SECRETARY_DRY_RUN" in os.environ:
            dry_run = _as_bool(os.environ["SECRETARY_DRY_RUN"], True)
        reminders_enabled = self.reminders_enabled
        if "SECRETARY_REMINDERS_ENABLED" in os.environ:
            reminders_enabled = _as_bool(
                os.environ["SECRETARY_REMINDERS_ENABLED"], False
            )
        account_id = os.getenv("WEIXIN_ACCOUNT_ID", "").strip() or self.account_id
        vault = _as_path(os.getenv("SECRETARY_VAULT_PATH", ""), self.project_root) or self.vault_path
        private_inbox = (
            _as_path(os.getenv("SECRETARY_PRIVATE_INBOX_PATH", ""), self.project_root)
            or self.private_inbox_path
        )
        return replace(
            self,
            dry_run=dry_run,
            reminders_enabled=reminders_enabled,
            allowed_users=allowed,
            account_id=account_id,
            vault_path=vault,
            private_inbox_path=private_inbox,
        )

    def runtime_errors(self, strict: bool = False) -> list[str]:
        errors: list[str] = []
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", self.profile_id):
            errors.append("profile_id 只能包含小写字母、数字、下划线或连字符")
        expected_profile = os.getenv("SECRETARY_PROFILE", "").strip()
        if expected_profile and expected_profile != self.profile_id:
            errors.append("所选运行档案与配置文件中的 profile_id 不一致")
        if self.timezone_name != "Asia/Shanghai":
            errors.append("timezone 必须为 Asia/Shanghai")
        if self.max_actions_per_message < 1 or self.max_actions_per_message > 3:
            errors.append("max_actions_per_message 必须在 1 到 3 之间")
        if strict and not self.allowed_users:
            errors.append("微信 allowlist 为空；为安全起见不会处理任何消息")
        if not self.dry_run:
            if self.vault_path is None:
                errors.append("真实写入模式缺少 Obsidian Vault 路径")
            if self.private_inbox_path is None:
                errors.append("真实写入模式缺少私密收件箱路径")
            if not self.obsidian_mapping_confirmed:
                errors.append("Obsidian 分类映射尚未确认")
            if not self.dida_mapping_confirmed:
                errors.append("滴答清单分类映射尚未确认")
            if not self.dida_schema_confirmed:
                errors.append("滴答 MCP create_task 尚未通过专用测试任务创建并回读确认")
            if os.getenv("SECRETARY_DIDA_CREATES_APPROVED", "").strip() != "1":
                errors.append("本次前台启动尚未显式允许创建滴答任务")
            if (
                os.getenv("SECRETARY_DIDA_COMPLETIONS_APPROVED", "").strip() == "1"
                and not self.dida_complete_schema_confirmed
            ):
                errors.append("滴答 MCP complete_task 参数结构尚未实测确认")
        return errors

    @classmethod
    def from_file(cls, path: Path, project_root: Path | None = None) -> "SecretarySettings":
        config_path = path.resolve(strict=True)
        root = (project_root or config_path.parent.parent).resolve(strict=False)
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)

        secretary = data.get("secretary", {})
        wechat = data.get("wechat", {})
        obsidian = data.get("obsidian", {})
        media = data.get("media", {})
        reminders = data.get("reminders", {})
        dida = data.get("dida", {})

        allowed_users = frozenset(
            str(item).strip()
            for item in wechat.get("allowed_users", [])
            if str(item).strip()
        )
        cache_roots = tuple(
            path_obj
            for item in media.get("allowed_cache_roots", [])
            if (path_obj := _as_path(item, root)) is not None
        )
        known_links = tuple(
            str(item).strip()
            for item in obsidian.get("known_links", [])
            if str(item).strip()
        )

        settings = cls(
            project_root=root,
            profile_id=str(secretary.get("profile_id", "owner")).strip() or "owner",
            dry_run=_as_bool(secretary.get("dry_run"), True),
            timezone_name=str(secretary.get("timezone", "Asia/Shanghai")),
            max_actions_per_message=int(secretary.get("max_actions_per_message", 3)),
            worker_limit=max(1, min(int(secretary.get("worker_limit", 4)), 8)),
            allowed_users=allowed_users,
            account_id=str(wechat.get("account_id", "")).strip(),
            dm_only=_as_bool(wechat.get("dm_only"), True),
            private_next_ttl_seconds=max(
                15, min(int(wechat.get("private_next_ttl_seconds", 120)), 600)
            ),
            completion_context_ttl_seconds=max(
                300, min(int(wechat.get("completion_context_ttl_seconds", 86400)), 604800)
            ),
            completion_confirmation_ttl_seconds=max(
                60,
                min(int(wechat.get("completion_confirmation_ttl_seconds", 300)), 1800),
            ),
            reminders_enabled=_as_bool(reminders.get("enabled"), False),
            reminder_poll_seconds=max(
                5, min(int(reminders.get("poll_seconds", 15)), 300)
            ),
            reminder_overdue_merge_seconds=max(
                300,
                min(int(reminders.get("overdue_merge_seconds", 7200)), 86400),
            ),
            reminder_retry_seconds=max(
                30, min(int(reminders.get("retry_seconds", 60)), 3600)
            ),
            vault_path=_as_path(obsidian.get("vault_path", ""), root),
            private_inbox_path=_as_path(obsidian.get("private_inbox_path", ""), root),
            default_note_path=str(
                obsidian.get("default_note_path", "Inbox/微信收件箱.md")
            ).strip(),
            max_links=max(0, min(int(obsidian.get("max_links", 3)), 3)),
            obsidian_mapping_confirmed=_as_bool(obsidian.get("mapping_confirmed"), False),
            known_links=known_links,
            folder_map=_string_map(obsidian.get("folder_map", {})),
            media_cache_roots=cache_roots,
            vision_enabled=_as_bool(media.get("vision_enabled"), False),
            voice_asr_enabled=_as_bool(media.get("voice_asr_enabled"), False),
            image_max_files=max(1, min(int(media.get("image_max_files", 4)), 8)),
            image_max_bytes=max(
                1024 * 1024,
                min(int(media.get("image_max_bytes", 16 * 1024 * 1024)), 32 * 1024 * 1024),
            ),
            image_max_dimension=max(
                512, min(int(media.get("image_max_dimension", 2048)), 4096)
            ),
            audio_max_bytes=max(
                1024 * 1024,
                min(int(media.get("audio_max_bytes", 25 * 1024 * 1024)), 25 * 1024 * 1024),
            ),
            media_text_max_chars=max(
                1000, min(int(media.get("media_text_max_chars", 12000)), 30000)
            ),
            asr_model=str(media.get("asr_model", "small")).strip() or "small",
            asr_language=str(media.get("asr_language", "zh")).strip() or "zh",
            dida_server=str(dida.get("server", "dida365")).strip() or "dida365",
            dida_mapping_confirmed=_as_bool(dida.get("mapping_confirmed"), False),
            dida_schema_confirmed=_as_bool(dida.get("schema_confirmed"), False),
            dida_complete_schema_confirmed=_as_bool(
                dida.get("complete_schema_confirmed"), False
            ),
            category_map=_string_map(dida.get("category_map", {})),
            tag_map=_string_map(dida.get("tag_map", {})),
        )
        return settings.with_environment()


def load_settings(project_root: Path | None = None) -> SecretarySettings:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve(strict=False)
    raw = os.getenv("SECRETARY_CONFIG", "config/secretary.toml")
    config_path = Path(raw)
    if not config_path.is_absolute():
        config_path = root / config_path
    return SecretarySettings.from_file(config_path, root)
