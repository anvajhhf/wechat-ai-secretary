from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .config import SecretarySettings
from .models import MessageEnvelope
from .path_security import is_within_any


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_AUDIO_SUFFIXES = {".silk", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac"}
_MAX_PREPARED_IMAGE_BYTES = 30 * 1024 * 1024


class MediaPreparationError(RuntimeError):
    """Safe, user-facing failure raised before any LLM call."""


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    file_name: str
    fingerprint: str

    def as_llm_input(self) -> dict[str, object]:
        return {
            "type": "image",
            "data": self.data,
            "mime_type": self.mime_type,
            "file_name": self.file_name,
        }


@dataclass(frozen=True)
class PreparedMedia:
    transcript_text: str = ""
    images: tuple[PreparedImage, ...] = ()
    fingerprints: tuple[str, ...] = ()

    @property
    def image_inputs(self) -> tuple[dict[str, object], ...]:
        return tuple(image.as_llm_input() for image in self.images)


class MediaPreprocessor(Protocol):
    def prepare(self, message: MessageEnvelope) -> PreparedMedia: ...


class DisabledMediaPreprocessor:
    def prepare(self, message: MessageEnvelope) -> PreparedMedia:
        if message.media_paths:
            raise MediaPreparationError("当前媒体处理组件未启用，请补充文字说明。")
        return PreparedMedia()


class LocalMediaPreprocessor:
    """Prepare non-private media without logging or persisting extracted bodies.

    Images are decoded, metadata-stripped and resized in memory before being sent
    to the configured DeepSeek vision auxiliary task. Voice is transcribed only
    through Hermes' already-installed local faster-whisper backend.
    """

    def __init__(self, settings: SecretarySettings):
        self.settings = settings

    def _safe_path(self, raw_path: str, *, max_bytes: int) -> tuple[Path, bytes, str]:
        candidate = Path(raw_path)
        if candidate.is_symlink():
            raise MediaPreparationError("媒体文件路径不安全，本条未处理。")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise MediaPreparationError("微信媒体文件不可读取，请重新发送。") from exc
        if not resolved.is_file() or not is_within_any(resolved, self.settings.media_cache_roots):
            raise MediaPreparationError("媒体文件不在允许的微信缓存目录中，本条未处理。")
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise MediaPreparationError("微信媒体文件不可读取，请重新发送。") from exc
        if size <= 0:
            raise MediaPreparationError("收到的媒体文件为空，请重新发送。")
        if size > max_bytes:
            raise MediaPreparationError("媒体文件过大，请压缩或拆分后重新发送。")
        try:
            payload = resolved.read_bytes()
        except OSError as exc:
            raise MediaPreparationError("微信媒体文件不可读取，请重新发送。") from exc
        return resolved, payload, hashlib.sha256(payload).hexdigest()

    def _prepare_image(self, raw_path: str, index: int) -> PreparedImage:
        if not self.settings.vision_enabled:
            raise MediaPreparationError("图片理解尚未启用，请补充文字说明。")
        _, payload, fingerprint = self._safe_path(
            raw_path, max_bytes=self.settings.image_max_bytes
        )
        try:
            from PIL import Image, ImageOps, UnidentifiedImageError
        except ImportError as exc:
            raise MediaPreparationError(
                "本地图片安全处理依赖未安装，图片没有发送给模型。"
            ) from exc

        try:
            Image.MAX_IMAGE_PIXELS = 40_000_000
            with Image.open(BytesIO(payload)) as probe:
                actual_format = str(probe.format or "").upper()
                if probe.width * probe.height > Image.MAX_IMAGE_PIXELS:
                    raise MediaPreparationError("图片像素尺寸过大，请缩小后重新发送。")
                probe.verify()
            if actual_format not in {"JPEG", "PNG", "GIF", "WEBP"}:
                raise MediaPreparationError("图片实际格式不受支持，请改用 JPEG、PNG、GIF 或 WebP。")

            with Image.open(BytesIO(payload)) as source:
                source.seek(0)
                image = ImageOps.exif_transpose(source).copy()
            longest = max(image.size)
            if longest > self.settings.image_max_dimension:
                scale = self.settings.image_max_dimension / longest
                target = (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
                image.thumbnail(target, Image.Resampling.LANCZOS)

            has_alpha = image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            )
            if has_alpha:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                image = background.convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")

            encoded = BytesIO()
            if actual_format == "PNG":
                image.save(encoded, format="PNG", optimize=True)
                mime_type, extension = "image/png", ".png"
            else:
                image.save(encoded, format="JPEG", quality=90, optimize=True)
                mime_type, extension = "image/jpeg", ".jpg"
            normalized = encoded.getvalue()
            if len(normalized) > self.settings.image_max_bytes:
                encoded = BytesIO()
                image.save(encoded, format="JPEG", quality=82, optimize=True)
                normalized = encoded.getvalue()
                mime_type, extension = "image/jpeg", ".jpg"
            if len(normalized) > self.settings.image_max_bytes:
                raise MediaPreparationError("图片处理后仍然过大，请压缩后重新发送。")
            return PreparedImage(
                data=normalized,
                mime_type=mime_type,
                file_name=f"wechat-image-{index}{extension}",
                fingerprint=fingerprint,
            )
        except MediaPreparationError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise MediaPreparationError("图片内容无法安全解码，请重新发送。") from exc

    def _transcribe_audio(self, raw_path: str) -> tuple[str, str]:
        if not self.settings.voice_asr_enabled:
            raise MediaPreparationError("语音转写尚未启用，请改发文字。")
        path, _, fingerprint = self._safe_path(
            raw_path, max_bytes=self.settings.audio_max_bytes
        )
        prepared_path = path
        temporary_audio: Path | None = None
        self.settings.media_work_dir.mkdir(parents=True, exist_ok=True)
        try:
            if path.suffix.casefold() == ".silk":
                try:
                    import pysilk
                except ImportError as exc:
                    raise MediaPreparationError(
                        "本地微信语音解码依赖未安装，语音没有发送给任何外部服务。"
                    ) from exc
                temporary_audio = (
                    self.settings.media_work_dir / f"voice-{uuid4().hex}.wav"
                )
                try:
                    pcm = BytesIO()
                    with path.open("rb") as silk_input:
                        pysilk.decode(silk_input, pcm, 24_000)
                    pcm.seek(0)
                    with temporary_audio.open("xb") as wav_file:
                        with wave.open(wav_file, "wb") as wav_output:
                            wav_output.setnchannels(1)
                            wav_output.setsampwidth(2)
                            wav_output.setframerate(24_000)
                            wav_output.writeframes(pcm.read())
                except Exception as exc:
                    raise MediaPreparationError("微信语音解码失败，请重新发送。") from exc
                if not temporary_audio.is_file() or temporary_audio.stat().st_size <= 0:
                    raise MediaPreparationError("微信语音解码失败，请重新发送。")
                prepared_path = temporary_audio

            try:
                from tools.transcription_tools import transcribe_audio_local_fallback
            except ImportError as exc:
                raise MediaPreparationError("Hermes 本地语音组件不可用，语音未处理。") from exc
            result = transcribe_audio_local_fallback(
                str(prepared_path), model=self.settings.asr_model
            )
            if not result.get("success") or result.get("provider") != "local":
                raise MediaPreparationError(
                    "本地语音转写失败，语音没有发送给任何外部服务。"
                )
            transcript = str(result.get("transcript") or "").strip()
            if not transcript:
                raise MediaPreparationError("没有可靠识别到语音内容，请改发文字。")
            return transcript, fingerprint
        except MediaPreparationError:
            raise
        except OSError as exc:
            raise MediaPreparationError("本地语音临时处理失败，请重新发送。") from exc
        finally:
            if temporary_audio is not None:
                try:
                    temporary_audio.unlink(missing_ok=True)
                except OSError as exc:
                    raise MediaPreparationError(
                        "本地语音临时文件清理失败，语音没有发送给任何外部服务。"
                    ) from exc

    def prepare(self, message: MessageEnvelope) -> PreparedMedia:
        if not message.media_paths:
            return PreparedMedia()
        if len(message.media_paths) > self.settings.image_max_files:
            raise MediaPreparationError(
                f"单条消息最多处理 {self.settings.image_max_files} 个图片或语音附件，请拆分发送。"
            )

        images: list[PreparedImage] = []
        transcripts: list[str] = []
        fingerprints: list[str] = []
        for index, raw_path in enumerate(message.media_paths, start=1):
            declared = (
                message.media_types[index - 1].casefold()
                if index - 1 < len(message.media_types)
                else ""
            )
            suffix = Path(raw_path).suffix.casefold()
            if declared.startswith("image/") or suffix in _IMAGE_SUFFIXES:
                image = self._prepare_image(raw_path, index)
                images.append(image)
                fingerprints.append(image.fingerprint)
                continue
            if declared.startswith("audio/") or suffix in _AUDIO_SUFFIXES:
                transcript, fingerprint = self._transcribe_audio(raw_path)
                transcripts.append(f"[语音转写 {index}]\n{transcript}")
                fingerprints.append(fingerprint)
                continue
            raise MediaPreparationError("当前仅处理微信图片和语音；其他附件请补充文字说明。")

        if sum(len(image.data) for image in images) > _MAX_PREPARED_IMAGE_BYTES:
            raise MediaPreparationError("本条消息的图片总量过大，请拆分后重新发送。")
        if len(transcripts) == 1:
            transcript_text = transcripts[0].split("\n", 1)[-1].strip()
        else:
            transcript_text = "\n\n".join(transcripts).strip()
        if len(transcript_text) > self.settings.media_text_max_chars:
            raise MediaPreparationError("语音转写内容过长，请拆分为多条语音发送。")
        return PreparedMedia(
            transcript_text=transcript_text,
            images=tuple(images),
            fingerprints=tuple(fingerprints),
        )
