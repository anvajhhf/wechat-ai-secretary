from __future__ import annotations

import math
import re
import threading
import wave
from pathlib import Path
from typing import Any

from .config import SecretarySettings
from .prefixes import parse_prefix


class SpeechTranscriptionError(RuntimeError):
    """A safe local-only speech failure, without audio text or backend details."""


class _UnreliableTranscript(SpeechTranscriptionError):
    """A decoded candidate contains definite corruption or repetition."""


class LocalSpeechTranscriber:
    """Use one explicitly selected local backend with per-profile options.

    No provider routing, dependency installation, cloud fallback, request history,
    or environment/configuration mutation is involved. Whisper's instance lock
    covers the lazy segment generator; ONNX uses its own full-operation lock.
    """

    def __init__(self, settings: SecretarySettings) -> None:
        self.settings = settings
        self._model: Any = None
        self._onnx: Any = None
        self._lock = threading.Lock()

    def _resolve_model_path(self) -> Path:
        name = self.settings.asr_model
        if len(name) > 40 or not re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", name):
            raise SpeechTranscriptionError("本地语音模型名称无效，没有下载或上传音频。")
        cache = (
            self.settings.project_root / "runtime" / "models" / "huggingface" / "hub"
        ).resolve()
        try:
            from faster_whisper.utils import download_model

            # Despite its name, this helper ONLY resolves the existing cache
            # when local_files_only=True; a cache miss must remain a failure.
            model_path = Path(download_model(
                name, cache_dir=str(cache), local_files_only=True,
            )).resolve(strict=True)
            if not model_path.is_relative_to(cache) or not model_path.is_dir():
                raise ValueError("model-outside-cache")
            for name in ("config.json", "model.bin", "tokenizer.json"):
                item = (model_path / name).resolve(strict=True)
                if not item.is_relative_to(cache) or not item.is_file() or item.stat().st_size <= 0:
                    raise ValueError("incomplete-model-cache")
            # Prechecking tokenizer.json is important: faster-whisper otherwise
            # calls Tokenizer.from_pretrained even when its model was local.
            return model_path
        except SpeechTranscriptionError:
            raise
        except Exception as exc:
            raise SpeechTranscriptionError(
                "本地语音模型缓存不完整或不可读取，没有自动下载；请先发文字。"
            ) from exc

    def _load_model(self) -> Any:
        model_path = self._resolve_model_path()
        try:
            from faster_whisper import WhisperModel

            return WhisperModel(
                str(model_path), device="cpu", compute_type="int8",
                local_files_only=True,
            )
        except Exception as exc:
            raise SpeechTranscriptionError("本地语音模型暂时无法加载，请先发文字。") from exc

    def _decode_options(self) -> dict[str, Any]:
        language = self.settings.asr_language.strip().lower()
        if language != "auto" and not re.fullmatch(r"[a-z]{2,3}", language):
            raise SpeechTranscriptionError("语音语言配置无效，请先发文字。")
        options: dict[str, Any] = {
            "language": None if language == "auto" else language,
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
        }
        vocabulary = self.settings.asr_vocabulary
        if vocabulary:
            # Static domain vocabulary, not a desired sentence or a correction
            # such as replacing 每天 with 明天. Neither dates nor user history
            # are injected. Natural prose avoids teaching ASR semicolon lists.
            terms = "、".join(vocabulary)
            options["initial_prompt"] = (
                "以下是简体中文的日常语音备忘录，可能涉及工作、生活、科研与采购，"
                f"常用词有{terms}。"
                if language == "zh" else f"Vocabulary: {', '.join(vocabulary)}."
            )
        return options

    @staticmethod
    def _wave_duration(path: str | Path) -> float | None:
        try:
            with wave.open(str(path), "rb") as source:
                return source.getnframes() / source.getframerate()
        except (OSError, EOFError, wave.Error, ZeroDivisionError):
            return None

    def _decode_result(self, path: str | Path, options: dict[str, Any]) -> str:
        segments, _ = self._model.transcribe(str(path), **options)
        texts: list[str] = []
        unreliable = False
        for segment in segments:
            text = getattr(segment, "text", "")
            if not isinstance(text, str):
                raise ValueError("invalid-segment-text")
            # A rejected first candidate must not lose a privacy signal when a
            # recovery pass is made. This refusal happens before quality gates.
            candidate = " ".join((*texts, text.strip())).strip()
            if parse_prefix(candidate, speech=True).private:
                raise SpeechTranscriptionError(
                    "我识别到语音中的私密标记，本条只在本地处理，没有外发或写入任务、笔记。"
                    "请先发送“私密：下一条”，再重发语音。"
                )
            no_speech = float(getattr(segment, "no_speech_prob", 0.0))
            logprob = float(getattr(segment, "avg_logprob", 0.0))
            compression = float(getattr(segment, "compression_ratio", 0.0))
            if not all(math.isfinite(value) for value in (no_speech, logprob, compression)):
                raise ValueError("invalid-segment-confidence")
            if not 0.0 <= no_speech <= 1.0 or compression < 0.0:
                raise ValueError("invalid-segment-confidence")
            if no_speech > 0.6 and logprob < -1.0:
                continue
            if text.strip():
                texts.append(text.strip())
                unreliable |= compression > 2.4
        transcript = " ".join(texts).strip()
        if not transcript:
            raise SpeechTranscriptionError("没有可靠识别到语音内容，请改发文字。")
        if len(transcript) > self.settings.media_text_max_chars:
            raise SpeechTranscriptionError("语音转写内容过长，请拆分为多条语音发送。")
        if unreliable or "\ufffd" in transcript or re.search(r"(?:[，,、。.!！？?；;：:]\s*){5,}", transcript):
            raise _UnreliableTranscript("这条语音没有可靠识别，请用文字补充关键日期和事项。")
        return transcript

    def transcribe(self, path: str | Path) -> str:
        if not isinstance(self.settings.asr_backend, str):
            raise SpeechTranscriptionError("本地语音后端配置无效，请先发文字。")
        if self.settings.asr_backend in {"sensevoice", "paraformer"}:
            with self._lock:
                if self._onnx is None:
                    from .speech_onnx import OnnxSpeechTranscriber

                    self._onnx = OnnxSpeechTranscriber(self.settings)
            return self._onnx.transcribe(path)
        if self.settings.asr_backend != "whisper":
            raise SpeechTranscriptionError("本地语音后端配置无效，请先发文字。")
        options = self._decode_options()
        duration = self._wave_duration(path)
        # Domain vocabulary helps longer task descriptions but can overwhelm a
        # short correction such as “不是每天，是明天”. Unknown formats are
        # conservative too; the ordinary decoder still supports them normally.
        if duration is None or duration <= 4.0:
            options.pop("initial_prompt", None)
        with self._lock:
            if self._model is None:
                self._model = self._load_model()
            try:
                try:
                    return self._decode_result(path, options)
                except _UnreliableTranscript:
                    if "initial_prompt" not in options:
                        raise
                    # One local, unprompted recovery only. Never retry silence,
                    # private requests, backend errors or ordinary word errors.
                    recovery_options = dict(options)
                    recovery_options.pop("initial_prompt")
                    return self._decode_result(path, recovery_options)
            except SpeechTranscriptionError:
                raise
            except Exception as exc:
                # Do not log backend exceptions: they can contain paths, input
                # text or model prompts. Never return an incomplete transcript.
                raise SpeechTranscriptionError(
                    "本地语音转写失败，音频没有发送到外部服务，请先发文字。"
                ) from exc
