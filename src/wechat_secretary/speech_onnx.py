"""Pinned offline CPU speech recognizers; no remote loading or cloud fallback."""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Any

from .config import SecretarySettings
from .prefixes import parse_prefix
from .speech import SpeechTranscriptionError


SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 120
MIN_AUDIO_SAMPLES = SAMPLE_RATE // 5
# (file size, digest, hash format). Source revisions are pinned in the setup
# script; token files use their authoritative Git blob hashes, not guessed IDs.
MODEL_ARTIFACTS = {
    "sensevoice": {
        "model.int8.onnx": (239233841, "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51", "sha256"),
        "tokens.txt": (315894, "2cfc92fc2ff26aaa690b7c01fd96b41109413881", "git-sha1"),
    },
    "paraformer": {
        "model.int8.onnx": (223385835, "9ada9127ca5b82320385ac12340eb8b05dee64fd45cf8cf593ec693826ec2fd7", "sha256"),
        "tokens.txt": (75756, "57bc045ddda0434ed4440c38e14287c595b258d9", "git-sha1"),
    },
}
_KNOWN_TAGS = frozenset({
    "zh", "en", "yue", "ja", "ko", "nospeech", "withitn", "woitn",
    "NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED",
    "EMO_UNKNOWN", "Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze",
    "Breath", "Cough", "Sing", "Speech_Noise", "Event_UNK",
})
_PRIVATE_REFUSAL = (
    "我识别到语音中的私密标记，本条只在本地处理，没有外发或写入任务、笔记。"
    "请先发送“私密：下一条”，再重发语音。"
)


class OnnxSpeechTranscriber:
    def __init__(self, settings: SecretarySettings):
        self.settings = settings
        self._model: Any = None
        self._lock = threading.Lock()

    def _validate_settings(self) -> str:
        backend = self.settings.asr_backend
        if not isinstance(backend, str) or backend not in MODEL_ARTIFACTS:
            raise SpeechTranscriptionError("本地语音后端配置无效，请先发文字。")
        if not isinstance(self.settings.asr_language, str):
            raise SpeechTranscriptionError("本地语音语言配置无效，请先发文字。")
        language = self.settings.asr_language.strip().lower()
        if type(self.settings.asr_threads) is not int or not 1 <= self.settings.asr_threads <= 4:
            raise SpeechTranscriptionError("本地语音线程数配置无效，请先发文字。")
        supported = {"zh", "en", "ja", "ko", "yue", "auto"} if backend == "sensevoice" else {"zh", "auto"}
        if language not in supported:
            raise SpeechTranscriptionError("当前本地语音模型不支持配置的语言，请先发文字。")
        return "" if language == "auto" else language

    def _resolve_model_files(self) -> tuple[Path, Path]:
        # Do not resolve the allowed root through an unexpected directory link.
        allowed = self.settings.project_root.resolve() / "runtime" / "models" / "speech"
        model_dir = allowed / self.settings.asr_backend
        resolved: list[Path] = []
        try:
            for name in ("model.int8.onnx", "tokens.txt"):
                expected_size, expected_digest, kind = MODEL_ARTIFACTS[self.settings.asr_backend][name]
                item = (model_dir / name).resolve(strict=True)
                if not item.is_relative_to(allowed) or not item.is_file() or item.stat().st_size != expected_size:
                    raise ValueError("missing-or-unexpected-artifact")
                with item.open("rb") as stream:
                    if kind == "git-sha1":
                        content = stream.read()
                        digest = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
                    else:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != expected_digest:
                    raise ValueError("artifact-hash-mismatch")
                resolved.append(item)
            return resolved[0], resolved[1]
        except Exception as exc:
            raise SpeechTranscriptionError(
                "本地语音模型缺失或校验失败，没有自动下载或切换模型，请先发文字。"
            ) from exc

    def _load_model(self) -> Any:
        language = self._validate_settings()
        model, tokens = self._resolve_model_files()
        import sherpa_onnx

        if sherpa_onnx.__version__ != "1.13.6":
            raise SpeechTranscriptionError("本地语音组件版本不匹配，请先发文字。")
        common = dict(tokens=str(tokens), num_threads=self.settings.asr_threads,
                      sample_rate=SAMPLE_RATE, feature_dim=80,
                      decoding_method="greedy_search", provider="cpu", debug=False)
        if self.settings.asr_backend == "sensevoice":
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model), language=language, use_itn=True, **common,
            )
        return sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=str(model), **common)

    @staticmethod
    def _read_audio(path: str | Path) -> Any:
        import av
        import numpy as np

        def refuse_nested_resource(*args: Any, **kwargs: Any) -> Any:
            # A mislabeled playlist must not open URLs or other local files.
            raise ValueError("indirect-audio-resource-refused")

        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks: list[Any] = []
        sample_count = 0

        def append_frames(frames: Any) -> None:
            nonlocal sample_count
            for frame in frames:
                chunk = frame.to_ndarray().reshape(-1)
                sample_count += chunk.size
                if sample_count > MAX_AUDIO_SECONDS * SAMPLE_RATE:
                    raise SpeechTranscriptionError("语音超过两分钟，请拆分后发送。")
                chunks.append(chunk)

        with Path(path).open("rb") as source:
            with av.open(source, mode="r", io_open=refuse_nested_resource) as container:
                if len(container.streams.audio) != 1:
                    raise SpeechTranscriptionError("语音音轨格式不明确，请重新发送。")
                for frame in container.decode(audio=0):
                    frame.pts = None
                    append_frames(resampler.resample(frame))
                append_frames(resampler.resample(None))
        if not chunks:
            raise SpeechTranscriptionError("没有识别到可处理的语音，请改发文字。")
        return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32) / 32768.0

    @staticmethod
    def _validate_audio(samples: Any) -> None:
        import numpy as np

        if (not isinstance(samples, np.ndarray) or samples.dtype != np.float32
                or samples.ndim != 1 or not samples.flags.c_contiguous
                or not MIN_AUDIO_SAMPLES <= samples.size <= MAX_AUDIO_SECONDS * SAMPLE_RATE
                or not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.0):
            raise SpeechTranscriptionError("语音长度或采样格式不可靠，请重新发送或改发文字。")

    @staticmethod
    def _has_speech(samples: Any) -> bool:
        # Reuse the already-installed local Silero asset, without trimming the
        # utterance: quiet privacy words at its beginning must not be cut away.
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        return bool(get_speech_timestamps(samples, vad_options=VadOptions(
            min_speech_duration_ms=100, min_silence_duration_ms=500,
        ), sampling_rate=SAMPLE_RATE))

    def _normalize_result(self, text: Any, *, language_tag: Any = None) -> str:
        if not isinstance(text, str):
            raise SpeechTranscriptionError("本地语音结果格式异常，请改发文字。")
        text = text.strip()
        # The pinned sherpa export normally separates these tags already. Only
        # known *leading* transport tags may be removed; never rewrite the body.
        tag_count = 0
        no_speech_tag = False
        while text.startswith("<|"):
            found = re.match(r"<\|([^<>|]+)\|>", text)
            if not found or found[1] not in _KNOWN_TAGS or tag_count >= 8:
                raise SpeechTranscriptionError("本地语音结果包含未知标记，请改发文字。")
            no_speech_tag |= found[1] == "nospeech"
            text = text[found.end():].lstrip()
            tag_count += 1
        # Paraformer may omit punctuation entirely. Treat a recognized spoken
        # 私密 prefix conservatively even without its colon; typed syntax stays
        # unchanged and quoted/body occurrences are not a leading prefix.
        if parse_prefix(text, speech=True).private or re.match(r"^私\s*密", text):
            raise SpeechTranscriptionError(_PRIVATE_REFUSAL)
        if language_tag is not None and not isinstance(language_tag, str):
            raise SpeechTranscriptionError("本地语音结果格式异常，请改发文字。")
        # The pinned SenseVoice binding normally removes the transport tags
        # from text and exposes the language/no-speech signal as result.lang.
        # Check it only after the local privacy gate, without inventing scores.
        if no_speech_tag or language_tag in ("nospeech", "<|nospeech|>"):
            raise SpeechTranscriptionError("没有可靠识别到语音内容，请改发文字。")
        if (not text or not re.search(r"[\w\u3400-\u9fff]", text)
                or len(text) > self.settings.media_text_max_chars
                or "\ufffd" in text or re.search(r"(?:[，,、。.!！？?；;：:]\s*){5,}", text)):
            raise SpeechTranscriptionError("这条语音没有可靠识别，请用文字补充关键日期和事项。")
        return text

    def transcribe(self, path: str | Path) -> str:
        self._validate_settings()
        with self._lock:
            try:
                samples = self._read_audio(path)
                self._validate_audio(samples)
                if not self._has_speech(samples):
                    raise SpeechTranscriptionError("没有可靠识别到语音内容，请改发文字。")
                if self._model is None:
                    self._model = self._load_model()
                # No hotwords/prompt: these native backends cannot support the
                # transducer-only contextual API and may exit instead of raise.
                stream = self._model.create_stream()
                stream.accept_waveform(SAMPLE_RATE, samples)
                self._model.decode_stream(stream)
                result = stream.result
                return self._normalize_result(
                    result.text,
                    language_tag=getattr(result, "lang", None)
                    if self.settings.asr_backend == "sensevoice" else None,
                )
            except SpeechTranscriptionError:
                raise
            except Exception as exc:
                raise SpeechTranscriptionError(
                    "本地语音识别失败，没有下载模型或外发音频，请先发文字。"
                ) from exc
