"""Independent offline ONNX boundaries: fake inference, synthetic fixtures only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import numpy as np

import test_voice_reminder_conversation as service_fixtures
from wechat_secretary.config import SecretarySettings
from wechat_secretary.media import LocalMediaPreprocessor
from wechat_secretary.models import ExecutionStatus
from wechat_secretary.speech import LocalSpeechTranscriber, SpeechTranscriptionError
from wechat_secretary.speech_onnx import OnnxSpeechTranscriber, SAMPLE_RATE


TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"
REMINDER_TEXT = "明天下午两点提醒我分选试剂盒询价"


class OnnxSpeechTests(unittest.TestCase):
    make_service = service_fixtures.VoiceReminderConversationTests.make_service

    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = (TEST_ROOT / f"onnx-speech-{uuid4().hex}").resolve()
        self.root.mkdir()
        self.addCleanup(self.cleanup_fixture)
        self.settings = SecretarySettings(
            project_root=self.root, voice_asr_enabled=True,
            media_cache_roots=(self.root,), asr_backend="sensevoice",
            asr_language="zh", asr_threads=2,
        )
        self.audio = self.root / "synthetic.wav"
        self.audio.write_bytes(b"synthetic-audio-decoding-is-mocked")
        self.samples = np.full(SAMPLE_RATE, 0.125, dtype=np.float32)
        self.real_audio_reader = OnnxSpeechTranscriber._read_audio
        self.audio_reader = self.enterContext(patch.object(
            OnnxSpeechTranscriber, "_read_audio", return_value=self.samples,
        ))
        self.output = REMINDER_TEXT
        self.streams = []
        self.model = SimpleNamespace(
            create_stream=Mock(side_effect=self.create_stream),
            decode_stream=Mock(side_effect=self.decode_stream),
        )
        self.sherpa = ModuleType("sherpa_onnx")
        self.sherpa.__version__ = "1.13.6"
        self.sherpa.OfflineRecognizer = SimpleNamespace(
            from_sense_voice=Mock(return_value=self.model),
            from_paraformer=Mock(return_value=self.model),
        )
        self.whisper = ModuleType("faster_whisper")
        self.whisper.WhisperModel = Mock(side_effect=AssertionError("real Whisper loading forbidden"))
        self.whisper_utils = ModuleType("faster_whisper.utils")
        self.whisper_utils.download_model = Mock(side_effect=AssertionError("cache downloads forbidden"))
        self.vad = ModuleType("faster_whisper.vad")
        self.vad.VadOptions = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        self.vad.get_speech_timestamps = Mock(return_value=[{"start": 100, "end": 15000}])
        self.whisper.vad = self.vad
        self.whisper.utils = self.whisper_utils
        self.enterContext(patch.dict(sys.modules, {
            "sherpa_onnx": self.sherpa,
            "faster_whisper": self.whisper,
            "faster_whisper.utils": self.whisper_utils,
            "faster_whisper.vad": self.vad,
            "tools.transcription_tools": None,
        }))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("network forbidden")))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("network forbidden")))

        payloads = {"model.int8.onnx": b"fake-int8-model", "tokens.txt": b"fake-token-table\n"}
        manifest = {}
        for backend in ("sensevoice", "paraformer"):
            directory = self.root / "runtime" / "models" / "speech" / backend
            directory.mkdir(parents=True)
            manifest[backend] = {}
            for name, content in payloads.items():
                (directory / name).write_bytes(content)
                if name == "tokens.txt":
                    digest = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
                    kind = "git-sha1"
                else:
                    digest, kind = hashlib.sha256(content).hexdigest(), "sha256"
                manifest[backend][name] = (len(content), digest, kind)
        self.enterContext(patch("wechat_secretary.speech_onnx.MODEL_ARTIFACTS", manifest))

    def cleanup_fixture(self):
        if self.root.parent != TEST_ROOT.resolve() or not self.root.name.startswith("onnx-speech-"):
            raise AssertionError("refusing cleanup outside owned ONNX fixture")
        shutil.rmtree(self.root)

    def create_stream(self):
        stream = SimpleNamespace(accept_waveform=Mock(), result=SimpleNamespace(text=""))
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        stream.result = SimpleNamespace(text=self.output)

    def transcriber(self, backend="sensevoice", **changes):
        return OnnxSpeechTranscriber(replace(self.settings, asr_backend=backend, **changes))

    def assert_no_native_constructor(self):
        self.sherpa.OfflineRecognizer.from_sense_voice.assert_not_called()
        self.sherpa.OfflineRecognizer.from_paraformer.assert_not_called()

    def test_both_backends_use_pinned_local_files_and_cpu_only(self):
        for backend in ("sensevoice", "paraformer"):
            with self.subTest(backend=backend):
                self.assertEqual(REMINDER_TEXT, self.transcriber(backend).transcribe(self.audio))
                constructor = getattr(self.sherpa.OfflineRecognizer, f"from_{'sense_voice' if backend == 'sensevoice' else backend}")
                kwargs = constructor.call_args.kwargs
                directory = self.root / "runtime" / "models" / "speech" / backend
                model_key = "model" if backend == "sensevoice" else "paraformer"
                self.assertEqual(directory / "model.int8.onnx", Path(kwargs[model_key]))
                self.assertEqual(directory / "tokens.txt", Path(kwargs["tokens"]))
                self.assertEqual("cpu", kwargs["provider"])
                self.assertEqual(2, kwargs["num_threads"])
                self.assertEqual(16000, kwargs["sample_rate"])
                self.assertEqual(80, kwargs["feature_dim"])
                self.assertEqual("greedy_search", kwargs["decoding_method"])
                self.assertIs(False, kwargs["debug"])
                if backend == "sensevoice":
                    self.assertEqual("zh", kwargs["language"])
                    self.assertIs(True, kwargs["use_itn"])
                self.assertFalse({"hotwords", "hotwords_file", "initial_prompt", "prompt"} & kwargs.keys())
        self.whisper_utils.download_model.assert_not_called()
        self.whisper.WhisperModel.assert_not_called()

    def test_model_is_loaded_once_but_each_utterance_has_a_new_stream(self):
        transcriber = self.transcriber()
        transcriber.transcribe(self.audio)
        transcriber.transcribe(self.audio)
        self.sherpa.OfflineRecognizer.from_sense_voice.assert_called_once()
        self.assertEqual(2, self.model.create_stream.call_count)
        self.assertIsNot(self.streams[0], self.streams[1])
        for stream in self.streams:
            stream.accept_waveform.assert_called_once()
            self.assertEqual(16000, stream.accept_waveform.call_args.args[0])
            self.assertIs(self.samples, stream.accept_waveform.call_args.args[1])
        for call in self.model.create_stream.call_args_list:
            self.assertEqual((), call.args)
            self.assertEqual({}, call.kwargs)

    def test_facade_dispatches_without_loading_whisper_or_changing_default(self):
        self.assertEqual("whisper", SecretarySettings(project_root=self.root).asr_backend)
        for backend in ("sensevoice", "paraformer"):
            with self.subTest(backend=backend):
                facade = LocalSpeechTranscriber(replace(self.settings, asr_backend=backend))
                self.assertEqual(REMINDER_TEXT, facade.transcribe(self.audio))
                self.assertIsNotNone(facade._onnx)
        self.whisper.WhisperModel.assert_not_called()
        self.whisper_utils.download_model.assert_not_called()

    def test_invalid_backend_names_fail_before_audio_or_native_loading(self):
        for backend in ("", "unknown", "../sensevoice", "sensevoice;exec", "https://example.invalid/model"):
            with self.subTest(backend=backend):
                with self.assertRaises(SpeechTranscriptionError):
                    LocalSpeechTranscriber(replace(self.settings, asr_backend=backend)).transcribe(self.audio)
        self.audio_reader.assert_not_called()
        self.assert_no_native_constructor()

    def test_nonstring_backend_and_language_objects_fail_with_safe_errors(self):
        for backend in (None, [], {}, 1, True):
            with self.subTest(backend=backend):
                settings = replace(self.settings, asr_backend=backend)
                for cls in (LocalSpeechTranscriber, OnnxSpeechTranscriber):
                    with self.assertRaises(SpeechTranscriptionError):
                        cls(settings).transcribe(self.audio)
        for language in (None, [], {}, 1, True):
            with self.subTest(language=language):
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber(asr_language=language).transcribe(self.audio)
        self.audio_reader.assert_not_called()
        self.assert_no_native_constructor()

    def test_invalid_thread_counts_and_languages_fail_before_reading_audio(self):
        for threads in (True, 0, -1, 5, 1.5, "2", None):
            with self.subTest(threads=threads):
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber(asr_threads=threads).transcribe(self.audio)
        for backend, language in (("sensevoice", "zh-CN"), ("sensevoice", "../en"), ("paraformer", "en"), ("paraformer", "ja")):
            with self.subTest(backend=backend, language=language):
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber(backend, asr_language=language).transcribe(self.audio)
        self.audio_reader.assert_not_called()
        self.assert_no_native_constructor()

    def test_auto_language_is_explicit_and_environment_is_unchanged(self):
        before = dict(os.environ)
        self.transcriber(asr_language="auto").transcribe(self.audio)
        self.assertEqual("", self.sherpa.OfflineRecognizer.from_sense_voice.call_args.kwargs["language"])
        self.assertEqual(before, dict(os.environ))

    def test_config_defaults_and_rejected_values_are_explicit(self):
        path = self.root / "fixture-config.toml"
        path.write_text("[media]\n", encoding="utf-8")
        defaults = SecretarySettings.from_file(path, project_root=self.root)
        self.assertEqual("whisper", defaults.asr_backend)
        self.assertEqual(2, defaults.asr_threads)
        for backend in ("sensevoice", "paraformer"):
            path.write_text(f'[media]\nasr_backend = "{backend}"\nasr_threads = 4\n', encoding="utf-8")
            settings = SecretarySettings.from_file(path, project_root=self.root)
            self.assertEqual(backend, settings.asr_backend)
            self.assertEqual(4, settings.asr_threads)
        for field, value in (("asr_backend", "remote"), ("asr_threads", True), ("asr_threads", 0), ("asr_threads", "2")):
            path.write_text(f"[media]\n{field} = {json.dumps(value)}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                SecretarySettings.from_file(path, project_root=self.root)

    def test_missing_model_or_tokens_never_calls_native_or_downloads(self):
        for name in ("model.int8.onnx", "tokens.txt"):
            with self.subTest(name=name):
                path = self.root / "runtime" / "models" / "speech" / "sensevoice" / name
                payload = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(SpeechTranscriptionError):
                        self.transcriber().transcribe(self.audio)
                finally:
                    path.write_bytes(payload)
        self.assert_no_native_constructor()
        self.whisper_utils.download_model.assert_not_called()

    def test_same_size_tampering_of_model_and_git_blob_tokens_is_rejected(self):
        for name in ("model.int8.onnx", "tokens.txt"):
            with self.subTest(name=name):
                path = self.root / "runtime" / "models" / "speech" / "sensevoice" / name
                payload = path.read_bytes()
                path.write_bytes(b"X" * len(payload))
                try:
                    with self.assertRaises(SpeechTranscriptionError):
                        self.transcriber().transcribe(self.audio)
                finally:
                    path.write_bytes(payload)
        self.assert_no_native_constructor()

    def test_resolved_artifact_cannot_escape_the_fixed_speech_directory(self):
        expected = self.root / "runtime" / "models" / "speech" / "sensevoice" / "model.int8.onnx"
        outside = self.root / "outside-model.onnx"
        outside.write_bytes(expected.read_bytes())
        original_resolve = Path.resolve

        def redirect(path, *args, **kwargs):
            return outside if path == expected else original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", redirect):
            with self.assertRaises(SpeechTranscriptionError):
                self.transcriber().transcribe(self.audio)
        self.assert_no_native_constructor()

    def test_dependency_absence_or_version_mismatch_has_no_fallback(self):
        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            with self.assertRaises(SpeechTranscriptionError):
                self.transcriber().transcribe(self.audio)
        self.sherpa.__version__ = "1.13.5"
        with self.assertRaises(SpeechTranscriptionError):
            self.transcriber().transcribe(self.audio)
        self.assert_no_native_constructor()
        self.whisper.WhisperModel.assert_not_called()
        self.whisper_utils.download_model.assert_not_called()

    def test_native_exception_is_not_retried_or_echoed(self):
        self.model.decode_stream.side_effect = RuntimeError("SYNTHETIC-SECRET-DETAIL")
        with self.assertRaises(SpeechTranscriptionError) as caught:
            self.transcriber().transcribe(self.audio)
        self.assertNotIn("SYNTHETIC-SECRET-DETAIL", str(caught.exception))
        self.model.decode_stream.assert_called_once()
        self.sherpa.OfflineRecognizer.from_paraformer.assert_not_called()

    def test_failed_constructor_is_not_cached_as_a_working_model(self):
        self.sherpa.OfflineRecognizer.from_sense_voice.side_effect = [RuntimeError("synthetic"), self.model]
        transcriber = self.transcriber()
        with self.assertRaises(SpeechTranscriptionError):
            transcriber.transcribe(self.audio)
        self.assertEqual(REMINDER_TEXT, transcriber.transcribe(self.audio))
        self.assertEqual(2, self.sherpa.OfflineRecognizer.from_sense_voice.call_count)

    def test_invalid_samples_never_reach_vad_or_native(self):
        invalid = (
            None, [0.1] * 3200, np.zeros(3200, dtype=np.float64),
            np.zeros((2, 3200), dtype=np.float32), np.zeros(3199, dtype=np.float32),
            np.zeros(120 * SAMPLE_RATE + 1, dtype=np.float32),
            np.full(3200, np.nan, dtype=np.float32), np.full(3200, np.inf, dtype=np.float32),
            np.full(3200, 1.01, dtype=np.float32), np.zeros(6400, dtype=np.float32)[::2],
        )
        for samples in invalid:
            with self.subTest(type=type(samples), shape=getattr(samples, "shape", None)):
                self.audio_reader.return_value = samples
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber().transcribe(self.audio)
        self.vad.get_speech_timestamps.assert_not_called()
        self.assert_no_native_constructor()

    def test_exact_sample_length_limits_are_accepted(self):
        for count in (3200, 120 * SAMPLE_RATE):
            with self.subTest(count=count):
                self.audio_reader.return_value = np.zeros(count, dtype=np.float32)
                self.assertEqual(REMINDER_TEXT, self.transcriber().transcribe(self.audio))

    def test_silence_fails_before_model_load_and_vad_does_not_trim_input(self):
        self.vad.get_speech_timestamps.return_value = []
        with self.assertRaises(SpeechTranscriptionError):
            self.transcriber().transcribe(self.audio)
        self.assert_no_native_constructor()
        self.vad.get_speech_timestamps.return_value = [{"start": 8000, "end": 9000}]
        self.transcriber().transcribe(self.audio)
        self.assertIs(self.samples, self.streams[-1].accept_waveform.call_args.args[1])
        self.assertEqual(16000, self.vad.get_speech_timestamps.call_args.kwargs["sampling_rate"])

    def test_vad_failure_is_fail_closed_without_loading_the_model(self):
        self.vad.get_speech_timestamps.side_effect = RuntimeError("SYNTHETIC-VAD-SECRET")
        with self.assertRaises(SpeechTranscriptionError) as caught:
            self.transcriber().transcribe(self.audio)
        self.assertNotIn("SYNTHETIC-VAD-SECRET", str(caught.exception))
        self.assert_no_native_constructor()

    def fake_av(self, *, audio_tracks=1, decoded=None, resampled=None, flushed=None):
        module = ModuleType("av")
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams = SimpleNamespace(audio=[object() for _ in range(audio_tracks)])
        source_frame = SimpleNamespace(pts=123)
        container.decode.return_value = iter([source_frame] if decoded is None else decoded)
        frames = [SimpleNamespace(to_ndarray=lambda: np.array([[-32768, 0, 32767]], dtype=np.int16))] if resampled is None else resampled
        resampler = SimpleNamespace(resample=Mock(side_effect=lambda frame: (flushed or []) if frame is None else frames))
        module.AudioResampler = Mock(return_value=resampler)
        module.open = Mock(return_value=container)
        self.enterContext(patch.dict(sys.modules, {"av": module}))
        return module, container, resampler, source_frame

    def test_audio_reader_uses_a_file_object_and_exact_mono_16khz_pcm(self):
        module, container, resampler, frame = self.fake_av()
        samples = self.real_audio_reader(self.audio)
        self.assertEqual(np.float32, samples.dtype)
        self.assertEqual(1, samples.ndim)
        self.assertTrue(samples.flags.c_contiguous)
        np.testing.assert_allclose(samples, [-1.0, 0.0, 32767 / 32768])
        module.AudioResampler.assert_called_once_with(format="s16", layout="mono", rate=16000)
        self.assertIsNone(frame.pts)
        self.assertEqual("r", module.open.call_args.kwargs["mode"])
        source = module.open.call_args.args[0]
        self.assertTrue(hasattr(source, "read"))
        self.assertTrue(source.closed)
        container.decode.assert_called_once_with(audio=0)
        self.assertEqual(2, resampler.resample.call_count)

    def test_audio_reader_refuses_nested_urls_or_local_resource_requests(self):
        module, _, _, _ = self.fake_av()

        def nested_open(source, **kwargs):
            return kwargs["io_open"]("https://must-not-open.invalid/private", 0, {})

        module.open.side_effect = nested_open
        with patch.object(OnnxSpeechTranscriber, "_read_audio", staticmethod(self.real_audio_reader)):
            with self.assertRaises(SpeechTranscriptionError) as caught:
                self.transcriber().transcribe(self.audio)
        self.assertNotIn("must-not-open", str(caught.exception))
        self.assert_no_native_constructor()
        self.vad.get_speech_timestamps.assert_not_called()

    def test_audio_reader_rejects_zero_or_multiple_tracks_before_decoding(self):
        for tracks in (0, 2):
            with self.subTest(tracks=tracks):
                _, container, resampler, _ = self.fake_av(audio_tracks=tracks)
                with self.assertRaises(SpeechTranscriptionError):
                    self.real_audio_reader(self.audio)
                container.decode.assert_not_called()
                resampler.resample.assert_not_called()

    def test_audio_reader_enforces_limit_during_decode_instead_of_truncating(self):
        huge = SimpleNamespace(to_ndarray=lambda: np.zeros((1, 120 * SAMPLE_RATE + 1), dtype=np.int16))
        module, container, _, _ = self.fake_av(resampled=[huge])
        with self.assertRaises(SpeechTranscriptionError):
            self.real_audio_reader(self.audio)
        self.assertTrue(module.open.call_args.args[0].closed)
        container.__exit__.assert_called_once()
        self.assert_no_native_constructor()

    def test_audio_reader_rejects_empty_output_but_preserves_resampler_flush(self):
        self.fake_av(decoded=[], resampled=[])
        with self.assertRaises(SpeechTranscriptionError):
            self.real_audio_reader(self.audio)
        tail = SimpleNamespace(to_ndarray=lambda: np.array([[42]], dtype=np.int16))
        self.fake_av(resampled=[], flushed=[tail])
        np.testing.assert_allclose(self.real_audio_reader(self.audio), [42 / 32768])

    def test_audio_decoder_failure_cannot_return_a_partial_transcript(self):
        _, container, _, _ = self.fake_av()

        def broken_decode(**kwargs):
            yield SimpleNamespace(pts=456)
            raise RuntimeError("SYNTHETIC-DECODER-SECRET")

        container.decode.side_effect = broken_decode
        with patch.object(OnnxSpeechTranscriber, "_read_audio", staticmethod(self.real_audio_reader)):
            with self.assertRaises(SpeechTranscriptionError) as caught:
                self.transcriber().transcribe(self.audio)
        self.assertNotIn("SYNTHETIC-DECODER-SECRET", str(caught.exception))
        self.vad.get_speech_timestamps.assert_not_called()
        self.assert_no_native_constructor()

    def test_known_leading_tags_are_removed_without_changing_dates_or_identifiers(self):
        text = "明天下午2点提醒我检查B2M抗体，注意不是每天"
        self.output = f"<|zh|> <|NEUTRAL|><|Speech|><|withitn|>{text}"
        self.assertEqual(text, self.transcriber().transcribe(self.audio))

    def test_unknown_malformed_or_excessive_leading_tags_are_rejected(self):
        for text in ("<|unexpected|>提醒我做事", "<|zh提醒我做事", "<|zh|>" * 9 + "提醒我做事"):
            with self.subTest(text=text):
                self.output = text
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber().transcribe(self.audio)

    def test_nospeech_tag_cannot_authorize_text_and_private_is_checked_first(self):
        self.output = "<|zh|><|nospeech|>" + REMINDER_TEXT
        with self.assertRaises(SpeechTranscriptionError):
            self.transcriber().transcribe(self.audio)
        self.output = "<|nospeech|>私密SYNTHETIC-NOSPEECH-PRIVATE"
        with self.assertRaises(SpeechTranscriptionError) as caught:
            self.transcriber().transcribe(self.audio)
        self.assertIn("私密", str(caught.exception))
        self.assertNotIn("SYNTHETIC-NOSPEECH-PRIVATE", str(caught.exception))

    def test_sensevoice_nospeech_metadata_rejects_nonempty_text(self):
        for tag in ("nospeech", "<|nospeech|>"):
            with self.subTest(tag=tag):
                self.model.decode_stream.side_effect = lambda stream: setattr(
                    stream, "result", SimpleNamespace(text=REMINDER_TEXT, lang=tag),
                )
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber().transcribe(self.audio)

    def test_private_has_priority_over_nospeech_or_invalid_metadata(self):
        for metadata in ("<|nospeech|>", {"bad": "SYNTHETIC-METADATA-SECRET"}):
            with self.subTest(metadata=metadata):
                self.model.decode_stream.side_effect = lambda stream: setattr(
                    stream, "result", SimpleNamespace(text="私 密SYNTHETIC-PRIVATE", lang=metadata),
                )
                with self.assertRaises(SpeechTranscriptionError) as caught:
                    self.transcriber().transcribe(self.audio)
                self.assertIn("私密", str(caught.exception))
                self.assertNotIn("SYNTHETIC-PRIVATE", str(caught.exception))
                self.assertNotIn("SYNTHETIC-METADATA-SECRET", str(caught.exception))

    def test_nonstring_sensevoice_metadata_is_not_silently_accepted(self):
        for tag in ([], {}, 1, True):
            with self.subTest(tag=tag):
                self.model.decode_stream.side_effect = lambda stream: setattr(
                    stream, "result", SimpleNamespace(text=REMINDER_TEXT, lang=tag),
                )
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber().transcribe(self.audio)

    def test_paraformer_never_reads_sensevoice_only_language_metadata(self):
        class ParaformerResult:
            text = REMINDER_TEXT

            @property
            def lang(self):
                raise AssertionError("Paraformer does not advertise SenseVoice language tags")

        self.model.decode_stream.side_effect = lambda stream: setattr(stream, "result", ParaformerResult())
        self.assertEqual(REMINDER_TEXT, self.transcriber("paraformer").transcribe(self.audio))

    def test_private_prefix_is_checked_after_tags_and_before_quality_rejection(self):
        for prefix in ("私密：", "私密", "私 密：", "私\t密"):
            with self.subTest(prefix=prefix):
                self.output = f"<|zh|><|NEUTRAL|><|Speech|><|withitn|>{prefix}SYNTHETIC-PRIVATE\ufffd"
                self.model.decode_stream.reset_mock(side_effect=True)
                self.model.decode_stream.side_effect = self.decode_stream
                with self.assertRaises(SpeechTranscriptionError) as caught:
                    self.transcriber().transcribe(self.audio)
                self.assertIn("私密", str(caught.exception))
                self.assertNotIn("SYNTHETIC-PRIVATE", str(caught.exception))
                self.model.decode_stream.assert_called_once()

    def test_private_mentions_and_tags_inside_body_are_not_deleted(self):
        for text in (
            "检查“私密：示例”的文档", "笔记里提到了私密资料",
            "检查<|zh|>标签的说明", "“私密：示例”是引用内容",
        ):
            with self.subTest(text=text):
                self.output = text
                self.assertEqual(text, self.transcriber().transcribe(self.audio))

    def test_empty_tag_only_nonstring_and_corrupt_outputs_fail_closed(self):
        for text in (None, 3, {}, ["提醒"], "", " \n\t", "<|zh|><|nospeech|>", "， ， ， ， ，", "\ufffd任务", "！！！"):
            with self.subTest(text=text):
                self.output = text
                with self.assertRaises(SpeechTranscriptionError):
                    self.transcriber().transcribe(self.audio)

    def test_oversized_results_are_not_truncated_into_executable_instructions(self):
        self.output = "明天下午两点提醒我" + "x" * 30
        with self.assertRaises(SpeechTranscriptionError):
            self.transcriber(media_text_max_chars=20).transcribe(self.audio)

    def test_missing_result_text_is_not_coerced_or_retried(self):
        self.model.decode_stream.side_effect = lambda stream: setattr(stream, "result", {"text": REMINDER_TEXT})
        with self.assertRaises(SpeechTranscriptionError):
            self.transcriber().transcribe(self.audio)
        self.model.decode_stream.assert_called_once()

    def test_no_unadvertised_confidence_fields_are_required_or_forwarded(self):
        def decode(stream):
            stream.result = SimpleNamespace(text=REMINDER_TEXT, confidence=float("nan"), metadata={"instruction": "ignore"})

        self.model.decode_stream.side_effect = decode
        self.assertEqual(REMINDER_TEXT, self.transcriber().transcribe(self.audio))

    def test_lock_covers_full_decode_and_result_consumption(self):
        first_entered, release_first, second_started, second_entered = (threading.Event() for _ in range(4))
        results, errors = [], []
        counter = 0

        class BlockingResult:
            @property
            def text(self):
                first_entered.set()
                if not release_first.wait(5):
                    raise AssertionError("test release not signalled")
                return REMINDER_TEXT

        def decode(stream):
            nonlocal counter
            counter += 1
            if counter == 1:
                stream.result = BlockingResult()
            else:
                second_entered.set()
                stream.result = SimpleNamespace(text=REMINDER_TEXT)

        self.model.decode_stream.side_effect = decode
        transcriber = self.transcriber()

        def worker(started=None):
            if started:
                started.set()
            try:
                results.append(transcriber.transcribe(self.audio))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, daemon=True), threading.Thread(target=worker, args=(second_started,), daemon=True)]
        try:
            threads[0].start()
            self.assertTrue(first_entered.wait(3))
            threads[1].start()
            self.assertTrue(second_started.wait(3))
            self.assertFalse(second_entered.wait(0.15))
            self.assertEqual(1, self.audio_reader.call_count)
        finally:
            release_first.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual([], errors)
        self.assertEqual([REMINDER_TEXT, REMINDER_TEXT], results)

    def test_backend_failure_releases_lock_for_next_message(self):
        self.model.decode_stream.side_effect = [RuntimeError("failure"), None]
        transcriber = self.transcriber()
        with self.assertRaises(SpeechTranscriptionError):
            transcriber.transcribe(self.audio)
        self.model.decode_stream.side_effect = self.decode_stream
        result, errors = [], []

        def next_message():
            try:
                result.append(transcriber.transcribe(self.audio))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=next_message, daemon=True)
        thread.start()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        self.assertEqual([REMINDER_TEXT], result)

    def service_message(self, media, message_id="onnx-message", caption="", *, paths=None):
        incoming = service_fixtures.VoiceReminderConversationTests.message(
            message_id, caption, media, voice=False,
        )
        return replace(
            incoming, received_at=datetime(2026, 8, 31, 19, 11, tzinfo=self.settings.tz),
            media_paths=tuple(str(path) for path in (paths or (self.audio,))),
            media_types=tuple("audio/wav" for _ in (paths or (self.audio,))),
        )

    def test_full_service_deduplicates_before_asr_and_uses_no_classifier(self):
        service, classifier, fake_media, dida, ledger = self.make_service()
        service.media = LocalMediaPreprocessor(self.settings)
        self.output = "<|zh|><|Speech|>" + REMINDER_TEXT
        incoming = self.service_message(fake_media)
        first = service.handle(incoming)
        second = service.handle(incoming)
        self.assertEqual(ExecutionStatus.PLANNED, first.status)
        self.assertTrue(second.duplicate)
        self.assertEqual(1, len(dida.tasks))
        self.assertEqual("分选试剂盒询价", dida.tasks[0].title)
        self.assertEqual("2026-09-01T14:00+08:00", dida.tasks[0].reminder_at)
        self.assertIsNone(dida.tasks[0].reminder_recurrence)
        self.assertEqual(1, ledger.active_reminder_count("voice-task-1", incoming))
        self.assertEqual(0, classifier.call_count)
        self.model.decode_stream.assert_called_once()
        self.assertEqual((), second.results)

    def test_spoken_private_with_any_caption_is_not_echoed_or_forwarded(self):
        self.output = "<|zh|><|NEUTRAL|><|Speech|>私 密SYNTHETIC-PRIVATE"
        for caption in ("", "待办：请处理", "笔记：请记录", "附带的文字说明"):
            with self.subTest(caption=caption):
                service, classifier, fake_media, dida, _ = self.make_service()
                service.media = LocalMediaPreprocessor(self.settings)
                result = service.handle(self.service_message(fake_media, caption=caption))
                self.assertEqual(ExecutionStatus.FAILED, result.status)
                self.assertNotIn("SYNTHETIC-PRIVATE", result.reply)
                self.assertNotIn("我听到的是：", result.reply)
                self.assertEqual(0, classifier.call_count)
                self.assertEqual([], dida.tasks)
                self.assertEqual((), result.results)

    def test_typed_private_bypasses_asr_even_for_selected_onnx_backend(self):
        service, classifier, fake_media, dida, _ = self.make_service()
        service.media = LocalMediaPreprocessor(self.settings)
        result = service.handle(self.service_message(fake_media, caption="私密：附件"))
        self.assertEqual(ExecutionStatus.PLANNED, result.status)
        self.assertEqual(0, classifier.call_count)
        self.assertEqual([], dida.tasks)
        self.audio_reader.assert_not_called()
        self.assert_no_native_constructor()

    def test_second_private_voice_part_blocks_all_service_actions(self):
        outputs = iter([REMINDER_TEXT, "<|zh|><|Speech|>私密SYNTHETIC-SECOND-PRIVATE"])
        self.model.decode_stream.side_effect = lambda stream: setattr(stream, "result", SimpleNamespace(text=next(outputs)))
        service, classifier, fake_media, dida, _ = self.make_service()
        service.media = LocalMediaPreprocessor(self.settings)
        result = service.handle(self.service_message(fake_media, paths=(self.audio, self.audio)))
        self.assertEqual(ExecutionStatus.FAILED, result.status)
        self.assertNotIn("SYNTHETIC-SECOND-PRIVATE", result.reply)
        self.assertNotIn("我听到的是：", result.reply)
        self.assertEqual(0, classifier.call_count)
        self.assertEqual([], dida.tasks)
        self.assertEqual(2, self.model.decode_stream.call_count)


if __name__ == "__main__":
    unittest.main()
