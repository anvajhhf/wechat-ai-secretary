"""Independent speech boundary tests: no real models, downloads or user audio."""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from wechat_secretary.config import SecretarySettings
from wechat_secretary.speech import LocalSpeechTranscriber, SpeechTranscriptionError


TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"


class VoiceSafetyRound3Tests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = (TEST_ROOT / f"voice-safety-{uuid4().hex}").resolve()
        self.root.mkdir()
        self.addCleanup(self.cleanup_fixture)
        self.settings = SecretarySettings(
            project_root=self.root,
            voice_asr_enabled=True,
            asr_model="small",
            asr_language="zh",
        )
        self.cache = self.root / "runtime" / "models" / "huggingface" / "hub"
        self.snapshot = self.cache / "models--Systran--faster-whisper-small" / "snapshots" / "fixture"
        self.snapshot.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (self.snapshot / name).write_bytes(b"synthetic-cache-fixture")
        self.audio = self.root / "synthetic.wav"
        self.write_wav(self.audio, seconds=6)

        # Replace the import boundary itself. Importing or constructing a real
        # model is forbidden even if an implementation accidentally changes.
        self.fake_whisper = ModuleType("faster_whisper")
        self.fake_utils = ModuleType("faster_whisper.utils")
        self.fake_utils.download_model = Mock(return_value=str(self.snapshot))
        self.model = SimpleNamespace(transcribe=Mock(side_effect=self.fake_transcribe))
        self.fake_whisper.WhisperModel = Mock(return_value=self.model)
        self.fake_whisper.utils = self.fake_utils
        modules_patch = patch.dict(sys.modules, {
            "faster_whisper": self.fake_whisper,
            "faster_whisper.utils": self.fake_utils,
            "tools.transcription_tools": None,
        })
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def cleanup_fixture(self):
        if self.root.parent != TEST_ROOT.resolve() or not self.root.name.startswith("voice-safety-"):
            raise AssertionError("refusing cleanup outside the owned voice test fixture")
        shutil.rmtree(self.root)

    @staticmethod
    def write_wav(path: Path, *, seconds: float):
        # A genuine header exercises duration gating while inference remains
        # completely mocked. These zero samples are never sent to a model.
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(b"\x00\x00" * round(seconds * 16_000))

    @staticmethod
    def segment(text="缓存语音结果", *, no_speech_prob=0.02, avg_logprob=-0.2, compression_ratio=1.0):
        return SimpleNamespace(
            text=text,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )

    def fake_transcribe(self, *args, **kwargs):
        return iter([self.segment()]), SimpleNamespace(language="zh", duration=1.0)

    def test_model_resolution_and_constructor_are_explicitly_offline(self):
        transcriber = LocalSpeechTranscriber(self.settings)
        self.assertEqual("缓存语音结果", transcriber.transcribe(self.audio))
        self.fake_utils.download_model.assert_called_once()
        resolved_call = self.fake_utils.download_model.call_args
        self.assertIs(True, resolved_call.kwargs.get("local_files_only"))
        self.assertEqual(self.cache.resolve(), Path(resolved_call.kwargs["cache_dir"]).resolve())
        constructor = self.fake_whisper.WhisperModel.call_args
        self.assertEqual(self.snapshot.resolve(), Path(constructor.args[0]).resolve())
        self.assertIs(True, constructor.kwargs.get("local_files_only"))
        self.assertEqual("cpu", constructor.kwargs.get("device"))
        self.assertEqual("int8", constructor.kwargs.get("compute_type"))

    def test_model_names_cannot_be_paths_urls_or_shell_fragments(self):
        for name in (
            "../small", "..\\small", "D:\\models\\small", "/models/small",
            "https://example.invalid/model", "Systran/faster-whisper-small",
            "small;echo", "small\nbase", "small --other", "a" * 41,
        ):
            with self.subTest(name=name):
                with self.assertRaises(SpeechTranscriptionError):
                    LocalSpeechTranscriber(replace(self.settings, asr_model=name)).transcribe(self.audio)
        self.fake_utils.download_model.assert_not_called()
        self.fake_whisper.WhisperModel.assert_not_called()

    def test_invalid_language_fails_before_cache_lookup_or_model_load(self):
        for language in ("", "zh-CN", "../zh", "zh en", "zh;exec", "zh\nen", "abcd"):
            with self.subTest(language=language):
                with self.assertRaises(SpeechTranscriptionError):
                    LocalSpeechTranscriber(replace(self.settings, asr_language=language)).transcribe(self.audio)
        self.fake_utils.download_model.assert_not_called()
        self.fake_whisper.WhisperModel.assert_not_called()

    def test_auto_language_is_an_explicit_none_without_environment_changes(self):
        before = dict(os.environ)
        LocalSpeechTranscriber(replace(self.settings, asr_language="auto")).transcribe(self.audio)
        self.assertIsNone(self.model.transcribe.call_args.kwargs["language"])
        self.assertEqual(before, dict(os.environ))

    def test_missing_tokenizer_refuses_before_any_model_is_created(self):
        (self.snapshot / "tokenizer.json").unlink()
        with self.assertRaises(SpeechTranscriptionError):
            LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.fake_whisper.WhisperModel.assert_not_called()
        self.model.transcribe.assert_not_called()

    def test_empty_model_refuses_before_constructor(self):
        (self.snapshot / "model.bin").write_bytes(b"")
        with self.assertRaises(SpeechTranscriptionError):
            LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.fake_whisper.WhisperModel.assert_not_called()

    def test_cache_resolver_cannot_return_a_directory_outside_model_cache(self):
        unrelated = self.root / "unrelated-cache"
        unrelated.mkdir()
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (unrelated / name).write_bytes(b"synthetic-cache-fixture")
        self.fake_utils.download_model.return_value = str(unrelated)
        with self.assertRaises(SpeechTranscriptionError):
            LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.fake_whisper.WhisperModel.assert_not_called()

    def test_no_environment_mutation_or_hermes_fallback(self):
        before = dict(os.environ)
        transcriber = LocalSpeechTranscriber(self.settings)
        self.assertEqual("缓存语音结果", transcriber.transcribe(self.audio))
        self.assertEqual(before, dict(os.environ))
        # tools.transcription_tools is intentionally unavailable in setUp.
        self.assertIsNone(sys.modules["tools.transcription_tools"])

    def test_decoding_is_explicit_and_does_not_seed_schedule_words(self):
        LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        kwargs = self.model.transcribe.call_args.kwargs
        self.assertEqual("zh", kwargs.get("language"))
        self.assertEqual(5, kwargs.get("beam_size"))
        self.assertEqual(0, kwargs.get("temperature"))
        self.assertIs(False, kwargs.get("condition_on_previous_text"))
        self.assertIs(True, kwargs.get("vad_filter"))
        prompt = str(kwargs.get("initial_prompt", ""))
        self.assertNotIn("明天", prompt)
        self.assertNotIn("每天", prompt)

    def test_short_wav_utterances_do_not_use_domain_prompt(self):
        for seconds in (1.8, 2.96, 4.0):
            with self.subTest(seconds=seconds):
                self.write_wav(self.audio, seconds=seconds)
                LocalSpeechTranscriber(self.settings).transcribe(self.audio)
                self.assertNotIn("initial_prompt", self.model.transcribe.call_args.kwargs)

    def test_longer_wav_utterance_retains_static_domain_prompt(self):
        self.write_wav(self.audio, seconds=4.1)
        LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.assertIn("试剂盒", self.model.transcribe.call_args.kwargs.get("initial_prompt", ""))

    def test_unknown_audio_duration_does_not_enable_domain_prompt(self):
        opaque = self.root / "synthetic.ogg"
        opaque.write_bytes(b"not-a-real-ogg-model-is-mocked")
        LocalSpeechTranscriber(self.settings).transcribe(opaque)
        self.assertNotIn("initial_prompt", self.model.transcribe.call_args.kwargs)

    def test_prompted_corrupt_text_gets_at_most_one_unprompted_recovery(self):
        bad_segments = (
            self.segment("任务\ufffd内容"),
            self.segment("，，，，，，，，，，，，，，"),
            self.segment("， ， ， ， ， ，"),
            self.segment("、、、、、"),
            self.segment("异常压缩的输出", compression_ratio=10.0),
        )
        for bad in bad_segments:
            with self.subTest(bad=bad):
                self.model.transcribe.reset_mock()
                self.model.transcribe.side_effect = [
                    (iter([bad]), None),
                    (iter([self.segment("不是每天，是明天")]), None),
                ]
                self.assertEqual("不是每天，是明天", LocalSpeechTranscriber(self.settings).transcribe(self.audio))
                self.assertEqual(2, self.model.transcribe.call_count)
                first, second = self.model.transcribe.call_args_list
                self.assertIn("initial_prompt", first.kwargs)
                self.assertNotIn("initial_prompt", second.kwargs)
                self.assertEqual(first.args, second.args)

    def test_recovery_candidate_must_pass_the_same_quality_gate(self):
        self.model.transcribe.side_effect = [
            (iter([self.segment("首轮\ufffd")]), None),
            (iter([self.segment("恢复\ufffd")]), None),
        ]
        with self.assertRaises(SpeechTranscriptionError):
            LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.assertEqual(2, self.model.transcribe.call_count)

    def test_short_unprompted_corruption_does_not_start_an_extra_attempt(self):
        self.write_wav(self.audio, seconds=2.96)
        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (iter([self.segment("短句\ufffd")]), None)
        with self.assertRaises(SpeechTranscriptionError):
            LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.assertEqual(1, self.model.transcribe.call_count)

    def test_backend_exception_is_not_retried_as_a_prompt_problem(self):
        self.model.transcribe.side_effect = RuntimeError("synthetic-backend-error")
        with self.assertRaises(SpeechTranscriptionError):
            LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.assertEqual(1, self.model.transcribe.call_count)

    def test_recovery_cannot_erase_a_private_prefix_from_the_first_candidate(self):
        from wechat_secretary.prefixes import parse_prefix

        self.model.transcribe.side_effect = [
            (iter([self.segment("私密：SYNTHETIC-PRIVATE\ufffd")]), None),
            (iter([self.segment("提醒我公开记录")]), None),
        ]
        try:
            result = LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        except SpeechTranscriptionError as exc:
            self.assertNotIn("SYNTHETIC-PRIVATE", str(exc))
        else:
            self.assertTrue(parse_prefix(result, speech=True).private, "recovery discarded a known private prefix")
        self.assertEqual(1, self.model.transcribe.call_count)

    def test_media_preserves_separate_voice_parts_for_the_privacy_gate(self):
        from wechat_secretary.media import LocalMediaPreprocessor

        media = LocalMediaPreprocessor(self.settings)
        texts = ("普通备忘录", "私密：SYNTHETIC-PRIVATE")
        incoming = SimpleNamespace(
            media_paths=("first.wav", "second.wav"),
            media_types=("audio/wav", "audio/wav"),
        )
        with patch.object(media, "_transcribe_audio", side_effect=[(texts[0], "a" * 64), (texts[1], "b" * 64)]):
            prepared = media.prepare(incoming)
        self.assertEqual(texts, prepared.transcript_parts)
        self.assertIn("[语音转写 2]", prepared.transcript_text)

    def test_language_settings_do_not_leak_between_instances(self):
        chinese = LocalSpeechTranscriber(self.settings)
        english = LocalSpeechTranscriber(replace(self.settings, profile_id="partner", asr_language="en"))
        chinese.transcribe(self.audio)
        english.transcribe(self.audio)
        chinese.transcribe(self.audio)
        languages = [call.kwargs["language"] for call in self.model.transcribe.call_args_list]
        self.assertEqual(["zh", "en", "zh"], languages)

    def test_successive_utterances_do_not_become_prompt_history(self):
        transcriber = LocalSpeechTranscriber(self.settings)
        secret_body = "SYNTHETIC-PRIVATE-NOT-A-PROMPT"
        self.model.transcribe.side_effect = lambda *a, **k: (iter([self.segment(secret_body)]), None)
        self.assertEqual(secret_body, transcriber.transcribe(self.audio))
        transcriber.transcribe(self.audio)
        first, second = self.model.transcribe.call_args_list
        self.assertEqual(first.kwargs, second.kwargs)
        self.assertNotIn(secret_body, str(second.kwargs))

    def test_empty_or_silence_only_transcript_is_not_accepted(self):
        for segments in ([], [self.segment(" ")], [self.segment("幻听", no_speech_prob=0.95, avg_logprob=-2.0)]):
            with self.subTest(segments=segments):
                self.model.transcribe.side_effect = None
                self.model.transcribe.return_value = (iter(segments), None)
                with self.assertRaises(SpeechTranscriptionError):
                    LocalSpeechTranscriber(self.settings).transcribe(self.audio)

    def test_nonfinite_or_invalid_confidence_is_not_accepted(self):
        for name in ("no_speech_prob", "avg_logprob", "compression_ratio"):
            for value in (float("nan"), float("inf"), float("-inf"), None, "invalid"):
                with self.subTest(field=name, value=value):
                    segment = self.segment()
                    setattr(segment, name, value)
                    self.model.transcribe.side_effect = None
                    self.model.transcribe.return_value = (iter([segment]), None)
                    with self.assertRaises(SpeechTranscriptionError):
                        LocalSpeechTranscriber(self.settings).transcribe(self.audio)

    def test_out_of_range_probabilities_do_not_trigger_recovery(self):
        for name, value in (("no_speech_prob", -0.1), ("no_speech_prob", 1.1), ("compression_ratio", -1.0)):
            with self.subTest(field=name, value=value):
                segment = self.segment()
                setattr(segment, name, value)
                self.model.transcribe.reset_mock()
                self.model.transcribe.side_effect = None
                self.model.transcribe.return_value = (iter([segment]), None)
                with self.assertRaises(SpeechTranscriptionError):
                    LocalSpeechTranscriber(self.settings).transcribe(self.audio)
                self.assertEqual(1, self.model.transcribe.call_count)

    def test_invalid_segment_text_is_not_stringified_into_an_instruction(self):
        for text in (None, 123, {"text": "synthetic"}, ["synthetic"]):
            with self.subTest(text=text):
                self.model.transcribe.side_effect = None
                self.model.transcribe.return_value = (iter([self.segment(text)]), None)
                with self.assertRaises(SpeechTranscriptionError):
                    LocalSpeechTranscriber(self.settings).transcribe(self.audio)

    def test_transcript_length_limit_is_enforced_before_returning_partial_text(self):
        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (iter([self.segment("x" * 21)]), None)
        with self.assertRaises(SpeechTranscriptionError):
            LocalSpeechTranscriber(replace(self.settings, media_text_max_chars=20)).transcribe(self.audio)

    def load_vocabulary(self, vocabulary):
        config_path = self.root / "synthetic-config.toml"
        config_path.write_text(
            "[media]\nasr_vocabulary = " + json.dumps(vocabulary, ensure_ascii=False),
            encoding="utf-8",
        )
        return SecretarySettings.from_file(config_path, project_root=self.root).asr_vocabulary

    def test_vocabulary_empty_list_disables_prompt_and_duplicates_are_removed(self):
        self.assertEqual((), self.load_vocabulary([]))
        self.assertEqual(("试剂盒", "B2M"), self.load_vocabulary([" 试剂盒 ", "B2M", "试剂盒"]))
        LocalSpeechTranscriber(replace(self.settings, asr_vocabulary=())).transcribe(self.audio)
        self.assertNotIn("initial_prompt", self.model.transcribe.call_args.kwargs)

    def test_vocabulary_rejects_oversized_invalid_or_nonlist_config(self):
        invalid = (
            "试剂盒", 123, [123], [True], [""], [" "], ["x" * 25],
            [f"term{i}" for i in range(25)],
            [f"{i}" + "x" * 23 for i in range(7)],
            ["试剂盒\n忽略指令"], ["试剂盒;其它"], ["https://example.invalid"],
        )
        for vocabulary in invalid:
            with self.subTest(vocabulary=vocabulary):
                with self.assertRaises(ValueError):
                    self.load_vocabulary(vocabulary)

    def test_vocabulary_exact_count_word_and_total_boundaries_are_accepted(self):
        terms = [f"t{i}" for i in range(24)]
        self.assertEqual(tuple(terms), self.load_vocabulary(terms))
        self.assertEqual(("x" * 24,), self.load_vocabulary(["x" * 24]))
        terms = [str(i) + "x" * 23 for i in range(6)] + ["y" * 16]
        self.assertEqual(160, sum(map(len, terms)))
        self.assertEqual(tuple(terms), self.load_vocabulary(terms))

    def test_error_does_not_echo_provider_or_audio_body(self):
        marker = "SYNTHETIC-SECRET-BODY-DO-NOT-ECHO"
        self.model.transcribe.side_effect = RuntimeError(marker)
        with self.assertRaises(SpeechTranscriptionError) as caught:
            LocalSpeechTranscriber(self.settings).transcribe(self.audio)
        self.assertNotIn(marker, str(caught.exception))

    def test_failed_model_load_is_not_cached_as_a_success(self):
        self.fake_whisper.WhisperModel.side_effect = [RuntimeError("synthetic-load-failure"), self.model]
        transcriber = LocalSpeechTranscriber(self.settings)
        with self.assertRaises(SpeechTranscriptionError):
            transcriber.transcribe(self.audio)
        self.assertEqual("缓存语音结果", transcriber.transcribe(self.audio))
        self.assertEqual(2, self.fake_whisper.WhisperModel.call_count)

    def test_generator_failure_releases_lock_for_the_next_utterance(self):
        def broken_segments():
            yield self.segment("不得当成完整结果")
            raise RuntimeError("synthetic-generator-failure")

        self.model.transcribe.side_effect = [
            (broken_segments(), None),
            (iter([self.segment()]), None),
        ]
        transcriber = LocalSpeechTranscriber(self.settings)
        with self.assertRaises(SpeechTranscriptionError):
            transcriber.transcribe(self.audio)
        results = []
        errors = []

        def followup():
            try:
                results.append(transcriber.transcribe(self.audio))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=followup, daemon=True)
        thread.start()
        thread.join(3)
        self.assertFalse(thread.is_alive(), "failed transcription retained its inference lock")
        self.assertEqual([], errors)
        self.assertEqual(["缓存语音结果"], results)

    def test_lock_covers_lazy_segment_consumption(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()
        errors = []
        results = []

        def lazy_transcribe(path, **kwargs):
            if str(path).endswith("second.wav"):
                second_entered.set()

            def segments():
                if str(path).endswith("first.wav"):
                    first_entered.set()
                    if not release_first.wait(5):
                        raise AssertionError("test release was not signalled")
                yield self.segment()

            return segments(), None

        transcriber = LocalSpeechTranscriber(self.settings)
        self.model.transcribe.side_effect = lazy_transcribe

        def worker(path, started=None):
            if started is not None:
                started.set()
            try:
                results.append(transcriber.transcribe(path))
            except BaseException as exc:
                errors.append(exc)

        first_path, second_path = self.root / "first.wav", self.root / "second.wav"
        self.write_wav(first_path, seconds=6)
        self.write_wav(second_path, seconds=6)
        threads = [
            threading.Thread(target=worker, args=(first_path,), daemon=True),
            threading.Thread(target=worker, args=(second_path, second_started), daemon=True),
        ]
        try:
            threads[0].start()
            self.assertTrue(first_entered.wait(3))
            threads[1].start()
            self.assertTrue(second_started.wait(3))
            self.assertFalse(second_entered.wait(0.15), "second inference entered while first generator was active")
        finally:
            release_first.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual([], errors)
        self.assertEqual(["缓存语音结果", "缓存语音结果"], results)
        self.fake_whisper.WhisperModel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
