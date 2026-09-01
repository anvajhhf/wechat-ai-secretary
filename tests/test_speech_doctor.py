"""Backend readiness checks using only synthetic files and fake components."""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import shutil
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from wechat_secretary import cli
from wechat_secretary.config import SecretarySettings
from wechat_secretary.speech import LocalSpeechTranscriber
from wechat_secretary.speech_onnx import OnnxSpeechTranscriber


TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"
SECRET = "SYNTHETIC-SECRET-DO-NOT-ECHO"
NOISY_IMPORT_MARKER = "SYNTHETIC-NOISY-IMPORT-MARKER"


class SpeechDoctorTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = (TEST_ROOT / f"speech-doctor-{uuid4().hex}").resolve()
        self.root.mkdir()
        self.addCleanup(self.cleanup_fixture)
        self.settings = SecretarySettings(
            project_root=self.root, dry_run=True, allowed_users=frozenset({"fixture-user"}),
            voice_asr_enabled=True, asr_backend="paraformer", asr_language="zh",
        )
        self.assets = self.root / "fake-components" / "assets"
        self.assets.mkdir(parents=True)
        self.vad_asset = self.assets / "silero_vad_v6.onnx"
        self.vad_asset.write_bytes(b"synthetic-vad-not-a-real-model")
        self.cache = self.root / "runtime" / "models" / "huggingface" / "hub"
        self.snapshot = self.cache / "models--Systran--faster-whisper-small" / "snapshots" / "fixture"
        self.snapshot.mkdir(parents=True)
        for name in ("config.json", "model.bin", "tokenizer.json"):
            (self.snapshot / name).write_bytes(b"synthetic-cache-not-loaded")

        manifest = {}
        for backend in ("sensevoice", "paraformer"):
            folder = self.root / "runtime" / "models" / "speech" / backend
            folder.mkdir(parents=True)
            manifest[backend] = {}
            for name, content in (("model.int8.onnx", b"synthetic-onnx"), ("tokens.txt", b"synthetic-tokens")):
                (folder / name).write_bytes(content)
                if name == "tokens.txt":
                    digest = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
                    kind = "git-sha1"
                else:
                    digest, kind = hashlib.sha256(content).hexdigest(), "sha256"
                manifest[backend][name] = (len(content), digest, kind)
        self.enterContext(patch("wechat_secretary.speech_onnx.MODEL_ARTIFACTS", manifest))

        self.modules = {}
        for name in ("av", "numpy", "pysilk", "onnxruntime", "faster_whisper", "faster_whisper.utils", "faster_whisper.vad", "sherpa_onnx"):
            self.modules[name] = ModuleType(name)
        self.forbidden_calls = []

        def forbidden():
            mocked = Mock(side_effect=AssertionError("doctor must not process audio or load models"))
            self.forbidden_calls.append(mocked)
            return mocked
        self.modules["av"].open = forbidden()
        self.modules["av"].AudioResampler = forbidden()
        self.modules["numpy"].ascontiguousarray = forbidden()
        self.modules["numpy"].isfinite = forbidden()
        self.modules["pysilk"].decode = forbidden()
        self.modules["onnxruntime"].InferenceSession = forbidden()
        self.modules["onnxruntime"].get_available_providers = Mock(return_value=["CPUExecutionProvider"])
        self.modules["faster_whisper"].WhisperModel = forbidden()
        self.modules["faster_whisper.utils"].get_assets_path = Mock(return_value=str(self.assets))
        self.resolver = self.modules["faster_whisper.utils"].download_model = Mock(return_value=str(self.snapshot))
        self.modules["faster_whisper.vad"].VadOptions = forbidden()
        self.modules["faster_whisper.vad"].get_speech_timestamps = forbidden()
        self.modules["faster_whisper.vad"].get_vad_model = forbidden()
        self.modules["sherpa_onnx"].__version__ = "1.13.6"
        self.modules["sherpa_onnx"].OfflineRecognizer = SimpleNamespace(
            from_sense_voice=forbidden(), from_paraformer=forbidden(),
        )
        self.enterContext(patch.dict(sys.modules, self.modules))
        self.version = self.enterContext(patch("wechat_secretary.cli.importlib.metadata.version", return_value="1.13.6"))
        for kind in (LocalSpeechTranscriber, OnnxSpeechTranscriber):
            for method in ("_load_model", "transcribe"):
                self.forbidden_calls.append(self.enterContext(patch.object(kind, method, side_effect=AssertionError("ASR loading forbidden"))))
        for method in ("_read_audio", "_has_speech"):
            self.forbidden_calls.append(self.enterContext(patch.object(OnnxSpeechTranscriber, method, side_effect=AssertionError("Audio access forbidden"))))
        self.forbidden_calls.append(self.enterContext(patch("sqlite3.connect", side_effect=AssertionError("Database access forbidden"))))
        self.forbidden_calls.append(self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))))
        self.forbidden_calls.append(self.enterContext(patch("socket.create_connection", side_effect=AssertionError("Network forbidden"))))
        self.enterContext(patch.dict(os.environ, {"DEEPSEEK_API_KEY": SECRET, "SECRETARY_PROFILE": "owner"}, clear=True))

    def tearDown(self):
        # A forbidden call may raise into a production exception sanitizer;
        # assert its absence even when that particular case expects failure.
        for mocked in self.forbidden_calls:
            mocked.assert_not_called()

    def cleanup_fixture(self):
        if self.root.parent != TEST_ROOT.resolve() or not self.root.name.startswith("speech-doctor-"):
            raise AssertionError("refusing cleanup outside owned doctor fixture")
        shutil.rmtree(self.root)

    def check(self, backend="paraformer", **changes):
        return cli._speech_readiness(replace(self.settings, asr_backend=backend, **changes))

    def doctor(self, *, strict=True, settings=None):
        hermes = self.root / ".venv" / "Scripts" / "hermes.exe"
        hermes.parent.mkdir(parents=True, exist_ok=True)
        hermes.write_bytes(b"synthetic-never-executed")
        plugin = self.root / ".hermes" / "plugins" / "wechat-secretary" / "plugin.yaml"
        plugin.parent.mkdir(parents=True, exist_ok=True)
        plugin.write_text("synthetic: true", encoding="utf-8")
        capture = io.StringIO()
        with (
            patch.object(cli, "PROJECT_ROOT", self.root),
            patch.object(cli, "load_settings", return_value=settings or self.settings),
            patch.object(cli.shutil, "which", return_value=None),
            patch.object(cli.importlib.util, "find_spec", return_value=object()),
            redirect_stdout(capture),
        ):
            code = cli.command_doctor(argparse.Namespace(strict=strict))
        output = capture.getvalue()
        self.assertNotIn(SECRET, output)
        return code, output

    def test_disabled_voice_does_not_probe_components_versions_or_models(self):
        with (
            patch.object(cli.importlib, "import_module", side_effect=AssertionError("no voice imports")),
            patch.object(LocalSpeechTranscriber, "_resolve_model_path", side_effect=AssertionError("no cache lookup")),
            patch.object(OnnxSpeechTranscriber, "_resolve_model_files", side_effect=AssertionError("no model lookup")),
        ):
            self.assertEqual(("未启用", ()), self.check(voice_asr_enabled=False))
            code, output = self.doctor(settings=replace(self.settings, voice_asr_enabled=False))
        self.assertEqual(0, code)
        self.assertIn("本地语音转写：未启用", output)
        self.version.assert_not_called()
        self.resolver.assert_not_called()

    def test_whisper_uses_only_runtime_local_cache_without_sherpa(self):
        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            status, errors = self.check("whisper")
        self.assertEqual("Whisper：已就绪", status)
        self.assertEqual((), errors)
        self.resolver.assert_called_once_with("small", cache_dir=str(self.cache.resolve()), local_files_only=True)
        self.version.assert_not_called()

    def test_onnx_backends_need_no_whisper_weights_or_resolver(self):
        shutil.rmtree(self.cache)
        for backend, label in (("sensevoice", "SenseVoice"), ("paraformer", "Paraformer")):
            with self.subTest(backend=backend):
                self.assertEqual((f"{label}：已就绪", ()), self.check(backend))
        self.resolver.assert_not_called()
        self.assertEqual({"sherpa-onnx", "sherpa-onnx-core"}, {call.args[0] for call in self.version.call_args_list})

    def test_whisper_missing_selected_cache_is_not_hidden_by_hf_home(self):
        self.resolver.side_effect = FileNotFoundError(SECRET)
        with patch.dict(os.environ, {"HF_HOME": str(self.snapshot)}):
            status, errors = self.check("whisper")
        self.assertIn("未就绪", status)
        self.assertTrue(errors)
        self.assertNotIn(SECRET, "".join(errors))
        self.assertEqual(str(self.cache.resolve()), self.resolver.call_args.kwargs["cache_dir"])

    def test_whisper_outside_cache_or_incomplete_snapshot_is_rejected(self):
        self.resolver.return_value = str(self.root)
        self.assertTrue(self.check("whisper")[1])
        self.resolver.return_value = str(self.snapshot)
        for name in ("tokenizer.json", "config.json", "model.bin"):
            path = self.snapshot / name
            payload = path.read_bytes()
            path.write_bytes(b"")
            try:
                self.assertTrue(self.check("whisper")[1])
            finally:
                path.write_bytes(payload)

    def test_whisper_pathlike_model_setting_is_never_echoed(self):
        status, errors = self.check("whisper", asr_model=f"../{SECRET}")
        self.assertIn("未就绪", status)
        self.assertNotIn(SECRET, "".join(errors))
        self.resolver.assert_not_called()

    def test_invalid_backends_are_safe_and_do_not_probe_dependencies(self):
        for backend in (None, {}, "remote", f"../{SECRET}"):
            with self.subTest(backend=backend), patch.object(cli.importlib, "import_module", side_effect=AssertionError("invalid backend")):
                status, errors = self.check(backend)
                self.assertEqual("未就绪", status)
                self.assertTrue(errors)
                self.assertNotIn(SECRET, "".join(errors))

    def test_onnx_language_and_thread_settings_are_checked(self):
        for changes in ({"asr_language": "en"}, {"asr_threads": 0}, {"asr_threads": True}):
            with self.subTest(changes=changes):
                self.assertTrue(self.check(**changes)[1])

    def test_onnx_missing_or_tampered_model_and_tokens_fail(self):
        for name in ("model.int8.onnx", "tokens.txt"):
            path = self.root / "runtime" / "models" / "speech" / "paraformer" / name
            payload = path.read_bytes()
            for content in (None, b"X" * len(payload), b""):
                with self.subTest(name=name, content=content):
                    if content is None:
                        path.unlink()
                    else:
                        path.write_bytes(content)
                    self.assertTrue(self.check()[1])
                    path.write_bytes(payload)

    def test_selected_onnx_only_checks_its_own_model(self):
        shutil.rmtree(self.root / "runtime" / "models" / "speech" / "sensevoice")
        self.assertEqual(("Paraformer：已就绪", ()), self.check())

    def test_onnx_resolved_file_outside_model_root_fails(self):
        original = Path.resolve
        target = self.root / "runtime" / "models" / "speech" / "paraformer" / "model.int8.onnx"
        outside = self.root / "outside.onnx"
        outside.write_bytes(target.read_bytes())

        def redirect(path, *args, **kwargs):
            return outside if path == target else original(path, *args, **kwargs)

        with patch.object(Path, "resolve", redirect):
            self.assertTrue(self.check()[1])

    def test_each_shared_dependency_missing_blocks_strict_start(self):
        for name in ("av", "numpy", "pysilk", "onnxruntime", "faster_whisper"):
            with self.subTest(name=name), patch.dict(sys.modules, {name: None}):
                code, output = self.doctor()
                self.assertEqual(1, code)
                self.assertIn(name, output)
                self.assertIn("未就绪", output)

    def test_incomplete_component_api_is_not_reported_ready(self):
        for name, attribute in (("av", "AudioResampler"), ("numpy", "isfinite"), ("pysilk", "decode"), ("onnxruntime", "InferenceSession")):
            with self.subTest(name=name), patch.object(self.modules[name], attribute, None):
                self.assertTrue(self.check()[1])

    def test_native_component_import_failure_has_safe_output(self):
        real_import = cli.importlib.import_module

        def failed_import(name, *args, **kwargs):
            if name == "av":
                raise OSError(f"{SECRET}/private-library.dll")
            return real_import(name, *args, **kwargs)

        with patch.object(cli.importlib, "import_module", side_effect=failed_import):
            code, output = self.doctor()
        self.assertEqual(1, code)
        self.assertNotIn("private-library", output)

    def test_import_prints_warnings_and_prebound_logs_never_escape(self):
        real_import = cli.importlib.import_module
        logger = logging.getLogger(f"speech-doctor-fixture-{uuid4().hex}")
        logged = io.StringIO()
        handler = logging.StreamHandler(logged)
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.WARNING)
        self.addCleanup(logger.removeHandler, handler)
        for fail in (False, True):
            with self.subTest(fail=fail):
                original_stdout, original_stderr = sys.stdout, sys.stderr
                original_filters = list(warnings.filters)
                disabled = logging.root.manager.disable

                def noisy_import(name, *args, **kwargs):
                    if name == "av":
                        # This test checks blanket import-output suppression. A
                        # non-secret marker avoids teaching static analysis that
                        # the test itself intentionally logs credential data.
                        print(NOISY_IMPORT_MARKER)
                        print(NOISY_IMPORT_MARKER, file=sys.stderr)
                        warnings.warn(NOISY_IMPORT_MARKER)
                        logger.warning(NOISY_IMPORT_MARKER)
                        if fail:
                            raise OSError(NOISY_IMPORT_MARKER)
                    return real_import(name, *args, **kwargs)

                captured_err = io.StringIO()
                with patch.object(cli.importlib, "import_module", side_effect=noisy_import), redirect_stderr(captured_err):
                    code, output = self.doctor()
                self.assertEqual(int(fail), code)
                self.assertNotIn(
                    NOISY_IMPORT_MARKER,
                    output + captured_err.getvalue() + logged.getvalue(),
                )
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)
                self.assertEqual(original_filters, warnings.filters)
                self.assertEqual(disabled, logging.root.manager.disable)

    def test_new_import_log_handlers_do_not_retain_closed_capture_streams(self):
        real_import = cli.importlib.import_module
        logger = logging.getLogger(f"speech-doctor-import-{uuid4().hex}")
        handlers = []

        def register_handler(name, *args, **kwargs):
            if name == "av":
                handler = logging.StreamHandler(sys.stderr)
                handlers.append(handler)
                logger.addHandler(handler)
                self.addCleanup(logger.removeHandler, handler)
            return real_import(name, *args, **kwargs)

        original = sys.stderr
        with patch.object(cli.importlib, "import_module", side_effect=register_handler):
            self.assertEqual(("Paraformer：已就绪", ()), self.check())
        self.assertEqual(1, len(handlers))
        self.assertIs(handlers[0].stream, original)

    def test_vad_dependency_asset_and_readability_are_required_without_inference(self):
        for name in ("faster_whisper.utils", "faster_whisper.vad"):
            with self.subTest(name=name), patch.dict(sys.modules, {name: None}):
                self.assertTrue(self.check()[1])
        payload = self.vad_asset.read_bytes()
        self.vad_asset.unlink()
        self.assertTrue(self.check()[1])
        self.vad_asset.write_bytes(b"")
        self.assertTrue(self.check()[1])
        self.vad_asset.write_bytes(payload)
        real_open = Path.open

        def unreadable(path, *args, **kwargs):
            if path == self.vad_asset:
                raise PermissionError(SECRET)
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", unreadable):
            status, errors = self.check()
        self.assertIn("未就绪", status)
        self.assertNotIn(SECRET, "".join(errors))
        self.modules["faster_whisper.vad"].get_vad_model.assert_not_called()
        self.modules["faster_whisper.vad"].get_speech_timestamps.assert_not_called()

    def test_vad_asset_redirection_outside_package_is_rejected(self):
        original = Path.resolve
        outside = self.root / "outside-vad.onnx"
        outside.write_bytes(b"synthetic")

        def redirect(path, *args, **kwargs):
            return outside if path == self.vad_asset else original(path, *args, **kwargs)

        with patch.object(Path, "resolve", redirect):
            self.assertTrue(self.check()[1])

    def test_cpu_provider_missing_or_query_failure_is_not_ready(self):
        providers = self.modules["onnxruntime"].get_available_providers
        providers.return_value = ["CUDAExecutionProvider"]
        self.assertTrue(self.check()[1])
        providers.side_effect = RuntimeError(SECRET)
        status, errors = self.check()
        self.assertIn("未就绪", status)
        self.assertNotIn(SECRET, "".join(errors))

    def test_both_sherpa_distributions_are_pinned_and_missing_versions_fail(self):
        for missing in ("sherpa-onnx", "sherpa-onnx-core"):
            for version in (None, "1.13.5", "1.13.6.dev0"):
                with self.subTest(distribution=missing, version=version):
                    def installed(name):
                        if name == missing:
                            if version is None:
                                raise PackageNotFoundError(SECRET)
                            return version
                        return "1.13.6"
                    self.version.side_effect = installed
                    status, errors = self.check()
                    self.assertIn("未就绪", status)
                    self.assertTrue(any(missing in error for error in errors))
                    self.assertNotIn(SECRET, "".join(errors))

    def test_missing_sherpa_import_and_mismatched_binding_version_fail(self):
        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            self.assertTrue(self.check()[1])
        with patch.object(self.modules["sherpa_onnx"], "__version__", "1.13.5"):
            self.assertTrue(self.check()[1])
        with patch.object(self.modules["sherpa_onnx"].OfflineRecognizer, "from_paraformer", None):
            self.assertTrue(self.check()[1])

    def test_backend_probe_exception_is_not_echoed_and_does_not_fallback(self):
        with patch.object(OnnxSpeechTranscriber, "_resolve_model_files", side_effect=OSError(SECRET)):
            code, output = self.doctor()
        self.assertEqual(1, code)
        self.assertIn("Paraformer", output)
        self.assertNotIn("Whisper", output)
        self.resolver.assert_not_called()

    def test_strict_doctor_accepts_onnx_without_whisper_cache(self):
        shutil.rmtree(self.cache)
        code, output = self.doctor()
        self.assertEqual(0, code)
        self.assertIn("Paraformer：已就绪", output)
        self.assertNotIn("Whisper", output)
        self.resolver.assert_not_called()

    def test_nonstrict_doctor_reports_unready_without_adding_startup_blockers(self):
        self.vad_asset.unlink()
        code, output = self.doctor(strict=False)
        self.assertEqual(0, code)
        self.assertIn("未就绪", output)
        self.assertIn("语音待就绪项", output)
        self.assertNotIn("阻止项", output)


if __name__ == "__main__":
    unittest.main()
