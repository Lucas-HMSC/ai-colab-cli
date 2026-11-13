"""Testes da padronização da CLI (v0.3.2): start --model, aliases e --help."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ai_colab as cli


class StartStandardizationTests(unittest.TestCase):
    def test_start_code_models_and_legacy_alias(self):
        cases = [
            (["start", "code"], "code", "code"),
            (["start", "code", "--model", "qwen"], "code", "code"),
            (["start", "code", "-m", "qwen-next"], "code", "code-next"),
            (["start", "code", "--model", "code-next"], "code", "code-next"),
            (["start", "code-next"], "code-next", "code-next"),
        ]
        for argv, expected_mode, expected_variant in cases:
            with self.subTest(argv=argv):
                args = cli.parser_make().parse_args(argv)
                mode, variant = cli.resolve_start_target(args)
                self.assertEqual(mode, expected_mode)
                self.assertEqual(variant, expected_variant)

    def test_start_voice_and_tts_aliases(self):
        cases = [
            (["start", "voice"], "voice", "custom"),
            (["start", "voice", "--model", "qwen"], "voice", "custom"),
            (["start", "tts", "--model", "voxcpm2"], "voice", "voxcpm2"),
            (["start", "tts", "-m", "voxcpm"], "voice", "voxcpm2"),
            (["start", "tts", "-m", "chatterbox"], "voice", "chatterbox-br"),
            (["start", "voice", "-m", "kokoro"], "voice", "kokoro"),
            (["start", "voice", "--voice-model", "design"], "voice", "design"),
        ]
        for argv, expected_mode, expected_variant in cases:
            with self.subTest(argv=argv):
                args = cli.parser_make().parse_args(argv)
                mode, variant = cli.resolve_start_target(args)
                self.assertEqual(mode, expected_mode)
                self.assertEqual(variant, expected_variant)
                self.assertEqual(args.mode, "voice")
                self.assertEqual(args.voice_model, expected_variant)

    def test_start_transcribe_and_asr_aliases(self):
        cases = [
            (["start", "transcribe"], "transcribe", "whisper"),
            (["start", "transcribe", "--model", "whisper"], "transcribe", "whisper"),
            (["start", "asr", "--model", "qwen"], "transcribe", "qwen"),
            (["start", "asr", "-m", "qwen-asr"], "transcribe", "qwen"),
            (["start", "transcribe", "--asr-model", "qwen"], "transcribe", "qwen"),
        ]
        for argv, expected_mode, expected_variant in cases:
            with self.subTest(argv=argv):
                args = cli.parser_make().parse_args(argv)
                mode, variant = cli.resolve_start_target(args)
                self.assertEqual(mode, expected_mode)
                self.assertEqual(variant, expected_variant)
                self.assertEqual(args.mode, "transcribe")
                self.assertEqual(args.asr_model, expected_variant)

    def test_start_video_aliases(self):
        cases = [
            (["start", "video"], "video", "hunyuan15"),
            (["start", "video", "--model", "hunyuan"], "video", "hunyuan15"),
            (["start", "video", "--model", "wan"], "video", "wan22-5b"),
            (["start", "video", "-m", "wan22"], "video", "wan22-5b"),
            (["start", "video", "--video-model", "wan22-5b"], "video", "wan22-5b"),
        ]
        for argv, expected_mode, expected_variant in cases:
            with self.subTest(argv=argv):
                args = cli.parser_make().parse_args(argv)
                mode, variant = cli.resolve_start_target(args)
                self.assertEqual(mode, expected_mode)
                self.assertEqual(variant, expected_variant)
                self.assertEqual(args.mode, "video")
                self.assertEqual(args.video_model, expected_variant)

    def test_invalid_or_cross_modality_model_fails_before_gpu(self):
        args = cli.parser_make().parse_args(["start", "voice", "--model", "whisper"])
        with mock.patch.object(cli, "ensure_key") as key, mock.patch.object(cli, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "transcribe"):
                cli.do_start(args)
            key.assert_not_called()
            run.assert_not_called()

        args_bad = cli.parser_make().parse_args(["start", "video", "--model", "nao-existe"])
        with mock.patch.object(cli, "ensure_key") as key, mock.patch.object(cli, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "hunyuan15"):
                cli.do_start(args_bad)
            key.assert_not_called()
            run.assert_not_called()

    def test_do_start_uses_normalized_model_and_records_state(self):
        with tempfile.TemporaryDirectory() as d:
            fake_state = Path(d)
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                return cli.subprocess.CompletedProcess(cmd, 0, "")

            with mock.patch.object(cli, "STATE_DIR", fake_state), \
                 mock.patch.object(cli, "STATE_FILE", fake_state / "state.json"), \
                 mock.patch.object(cli, "ensure_key", return_value=fake_state / "id_ed25519"), \
                 mock.patch.object(cli, "cli_base", return_value=["/usr/bin/colab", "--auth=oauth2"]), \
                 mock.patch.object(cli.shutil, "which", return_value="/usr/bin/ssh"), \
                 mock.patch.object(cli, "remote_session_exists", side_effect=[False, True]), \
                 mock.patch.object(cli, "run", side_effect=fake_run), \
                 mock.patch.object(cli, "launch_process", return_value=777), \
                 mock.patch.object(cli, "tunnel_start", side_effect=lambda s: s), \
                 mock.patch.object(cli, "watchdog_start"), \
                 mock.patch.object(cli, "wait_ready"):
                args = cli.parser_make().parse_args(["start", "tts", "--model", "kokoro", "--port", "28400"])
                cli.do_start(args)
                saved = cli.state_read()
                self.assertEqual(saved["mode"], "voice")
                self.assertEqual(saved["voice_variant"], "kokoro")
                self.assertEqual(saved["model"], cli.VOICE_MODELS["kokoro"])
                self.assertEqual(saved["gpu"], "T4")


class SymmetryAndCatalogTests(unittest.TestCase):
    def test_tts_alias_maps_to_voice_and_models(self):
        gen = cli.parser_make().parse_args(["tts", "generate", "--text", "Oi", "--output", "/tmp/out.wav"])
        self.assertEqual(gen.func, cli.do_voice)
        self.assertEqual(gen.action, "generate")

        models_args = cli.parser_make().parse_args(["tts", "models"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            models_args.func(models_args)
        text = out.getvalue()
        self.assertIn("voxcpm2", text)
        self.assertIn("kokoro", text)
        self.assertNotIn("higgs", text.lower())

        json_args = cli.parser_make().parse_args(["voice", "models", "--json"])
        jout = io.StringIO()
        with contextlib.redirect_stdout(jout):
            json_args.func(json_args)
        payload = json.loads(jout.getvalue())
        self.assertIn("voice", payload["modalities"])
        self.assertNotIn("higgs3", payload["modalities"]["voice"]["models"])

    def test_asr_alias_maps_to_transcribe_and_models(self):
        args = cli.parser_make().parse_args(["asr", "--file", "som.wav"])
        self.assertEqual(args.func, cli.do_transcribe)
        self.assertEqual(args.file, "som.wav")

        models_args = cli.parser_make().parse_args(["asr", "models"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            models_args.func(models_args)
        text = out.getvalue()
        self.assertIn("whisper", text)
        self.assertIn("qwen", text)

        json_args = cli.parser_make().parse_args(["transcribe", "models", "--json"])
        jout = io.StringIO()
        with contextlib.redirect_stdout(jout):
            json_args.func(json_args)
        payload = json.loads(jout.getvalue())
        self.assertIn("transcribe", payload["modalities"])
        self.assertIn("whisper", payload["modalities"]["transcribe"]["models"])

        missing_file = cli.parser_make().parse_args(["asr"])
        with self.assertRaisesRegex(RuntimeError, "--file"):
            missing_file.func(missing_file)

    def test_code_alias_maps_to_claude(self):
        args = cli.parser_make().parse_args(["code", "--", "-p", "OK"])
        self.assertEqual(args.func, cli.do_claude)
        self.assertEqual(args.extra, ["--", "-p", "OK"])

    def test_global_models_lists_all_modalities_and_supports_filter_and_json(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.do_models(None)
        full = out.getvalue()
        for expected in ("code", "voice", "transcribe", "video", "hunyuan15", "wan22-5b", "whisper", "voxcpm2", "start "):
            self.assertIn(expected, full)
        self.assertNotIn("higgs", full.lower())

        v_out = io.StringIO()
        v_args = cli.parser_make().parse_args(["models", "video"])
        with contextlib.redirect_stdout(v_out):
            v_args.func(v_args)
        self.assertIn("hunyuan15", v_out.getvalue())
        self.assertIn("wan22-5b", v_out.getvalue())
        self.assertNotIn("voxcpm2", v_out.getvalue())

        j_out = io.StringIO()
        j_args = cli.parser_make().parse_args(["models", "--json"])
        with contextlib.redirect_stdout(j_out):
            j_args.func(j_args)
        payload = json.loads(j_out.getvalue())
        self.assertEqual(set(payload["modalities"]), {"code", "voice", "transcribe", "video"})


class HelpDocumentationTests(unittest.TestCase):
    def _capture_help(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                cli.parser_make().parse_args(argv)
            self.assertEqual(cm.exception.code, 0)
        return buf.getvalue()

    def test_no_args_prints_rich_help_and_returns_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main([])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        for token in ("code", "voice", "transcribe", "video", "qwen", "voxcpm2", "whisper", "hunyuan15", "wan22-5b", "--model"):
            self.assertIn(token, text)
        self.assertNotIn("higgs", text.lower())

    def test_subcommand_helps_document_models_examples_and_hide_legacy_flags(self):
        start_help = self._capture_help(["start", "--help"])
        self.assertIn("--model", start_help)
        self.assertIn("hunyuan15", start_help)
        self.assertIn("voxcpm2", start_help)
        self.assertNotIn("--voice-model", start_help)
        self.assertNotIn("--asr-model", start_help)
        self.assertNotIn("--video-model", start_help)
        self.assertNotIn("higgs", start_help.lower())

        for subcmd, expected_tokens in [
            (["voice", "--help"], ("custom", "design", "clone", "voxcpm2", "kokoro", "generate")),
            (["transcribe", "--help"], ("whisper", "qwen", "--file", "models")),
            (["video", "--help"], ("hunyuan15", "wan22-5b", "generate", "jobs")),
            (["models", "--help"], ("code", "voice", "transcribe", "video", "--json")),
            (["code", "--help"], ("ai-colab start code", "COLAB_AI_VISION_GUIDANCE")),
        ]:
            with self.subTest(subcmd=subcmd):
                text = self._capture_help(subcmd)
                for token in expected_tokens:
                    self.assertIn(token, text)
                self.assertNotIn("higgs", text.lower())


if __name__ == "__main__":
    unittest.main()
