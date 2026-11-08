"""Regressão: Higgs suspenso no catálogo, sem provisionar GPU."""
import importlib.util, io, contextlib, unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("ai_colab_v025",ROOT/"ai_colab.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

class CatalogV025Tests(unittest.TestCase):
    def test_disabled_higgs_not_offered(self):
        self.assertNotIn("higgs3",cli.ACTIVE_VOICE_MODELS)
        output=io.StringIO()
        with contextlib.redirect_stdout(output): cli.do_models(None)
        self.assertNotIn("higgs",output.getvalue().lower())
        for model in ("custom","design","clone","voxcpm2","chatterbox-br","kokoro"):
            self.assertIn(model, cli.ACTIVE_VOICE_MODELS)
    def test_higgs_cannot_create_gpu(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.parser_make().parse_args(["start","voice","--voice-model","higgs3","--gpu","G4"])
        with mock.patch.object(cli,"run") as run:
            arg=cli.parser_make().parse_args(["start","voice","--voice-model","kokoro","--gpu","T4"])
            arg.voice_model="higgs3" # even forged input cannot provision
            with self.assertRaisesRegex(RuntimeError,"desativado"):
                cli.do_start(arg)
            run.assert_not_called()
    def test_transcribe_registered_in_cli_without_calling_gpu(self):
        args = cli.parser_make().parse_args(["start", "transcribe", "--asr-model", "whisper"])
        self.assertEqual(args.asr_model, "whisper")
        self.assertEqual(cli.ASR_DEFAULT_GPU["whisper"], "T4")
        self.assertEqual(cli.ASR_DEFAULT_GPU["qwen"], "L4")
        with mock.patch.object(cli, "ensure_key", side_effect=RuntimeError("pré-verificação")):
            with mock.patch.object(cli, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "pré-verificação"):
                    cli.do_start(args)
                run.assert_not_called()
if __name__=="__main__": unittest.main()
