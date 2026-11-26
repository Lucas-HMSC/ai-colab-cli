import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import wave
from pathlib import Path
from unittest import TestCase, mock


def _load_transcribe_api(conf_dict=None, suffix="f6b"):
    conf_data = json.dumps(
        conf_dict or {"model": "teste/model", "asr_variant": "whisper", "hf_token": "hf_test_123"}
    )
    original_read_text = Path.read_text

    def fake_read_text(path, *args, **kwargs):
        if str(path) == "/content/colab-ai-config.json":
            return conf_data
        return original_read_text(path, *args, **kwargs)

    module_name = f"asr_f6b_{suffix}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "remote/transcribe_api.py",
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(Path, "read_text", new=fake_read_text):
        spec.loader.exec_module(module)
    return module


def _write_pcm16_wav(path: Path, samples_count: int = 1600) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x01\x00" * samples_count)


class _FakeTensor:
    def __init__(self, array, shape=None):
        self._array = array
        self.dtype = "float32"
        self.shape = shape if shape is not None else (len(array),)

    def unsqueeze(self, dim):
        assert dim == 0
        return _FakeTensor(self._array, shape=(1, len(self._array)))


def _make_fake_torch(cuda_available=True):
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "float32"
    torch_mod.from_numpy = lambda arr: _FakeTensor(arr)
    torch_mod.device = lambda name: f"device:{name}"
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: cuda_available
    torch_mod.cuda = cuda_mod
    return torch_mod


class TranscribeF6bDiarizeTests(TestCase):
    def test_diarize_audio_passes_in_memory_waveform_dict_without_torchcodec(self):
        mod = _load_transcribe_api(suffix="waveform")
        fake_torch = _make_fake_torch(cuda_available=False)

        turn = types.SimpleNamespace(start=0.1, end=0.8)
        diarize_output = types.SimpleNamespace(speaker_diarization=[(turn, None, "SPEAKER_00")])
        captured = {}

        def fake_diarizer(inp):
            if isinstance(inp, str | Path):
                raise ValueError("torchcodec disabled: pass waveform dict")
            captured["inp"] = inp
            return diarize_output

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            _write_pcm16_wav(wav_path, 3200)
            with mock.patch.object(mod, "load_diarizer", return_value=fake_diarizer), \
                    mock.patch.dict(sys.modules, {"torch": fake_torch}):
                turns = mod.diarize_audio(wav_path)

        self.assertIn("inp", captured)
        self.assertIsInstance(captured["inp"], dict)
        self.assertEqual(captured["inp"]["sample_rate"], 16000)
        self.assertEqual(captured["inp"]["waveform"].shape, (1, 3200))
        self.assertEqual(captured["inp"]["waveform"].dtype, "float32")
        self.assertEqual(turns, [(0.1, 0.8, "SPEAKER_00")])

    def test_diarize_audio_preserves_itertracks_and_iterable_fallbacks(self):
        mod = _load_transcribe_api(suffix="itertracks")
        fake_torch = _make_fake_torch(cuda_available=False)

        turn1 = types.SimpleNamespace(start=0.0, end=0.5)
        turn2 = types.SimpleNamespace(start=0.5, end=1.0)

        class _Annotation:
            def itertracks(self, yield_label=False):
                self.yield_label = yield_label
                return [(turn1, "t1", "SPK_A")]

        ann_obj = _Annotation()
        iterable_obj = [(turn2, "t2", "SPK_B")]

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            _write_pcm16_wav(wav_path, 1600)
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with mock.patch.object(mod, "load_diarizer", return_value=lambda _inp: ann_obj):
                    res_ann = mod.diarize_audio(wav_path)
                with mock.patch.object(
                    mod, "load_diarizer", return_value=lambda _inp: iterable_obj
                ):
                    res_iter = mod.diarize_audio(wav_path)

        self.assertEqual(res_ann, [(0.0, 0.5, "SPK_A")])
        self.assertEqual(res_iter, [(0.5, 1.0, "SPK_B")])

    def test_load_diarizer_moves_pipeline_to_cuda_and_preserves_missing_token_value_error(self):
        mod_notok = _load_transcribe_api(
            {"model": "teste/model", "asr_variant": "whisper"}, suffix="notok"
        )
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
                mod_notok.load_diarizer()

        mod_cuda = _load_transcribe_api(suffix="cuda")
        fake_torch = _make_fake_torch(cuda_available=True)
        pipeline_obj = mock.MagicMock()
        pipeline_obj.to.return_value = pipeline_obj

        fake_pipeline_cls = mock.MagicMock()
        fake_pipeline_cls.from_pretrained.return_value = pipeline_obj
        fake_pyannote = types.ModuleType("pyannote")
        fake_pyannote_audio = types.ModuleType("pyannote.audio")
        fake_pyannote_audio.Pipeline = fake_pipeline_cls
        fake_pyannote.audio = fake_pyannote_audio

        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.dict(sys.modules, {
                    "torch": fake_torch,
                    "pyannote": fake_pyannote,
                    "pyannote.audio": fake_pyannote_audio,
                }):
            loaded = mod_cuda.load_diarizer()

        self.assertIs(loaded, pipeline_obj)
        pipeline_obj.to.assert_called_once_with("device:cuda")

        mod_str = _load_transcribe_api(suffix="strpipe")
        fake_pipeline_cls.from_pretrained.return_value = "pipeline-instance"
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.dict(sys.modules, {
                    "torch": fake_torch,
                    "pyannote": fake_pyannote,
                    "pyannote.audio": fake_pyannote_audio,
                }):
            loaded_str = mod_str.load_diarizer()
        self.assertEqual(loaded_str, "pipeline-instance")

    def test_infer_with_speakers_preserves_diarize_audio_path_call(self):
        mod = _load_transcribe_api(suffix="inferpath")
        seg = mock.MagicMock()
        seg.start, seg.end, seg.text, seg.words = 0.0, 1.0, "Olá", []
        engine = mock.MagicMock()
        engine.transcribe.return_value = (iter([seg]), mock.MagicMock(language="pt"))

        with (
            mock.patch.object(mod, "whisper_pcm16_to_array", return_value=object()),
            mock.patch.object(
                mod, "diarize_audio", return_value=[(0.0, 1.0, "SPK_0")]
            ) as diarize,
        ):
            payload = mod.infer(engine, "norm.wav", 1.0, diarize=True)

        diarize.assert_called_once_with("norm.wav")
        self.assertTrue(payload["has_speaker_labels"])

    def test_do_post_separates_422_input_errors_from_500_sanitized_infer_errors(self):
        mod = _load_transcribe_api(suffix="dopost")

        def _make_handler(body_bytes):
            with mock.patch.object(mod.BaseHTTPRequestHandler, "handle", lambda self: None):
                handler = mod.handler_for(object())(
                    mock.MagicMock(), ("127.0.0.1", 12345), mock.MagicMock()
                )
            handler.path = "/_colab_ai/transcribe"
            handler.headers = {
                "Content-Length": str(len(body_bytes) + 10),
                "X-Colab-Ai-Extension": ".wav",
                "X-Colab-Ai-Language": "pt",
                "X-Colab-Ai-Diarize": "true",
            }
            handler.rfile = io.BytesIO(body_bytes)
            return handler

        # 1) Upload truncado -> 422
        h_trunc = _make_handler(b"abc")
        with mock.patch.object(h_trunc, "reply") as reply:
            h_trunc.do_POST()
        self.assertEqual(reply.call_args.args[0], 422)

        # 2) duration_for ValueError / to_wav TimeoutExpired -> 422
        for side_effect in (ValueError("duracao invalida"), subprocess.TimeoutExpired("ffmpeg", 5)):
            h_in = _make_handler(b"1234")
            h_in.headers["Content-Length"] = "4"
            with mock.patch.object(mod, "duration_for", side_effect=side_effect), \
                    mock.patch.object(h_in, "reply") as reply:
                h_in.do_POST()
            self.assertEqual(reply.call_args.args[0], 422)

        # 3) infer() ValueError('detalhe sensivel') -> 500 sanitizado sem str(exc)
        h_infer = _make_handler(b"1234")
        h_infer.headers["Content-Length"] = "4"
        with mock.patch.object(mod, "duration_for", return_value=1.0), \
                mock.patch.object(mod, "to_wav", return_value=None), \
                mock.patch.object(mod, "infer", side_effect=ValueError("detalhe sensivel")), \
                mock.patch.object(h_infer, "reply") as reply:
            h_infer.do_POST()
        status, body = reply.call_args.args
        self.assertEqual(status, 500)
        self.assertEqual(body.get("error"), "Falha na transcrição")
        self.assertNotIn("detalhe sensivel", json.dumps(body, ensure_ascii=False))
