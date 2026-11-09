"""Regressões Whisper/PyAV 19 e Qwen-ASR sem Colab e sem GPU."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]


def load_asr(variant):
    original = Path.read_text
    def fake_read(path, *args, **kwargs):
        if str(path) == '/content/colab-ai-config.json':
            return json.dumps({'model': 'teste/model', 'asr_variant': variant})
        return original(path, *args, **kwargs)
    spec = importlib.util.spec_from_file_location('asr_test_v027_' + variant, ROOT / 'remote/transcribe_api.py')
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(Path, 'read_text', new=fake_read):
        spec.loader.exec_module(module)
    return module


def make_wav(path, samples, rate=16000, nchannels=1):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(nchannels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples)


class WhisperPyAVRegression(unittest.TestCase):
    def test_whisper_wav_to_float32_no_pyav(self):
        import numpy as np
        mod = load_asr('whisper')
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / 'sample.wav'
            vals = np.array([0, 32767, -32768, 16384, -16384], dtype='<i2')
            make_wav(wav, vals.tobytes())
            audio = mod.whisper_pcm16_to_array(wav)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (5,))
        self.assertAlmostEqual(float(audio[1]), 32767/32768, places=5)
        self.assertAlmostEqual(float(audio[2]), -1, places=5)

    def test_whisper_passes_array_not_path_to_engine(self):
        import numpy as np
        from types import SimpleNamespace
        mod = load_asr('whisper')
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / 'sample.wav'
            make_wav(wav, np.array([10]*16000,dtype='<i2').tobytes())
            fake = mock.Mock()
            fake.transcribe.return_value = (
                iter([SimpleNamespace(start=0.0,end=.9,text=' Olá!')]),
                SimpleNamespace(language='pt'))
            response = mod.infer(fake,wav,1.0)
            args,kwargs = fake.transcribe.call_args
            self.assertIsInstance(args[0],np.ndarray)
            self.assertEqual(args[0].dtype,np.float32)
            self.assertEqual(kwargs['language'],'pt')
            self.assertEqual(response['text'],'Olá!')
            self.assertEqual(response['segments'][0]['end'],.9)

    def test_wav_format_guard(self):
        import numpy as np
        mod = load_asr('whisper')
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td)/'stereo.wav'
            make_wav(bad,np.array([1,2,3,4],dtype='<i2').tobytes(),nchannels=2)
            with self.assertRaisesRegex(ValueError,'PCM16 mono'):
                mod.whisper_pcm16_to_array(bad)
            wrong_rate = Path(td)/'wrong.wav'
            make_wav(wrong_rate,np.array([1,2,3,4],dtype='<i2').tobytes(),rate=8000)
            with self.assertRaisesRegex(ValueError,'PCM16 mono'):
                mod.whisper_pcm16_to_array(wrong_rate)

    def test_qwen_inference_unchanged(self):
        from types import SimpleNamespace
        mod = load_asr('qwen')
        result = SimpleNamespace(text='Tudo certo',language='Portuguese',time_stamps=None)
        model = mock.Mock(transcribe=mock.Mock(return_value=[result]))
        response = mod.infer(model,Path('/tmp/sample.wav'),2.5)
        self.assertEqual(response['text'],'Tudo certo')
        model.transcribe.assert_called_once_with(audio='/tmp/sample.wav',language='Portuguese',return_time_stamps=True)

    def test_dependency_pin_and_normalization(self):
        bootstrap = (ROOT/'remote/transcribe_bootstrap.py').read_text()
        worker = (ROOT/'remote/transcribe_api.py').read_text()
        self.assertIn("'av>=11,<19'",bootstrap)
        self.assertIn('.colab-asr-v027-ready',bootstrap)
        self.assertIn("'-c:a', 'pcm_s16le'",worker)
        self.assertIn('whisper_pcm16_to_array(audio_path)', worker)

if __name__=='__main__':
    unittest.main()
