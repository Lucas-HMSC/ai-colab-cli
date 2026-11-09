"""Regressões para falhas CUDA no primeiro encode (sem precisar GPU)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load_worker():
    read=Path.read_text
    def patched(path,*a,**kw):
        if str(path)=='/content/colab-ai-config.json':
            return json.dumps({'model':'fake/whisper', 'asr_variant':'whisper'})
        return read(path,*a,**kw)
    spec=importlib.util.spec_from_file_location('whisper_cuda_worker',ROOT/'remote/transcribe_api.py')
    mod=importlib.util.module_from_spec(spec)
    with mock.patch.object(Path,'read_text', new=patched):
        spec.loader.exec_module(mod)
    return mod

class Tests(unittest.TestCase):
    def test_cuda_requirement_and_loader(self):
        src=(ROOT/'remote/transcribe_bootstrap.py').read_text()
        self.assertIn("'nvidia-cublas-cu12'",src)
        self.assertIn("'nvidia-cudnn-cu12>=9,<10'",src)
        self.assertIn('LD_LIBRARY_PATH',src)
        self.assertIn('libdirs',src)
        self.assertIn('.colab-asr-v028-ready',src)

    def test_lazy_generator_is_forced_before_ready(self):
        m=load_worker()
        engine=mock.Mock()
        calls=[]
        def deferred():
            calls.append('iterated')
            raise RuntimeError('Library libcublas.so.12 is not found or cannot be loaded')
            yield None
        engine.transcribe.return_value=(deferred(),mock.Mock())
        with self.assertRaisesRegex(RuntimeError,'libcublas'):
            m.whisper_preflight(engine)
        self.assertEqual(calls,['iterated'])
        waveform=engine.transcribe.call_args.args[0]
        self.assertEqual(waveform.dtype,np.float32)
        self.assertTrue(np.all(waveform==0))
        self.assertFalse(engine.transcribe.call_args.kwargs['vad_filter'])

    def test_preflight_success_and_first_encode(self):
        m=load_worker()
        engine=mock.Mock()
        engine.transcribe.return_value=(iter([]),mock.Mock())
        m.whisper_preflight(engine)
        engine.transcribe.assert_called_once()

    def test_cuda_errors_classified_without_audio(self):
        m=load_worker()
        for message,code in [
            ('Library libcublas.so.12 is not found or cannot be loaded','cuda-libraries'),
            ('Library libcudnn.so.9 cannot be loaded','cuda-libraries'),
            ('CUDA failed with error out of memory','gpu-memory'),
            ('cuBLAS failed with status CUBLAS_STATUS_NOT_SUPPORTED','cuda-kernel'),
            ('an unrelated backend failure','inference')]:
            self.assertEqual(m.classify_whisper_failure(RuntimeError(message))[0],code)

    def test_qwen_no_preflight_change(self):
        src=(ROOT/'remote/transcribe_api.py').read_text()
        self.assertIn("if VARIANT == 'whisper':\n        try:\n            whisper_preflight(loaded)",src)
        self.assertIn("forced_aligner='Qwen/Qwen3-ForcedAligner-0.6B'",src)

if __name__=='__main__':
    unittest.main()
