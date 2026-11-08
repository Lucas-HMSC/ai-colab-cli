"""Verifica geração de WAV com uma versão simulada do Qwen: NÃO usa GPU."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def generate_custom_voice(self, **kw):
        self.last = kw
        return [np.zeros(8000, dtype=np.float32)], 16000

    def generate_voice_design(self, **kw):
        return [np.zeros(8000, dtype=np.float32)], 16000

    def generate_voice_clone(self, **kw):
        self.last = kw
        return [np.zeros(8000, dtype=np.float32)], 16000


class VoiceServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeModel()
        fq = types.ModuleType('qwen_tts')
        class Factory:
            @staticmethod
            def from_pretrained(*args, **kw):
                return cls.fake
        fq.Qwen3TTSModel = Factory
        original_read = Path.read_text
        def patched_read(self, *args, **kw):
            if str(self) == '/content/colab-ai-config.json':
                return json.dumps({'model':'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', 'voice_variant':'custom'})
            return original_read(self, *args, **kw)
        spec = importlib.util.spec_from_file_location('colab_tts_remote_test',ROOT/'remote/voice_api.py')
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'qwen_tts':fq}), \
             mock.patch.object(Path,'read_text', patched_read), \
             mock.patch('http.server.ThreadingHTTPServer'):
            spec.loader.exec_module(module)
        cls.mod = module

    def test_custom_produces_real_valid_pcm_wav(self):
        self.mod.VARIANT='custom'
        audio=self.mod.synthesize({'text':'Olá, você está bem?', 'language':'Portuguese','speaker':'Ryan'})
        self.assertTrue(audio.startswith(b'RIFF'))
        info=sf.info(io.BytesIO(audio))
        self.assertEqual(info.samplerate,16000)
        self.assertEqual(info.channels,1)
        self.assertEqual(self.fake.last['speaker'],'Ryan')

    def test_voice_design_requires_description(self):
        self.mod.VARIANT='design'
        with self.assertRaisesRegex(self.mod.UserError,'descrição'):
            self.mod.synthesize({'text':'Olá'})
        audio=self.mod.synthesize({'text':'Olá', 'instruct':'Voz brasileira grave'})
        self.assertTrue(audio.startswith(b'RIFF'))

    def test_clone_requires_consent_and_valid_wav(self):
        self.mod.VARIANT='clone'
        wav = io.BytesIO()
        sf.write(wav, np.zeros(16000, dtype=np.float32), 16000,format='WAV')
        payload={'text':'Olá', 'ref_text':'Um exemplo',
                 'ref_audio_b64':base64.b64encode(wav.getvalue()).decode('ascii')}
        with self.assertRaisesRegex(self.mod.UserError,'autorização'):
            self.mod.synthesize(payload)
        payload['consent']=True
        output=self.mod.synthesize(payload)
        self.assertTrue(output.startswith(b'RIFF'))
        self.assertEqual(self.fake.last['ref_text'],'Um exemplo')

    def test_clone_rejects_invalid_audio(self):
        self.mod.VARIANT='clone'
        with self.assertRaisesRegex(self.mod.UserError,'Referência inválida'):
            self.mod.synthesize({'text':'oi','consent':True,'ref_text':'Oi',
                                 'ref_audio_b64':base64.b64encode(b'not an audio file').decode('ascii')})


if __name__=='__main__': unittest.main()
