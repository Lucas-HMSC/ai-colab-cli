"""Testes sem GPU para roteamento de modelos e geração simulada WAV.
Nunca invocam Colab, não fazem downloads e não criam instâncias.
"""
import base64
from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
CLI_SPEC = importlib.util.spec_from_file_location('cli_v020', ROOT / 'ai_colab.py')
app = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(app)


def fake_voice_api(variant):
    model = app.VOICE_MODELS[variant]
    fake = types.SimpleNamespace()
    fake.tts_model = types.SimpleNamespace(sample_rate=48000)
    fake.generate = mock.Mock(return_value=np.zeros(4800, dtype=np.float32))
    fake.t3 = types.SimpleNamespace(load_state_dict=mock.Mock(), eval=mock.Mock())
    fake.sr = 24000
    if variant == 'chatterbox-br':
        fake.generate.return_value = torch.zeros(1, 2400)
    pkg = {}
    if variant == 'voxcpm2':
        pkg['voxcpm'] = types.SimpleNamespace(VoxCPM=types.SimpleNamespace(from_pretrained=lambda *a, **k:fake))
    if variant == 'kokoro':
        pkg['kokoro'] = types.SimpleNamespace(KPipeline=lambda **kw: lambda *a, **k: [(None,None,np.zeros(2400, dtype=np.float32))])
    if variant == 'chatterbox-br':
        cb = types.SimpleNamespace(ChatterboxMultilingualTTS=types.SimpleNamespace(from_pretrained=lambda **kw:fake))
        pkg['chatterbox'] = types.ModuleType('chatterbox')
        pkg['chatterbox.mtl_tts'] = cb
        pkg['safetensors.torch'] = types.SimpleNamespace(load_file=lambda *a: {})
        pkg['huggingface_hub'] = types.SimpleNamespace(hf_hub_download=lambda **kw:'/tmp/mock-ptbr.safetensors')
    if variant == 'higgs3':
        pkg['transformers'] = types.SimpleNamespace(pipeline=lambda *a, **kw: lambda text: {
            'audio': np.zeros(2400, dtype=np.float32), 'sampling_rate':24000})
    original_read = Path.read_text
    def read_override(path,*a,**kw):
        if str(path) == '/content/colab-ai-config.json':
            return json.dumps({'voice_variant':variant,'model':model})
        return original_read(path,*a,**kw)
    spec=importlib.util.spec_from_file_location(f'mock_voice_{variant.replace("-","_")}', ROOT/'remote/voice_api.py')
    mod=importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, pkg),mock.patch.object(Path, 'read_text', read_override), \
         mock.patch('http.server.ThreadingHTTPServer'):
        spec.loader.exec_module(mod)
    return mod, fake


class VoiceV020Tests(unittest.TestCase):
    def test_model_registry_gpu_and_capabilities(self):
        for variant in ('voxcpm2','chatterbox-br','kokoro','higgs3'):
            with self.subTest(variant=variant):
                self.assertIn(variant, app.VOICE_MODELS)
                self.assertIn(variant, app.VOICE_DEFAULT_GPU)
                args=app.parser_make().parse_args(['start','voice','--voice-model',variant])
                self.assertEqual(args.voice_model,variant)
        self.assertEqual(app.VOICE_DEFAULT_GPU['kokoro'],'T4')
        self.assertEqual(app.VOICE_DEFAULT_GPU['higgs3'],'G4')
        self.assertNotIn('clone',app.VOICE_CAPABILITIES['higgs3'])
        self.assertIn('clone',app.VOICE_CAPABILITIES['voxcpm2'])

    def test_voxcpm_design_and_clone(self):
        server, engine=fake_voice_api('voxcpm2')
        audio=server.synthesize({'text':'Olá', 'instruct':'Voz brasileira tranquila', 'inference_timesteps':10})
        self.assertEqual(sf.info(io.BytesIO(audio)).samplerate,48000)
        self.assertTrue(engine.generate.call_args.kwargs['text'].startswith('(Voz brasileira tranquila)'))
        ref=io.BytesIO();sf.write(ref,np.zeros(16000,dtype=np.float32),16000,format='WAV')
        payload={'text':'Oi','consent':True,'ref_audio_b64':base64.b64encode(ref.getvalue()).decode()}
        audio=server.synthesize(payload)
        self.assertTrue(audio.startswith(b'RIFF'))
        self.assertIn('reference_wav_path',engine.generate.call_args.kwargs)
        self.assertFalse(Path(engine.generate.call_args.kwargs['reference_wav_path']).exists())

    def test_kokoro_brazilian_voices(self):
        server,_=fake_voice_api('kokoro')
        result=server.synthesize({'text':'Tudo bem?', 'speaker':'pf_dora','speed':1.0})
        self.assertEqual(sf.info(io.BytesIO(result)).samplerate,24000)
        with self.assertRaises(server.UserError):
            server.synthesize({'text':'oi','speaker':'Ryan'})

    def test_chatterbox_loads_brazilian_t3(self):
        server,engine=fake_voice_api('chatterbox-br')
        engine.t3.load_state_dict.assert_called_once_with({},strict=True)
        audio=server.synthesize({'text':'Bom dia pessoal','language':'Portuguese'})
        self.assertTrue(audio.startswith(b'RIFF'))
        self.assertEqual(engine.generate.call_args.kwargs['language_id'],'pt')
        with self.assertRaisesRegex(server.UserError,'450'):
            server.synthesize({'text':'a'*451})

    def test_higgs_simple_tts(self):
        server,_=fake_voice_api('higgs3')
        fake_wav = io.BytesIO()
        sf.write(fake_wav, np.zeros(2400,dtype=np.float32),24000,format='WAV')
        response = mock.MagicMock()
        response.read.return_value = fake_wav.getvalue()
        response.__enter__.return_value = response
        with mock.patch.object(server.urllib.request, 'urlopen', return_value=response) as request:
            wav=server.synthesize({'text':'Olá!', 'instruct':'calmo'})
        self.assertEqual(sf.info(io.BytesIO(wav)).samplerate,24000)
        self.assertIn('/v1/audio/speech',request.call_args.args[0].full_url)
        with self.assertRaisesRegex(server.UserError, 'não disponível'):
            server.synthesize({'text':'Olá', 'consent':True,'ref_audio_b64':base64.b64encode(b'abc').decode()})

    def test_incompatible_action_fails_without_remote_request(self):
        args = app.parser_make().parse_args(['voice','design','--text','Oi',
                                              '--description','Voz calma','--output','/tmp/out.wav'])
        with mock.patch.object(app,'state_read',return_value={'mode':'voice','voice_variant':'kokoro'}), \
             mock.patch.object(app.urllib.request,'urlopen') as remote:
            with self.assertRaisesRegex(RuntimeError,'--voice-model design'):
                app.do_voice(args)
            remote.assert_not_called()

    def test_kokoro_cli_uses_ptbr_speaker_default(self):
        args = app.parser_make().parse_args(['voice','generate','--text','Oi',
                                              '--output','/tmp/out.wav'])
        with tempfile.TemporaryDirectory() as folder:
            args.output = str(Path(folder)/'out.wav')
            state={'mode':'voice','voice_variant':'kokoro','port':18001,'token':'testing'}
            reply = io.BytesIO(b'RIFF'+b'\0'*100)
            reply.headers={'Content-Type':'audio/wav'}
            with mock.patch.object(app,'state_read',return_value=state), \
                 mock.patch.object(app.urllib.request,'urlopen',return_value=reply) as request:
                app.do_voice(args)
            sent=json.loads(request.call_args.args[0].data)
            self.assertEqual(sent['speaker'],'pf_dora')
            self.assertTrue(Path(args.output).exists())

    def test_code_mode_parser_still_preserves_262k(self):
        args=app.parser_make().parse_args(['start','code','--context','262144'])
        self.assertEqual((args.mode,args.context),('code',262144))



if __name__=='__main__':unittest.main()
