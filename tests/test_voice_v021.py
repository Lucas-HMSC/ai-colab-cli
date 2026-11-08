"""Regressões da v0.2.1: sem GPU, internet, downloads ou alteração de conta."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ai_colab_v021',ROOT/'ai_colab.py')
app=importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
from test_voice_v020 import fake_voice_api


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_registry_preserves_code_and_qwen(self):
        self.assertEqual(app.VERSION, '0.2.4')
        self.assertEqual(app.PROFILES['code']['model'],'Qwen/Qwen3.8-27B-FP8')
        self.assertIn('262144',str(app.parser_make().parse_args(['start','code','--context','262144'])))
        for q in ('custom','design','clone'):
            self.assertIn(q,app.VOICE_MODELS)

    def test_runtime_isolation_and_pytorch_pair(self):
        source=(ROOT/'remote/voice_bootstrap.py').read_text()
        self.assertIn("else '3.12' if variant == 'higgs3' else '3.11'",source)
        self.assertIn("git+https://github.com/resemble-ai/chatterbox.git@5de7a54",source)
        self.assertIn("'kokoro==0.9.4', 'misaki[en]==0.9.4'",source)
        self.assertIn('vllm-omni.git@36e048fd',source)
        self.assertIn("'--torch-backend=cu128'",source)
        self.assertIn('venv_python = Path(sys.executable) if qwen_engine',source)
        self.assertIn("'optimize=False'".strip("'"), (ROOT/'remote/voice_api.py').read_text())

    def test_higgs_preflight_blocks_undersized_a100_before_gpu(self):
        args=app.parser_make().parse_args(['start','voice','--voice-model','higgs3','--gpu','A100'])
        with mock.patch.object(app,'ensure_key') as ssh, mock.patch.object(app,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'A100 padrão'):
                app.do_start(args)
            ssh.assert_not_called()
            run.assert_not_called()

    def test_higgs_default_gpu_and_privacy(self):
        self.assertEqual(app.VOICE_DEFAULT_GPU['higgs3'],'H100')
        self.assertGreaterEqual(app.VOICE_MIN_GIB['higgs3'],70)
        self.assertNotIn('clone',app.VOICE_CAPABILITIES['higgs3'])
        self.assertEqual(app.parser_make().parse_args(['logs','--source','omni']).source,'omni')

    def test_higgs_worker_routes_to_omni_not_transformers(self):
        server,_=fake_voice_api('higgs3')
        b=io.BytesIO();sf.write(b,np.zeros(1600,dtype=np.float32),24000,format='WAV')
        resp=mock.MagicMock();resp.read.return_value=b.getvalue();resp.__enter__.return_value=resp
        with mock.patch.object(server.urllib.request,'urlopen',return_value=resp) as urlopen:
            out=server.synthesize({'text':'Testando voz', 'instruct':'calmo'})
        self.assertEqual(sf.info(io.BytesIO(out)).samplerate,24000)
        req=urlopen.call_args.args[0]
        self.assertEqual(req.full_url,'http://127.0.0.1:8002/v1/audio/speech')
        posted=json.loads(req.data)
        self.assertIn('<|emotion:contentment|>',posted['input'])
        self.assertNotIn('pipeline(', (ROOT/'remote/voice_api.py').read_text())

    def test_vox_initialization_disables_torch_compile(self):
        server, _ = fake_voice_api('voxcpm2')
        self.assertEqual(sf.info(io.BytesIO(server.synthesize({'text':'Oi!'}))).samplerate,48000)
        self.assertIn('optimize=False',(ROOT/'remote/voice_api.py').read_text())

    def test_logs_never_share_user_text_on_inference_exception(self):
        server,engine=fake_voice_api('voxcpm2')
        engine.generate.side_effect=TypeError('private_example_voice_text expected argument')
        with mock.patch('builtins.print') as output:
            try:
                server.synthesize({'text':'private_example_voice_text'})
            except TypeError:
                pass
        # The HTTP handler catches and logs the error type/line, never the text.
        source=(ROOT/'remote/voice_api.py').read_text()
        self.assertIn("'location': loc", source)
        self.assertNotIn('str(err)[:400]',source)

    def test_omni_fail_status_exposed(self):
        source=(ROOT/'remote/proxy.py').read_text()
        self.assertIn('omni_alive is False',source)
        self.assertIn("alive = False",source)
        self.assertIn('Higgs Omni',source)


if __name__=='__main__': unittest.main()
