"""Regressões locais dos perfis corrigidos: não contatam GPUs nem fazem downloads."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ai_colab_cli_v023', ROOT / 'ai_colab.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

class VoiceV023Tests(unittest.TestCase):
    def test_chatterbox_loader_specific_for_region_and_decoder(self):
        api=(ROOT/'remote/voice_api.py').read_text()
        boot=(ROOT/'remote/voice_bootstrap.py').read_text()
        self.assertIn("T3(T3Config.multilingual())",api)
        self.assertIn("t3_pt_br.safetensors",api)
        self.assertIn("s3gen_v3.pt",api)
        self.assertIn("strict=False",api)
        self.assertIn("Conditionals.load",api)
        self.assertNotIn("from_pretrained(device='cuda', t3_model='v3')", api)
        self.assertIn('github.com/resemble-ai/chatterbox.git@5de7a54', boot)

    def test_higgs_profile_uses_official_recipe_version(self):
        boot=(ROOT/'remote/voice_bootstrap.py').read_text()
        self.assertIn("'vllm==0.22.0'",boot)
        self.assertIn('vllm-omni.git@36e048fd',boot)
        self.assertIn("'--omni'",boot)
        self.assertIn("'/bin/vllm'",boot)
        self.assertEqual(app.VOICE_DEFAULT_GPU['higgs3'],'H100')
        self.assertEqual(app.VERSION,'0.3.2')

    def test_qwen_voice_design_does_not_change_official_api(self):
        api=(ROOT/'remote/voice_api.py').read_text()
        self.assertIn('MODEL_OBJ.generate_voice_design(**args)',api)
        self.assertIn("args['instruct']",api)
        self.assertIn("'cuda_memory'",api)
        self.assertIn("'api_compatibility'",api)

    def test_programming_context_is_preserved(self):
        args=app.parser_make().parse_args(['start','code','--gpu','G4','--context','262144'])
        self.assertEqual(args.context,262144)
        self.assertEqual(args.gpu,'G4')

if __name__ == '__main__': unittest.main()
