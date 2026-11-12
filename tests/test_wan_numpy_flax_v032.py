"""Regressão v0.3.2: Wan não deve importar Flax/JAX e deve checar imports antes dos pesos."""
import ast
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WanFlaxRegressionTests(unittest.TestCase):
    def test_backends_disabled_before_optional_framework_imports(self):
        bootstrap = (ROOT / 'remote/video_bootstrap.py').read_text()
        api = (ROOT / 'remote/video_api.py').read_text()
        generator = (ROOT / 'remote/video_generate.py').read_text()
        for file, src in [('bootstrap', bootstrap), ('api', api), ('generator', generator)]:
            with self.subTest(file=file):
                self.assertIn("os.environ['USE_FLAX'] = '0'" if file != 'generator' else 'os.environ["USE_FLAX"] = "0"', src)
                self.assertIn("os.environ['USE_TF'] = '0'" if file != 'generator' else 'os.environ["USE_TF"] = "0"', src)
                ast.parse(src)
        self.assertLess(generator.index('os.environ["USE_FLAX"]'), generator.index('import torch'))
        self.assertLess(bootstrap.index("os.environ['USE_FLAX']"), bootstrap.index('    import torch'))

    def test_child_process_inherits_disabled_frameworks(self):
        api = (ROOT / 'remote/video_api.py').read_text()
        generator = (ROOT / 'remote/video_generate.py').read_text()
        self.assertRegex(api, r'env=\{\*\*os\.environ, "USE_FLAX": "0", "USE_TF": "0"\}')
        self.assertIn("env={**os.environ, 'USE_FLAX': '0', 'USE_TF': '0'}", generator)
        self.assertIn('env={**os.environ, "USE_FLAX": "0", "USE_TF": "0"}', generator)

    def test_preflight_before_snapshot_download(self):
        source = (ROOT / 'remote/video_generate.py').read_text()
        fn = source[source.index('def generate_wan('):source.index('\ndef main(',source.index('def generate_wan('))]
        self.assertLess(fn.index("subprocess.run("), fn.index('snapshot_download(WAN_REPO'))
        self.assertIn('import torch, transformers, diffusers, wan;', fn)
        self.assertIn('if check.returncode != 0:', fn)
        self.assertIn('raise RuntimeError(', fn)
        self.assertIn('ANTES do download dos pesos', fn)

    def test_environment_overrides_do_not_modify_other_modules(self):
        for file in ('bootstrap.py', 'voice_bootstrap.py', 'transcribe_bootstrap.py'):
            src = (ROOT / 'remote' / file).read_text()
            if file != 'bootstrap.py':
                self.assertNotIn('USE_FLAX', src)
        self.assertIn('VERSION = "0.3.2"', (ROOT/'ai_colab.py').read_text())

if __name__ == '__main__':
    unittest.main()
