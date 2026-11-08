"""Regression: PyPI voxcpm 2.0.3 does not accept seed= in _generate()."""
import io
import unittest

import numpy as np
import soundfile as sf

from test_voice_v020 import fake_voice_api


class VoxCPMReleaseCompatibilityTests(unittest.TestCase):
    def test_voxcpm2_generate_uses_only_pypi_203_supported_arguments(self):
        server, engine = fake_voice_api('voxcpm2')
        called = []

        # Enforce documented 2.0.3 signature instead of Mock(**kwargs).
        def pypi_203_generate(*, text, cfg_value=2.0, inference_timesteps=10,
                              prompt_wav_path=None, prompt_text=None,
                              reference_wav_path=None):
            called.append((text, cfg_value, inference_timesteps))
            return np.zeros(4800, dtype=np.float32)

        engine.generate = pypi_203_generate
        wav = server.synthesize({
            'text': 'Olá! Testando compatibilidade com o VoxCPM2.',
            'instruct': 'Voz feminina brasileira, suave e acolhedora',
            'inference_timesteps': 10,
        })
        self.assertTrue(wav.startswith(b'RIFF'))
        self.assertEqual(sf.info(io.BytesIO(wav)).samplerate, 48000)
        self.assertEqual(len(called), 1)
        self.assertTrue(called[0][0].startswith('(Voz feminina brasileira'))

    def test_released_package_is_pinned(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'remote' / 'voice_bootstrap.py').read_text()
        self.assertIn("'voxcpm==2.0.3'", source)


if __name__ == '__main__':
    unittest.main()
