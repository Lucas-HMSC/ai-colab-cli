"""Regressões sem GPUs para tratamento de erros na inicialização."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]

class DiagnosticsV024(unittest.TestCase):
    def test_diagnostics_declare_higgs_backend_health_check(self):
        source=(ROOT/'remote/voice_bootstrap.py').read_text()
        ast.parse(source)
        self.assertIn("urlopen('http://127.0.0.1:8002/health'", source)
        self.assertIn('wait_omni_backend(aux_pidfile, aux_log)', source)
        self.assertIn('omni_failure_summary', source)

    def test_chatterbox_reports_phases(self):
        source=(ROOT/'remote/voice_api.py').read_text()
        ast.parse(source)
        for i in range(1, 6):
            self.assertIn(f'Chatterbox PT-BR [{i}/5]', source)

    def test_no_automatic_gpu_restart_on_network_failure(self):
        source=(ROOT/'ai_colab.py').read_text()
        self.assertIn('Se persistir, examine sem reiniciar a GPU:', source)
        self.assertIn('ai-colab logs --source bootstrap --lines 80', source)

if __name__ == '__main__': unittest.main()
