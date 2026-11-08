"""Executado por 'colab exec -f': mantém serviço real (não keepalive falso)."""
from pathlib import Path
import json
import importlib.util

config = json.loads(Path("/content/colab-ai-config.json").read_text())
module = "/content/colab-ai-proxy.py"
spec = importlib.util.spec_from_file_location("colab_ai_proxy", module)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
proxy.serve(config["key"])
