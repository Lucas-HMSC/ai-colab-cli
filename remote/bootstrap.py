"""Executado via 'colab exec' no kernel da VM. Inicia vLLM desacoplado."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import time

config = json.loads(Path("/content/colab-ai-config.json").read_text())
PID_FILE = Path("/content/colab-ai-vllm.pid")
LOG_FILE = Path("/content/colab-ai-vllm.log")


def old_vllm_alive():
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").lower()
        if b"vllm" in cmdline:
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().split(") ", 1)[1][:1] == "Z":
                return False
            return True
        return False
    except (OSError, ValueError, ProcessLookupError):
        return False

model = config["model"]
model_path = model
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1' if config.get('hf_mode') == 'fast' else '0'
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
if config.get("hf_token"):
    os.environ["HF_TOKEN"] = config["hf_token"]
info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()[0]
gpu, mem = info.split(",", 1)
match = re.search(r"([\d.]+)\s*MiB", mem)
if not match:
    raise RuntimeError(f"GPU inesperada: {info}")
gib = float(match.group(1)) / 1024
print(f"GPU: {gpu.strip()} | {gib:.1f} GiB", flush=True)
if gib < config["min_gib"]:
    raise RuntimeError(f"GPU insuficiente para perfil {model}: mínimo ~{config['min_gib']} GiB")
# Atualizar o vLLM apenas quando está ausente ou incompatível com requisitos do perfil.
try:
    import importlib.metadata as md
    from packaging.version import Version
    good = Version(md.version("vllm")) >= Version("0.28.0") and Version(md.version("transformers")) >= Version("5.8.0")
except (ImportError, md.PackageNotFoundError, ValueError):
    good = False
if not good:
    print("Instalando vLLM>=0.28 / transformers>=5.8 (primeira vez na VM)...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "vllm>=0.28.0", "transformers>=5.8.0", "hf_xet"])

try:
    import hf_xet
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'hf_xet'])
print('Download HF: hf_xet', config.get('hf_mode', 'fast'), flush=True)

# Evitar duas instâncias pesadas no mesmo runtime.
from urllib.request import urlopen
try:
    with urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
        existing = (r.status == 200)
except OSError:
    existing = False
if existing:
    with urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as resp:
        ids = [x.get("id") for x in json.load(resp).get("data", [])]
    if model not in ids:
        raise RuntimeError("Já existe outro modelo na VM. Encerre a sessão antes de trocar de perfil.")
    print(f"Reutilizando vLLM já ativo: {model}", flush=True)
elif old_vllm_alive():
    print("vLLM já está inicializando na VM; preservando o processo e os downloads.", flush=True)
else:
    PID_FILE.unlink(missing_ok=True)
    cmd = [shutil.which("vllm") or "/usr/local/bin/vllm", "serve", model_path,
           "--host", "127.0.0.1", "--port", "8000",
           "--served-model-name", model,
           "--generation-config", "vllm", "--tensor-parallel-size", "1",
           "--gpu-memory-utilization", str(config["gpu_util"]),
           "--max-model-len", str(config["context"]),
           "--max-num-seqs", "1", "--max-num-batched-tokens", "8192",
           "--kv-cache-dtype", "fp8", "--enable-prefix-caching",
           "--enable-auto-tool-choice", "--tool-call-parser", config["tool_parser"]]
    if config.get("reasoning_parser"):
        cmd += ["--reasoning-parser", config["reasoning_parser"]]
    logfile = LOG_FILE
    with logfile.open("ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True,
                                env=os.environ.copy())
    PID_FILE.write_text(str(proc.pid))
    print(f"vLLM iniciado, PID={proc.pid}. Log: {logfile} | modelo={model}", flush=True)
    # Verificar falha imediata (ex.: flag desconhecida) sem bloquear o kernel por horas.
    time.sleep(3)
    if proc.poll() is not None:
        print("\n".join(logfile.read_text(errors="replace").splitlines()[-35:]))
        raise RuntimeError(f"vLLM encerrou ao inicializar, código {proc.returncode}")
    print("Modelo carregando em segundo plano. A célula real de API ficará em execução contínua.", flush=True)
