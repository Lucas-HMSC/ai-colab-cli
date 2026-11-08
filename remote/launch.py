"""Executado em colab exec: apenas dispara preparação; retorna imediatamente.

Download de pesos e pip install acontecem em processo separado, sem bloquear
`execute_reply` do kernel Jupyter (timeout do Colab CLI).
"""
from pathlib import Path
import os
import json
import subprocess
import sys
from urllib.request import urlopen

CONF = json.loads(Path("/content/colab-ai-config.json").read_text())
VOICE = CONF.get("mode") == "voice"
BOOTSTRAP = Path("/content/colab-ai-voice-bootstrap.py" if VOICE else "/content/colab-ai-bootstrap.py")
BOOT_LOG = Path("/content/colab-ai-bootstrap.log")
BOOT_PID = Path("/content/colab-ai-bootstrap.pid")
VLLM_PID = Path("/content/colab-ai-voice.pid" if VOICE else "/content/colab-ai-vllm.pid")


def alive(path, marker):
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split(") ", 1)[1].startswith("Z"):
            return False
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
        return marker.encode("utf-8") in cmd
    except (OSError, ValueError, ProcessLookupError):
        return False

try:
    with urlopen("http://127.0.0.1:8000/health", timeout=1) as r:
        ready = r.status == 200
except OSError:
    ready = False

if ready or alive(VLLM_PID, "colab-ai-voice-api.py" if VOICE else "vllm"):
    print("Servidor do modelo já está ativo/inicializando; não foi criado outro processo.", flush=True)
elif alive(BOOT_PID, str(BOOTSTRAP)):
    print("Preparação remota já em andamento; evitando instalação duplicada.", flush=True)
else:
    with BOOT_LOG.open("ab") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(BOOTSTRAP)],
                                stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                close_fds=True)
    BOOT_PID.write_text(str(proc.pid))
    print(f"Preparação remota iniciada: PID={proc.pid}. Log={BOOT_LOG}", flush=True)
