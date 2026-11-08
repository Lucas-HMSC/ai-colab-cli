"""Proxy HTTP de streaming para API vLLM (Anthropic + OpenAI).

Escuta APENAS 127.0.0.1 no Colab; cliente usa túnel SSH.
"""
import http.client
import os
from pathlib import Path
from urllib.request import urlopen
import json
import secrets
import subprocess
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROUTES = {
    ("GET", "/v1/models"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/responses"),
    ("POST", "/_colab_ai/voice/generate"),
}


try:
    CONF = json.loads(Path("/content/colab-ai-config.json").read_text())
except (FileNotFoundError, ValueError):
    CONF = {}  # testabilidade local; em produção o arquivo é obrigatório em serve.py
VOICE_MODE = CONF.get("mode") == "voice"
PID_FILE = Path("/content/colab-ai-voice.pid" if VOICE_MODE else "/content/colab-ai-vllm.pid")
LOG_FILE = Path("/content/colab-ai-voice.log" if VOICE_MODE else "/content/colab-ai-vllm.log")
BOOT_FILE = Path("/content/colab-ai-bootstrap.log")
BOOT_PID_FILE = Path("/content/colab-ai-bootstrap.pid")



def process_alive(pid):
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        # Um processo zumbi não está servindo, mesmo se kill(pid, 0) funcionar.
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split(") ", 1)[1][:1] == "Z":
            return False
        return True
    except (ProcessLookupError, PermissionError, OSError, ValueError, IndexError):
        return False


def collect_metrics():
    """Read hardware, not inferred estimates. Bounded runtime/timeout."""
    output = {"timestamp": time.time()}
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu",
               "--format=csv,noheader,nounits"]
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3)
        if p.returncode == 0 and p.stdout.strip():
            parts = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
            if len(parts) >= 6:
                output["gpu"] = {"name": parts[0], "utilization_pct": float(parts[1]),
                                 "memory_total_mib": float(parts[2]), "memory_used_mib": float(parts[3]),
                                 "memory_free_mib": float(parts[4]), "temperature_c": float(parts[5])}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        m = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                m[key] = int(value.strip().split()[0]) * 1024
        if all(x in m for x in ("MemTotal", "MemAvailable")):
            output["ram"] = {"total_bytes": m["MemTotal"],
                             "used_bytes": m["MemTotal"] - m["MemAvailable"],
                             "available_bytes": m["MemAvailable"]}
        d = shutil.disk_usage("/content")
        output["disk"] = {"total_bytes": d.total, "used_bytes": d.used, "free_bytes": d.free}
    except (OSError, ValueError):
        pass
    return output


def vllm_status(include_metrics=False):
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        pid = None
    alive = process_alive(pid)
    omni_log_tail = ""
    omni_alive = None
    if VOICE_MODE and CONF.get('voice_variant') == 'higgs3':
        omni_file = Path('/content/colab-ai-voice-omni.pid')
        try:
            omni_pid = int(omni_file.read_text().strip())
            omni_alive = process_alive(omni_pid)
        except (OSError, ValueError):
            omni_alive = None
        if omni_alive is False:
            alive = False  # expor falha real do backend, não apenas worker HTTP ativo
        try:
            omni_log_tail = Path('/content/colab-ai-voice-omni.log').read_text(errors='replace')[-2000:]
        except OSError:
            pass
    ready = False
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=0.8) as resp:
            ready = resp.status == 200
    except OSError:
        pass
    log_tail = ""
    log_bytes = 0
    try:
        with LOG_FILE.open("rb") as f:
            log_bytes = f.seek(0, 2)
            f.seek(max(0, log_bytes - 8192))
            log_tail = f.read().decode("utf-8", "replace")[-3300:]
    except OSError:
        pass
    bootstrap_pid = None
    try:
        bootstrap_pid = int(BOOT_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        pass
    bootstrap_alive = process_alive(bootstrap_pid)
    bootstrap_log_tail = ""
    try:
        with BOOT_FILE.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - 8192))
            bootstrap_log_tail = f.read().decode("utf-8", "replace")[-3300:]
    except OSError:
        pass
    text = log_tail.lower()
    if ready:
        phase = "modelo pronto"
    elif alive is False:
        phase = "processo do modelo encerrado (ver logs)"
    elif alive is None and bootstrap_alive is True:
        phase = "preparando dependências / iniciando o servidor"
    elif alive is None and bootstrap_alive is False:
        phase = "preparação encerrou sem iniciar o servidor (ver bootstrap log)"
    elif any(k in text for k in ("cuda out of memory", "outofmemoryerror", "traceback", "error loading")):
        phase = "possível erro no carregamento (ver logs)"
    elif any(k in text for k in ("downloading", "fetching", "model.safetensors", "downloaded")):
        phase = "download dos pesos / cache"
    elif any(k in text for k in ("loading", "checkpoint", "model runner", "initializing", "enginecore")):
        phase = "inicializando / carregando pesos"
    else:
        phase = "inicialização em andamento" if alive else "processo ainda não detectado"
    if omni_log_tail and alive is False:
        log_tail = (log_tail + "\n--- Higgs Omni ---\n" + omni_log_tail)[-3500:]
    status = {"vllm_pid": pid, "vllm_alive": alive, "vllm_ready": ready,
              "omni_alive": omni_alive if VOICE_MODE and CONF.get('voice_variant') == 'higgs3' else None,
              "phase": phase, "log_bytes": log_bytes, "log_tail": log_tail,
              "bootstrap_pid": bootstrap_pid, "bootstrap_alive": bootstrap_alive,
              "bootstrap_log_tail": bootstrap_log_tail, "mode": "voice" if VOICE_MODE else "code"}
    if include_metrics:
        status["metrics"] = collect_metrics()
    return status


def make_handler(api_key):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            parsed = urlsplit(self.path)
            if (self.command, parsed.path) not in ROUTES and not (
                self.command == "GET" and parsed.path == "/_colab_ai/status"
            ):
                self.send_error(404, "Rota indisponível")
                return
            authorization = self.headers.get("Authorization", "")
            x_key = self.headers.get("X-Api-Key", "")
            if not (secrets.compare_digest(authorization, "Bearer " + api_key)
                    or secrets.compare_digest(x_key, api_key)):
                self.send_error(401, "Chave inválida")
                return
            if self.command == "GET" and parsed.path == "/_colab_ai/status":
                data = json.dumps(vllm_status(include_metrics=("metrics=1" in parsed.query))).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/_colab_ai/voice/generate" and not VOICE_MODE:
                self.send_error(404, "Voice não iniciado nesta sessão")
                return
            data = None
            if self.command == "POST":
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    size = -1
                if size < 0 or size > 20 * 1024 * 1024:
                    self.send_error(413, "Corpo ausente ou muito grande")
                    return
                data = self.rfile.read(size)
                # Intercepta SOMENTE o campo de esforço no topo; não mexe no prompt.
                # Claude Code pode enviar 'high', enquanto o Qwen3.8 admite 'xhigh'.
                if parsed.path == "/v1/messages" and data:
                    try:
                        payload = json.loads(data)
                        if payload.get("reasoning_effort") == "high":
                            payload["reasoning_effort"] = "xhigh"
                            data = json.dumps(payload).encode("utf-8")
                    except (ValueError, TypeError, AttributeError):
                        pass
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json"),
                       "Accept": self.headers.get("Accept", "application/json"),
                       "Accept-Encoding": "identity"}
            for key in ("Anthropic-Version", "Anthropic-Beta"):
                if self.headers.get(key):
                    headers[key] = self.headers[key]
            upstream = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3600)
            started = False
            try:
                url = parsed.path + (("?" + parsed.query) if parsed.query else "")
                upstream.request(self.command, url, body=data, headers=headers)
                response = upstream.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                started = True
                while True:
                    chunk = response.read1(16384)
                    if not chunk: break
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    self.wfile.write(chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                if not started:
                    self.send_error(502, f"vLLM indisponível: {type(error).__name__}")
                else:
                    self.close_connection = True
            finally:
                upstream.close()

        def log_message(self, fmt, *args):
            # Não gravar corpo das requisições, prompts ou chaves.
            pass
    return Proxy


def serve(key):
    server = ThreadingHTTPServer(("127.0.0.1", 8001), make_handler(key))
    server.daemon_threads = True
    print("Proxy real ativo: 127.0.0.1:8001", flush=True)
    server.serve_forever(poll_interval=0.25)
