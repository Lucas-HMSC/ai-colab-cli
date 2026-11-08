#!/usr/bin/env python3
"""Colab AI CLI: runtime multimodal sob demanda para Linux.

Usa somente stdlib no host; controle via google-colab-cli. Modos adicionais
ficam definidos, mas sem iniciar uma GPU até terem um backend implementado.
"""
from __future__ import annotations
import argparse
import fcntl
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import tempfile
from collections import deque
import urllib.error
import urllib.request

VERSION = "0.2.1"
ROOT = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "colab-ai"
STATE_FILE = STATE_DIR / "state.json"
PREFS_FILE = STATE_DIR / "preferences.json"
LOCK_FILE = STATE_DIR / "tunnel.lock"
VOICE_BOOTSTRAP_REMOTE = "/content/colab-ai-voice-bootstrap.py"
VOICE_API_REMOTE = "/content/colab-ai-voice-api.py"

SESSION = "colab-ai-code"
REMOTE_CONFIG = "/content/colab-ai-config.json"
REMOTE_SERVER = "/content/colab-ai-proxy.py"
REMOTE_BOOTSTRAP = "/content/colab-ai-bootstrap.py"
LOCAL_PORT = 18001
REMOTE_PROXY_PORT = 8001

PROFILES = {
    "code": {
        "model": "Qwen/Qwen3.8-27B-FP8", "context": 131072,
        "min_gib": 38, "gpu_util": 0.88, "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3", "effort": "medium",
        "note": "Padrão recomendado: qualidade agentiva; já testado no Colab G4 via notebook.",
    },
    "code-next": {
        "model": "Qwen/Qwen3-Coder-Next-FP8", "context": 32768,
        "min_gib": 85, "gpu_util": 0.95, "tool_parser": "qwen3_coder",
        "reasoning_parser": None, "effort": "medium",
        "note": "Experimental: 80B totais, possível OOM em 96 GB; sem modo thinking.",
    },
}
COMING = {
    "image": "Geração e edição de imagens (planejada)",
    "video": "Geração de vídeos (planejada)",
    "transcribe": "Transcrição de áudio / ASR (planejada)",
}

VOICE_MODELS = {
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voxcpm2": "openbmb/VoxCPM2",
    "chatterbox-br": "ResembleAI/Chatterbox-Multilingual-pt-br",
    "kokoro": "hexgrad/Kokoro-82M",
    "higgs3": "bosonai/higgs-tts-3-4b",
}
VOICE_DEFAULT_GPU = {"custom": "L4", "design": "L4", "clone": "L4",
                     "voxcpm2": "L4", "chatterbox-br": "L4", "kokoro": "T4", "higgs3": "G4"}
VOICE_MIN_GIB = {"custom": 10, "design": 10, "clone": 10,
                 "voxcpm2": 10, "chatterbox-br": 10, "kokoro": 5, "higgs3": 70}
VOICE_CAPABILITIES = {
    "custom": {"generate"}, "design": {"design"}, "clone": {"clone"},
    "voxcpm2": {"generate", "design", "clone"},
    "chatterbox-br": {"generate", "clone"},
    "kokoro": {"generate"}, "higgs3": {"generate"},
}
VOICE_NOTES = {"voxcpm2": "Design e clonagem; Apache-2.0",
               "chatterbox-br": "PT-BR com pesos regionais sobre V3; MIT",
               "kokoro": "82M, vozes brasileiras; Apache-2.0",
               "higgs3": "EXPERIMENTAL (vLLM-Omni), GPU G4/H100; licença Research/Creator Grant"}
SPEAKERS = ("Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")



def state_read() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        raise RuntimeError(f"Estado corrompido: {STATE_FILE}")


def state_write(data: dict) -> None:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    STATE_DIR.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix=".colab-ai-state-", suffix=".tmp", dir=STATE_DIR)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
        os.replace(temporary, STATE_FILE)
        STATE_FILE.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def preferences_read() -> dict:
    """Compatível com preferências 0.1.8 sem reativar cache do Drive."""
    try:
        value = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"hf_mode": "fast"}
    except (ValueError, OSError) as error:
        raise RuntimeError(f"Preferências inválidas: {error}")
    return {"hf_mode": value.get("hf_mode", "fast") if value.get("hf_mode") in ("fast", "normal") else "fast"}


def preferences_write(value: dict):
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = PREFS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(PREFS_FILE)


def do_hf(args):
    value = preferences_read()
    if args.action in ("fast", "normal"):
        value["hf_mode"] = args.action
        preferences_write(value)
        print("Próximas sessões: hf_xet", args.action, "— configuração aplicada após novo start")
    else:
        print("Download Hugging Face:", value["hf_mode"], "(hf_xet; cache local efêmero)")


@contextmanager
def tunnel_lock():
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    with LOCK_FILE.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run(cmd: list[str], *, check=True, timeout=None, quiet=False) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if not quiet or result.returncode != 0:
        if result.stdout.strip():
            print(result.stdout.strip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Falha ({result.returncode}): {shlex.join(cmd[:5])}\n{result.stdout[-2500:]}")
    return result


def run_interactive(cmd: list[str], *, timeout: int = 720):
    """Stream stdout/stderr for OAuth Drive mounting; retain live stdin prompts."""
    print("Autorização do Google Drive: siga as instruções exibidas abaixo, se solicitadas.", flush=True)
    result = subprocess.run(cmd, stdin=None, stdout=None, stderr=None, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Falha ({result.returncode}) ao montar Drive. "
                           "A VM já pode estar ativa: use 'ai-colab status' e 'ai-colab stop' se necessário.")
    return result


def cli_base(auth: str) -> list[str]:
    binary = shutil.which("colab")
    if not binary:
        raise RuntimeError("Colab CLI não encontrada. Instale: uv tool install google-colab-cli")
    return [binary, f"--auth={auth}"]


def remote_session_exists(base: list[str], name: str) -> bool:
    """Usa o TEXTO de `colab status`, não só o exit code.

    google-colab-cli `status -s` retorna zero inclusive quando a sessão
    solicitada não existe. Em casos desconhecidos falha de modo seguro,
    evitando atribuir uma segunda GPU por engano.
    """
    result = run(base + ["status", "-s", name], check=False, quiet=True)
    output = result.stdout.strip()
    if result.returncode != 0:
        raise RuntimeError(f"Não foi possível consultar a sessão Colab: {output[-1800:]}")
    if re.search(rf"(?m)^\[{re.escape(name)}\]\s+", output):
        return True
    if f"Session '{name}' not found." in output:
        return False
    if "No active sessions" in output:
        return False
    raise RuntimeError(
        "Resposta inesperada do comando 'colab status'; não criarei outra GPU "
        f"sem confirmar o estado remoto. Saída: {output[-1200:]}"
    )


def ensure_key() -> Path:
    """Prepara uma identidade exclusiva para o túnel, antes de alocar GPU.

    Nunca utiliza, altera, sobrescreve nem solicita a senha das chaves SSH
    pessoais do usuário. O Colab CLI precisa extrair a chave pública com
    `ssh-keygen -y`; uma chave pessoal protegida por senha pode falhar em
    execução desacoplada, mesmo que ~/.ssh/id_ed25519.pub exista.
    """
    keygen = shutil.which("ssh-keygen")
    if not keygen:
        raise RuntimeError("OpenSSH ssh-keygen não encontrado. Instale openssh-client.")

    key_dir = STATE_DIR / "ssh"
    key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    key_dir.chmod(0o700)
    key = key_dir / "id_ed25519"
    pub = key_dir / "id_ed25519.pub"

    if not key.exists():
        if pub.exists():
            raise RuntimeError(
                f"Chave privada da ferramenta não existe, mas existe {pub}. "
                "Verifique esse arquivo manualmente. Nada será sobrescrito.")
        result = subprocess.run(
            [keygen, "-q", "-t", "ed25519", "-N", "",
             "-C", "colab-ai-cli", "-f", str(key)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Não foi possível criar chave SSH exclusiva para o Colab AI: "
                + result.stderr.strip()[-500:])
        print("Chave SSH exclusiva do Colab AI CLI criada. Suas chaves pessoais não foram modificadas.")

    key.chmod(0o600)
    check = subprocess.run(
        [keygen, "-y", "-P", "", "-f", str(key)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=15,
    )
    if check.returncode != 0 or not check.stdout.startswith("ssh-ed25519 "):
        raise RuntimeError(
            f"Chave SSH exclusiva inválida: {key}. "
            "A GPU NÃO será iniciada. Verifique permissões/conteúdo, "
            "sem sobrescrever sua chave pessoal. Detalhe: "
            + check.stderr.strip()[-350:])
    derived_pub = check.stdout.strip() + "\n"
    if pub.exists() and pub.read_text(encoding="utf-8").split()[:2] != derived_pub.split()[:2]:
        raise RuntimeError(f"Chave pública não corresponde à chave privada em {pub}. "
                           "Não modifiquei o arquivo. Revise-o manualmente.")
    if not pub.exists():
        pub.write_text(derived_pub, encoding="utf-8")
    pub.chmod(0o644)
    return key


def do_ssh_check(args):
    """Valida a chave local sem criar sessão e sem consumir unidades."""
    key = ensure_key()
    print(f"SSH pronto: identidade exclusiva validada em {key}")
    print("Nenhuma instância Colab foi criada.")


def pid_running(pid: int | None) -> bool:
    if not pid or not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.exists() and proc_stat.read_text().split(") ", 1)[1][:1] == "Z":
            return False
    except (ProcessLookupError, PermissionError, OSError, IndexError):
        return False
    return True


def kill_group(pid: int | None):
    if not pid_running(pid):
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def local_api(state: dict, timeout: int = 4) -> tuple[bool, str]:
    if not state:
        return False, "não iniciado"
    url = f"http://127.0.0.1:{state.get('port',LOCAL_PORT)}/v1/models"
    req = urllib.request.Request(url, headers={"X-Api-Key": state.get("token", "")})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = json.load(r)
            names = [p.get("id") for p in b.get("data", [])]
            return (state.get("model") in names), ", ".join(names) or "sem modelos"
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def launch_process(cmd: list[str], logname: str) -> int:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = STATE_DIR / logname
    with path.open("ab") as log:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             close_fds=True, start_new_session=True)
    return p.pid


def runtime_status(state: dict, timeout: int = 5, *, metrics: bool = False) -> tuple[bool, dict | str]:
    """Consulta status autenticado do servidor, mesmo antes do vLLM estar pronto.

    O endpoint do proxy e o túnel continuam disponíveis durante downloads e
    carregamento. Assim é possível diferenciar demora, falha e API pronta.
    """
    url = f"http://127.0.0.1:{state.get('port', LOCAL_PORT)}/_colab_ai/status" + ("?metrics=1" if metrics else "")
    req = urllib.request.Request(url, headers={"X-Api-Key": state.get("token", "")})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.load(r)
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def tunnel_start(state: dict, *, force: bool = False) -> dict:
    """Start/reuse SSH tunnel without reallocating GPU or restarting vLLM.

    Do not kill a working forwarding socket merely because an HTTP request is
    slow (inference can saturate the remote VM). force=True is reserved for a
    confirmed local socket failure (three checks) or manual reconnect.
    """
    with tunnel_lock():
        latest = state_read()
        if not latest or latest.get("token") != state.get("token"):
            raise RuntimeError("Estado mudou ou sessão foi encerrada; túnel não será recriado")
        state = latest
        pid = state.get("tunnel_pid")
        if pid_running(pid):
            if not force:
                return state
            # Never interrupt in-flight Claude requests when local SSH still
            # listens. A slow upstream API is NOT a tunnel failure.
            with socket.socket() as sk:
                sk.settimeout(1)
                if sk.connect_ex(("127.0.0.1", int(state["port"]))) == 0:
                    return state
            kill_group(pid)
            time.sleep(0.5)
        key = ensure_key()
        colab_bin = cli_base(state["auth"])[0]
        proxy_cmd = shlex.join([colab_bin, f"--auth={state['auth']}", "ssh", "--proxy-mode",
                                "-s", state["session"], "-i", str(key)])
        ssh = shutil.which("ssh")
        if not ssh:
            raise RuntimeError("OpenSSH client ausente. Instale openssh-client.")
        cmd = [ssh, "-F", "/dev/null", "-i", str(key), "-o", "BatchMode=yes",
               "-o", "IdentitiesOnly=yes", "-o", "ExitOnForwardFailure=yes",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
               "-o", "StrictHostKeyChecking=accept-new",
               "-o", f"UserKnownHostsFile={STATE_DIR / 'known_hosts'}",
               "-o", f"ProxyCommand={proxy_cmd}", "-N",
               "-L", f"127.0.0.1:{state['port']}:127.0.0.1:{REMOTE_PROXY_PORT}",
               "root@colab"]
        state["tunnel_pid"] = launch_process(cmd, "ssh-tunnel.log")
        state_write(state)
        return state


def watchdog_start(state: dict) -> dict:
    with tunnel_lock():
        latest = state_read()
        if not latest or latest.get("token") != state.get("token"):
            raise RuntimeError("Sessão alterada ou encerrada antes de iniciar o watchdog")
        if pid_running(latest.get("watchdog_pid")):
            return latest
        latest["watchdog_pid"] = launch_process(
            [sys.executable, str(ROOT / "ai_colab.py"), "_watchdog"], "watchdog.log")
        state_write(latest)
        return latest


def do_watchdog(args):
    """Local monitor; NEVER calls colab new or starts the model."""
    myself = os.getpid()
    errors = 0
    consecutive_restarts = 0
    delay = 10
    time.sleep(2)  # parent commits state before worker reads it
    while True:
        state = state_read()
        if not state or state.get("watchdog_pid") != myself:
            return
        pid = state.get("tunnel_pid")
        listening = False
        with socket.socket() as sk:
            sk.settimeout(2)
            try:
                listening = sk.connect_ex(("127.0.0.1", int(state["port"]))) == 0
            except OSError:
                pass
        if pid_running(pid) and listening:
            errors = 0
            consecutive_restarts = 0
            delay = 10
        else:
            errors += 1
            if errors >= 3 or not pid_running(pid):
                try:
                    tunnel_start(state, force=True)
                    # Wait for local SSH listener before declaring recovery.
                    errors = 0
                    consecutive_restarts += 1
                    delay = min(120, 10 * (2 ** min(4, consecutive_restarts)))
                    with (STATE_DIR / "watchdog.log").open("a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - nova tentativa de túnel, sem reiniciar GPU\n")
                except (RuntimeError, OSError) as err:
                    with (STATE_DIR / "watchdog.log").open("a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - SSH indisponível: {type(err).__name__}: {err}\n")
                    delay = min(120, max(15, delay * 2))
        time.sleep(delay)


def _progress_line(snapshot: dict) -> str:
    """Exibe progresso útil, sem corpo de requisição ou segredos."""
    log = snapshot.get("log_tail", "") or snapshot.get("bootstrap_log_tail", "")
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
             for line in re.split(r"[\r\n]+", log) if line.strip()]
    line = (lines[-1] if lines else "ainda sem linhas no log vLLM")
    return line[-220:]


def wait_ready(state: dict, seconds: int = 900, *, interval: float = 5) -> None:
    if seconds < 1:
        raise RuntimeError("Tempo de espera precisa ser >= 1 segundo.")
    print(f"Aguardando modelo e API (limite {seconds}s). Monitorando o servidor...", flush=True)
    begun = time.monotonic()
    deadline = begun + seconds
    last_phase = None
    last_report = begun - 30
    last_snapshot = None
    last_failure = None
    while time.monotonic() < deadline:
        current = state_read()
        if current and current.get("token") == state.get("token"):
            state = current
        if not pid_running(state.get("tunnel_pid")):
            # Restore only the SSH channel. No VM/model restart.
            try:
                state = tunnel_start(state)
                time.sleep(2)
            except (RuntimeError, OSError) as exc:
                if time.monotonic() - last_report >= 30:
                    print(f"Túnel indisponível: {exc}", flush=True)
                    last_report = time.monotonic()
                time.sleep(min(interval, max(0, deadline - time.monotonic())))
                continue
        if not pid_running(state.get("serve_pid")):
            raise RuntimeError("Processo que hospeda o proxy no kernel terminou. "
                               "Veja ai-colab diagnostics e ai-colab logs; "
                               "a GPU não foi desligada.")
        healthy, snapshot = runtime_status(state, timeout=3)
        if healthy and isinstance(snapshot, dict):
            last_snapshot = snapshot
            if snapshot.get("vllm_alive") is False:
                tail = snapshot.get("log_tail", "")[-2200:]
                raise RuntimeError("Servidor do modelo encerrou durante o carregamento. Últimas linhas:\n"
                                   + tail + "\nA GPU segue ativa: avalie logs antes de ai-colab stop.")
            if snapshot.get("bootstrap_alive") is False and snapshot.get("vllm_alive") is None:
                tail = snapshot.get("bootstrap_log_tail", "")[-2500:]
                raise RuntimeError("Preparação remota terminou sem iniciar o servidor. "
                                   "Log da preparação:\n" + tail
                                   + "\nA GPU segue ativa: use ai-colab stop se não for corrigir agora.")
            if snapshot.get("vllm_ready") is True:
                good, models = local_api(state, timeout=5)
                if good:
                    print(f"API pronta: {models} ({time.monotonic()-begun:.0f}s de espera)")
                    return
                last_failure = models
            phase = snapshot.get("phase", "desconhecida")
            if phase != last_phase or time.monotonic() - last_report >= 30:
                print(f"[{time.monotonic()-begun:.0f}s] {phase} | "
                      f"servidor: {'ativo' if snapshot.get('vllm_alive') else 'não detectado'} | "
                      + _progress_line(snapshot), flush=True)
                last_phase = phase
                last_report = time.monotonic()
        else:
            last_failure = snapshot
            # Compatibilidade com uma VM que ainda tenha proxy da versão anterior.
            good, models = local_api(state, timeout=3)
            if good:
                print(f"API pronta: {models}")
                return
            if time.monotonic() - last_report >= 30:
                print(f"[{time.monotonic()-begun:.0f}s] Aguardando proxy/túnel: {snapshot}", flush=True)
                last_report = time.monotonic()
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    summary = ((f"Fase: {last_snapshot.get('phase')} | "
                f"servidor: {last_snapshot.get('vllm_alive')} | "
                f"Log: {_progress_line(last_snapshot)}") if last_snapshot else
               f"Status do proxy indisponível: {last_failure}")
    raise RuntimeError(f"Limite de {seconds}s atingido. {summary}\n"
                       "A GPU NÃO foi desligada. Para continuar aguardando sem reiniciar: "
                       "ai-colab wait --seconds 1800. Para encerrar e economizar: ai-colab stop.")



def _human_gib(n):
    return f"{float(n) / (1024**3):.1f} GiB"


def stats_text(state: dict) -> str:
    ok, snapshot = runtime_status(state, timeout=6, metrics=True)
    if not ok or not isinstance(snapshot, dict):
        return "API/túnel indisponível: " + str(snapshot)
    met = snapshot.get("metrics", {})
    result = ["Modelo: " + str(snapshot.get("phase", "desconhecido"))]
    gpu = met.get("gpu", {})
    if gpu:
        result += [f"GPU {gpu.get('name', '?')}: {gpu.get('utilization_pct', 0):g}% | "
                   f"VRAM {gpu.get('memory_used_mib', 0) / 1024:.1f}/"
                   f"{gpu.get('memory_total_mib', 0) / 1024:.1f} GiB "
                   f"| {gpu.get('temperature_c', 0):g}°C"]
    ram = met.get("ram", {})
    if ram:
        result.append("RAM: " + _human_gib(ram.get('used_bytes', 0)) + " / " + _human_gib(ram.get('total_bytes', 0)))
    disk = met.get("disk", {})
    if disk:
        result.append("Disco /content livre: " + _human_gib(disk.get('free_bytes', 0)))
    return "\n".join(result)


def do_stats(args):
    state = state_read()
    if not state:
        print("Nenhuma sessão registrada.")
        return
    print(stats_text(state))
    if args.usage:
        print("\n=== Unidades de computação (Colab CLI) ===")
        run(cli_base(state["auth"]) + ["usage"], check=False)


def do_monitor(args):
    if args.interval < 3:
        raise RuntimeError("Intervalo mínimo: 3 segundos")
    if args.usage_every and args.usage_every < 60:
        raise RuntimeError("Consultas de unidades: intervalo mínimo 60 segundos")
    state = state_read()
    if not state:
        raise RuntimeError("Nenhuma sessão registrada")
    print("Monitorando a GPU (Ctrl+C para sair; a GPU continuará ativa).")
    last_usage = 0
    try:
        while True:
            print("\n" + time.strftime("%H:%M:%S") + " | " + stats_text(state), flush=True)
            if args.usage_every and (time.monotonic() - last_usage > args.usage_every or last_usage == 0):
                run(cli_base(state["auth"]) + ["usage"], check=False, timeout=30)
                last_usage = time.monotonic()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Monitor encerrado. Use 'ai-colab stop' para liberar a GPU.")

def do_wait(args):
    """Retoma SOMENTE a espera de uma sessão existente; nunca cria GPU."""
    state = state_read()
    if not state:
        raise RuntimeError("Nenhuma sessão registrada. Execute ai-colab start voice ou ai-colab start code.")
    base = cli_base(state["auth"])
    if not remote_session_exists(base, state["session"]):
        raise RuntimeError("A GPU foi encerrada. 'wait' não cria nova VM. Use ai-colab start voice ou ai-colab start code.")
    if not pid_running(state.get("serve_pid")):
        raise RuntimeError("O processo local que mantém o proxy remoto está inativo. "
                           "Use ai-colab diagnostics; em seguida ai-colab start "
                           "para tentar retomar na mesma GPU.")
    tunnel_start(state)
    wait_ready(state, args.seconds)


def do_start(args) -> None:
    mode = args.mode
    if mode in COMING:
        print(f"{COMING[mode]}. A ferramenta NÃO criará uma GPU para esse modo ainda.")
        return
    if mode not in PROFILES and mode != "voice":
        raise RuntimeError(f"Modalidade desconhecida: {mode}")
    if mode == "voice":
        if args.voice_model == "higgs3":
            if args.gpu == "A100":
                raise RuntimeError("Higgs TTS 3 requer preferencialmente G4/H100 (~80 GiB ou mais). "
                                   "A100 padrão costuma ter 40 GiB; selecione --gpu G4 antes de iniciar a VM.")
            print("ATENÇÃO: Higgs TTS 3 é experimental e usa licença Boson Research/Non-Commercial. "
                  "Uso como serviço ou produto comercial exige outra licença. "
                  "Conteúdo de criadores pode exigir atribuição; consulte a licença oficial.")
        if args.context is not None:
            raise RuntimeError("--context é exclusivo da programação; Voice não usa janela de tokens do vLLM")
        profile = {"mode": "voice", "model": VOICE_MODELS[args.voice_model],
                   "context": 0, "min_gib": VOICE_MIN_GIB[args.voice_model], "gpu_util": None, "tool_parser": None,
                   "reasoning_parser": None, "effort": None, "voice_variant": args.voice_model}
    else:
        profile = dict(PROFILES[mode]); profile["mode"] = mode
        if args.context:
            profile["context"] = args.context
    state = state_read()
    if state and (state.get("mode") != mode or state.get("model") != profile["model"]):
        raise RuntimeError(f"Sessão '{state.get('mode')}' / modelo '{state.get('model')}' registrada. "
                           "Use 'ai-colab stop' antes de trocar de modelo ou variante.")
    if state and mode != "voice" and state.get("context") != profile["context"]:
        if args.context is not None:
            raise RuntimeError("Há uma sessão com outro contexto; encerre com ai-colab stop antes de alterar.")
        # Sessões já abertas podem ter o contexto anterior de 65K.
        # O perfil padrão novo não deve causar nova GPU nem mudar o vLLM à força.
        profile["context"] = state["context"]
        print(f"Mantendo contexto da sessão existente: {profile['context']} tokens. "
              "Para aumentar, encerre a GPU e inicie uma nova sessão.")
    if state:
        ok, msg = local_api(state, timeout=3)
        if ok:
            watchdog_start(state)
            print(f"Já ativa: {msg}. " + ("Use ai-colab voice generate --help" if mode == "voice" else "Use claude-colab para programar."))
            return
    # Verifica requisitos antes de provisionar a GPU.
    ensure_key()
    if not shutil.which("ssh"):
        raise RuntimeError("OpenSSH não encontrado. Instale openssh-client antes de iniciar a GPU.")
    if not (1024 <= args.port <= 65535):
        raise RuntimeError("Porta local deve estar entre 1024 e 65535.")
    if mode != "voice" and profile["context"] < 1024:
        raise RuntimeError("Contexto inválido (mínimo 1024).")
    if not state:
        state = {"mode": mode, "model": profile["model"], "session": args.session,
                 "auth": args.auth, "gpu": (args.gpu or (VOICE_DEFAULT_GPU[args.voice_model] if mode == "voice" else "G4")), "port": args.port,
                 "token": secrets.token_urlsafe(36), "context": profile["context"],
                 "tunnel_pid": None, "serve_pid": None,
                 "watchdog_pid": None, "hf_mode": (args.hf_mode or preferences_read()["hf_mode"]),
                 "voice_variant": profile.get("voice_variant")}
        state_write(state)
    base = cli_base(state["auth"])
    # `status -s` pode terminar com RC=0 MESMO com "Session not found".
    # Conferir a linha [NOME] evita reutilizar uma VM inexistente.
    exists = remote_session_exists(base, state["session"])
    if not exists:
        # Uma VM antiga sumiu: se o usuário não especificou --context,
        # aplicar o novo padrão da versão instalada na próxima sessão.
        if mode != "voice" and args.context is None and state.get("context") != PROFILES[mode]["context"]:
            state["context"] = PROFILES[mode]["context"]
            profile["context"] = state["context"]
            print(f"Nova sessão: contexto padrão atualizado para {state['context']} tokens.")
        # Corrige estado local órfão após queda/occiosidade do Colab.
        for field in ("tunnel_pid", "serve_pid", "watchdog_pid"):
            kill_group(state.get(field))
            state[field] = None
        state_write(state)
        (STATE_DIR / "known_hosts").unlink(missing_ok=True)
        print("Nenhuma sessão Colab ativa com esse nome. Criando GPU...")
        run(base + ["new", "-s", state["session"], "--gpu", state["gpu"]])
        # Verificação pós-criação: não passar diretamente ao upload se a
        # CLI não confirmou uma sessão de verdade.
        if not remote_session_exists(base, state["session"]):
            raise RuntimeError("O comando 'colab new' não confirmou a sessão. Confira 'colab sessions'.")
    else:
        print("Sessão Colab existente confirmada pelo servidor.")
        if pid_running(state.get("serve_pid")):
            print("Execução remota ainda conectada. Recuperando túnel, sem recarregar o modelo...")
            state = tunnel_start(state)
            watchdog_start(state)
            wait_ready(state, args.wait)
            return
    if not pid_running(state.get("tunnel_pid")):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", state["port"]))
            except OSError:
                raise RuntimeError(f"Porta local {state['port']} já está em uso. Encerre o serviço conflitante antes de prosseguir.")
    payload = {"model": profile["model"], "context": profile["context"],
               "min_gib": profile["min_gib"], "gpu_util": profile["gpu_util"],
               "tool_parser": profile["tool_parser"], "reasoning_parser": profile["reasoning_parser"],
               "key": state["token"], "mode": mode, "voice_variant": profile.get("voice_variant"),
               "hf_mode": state.get("hf_mode", "fast")}
    config_file = STATE_DIR / "remote-config.json"
    config_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    config_file.chmod(0o600)
    print("Enviando configuração e proxy para a VM...")
    run(base + ["upload", "-s", state["session"], str(config_file), REMOTE_CONFIG], quiet=True)
    run(base + ["upload", "-s", state["session"], str(ROOT / "remote/proxy.py"), REMOTE_SERVER], quiet=True)
    run(base + ["upload", "-s", state["session"], str(ROOT / "remote/bootstrap.py"), REMOTE_BOOTSTRAP], quiet=True)
    if mode == "voice":
        run(base + ["upload", "-s", state["session"],
                    str(ROOT / "remote/voice_bootstrap.py"), VOICE_BOOTSTRAP_REMOTE], quiet=True)
        run(base + ["upload", "-s", state["session"],
                    str(ROOT / "remote/voice_api.py"), VOICE_API_REMOTE], quiet=True)
    print("Iniciando preparação remota em segundo plano (sem bloquear o kernel)...", flush=True)
    # O kernel da Colab CLI pode não devolver execute_reply mesmo após disparar
    # o bootstrap. Antes de declarar falha, confirmar o efeito remoto por
    # download de PID/log; não criar outra VM nem dar duas partidas no modelo.
    def remote_launch_observed() -> bool:
        with tempfile.TemporaryDirectory(prefix='colab-ai-probe-') as folder:
            for filename in ('/content/colab-ai-bootstrap.pid', '/content/colab-ai-bootstrap.log'):
                local = Path(folder) / Path(filename).name
                try:
                    result = run(base + ['download', '-s', state['session'], filename, str(local)],
                                 check=False, timeout=45, quiet=True)
                    if result.returncode == 0 and local.exists():
                        return True
                except subprocess.TimeoutExpired:
                    pass
        return False
    try:
        run(base + ["exec", "-s", state["session"], "--timeout", "180",
                    "-f", str(ROOT / "remote/launch.py")], timeout=240)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        print("Lançamento remoto não respondeu; verificando se o bootstrap já começou...", flush=True)
        if remote_launch_observed():
            print("Bootstrap detectado na VM. Continuando sem repetir a partida.", flush=True)
        else:
            raise RuntimeError("Não foi possível confirmar o lançamento remoto. "
                               "Não inicialize outra GPU: confira ai-colab status --remote e ai-colab logs --source bootstrap; "
                               "se a VM ainda estiver ativa, tente ai-colab start voice "
                               "(ou start code) novamente, preservando a sessão. "
                               "Ao desistir, execute ai-colab stop.") from exc
    if not pid_running(state.get("serve_pid")):
        # Este exec executa um servidor HTTP real no kernel; o prazo é maior
        # que a duração típica de uma sessão de programação (2 h).
        state["serve_pid"] = launch_process(base + ["exec", "-s", state["session"],
                                                  "--timeout", "14400",
                                                  "-f", str(ROOT / "remote/serve.py")], "kernel-serve.log")
        state_write(state)
    print("Conectando túnel SSH privado (sem ngrok)...")
    state = tunnel_start(state)
    watchdog_start(state)
    wait_ready(state, args.wait)
    print("\nPronto! Execute: " + ("ai-colab voice generate --text \"Olá, mundo!\" --output teste.wav" if mode == "voice" else "claude-colab  (ou ai-colab claude)"))


def do_stop(args) -> None:
    state = state_read()
    if not state:
        print("Nenhuma instância registrada.")
        return
    # Primeiro libera a GPU. Só limpa o estado local se a operação no Colab der certo.
    base = cli_base(state["auth"])
    print(f"Encerrando GPU da sessão {state['session']}...")
    result = run(base + ["stop", "-s", state["session"]], check=False, quiet=True)
    if result.returncode != 0 and not args.force_local:
        print("O Colab não confirmou o encerramento. Estado mantido. Use 'ai-colab stop --force-local' apenas se a sessão realmente não existir.")
        return
    with tunnel_lock():
        # The watchdog cannot commit a new tunnel after session deletion.
        latest = state_read()
        if latest:
            state = latest
        kill_group(state.get("watchdog_pid"))
        kill_group(state.get("tunnel_pid"))
        kill_group(state.get("serve_pid"))
        STATE_FILE.unlink(missing_ok=True)
        (STATE_DIR / "remote-config.json").unlink(missing_ok=True)
        (STATE_DIR / "known_hosts").unlink(missing_ok=True)
    print("Sessão local limpa. Confirme 'colab sessions' se a GPU já estava indisponível.")


def do_status(args) -> None:
    state = state_read()
    if not state:
        print("Nenhuma sessão da ferramenta registrada. GPU não iniciada por esta ferramenta.")
        return
    print(f"Perfil: {state['mode']} | Modelo: {state['model']} | GPU solicitada: {state['gpu']}")
    print(f"Túnel SSH (PID {state.get('tunnel_pid')}): {'ativo' if pid_running(state.get('tunnel_pid')) else 'inativo'}")
    print(f"Servidor Colab CLI (PID {state.get('serve_pid')}): {'ativo' if pid_running(state.get('serve_pid')) else 'inativo'}")
    print(f"Watchdog SSH (PID {state.get('watchdog_pid')}): "
          f"{'ativo' if pid_running(state.get('watchdog_pid')) else 'inativo'}")
    good, detail = local_api(state)
    print(f"API: {'pronta' if good else 'indisponível'} ({detail})")
    snapshot_ok, snapshot = runtime_status(state)
    if snapshot_ok and isinstance(snapshot, dict):
        print(f"Servidor: {snapshot.get('phase')} | processo: "
              f"{'ativo' if snapshot.get('vllm_alive') else 'não detectado'} | "
              f"pronto: {snapshot.get('vllm_ready')}")
        if not good:
            print(f"Log: {_progress_line(snapshot)}")
    if args.remote:
        run(cli_base(state["auth"]) + ["status", "-s", state["session"]], check=False)


def do_reconnect(args):
    state = state_read()
    if not state:
        raise RuntimeError("Nenhuma sessão. Execute ai-colab start voice ou ai-colab start code.")
    state = tunnel_start(state, force=True)
    watchdog_start(state)
    good, detail = local_api(state, timeout=8)
    if good:
        print(f"Reconectado: {detail}")
    else:
        print(f"Túnel criado; API ainda não disponível ({detail}). Veja diagnostics/logs.")


def do_claude(args):
    state = state_read()
    if not state:
        raise RuntimeError("GPU não iniciada. Execute: ai-colab start code")
    if state.get("mode") == "voice":
        raise RuntimeError("Sessão Voice ativa. Claude Code exige um perfil de programação; use stop/start code")
    good, detail = local_api(state, timeout=3)
    if not good:
        do_reconnect(args)
        good, detail = local_api(state, timeout=5)
    if not good:
        raise RuntimeError(f"Endpoint indisponível: {detail}. Use 'ai-colab diagnostics'.")
    watchdog_start(state)
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("Claude Code CLI não encontrado no PATH.")
    env = os.environ.copy()
    for key in ["ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"]:
        env.pop(key, None)
    env.update({
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{state['port']}",
        "ANTHROPIC_AUTH_TOKEN": state["token"],
        "ANTHROPIC_MODEL": state["model"],
        "ANTHROPIC_DEFAULT_OPUS_MODEL": state["model"],
        "ANTHROPIC_DEFAULT_SONNET_MODEL": state["model"],
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": state["model"],
        "CLAUDE_CODE_EFFORT_LEVEL": "medium",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384",
        # Um gateway com modelo desconhecido não informa automaticamente
        # a capacidade real. Usar o limite desta sessão (131072 no perfil
        # code novo; manter o valor legado de sessões já existentes).
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(
            state.get("context") or PROFILES.get(state.get("mode"), PROFILES["code"])["context"]),
        # Iniciar compactação antecipada, antes que um grande resultado de
        # ferramenta leve o prompt para além do limite do vLLM.
        # O usuário pode ajustar a porcentagem ao editar o launcher.
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "75",
    })
    argv = list(args.extra)
    if argv and argv[0] == "--":
        argv = argv[1:]
    # Pedido expresso: modo sem prompts de permissão para a CLI dedicada.
    # Nunca altera o comando `claude` do usuário, nem settings.json.
    # O opt-out é útil ao abrir repositórios não confiáveis.
    bypass = os.environ.get("COLAB_AI_SKIP_PERMISSIONS", "1").strip().lower()
    if bypass not in ("0", "false", "no", "off") and "--dangerously-skip-permissions" not in argv:
        argv.insert(0, "--dangerously-skip-permissions")
    os.execvpe(claude, [claude, "--model", state["model"], *argv], env)


def do_diagnostics(args):
    state = state_read()
    if not state:
        print("Sem estado."); return
    for filename in ("kernel-serve.log", "ssh-tunnel.log", "watchdog.log"):
        path = STATE_DIR / filename
        print(f"\n=== {filename} ===")
        if path.exists():
            print("\n".join(path.read_text(errors="replace").splitlines()[-30:]))
        else:
            print("(sem log)")
    do_status(argparse.Namespace(remote=True))


def do_logs(args):
    """Baixa logs do runtime sem depender de `colab ssh -- CMD`.

    --source bootstrap ajuda a diagnosticar instalações que ainda não
    chegaram à etapa de iniciar o vLLM. Nunca cria uma nova GPU.
    """
    if args.lines < 1 or args.lines > 5000:
        raise RuntimeError("--lines precisa estar entre 1 e 5000.")
    state = state_read()
    if not state:
        raise RuntimeError("Nenhuma sessão local. Use ai-colab status --remote para verificar.")
    base = cli_base(state["auth"])
    if not remote_session_exists(base, state["session"]):
        raise RuntimeError("A sessão Colab não está ativa. Logs da VM encerrada não ficam disponíveis.")
    filename = {"vllm": "/content/colab-ai-voice.log" if state.get("mode") == "voice" else "/content/colab-ai-vllm.log",
                "bootstrap": "/content/colab-ai-bootstrap.log",
                "omni": "/content/colab-ai-voice-omni.log"}[args.source]
    if args.source == 'omni' and state.get('voice_variant') != 'higgs3':
        raise RuntimeError('Log Omni disponível apenas no perfil Higgs TTS 3')
    with tempfile.TemporaryDirectory(prefix="colab-ai-log-") as temp:
        local_path = Path(temp) / Path(filename).name
        cmd = base + ["download", "-s", state["session"], filename, str(local_path)]
        result = run(cmd, check=False, quiet=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError("Falha ao baixar logs remotos: " + result.stdout[-900:])
        if not local_path.exists():
            raise RuntimeError("A CLI não gerou o arquivo de log solicitado.")
        for line in deque(local_path.read_text(errors="replace").splitlines(), maxlen=args.lines):
            print(line)


def do_test(args):
    state = state_read()
    if not state:
        raise RuntimeError("Sem sessão. Execute ai-colab start code ou start voice primeiro.")
    if state.get("mode") == "voice":
        voice_args = argparse.Namespace(action="generate", text="Olá! Teste de geração de voz no Colab.",
            file=None, output=str(STATE_DIR / "voice-test.wav"), language="Portuguese",
            speaker="Ryan", style="Fale de maneira natural e acolhedora.", description=None,
            reference=None, reference_text=None, consent=False, timeout=args.timeout)
        if state.get("voice_variant") == "design":
            voice_args.action = "design"
            voice_args.description = "Voz brasileira calorosa e profissional, boa dicção."
        elif state.get("voice_variant") == "clone":
            print("Teste de clonagem requer áudio autorizado. Use 'ai-colab voice clone --help'.")
            return
        if state.get('voice_variant') not in ('custom', 'design', 'clone'):
            voice_args.action = 'generate'
            voice_args.speaker = 'pf_dora' if state.get('voice_variant') == 'kokoro' else 'Ryan'
            voice_args.speed = 1.0
            voice_args.temperature = 0.8
            voice_args.exaggeration = 0.5
            voice_args.cfg_weight = 0.5
            voice_args.steps = 10
        do_voice(voice_args)
        return
    url = f"http://127.0.0.1:{state['port']}/v1/messages"
    body = {"model": state["model"], "max_tokens": 256,
            "messages": [{"role": "user", "content": "Responda apenas OK."}]}
    request = urllib.request.Request(url, method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Api-Key": state["token"],
                 "Anthropic-Version": "2023-06-01"})
    begun = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.load(response)
            print(f"HTTP {response.status} | {time.monotonic()-begun:.1f}s | modelo: {payload.get('model',state['model'])}")
            pieces = payload.get("content", [])
            message = " ".join(x.get("text", "") for x in pieces if x.get("type") == "text")
            print("Resposta:", message[:1000] if message else "[Sem texto dentro do limite de tokens]")
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail}")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Falha na inferência: {exc}")


def do_usage(args):
    state = state_read(); auth = (state.get("auth") or args.auth)
    run(cli_base(auth) + ["usage"])


def do_models(args):
    print("Modelos e modalidades:")
    for m, p in PROFILES.items():
        print(f"  {m:13} {p['model']} — {p['note']}")
    for variant, model in VOICE_MODELS.items():
        extra = VOICE_NOTES.get(variant, "Qwen3-TTS")
        print(f"  {'voice/'+variant:18} {model} — GPU {VOICE_DEFAULT_GPU[variant]} | {extra}")
    for m, desc in COMING.items():
        print(f"  {m:13} {desc}")


def do_voice(args):
    """Solicita WAV através do proxy autenticado; saída escrita atomicamente."""
    import base64
    state = state_read()
    if not state or state.get('mode') != 'voice':
        raise RuntimeError("Inicie a modalidade Voice: ai-colab start voice --gpu L4")
    selected = state.get('voice_variant', 'custom')
    if args.action not in VOICE_CAPABILITIES.get(selected, set()):
        needed = {'generate': 'custom', 'design': 'design', 'clone': 'clone'}[args.action]
        raise RuntimeError(f"Modelo em execução: {selected}. "
                           f"Para '{args.action}', execute ai-colab stop e depois "
                           f"ai-colab start voice --voice-model {needed} --gpu L4")
    if args.action == 'clone' and not args.consent:
        raise RuntimeError("Clonagem requer --consent, confirmando autorização expressa do titular da voz")
    if args.action == 'clone' and selected == 'clone' and not args.reference_text.strip():
        raise RuntimeError("Qwen Base exige --reference-text com a transcrição da referência")
    content = (Path(args.file).expanduser().read_text(encoding='utf-8') if args.file else args.text)
    if not (1 <= len(content.strip()) <= 1800):
        raise RuntimeError("Texto deve conter entre 1 e 1800 caracteres; divida textos longos em partes")
    out = Path(args.output).expanduser().resolve()
    if not out.parent.is_dir():
        raise RuntimeError(f"Pasta de saída não existe: {out.parent}")
    if args.timeout < 1:
        raise RuntimeError("Timeout precisa ser positivo")
    payload = {'text': content.strip(), 'language': args.language}
    if args.action == 'generate':
        payload.update({'speaker': ('pf_dora' if selected == 'kokoro' and args.speaker == 'Ryan' else args.speaker), 'instruct': args.style,
                        'speed': args.speed, 'temperature': args.temperature,
                        'exaggeration': args.exaggeration, 'cfg_weight': args.cfg_weight,
                        'inference_timesteps': args.steps})
    elif args.action == 'design':
        payload['instruct'] = args.description
        payload['inference_timesteps'] = args.steps
    if args.action == 'clone':
        ref = Path(args.reference).expanduser().resolve()
        if not ref.is_file():
            raise RuntimeError(f"Referência não encontrada: {ref}")
        if ref.stat().st_size > 4 * 1024 * 1024:
            raise RuntimeError("Referência excede 4 MB; use um trecho curto")
        payload.update({'consent': True, 'ref_text': args.reference_text or '',
                        'ref_audio_b64': base64.b64encode(ref.read_bytes()).decode('ascii')})
    url = f"http://127.0.0.1:{state['port']}/_colab_ai/voice/generate"
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'X-Api-Key': state['token']}, method='POST')
    print('Gerando voz no Colab; o resultado será salvo somente no arquivo indicado...', flush=True)
    begin = time.monotonic()
    with tempfile.NamedTemporaryFile(prefix='.colab-voice-', suffix='.wav',
                                     dir=out.parent, delete=False) as temp:
        tmp = Path(temp.name)
        try:
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    if resp.headers.get('Content-Type', '').split(';')[0] != 'audio/wav':
                        raise RuntimeError('Servidor não retornou WAV')
                    total = 0
                    while True:
                        chunk = resp.read(65536)
                        if not chunk: break
                        total += len(chunk)
                        if total > 40 * 1024 * 1024:
                            raise RuntimeError('WAV ultrapassa 40 MB')
                        temp.write(chunk)
            except urllib.error.HTTPError as exc:
                detail = exc.read(900).decode('utf-8', errors='replace')
                raise RuntimeError(f'API de voz HTTP {exc.code}: {detail}') from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f'API indisponível: {exc}. Use ai-colab status/reconnect') from exc
            temp.flush()
            if total < 44:
                raise RuntimeError('Resposta WAV muito curta')
            temp.close()
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
    print(f'Arquivo criado: {out} | {total / 1024:.1f} KiB | {time.monotonic()-begin:.1f}s')


def parser_make():
    p = argparse.ArgumentParser(prog="ai-colab", description="Gerenciador multimodal de IA no Colab via CLI + SSH.")
    p.add_argument("--version", action="version", version=f"Colab AI CLI {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("start", help="Inicia uma modalidade")
    a.add_argument("mode", choices=[*PROFILES, "voice", *COMING], default="code", nargs="?")
    a.add_argument("--gpu", default=None, help="Voice: L4 por padrão, T4 para kokoro, A100 para higgs3; Code: G4")
    a.add_argument("--voice-model", choices=list(VOICE_MODELS), default="custom",
                   help="custom/design/clone (Qwen), voxcpm2, chatterbox-br, kokoro, higgs3")
    a.add_argument("--hf-mode", choices=["fast", "normal"], help="Sobrescreve o padrão hf_xet para esta sessão")
    a.add_argument("--session", default=SESSION)
    a.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    a.add_argument("--port", type=int, default=LOCAL_PORT)
    a.add_argument("--context", type=int, help="Tokens de contexto. Experimente 32768 se houver OOM.")
    a.add_argument("--wait", type=int, default=900, help="Espera em segundos para download+modelo (padrão 900)")
    a.set_defaults(func=do_start)
    a = sub.add_parser("wait", help="Aguarda modelo na GPU já ativa; não reinstala nem cria sessão")
    a.add_argument("--seconds", type=int, default=1800, help="Espera adicional em segundos (padrão 1800)")
    a.set_defaults(func=do_wait)
    a = sub.add_parser("stop", help="Libera GPU e encerra sessão")
    a.add_argument("--force-local", action="store_true", help="Limpa somente estado local caso a VM já tenha sumido")
    a.set_defaults(func=do_stop)
    a = sub.add_parser("status", help="Mostra estado e API")
    a.add_argument("--remote", action="store_true")
    a.set_defaults(func=do_status)
    sub.add_parser("reconnect", help="Restaura somente túnel SSH").set_defaults(func=do_reconnect)
    sub.add_parser("stats", help="VRAM, GPU, RAM, disco e unidades").add_argument("--usage", action="store_true")
    sub.choices["stats"].set_defaults(func=do_stats)
    a = sub.add_parser("monitor", help="Monitora recursos continuamente (não inicia GPU)")
    a.add_argument("--interval", type=int, default=10)
    a.add_argument("--usage-every", type=int, default=300,
                   help="Atualizar unidades do Colab a cada N segundos (0 para desativar)")
    a.set_defaults(func=do_monitor)
    a = sub.add_parser("hf", help="Configura download com hf_xet (fast / normal)")
    a.add_argument("action", choices=["fast", "normal", "status"], nargs="?", default="status")
    a.set_defaults(func=do_hf)
    sub.add_parser("_watchdog", help=argparse.SUPPRESS).set_defaults(func=do_watchdog)
    sub.add_parser("models", help="Lista perfis e modalidades").set_defaults(func=do_models)
    sub.add_parser("ssh-check", help="Valida chave SSH exclusiva, sem criar GPU").set_defaults(func=do_ssh_check)
    sub.add_parser("diagnostics", help="Logs locais + estado remoto").set_defaults(func=do_diagnostics)
    a = sub.add_parser("logs", help="Últimas linhas do servidor do modelo")
    a.add_argument("--lines", type=int, default=80)
    a.add_argument("--source", choices=["vllm", "bootstrap", "omni"], default="vllm",
                   help="vllm/voice, bootstrap (instalação) ou omni (backend Higgs)")
    a.set_defaults(func=do_logs)
    a = sub.add_parser("test", help="Executa geração real pela API")
    a.add_argument("--timeout", type=int, default=180)
    a.set_defaults(func=do_test)
    voice = sub.add_parser("voice", help="Gera WAV usando a engine Voice selecionada")
    voice_sub = voice.add_subparsers(dest="action", required=True)
    for action in ("generate", "design", "clone"):
        v = voice_sub.add_parser(action, help={"generate":"TTS de voz predefinida", "design":"Criar voz por descrição", "clone":"Clonar voz autorizada"}[action])
        group = v.add_mutually_exclusive_group(required=True)
        group.add_argument("--text", help="Texto para sintetizar")
        group.add_argument("--file", help="Arquivo de texto UTF-8")
        v.add_argument("--output", required=True, help="Arquivo WAV de destino no computador")
        v.add_argument("--language", default="Portuguese")
        v.add_argument("--timeout", type=int, default=600)
        if action == "generate":
            v.add_argument("--speaker", default="Ryan", help="Qwen: Ryan etc.; Kokoro: pf_dora/pm_alex/pm_santa")
            v.add_argument("--style", default="", help="VoxCPM/Higgs: descrição ou orientação expressiva")
            v.add_argument("--speed", type=float, default=1., help="Kokoro: velocidade 0.5–2.0")
            v.add_argument("--temperature", type=float, default=.8, help="Chatterbox: temperatura 0.05–2")
            v.add_argument("--exaggeration", type=float, default=.5, help="Chatterbox: expressividade 0.25–2")
            v.add_argument("--cfg-weight", type=float, default=.5, help="Chatterbox: 0–1")
            v.add_argument("--steps", type=int, default=10, help="VoxCPM: etapas de inferência")
        if action == "design":
            v.add_argument("--description", required=True, help="Descrição da voz")
            v.add_argument("--steps", type=int, default=10, help="VoxCPM: etapas de inferência")
        if action == "clone":
            v.add_argument("--reference", required=True, help="Arquivo WAV/FLAC/OGG de referência")
            v.add_argument("--reference-text", default="", help="Transcrição literal; obrigatória para Qwen; opcional para VoxCPM")
            v.add_argument("--consent", action="store_true", help="Confirma autorização para reproduzir esta voz")
        v.set_defaults(func=do_voice)
    a = sub.add_parser("usage", help="Unidades computacionais no Colab")
    a.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    a.set_defaults(func=do_usage)
    a = sub.add_parser("claude", help="Claude Code com Qwen; conta Anthropic normal preservada")
    a.add_argument("extra", nargs=argparse.REMAINDER)
    a.set_defaults(func=do_claude)
    return p


def main(argv=None):
    args = parser_make().parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
