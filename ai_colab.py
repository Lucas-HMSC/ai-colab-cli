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

VERSION = "0.3.2"
ROOT = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "colab-ai"
STATE_FILE = STATE_DIR / "state.json"
PREFS_FILE = STATE_DIR / "preferences.json"
LOCK_FILE = STATE_DIR / "tunnel.lock"
VOICE_BOOTSTRAP_REMOTE = "/content/colab-ai-voice-bootstrap.py"
VOICE_API_REMOTE = "/content/colab-ai-voice-api.py"
ASR_BOOTSTRAP_REMOTE = "/content/colab-ai-transcribe-bootstrap.py"
ASR_API_REMOTE = "/content/colab-ai-transcribe-api.py"
VIDEO_BOOTSTRAP_REMOTE = "/content/colab-ai-video-bootstrap.py"
VIDEO_API_REMOTE = "/content/colab-ai-video-api.py"
VIDEO_GENERATOR_REMOTE = "/content/colab-ai-video-generate.py"

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
    }

# Catálogo de vídeo; Hunyuan 1.5 e Wan 2.2-5B com backends implementados.
VIDEO_MODELS = {
    "hunyuan15": {
        "name": "HunyuanVideo-1.5 (8.3B)",
        "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v / 480p_i2v_step_distilled",
        "gpu": "L4 (G4 opcional; T4 não recomendado nesta implementação BF16)",
        "tasks": ("t2v", "i2v"),
        "memory": ">=14 GB de VRAM com offload (fabricante)",
        "note": "Implementado 480p; T2V 50 passos; I2V step-distilled 12 passos (pesos separados); sem áudio",
        "license": "Tencent Hunyuan Community License",
    },
    "wan22-5b": {
        "name": "Wan2.2 TI2V-5B",
        "repo": "Wan-AI/Wan2.2-TI2V-5B",
        "gpu": "G4 (L4 24 GB com offload)",
        "tasks": ("t2v", "i2v"),
        "memory": ">=24 GB de VRAM com offload (fabricante)",
        "note": "720p, 24 fps; sem áudio nativo",
        "license": "Apache-2.0 (verificar termos de uso dos pesos)",
    },
    "wan22-14b": {
        "name": "Wan2.2 T2V/I2V A14B",
        "repo": "Wan-AI/Wan2.2-T2V-A14B e Wan-AI/Wan2.2-I2V-A14B",
        "gpu": "G4 ~96 GB",
        "tasks": ("t2v", "i2v"),
        "memory": ">=80 GB de VRAM (fabricante)",
        "note": "480p/720p; maior peso e custo computacional; sem áudio nativo",
        "license": "Apache-2.0 (verificar termos de uso dos pesos)",
    },
    "ltx25": {
        "name": "LTX-2.5 Fast",
        "repo": "Lightricks/LTX-2.5",
        "gpu": "G4 (L4 ~24 GB com offload, sujeito a RAM/disco)",
        "tasks": ("t2v", "i2v", "audio-video"),
        "memory": ">=24 GB de VRAM recomendado para Linux na ferramenta oficial",
        "note": "Vídeo + áudio sincronizados; pesos com acesso condicionado e requisitos de disco elevados",
        "license": "LTX-2.x Community License (avaliar uso comercial)",
    },
}

# Orientação restrita ao launcher claude-colab, não ao comando claude original
# nem aos arquivos CLAUDE.md do usuário. Não bloqueia tecnicamente ferramentas.
GLOBAL_VISION_GUIDANCE = (
    "Ao lidar com imagens locais (incluindo screenshots, referências visuais e "
    "catalogação de diretórios), primeiro enumere os arquivos com as ferramentas "
    "locais adequadas e utilize a ferramenta nativa Read do Claude Code para "
    "abrir cada imagem e inspecionar os pixels. Baseie cada descrição no "
    "conteúdo realmente observado, não apenas no nome do arquivo. "
    "Não busque API keys, não use APIs de visão externas, OCR de terceiros ou "
    "outros provedores para essas tarefas, exceto mediante pedido explícito "
    "do usuário. Se Read não conseguir abrir uma imagem, explique a limitação "
    "e não invente o conteúdo. Para diretórios grandes, processe imagens em "
    "lotes e evite reproduzir dados confidenciais desnecessariamente."
)

VIDEO_ACTIVE_MODELS = frozenset({"hunyuan15", "wan22-5b"})
VIDEO_DEFAULT_GPU = {"hunyuan15": "L4", "wan22-5b": "G4"}
VIDEO_MIN_GIB = {"hunyuan15": 14, "wan22-5b": 21}
VIDEO_MODEL_IDS = {"hunyuan15": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
                   "wan22-5b": "Wan-AI/Wan2.2-TI2V-5B"}

# Duas engines ASR PT-BR isoladas por Python 3.12, sem cache no Drive.
ASR_MODELS = {
    "whisper": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "qwen": "Qwen/Qwen3-ASR-1.7B",
}
ASR_DEFAULT_GPU = {"whisper": "T4", "qwen": "L4"}
ASR_MIN_GIB = {"whisper": 7, "qwen": 12}
ASR_NOTES = {
    "whisper": "faster-whisper, legendas por segmentos, PT-BR",
    "qwen": "Qwen3-ASR + ForcedAligner-0.6B, legendas com tempos reais",
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
                     "voxcpm2": "L4", "chatterbox-br": "L4", "kokoro": "T4", "higgs3": "H100"}
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
               "higgs3": "EXPERIMENTAL: H100 é a GPU da receita oficial; G4 não validada; licença Research/Creator Grant"}
# O backend Higgs fica arquivado apenas para possível restauração futura.
# Ele NÃO aparece no catálogo nem pode ser selecionado pelo usuário.
HIDDEN_VOICE_MODELS = frozenset({"higgs3"})
ACTIVE_VOICE_MODELS = {name: repo for name, repo in VOICE_MODELS.items() if name not in HIDDEN_VOICE_MODELS}

SPEAKERS = ("Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")

MODALITY_ALIASES = {
    "tts": "voice",
    "asr": "transcribe",
}

MODEL_CATALOG = {
    "code": {
        "qwen": {
            "variant": "code",
            "aliases": ("code", "default", "27b", "qwen3.8-27b"),
            "gpu": "G4",
            "repo": PROFILES["code"]["model"],
            "note": PROFILES["code"]["note"],
        },
        "qwen-next": {
            "variant": "code-next",
            "aliases": ("code-next", "next", "80b"),
            "gpu": "G4",
            "repo": PROFILES["code-next"]["model"],
            "note": PROFILES["code-next"]["note"],
        },
    },
    "voice": {
        "custom": {
            "variant": "custom",
            "aliases": ("qwen", "default"),
            "gpu": VOICE_DEFAULT_GPU["custom"],
            "repo": VOICE_MODELS["custom"],
            "note": "Qwen3-TTS voz predefinida (PT-BR, speakers Ryan/Vivian/etc.)",
        },
        "design": {
            "variant": "design",
            "aliases": (),
            "gpu": VOICE_DEFAULT_GPU["design"],
            "repo": VOICE_MODELS["design"],
            "note": "Qwen3-TTS criação de voz por descrição textual",
        },
        "clone": {
            "variant": "clone",
            "aliases": (),
            "gpu": VOICE_DEFAULT_GPU["clone"],
            "repo": VOICE_MODELS["clone"],
            "note": "Qwen3-TTS clonagem de voz com áudio + transcrição",
        },
        "voxcpm2": {
            "variant": "voxcpm2",
            "aliases": ("voxcpm",),
            "gpu": VOICE_DEFAULT_GPU["voxcpm2"],
            "repo": VOICE_MODELS["voxcpm2"],
            "note": VOICE_NOTES["voxcpm2"],
        },
        "chatterbox-br": {
            "variant": "chatterbox-br",
            "aliases": ("chatterbox",),
            "gpu": VOICE_DEFAULT_GPU["chatterbox-br"],
            "repo": VOICE_MODELS["chatterbox-br"],
            "note": VOICE_NOTES["chatterbox-br"],
        },
        "kokoro": {
            "variant": "kokoro",
            "aliases": (),
            "gpu": VOICE_DEFAULT_GPU["kokoro"],
            "repo": VOICE_MODELS["kokoro"],
            "note": VOICE_NOTES["kokoro"],
        },
    },
    "transcribe": {
        "whisper": {
            "variant": "whisper",
            "aliases": ("default",),
            "gpu": ASR_DEFAULT_GPU["whisper"],
            "repo": ASR_MODELS["whisper"],
            "note": ASR_NOTES["whisper"],
        },
        "qwen": {
            "variant": "qwen",
            "aliases": ("qwen-asr",),
            "gpu": ASR_DEFAULT_GPU["qwen"],
            "repo": ASR_MODELS["qwen"],
            "note": ASR_NOTES["qwen"],
        },
    },
    "video": {
        "hunyuan15": {
            "variant": "hunyuan15",
            "aliases": ("hunyuan", "default"),
            "gpu": VIDEO_DEFAULT_GPU["hunyuan15"],
            "repo": VIDEO_MODELS["hunyuan15"]["repo"],
            "note": VIDEO_MODELS["hunyuan15"]["note"],
        },
        "wan22-5b": {
            "variant": "wan22-5b",
            "aliases": ("wan", "wan22"),
            "gpu": VIDEO_DEFAULT_GPU["wan22-5b"],
            "repo": VIDEO_MODELS["wan22-5b"]["repo"],
            "note": VIDEO_MODELS["wan22-5b"]["note"],
        },
    },
}


def _find_model_entry(catalog_key: str, model_name: str) -> tuple[str, dict] | None:
    target = model_name.strip().lower()
    for canonical, info in MODEL_CATALOG[catalog_key].items():
        if target == canonical.lower() or target in info["aliases"] or target == info["variant"].lower():
            return canonical, info
    return None


def resolve_start_target(args) -> tuple[str, str]:
    """Normaliza modalidade e modelo a partir de --model e flags legadas."""
    raw_mode = getattr(args, "mode", "code") or "code"
    mode = MODALITY_ALIASES.get(raw_mode, raw_mode)
    if mode in COMING:
        args.mode = mode
        return mode, mode

    if mode not in PROFILES and mode not in ("voice", "transcribe", "video"):
        raise RuntimeError(f"Modalidade desconhecida: {raw_mode}")

    catalog_key = "code" if mode in PROFILES else mode
    requested = getattr(args, "model", None)

    if requested is not None:
        match = _find_model_entry(catalog_key, requested)
        if not match:
            valid = ", ".join(MODEL_CATALOG[catalog_key].keys())
            for other_key in MODEL_CATALOG:
                if other_key == catalog_key:
                    continue
                other_match = _find_model_entry(other_key, requested)
                if other_match:
                    other_canonical, _ = other_match
                    raise RuntimeError(
                        f"Modelo '{requested}' pertence à modalidade '{other_key}', não '{mode}'. "
                        f"Modelos válidos para '{mode}': {valid}. "
                        f"Use: ai-colab start {other_key} --model {other_canonical}"
                    )
            raise RuntimeError(
                f"Modelo desconhecido para '{mode}': '{requested}'. "
                f"Modelos válidos: {valid}. Consulte: ai-colab models {mode}"
            )
        _, info = match
        variant = info["variant"]
    else:
        if mode in PROFILES:
            variant = mode
        elif mode == "voice":
            variant = getattr(args, "voice_model", "custom")
        elif mode == "transcribe":
            variant = getattr(args, "asr_model", "whisper")
        else:
            variant = getattr(args, "video_model", "hunyuan15")

    args.mode = mode
    if mode == "voice":
        if variant not in ACTIVE_VOICE_MODELS:
            raise RuntimeError(f"Modelo de voz desativado ou desconhecido: {variant}")
        args.voice_model = variant
    elif mode == "transcribe":
        if variant not in ASR_MODELS:
            raise RuntimeError(f"Modelo ASR desconhecido: {variant}")
        args.asr_model = variant
    elif mode == "video":
        if variant not in VIDEO_ACTIVE_MODELS:
            raise RuntimeError(f"Modelo de vídeo não implementado: {variant}")
        args.video_model = variant

    return mode, variant



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


def runtime_status(state: dict, timeout: int = 5, *, metrics: bool = False, deep: bool = False) -> tuple[bool, dict | str]:
    """Consulta status autenticado do servidor, mesmo antes do vLLM estar pronto.

    O endpoint do proxy e o túnel continuam disponíveis durante downloads e
    carregamento. Assim é possível diferenciar demora, falha e API pronta.
    """
    url = f"http://127.0.0.1:{state.get('port', LOCAL_PORT)}/_colab_ai/status" + ("?metrics=1" + ("&deep=1" if deep else "") if metrics else "")
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
                elapsed = time.monotonic()-begun
                print(f"[{elapsed:.0f}s] Aguardando proxy/túnel: {snapshot}", flush=True)
                if elapsed > 120 and state.get('mode') in ('voice', 'transcribe'):
                    print("Se persistir, examine sem reiniciar a GPU: "
                          "ai-colab logs --source bootstrap --lines 80; "
                          "ai-colab status --remote; ai-colab diagnostics", flush=True)
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


def do_gpu_status(args):
    """Lê métricas pelo túnel autenticado EXISTENTE, sem colab ssh."""
    state = state_read()
    if not state:
        raise RuntimeError("Nenhuma sessão iniciada. Use ai-colab start antes de consultar GPU.")
    if args.watch is not None and args.watch < 3:
        raise RuntimeError("--watch precisa ser >= 3 segundos")
    try:
        while True:
            ok, response = runtime_status(state, timeout=15 if args.deep else 7,
                                          metrics=True, deep=args.deep)
            if not ok:
                raise RuntimeError("Túnel/API indisponível: " + str(response) +
                                   ". Verifique ai-colab status ou ai-colab reconnect.")
            if args.json:
                print(json.dumps(response.get('metrics', {}), ensure_ascii=False, indent=2), flush=True)
            else:
                print("=== GPU remota (conexão SSH existente) ===", flush=True)
                print(stats_text_from_snapshot(response), flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("Monitor de GPU encerrado; a sessão Colab continua ativa.")


def stats_text_from_snapshot(snapshot):
    met = snapshot.get('metrics') or {}
    lines = ["Estado: " + str(snapshot.get('phase', 'desconhecido'))]
    gpu = met.get('gpu') or {}
    if gpu:
        lines += [f"GPU: {gpu.get('name','?')} | uso {gpu.get('utilization_pct', 0):g}% "
                  f"| VRAM {gpu.get('memory_used_mib',0)/1024:.1f}/{gpu.get('memory_total_mib',0)/1024:.1f} GiB "
                  f"| temperatura {gpu.get('temperature_c',0):g}°C"]
        if gpu.get('driver_version'):
            lines += [f"Driver NVIDIA: {gpu['driver_version']}"]
    else:
        lines.append("nvidia-smi não disponibilizou métricas; consulte logs/status.")
    if met.get('torch'):
        t = met['torch']
        lines.append(f"PyTorch: {t.get('torch_version', '?')} | CUDA do torch: {t.get('cuda_version', '?')} "
                     f"| disponível: {t.get('cuda_available', False)} | dispositivo: {t.get('device', '?')}")
        if t.get('error'):
            lines.append('Diagnóstico PyTorch: ' + str(t['error']))
    for proc in met.get('gpu_processes', []):
        lines.append(f"Processo GPU PID {proc.get('pid','?')} | {proc.get('memory_used_mib','?')} MiB | "
                     f"{proc.get('process_name','?')}")
    return "\n".join(lines)


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
        raise RuntimeError("Nenhuma sessão registrada. Execute ai-colab start code/voice/transcribe.")
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
    raw_mode = getattr(args, "mode", "code") or "code"
    mode = MODALITY_ALIASES.get(raw_mode, raw_mode)
    if mode in COMING:
        print(f"{COMING[mode]}. A ferramenta NÃO criará uma GPU para esse modo ainda.")
        return
    mode, variant = resolve_start_target(args)
    if mode == "voice":
        if variant not in ACTIVE_VOICE_MODELS:
            raise RuntimeError(f"Modelo de voz desativado ou desconhecido: {variant}")
        if variant == "higgs3":
            if args.gpu == "A100":
                raise RuntimeError("Higgs TTS 3: A100 padrão 40 GiB não é a GPU validada na receita oficial. "
                                   "Use --gpu H100, ou --gpu G4 por sua conta e risco experimental.")
            if args.gpu == "G4":
                print("ATENÇÃO: Higgs TTS 3 só está documentado oficialmente na H100. "
                      "G4 é experimental e pode falhar na inicialização; consulte logs --source omni.")
            print("ATENÇÃO: Higgs TTS 3 é experimental e usa licença Boson Research/Non-Commercial. "
                  "Uso como serviço ou produto comercial exige outra licença. "
                  "Conteúdo de criadores pode exigir atribuição; consulte a licença oficial.")
        if args.context is not None:
            raise RuntimeError("--context é exclusivo da programação; Voice não usa janela de tokens do vLLM")
        profile = {"mode": "voice", "model": VOICE_MODELS[variant],
                   "context": 0, "min_gib": VOICE_MIN_GIB[variant], "gpu_util": None, "tool_parser": None,
                   "reasoning_parser": None, "effort": None, "voice_variant": variant}
    elif mode == "video":
        if variant not in VIDEO_ACTIVE_MODELS:
            raise RuntimeError(f"Modelo de vídeo não implementado: {variant}")
        if args.context is not None:
            raise RuntimeError("--context é exclusivo da programação")
        if variant == "hunyuan15" and args.gpu == "T4":
            raise RuntimeError("HunyuanVideo15 BF16 desta implementação requer L4 ou G4, não T4")
        profile = {"mode": "video", "model": VIDEO_MODEL_IDS[variant],
                   "context": 0, "min_gib": VIDEO_MIN_GIB[variant],
                   "gpu_util": None, "tool_parser": None, "reasoning_parser": None,
                   "effort": None, "video_variant": variant}
    elif mode == "transcribe":
        if variant not in ASR_MODELS:
            raise RuntimeError(f"Modelo ASR desconhecido: {variant}")
        if args.context is not None:
            raise RuntimeError("--context é exclusivo da programação; transcrição não usa contexto do vLLM")
        profile = {"mode": "transcribe", "model": ASR_MODELS[variant],
                   "context": 0, "min_gib": ASR_MIN_GIB[variant], "gpu_util": None,
                   "tool_parser": None, "reasoning_parser": None, "effort": None,
                   "asr_variant": variant}
    else:
        profile = dict(PROFILES[variant]); profile["mode"] = mode
        if args.context:
            profile["context"] = args.context
    state = state_read()
    if state and (state.get("mode") != mode or state.get("model") != profile["model"]):
        raise RuntimeError(f"Sessão '{state.get('mode')}' / modelo '{state.get('model')}' registrada. "
                           "Use 'ai-colab stop' antes de trocar de modelo ou variante.")
    if state and mode in PROFILES and state.get("context") != profile["context"]:
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
            print(f"Já ativa: {msg}. " + ("Use ai-colab voice generate --help" if mode == "voice" else
                                                "Use ai-colab transcribe --help" if mode == "transcribe" else
                                                "Use ai-colab video generate --help" if mode == "video" else
                                                "Use claude-colab para programar."))
            return
    # Verifica requisitos antes de provisionar a GPU.
    ensure_key()
    if not shutil.which("ssh"):
        raise RuntimeError("OpenSSH não encontrado. Instale openssh-client antes de iniciar a GPU.")
    if not (1024 <= args.port <= 65535):
        raise RuntimeError("Porta local deve estar entre 1024 e 65535.")
    if mode in PROFILES and profile["context"] < 1024:
        raise RuntimeError("Contexto inválido (mínimo 1024).")
    if not state:
        state = {"mode": mode, "model": profile["model"], "session": args.session,
                 "auth": args.auth, "gpu": (args.gpu or (VOICE_DEFAULT_GPU[args.voice_model] if mode == "voice" else
                                          ASR_DEFAULT_GPU[args.asr_model] if mode == "transcribe" else
                                          VIDEO_DEFAULT_GPU[args.video_model] if mode == "video" else "G4")), "port": args.port,
                 "token": secrets.token_urlsafe(36), "context": profile["context"],
                 "tunnel_pid": None, "serve_pid": None,
                 "watchdog_pid": None, "hf_mode": (args.hf_mode or preferences_read()["hf_mode"]),
                 "voice_variant": profile.get("voice_variant"),
                 "asr_variant": profile.get("asr_variant"),
                 "video_variant": profile.get("video_variant")}
        state_write(state)
    base = cli_base(state["auth"])
    # `status -s` pode terminar com RC=0 MESMO com "Session not found".
    # Conferir a linha [NOME] evita reutilizar uma VM inexistente.
    exists = remote_session_exists(base, state["session"])
    if not exists:
        # Uma VM antiga sumiu: se o usuário não especificou --context,
        # aplicar o novo padrão da versão instalada na próxima sessão.
        if mode in PROFILES and args.context is None and state.get("context") != PROFILES[variant]["context"]:
            state["context"] = PROFILES[variant]["context"]
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
               "asr_variant": profile.get("asr_variant"),
               "video_variant": profile.get("video_variant"),
               "hf_mode": state.get("hf_mode", "fast")}
    if mode == "transcribe" and os.environ.get("HF_TOKEN", "").strip():
        payload["hf_token"] = os.environ["HF_TOKEN"].strip()
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
    elif mode == "transcribe":
        run(base + ["upload", "-s", state["session"],
                    str(ROOT / "remote/transcribe_bootstrap.py"), ASR_BOOTSTRAP_REMOTE], quiet=True)
        run(base + ["upload", "-s", state["session"],
                    str(ROOT / "remote/transcribe_api.py"), ASR_API_REMOTE], quiet=True)
    elif mode == "video":
        for source, destination in (("video_bootstrap.py", VIDEO_BOOTSTRAP_REMOTE),
                                    ("video_api.py", VIDEO_API_REMOTE),
                                    ("video_generate.py", VIDEO_GENERATOR_REMOTE)):
            run(base + ["upload", "-s", state["session"],
                        str(ROOT / "remote" / source), destination], quiet=True)
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
                               "se a VM ainda estiver ativa, tente ai-colab start "
                               "novamente com o mesmo perfil, preservando a sessão. "
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
    print("\nPronto! Execute: " + (
        "ai-colab voice generate --text \"Olá, mundo!\" --output teste.wav" if mode == "voice" else
        "ai-colab transcribe --file audio.mp3 --output transcricao.txt" if mode == "transcribe" else
        "ai-colab video generate --prompt \"um gato caminhando\" --output ~/Videos/gato.mp4" if mode == "video" else
        "claude-colab  (ou ai-colab claude)"))


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
    if getattr(args, "json", False):
        if not state:
            print(json.dumps({"active": False}, ensure_ascii=False))
            return
        good, detail = local_api(state)
        snapshot_ok, snapshot = runtime_status(state)
        print(json.dumps({
            "active": True,
            "mode": state.get("mode"),
            "model": state.get("model"),
            "gpu": state.get("gpu"),
            "port": state.get("port"),
            "token": state.get("token"),
            "tunnel_active": pid_running(state.get("tunnel_pid")),
            "serve_active": pid_running(state.get("serve_pid")),
            "watchdog_active": pid_running(state.get("watchdog_pid")),
            "api_ready": good,
            "api_detail": detail,
            "runtime": snapshot if snapshot_ok else None,
        }, ensure_ascii=False, indent=2))
        return
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
        raise RuntimeError("Nenhuma sessão. Execute ai-colab start code/voice/transcribe.")
    state = tunnel_start(state, force=True)
    watchdog_start(state)
    good, detail = local_api(state, timeout=8)
    if good:
        print(f"Reconectado: {detail}")
    else:
        print(f"Túnel criado; API ainda não disponível ({detail}). Veja diagnostics/logs.")


def apply_global_vision_guidance(argv: list[str], env: dict[str, str]) -> list[str]:
    """Acrescenta instrução global a este launcher sem modificar preferências do Claude.

    Se usuário já definiu --append-system-prompt, mescla em vez de repetir a
    flag (CLI de terceiros pode aceitar apenas a última ocorrência).
    """
    result = list(argv)
    enabled = env.get("COLAB_AI_VISION_GUIDANCE", "1").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        return result
    for idx, arg in enumerate(result):
        if arg == "--append-system-prompt":
            if idx + 1 >= len(result):
                # Deixa o Claude reportar o erro do usuário sem ocultá-lo.
                return result
            result[idx + 1] += "\n\n" + GLOBAL_VISION_GUIDANCE
            return result
        if arg.startswith("--append-system-prompt="):
            result[idx] += "\n\n" + GLOBAL_VISION_GUIDANCE
            return result
    # --append-system-prompt preserva o prompt nativo (em vez de substituí-lo).
    return ["--append-system-prompt", GLOBAL_VISION_GUIDANCE, *result]


def do_video(args):
    """Catálogo informativo; não provisiona GPU e não afirma gerar vídeos."""
    if args.action != "models":
        raise RuntimeError("Subcomando video desconhecido.")
    if getattr(args, "json", False):
        print(json.dumps({"stage": "hunyuan15_wan22-5b_available", "models": VIDEO_MODELS},
                         ensure_ascii=False, indent=2))
        return
    print("Módulo vídeo: HunyuanVideo-1.5 e Wan2.2 TI2V-5B implementados (GPU ainda não validada).")
    print("O catálogo não cria GPU. Para executar, use ai-colab start video --video-model MODELO.\n")
    for key, info in VIDEO_MODELS.items():
        print(f"  {key:13} {info['name']} | {','.join(info['tasks'])}")
        print(f"    GPU sugerida: {info['gpu']} | {info['memory']}")
        print(f"    {info['note']}")
        print(f"    Pesos: {info['repo']} | Licença: {info['license']}")
    print("\nConsulte RESEARCH_VIDEO_2026.md para seleção, requisitos e fontes.")


def do_claude(args):
    state = state_read()
    if not state:
        raise RuntimeError("GPU não iniciada. Execute: ai-colab start code")
    if state.get("mode") in ("voice", "transcribe", "video"):
        raise RuntimeError("Sessão não Code ativa. Claude Code exige Code: use stop/start code")
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
    argv = apply_global_vision_guidance(argv, env)
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
    filename = {"vllm": "/content/colab-ai-voice.log" if state.get("mode") == "voice" else
                        "/content/colab-ai-asr.log" if state.get("mode") == "transcribe" else
                        "/content/colab-ai-video.log" if state.get("mode") == "video" else
                        "/content/colab-ai-vllm.log",
                "bootstrap": "/content/colab-ai-bootstrap.log",
                "omni": "/content/colab-ai-voice-omni.log",
                "video": "/content/colab-ai-video.log"}[args.source]
    if args.source == 'video' and state.get('mode') != 'video':
        raise RuntimeError("Log Video somente para sessões Video")
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
    if state.get("mode") == "video":
        ok, message = local_api(state, timeout=args.timeout)
        if not ok:
            raise RuntimeError("Video não respondeu ao endpoint de saúde: " + message)
        print("Video API responde, mas inferência ainda deve ser validada por video generate.")
        return
    if state.get("mode") == "transcribe":
        if args.file:
            return do_transcribe(argparse.Namespace(file=args.file, output=None, format='txt',
                                                     language='pt', timeout=args.timeout, overwrite=False))
        good, models = local_api(state, timeout=8)
        if not good:
            raise RuntimeError(f"Servidor ASR indisponível: {models}")
        print(f"API ASR pronta: {models}. Para testar inferência real, use:")
        print('  ai-colab transcribe --file /caminho/audio.mp3 --output transcricao.txt')
        return
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


def _build_models_payload(modality: str = "all") -> dict:
    norm = MODALITY_ALIASES.get(modality, modality)
    if norm == "all":
        selected = list(MODEL_CATALOG.keys())
    elif norm in MODEL_CATALOG:
        selected = [norm]
    else:
        raise RuntimeError(f"Modalidade desconhecida para catálogo: {modality}")

    use_aliases = {
        "code": ["code", "claude"],
        "voice": ["voice", "tts"],
        "transcribe": ["transcribe", "asr"],
        "video": ["video"],
    }
    return {
        "modalities": {
            mod: {
                "commands": use_aliases.get(mod, [mod]),
                "models": {
                    name: {
                        "variant": info["variant"],
                        "aliases": [a for a in info["aliases"] if a != "default"],
                        "gpu": info["gpu"],
                        "repo": info["repo"],
                        "note": info["note"],
                        "start_command": f"ai-colab start {mod} --model {name}",
                    }
                    for name, info in MODEL_CATALOG[mod].items()
                },
            }
            for mod in selected
        },
        "planned": dict(COMING) if norm == "all" else {},
    }


def do_models(args=None):
    modality = getattr(args, "modality", "all") if args is not None else "all"
    as_json = bool(getattr(args, "json", False)) if args is not None else False
    payload = _build_models_payload(modality)
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Modelos e modalidades:")
    for mod, section in payload["modalities"].items():
        cmds = "/".join(section["commands"])
        print(f"\n[{mod}] (comando pós-start: ai-colab {cmds})")
        for name, info in section["models"].items():
            alias_note = f" (aliases: {', '.join(info['aliases'])})" if info["aliases"] else ""
            print(f"  {name:14} {info['repo']} — GPU {info['gpu']}{alias_note}")
            print(f"    {info['note']}")
            print(f"    Iniciar: {info['start_command']}")
    if payload["planned"]:
        print("\n[planejadas]")
        for m, desc in payload["planned"].items():
            print(f"  {m:14} {desc}")


def do_voice(args):
    """Solicita WAV através do proxy autenticado; saída escrita atomicamente."""
    import base64
    state = state_read()
    if not state or state.get('mode') != 'voice':
        raise RuntimeError("Inicie a modalidade Voice: ai-colab start voice --model custom --gpu L4")
    selected = state.get('voice_variant', 'custom')
    if args.action not in VOICE_CAPABILITIES.get(selected, set()):
        needed = {'generate': 'custom', 'design': 'design', 'clone': 'clone'}[args.action]
        raise RuntimeError(f"Modelo em execução: {selected}. "
                           f"Para '{args.action}', execute ai-colab stop e depois "
                           f"ai-colab start voice --model {needed} (ou --voice-model {needed}) --gpu L4")
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


def _srt_clock(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or not 0 <= seconds < 999999:
        raise RuntimeError("Timestamp ASR inválido")
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    whole, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole:02},{millis:03}"


def _formats_from_options(input_path: Path, output: str | None, fmt: str):
    """Monta caminhos locais sem aceitar extensão enganosa."""
    if fmt == "all":
        base = Path(output).expanduser() if output else input_path.with_name(input_path.stem + '-transcricao')
        if base.suffix.lower() in ('.txt', '.srt', '.json'):
            base = base.with_suffix('')
        return {ext: base.with_suffix('.' + ext) for ext in ('txt', 'srt', 'json')}
    if output:
        path = Path(output).expanduser()
        if path.suffix.lower() != '.' + fmt:
            raise RuntimeError(f"A extensão do --output deve ser .{fmt}; ou use --format apropriado")
    else:
        path = input_path.with_name(input_path.stem + '-transcricao.' + fmt)
    return {fmt: path}


def _render_transcription(payload: dict, fmt: str) -> str:
    segments = payload.get('segments') or []
    if fmt == 'txt':
        if any(seg.get('speaker') for seg in segments):
            # Com diarização: uma linha por segmento, prefixada quando houver falante.
            return ''.join(f"[{seg['speaker']}] {seg['text'].strip()}\n" if seg.get('speaker')
                           else f"{seg['text'].strip()}\n" for seg in segments)
        return (payload.get('text') or '').strip() + '\n'
    if fmt == 'json':
        return json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
    if not segments:
        raise RuntimeError("Modelo não retornou timestamps reais. Não será gerado SRT fictício.")
    parts = []
    for number, seg in enumerate(segments, 1):
        start, end = float(seg['start']), float(seg['end'])
        if end < start:
            raise RuntimeError('Timestamp inválido retornado pela engine')
        text = seg['text'].strip()
        if seg.get('speaker'):
            text = f"[{seg['speaker']}] {text}"
        parts.append(f"{number}\n{_srt_clock(start)} --> {_srt_clock(end)}\n{text}\n")
    return '\n'.join(parts) + '\n'


def do_transcribe(args):
    """Envia áudio em streaming via SSH privado, devolve somente texto/legendas."""
    import http.client
    if getattr(args, "action", None) == "models":
        do_models(argparse.Namespace(modality="transcribe", json=getattr(args, "json", False)))
        return
    if not getattr(args, "file", None):
        raise RuntimeError("Informe --file/--input com o arquivo de áudio/vídeo, ou use: ai-colab transcribe models")
    # --json com --file equivale a --format json --stdout (modo stdout).
    stdout_mode = bool(getattr(args, "stdout", False))
    if getattr(args, "json", False):
        args.format = "json"
        stdout_mode = True
    if stdout_mode and args.format == "all":
        raise RuntimeError('Não é possível usar --stdout com --format all.')
    state = state_read()
    if not state or state.get('mode') != 'transcribe':
        raise RuntimeError('Inicie ASR primeiro: ai-colab start transcribe --model whisper')
    source = Path(args.file).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f'Arquivo de entrada não encontrado: {source}')
    extensions = {'.wav','.mp3','.m4a','.mp4','.ogg','.oga','.opus',
                  '.flac','.aac','.webm','.mov','.mpeg','.mpga','.wma'}
    if source.suffix.lower() not in extensions:
        raise RuntimeError('Formato não suportado. Use WAV/MP3/M4A/MP4/FLAC/OGG/OPUS/WEBM.')
    size = source.stat().st_size
    if not 0 < size <= 150 * 1024 * 1024:
        raise RuntimeError('Áudio deve ter entre 1 byte e 150 MiB.')
    if args.timeout < 1:
        raise RuntimeError('Timeout deve ser positivo')
    if stdout_mode:
        # Sem --file de destino: nada no disco, só stdout.
        formats = {}
    else:
        formats = _formats_from_options(source, args.output, args.format)
        for dest in formats.values():
            if not dest.parent.exists() or not dest.parent.is_dir():
                raise RuntimeError(f'Pasta de saída ausente: {dest.parent}')
            if dest.exists() and not args.overwrite:
                raise RuntimeError(f'Arquivo já existe: {dest}. Para substituir, use --overwrite.')
    conn = http.client.HTTPConnection('127.0.0.1', int(state['port']), timeout=args.timeout)
    begun = time.monotonic()
    try:
        conn.putrequest('POST', '/_colab_ai/transcribe')
        headers = {
            'X-Api-Key': state['token'], 'Content-Type': 'application/octet-stream',
            'Content-Length': str(size), 'X-Colab-Ai-Extension': source.suffix.lower(),
            'X-Colab-Ai-Language': args.language,
        }
        if getattr(args, "word_timestamps", False):
            headers['X-Colab-Ai-Word-Timestamps'] = 'true'
        if getattr(args, "diarize", False):
            headers['X-Colab-Ai-Diarize'] = 'true'
        for header, value in headers.items():
            conn.putheader(header, value)
        conn.endheaders()
        with source.open('rb') as src:
            while True:
                chunk = src.read(65536)
                if not chunk:
                    break
                conn.send(chunk)
        if not stdout_mode:
            print(f'Transcrevendo com {state["model"]} no Colab...', flush=True)
        response = conn.getresponse()
        raw = response.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise RuntimeError('Resposta ASR ultrapassou 16 MiB')
        if response.status != 200:
            raise RuntimeError(f'API ASR HTTP {response.status}: {raw.decode("utf-8", "replace")[:450]}')
        result = json.loads(raw)
        if not isinstance(result, dict) or not isinstance(result.get('text'), str):
            raise RuntimeError('Resposta inválida da engine ASR')
        if stdout_mode:
            # Sem mensagens de progresso/resumo: stdout 100% limpo.
            print(_render_transcription(result, args.format), end="")
        else:
            outputs = {k: _render_transcription(result, k) for k in formats}
            # Prepara todos os arquivos antes de substituir qualquer destino.
            prepared = {}
            try:
                for fmt, destination in formats.items():
                    with tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                                      prefix='.colab-asr-', suffix='.tmp',
                                                      dir=destination.parent, delete=False) as tmp:
                        tmp.write(outputs[fmt])
                        prepared[fmt] = Path(tmp.name)
                for fmt, destination in formats.items():
                    os.replace(prepared[fmt], destination)
                    print(f'Arquivo criado: {destination}')
            finally:
                for pth in prepared.values():
                    pth.unlink(missing_ok=True)
            print(f'Concluído em {time.monotonic() - begun:.1f}s | idioma={result.get("language", "?")} '
                  f'| segmentos={len(result.get("segments") or [])}')
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise RuntimeError(f'Falha na transcrição/comunicação: {type(exc).__name__}: {exc}') from exc
    finally:
        conn.close()


def _video_session() -> dict:
    state = state_read()
    if not state or state.get('mode') != 'video':
        raise RuntimeError('Inicie Video: ai-colab start video --model hunyuan15 (ou wan22-5b)')
    return state


def _video_request(state: dict, path: str, *, data: dict | None = None,
                   timeout: int = 30) -> dict:
    method = 'POST' if data is not None else 'GET'
    request = urllib.request.Request(
        f"http://127.0.0.1:{state.get('port', LOCAL_PORT)}{path}",
        data=json.dumps(data, ensure_ascii=False).encode('utf-8') if data is not None else None,
        headers={'Content-Type': 'application/json', 'X-Api-Key': state['token']},
        method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = resp.read(2 * 1024 * 1024 + 1)
            if len(payload) > 2 * 1024 * 1024:
                raise RuntimeError('Resposta do servidor Video acima de 2 MiB')
            return json.loads(payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode('utf-8', 'replace')
        raise RuntimeError(f'Video API HTTP {exc.code}: {detail}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Video API indisponível: {exc}. Use ai-colab reconnect/status') from exc


def _video_job_id(value: str):
    if not re.fullmatch('[a-f0-9]{32}', value):
        raise RuntimeError('ID de trabalho inválido (32 caracteres hexadecimais)')
    return value


def _video_output(args):
    dest = Path(args.output).expanduser().resolve()
    if dest.suffix.lower() != '.mp4':
        raise RuntimeError('O arquivo --output precisa terminar com .mp4')
    if not dest.parent.is_dir():
        raise RuntimeError(f'Pasta de saída não existe: {dest.parent}')
    if dest.exists() and not args.overwrite:
        raise RuntimeError(f'Arquivo existente: {dest}. Use --overwrite se quiser substituir.')
    return dest


def do_video_jobs(args):
    state = _video_session()
    result = _video_request(state, '/_colab_ai/video/jobs')
    print(json.dumps(result, ensure_ascii=False, indent=2))


def do_video_status(args):
    state = _video_session()
    job = _video_request(state, f'/_colab_ai/video/jobs/{_video_job_id(args.job_id)}')
    print(json.dumps(job, ensure_ascii=False, indent=2))


def do_video_cancel(args):
    state = _video_session()
    job = _video_request(state, f'/_colab_ai/video/jobs/{_video_job_id(args.job_id)}/cancel',
                         data={})
    print(f'Job {job["id"]}: {job["status"]}')


def do_video_fetch(args):
    state = _video_session()
    dest = _video_output(args)
    jid = _video_job_id(args.job_id)
    request = urllib.request.Request(
        f"http://127.0.0.1:{state.get('port', LOCAL_PORT)}/_colab_ai/video/jobs/{jid}/file",
        headers={'X-Api-Key': state['token']}, method='GET')
    path = None
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            if response.headers.get('Content-Type', '').split(';')[0] != 'video/mp4':
                raise RuntimeError('Servidor não retornou um MP4')
            with tempfile.NamedTemporaryFile(prefix='.colab-video-', suffix='.mp4.part',
                                             dir=dest.parent, delete=False) as temp:
                path = Path(temp.name)
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > 256 * 1024 * 1024:
                        raise RuntimeError('MP4 maior que 256 MiB; interrompendo transferência')
                    temp.write(chunk)
        if downloaded < 128:
            raise RuntimeError('MP4 muito curto')
        with path.open('rb') as fd:
            if b'ftyp' not in fd.read(16)[4:16]:
                raise RuntimeError('Arquivo retornado não possui assinatura MP4')
        os.replace(path, dest)
        print(f'MP4 salvo: {dest} | {downloaded / 1024**2:.1f} MiB')
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Erro ao baixar MP4 HTTP {exc.code}: '
                           + exc.read(800).decode('utf-8', 'replace')) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Conexão interrompida ao baixar MP4: {exc}') from exc
    finally:
        if path:
            path.unlink(missing_ok=True)


def video_dimensions(variant: str, aspect: str, size: str | None, has_image: bool) -> tuple[int, int]:
    """Define dimensões exatas suportadas por cada engine antes de enviar job."""
    defaults = {'hunyuan15': {'9:16': (480, 848), '16:9': (848, 480)},
                'wan22-5b': {'9:16': (704, 1280), '16:9': (1280, 704)}}
    if variant not in defaults:
        raise RuntimeError('Modelo de vídeo inválido')
    w, h = defaults[variant][aspect]
    if size:
        match = re.fullmatch(r'(\d{2,4})[xX](\d{2,4})', size)
        if not match:
            raise RuntimeError('--size deve ser LARGURAxALTURA (ex.: 480x848)')
        w, h = map(int, match.groups())
    if (w < 256 or h < 256 or w % 16 or h % 16):
        raise RuntimeError('Dimensões devem ser >=256 e múltiplas de 16')
    if (w < h) != (aspect == '9:16'):
        raise RuntimeError('--size conflita com --aspect; ajuste ambos')
    if variant == 'wan22-5b':
        if (w,h) != defaults[variant][aspect]:
            raise RuntimeError('Wan TI2V-5B suporta apenas 704x1280 (9:16) ou 1280x704 (16:9)')
    elif w*h > 848*480 or max(w,h) > 1024:
        raise RuntimeError('Hunyuan 480p: resolução acima do limite seguro deste checkpoint')
    return w,h


def video_steps(variant: str, has_image: bool, steps: int | None, quality: str) -> int:
    if variant == 'hunyuan15' and has_image:
        if steps is not None and steps != 12:
            raise RuntimeError('Hunyuan I2V step-distilled exige --steps 12')
        return 12
    if steps is not None:
        if not 1 <= steps <= 60:
            raise RuntimeError('--steps fora do intervalo 1-60')
        return steps
    return {'draft': 20, 'standard': 32, 'high': 50}[quality]


def do_video_generate(args):
    import base64
    state = _video_session()
    _video_output(args)
    if not isinstance(args.prompt, str) or not 1 <= len(args.prompt.strip()) <= 4000:
        raise RuntimeError('Prompt obrigatório de 1 a 4000 caracteres')
    if not 17 <= args.frames <= 121 or args.frames % 4 != 1:
        raise RuntimeError('--frames precisa ser 4n+1 entre 17 e 121 (ex.: 49, 73, 97, 121)')
    if not 0 <= args.seed < 2**32:
        raise RuntimeError('--seed precisa estar entre 0 e 4294967295')
    if not 60 <= args.timeout <= 43200:
        raise RuntimeError('--timeout entre 60 segundos e 12 horas')
    image_b64 = None
    if args.image:
        source = Path(args.image).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            raise RuntimeError('--image deve indicar PNG/JPG/WEBP existente')
        size = source.stat().st_size
        if size < 16 or size > 12 * 1024 * 1024:
            raise RuntimeError('Imagem deve ter no máximo 12 MiB')
        image_b64 = base64.b64encode(source.read_bytes()).decode('ascii')
    variant = state.get('video_variant')
    w, h = video_dimensions(variant, args.aspect, args.size, bool(image_b64))
    steps = video_steps(variant, bool(image_b64), args.steps, args.quality)
    print('Enviando trabalho ao Colab pelo túnel SSH privado...', flush=True)
    job = _video_request(state, '/_colab_ai/video/jobs', data={
        'prompt': args.prompt, 'frames': args.frames, 'steps': steps,
        'seed': args.seed, 'aspect': args.aspect, 'width': w, 'height': h,
        'offload': args.offload, 'image_fit': args.image_fit, 'image_b64': image_b64,
    }, timeout=100)
    jid = _video_job_id(job['id'])
    print(f'Trabalho enviado: {jid} | modelo={state["model"]} | {w}x{h} | quadros={args.frames} | passos={steps} | offload={args.offload}')
    print(f'Se a conexão cair: ai-colab video status {jid}; ai-colab video fetch {jid} --output "{args.output}"', flush=True)
    began = time.monotonic()
    last = None
    try:
        while time.monotonic() - began < args.timeout:
            time.sleep(10)
            job = _video_request(state, f'/_colab_ai/video/jobs/{jid}', timeout=30)
            status = job['status']
            tail = (job.get('log_tail') or '').strip().splitlines()
            info = (tail[-1][:150] if tail else 'preparando')
            g = job.get('gpu') or {}
            gpu_note = (f" | GPU {g.get('utilization_pct', 0):g}% | VRAM {g.get('memory_used_mib', 0)/1024:.1f} GiB"
                        if g else '')
            digest = (status, info, int(g.get('timestamp',0))//30)
            if digest != last:
                print(f'[{int(time.monotonic()-began)}s] {status}: {info}{gpu_note}', flush=True)
                last = digest
            if status == 'completed':
                do_video_fetch(args.__class__(**{**vars(args), 'job_id': jid}))
                print(f'Geração concluída em {time.monotonic()-began:.0f}s')
                return
            if status in ('failed', 'cancelled'):
                raise RuntimeError(f'Vídeo {status}: {job.get("error", "")}\n{job.get("log_tail", "")[-2500:]}')
    except KeyboardInterrupt:
        print(f'Interrompeu a espera, mas o trabalho pode continuar na VM. ID: {jid}')
        print(f'Para cancelar: ai-colab video cancel {jid}')
        return
    raise RuntimeError(f'Tempo de espera excedido. O job {jid} NÃO foi cancelado. '
                       f'Use ai-colab video status {jid} e video fetch quando concluir.')


HELP_EPILOG = """\
Comandos por categoria:
  Ciclo de vida:
    start <modalidade> --model <modelo>   Inicia GPU Colab e sobe o servidor
    wait                                  Aguarda modelo em GPU já ativa
    reconnect                             Restaura apenas o túnel SSH privado
    stop                                  Libera a GPU Colab e limpa estado local

  Uso das modalidades:
    code | claude                         Abre Claude Code conectado ao modelo Code
    voice | tts                           Sintetiza fala (generate, design, clone, models)
    transcribe | asr                      Transcreve áudio/vídeo em TXT/SRT/JSON
    video                                 Gera e gerencia vídeos MP4 (generate, jobs, fetch)
    models [MODALIDADE] [--json]          Lista modalidades, modelos e comandos start

  Monitoramento e diagnóstico:
    status | gpu status | stats | monitor Consulta estado, VRAM, temperatura e uso
    logs | diagnostics | test | ssh-check Logs remotos, saúde da API e chave SSH
    usage | hf                            Unidades do Colab e modo hf_xet

Modalidades, modelos (--model) e GPUs recomendadas:
  code              qwen (padrão, G4), qwen-next (80B, G4)
  voice (tts)       custom (padrão, L4), design (L4), clone (L4),
                    voxcpm2 (L4), chatterbox-br (L4), kokoro (T4)
  transcribe (asr)  whisper (padrão, T4), qwen (L4)
  video             hunyuan15 (padrão, L4), wan22-5b (G4)

Exemplos rápidos:
  ai-colab models
  ai-colab start code --model qwen --gpu G4 && ai-colab code
  ai-colab start tts --model voxcpm2 && ai-colab tts generate --text "Olá!" --output voz.wav
  ai-colab start asr --model whisper && ai-colab asr --file audio.mp3 --format srt
  ai-colab start video --model hunyuan15 && ai-colab video generate --prompt "Pôr do sol" --output out.mp4
  ai-colab stop

Use "ai-colab <comando> --help" para ver opções e exemplos de cada comando.
"""

START_EPILOG = """\
Modelos por modalidade (--model / -m):
  code              qwen (padrão, Qwen3.8-27B, G4), qwen-next (80B, G4)
  voice | tts       custom (padrão, L4), design (L4), clone (L4),
                    voxcpm2 (L4), chatterbox-br (L4), kokoro (T4)
  transcribe | asr  whisper (padrão, T4), qwen (L4)
  video             hunyuan15 (padrão, L4), wan22-5b (G4)

Exemplos:
  ai-colab start code --model qwen --gpu G4
  ai-colab start tts --model kokoro
  ai-colab start asr --model qwen --gpu L4
  ai-colab start video --model wan22-5b --gpu G4 --wait 1800
"""

MODELS_EPILOG = """\
Exemplos:
  ai-colab models
  ai-colab models video
  ai-colab models tts --json
"""

VOICE_EPILOG = """\
Modelos compatíveis (inicie antes com "ai-colab start voice --model <modelo>"):
  custom         Qwen3-TTS voz predefinida (generate) — GPU L4
  design         Qwen3-TTS voz por descrição (design) — GPU L4
  clone          Qwen3-TTS clonagem com áudio + transcrição (clone) — GPU L4
  voxcpm2        VoxCPM2 (generate, design, clone) — GPU L4
  chatterbox-br  Chatterbox PT-BR (generate, clone) — GPU L4
  kokoro         Kokoro 82M PT-BR (generate) — GPU T4

Exemplos:
  ai-colab start tts --model voxcpm2
  ai-colab tts generate --text "Olá, mundo!" --output voz.wav
  ai-colab voice design --text "Bem-vindo" --description "Voz grave calma" --output locucao.wav
  ai-colab voice models
"""

TRANSCRIBE_EPILOG = """\
Modelos compatíveis (inicie antes com "ai-colab start transcribe --model <modelo>"):
  whisper  faster-whisper-large-v3-turbo (T4, legendas por segmentos)
  qwen     Qwen3-ASR-1.7B + ForcedAligner (L4, timestamps por palavra)

Exemplos:
  ai-colab transcribe models
  ai-colab start asr --model whisper
  ai-colab asr --file ~/Downloads/audio.mp3 --format all --output ~/Downloads/resultado
"""

VIDEO_EPILOG = """\
Modelos compatíveis (inicie antes com "ai-colab start video --model <modelo>"):
  hunyuan15  HunyuanVideo-1.5 480p (L4; T2V 50 passos, I2V 12 passos)
  wan22-5b   Wan2.2 TI2V-5B 720p (G4; T2V e I2V)

Exemplos:
  ai-colab video models
  ai-colab start video --model hunyuan15 --gpu L4 --wait 1800
  ai-colab video generate --prompt "Nave futurista ao pôr do sol" --quality draft --output ~/Videos/out.mp4
  ai-colab video jobs
"""

CODE_EPILOG = """\
Pré-requisito: inicie a sessão com "ai-colab start code --model qwen --gpu G4".

Variáveis de ambiente suportadas:
  COLAB_AI_VISION_GUIDANCE=0  Desativa orientação global de leitura de imagens via Read
  COLAB_AI_SKIP_PERMISSIONS=0 Preserva prompts interativos de permissão do Claude Code

Exemplos:
  ai-colab code
  ai-colab claude -- -p "Explique o repositório"
"""


def parser_make():
    p = argparse.ArgumentParser(
        prog="ai-colab",
        description="Gerenciador multimodal de IA no Google Colab via CLI + túnel SSH privado.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"Colab AI CLI {VERSION}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMANDO")
    a = sub.add_parser(
        "start",
        help="Inicia uma modalidade (code, voice/tts, transcribe/asr, video)",
        epilog=START_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a.add_argument("mode", choices=["code", "code-next", "voice", "tts", "transcribe", "asr", "video", *COMING],
                   default="code", nargs="?",
                   help="Modalidade a iniciar (padrão: code). Aliases: tts=voice, asr=transcribe")
    a.add_argument("-m", "--model", dest="model", default=None, metavar="MODELO",
                   help="Modelo da modalidade (ex.: qwen, qwen-next, custom, voxcpm2, kokoro, whisper, hunyuan15, wan)")
    a.add_argument("--gpu", default=None, help="GPU Colab (ex.: T4, L4, G4). Se omitida, usa a recomendada do modelo")
    a.add_argument("--voice-model", choices=list(ACTIVE_VOICE_MODELS), default="custom",
                   help=argparse.SUPPRESS)
    a.add_argument("--video-model", choices=sorted(VIDEO_ACTIVE_MODELS), default="hunyuan15",
                   help=argparse.SUPPRESS)
    a.add_argument("--asr-model", choices=list(ASR_MODELS), default="whisper",
                   help=argparse.SUPPRESS)
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
    a.add_argument("--json", action="store_true", help="Saída em JSON para automação")
    a.set_defaults(func=do_status)
    sub.add_parser("reconnect", help="Restaura somente túnel SSH").set_defaults(func=do_reconnect)
    gpu = sub.add_parser('gpu', help='Diagnóstico da GPU pelo túnel atual (sem nova sessão SSH)')
    gpu_sub = gpu.add_subparsers(dest='gpu_action', required=True)
    gs = gpu_sub.add_parser('status', help='Consulta nome, VRAM, uso e processos da GPU')
    gs.add_argument('--json', action='store_true', help='Métricas em JSON')
    gs.add_argument('--deep', action='store_true', help='Inclui teste PyTorch no worker (mais demorado)')
    gs.add_argument('--watch', type=int, nargs='?', const=10, default=None, metavar='SEGUNDOS',
                    help='Atualiza a cada N segundos (mínimo 3; Ctrl+C encerra apenas o monitor)')
    gs.set_defaults(func=do_gpu_status)
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
    sub.add_parser("_watchdog").set_defaults(func=do_watchdog)
    a = sub.add_parser(
        "models",
        help="Lista modalidades, modelos (--model), GPUs e comandos start",
        epilog=MODELS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a.add_argument("modality", nargs="?", default="all",
                   choices=["all", "code", "voice", "tts", "transcribe", "asr", "video"],
                   help="Filtrar por modalidade (padrão: all)")
    a.add_argument("--json", action="store_true", help="Exibe catálogo estruturado em JSON")
    a.set_defaults(func=do_models)
    video = sub.add_parser(
        "video",
        help="Geração de vídeos MP4 na GPU Colab",
        epilog=VIDEO_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    video_sub = video.add_subparsers(dest="action", required=True)
    v = video_sub.add_parser("models", help="Lista modelos candidatos, GPU, modalidades e licenças")
    v.add_argument("--json", action="store_true", help="Exibe catálogo em JSON para futura automação")
    v.set_defaults(func=do_video)
    v = video_sub.add_parser("generate", help="Gera vídeo do prompt ou imagem e baixa MP4")
    v.add_argument("--prompt", required=True, help="Cena a gerar")
    v.add_argument("--image", help="Imagem PNG/JPG/WEBP local para animação (I2V)")
    v.add_argument("--output", required=True, help="MP4 de saída no Linux")
    v.add_argument("--frames", type=int, default=49, help="4n+1 quadros (padrão 49)")
    v.add_argument("--steps", type=int, default=None, help="Passos (Hunyuan T2V 50, I2V 12, Wan 50)")
    v.add_argument("--aspect", choices=["16:9", "9:16"], default="9:16",
                   help="Formato do vídeo (padrão vertical 9:16)")
    v.add_argument('--size', metavar='LARGURAxALTURA',
                   help='Hunyuan: custom até 480p; Wan: 704x1280 ou 1280x704')
    v.add_argument('--quality', choices=['draft','standard','high'], default='high',
                   help='Define 20, 32 ou 50 passos quando --steps não é informado')
    v.add_argument('--offload', choices=['auto','on','off'], default='auto',
                   help='offload automático por VRAM (forçar off exige >=80 GiB)')
    v.add_argument('--image-fit', choices=['cover','contain'], default='cover',
                   help='Imagem de referência: cobrir recortando ou conter com bordas')
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--timeout", type=int, default=10800, help="Tempo máximo total em segundos")
    v.add_argument("--overwrite", action="store_true")
    v.set_defaults(func=do_video_generate)
    v = video_sub.add_parser("jobs", help="Lista jobs da sessão (inclusive após perda de conexão)")
    v.set_defaults(func=do_video_jobs)
    v = video_sub.add_parser("status", help="Consulta progresso de geração por ID")
    v.add_argument("job_id", help="ID de geração")
    v.set_defaults(func=do_video_status)
    v = video_sub.add_parser("fetch", help="Baixa MP4 de uma geração finalizada")
    v.add_argument("job_id")
    v.add_argument("--output", required=True)
    v.add_argument("--overwrite", action="store_true")
    v.set_defaults(func=do_video_fetch)
    v = video_sub.add_parser("cancel", help="Interrompe trabalho de vídeo em andamento")
    v.add_argument("job_id")
    v.set_defaults(func=do_video_cancel)
    sub.add_parser("ssh-check", help="Valida chave SSH exclusiva, sem criar GPU").set_defaults(func=do_ssh_check)
    sub.add_parser("diagnostics", help="Logs locais + estado remoto").set_defaults(func=do_diagnostics)
    a = sub.add_parser("logs", help="Últimas linhas do servidor do modelo")
    a.add_argument("--lines", type=int, default=80)
    a.add_argument("--source", choices=["vllm", "bootstrap", "omni", "video"], default="vllm",
                   help="vllm/voice, bootstrap (instalação), video ou omni")
    a.set_defaults(func=do_logs)
    a = sub.add_parser("test", help="Executa geração real pela API")
    a.add_argument("--timeout", type=int, default=180)
    a.add_argument("--file", help="Áudio de teste para o modo transcribe (sem arquivo: somente health)")
    a.set_defaults(func=do_test)
    a = sub.add_parser(
        "transcribe",
        aliases=["asr"],
        help="Transcreve áudio no Colab e salva TXT/SRT/JSON no Linux",
        epilog=TRANSCRIBE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a.add_argument("action", nargs="?", choices=["models"],
                   help="Opcional: 'models' lista os modelos ASR disponíveis sem iniciar GPU")
    a.add_argument("--file", "--input", dest="file", required=False,
                   help="Arquivo de áudio/vídeo local (WAV, MP3, MP4, M4A, FLAC, OGG, ...)")
    a.add_argument("--output", help="Destino (inferido do formato; padrão ao lado do áudio)")
    a.add_argument("--format", choices=["txt", "srt", "json", "all"], default="txt",
                   help="Formato de saída; all produz três arquivos")
    a.add_argument("--language", choices=["pt", "auto"], default="pt",
                   help="Forçar português ou detectar automaticamente")
    a.add_argument("--timeout", type=int, default=1800, help="Timeout em segundos para ASR")
    a.add_argument("--overwrite", action="store_true", help="Permitir substituir arquivos existentes")
    a.add_argument("--json", action="store_true", help="Saída em JSON quando usado com 'models'")
    a.add_argument("--stdout", action="store_true",
                   help="Imprime a transcrição direto no stdout, sem gravar arquivo (ignora --output)")
    a.add_argument("--word-timestamps", action="store_true",
                   help="Inclui timestamps por palavra no JSON da transcrição")
    a.add_argument("--diarize", action="store_true",
                   help="Identifica os falantes e marca cada segmento como [Falante 1], [Falante 2], ...")
    a.set_defaults(func=do_transcribe)
    voice = sub.add_parser(
        "voice",
        aliases=["tts"],
        help="Gera WAV usando a engine Voice selecionada",
        epilog=VOICE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    voice_sub = voice.add_subparsers(dest="action", required=True)
    vm = voice_sub.add_parser("models", help="Lista modelos de voz/TTS disponíveis sem iniciar GPU")
    vm.add_argument("--json", action="store_true", help="Exibe catálogo de voz em JSON")
    vm.set_defaults(func=do_models, modality="voice")
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
            v.add_argument("--style", default="", help="VoxCPM/Qwen: descrição ou orientação expressiva")
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
    a = sub.add_parser(
        "claude",
        aliases=["code"],
        help="Abre Claude Code conectado ao modelo Code ativo",
        epilog=CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a.add_argument("extra", nargs=argparse.REMAINDER)
    a.set_defaults(func=do_claude)
    return p


def main(argv=None):
    parser = parser_make()
    effective_argv = sys.argv[1:] if argv is None else argv
    if not effective_argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
