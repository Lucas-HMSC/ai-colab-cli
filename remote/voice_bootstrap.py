"""Bootstrap de TTS isolado por engine.

O kernel do Colab só inicia este processo; downloads/instalações não bloqueiam colab exec.
Cada perfil usa seu próprio ambiente e Python compatível para evitar conflitos
entre Torch, torchvision, Transformers e pacotes de geração de voz.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

conf = json.loads(Path('/content/colab-ai-config.json').read_text())
model = conf['model']
variant = conf['voice_variant']
logfile = Path('/content/colab-ai-voice.log')
pidfile = Path('/content/colab-ai-voice.pid')
aux_log = Path('/content/colab-ai-voice-omni.log')
aux_pidfile = Path('/content/colab-ai-voice-omni.pid')

info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                '--format=csv,noheader'], text=True).strip().splitlines()[0]
gpu, mem = info.split(',', 1)
match = re.search(r'([\d.]+)\s*MiB', mem)
if not match:
    raise RuntimeError('GPU não identificada: ' + info)
gib = float(match.group(1)) / 1024
print(f'GPU Voice: {gpu.strip()} | {gib:.1f} GiB', flush=True)
if gib < conf['min_gib']:
    raise RuntimeError(f'GPU menor que {conf["min_gib"]} GiB para {variant}. Use outra GPU.')

os.environ['HF_XET_HIGH_PERFORMANCE'] = '1' if conf.get('hf_mode') == 'fast' else '0'
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# O Colab CLI usa Python 3.13. Preserve o Qwen3-TTS que já funcionava no
# runtime original. Os backends alternativos têm Python e pacotes isolados.
qwen_engine = variant in ('custom', 'design', 'clone')
python_version = (f'{sys.version_info.major}.{sys.version_info.minor}' if qwen_engine
                  else '3.12' if variant == 'higgs3' else '3.11')
venv = Path('/content/colab-ai-voice-venvs') / (variant + '-py' + python_version.replace('.', ''))
venv.mkdir(mode=0o700, parents=True, exist_ok=True)
venv_python = Path(sys.executable) if qwen_engine else venv / 'bin/python'

if not qwen_engine:
    uv = shutil.which('uv')
    if not uv:
        print('Instalando gerenciador uv no Python do sistema...', flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
        uv = shutil.which('uv') or '/usr/local/bin/uv'
    print(f'Preparando Python {python_version} isolado para {variant}...', flush=True)
    if not venv_python.exists():
        subprocess.check_call([uv, 'venv', '--python', python_version, '--seed', str(venv)])
else:
    print('Qwen3-TTS: preservando Python do Colab, conhecido como funcional.', flush=True)

marker = venv / '.colab-ai-ready-v021'
if not marker.exists():
    common = ['huggingface_hub>=0.34', 'hf_xet', 'numpy', 'soundfile']
    if qwen_engine:
        packages = ['qwen-tts']
    elif variant == 'voxcpm2':
        packages = ['voxcpm==2.0.3', 'setuptools']
    elif variant == 'chatterbox-br':
        # Compatibilidade oficial do Chatterbox 0.1.7 com PyTorch 2.6.0.
        packages = ['chatterbox-tts==0.1.7', 'torchvision==0.21.0']
    elif variant == 'kokoro':
        packages = ['kokoro==0.9.4', 'misaki[en]==0.9.4']
    elif variant == 'higgs3':
        packages = []
    else:
        raise RuntimeError('Engine Voice desconhecida: ' + variant)
    if variant == 'higgs3':
        print('Instalando vLLM e vLLM-Omni 0.32.0 (pode exigir vários GB)...', flush=True)
        subprocess.check_call([uv, 'pip', 'install', '--python', str(venv_python),
                               '--torch-backend=cu130', 'vllm==0.32.0'])
        subprocess.check_call([uv, 'pip', 'install', '--python', str(venv_python),
                               'vllm-omni==0.32.0'] + common)
    elif qwen_engine:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U'] + common + packages)
    else:
        print('Instalando engine Voice compatível:', variant, packages, flush=True)
        subprocess.check_call([uv, 'pip', 'install', '--python', str(venv_python)] + common + packages)
    marker.write_text('ready\n')

if variant == 'kokoro':
    print('Verificando eSpeak NG para pronúncia PT-BR...', flush=True)
    if not shutil.which('espeak-ng'):
        subprocess.check_call(['apt-get', 'update', '-qq'], stdout=subprocess.DEVNULL)
        subprocess.check_call(['apt-get', 'install', '-y', '-qq', 'espeak-ng'], stdout=subprocess.DEVNULL)

# Baixar pesos pelo Python da engine (não pelo Python 3.13 do kernel).
# Pacotes incompletos não recebem marcador de conclusão.
downloaded_marker = venv / '.model-downloaded'
if not downloaded_marker.exists():
    if variant == 'chatterbox-br':
        # ChatterboxMultilingualTTS baixa sozinho o checkpoint base V3 no load.
        # Só é preciso pré-baixar o ajuste regional, sem duplicar os demais arquivos.
        command = '''from huggingface_hub import snapshot_download
snapshot_download(repo_id="ResembleAI/Chatterbox-Multilingual-pt-br", allow_patterns=["t3_pt_br.safetensors"])
'''
    elif variant == 'kokoro':
        command = '''from huggingface_hub import snapshot_download
snapshot_download(repo_id="hexgrad/Kokoro-82M", allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/pf_dora.pt", "voices/pm_alex.pt", "voices/pm_santa.pt"])
'''
    else:
        command = 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=' + repr(model) + ')'
    print('Baixando pesos via hf_xet:', model, flush=True)
    subprocess.check_call([str(venv_python), '-c', command])
    downloaded_marker.write_text(model + '\n')


def process_alive(path, marker):
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        stat = Path(f'/proc/{pid}/stat')
        return marker in cmd and not (stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'))
    except (OSError, ValueError, ProcessLookupError):
        return False

try:
    with urlopen('http://127.0.0.1:8000/health', timeout=2) as resp:
        ready = resp.status == 200
except OSError:
    ready = False
if ready:
    with urlopen('http://127.0.0.1:8000/v1/models', timeout=5) as resp:
        ids = [x['id'] for x in json.load(resp).get('data', [])]
    if model not in ids:
        raise RuntimeError('Outro modelo já utiliza porta 8000; execute stop antes da troca')
    print('Reutilizando servidor Voice pronto.', flush=True)
    sys.exit(0)

if process_alive(pidfile, b'colab-ai-voice-api.py'):
    print('Worker Voice já ativo/inicializando. Preservando processo.', flush=True)
    sys.exit(0)

if variant == 'higgs3':
    if not process_alive(aux_pidfile, b'vllm-omni'):
        aux_pidfile.unlink(missing_ok=True)
        omni = venv / 'bin/vllm-omni'
        if not omni.exists():
            raise RuntimeError('Executável vllm-omni ausente após instalação; consulte bootstrap')
        omni_cmd = [str(omni), 'serve', model, '--host', '127.0.0.1',
                    '--port', '8002', '--trust-remote-code', '--omni']
        with aux_log.open('ab') as log:
            backend = subprocess.Popen(omni_cmd, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, close_fds=True,
                                       start_new_session=True, env=os.environ.copy())
        aux_pidfile.write_text(str(backend.pid))
        print('Higgs via vLLM-Omni iniciado na porta privada 8002. PID:', backend.pid, flush=True)
        time.sleep(3)
        if backend.poll() is not None:
            print('\n'.join(aux_log.read_text(errors='replace').splitlines()[-25:]), flush=True)
            raise RuntimeError('vLLM-Omni encerrou na inicialização')

pidfile.unlink(missing_ok=True)
with logfile.open('ab') as log:
    child = subprocess.Popen([str(venv_python), '-u', '/content/colab-ai-voice-api.py'],
                             stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, close_fds=True,
                             start_new_session=True, env=os.environ.copy())
pidfile.write_text(str(child.pid))
print('Worker Voice iniciado:', child.pid, '| Python:', python_version, flush=True)
time.sleep(3)
if child.poll() is not None:
    print('\n'.join(logfile.read_text(errors='replace').splitlines()[-35:]), flush=True)
    raise RuntimeError(f'Worker Voice terminou com código {child.returncode}')
