"""Prepara ASR no Colab em ambiente isolado e inicia o worker HTTP privado."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

CONF = json.loads(Path('/content/colab-ai-config.json').read_text())
VARIANT = CONF['asr_variant']
MODEL = CONF['model']
LOG = Path('/content/colab-ai-asr.log')
PID = Path('/content/colab-ai-asr.pid')
ENV = Path('/content/colab-ai-asr-venvs') / (VARIANT + '-py312')
info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                '--format=csv,noheader'], text=True).splitlines()[0]
gpu, memory = info.split(',', 1)
match = re.search(r'[\d.]+', memory)
gib = float(match.group()) / 1024
print(f'GPU ASR: {gpu.strip()} | {gib:.1f} GiB', flush=True)
if gib < CONF['min_gib']:
    raise RuntimeError(f'GPU insuficiente para ASR: mínimo {CONF["min_gib"]} GiB')

os.environ['HF_XET_HIGH_PERFORMANCE'] = '1' if CONF.get('hf_mode') == 'fast' else '0'
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
if CONF.get('hf_token'):
    os.environ['HF_TOKEN'] = CONF['hf_token']

uv = shutil.which('uv')
if not uv:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    uv = shutil.which('uv') or '/usr/local/bin/uv'
python_bin = ENV / 'bin/python'
if not python_bin.exists():
    print('Preparando Python 3.12 isolado para ASR...', flush=True)
    subprocess.check_call([uv, 'venv', '--python', '3.12', '--seed', str(ENV)])
marker = ENV / ('.colab-asr-v028-diarize-ready' if CONF.get('hf_token') else '.colab-asr-v028-ready')
if not marker.exists():
    print('Instalando dependências ASR:', VARIANT, flush=True)
    if VARIANT == 'whisper':
        packages = ['faster-whisper>=1.1,<2', 'av>=11,<19', 'huggingface_hub>=0.34', 'hf_xet',
                    'nvidia-cublas-cu12', 'nvidia-cudnn-cu12>=9,<10']
    elif VARIANT == 'qwen':
        packages = ['qwen-asr', 'huggingface_hub>=0.34', 'hf_xet', 'soundfile', 'librosa']
    else:
        raise RuntimeError('Perfil ASR desconhecido')
    if CONF.get('hf_token'):
        packages.append('pyannote.audio>=3.1')
    subprocess.check_call([uv, 'pip', 'install', '--python', str(python_bin), *packages])
    marker.write_text('ready\n')
# As bibliotecas CUDA usadas pelo CTranslate2 precisam estar no caminho do
# carregador dinâmico ANTES de iniciar o Python do worker. Somente ter nvidia-smi
# funcional nao comprova a disponibilidade de cuBLAS/cuDNN na primeira inferência.
if VARIANT == 'whisper':
    libsite = subprocess.check_output(
        [str(python_bin), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'],
        text=True).strip()
    from pathlib import Path as _Path
    libdirs = [_Path(libsite) / 'nvidia' / 'cublas' / 'lib',
               _Path(libsite) / 'nvidia' / 'cudnn' / 'lib']
    for directory in libdirs:
        if not directory.is_dir():
            raise RuntimeError(f'Biblioteca CUDA ausente: {directory.name}; verifique dependências ASR')
    previous = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ':'.join([*(str(x) for x in libdirs), *([previous] if previous else [])])
    print('CTranslate2: cuBLAS CUDA 12 + cuDNN 9 disponíveis no ambiente do worker', flush=True)

if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
    print('Instalando FFmpeg para normalização de áudio...', flush=True)
    subprocess.check_call(['apt-get', 'update', '-qq'], stdout=subprocess.DEVNULL)
    subprocess.check_call(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], stdout=subprocess.DEVNULL)

def alive():
    try:
        pid = int(PID.read_text())
        os.kill(pid, 0)
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        stat = Path(f'/proc/{pid}/stat')
        return b'colab-ai-transcribe-api.py' in cmd and not (stat.exists() and stat.read_text().split(') ',1)[1].startswith('Z'))
    except (ValueError, ProcessLookupError, OSError):
        return False

if alive():
    print('Worker ASR já ativo. Não foi iniciado novamente.', flush=True)
    sys.exit(0)
try:
    with urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
        if response.status == 200:
            raise RuntimeError('Porta 8000 em uso por outro modelo; execute stop antes de trocar')
except OSError:
    pass

with LOG.open('ab') as log:
    process = subprocess.Popen([str(python_bin), '-u', '/content/colab-ai-transcribe-api.py'],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               close_fds=True, start_new_session=True, env=os.environ.copy())
PID.write_text(str(process.pid))
print(f'Worker ASR iniciado: PID {process.pid}, modelo={MODEL}', flush=True)
time.sleep(3)
if process.poll() is not None:
    print('\n'.join(LOG.read_text(errors='replace').splitlines()[-45:]), flush=True)
    raise RuntimeError('Worker ASR encerrou durante inicialização')
