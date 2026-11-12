"""Colab VM: prepara worker de vídeo isolado, sem baixar checkpoints até gerar."""
from pathlib import Path
from urllib.request import urlopen
import json
import os
import re
import shutil
import subprocess
import sys
import time

CONFIG = json.loads(Path('/content/colab-ai-config.json').read_text())
VARIANT = CONFIG['video_variant']
assert VARIANT in {'hunyuan15', 'wan22-5b'}, 'Perfil de vídeo inválido'
PID = Path('/content/colab-ai-video.pid')
LOG = Path('/content/colab-ai-video.log')
VENV = Path('/content/colab-ai-video-venv')
PY = VENV / 'bin/python'

info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                '--format=csv,noheader'], text=True).splitlines()[0]
name, memory = info.split(',', 1)
match = re.search(r'[\d.]+', memory)
if not match:
    raise RuntimeError('Não foi possível identificar VRAM')
gib = float(match.group()) / 1024
print(f'GPU Video: {name.strip()} | {gib:.1f} GiB', flush=True)
try:
    import torch
    print(f'PyTorch do sistema: {torch.__version__} | CUDA compilado: {torch.version.cuda} | '
          f'CUDA disponível: {torch.cuda.is_available()}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('nvidia-smi encontrou GPU, mas PyTorch CUDA indisponível; verificar ambiente Colab')
except ImportError as err:
    raise RuntimeError('O PyTorch CUDA do Colab é obrigatório para o worker de vídeo') from err
if gib < CONFIG['min_gib']:
    raise RuntimeError(f'GPU insuficiente: mínimo {CONFIG["min_gib"]} GiB para {VARIANT}')

os.environ['HF_XET_HIGH_PERFORMANCE'] = '1' if CONFIG.get('hf_mode') == 'fast' else '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
if not shutil.which('ffmpeg'):
    subprocess.check_call(['apt-get', 'update', '-qq'], stdout=subprocess.DEVNULL)
    subprocess.check_call(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], stdout=subprocess.DEVNULL)

uv = shutil.which('uv')
if not uv:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    uv = shutil.which('uv') or '/usr/local/bin/uv'
if not PY.exists():
    # Aproveita o PyTorch com CUDA disponível no Colab; evita baixar outro torch.
    subprocess.check_call([uv, 'venv', '--system-site-packages', '--python', sys.executable, str(VENV)])
marker = VENV / ('.video-ready-' + VARIANT + '-v030')
if not marker.exists():
    packages = ['huggingface_hub>=0.34', 'hf_xet', 'pillow>=10', 'imageio[ffmpeg]', 'accelerate']
    if VARIANT == 'hunyuan15':
        packages += ['diffusers>=0.36.0', 'transformers>=4.56,<5', 'sentencepiece', 'protobuf']
    else:
        # O requirements.txt oficial do Wan exige flash_attn (build de CUDA)
        # e transformers<=4.51.3; instalar esses pacotes em venv dedicado.
        packages += ['transformers>=4.49.0,<=4.51.3', 'diffusers>=0.31.0',
                     'opencv-python-headless>=4.9', 'easydict', 'ftfy', 'dashscope',
                     'imageio-ffmpeg', 'sentencepiece', 'numpy>=1.23.5,<2']
    print('Instalando dependências de vídeo isoladas:', VARIANT, flush=True)
    subprocess.check_call([uv, 'pip', 'install', '--python', str(PY), *packages])
    # O Wan usa kernels flash-attn via import em algumas variantes. A instalação
    # pode ser custosa; tentar somente se necessário durante geração.
    marker.write_text('ok\n')

def alive():
    try:
        pid = int(PID.read_text())
        os.kill(pid, 0)
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
        stat = Path(f'/proc/{pid}/stat')
        return b'colab-ai-video-api.py' in raw and not (stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'))
    except (OSError, ValueError):
        return False

if alive():
    print('Worker Video já iniciado; preservando jobs.', flush=True)
    sys.exit(0)
try:
    with urlopen('http://127.0.0.1:8000/health', timeout=2) as r:
        if r.status == 200:
            raise RuntimeError('Porta 8000 já possui outro modelo; use stop antes de trocar')
except OSError:
    pass
with LOG.open('ab') as log:
    p = subprocess.Popen([str(PY), '-u', '/content/colab-ai-video-api.py'],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True, env=os.environ.copy())
PID.write_text(str(p.pid))
print(f'Worker Video iniciado: PID={p.pid}, {VARIANT} (pesos baixados na primeira geração)', flush=True)
time.sleep(3)
if p.poll() is not None:
    raise RuntimeError('Worker Video encerrou: \n' + LOG.read_text(errors='replace')[-3500:])
