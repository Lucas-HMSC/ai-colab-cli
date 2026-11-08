"""Instala somente a engine Voice selecionada e inicia serviço de inferência.
Executado em processo separado; nunca mantém o kernel do Colab CLI bloqueado.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen

conf = json.loads(Path('/content/colab-ai-config.json').read_text())
model = conf['model']
variant = conf['voice_variant']
logfile = Path('/content/colab-ai-voice.log')
pidfile = Path('/content/colab-ai-voice.pid')

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

# Engine isolada por VM/sessão. Não misturar Qwen-Code vLLM com bibliotecas TTS.
base = [sys.executable, '-m', 'pip', 'install', '-q', '-U']
subprocess.check_call(base + ['huggingface_hub>=0.34', 'hf_xet', 'soundfile'])
if variant in ('custom', 'design', 'clone'):
    packages = ['qwen-tts']
elif variant == 'voxcpm2':
    packages = ['voxcpm']
elif variant == 'chatterbox-br':
    packages = ['chatterbox-tts', 'safetensors']
elif variant == 'kokoro':
    packages = ['kokoro>=0.9', 'misaki[en]']
    # G2P de português brasileiro usa eSpeak.
    subprocess.check_call(['apt-get', 'update', '-qq'], stdout=subprocess.DEVNULL)
    subprocess.check_call(['apt-get', 'install', '-y', '-qq', 'espeak-ng'],
                          stdout=subprocess.DEVNULL)
elif variant == 'higgs3':
    packages = ['transformers>=5.0', 'accelerate', 'sentencepiece']
else:
    raise RuntimeError('Engine Voice desconhecida: ' + variant)
print('Instalando dependências específicas de', variant, packages, flush=True)
subprocess.check_call(base + packages)

from huggingface_hub import snapshot_download

if variant == 'chatterbox-br':
    # Para Chatterbox, a implementação oficial busca componentes compartilhados
    # de ResembleAI/chatterbox. A versão PT-BR contém somente checkpoints locais.
    print('Baixando checkpoint PT-BR: t3_pt_br.safetensors...', flush=True)
    snapshot_download(repo_id=model,
                      allow_patterns=['t3_pt_br.safetensors'])
    print('Baixando componentes compartilhados do Chatterbox V3...', flush=True)
    snapshot_download(repo_id='ResembleAI/chatterbox',
                      allow_patterns=['ve.pt', 't3_mtl23ls_v3.safetensors',
                                      's3gen.pt', 'grapheme_mtl_merged_expanded_v1.json',
                                      'conds.pt', 'Cangjie5_TC.json'])
elif variant == 'kokoro':
    print('Baixando Kokoro e vozes PT-BR...', flush=True)
    snapshot_download(repo_id=model,
                      allow_patterns=['config.json', 'kokoro-v1_0.pth',
                                      'voices/pf_dora.pt', 'voices/pm_alex.pt', 'voices/pm_santa.pt'])
else:
    print('Download via hf_xet:', model, flush=True)
    snapshot_download(repo_id=model)

try:
    with urlopen('http://127.0.0.1:8000/health', timeout=2) as resp:
        ready = resp.status == 200
except OSError:
    ready = False
if ready:
    with urlopen('http://127.0.0.1:8000/v1/models', timeout=5) as resp:
        models = [x['id'] for x in json.load(resp).get('data', [])]
    if model not in models:
        raise RuntimeError('Outro modelo já utiliza porta 8000; execute stop antes da troca')
    print('Reutilizando servidor Voice pronto.', flush=True)
else:
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'colab-ai-voice-api.py' in cmd:
                print('Servidor Voice já inicializando, preservado', flush=True)
                sys.exit(0)
        except (OSError, ValueError, ProcessLookupError):
            pass
    pidfile.unlink(missing_ok=True)
    with logfile.open('ab') as log:
        child = subprocess.Popen([sys.executable, '-u', '/content/colab-ai-voice-api.py'],
                                 stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, close_fds=True,
                                 start_new_session=True, env=os.environ.copy())
    pidfile.write_text(str(child.pid))
    print('Voice worker iniciado:', child.pid, flush=True)
    time.sleep(3)
    if child.poll() is not None:
        print('\n'.join(logfile.read_text(errors='replace').splitlines()[-30:]))
        raise RuntimeError(f'Worker Voice terminou com código {child.returncode}')
