"""API privada de jobs de vídeo. Apenas stdlib; modelos rodam em subprocessos."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import base64
import binascii
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time

# Evita imports acidentais de JAX/Flax no venv com --system-site-packages.
os.environ['USE_FLAX'] = '0'
os.environ['USE_TF'] = '0'

CONF = json.loads(Path(os.environ.get('COLAB_AI_VIDEO_CONFIG', '/content/colab-ai-config.json')).read_text())
if CONF.get('hf_token'):
    os.environ['HF_TOKEN'] = CONF['hf_token']
VARIANT = CONF['video_variant']
MODEL = CONF['model']
ROOT = Path(os.environ.get('COLAB_AI_VIDEO_ROOT', '/content/colab-ai-videos'))
ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
JOBS = {}
GUARD = threading.RLock()
MAX_IMAGE = 12 * 1024 * 1024
MAX_MP4 = 256 * 1024 * 1024
MAX_BODY = 17 * 1024 * 1024
JOB_PATH = re.compile(r'^/_colab_ai/video/jobs/([a-f0-9]{32})(/file|/cancel)?$')


def validate_size(variant, aspect, width, height):
    sizes = {'hunyuan15': {'9:16': (480,848), '16:9': (848,480)},
             'wan22-5b': {'9:16': (704,1280), '16:9': (1280,704)}}
    if aspect not in ('9:16', '16:9'):
        raise ValueError('Aspecto inválido')
    default_w, default_h = sizes[variant][aspect]
    w = default_w if width is None else width
    h = default_h if height is None else height
    if type(w) is not int or type(h) is not int or min(w,h) < 256 or w % 16 or h % 16:
        raise ValueError('Resolução inválida: largura e altura >=256, múltiplas de 16')
    if (w < h) != (aspect == '9:16'):
        raise ValueError('Dimensões incompatíveis com formato do vídeo')
    if variant == 'wan22-5b' and (w,h) != (default_w,default_h):
        raise ValueError('Wan TI2V-5B: somente 704x1280 ou 1280x704')
    if variant == 'hunyuan15' and (w*h > 480*848 or max(w,h)>1024):
        raise ValueError('Hunyuan 480p: acima dos limites do checkpoint')
    return w,h


def gpu_sample():
    """Snapshot leve: sem PyTorch nem novo SSH; pode falhar durante download."""
    try:
        output = subprocess.run(['nvidia-smi',
            '--query-gpu=name,utilization.gpu,memory.total,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if output.returncode != 0 or not output.stdout.strip():
            return None
        a = [x.strip() for x in output.stdout.splitlines()[0].split(',')]
        return {'name': a[0], 'utilization_pct': float(a[1]),
                'memory_total_mib': float(a[2]), 'memory_used_mib': float(a[3]),
                'timestamp': time.time()}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def monitor_job_gpu(job, stop_event):
    while not stop_event.is_set():
        sample = gpu_sample()
        if sample:
            with GUARD:
                job['gpu'] = sample
                job['gpu_samples'] = job.get('gpu_samples', 0) + 1
                job['peak_vram_mib'] = max(job.get('peak_vram_mib', 0), sample['memory_used_mib'])
            try:
                with job['log'].open('ab') as log:
                    log.write((f"[GPU] {sample['utilization_pct']:.0f}% | "
                               f"VRAM {sample['memory_used_mib']/1024:.1f}/"
                               f"{sample['memory_total_mib']/1024:.1f} GiB\n").encode('utf-8'))
            except OSError:
                pass
        stop_event.wait(30)


def check_request(request):
    if not isinstance(request, dict):
        raise ValueError('Metadados de vídeo inválidos')
    prompt = request.get('prompt')
    frames = request.get('frames', 49)
    seed = request.get('seed', 42)
    image_b64 = request.get('image_b64')
    aspect = request.get('aspect', '9:16')
    width = request.get('width')
    height = request.get('height')
    offload = request.get('offload', 'auto')
    image_fit = request.get('image_fit', 'cover')
    steps = request.get('steps', 12 if (VARIANT == 'hunyuan15' and image_b64) else 50)
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
        raise ValueError('Prompt obrigatório (1-4000 caracteres)')
    if type(frames) is not int or not 17 <= frames <= 121 or frames % 4 != 1:
        raise ValueError('Frames precisa estar em 4n+1, de 17 a 121')
    if type(steps) is not int or not 1 <= steps <= 60:
        raise ValueError('Passos fora de 1-60')
    if VARIANT == 'hunyuan15' and image_b64 and steps != 12:
        raise ValueError('Hunyuan I2V step-distilled exige 12 passos')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed inválido')
    width,height = validate_size(VARIANT, aspect, width, height)
    if offload not in ('auto','on','off'):
        raise ValueError('Offload inválido')
    if image_fit not in ('cover','contain'):
        raise ValueError('image_fit inválido')
    image = None
    extension = None
    if image_b64 is not None:
        if not isinstance(image_b64, str) or len(image_b64) > (MAX_IMAGE * 4 // 3 + 8):
            raise ValueError('Imagem excede 12 MiB')
        try:
            image = base64.b64decode(image_b64, validate=True)
        except (ValueError, binascii.Error) as err:
            raise ValueError('Base64 inválido') from err
        if len(image) > MAX_IMAGE or not image:
            raise ValueError('Imagem vazia ou maior que 12 MiB')
        if image.startswith(b'\x89PNG\r\n\x1a\n'):
            extension = '.png'
        elif image.startswith(b'\xff\xd8\xff'):
            extension = '.jpg'
        elif image.startswith(b'RIFF') and image[8:12] == b'WEBP':
            extension = '.webp'
        else:
            raise ValueError('Imagem deve ser PNG/JPG/WEBP válida')
    return {'prompt': prompt.strip(), 'frames': frames, 'steps': steps,
            'aspect': aspect, 'width': width, 'height': height, 'offload': offload,
            'image_fit': image_fit, 'seed': seed, 'image': image, 'extension': extension}


def _public_status(job):
    out = {k: job[k] for k in ('id', 'status', 'created_at', 'updated_at', 'model')}
    out['bytes'] = job['output'].stat().st_size if job['output'].is_file() else 0
    path = job['log']
    try:
        with path.open('rb') as log:
            size = log.seek(0, 2)
            log.seek(max(0, size - 6000))
            out['log_tail'] = log.read().decode('utf-8', 'replace')[-3500:]
    except OSError:
        out['log_tail'] = ''
    out['generation'] = job.get('generation', {})
    if job.get('gpu'):
        out['gpu'] = job['gpu']
        out['gpu_samples'] = job.get('gpu_samples', 0)
        out['peak_vram_mib'] = job.get('peak_vram_mib', 0)
    out['elapsed_seconds'] = round(time.time() - job['created_at']) if job['status'] in ('running','queued') else job.get('elapsed_seconds')
    if job.get('error'):
        out['error'] = job['error']
    return out


def valid_mp4(path):
    with path.open("rb") as f:
        return b"ftyp" in f.read(16)[4:16]


def _run(job, params):
    image_file = job.get('image')
    cmd = [sys.executable, '-u', '/content/colab-ai-video-generate.py',
           '--variant', VARIANT, '--prompt', params['prompt'],
           '--frames', str(params['frames']), '--steps', str(params['steps']),
           '--aspect', params['aspect'], '--width', str(params['width']),
           '--height', str(params['height']), '--offload', params['offload'],
           '--image-fit', params['image_fit'], '--seed', str(params['seed']),
           '--output', str(job['output'])]
    if image_file:
        cmd += ['--image', str(image_file)]
    stop_metrics = threading.Event()
    monitor = threading.Thread(target=monitor_job_gpu, args=(job, stop_metrics), daemon=True)
    try:
        with job['log'].open('ab') as log:
            with GUARD:
                if job['status'] == 'cancelled':
                    return
                job['status'] = 'running'
                job['updated_at'] = time.time()
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        close_fds=True, start_new_session=True,
                                        env={**os.environ, "USE_FLAX": "0", "USE_TF": "0"})
                job['process'] = proc
            monitor.start()
            code = proc.wait()
        with GUARD:
            if job['status'] in ('cancelled', 'cancelling'):
                job['status'] = 'cancelled'
                job['output'].unlink(missing_ok=True)
            elif code != 0:
                job['status'] = 'failed'
                job['error'] = f'Inferência encerrou com status {code}; consulte log_tail'
                job['output'].unlink(missing_ok=True)
            elif not job['output'].exists() or not 128 <= job['output'].stat().st_size <= MAX_MP4:
                job['status'] = 'failed'
                job['error'] = 'MP4 ausente ou excede o limite 256 MiB'
                job['output'].unlink(missing_ok=True)
            elif not valid_mp4(job['output']):
                job['status'] = 'failed'
                job['error'] = 'Resposta não é MP4'
                job['output'].unlink(missing_ok=True)
            else:
                job['status'] = 'completed'
            job['updated_at'] = time.time()
    except Exception as exc:
        with GUARD:
            if job['status'] != 'cancelled':
                job['status'] = 'failed'
                job['error'] = f'{type(exc).__name__}: {str(exc)[:220]}'
                job['updated_at'] = time.time()
    finally:
        if image_file:
            image_file.unlink(missing_ok=True)
        stop_metrics.set()
        job['elapsed_seconds'] = round(time.time() - job['created_at'])
        print(f'Video job {job["id"]}: {job["status"]}', flush=True)


def handler_factory():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def json_response(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/health':
                return self.json_response(200, {'ok': True, 'model': MODEL, 'note': 'worker pronto; pesos carregados no primeiro job'})
            if path == '/v1/models':
                return self.json_response(200, {'object': 'list', 'data': [{'id': MODEL, 'object': 'model'}]})
            if path == '/_colab_ai/video/jobs':
                with GUARD:
                    summary = [_public_status(x) for x in JOBS.values()]
                return self.json_response(200, {'jobs': summary})
            found = JOB_PATH.fullmatch(path)
            if not found or found.group(2) == '/cancel':
                return self.json_response(404, {'error': 'Rota não encontrada'})
            with GUARD:
                job = JOBS.get(found.group(1))
                if not job:
                    return self.json_response(404, {'error': 'Job inexistente nesta sessão'})
                status = job['status']
            if found.group(2) != '/file':
                return self.json_response(200, _public_status(job))
            if status != 'completed':
                return self.json_response(409, {'error': 'Vídeo ainda não concluído', 'status': status})
            out = job['output']
            if not out.is_file() or out.stat().st_size > MAX_MP4:
                return self.json_response(500, {'error': 'MP4 indisponível'})
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(out.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with out.open('rb') as fd:
                while True:
                    block = fd.read(65536)
                    if not block:
                        break
                    self.wfile.write(block)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path == '/_colab_ai/video/jobs':
                with GUARD:
                    if any(j['status'] in ('queued', 'running', 'cancelling') for j in JOBS.values()):
                        return self.json_response(409, {'error': 'Uma geração já está em andamento'})
                    if len(JOBS) >= 8:
                        return self.json_response(429, {'error': 'Limite de 8 jobs na sessão; reinicie se necessário'})
                try:
                    size = int(self.headers.get('Content-Length', '-1'))
                    if not 0 < size <= MAX_BODY:
                        return self.json_response(413, {'error': 'Entrada excede 17 MiB'})
                    params = check_request(json.loads(self.rfile.read(size)))
                except (ValueError, TypeError, UnicodeError) as exc:
                    return self.json_response(400, {'error': str(exc)[:300]})
                jid = secrets.token_hex(16)
                now = time.time()
                folder = ROOT / jid
                folder.mkdir(mode=0o700)
                image_file = folder / ('source' + params['extension']) if params['image'] else None
                if image_file:
                    image_file.write_bytes(params['image'])
                    image_file.chmod(0o600)
                job = {'id': jid, 'status': 'queued', 'model': MODEL,
                       'created_at': now, 'updated_at': now,
                       'output': folder / 'video.mp4', 'log': folder / 'job.log',
                       'image': image_file, 'process': None,
                       'generation': {k:params[k] for k in ('aspect','width','height','frames','steps','offload','image_fit')},
                       'gpu': None, 'gpu_samples': 0, 'peak_vram_mib': 0}
                with GUARD:
                    if any(j['status'] in ('queued','running') for j in JOBS.values()):
                        return self.json_response(409, {'error': 'Já existe job ativo'})
                    JOBS[jid] = job
                thread = threading.Thread(target=_run, args=(job, params), daemon=True)
                thread.start()
                return self.json_response(202, _public_status(job))
            found = JOB_PATH.fullmatch(path)
            if not found or found.group(2) != '/cancel':
                return self.json_response(404, {'error': 'Rota não encontrada'})
            with GUARD:
                job = JOBS.get(found.group(1))
                if not job:
                    return self.json_response(404, {'error': 'Job inexistente'})
                if job['status'] not in ('queued', 'running'):
                    return self.json_response(409, {'error': 'Job não está em andamento', 'status': job['status']})
                job['status'] = 'cancelling' if job.get('process') is not None else 'cancelled'
                job['updated_at'] = time.time()
                proc = job.get('process')
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            return self.json_response(200, {'id': found.group(1), 'status': job['status']})

        def log_message(self, fmt, *args):
            pass
    return Handler


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 8000), handler_factory())
    server.daemon_threads = True
    print('Video API privada em 127.0.0.1:8000', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
