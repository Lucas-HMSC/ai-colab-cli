"""Servidor ASR privado: recebe áudio e retorna transcrição estruturada.

O SSH e o proxy autenticado são a única via de acesso. Áudios temporários são
excluídos no final da requisição; nenhum conteúdo de áudio ou texto é logado.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONFIG_PATH = Path('/content/colab-ai-config.json')
CONF = json.loads(CONFIG_PATH.read_text())
MODEL_ID = CONF['model']
VARIANT = CONF['asr_variant']
MAX_UPLOAD = 150 * 1024 * 1024
MAX_AUDIO_SECONDS = 2 * 60 * 60
SUPPORTED_EXT = {'.wav', '.mp3', '.m4a', '.mp4', '.ogg', '.oga', '.opus',
                 '.flac', '.aac', '.webm', '.mov', '.mpeg', '.mpga', '.wma'}
LOCK = threading.Lock()


def align_to_segments(items, max_words=12, max_duration=6.0):
    """Agrupa palavras com tempos reais em legendas, sem estimar timestamps."""
    result = []
    group = []
    for item in items:
        text = str(getattr(item, 'text', '')).strip()
        start = float(getattr(item, 'start_time', 0))
        end = float(getattr(item, 'end_time', start))
        if not text or start < 0 or end < start:
            continue
        if group and (len(group) >= max_words or end - group[0]['start'] > max_duration
                      or start - group[-1]['end'] > 1.5):
            result.append({'start': group[0]['start'], 'end': group[-1]['end'],
                           'text': ' '.join(w['text'] for w in group)})
            group = []
        group.append({'start': start, 'end': end, 'text': text})
        if text.endswith(('.', '!', '?', '…')) and len(group) >= 4:
            result.append({'start': group[0]['start'], 'end': group[-1]['end'],
                           'text': ' '.join(w['text'] for w in group)})
            group = []
    if group:
        result.append({'start': group[0]['start'], 'end': group[-1]['end'],
                       'text': ' '.join(w['text'] for w in group)})
    return result


def duration_for(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
                       capture_output=True, text=True, timeout=20)
    if p.returncode or not p.stdout.strip():
        raise ValueError('Áudio ilegível ou formato sem duração detectável')
    sec = float(p.stdout.strip())
    if not 0 < sec <= MAX_AUDIO_SECONDS:
        raise ValueError('Áudio deve ter duração entre 0 e 120 minutos')
    return round(sec, 3)


def to_wav(path, output):
    p = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', str(path), '-vn', '-ac', '1', '-ar', '16000',
                        '-f', 'wav', str(output)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=300)
    if p.returncode or not output.exists() or output.stat().st_size < 44:
        raise ValueError('Não foi possível decodificar o áudio (FFmpeg)')


def load_model():
    if VARIANT == 'whisper':
        from faster_whisper import WhisperModel
        return WhisperModel(MODEL_ID, device='cuda', compute_type='float16')
    if VARIANT == 'qwen':
        import torch
        from qwen_asr import Qwen3ASRModel
        return Qwen3ASRModel.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map='cuda:0',
            max_inference_batch_size=1, max_new_tokens=2048,
            forced_aligner='Qwen/Qwen3-ForcedAligner-0.6B',
            forced_aligner_kwargs={'dtype': torch.bfloat16, 'device_map': 'cuda:0'})
    raise RuntimeError('Perfil ASR desconhecido')


def infer(model, audio_path, duration, language='pt'):
    if VARIANT == 'whisper':
        seg_iter, info = model.transcribe(str(audio_path), language=None if language == 'auto' else 'pt',
                                         task='transcribe', beam_size=5, vad_filter=True,
                                         condition_on_previous_text=True,
                                         word_timestamps=False)
        segments = [{'start': round(float(s.start), 3), 'end': round(float(s.end), 3),
                     'text': s.text.strip()} for s in seg_iter if s.text.strip()]
        transcript = ' '.join(s['text'] for s in segments)
        detected = info.language or 'desconhecido'
    else:
        result = model.transcribe(audio=str(audio_path),
                                  language=None if language == 'auto' else 'Portuguese',
                                  return_time_stamps=True)[0]
        transcript = result.text.strip()
        detected = result.language or 'desconhecido'
        items = result.time_stamps.items if result.time_stamps is not None else []
        segments = align_to_segments(items)
    return {'model': MODEL_ID, 'engine': VARIANT, 'language': detected,
            'duration_seconds': duration, 'text': transcript, 'segments': segments,
            'timestamps': bool(segments)}


def handler_for(model):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def reply(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/health':
                return self.reply(200, {'ready': True})
            if self.path == '/v1/models':
                return self.reply(200, {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model'}]})
            self.reply(404, {'error': 'Rota desconhecida'})

        def do_POST(self):
            if self.path != '/_colab_ai/transcribe':
                return self.reply(404, {'error': 'Rota desconhecida'})
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                ext = self.headers.get('X-Colab-Ai-Extension', '').lower()
                language = self.headers.get('X-Colab-Ai-Language', 'pt').lower()
                if not 0 < size <= MAX_UPLOAD or ext not in SUPPORTED_EXT or language not in ('pt', 'auto'):
                    return self.reply(400, {'error': 'Arquivo, extensão ou idioma inválido (máx. 150 MiB)'})
                with tempfile.TemporaryDirectory(prefix='colab-asr-') as directory:
                    source = Path(directory) / ('entrada' + ext)
                    remaining = size
                    with source.open('wb') as output:
                        while remaining > 0:
                            block = self.rfile.read(min(65536, remaining))
                            if not block:
                                raise ValueError('Upload interrompido')
                            output.write(block)
                            remaining -= len(block)
                    wav = Path(directory) / 'audio.wav'
                    duration = duration_for(source)
                    to_wav(source, wav)
                    with LOCK:
                        response = infer(model, wav, duration, language=language)
                self.reply(200, response)
            except (ValueError, subprocess.TimeoutExpired) as exc:
                self.reply(422, {'error': type(exc).__name__, 'message': str(exc)[:180]})
            except Exception as exc:
                # Não registrar nomes, textos ou conteúdo de áudio.
                import traceback
                frame = traceback.extract_tb(exc.__traceback__)[-1]
                print(f'Falha ASR: {type(exc).__name__} @ {frame.name}:{frame.lineno}', flush=True)
                self.reply(500, {'error': 'Falha na transcrição', 'type': type(exc).__name__,
                                 'location': f'{frame.name}:{frame.lineno}',
                                 'hint': 'Use ai-colab logs --source vllm --lines 100'})

        def log_message(self, fmt, *args):
            pass
    return Handler


if __name__ == '__main__':
    print('Carregando ASR:', VARIANT, MODEL_ID, flush=True)
    loaded = load_model()
    print('ASR pronto:', MODEL_ID, flush=True)
    server = ThreadingHTTPServer(('127.0.0.1', 8000), handler_for(loaded))
    server.daemon_threads = True
    server.serve_forever(poll_interval=0.3)
