"""Servidor TTS privado (127.0.0.1:8000); acessível somente via proxy SSH.
Um checkpoint por sessão. Carga de modelo só no worker, nunca no kernel Jupyter.
"""
from __future__ import annotations
import base64
import binascii
import io
import json
import traceback
import urllib.error
import urllib.request
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import numpy as np
import soundfile as sf
import torch

CONF = json.loads(Path('/content/colab-ai-config.json').read_text())
MODEL = CONF['model']
VARIANT = CONF['voice_variant']
MODEL_DIR = CONF.get('model_local_path') or MODEL
MAX_REF_BYTES = 4 * 1024 * 1024
MAX_TEXT_CHARS = 1800
LANGS = {'Auto', 'Chinese', 'English', 'Japanese', 'Korean', 'German', 'French',
         'Russian', 'Portuguese', 'Spanish', 'Italian'}
SPEAKERS = {'Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ryan',
            'Aiden', 'Ono_Anna', 'Sohee'}
KOKORO_VOICES = {'pf_dora', 'pm_alex', 'pm_santa'}
GEN_LOCK = threading.Lock()

print(f'Carregando TTS: {MODEL}, perfil={VARIANT}', flush=True)
if VARIANT in ('custom', 'design', 'clone'):
    from qwen_tts import Qwen3TTSModel
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    MODEL_OBJ = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map='cuda:0', dtype=dtype, attn_implementation='sdpa')
elif VARIANT == 'voxcpm2':
    from voxcpm import VoxCPM
    MODEL_OBJ = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, optimize=False)
elif VARIANT == 'kokoro':
    from kokoro import KPipeline
    MODEL_OBJ = KPipeline(lang_code='p', device='cuda')
elif VARIANT == 'chatterbox-br':
    # O repositório PT-BR traz somente pesos regionais. Carregamos os
    # componentes base V3 oficiais e substituímos o T3 pelo checkpoint PT-BR.
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    MODEL_OBJ = ChatterboxMultilingualTTS.from_pretrained(device='cuda', t3_model='v3')
    regional_path = hf_hub_download(repo_id=MODEL,
                                   filename='t3_pt_br.safetensors')
    regional_state = load_file(regional_path)
    MODEL_OBJ.t3.load_state_dict(regional_state, strict=True)
    MODEL_OBJ.t3.eval()
    print('Pesos PT-BR t3_pt_br.safetensors aplicados à base Chatterbox V3.', flush=True)
elif VARIANT == 'higgs3':
    # Higgs TTS 3 NÃO é um pipeline Transformers genérico.
    # O bootstrap inicia vLLM-Omni (porta privada 8002); este worker faz proxy
    # dos pedidos de geração e mantém a autenticação no proxy da CLI.
    MODEL_OBJ = None
else:
    raise RuntimeError('Perfil Voice desconhecido: ' + VARIANT)
print(f'Voice pronto: {MODEL} | perfil={VARIANT}', flush=True)


class UserError(ValueError):
    pass


def reference_audio(data: dict):
    raw = data.get('ref_audio_b64')
    if not isinstance(raw, str) or len(raw) > MAX_REF_BYTES * 4 // 3 + 16:
        raise UserError('Referência ausente ou maior que 4 MB')
    try:
        decoded = base64.b64decode(raw, validate=True)
        if not 0 < len(decoded) <= MAX_REF_BYTES:
            raise ValueError('tamanho inválido')
        info = sf.info(io.BytesIO(decoded))
        if not .5 <= info.duration <= 60 or info.channels > 8:
            raise ValueError('referência exige 0,5–60 segundos, até 8 canais')
        audio, sr = sf.read(io.BytesIO(decoded), dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr
    except (ValueError, TypeError, RuntimeError, binascii.Error) as err:
        raise UserError('Referência inválida: use WAV/FLAC/OGG de até 4 MB: ' + str(err))


def file_for_reference(audio, sr):
    with tempfile.NamedTemporaryFile(prefix='colab-ai-ref-', suffix='.wav', delete=False) as f:
        path = f.name
    sf.write(path, audio, sr)
    return path


def synthesize(data: dict) -> bytes:
    if not isinstance(data, dict):
        raise UserError('JSON deve ser objeto')
    text = data.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise UserError(f'text deve ter entre 1 e {MAX_TEXT_CHARS} caracteres')
    text = text.strip()
    lang = data.get('language', 'Portuguese')
    if VARIANT in ('custom', 'design', 'clone') and lang not in LANGS:
        raise UserError('Idioma Qwen não suportado')
    ref_requested = 'ref_audio_b64' in data
    if ref_requested and VARIANT in ('higgs3', 'kokoro'):
        raise UserError('Áudio de referência não disponível neste backend')
    if ref_requested and data.get('consent') is not True:
        raise UserError('Áudio de referência exige autorização explícita (--consent)')
    ref_audio, ref_sr = reference_audio(data) if ref_requested else (None, None)
    with GEN_LOCK:
        if VARIANT in ('custom', 'design', 'clone'):
            args = {'text': text, 'language': lang}
            with torch.inference_mode():
                if VARIANT == 'custom':
                    speaker = data.get('speaker', 'Ryan')
                    if speaker not in SPEAKERS:
                        raise UserError('Speaker Qwen inválido')
                    args['speaker'] = speaker
                    if data.get('instruct'):
                        args['instruct'] = str(data['instruct'])[:500]
                    wavs, sr = MODEL_OBJ.generate_custom_voice(**args)
                elif VARIANT == 'design':
                    instruct = data.get('instruct')
                    if not isinstance(instruct, str) or not instruct.strip():
                        raise UserError('VoiceDesign exige descrição (--description)')
                    args['instruct'] = instruct[:500]
                    wavs, sr = MODEL_OBJ.generate_voice_design(**args)
                else:
                    if ref_audio is None or not data.get('ref_text'):
                        raise UserError('Clonagem Qwen exige referência e --reference-text')
                    args.update({'ref_audio': (ref_audio, ref_sr),
                                 'ref_text': data['ref_text']})
                    wavs, sr = MODEL_OBJ.generate_voice_clone(**args)
            wav = np.asarray(wavs[0], dtype=np.float32)
        elif VARIANT == 'voxcpm2':
            control = str(data.get('instruct', '') or '').strip()
            if control:
                # O mecanismo de design oficial usa instrução em parênteses.
                control = control.replace('(', '').replace(')', '')[:400]
                text = '(' + control + ')' + text
            steps = int(data.get('inference_timesteps', 10))
            if not 4 <= steps <= 30:
                raise UserError('--steps precisa estar entre 4 e 30')
            kwargs = {'text': text, 'cfg_value': 2.0,
                      'inference_timesteps': steps, 'seed': 42}
            path = None
            try:
                if ref_audio is not None:
                    path = file_for_reference(ref_audio, ref_sr)
                    if data.get('ref_text'):
                        kwargs.update({'prompt_wav_path': path, 'prompt_text': data['ref_text']})
                    else:
                        kwargs['reference_wav_path'] = path
                wav = np.asarray(MODEL_OBJ.generate(**kwargs), dtype=np.float32)
                sr = MODEL_OBJ.tts_model.sample_rate
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)
        elif VARIANT == 'chatterbox-br':
            if len(text) > 450:
                raise UserError('Chatterbox: limite de 450 caracteres por requisição; divida a narração')
            temp = float(data.get('temperature', .8))
            exaggeration = float(data.get('exaggeration', .5))
            cfg = float(data.get('cfg_weight', .5))
            if not .05 <= temp <= 2 or not .25 <= exaggeration <= 2 or not 0 <= cfg <= 1:
                raise UserError('Chatterbox: temperatura, exaggeration ou CFG fora do intervalo')
            kwargs = {'language_id': 'pt', 'temperature': temp,
                      'exaggeration': exaggeration, 'cfg_weight': cfg}
            path = None
            try:
                if ref_audio is not None:
                    path = file_for_reference(ref_audio, ref_sr)
                    kwargs['audio_prompt_path'] = path
                wav = MODEL_OBJ.generate(text, **kwargs).squeeze(0).detach().cpu().numpy()
                sr = MODEL_OBJ.sr
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)
        elif VARIANT == 'kokoro':
            voice = data.get('speaker', 'pf_dora')
            if voice not in KOKORO_VOICES:
                raise UserError('Kokoro: speaker deve ser pf_dora, pm_alex ou pm_santa')
            speed = float(data.get('speed', 1))
            if not .5 <= speed <= 2:
                raise UserError('Kokoro: velocidade deve ficar entre 0.5 e 2')
            chunks = [np.asarray(audio, dtype=np.float32)
                      for _, _, audio in MODEL_OBJ(text, voice=voice, speed=speed) if audio is not None]
            if not chunks:
                raise UserError('Kokoro não gerou fala; revise o texto')
            wav = np.concatenate(chunks)
            sr = 24000
        elif VARIANT == 'higgs3':
            if ref_audio is not None:
                raise UserError('Clonagem Higgs ainda não implementada nesta interface')
            style = str(data.get('instruct', '') or '').strip()
            if style:
                allowed = {'sussurrar': '<|style:whispering|>',
                           'alegre': '<|emotion:enthusiasm|>',
                           'calmo': '<|emotion:contentment|>'}
                text = allowed.get(style.lower(), '') + text
            req = urllib.request.Request(
                'http://127.0.0.1:8002/v1/audio/speech',
                data=json.dumps({'model': MODEL, 'input': text}).encode('utf-8'),
                headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    raw_wav = response.read(40 * 1024 * 1024 + 1)
                if len(raw_wav) > 40 * 1024 * 1024:
                    raise UserError('Higgs retornou áudio maior que 40 MB')
                # Verifica a resposta real de vLLM-Omni antes de devolver WAV.
                with sf.SoundFile(io.BytesIO(raw_wav)) as f:
                    sr = f.samplerate
                    wav = f.read(dtype='float32', always_2d=False)
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)
                wav = np.asarray(wav, dtype=np.float32)
            except urllib.error.HTTPError as error:
                # Não registrar resposta upstream, que poderia incluir texto do usuário.
                raise RuntimeError(f'vLLM-Omni respondeu HTTP {error.code}') from error
        else:
            raise RuntimeError('Perfil Voice não implementado: ' + VARIANT)

    if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all():
        raise RuntimeError('Modelo retornou áudio inválido')
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def reply(self, code, payload, typ='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', typ)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/health':
            if VARIANT == 'higgs3':
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=.8) as upstream:
                        if upstream.status != 200:
                            return self.reply(503, b'{"status":"loading"}')
                except (OSError, urllib.error.HTTPError):
                    return self.reply(503, b'{"status":"loading"}')
            return self.reply(200, b'{"status":"ok"}')
        if self.path == '/v1/models':
            body = json.dumps({'object':'list','data':[{'id':MODEL,'object':'model'}]}).encode()
            return self.reply(200, body)
        self.send_error(404)

    def do_POST(self):
        if self.path != '/_colab_ai/voice/generate':
            return self.send_error(404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 6 * 1024 * 1024:
                return self.send_error(413, 'Limite do corpo excedido')
            data = json.loads(self.rfile.read(size))
            wav = synthesize(data)
            if len(wav) > 40 * 1024 * 1024:
                raise UserError('WAV maior que 40 MB; divida o texto')
            self.reply(200, wav, 'audio/wav')
        except (UserError, json.JSONDecodeError) as err:
            self.reply(400, json.dumps({'error': str(err)}).encode())
        except Exception as err:
            # Logar tipo e localização da exceção, sem prompts, áudio ou texto
            # livre do erro (que pode conter partes da entrada do usuário).
            tb = traceback.extract_tb(err.__traceback__)
            loc = (tb[-1].filename.rsplit('/', 1)[-1] + ':' + str(tb[-1].lineno)) if tb else 'desconhecido'
            print('Falha inferência ' + VARIANT + ': ' + type(err).__name__ + ' em ' + loc, flush=True)
            payload = {'error': 'Falha inferência: ' + type(err).__name__,
                       'location': loc, 'hint': 'ai-colab logs --source vllm --lines 80'}
            self.reply(500, json.dumps(payload).encode())

    def log_message(self, *args):
        pass


print('Servidor Voice privado em 127.0.0.1:8000', flush=True)
ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever(poll_interval=.25)
