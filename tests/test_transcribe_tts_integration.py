import importlib.util
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

MODULE = Path(__file__).resolve().parents[1] / 'ai_colab.py'
spec = importlib.util.spec_from_file_location('colab_ai', MODULE)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


SESSION_STATE = {
    "mode": "code",
    "model": "Qwen/Qwen3.8-27B-FP8",
    "gpu": "G4",
    "port": 6420,
    "token": "tok-123",
    "session": "colab-ai-code",
    "auth": "x",
    "tunnel_pid": 111,
    "serve_pid": 222,
    "watchdog_pid": 333,
}


class StatusJsonTests(TestCase):
    def test_status_json_without_session(self):
        args = app.parser_make().parse_args(['status', '--json'])
        with mock.patch.object(app, 'state_read', return_value={}), \
                mock.patch.object(app, 'local_api') as local_api, \
                mock.patch.object(app, 'runtime_status') as runtime_status:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                app.do_status(args)
        self.assertEqual(json.loads(buffer.getvalue()), {"active": False})
        local_api.assert_not_called()
        runtime_status.assert_not_called()

    def test_status_json_active_session(self):
        snapshot = {"phase": "pronto", "vllm_alive": True, "vllm_ready": True}
        args = app.parser_make().parse_args(['status', '--json'])
        with mock.patch.object(app, 'state_read', return_value=dict(SESSION_STATE)), \
                mock.patch.object(app, 'pid_running', return_value=True), \
                mock.patch.object(app, 'local_api', return_value=(True, 'http 200')) as local_api, \
                mock.patch.object(app, 'runtime_status', return_value=(True, snapshot)) as runtime_status:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                app.do_status(args)
        data = json.loads(buffer.getvalue())
        self.assertTrue(data['active'])
        self.assertEqual(data['mode'], 'code')
        self.assertEqual(data['model'], 'Qwen/Qwen3.8-27B-FP8')
        self.assertEqual(data['gpu'], 'G4')
        self.assertEqual(data['port'], 6420)
        self.assertEqual(data['token'], 'tok-123')
        self.assertTrue(data['tunnel_active'])
        self.assertTrue(data['serve_active'])
        self.assertTrue(data['watchdog_active'])
        self.assertTrue(data['api_ready'])
        self.assertEqual(data['api_detail'], 'http 200')
        self.assertEqual(data['runtime'], snapshot)
        local_api.assert_called_once()
        runtime_status.assert_called_once()

    def test_status_json_runtime_unavailable(self):
        args = app.parser_make().parse_args(['status', '--json'])
        with mock.patch.object(app, 'state_read', return_value=dict(SESSION_STATE)), \
                mock.patch.object(app, 'pid_running', return_value=False), \
                mock.patch.object(app, 'local_api', return_value=(False, 'timeout')), \
                mock.patch.object(app, 'runtime_status', return_value=(False, 'timeout')):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                app.do_status(args)
        data = json.loads(buffer.getvalue())
        self.assertTrue(data['active'])
        self.assertFalse(data['tunnel_active'])
        self.assertFalse(data['api_ready'])
        self.assertIsNone(data['runtime'])


def _fake_asr_response():
    payload = {"text": "Olá mundo.", "language": "pt",
               "segments": [{"start": 0.0, "end": 1.0, "text": "Olá mundo."}]}
    response = mock.MagicMock()
    response.status = 200
    response.read.return_value = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    connection = mock.MagicMock()
    connection.getresponse.return_value = response
    return connection


class TranscribeStdoutTests(TestCase):
    def _audio(self, temp):
        audio = Path(temp) / 'entrada.wav'
        audio.write_bytes(b'RIFF' + b'\x00' * 32)
        return audio

    def _state(self):
        return {"mode": "transcribe", "model": "whisper", "port": 6420, "token": "tok-123"}

    def test_transcribe_stdout_txt_clean_output_no_file(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = self._audio(temp)
            args = app.parser_make().parse_args(
                ['transcribe', '--file', str(audio), '--stdout', '--format', 'txt'])
            with mock.patch.object(app, 'state_read', return_value=self._state()), \
                    mock.patch('http.client.HTTPConnection', return_value=_fake_asr_response()):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    app.do_transcribe(args)
            # stdout limpo: apenas o texto, sem progresso/resumo.
            self.assertEqual(buffer.getvalue(), 'Olá mundo.\n')
            # Nenhum arquivo gravado no disco.
            self.assertEqual([p.name for p in Path(temp).iterdir()], ['entrada.wav'])

    def test_transcribe_json_flag_outputs_valid_json_no_file(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = self._audio(temp)
            args = app.parser_make().parse_args(['transcribe', '--file', str(audio), '--json'])
            with mock.patch.object(app, 'state_read', return_value=self._state()), \
                    mock.patch('http.client.HTTPConnection', return_value=_fake_asr_response()):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    app.do_transcribe(args)
            data = json.loads(buffer.getvalue())  # JSON válido
            self.assertEqual(data['text'], 'Olá mundo.')
            self.assertEqual([p.name for p in Path(temp).iterdir()], ['entrada.wav'])

    def test_transcribe_stdout_format_all_fails_before_connect(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = self._audio(temp)
            args = app.parser_make().parse_args(
                ['transcribe', '--file', str(audio), '--stdout', '--format', 'all'])
            with mock.patch('http.client.HTTPConnection') as connection, \
                    mock.patch.object(app, 'state_read', return_value=self._state()):
                with self.assertRaisesRegex(
                        RuntimeError, 'Não é possível usar --stdout com --format all'):
                    app.do_transcribe(args)
                connection.assert_not_called()


class TranscribeWordTimestampsClientTests(TestCase):
    def test_word_timestamps_flag_sends_header(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / 'entrada.wav'
            audio.write_bytes(b'RIFF' + b'\x00' * 32)
            # Com --word-timestamps: header presente.
            args = app.parser_make().parse_args(
                ['transcribe', '--file', str(audio), '--json', '--word-timestamps'])
            connection = _fake_asr_response()
            with mock.patch.object(app, 'state_read',
                                   return_value={"mode": "transcribe", "model": "whisper",
                                                 "port": 6420, "token": "tok-123"}), \
                    mock.patch('http.client.HTTPConnection', return_value=connection):
                with redirect_stdout(io.StringIO()):
                    app.do_transcribe(args)
            sent = {c.args[0]: c.args[1] for c in connection.putheader.call_args_list}
            self.assertEqual(sent.get('X-Colab-Ai-Word-Timestamps'), 'true')
            # Sem a flag: header ausente.
            args = app.parser_make().parse_args(['transcribe', '--file', str(audio), '--json'])
            connection = _fake_asr_response()
            with mock.patch.object(app, 'state_read',
                                   return_value={"mode": "transcribe", "model": "whisper",
                                                 "port": 6420, "token": "tok-123"}), \
                    mock.patch('http.client.HTTPConnection', return_value=connection):
                with redirect_stdout(io.StringIO()):
                    app.do_transcribe(args)
            sent = {c.args[0] for c in connection.putheader.call_args_list}
            self.assertNotIn('X-Colab-Ai-Word-Timestamps', sent)


def _load_api(variant):
    """Carrega remote/transcribe_api.py com config mockado (mesmo padrão do v028)."""
    original = Path.read_text
    conf = json.dumps({'model': 'teste/model', 'asr_variant': variant})

    def fake_read(path, *args, **kwargs):
        if str(path) == '/content/colab-ai-config.json':
            return conf
        return original(path, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(
        'asr_word_ts_' + variant, Path(__file__).resolve().parents[1] / 'remote/transcribe_api.py')
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(Path, 'read_text', new=fake_read):
        spec.loader.exec_module(module)
    return module


class TranscribeApiWordTimestampsTests(TestCase):
    def _whisper_engine(self):
        seg = mock.MagicMock()
        seg.start, seg.end, seg.text = 0.0, 1.25, 'Olá mundo. '
        w1 = mock.MagicMock(); w1.start, w1.end, w1.word = 0.0, 0.6, 'Olá'
        w2 = mock.MagicMock(); w2.start, w2.end, w2.word = 0.7, 1.2, ' mundo.'
        w3 = mock.MagicMock(); w3.start, w3.end, w3.word = 1.2, 1.3, '  '
        seg.words = [w1, w2, w3]
        engine = mock.MagicMock()
        engine.transcribe.return_value = (iter([seg]), mock.MagicMock(language='pt'))
        return engine

    def test_infer_whisper_with_word_timestamps(self):
        mod = _load_api('whisper')
        engine = self._whisper_engine()
        with mock.patch.object(mod, 'whisper_pcm16_to_array', return_value=object()):
            payload = mod.infer(engine, 'x.wav', 1.0, word_timestamps=True)
        self.assertTrue(engine.transcribe.call_args.kwargs['word_timestamps'])
        self.assertTrue(payload['word_timestamps'])
        self.assertEqual(payload['words'], [
            {'start': 0.0, 'end': 0.6, 'word': 'Olá'},
            {'start': 0.7, 'end': 1.2, 'word': 'mundo.'}])  # palavra vazia é ignorada
        self.assertEqual(payload['segments'], [{'start': 0.0, 'end': 1.25, 'text': 'Olá mundo.'}])
        self.assertEqual(payload['text'], 'Olá mundo.')

    def test_infer_whisper_without_word_timestamps(self):
        mod = _load_api('whisper')
        engine = self._whisper_engine()
        with mock.patch.object(mod, 'whisper_pcm16_to_array', return_value=object()):
            payload = mod.infer(engine, 'x.wav', 1.0)
        self.assertFalse(engine.transcribe.call_args.kwargs['word_timestamps'])
        self.assertFalse(payload['word_timestamps'])
        self.assertNotIn('words', payload)

    def _qwen_engine(self):
        ok = mock.Mock(); ok.text, ok.start_time, ok.end_time = 'Olá', 0.1234, 0.5678
        blank = mock.Mock(); blank.text, blank.start_time, blank.end_time = '  ', 1.0, 2.0
        negative = mock.Mock(); negative.text, negative.start_time, negative.end_time = 'x', -1.0, 1.0
        inverted = mock.Mock(); inverted.text, inverted.start_time, inverted.end_time = 'y', 2.0, 1.0
        stamps = mock.MagicMock(); stamps.items = [ok, blank, negative, inverted]
        result = mock.MagicMock()
        result.text, result.language, result.time_stamps = 'Olá', 'Portuguese', stamps
        engine = mock.MagicMock()
        engine.transcribe.return_value = [result]
        return engine

    def test_infer_qwen_with_word_timestamps(self):
        mod = _load_api('qwen')
        payload = mod.infer(self._qwen_engine(), 'x.wav', 1.0, word_timestamps=True)
        self.assertTrue(payload['word_timestamps'])
        # Vazios e tempos inválidos (start<0, end<start) são filtrados.
        self.assertEqual(payload['words'], [{'start': 0.123, 'end': 0.568, 'word': 'Olá'}])

    def test_infer_qwen_without_word_timestamps(self):
        mod = _load_api('qwen')
        payload = mod.infer(self._qwen_engine(), 'x.wav', 1.0)
        self.assertFalse(payload['word_timestamps'])
        self.assertNotIn('words', payload)

    def test_do_post_rejects_invalid_word_timestamps_header(self):
        mod = _load_api('whisper')
        # Sem ciclo real de requisição: o __init__ do handler só instancia.
        with mock.patch.object(mod.BaseHTTPRequestHandler, 'handle', lambda self: None):
            handler = mod.handler_for(object())(mock.MagicMock(), ('127.0.0.1', 12345),
                                                mock.MagicMock())
        handler.path = '/_colab_ai/transcribe'
        handler.headers = {'Content-Length': '10', 'X-Colab-Ai-Extension': '.wav',
                           'X-Colab-Ai-Language': 'pt',
                           'X-Colab-Ai-Word-Timestamps': 'maybe'}
        with mock.patch.object(handler, 'reply') as reply:
            handler.do_POST()
        reply.assert_called_once()
        self.assertEqual(reply.call_args.args[0], 400)
