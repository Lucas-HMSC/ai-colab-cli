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


class TranscribeDiarizeClientTests(TestCase):
    def test_diarize_flag_sends_header(self):
        state = {"mode": "transcribe", "model": "whisper", "port": 6420, "token": "tok-123"}
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / 'entrada.wav'
            audio.write_bytes(b'RIFF' + b'\x00' * 32)
            # Com --diarize: header presente.
            args = app.parser_make().parse_args(
                ['transcribe', '--file', str(audio), '--json', '--diarize'])
            connection = _fake_asr_response()
            with mock.patch.object(app, 'state_read', return_value=state), \
                    mock.patch('http.client.HTTPConnection', return_value=connection):
                with redirect_stdout(io.StringIO()):
                    app.do_transcribe(args)
            sent = {c.args[0]: c.args[1] for c in connection.putheader.call_args_list}
            self.assertEqual(sent.get('X-Colab-Ai-Diarize'), 'true')
            # Sem a flag: header ausente.
            args = app.parser_make().parse_args(['transcribe', '--file', str(audio), '--json'])
            connection = _fake_asr_response()
            with mock.patch.object(app, 'state_read', return_value=state), \
                    mock.patch('http.client.HTTPConnection', return_value=connection):
                with redirect_stdout(io.StringIO()):
                    app.do_transcribe(args)
            sent = {c.args[0] for c in connection.putheader.call_args_list}
            self.assertNotIn('X-Colab-Ai-Diarize', sent)


class RenderSpeakerTests(TestCase):
    def test_txt_without_speaker_is_unchanged(self):
        payload = {'text': 'Olá mundo.',
                   'segments': [{'start': 0.0, 'end': 1.0, 'text': 'Olá mundo.'}]}
        self.assertEqual(app._render_transcription(payload, 'txt'), 'Olá mundo.\n')

    def test_txt_with_speaker_prefixes_lines(self):
        payload = {'text': 'Olá mundo.', 'segments': [
            {'start': 0.0, 'end': 1.0, 'text': 'Olá', 'speaker': 'Falante 1'},
            {'start': 1.0, 'end': 2.0, 'text': 'mundo.'}]}
        self.assertEqual(app._render_transcription(payload, 'txt'),
                         '[Falante 1] Olá\nmundo.\n')

    def test_srt_with_speaker_prefixes_only_text_lines(self):
        base = {'segments': [{'start': 0.0, 'end': 1.0, 'text': 'Olá mundo.'}]}
        payload = {'segments': [{'start': 0.0, 'end': 1.0, 'text': 'Olá mundo.',
                                 'speaker': 'Falante 1'}]}
        plain = app._render_transcription(base, 'srt').split('\n')
        spaky = app._render_transcription(payload, 'srt').split('\n')
        # Mesma estrutura; apenas a linha de legenda ganha o prefixo.
        for i, line in enumerate(plain):
            self.assertEqual(spaky[i], ('[Falante 1] ' + line) if i == 2 else line)


class AssignSpeakersTests(TestCase):
    def test_normaliza_ids_em_ordem_da_primeira_aparição(self):
        mod = _load_api('whisper')
        items = [{'start': 0.0, 'end': 1.0}, {'start': 2.0, 'end': 3.0}]
        turns = [(0.0, 1.0, 'SPEAKER_00'), (2.0, 3.0, 'SPEAKER_01')]
        result = mod.assign_speakers(items, turns)
        self.assertEqual([i['speaker'] for i in result], ['Falante 1', 'Falante 2'])

    def test_maior_sobreposição_e_sem_sobreposição(self):
        mod = _load_api('whisper')
        items = [{'start': 0.0, 'end': 2.0}, {'start': 2.5, 'end': 5.5},
                 {'start': 6.0, 'end': 8.0}, {'start': 1.0, 'end': 6.0},
                 {'start': 0.5, 'end': 1.5}]
        turns = [(0.0, 2.0, 'SPEAKER_00'), (3.0, 5.0, 'SPEAKER_01')]
        result = mod.assign_speakers(items, turns)
        self.assertEqual([i['speaker'] for i in result],
                         ['Falante 1', 'Falante 2', None, 'Falante 2', 'Falante 1'])


class InferDiarizeTests(TestCase):
    def _whisper_engine(self):
        seg = mock.MagicMock()
        seg.start, seg.end, seg.text = 0.0, 1.25, 'Olá mundo. '
        w1 = mock.MagicMock(); w1.start, w1.end, w1.word = 0.0, 0.6, 'Olá'
        w2 = mock.MagicMock(); w2.start, w2.end, w2.word = 0.7, 1.2, ' mundo.'
        seg.words = [w1, w2]
        engine = mock.MagicMock()
        engine.transcribe.return_value = (iter([seg]), mock.MagicMock(language='pt'))
        return engine

    def test_infer_diarize_true_segments_and_words(self):
        mod = _load_api('whisper')
        turns = [(0.0, 1.0, 'SPK_A'), (0.7, 1.3, 'SPK_B')]
        engine = self._whisper_engine()
        with mock.patch.object(mod, 'whisper_pcm16_to_array', return_value=object()), \
                mock.patch.object(mod, 'diarize_audio', return_value=turns) as diarize:
            payload = mod.infer(engine, 'x.wav', 1.0, word_timestamps=True, diarize=True)
        diarize.assert_called_once_with('x.wav')
        self.assertTrue(payload['has_speaker_labels'])
        self.assertEqual(payload['segments'][0]['speaker'], 'Falante 1')
        self.assertEqual(payload['words'][0]['speaker'], 'Falante 1')  # 0.0–0.6 → SPK_A
        self.assertEqual(payload['words'][1]['speaker'], 'Falante 2')  # 0.7–1.2 → SPK_B

    def test_infer_diarize_consistency_between_segments_and_words(self):
        mod = _load_api('whisper')
        turns = [(0.0, 1.0, 'SPK_A'), (0.7, 1.3, 'SPK_B')]
        seg = mock.MagicMock()
        seg.start, seg.end, seg.text = 0.0, 1.25, 'Olá mundo. '
        w1 = mock.MagicMock(); w1.start, w1.end, w1.word = 0.7, 1.2, 'mundo.'
        w2 = mock.MagicMock(); w2.start, w2.end, w2.word = 0.0, 0.6, 'Olá'
        seg.words = [w1, w2]
        engine = mock.MagicMock()
        engine.transcribe.return_value = (iter([seg]), mock.MagicMock(language='pt'))
        with mock.patch.object(mod, 'whisper_pcm16_to_array', return_value=object()), \
                mock.patch.object(mod, 'diarize_audio', return_value=turns):
            payload = mod.infer(engine, 'x.wav', 1.0, word_timestamps=True, diarize=True)
        # O 1º word cai em SPK_B antes de SPK_A aparecer em words: os IDs
        # devem ser compartilhados com os segmentos (não reiniciados).
        self.assertEqual(payload['segments'][0]['speaker'], 'Falante 1')  # SPK_A
        self.assertEqual(payload['words'][0]['speaker'], 'Falante 2')     # SPK_B
        self.assertEqual(payload['words'][1]['speaker'], 'Falante 1')     # SPK_A, idêntico ao segmento

    def test_infer_diarize_false_default(self):
        mod = _load_api('whisper')
        engine = self._whisper_engine()
        with mock.patch.object(mod, 'whisper_pcm16_to_array', return_value=object()), \
                mock.patch.object(mod, 'diarize_audio') as diarize:
            payload = mod.infer(engine, 'x.wav', 1.0)
        diarize.assert_not_called()
        self.assertFalse(payload['has_speaker_labels'])
        self.assertNotIn('speaker', payload['segments'][0])


class DiarizeHeaderTests(TestCase):
    def test_do_post_rejects_invalid_diarize_header(self):
        mod = _load_api('whisper')
        with mock.patch.object(mod.BaseHTTPRequestHandler, 'handle', lambda self: None):
            handler = mod.handler_for(object())(mock.MagicMock(), ('127.0.0.1', 12345),
                                                mock.MagicMock())
        handler.path = '/_colab_ai/transcribe'
        handler.headers = {'Content-Length': '10', 'X-Colab-Ai-Extension': '.wav',
                           'X-Colab-Ai-Language': 'pt', 'X-Colab-Ai-Diarize': 'sim'}
        with mock.patch.object(handler, 'reply') as reply:
            handler.do_POST()
        reply.assert_called_once()
        self.assertEqual(reply.call_args.args[0], 400)


class LoadDiarizerTests(TestCase):
    def _load_with_conf(self, conf_dict, suffix):
        original = Path.read_text
        conf = json.dumps(conf_dict)

        def fake_read(path, *args, **kwargs):
            if str(path) == '/content/colab-ai-config.json':
                return conf
            return original(path, *args, **kwargs)

        spec = importlib.util.spec_from_file_location(
            'asr_diarizer_' + suffix, Path(__file__).resolve().parents[1] / 'remote/transcribe_api.py')
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(Path, 'read_text', new=fake_read):
            spec.loader.exec_module(module)
        return module

    def test_load_diarizer_requires_hf_token(self):
        mod = self._load_with_conf({'model': 'teste/model', 'asr_variant': 'whisper'}, 'notok')
        with mock.patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'HF_TOKEN'):
                mod.load_diarizer()

    def test_load_diarizer_uses_configured_hf_token_and_caches(self):
        import sys
        import types
        mod = self._load_with_conf(
            {'model': 'teste/model', 'asr_variant': 'whisper', 'hf_token': 'hf_test_123'},
            'withtok',
        )
        fake_pipeline_cls = mock.MagicMock()
        fake_pipeline_cls.from_pretrained.return_value = 'pipeline-instance'
        fake_pyannote = types.ModuleType('pyannote')
        fake_pyannote_audio = types.ModuleType('pyannote.audio')
        fake_pyannote_audio.Pipeline = fake_pipeline_cls
        fake_pyannote.audio = fake_pyannote_audio
        with mock.patch.dict('os.environ', {}, clear=True), \
                mock.patch.dict(sys.modules, {'pyannote': fake_pyannote,
                                              'pyannote.audio': fake_pyannote_audio}):
            first = mod.load_diarizer()
            second = mod.load_diarizer()
        self.assertEqual(first, 'pipeline-instance')
        self.assertIs(first, second)
        fake_pipeline_cls.from_pretrained.assert_called_once_with(
            'pyannote/speaker-diarization-community-1', token='hf_test_123')


class HfTokenPropagationTests(TestCase):
    def _run_start_capture_config(self, env_vars):
        with tempfile.TemporaryDirectory() as temp:
            state_dir = Path(temp)
            captured = {}

            class _StopUpload(Exception):
                pass

            def fake_run(cmd, **kwargs):
                if 'upload' in cmd:
                    captured['payload'] = json.loads((state_dir / 'remote-config.json').read_text(encoding='utf-8'))
                    raise _StopUpload()
                return mock.MagicMock(returncode=0, stdout='')

            args = app.parser_make().parse_args(['start', 'code'])
            with mock.patch.dict('os.environ', env_vars, clear=True), \
                    mock.patch.object(app, 'STATE_DIR', state_dir), \
                    mock.patch.object(app, 'state_read', return_value={}), \
                    mock.patch.object(app, 'state_write'), \
                    mock.patch.object(app, 'ensure_key'), \
                    mock.patch.object(app, 'cli_base', return_value=['colab']), \
                    mock.patch.object(app, 'remote_session_exists', return_value=True), \
                    mock.patch.object(app, 'pid_running', return_value=False), \
                    mock.patch.object(app, 'run', side_effect=fake_run):
                with self.assertRaises(_StopUpload):
                    app.do_start(args)
            return captured['payload']

    def test_do_start_code_writes_hf_token_when_present(self):
        payload = self._run_start_capture_config({'HF_TOKEN': 'hf_all_modes_123', 'PATH': '/usr/bin:/bin'})
        self.assertEqual(payload['mode'], 'code')
        self.assertEqual(payload.get('hf_token'), 'hf_all_modes_123')

    def test_do_start_code_omits_hf_token_when_absent(self):
        payload = self._run_start_capture_config({'PATH': '/usr/bin:/bin'})
        self.assertEqual(payload['mode'], 'code')
        self.assertNotIn('hf_token', payload)

    def test_remote_scripts_propagate_hf_token(self):
        root = Path(__file__).resolve().parents[1] / 'remote'
        for name in ('launch.py', 'bootstrap.py', 'voice_bootstrap.py',
                     'transcribe_bootstrap.py', 'video_bootstrap.py', 'video_api.py'):
            text = (root / name).read_text(encoding='utf-8')
            self.assertIn('hf_token', text, f'{name} não lê hf_token do config')
            self.assertIn('HF_TOKEN', text, f'{name} não exporta HF_TOKEN')

    def test_code_bootstrap_uses_uv_pip_system(self):
        text = (Path(__file__).resolve().parents[1] / 'remote/bootstrap.py').read_text(encoding='utf-8')
        self.assertIn('"pip", "install", "--system"', text)
        self.assertIn('vllm>=0.28.0', text)
        self.assertIn('Instalando vLLM>=0.28 / transformers>=5.8 (primeira vez na VM)...', text)


class HarnessAndOnboardingTests(TestCase):
    def _code_state(self):
        return {
            'mode': 'code',
            'model': 'Qwen/Qwen3.8-27B-FP8',
            'gpu': 'G4',
            'port': 18001,
            'token': 'tok-opencode',
            'context': 131072,
        }

    def test_configure_opencode_provider_creates_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = Path(temp) / 'opencode' / 'opencode.json'
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({
                'providers': {'custom': {'name': 'Outro'}},
                'theme': 'dark',
            }), encoding='utf-8')
            written = app.configure_opencode_provider(self._code_state(), config_path=cfg)
            self.assertEqual(written, cfg)
            self.assertEqual(cfg.stat().st_mode & 0o777, 0o600)
            data = json.loads(cfg.read_text(encoding='utf-8'))
            self.assertEqual(data['$schema'], 'https://opencode.ai/config.json')
            self.assertEqual(data['theme'], 'dark')
            self.assertEqual(data['providers']['custom'], {'name': 'Outro'})
            self.assertEqual(data['model'], 'colab-qwen/qwen-coding')
            prov = data['providers']['colab-qwen']
            self.assertEqual(prov['settings']['baseURL'], 'http://127.0.0.1:18001/v1')
            self.assertEqual(prov['settings']['apiKey'], 'tok-opencode')
            mdl = prov['models']['qwen-coding']
            self.assertEqual(mdl['modelID'], 'Qwen/Qwen3.8-27B-FP8')
            self.assertEqual(mdl['limit']['context'], 131072)

    def test_do_opencode_execs_with_standalone_auto_and_opt_out(self):
        state = self._code_state()
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict('os.environ', {'XDG_CONFIG_HOME': temp, 'COLAB_AI_SKIP_PERMISSIONS': '1'}), \
                    mock.patch.object(app, 'state_read', return_value=state), \
                    mock.patch.object(app, 'local_api', return_value=(True, 'ok')), \
                    mock.patch.object(app, 'watchdog_start'), \
                    mock.patch.object(app.shutil, 'which', return_value='/usr/bin/opencode'), \
                    mock.patch.object(app.os, 'execvpe') as ev:
                args = app.parser_make().parse_args(['opencode', '--', 'run', 'oi'])
                app.do_opencode(args)
                exe, argv, _env = ev.call_args.args
                self.assertEqual(exe, '/usr/bin/opencode')
                self.assertEqual(argv, ['/usr/bin/opencode', '--standalone', '--auto', 'run', 'oi'])
                cfg_file = Path(temp) / 'opencode' / 'opencode.json'
                self.assertTrue(cfg_file.is_file())
            with mock.patch.dict('os.environ', {'XDG_CONFIG_HOME': temp, 'COLAB_AI_SKIP_PERMISSIONS': '0'}), \
                    mock.patch.object(app, 'state_read', return_value=state), \
                    mock.patch.object(app, 'local_api', return_value=(True, 'ok')), \
                    mock.patch.object(app, 'watchdog_start'), \
                    mock.patch.object(app.shutil, 'which', return_value='/usr/bin/opencode'), \
                    mock.patch.object(app.os, 'execvpe') as ev:
                app.do_opencode(app.parser_make().parse_args(['opencode']))
                self.assertEqual(ev.call_args.args[1], ['/usr/bin/opencode', '--standalone'])

    def test_code_harness_flag_and_auto_fallback_to_opencode(self):
        state = self._code_state()
        args = app.parser_make().parse_args(['code', '--harness', 'opencode', '--', 'run', 'ola'])
        with mock.patch.object(app, 'do_opencode') as op:
            app.do_claude(args)
            op.assert_called_once_with(args)

        def fake_which(name):
            return '/usr/bin/opencode' if name == 'opencode' else None

        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict('os.environ', {'XDG_CONFIG_HOME': temp}), \
                    mock.patch.object(app, 'state_read', return_value=state), \
                    mock.patch.object(app, 'local_api', return_value=(True, 'ok')), \
                    mock.patch.object(app, 'watchdog_start'), \
                    mock.patch.object(app.shutil, 'which', side_effect=fake_which), \
                    mock.patch.object(app.os, 'execvpe') as ev:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    app.do_claude(app.parser_make().parse_args(['code']))
                self.assertIn('OpenCode', buf.getvalue())
                self.assertEqual(ev.call_args.args[0], '/usr/bin/opencode')

    def test_ensure_claude_onboarding_creates_and_preserves(self):
        with tempfile.TemporaryDirectory() as temp:
            claude_json = Path(temp) / '.claude.json'
            app.ensure_claude_onboarding(claude_json=claude_json)
            self.assertEqual(claude_json.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(claude_json.read_text(encoding='utf-8')),
                             {'hasCompletedOnboarding': True})

            claude_json.write_text(json.dumps({'theme': 'light', 'custom': 42}), encoding='utf-8')
            app.ensure_claude_onboarding(claude_json=claude_json)
            data = json.loads(claude_json.read_text(encoding='utf-8'))
            self.assertEqual(data, {'theme': 'light', 'custom': 42, 'hasCompletedOnboarding': True})

    def test_proxy_forwards_word_timestamps_and_diarize_headers(self):
        proxy = importlib.util.spec_from_file_location(
            'asr_proxy_headers', Path(__file__).resolve().parents[1] / 'remote/proxy.py')
        mod = importlib.util.module_from_spec(proxy)
        proxy.loader.exec_module(mod)
        mod.ASR_MODE = True
        captured_headers = {}

        class FakeResp:
            status = 200
            def getheader(self, name, default=None):
                return 'application/json' if name == 'Content-Type' else default
            def read1(self, _n):
                if not getattr(self, '_done', False):
                    self._done = True
                    return b'{"text":"ok"}'
                return b''

        class FakeConn:
            def __init__(self, *a, **kw):
                pass
            def putrequest(self, method, path):
                captured_headers['_req'] = (method, path)
            def putheader(self, k, v):
                captured_headers[k] = v
            def endheaders(self):
                pass
            def send(self, _b):
                pass
            def getresponse(self):
                return FakeResp()
            def close(self):
                pass

        with mock.patch.object(mod.http.client, 'HTTPConnection', FakeConn), \
                mock.patch.object(mod.BaseHTTPRequestHandler, 'handle', lambda self: None):
            h = mod.make_handler('tok')(mock.MagicMock(), ('127.0.0.1', 12345), mock.MagicMock())
            h.command = 'POST'
            h.path = '/_colab_ai/transcribe'
            h.requestline = 'POST /_colab_ai/transcribe HTTP/1.1'
            h.request_version = 'HTTP/1.1'
            h.headers = {
                'Authorization': 'Bearer tok',
                'Content-Length': '3',
                'X-Colab-Ai-Extension': '.wav',
                'X-Colab-Ai-Language': 'pt',
                'X-Colab-Ai-Word-Timestamps': 'true',
                'X-Colab-Ai-Diarize': 'true',
            }
            h.rfile = io.BytesIO(b'abc')
            h.wfile = io.BytesIO()
            h.do_POST()
        self.assertEqual(captured_headers.get('X-Colab-Ai-Word-Timestamps'), 'true')
        self.assertEqual(captured_headers.get('X-Colab-Ai-Diarize'), 'true')
