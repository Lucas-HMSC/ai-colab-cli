"""Testes de compatibilidade voice sem GPU, login ou dados remotos."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ai_colab_voice_tests', ROOT / 'ai_colab.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def voice_state(variant='custom'):
    return {'mode': 'voice', 'voice_variant': variant, 'model': app.VOICE_MODELS[variant],
            'port': 18001, 'token': 'fake-token', 'gpu': 'L4', 'session': 'colab-ai-code'}


class FakeResponse(io.BytesIO):
    headers = {'Content-Type': 'audio/wav'}
    status = 200


class VoiceCliTests(unittest.TestCase):
    def test_default_start_gpu_is_l4(self):
        args = app.parser_make().parse_args(['start', 'voice'])
        self.assertIsNone(args.gpu)
        self.assertEqual(args.voice_model, 'custom')
        stored = {}
        def save(s): stored['state'] = s
        def read(): return stored.get('state', {})
        calls = []
        with mock.patch.object(app, 'state_read', side_effect=read), \
             mock.patch.object(app, 'state_write', side_effect=save), \
             mock.patch.object(app, 'ensure_key'), \
             mock.patch.object(app, 'cli_base', return_value=['colab','--auth=oauth2']), \
             mock.patch.object(app, 'remote_session_exists', side_effect=[False, True]), \
             mock.patch.object(app, 'run', side_effect=lambda cmd,**kw: calls.append(cmd)), \
             mock.patch.object(app, 'pid_running', return_value=False), \
             mock.patch.object(app, 'tunnel_start', side_effect=lambda s:s), \
             mock.patch.object(app, 'watchdog_start'), \
             mock.patch.object(app, 'wait_ready'), \
             mock.patch.object(app, 'launch_process', return_value=222), \
             mock.patch.object(app.shutil, 'which', return_value='/usr/bin/ssh'), \
             mock.patch.object(app.socket, "socket") as sock:
            app.do_start(args)
        self.assertEqual(stored['state']['gpu'], 'L4')
        self.assertEqual(stored['state']['model'], app.VOICE_MODELS['custom'])
        self.assertTrue(any('colab-ai-voice-api.py' in ' '.join(c) for c in calls))
        self.assertTrue(any(c[-2:] == ['--gpu','L4'] for c in calls))

    def test_voice_generate_writes_wav_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out.wav'
            args=app.parser_make().parse_args(['voice','generate','--text','Olá, tudo bem?',
                                                  '--output',str(output)])
            response=FakeResponse(b'RIFF' + b'\x00'*50)
            with mock.patch.object(app, 'state_read', return_value=voice_state()), \
                 mock.patch.object(app.urllib.request, 'urlopen', return_value=response) as post:
                app.do_voice(args)
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes()[:4], b'RIFF')
            sent=json.loads(post.call_args.args[0].data)
            self.assertEqual(sent['speaker'], 'Ryan')
            self.assertEqual(sent['language'], 'Portuguese')
            self.assertEqual(post.call_args.args[0].headers['X-api-key'], 'fake-token')

    def test_voice_design_requires_active_variant(self):
        args=app.parser_make().parse_args(['voice','design','--text','Olá','--description','Voz brasileira', '--output','/tmp/out.wav'])
        with mock.patch.object(app, 'state_read', return_value=voice_state('custom')):
            with self.assertRaisesRegex(RuntimeError, '--voice-model design'):
                app.do_voice(args)

    def test_clone_requires_consent_before_reading_reference(self):
        args=app.parser_make().parse_args(['voice','clone','--text','Olá', '--reference','/tmp/no-file.wav',
            '--reference-text','Eu falo bem.', '--output','/tmp/out.wav'])
        with mock.patch.object(app, 'state_read', return_value=voice_state('clone')):
            with self.assertRaisesRegex(RuntimeError, '--consent'):
                app.do_voice(args)

    def test_expired_clone_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Path(directory)/'ref.wav'; f.write_bytes(b'0'*(4*1024*1024+1))
            args=app.parser_make().parse_args(['voice','clone','--text','Olá','--reference',str(f),
                '--reference-text','Minha voz', '--consent', '--output',str(Path(directory)/'out.wav')])
            with mock.patch.object(app,'state_read',return_value=voice_state('clone')):
                with self.assertRaisesRegex(RuntimeError,'4 MB'):
                    app.do_voice(args)

    def test_no_voice_when_code_running(self):
        args=app.parser_make().parse_args(['voice','generate','--text','Oi','--output','/tmp/x.wav'])
        with mock.patch.object(app,'state_read',return_value={'mode':'code'}):
            with self.assertRaisesRegex(RuntimeError,'Inicie a modalidade Voice'):
                app.do_voice(args)

    def test_voice_fails_when_very_long_without_starting_api(self):
        args=app.parser_make().parse_args(['voice','generate','--text','T'*1801,'--output','/tmp/out.wav'])
        with mock.patch.object(app,'state_read',return_value=voice_state()),mock.patch.object(app.urllib.request,'urlopen') as post:
            with self.assertRaisesRegex(RuntimeError,'1800'):
                app.do_voice(args)
            post.assert_not_called()

    def test_cache_drive_removed_preferences_migrate_to_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            prefs=Path(directory)/'prefs.json'; prefs.write_text('{"cache":"drive"}')
            with mock.patch.object(app, 'PREFS_FILE',prefs):
                self.assertEqual(app.preferences_read(),{'hf_mode':'fast'})
                self.assertNotIn('cache',app.parser_make()._subparsers._group_actions[0].choices)

    def test_hf_performance_switch(self):
        args=app.parser_make().parse_args(['hf','normal'])
        with mock.patch.object(app, 'preferences_read', return_value={'hf_mode':'fast'}), \
             mock.patch.object(app, 'preferences_write') as save:
            app.do_hf(args)
            self.assertEqual(save.call_args.args[0]['hf_mode'],'normal')

    def test_cli_preserves_code_context(self):
        a=app.parser_make().parse_args(['start','code','--gpu','G4','--context','262144'])
        self.assertEqual(a.context,262144)
        self.assertEqual(a.mode,'code')


if __name__ == '__main__': unittest.main()
