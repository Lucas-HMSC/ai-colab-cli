import importlib.util
import json
import os
from pathlib import Path
from unittest import TestCase, mock
import tempfile

MODULE = Path(__file__).resolve().parents[1] / 'ai_colab.py'
spec = importlib.util.spec_from_file_location('colab_ai', MODULE)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class CliTests(TestCase):
    def test_mode_catalog(self):
        self.assertEqual(app.PROFILES['code']['model'], 'Qwen/Qwen3.8-27B-FP8')
        self.assertIn('code-next', app.PROFILES)
        self.assertIn('video', app.COMING)
        self.assertNotIn('video', app.PROFILES)

    def test_context_default_increases_to_131k(self):
        self.assertEqual(app.PROFILES['code']['context'], 131072)

    def test_state_secret_permissions(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(app, 'STATE_DIR', Path(temp)), mock.patch.object(app, 'STATE_FILE', Path(temp)/'state.json'):
                app.state_write({'token':'secret', 'mode':'code'})
                self.assertEqual(app.state_read()['token'], 'secret')
                self.assertEqual((Path(temp)/'state.json').stat().st_mode & 0o777, 0o600)

    def test_unimplemented_mode_never_invokes_gpu(self):
        with mock.patch.object(app, 'run') as r:
            rc = app.main(['start', 'image'])
            self.assertEqual(rc, 0)
            r.assert_not_called()

    def test_reject_unknown_mode(self):
        with self.assertRaises(SystemExit):
            app.parser_make().parse_args(['start', 'nonexistent'])

    def test_launcher_flags(self):
        a=app.parser_make().parse_args(['claude','--','-p','Responda OK'])
        self.assertEqual(a.extra, ['--','-p','Responda OK'])

    def test_ssh_tunnel_private_localhost(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(app, 'STATE_DIR', Path(temp)), mock.patch.object(app,'STATE_FILE',Path(temp)/'state.json'), \
                 mock.patch.object(app,'ensure_key', return_value=Path(temp)/'id_ed25519'), \
                 mock.patch.object(app, 'cli_base', return_value=['/usr/bin/colab','--auth=oauth2']), \
                 mock.patch.object(app.shutil, 'which', return_value='/usr/bin/ssh'), \
                 mock.patch.object(app, 'launch_process', return_value=88888) as launch:
                state={'auth':'oauth2','port':18001,'session':'colab-ai-code','token':'foo'}
                app.state_write(state)
                app.tunnel_start(state)
                argv=launch.call_args[0][0]
                self.assertIn('127.0.0.1:18001:127.0.0.1:8001',argv)
                self.assertIn('root@colab',argv)
                self.assertIn('-N',argv)
                self.assertIn('--proxy-mode', ' '.join(argv))
                self.assertEqual(app.state_read()['tunnel_pid'], 88888)

    def test_no_config_modify_original_claude(self):
        with mock.patch.object(app,'state_read',return_value={'mode':'code','model':'Qwen/Qwen3.8-27B-FP8','port':18001,'token':'abc'}), \
             mock.patch.object(app,'local_api',return_value=(True,'ok')), \
             mock.patch.object(app.shutil,'which',return_value='/usr/bin/claude'), \
             mock.patch.object(app,'watchdog_start') as _watchdog, \
             mock.patch.object(app.os,'execvpe') as ev:
            app.do_claude(app.parser_make().parse_args(['claude','--','-p','OK']))
            arguments=ev.call_args[0]
            self.assertEqual(arguments[1][:3], ['/usr/bin/claude','--model','Qwen/Qwen3.8-27B-FP8'])
            self.assertEqual(arguments[2]['CLAUDE_CODE_EFFORT_LEVEL'],'medium')
            self.assertEqual(arguments[2]['ANTHROPIC_AUTH_TOKEN'],'abc')
            self.assertEqual(arguments[2]['CLAUDE_CODE_MAX_CONTEXT_TOKENS'],'131072')
            self.assertEqual(arguments[2]['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'],'75')
            self.assertEqual(arguments[1].count('--dangerously-skip-permissions'),1)
            self.assertEqual(arguments[1][-2:], ['-p','OK'])


    def test_claude_launcher_respects_custom_session_context(self):
        state = {'mode':'code', 'model':'Qwen/Qwen3.8-27B-FP8', 'port':18001,
                 'token':'abc', 'context':131072}
        with mock.patch.object(app,'state_read',return_value=state), \
             mock.patch.object(app,'local_api',return_value=(True,'ok')), \
             mock.patch.object(app.shutil,'which',return_value='/usr/bin/claude'), \
             mock.patch.dict(os.environ, {'COLAB_AI_SKIP_PERMISSIONS':'1'}), \
             mock.patch.object(app,'watchdog_start') as _watchdog, \
             mock.patch.object(app.os,'execvpe') as ev:
            app.do_claude(app.parser_make().parse_args(['claude']))
            argv, env = ev.call_args.args[1:3]
            self.assertEqual(env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'],'131072')
            self.assertEqual(env['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'],'75')
            self.assertIn('--dangerously-skip-permissions',argv)

    def test_claude_launcher_skip_flag_is_not_duplicated(self):
        state = {'mode':'code', 'model':'Qwen/Qwen3.8-27B-FP8', 'port':18001,
                 'token':'abc', 'context':65536}
        with mock.patch.object(app,'state_read',return_value=state), \
             mock.patch.object(app,'local_api',return_value=(True,'ok')), \
             mock.patch.object(app.shutil,'which',return_value='/usr/bin/claude'), \
             mock.patch.dict(os.environ, {'COLAB_AI_SKIP_PERMISSIONS':'1'}), \
             mock.patch.object(app,'watchdog_start') as _watchdog, \
             mock.patch.object(app.os,'execvpe') as ev:
            app.do_claude(app.parser_make().parse_args([
                'claude', '--','--dangerously-skip-permissions','-p','OK']))
            argv=ev.call_args.args[1]
            self.assertEqual(argv.count('--dangerously-skip-permissions'),1)

    def test_claude_launcher_opt_out_preserves_normal_permissions(self):
        state = {'mode':'code', 'model':'Qwen/Qwen3.8-27B-FP8', 'port':18001,
                 'token':'abc', 'context':65536}
        with mock.patch.object(app,'state_read',return_value=state), \
             mock.patch.object(app,'local_api',return_value=(True,'ok')), \
             mock.patch.object(app.shutil,'which',return_value='/usr/bin/claude'), \
             mock.patch.dict(os.environ, {'COLAB_AI_SKIP_PERMISSIONS':'0'}), \
             mock.patch.object(app,'watchdog_start') as _watchdog, \
             mock.patch.object(app.os,'execvpe') as ev:
            app.do_claude(app.parser_make().parse_args(['claude', '--','-p','OK']))
            self.assertNotIn('--dangerously-skip-permissions',ev.call_args.args[1])
            self.assertEqual(ev.call_args.args[1][-2:],['-p','OK'])

    def test_old_session_context_kept_by_launcher(self):
        state={'mode':'code','model':app.PROFILES['code']['model'], 'port':18001,
               'token':'abc','context':65536}
        with mock.patch.object(app,'state_read',return_value=state), \
             mock.patch.object(app,'local_api',return_value=(True,'ok')), \
             mock.patch.object(app.shutil,'which',return_value='/usr/bin/claude'), \
             mock.patch.object(app,'watchdog_start') as _watchdog, \
             mock.patch.object(app.os,'execvpe') as ev:
            app.do_claude(app.parser_make().parse_args(['claude']))
            env=ev.call_args.args[2]
            self.assertEqual(env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'],'65536')
            self.assertEqual(env['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'],'75')

class SessionDetectionRegressionTests(TestCase):
    """Regressões de `colab status -s`: 0 pode significar inexistente."""

    def test_missing_session_with_exit_code_zero(self):
        with mock.patch.object(app, 'run', return_value=app.subprocess.CompletedProcess(
                ['colab','status'], 0, "[colab] Session 'colab-ai-code' not found.\n")) as runner:
            self.assertFalse(app.remote_session_exists(['colab','--auth=oauth2'], 'colab-ai-code'))
            runner.assert_called_once()

    def test_present_session(self):
        with mock.patch.object(app, 'run', return_value=app.subprocess.CompletedProcess(
                ['colab','status'], 0,
                '[colab-ai-code] abc-123 | Hardware: G4 | Shape: STANDARD | Variant: GPU | Status: IDLE\n')):
            self.assertTrue(app.remote_session_exists(['colab','--auth=oauth2'], 'colab-ai-code'))

    def test_unexpected_output_does_not_allocate_gpu(self):
        with mock.patch.object(app, 'run', return_value=app.subprocess.CompletedProcess(
                ['colab','status'], 0, 'Unknown output format')):
            with self.assertRaisesRegex(RuntimeError, 'Resposta inesperada'):
                app.remote_session_exists(['colab','--auth=oauth2'], 'colab-ai-code')

    def test_missing_session_is_created_before_upload(self):
        with tempfile.TemporaryDirectory() as d:
            fake_state = Path(d)
            events=[]
            statuses=[False, True]
            def exists(_base, _name):
                events.append('status')
                return statuses.pop(0)
            def runner(cmd, **kwargs):
                events.append(cmd[2])
                return app.subprocess.CompletedProcess(cmd,0, '')
            with mock.patch.object(app, 'STATE_DIR', fake_state), \
                 mock.patch.object(app, 'STATE_FILE', fake_state/'state.json'), \
                 mock.patch.object(app, 'ensure_key', return_value=fake_state/'id_ed25519'), \
                 mock.patch.object(app, 'cli_base', return_value=['/usr/bin/colab','--auth=oauth2']), \
                 mock.patch.object(app.shutil, 'which', return_value='/usr/bin/ssh'), \
                 mock.patch.object(app, 'remote_session_exists', side_effect=exists), \
                 mock.patch.object(app, 'run', side_effect=runner), \
                 mock.patch.object(app, 'launch_process', return_value=99999), \
                 mock.patch.object(app, 'tunnel_start'), \
                 mock.patch.object(app, 'watchdog_start'), \
                 mock.patch.object(app, 'wait_ready'):
                app.do_start(app.parser_make().parse_args(['start', 'code', '--port', '28379']))
                self.assertEqual(events, ['status','new','status','upload','upload','upload','exec'])
                self.assertEqual(app.state_read()['serve_pid'], 99999)
                self.assertEqual(app.state_read()['session'],'colab-ai-code')

    def test_stale_state_recreates_session(self):
        with tempfile.TemporaryDirectory() as d:
            fake_state = Path(d)
            calls=[]
            def fake_run(cmd, **kwargs):
                calls.append(cmd[2]); return app.subprocess.CompletedProcess(cmd,0,'')
            with mock.patch.object(app, 'STATE_DIR', fake_state), \
                 mock.patch.object(app, 'STATE_FILE', fake_state/'state.json'), \
                 mock.patch.object(app, 'ensure_key', return_value=fake_state/'id_ed25519'), \
                 mock.patch.object(app, 'cli_base', return_value=['/usr/bin/colab','--auth=oauth2']), \
                 mock.patch.object(app.shutil, 'which', return_value='/usr/bin/ssh'), \
                 mock.patch.object(app, 'local_api', return_value=(False,'unavailable')), \
                 mock.patch.object(app, 'remote_session_exists', side_effect=[False,True]), \
                 mock.patch.object(app, 'pid_running', return_value=False), \
                 mock.patch.object(app, 'run', side_effect=fake_run), \
                 mock.patch.object(app, 'launch_process', return_value=99999), \
                 mock.patch.object(app, 'tunnel_start'), \
                 mock.patch.object(app, 'watchdog_start'), \
                 mock.patch.object(app, 'wait_ready'):
                app.state_write({'mode':'code','model':app.PROFILES['code']['model'],
                     'session':'colab-ai-code','auth':'oauth2','gpu':'G4','port':28379,
                     'token':'secret','context':65536,'tunnel_pid':1234,'serve_pid':5678})
                app.do_start(app.parser_make().parse_args(['start','code']))
                self.assertIn('new',calls)
                self.assertEqual(app.state_read()['context'],131072)
