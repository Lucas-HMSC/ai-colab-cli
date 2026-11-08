"""Regressões locais da arquitetura v0.1.6: não acessar GPU real."""
import importlib.util
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
from unittest import TestCase, mock

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('colab_ai_v016',root/'ai_colab.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
spec = importlib.util.spec_from_file_location('proxy_v016',root/'remote/proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)

class ExecTimeoutTests(TestCase):
    def test_start_exec_is_short_and_server_has_explicit_timeout(self):
        commands=[]; spawned=[]
        def fake_run(cmd,**kwargs):
            commands.append(cmd)
            return CompletedProcess(cmd,0,'')
        def fake_launch(cmd,logname):
            spawned.append((cmd,logname)); return 83922
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(app,'STATE_DIR',Path(d)), \
             mock.patch.object(app,'STATE_FILE',Path(d)/'state.json'), \
             mock.patch.object(app,'ensure_key',return_value=Path(d)/'ssh'/'key'), \
             mock.patch.object(app,'cli_base',return_value=['colab','--auth=oauth2']), \
             mock.patch.object(app.shutil,'which',return_value='/usr/bin/ssh'), \
             mock.patch.object(app,'remote_session_exists',side_effect=[False,True]), \
             mock.patch.object(app,'run',side_effect=fake_run), \
             mock.patch.object(app,'launch_process',side_effect=fake_launch), \
             mock.patch.object(app,'tunnel_start'), \
             mock.patch.object(app,'watchdog_start'), \
             mock.patch.object(app,'wait_ready'):
            app.do_start(app.parser_make().parse_args(['start','code','--port','28565']))
        remote_exec=[c for c in commands if len(c)>2 and c[2]=='exec']
        self.assertEqual(len(remote_exec),1)
        self.assertIn('180',remote_exec[0])
        self.assertTrue(remote_exec[0][-1].endswith('remote/launch.py'))
        self.assertFalse(any('bootstrap.py' in ' '.join(c) for c in remote_exec))
        self.assertEqual(len(spawned),1)
        self.assertIn('14400',spawned[0][0])
        self.assertTrue(spawned[0][0][-1].endswith('remote/serve.py'))
        self.assertEqual(spawned[0][1],'kernel-serve.log')
        self.assertEqual(len([c for c in commands if c[2]=='upload']),3)

    def test_bootstrap_death_detected_without_long_wait(self):
        state={'tunnel_pid':1,'serve_pid':2,'model':'model','token':'abc'}
        snapshot={'vllm_alive':None,'bootstrap_alive':False,
                  'bootstrap_log_tail':'pip failed: exit code 1'}
        with mock.patch.object(app,'pid_running',return_value=True), \
             mock.patch.object(app,'runtime_status',return_value=(True,snapshot)):
            with self.assertRaisesRegex(RuntimeError,'pip failed'):
                app.wait_ready(state,seconds=500,interval=0)

    def test_short_exec_timeout_explains_no_new_gpu(self):
        # Texto de erro explicativo regressão sem testes remotos
        import inspect
        src=inspect.getsource(app.do_start)
        self.assertIn('Não inicialize outra GPU',src)
        self.assertIn('ai-colab logs --source bootstrap',src)

    def test_bootstrap_phase_progress_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            bp=d/'bootstrap.pid';bp.write_text(str(__import__('os').getpid()))
            bl=d/'bootstrap.log';bl.write_text('Installing vLLM, please wait')
            with mock.patch.object(proxy,'BOOT_PID_FILE',bp), \
                 mock.patch.object(proxy,'BOOT_FILE',bl), \
                 mock.patch.object(proxy,'PID_FILE',d/'vllm_missing.pid'), \
                 mock.patch.object(proxy,'LOG_FILE',d/'vllm_missing.log'), \
                 mock.patch.object(proxy,'urlopen',side_effect=OSError('no server')):
                status=proxy.vllm_status()
            self.assertEqual(status['bootstrap_alive'],True)
            self.assertIn('preparando dependências',status['phase'])
            self.assertIn('Installing vLLM',status['bootstrap_log_tail'])

    def test_bootstrap_log_command_available(self):
        parsed=app.parser_make().parse_args(['logs','--source','bootstrap','--lines','50'])
        self.assertEqual(parsed.source,'bootstrap')
        self.assertEqual(parsed.lines,50)
