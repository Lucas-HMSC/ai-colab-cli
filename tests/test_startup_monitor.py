"""Testes locais de regressão: sem consumir GPU nem conta Colab."""
import importlib.util
import json
from pathlib import Path
import threading
import time
import unittest
from unittest import mock
from urllib import request, error
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

app = load(ROOT/'ai_colab.py', 'ai_colab_monitor_test')
proxy = load(ROOT/'remote'/'proxy.py', 'remote_proxy_test')


class WaitTests(unittest.TestCase):
    STATE = {'model': 'Qwen/Qwen3.8-27B-FP8', 'tunnel_pid': 101, 'serve_pid': 102,
             'auth':'oauth2', 'session':'colab-ai-code', 'port':18001, 'token':'test'}

    def test_vllm_dead_fails_without_waiting_15_minutes(self):
        snapshot = {'vllm_alive': False, 'phase':'processo encerrado', 'log_tail':'CUDA out of memory'}
        with mock.patch.object(app, 'pid_running', return_value=True), \
             mock.patch.object(app, 'runtime_status', return_value=(True, snapshot)), \
             mock.patch.object(app, 'local_api') as api:
            with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
                app.wait_ready(self.STATE, seconds=900, interval=0)
            api.assert_not_called()

    def test_progress_polls_until_model_ready(self):
        snapshots = [
            (True, {'vllm_alive':True,'vllm_ready':False,'phase':'download dos pesos / cache','log_tail':'Downloading checkpoint 12%'}),
            (True, {'vllm_alive':True,'vllm_ready':True,'phase':'modelo pronto','log_tail':'Application startup complete'}),
        ]
        with mock.patch.object(app, 'pid_running', return_value=True), \
             mock.patch.object(app, 'runtime_status', side_effect=snapshots), \
             mock.patch.object(app, 'local_api', return_value=(True,self.STATE['model'])), \
             mock.patch.object(app.time, 'sleep'):
            app.wait_ready(self.STATE, seconds=50, interval=0.01)

    def test_continue_wait_does_not_call_gpu_creation_or_bootstrap(self):
        with mock.patch.object(app, 'state_read', return_value=self.STATE), \
             mock.patch.object(app,'cli_base',return_value=['colab','--auth=oauth2']), \
             mock.patch.object(app,'remote_session_exists',return_value=True), \
             mock.patch.object(app,'pid_running',return_value=True), \
             mock.patch.object(app,'tunnel_start') as tunnel, \
             mock.patch.object(app,'wait_ready') as ready, \
             mock.patch.object(app,'run') as run:
            app.do_wait(app.parser_make().parse_args(['wait','--seconds','300']))
            tunnel.assert_called_once()
            ready.assert_called_once_with(self.STATE, 300)
            run.assert_not_called()

    def test_wait_never_creates_gpu_if_session_disappeared(self):
        with mock.patch.object(app,'state_read',return_value=self.STATE), \
             mock.patch.object(app,'cli_base',return_value=['colab','--auth=oauth2']), \
             mock.patch.object(app,'remote_session_exists',return_value=False), \
             mock.patch.object(app,'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'não cria nova VM'):
                app.do_wait(app.parser_make().parse_args(['wait']))
            run.assert_not_called()

    def test_tunnel_is_preserved_during_model_loading(self):
        with mock.patch.object(app, 'pid_running',return_value=True), \
             mock.patch.object(app,'runtime_status',return_value=(True, {'vllm_ready':False})), \
             mock.patch.object(app,'kill_group') as kill, \
             mock.patch.object(app,'state_read',return_value=self.STATE.copy()):
            state = app.tunnel_start(self.STATE.copy())
            self.assertEqual(state['tunnel_pid'],101)
            kill.assert_not_called()

    def test_wait_cli_exists(self):
        self.assertEqual(app.parser_make().parse_args(['wait']).seconds,1800)
        self.assertEqual(app.VERSION,'0.2.4')


class ProxyStatusTests(unittest.TestCase):
    def test_status_requires_token_and_returns_process_info(self):
        server = ThreadingHTTPServer(('127.0.0.1',0), proxy.make_handler('a-test-token'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_address[1]}/_colab_ai/status'
        try:
            with mock.patch.object(proxy,'vllm_status',return_value={'phase':'download dos pesos','vllm_alive':True,'vllm_ready':False}):
                with self.assertRaises(error.HTTPError) as ctx:
                    request.urlopen(url, timeout=2)
                self.assertEqual(ctx.exception.code,401)
                req = request.Request(url, headers={'X-Api-Key':'a-test-token'})
                with request.urlopen(req, timeout=2) as resp:
                    self.assertEqual(resp.status,200)
                    self.assertEqual(json.load(resp)['phase'],'download dos pesos')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_process_alive_missing_pid_is_unknown(self):
        self.assertIsNone(proxy.process_alive(None))


if __name__ == '__main__':
    unittest.main()
