import importlib.util
import io
import json
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
