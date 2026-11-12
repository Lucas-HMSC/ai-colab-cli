"""Testes de transporte e fluxo Video v0.3.0, sem consumir GPU."""
import base64
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import ai_colab as cli


class VideoCliTests(unittest.TestCase):
    def test_parser_and_catalog(self):
        arg = cli.parser_make().parse_args(['start', 'video', '--video-model', 'wan22-5b'])
        self.assertEqual(arg.video_model, 'wan22-5b')
        self.assertNotIn('video', cli.COMING)
        for cmd in ('models', 'generate', 'jobs', 'status', 'fetch', 'cancel'):
            if cmd == 'generate':
                cmdline = ['video', cmd, '--prompt', 'test', '--output', 'x.mp4']
            elif cmd in ('status', 'cancel'):
                cmdline = ['video', cmd, 'a'*32]
            elif cmd == 'fetch':
                cmdline = ['video', cmd, 'a'*32, '--output', 'x.mp4']
            else:
                cmdline = ['video', cmd]
            self.assertEqual(cli.parser_make().parse_args(cmdline).action, cmd)

    def test_start_invalid_model_no_gpu(self):
        args = cli.parser_make().parse_args(['start','video','--video-model','hunyuan15','--gpu','T4'])
        with mock.patch.object(cli,'run') as no_gpu, mock.patch.object(cli,'ensure_key') as no_key:
            with self.assertRaisesRegex(RuntimeError, 'não T4'):
                cli.do_start(args)
            no_gpu.assert_not_called()
            no_key.assert_not_called()

    def test_generate_rejects_bad_frames_before_transfer(self):
        args = cli.parser_make().parse_args(['video','generate','--prompt','oi',
                  '--output','/tmp/test.mp4','--frames','50'])
        state = {'mode':'video','port':18001,'token':'secret','video_variant':'wan22-5b','model':'Wan-AI/Wan2.2-TI2V-5B'}
        with mock.patch.object(cli, 'state_read', return_value=state):
            with self.assertRaisesRegex(RuntimeError, '4n'):
                cli.do_video_generate(args)

    def test_generation_send_and_fetch(self):
        with tempfile.TemporaryDirectory() as temp:
            outfile = str(Path(temp) / 'video.mp4')
            args = cli.parser_make().parse_args(['video','generate','--prompt','teste de vídeo',
                     '--output',outfile,'--frames','49'])
            state = {'mode':'video','port':18001,'token':'tok','video_variant':'hunyuan15',
                     'model':'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v'}
            request_ids = []
            def fake_request(st, route, data=None, timeout=30):
                self.assertEqual(st, state)
                if data is not None:
                    self.assertEqual(data['steps'], 50)
                    self.assertIsNone(data['image_b64'])
                    return {'id':'f'*32,'status':'queued'}
                return {'status':'completed'}
            def fake_fetch(a):
                request_ids.append(a.job_id)
            with mock.patch.object(cli,'state_read',return_value=state), \
                 mock.patch.object(cli,'_video_request',side_effect=fake_request), \
                 mock.patch.object(cli,'do_video_fetch',side_effect=fake_fetch), \
                 mock.patch.object(cli.time,'sleep',return_value=None):
                cli.do_video_generate(args)
            self.assertEqual(request_ids,['f'*32])


class RemoteVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        config = Path(self.tmp.name)/'config.json'
        config.write_text(json.dumps({'model': 'Wan-AI/Wan2.2-TI2V-5B', 'video_variant':'wan22-5b'}))
        patch = mock.patch.dict(os.environ, {'COLAB_AI_VIDEO_CONFIG': str(config),
                                            'COLAB_AI_VIDEO_ROOT': str(Path(self.tmp.name)/'jobs')})
        patch.start()
        self.addCleanup(patch.stop)
        path = Path(__file__).resolve().parents[1]/'remote'/'video_api.py'
        spec = importlib.util.spec_from_file_location('video_api_test_v030', path)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)

    def test_validation(self):
        p = {'prompt':'Teste', 'frames':49, 'steps':50}
        valid = self.api.check_request(p)
        self.assertEqual(valid['frames'],49)
        for key, val in [('frames', 50), ('steps',70), ('aspect','4:3'), ('seed',-1), ('prompt','')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.check_request({**p, key:val})
        with self.assertRaises(ValueError):
            self.api.check_request({**p, 'image_b64':base64.b64encode(b'NOT AN IMAGE').decode()})
        valid_image = self.api.check_request({**p,'image_b64':base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'X'*10).decode()})
        self.assertEqual(valid_image['extension'], '.png')

    def test_inmemory_http_jobs_and_download(self):
        from http.server import ThreadingHTTPServer
        server = ThreadingHTTPServer(('127.0.0.1',0), self.api.handler_factory())
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port=server.server_address[1]
        def fake_run(job, params):
            job['output'].write_bytes(b'\x00\x00\x00\x18ftypisom'+b'X'*200)
            job['status']='completed'
        with mock.patch.object(self.api,'_run',side_effect=fake_run):
            connection = http.client.HTTPConnection('127.0.0.1',port,timeout=3)
            data=json.dumps({'prompt':'teste','frames':49,'steps':50}).encode()
            connection.request('POST','/_colab_ai/video/jobs',data,{'Content-Type':'application/json'})
            resp=connection.getresponse()
            self.assertEqual(resp.status,202)
            job=json.loads(resp.read())
            jid=job['id']
            # Jobs são assíncronos: a resposta 202 não garante término imediato.
            deadline = time.monotonic() + 3
            while True:
                connection.request('GET',f'/_colab_ai/video/jobs/{jid}')
                r=connection.getresponse()
                status = json.loads(r.read())['status']
                if status == 'completed' or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            self.assertEqual(status, 'completed')
            connection.request('GET',f'/_colab_ai/video/jobs/{jid}/file')
            r=connection.getresponse()
            self.assertEqual(r.status,200)
            self.assertEqual(r.headers.get('Content-Type'),'video/mp4')
            self.assertIn(b'ftyp',r.read(16))
            r.read()
            connection.request('GET','/_colab_ai/video/jobs/../../etc/passwd')
            r=connection.getresponse()
            self.assertEqual(r.status,404)
            r.read()
            connection.close()

    def test_hunyuan_12_steps_only_i2v(self):
        self.api.VARIANT = 'hunyuan15'
        image = base64.b64encode(b'\xff\xd8\xff'+b'JPEG').decode()
        good = self.api.check_request({'prompt':'test','image_b64': image})
        self.assertEqual(good['steps'],12)
        with self.assertRaisesRegex(ValueError,'12 passos'):
            self.api.check_request({'prompt':'test','image_b64':image,'steps':50})


if __name__ == '__main__':
    unittest.main()
