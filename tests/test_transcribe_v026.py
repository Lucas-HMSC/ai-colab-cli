"""ASR v0.2.6: valida rotas, formatos, registros, segurança e streaming."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

app=load('asr_cli',ROOT/'ai_colab.py')
server=load('asr_worker',ROOT/'remote/transcribe_api.py') if False else None
# transcribe_api depends on Colab config at import-time: import with temp stub.

class CLITranscribeTests(unittest.TestCase):
    def test_profiles_present_and_voice_preserved(self):
        self.assertEqual(set(app.ASR_MODELS), {'whisper','qwen'})
        self.assertNotIn('transcribe', app.COMING)
        self.assertIn('voxcpm2', app.ACTIVE_VOICE_MODELS)
        self.assertNotIn('higgs3', app.ACTIVE_VOICE_MODELS)

    def test_asr_parse(self):
        args=app.parser_make().parse_args(['start','transcribe','--asr-model','qwen'])
        self.assertEqual(args.asr_model,'qwen')
        t=app.parser_make().parse_args(['transcribe','--input','som.wav','--format','all'])
        self.assertEqual((t.file,t.format),('som.wav','all'))
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                app.parser_make().parse_args(['start','transcribe','--asr-model','higgs3'])

    def test_output_naming_and_atomic_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audio.wav'
            outs=app._formats_from_options(p,None,'all')
            self.assertEqual({x.suffix for x in outs.values()},{'.txt','.srt','.json'})
            self.assertEqual(app._srt_clock(3600.325),'01:00:00,325')
            data={'text':'Olá mundo.', 'segments':[{'start':1.25,'end':2.3,'text':'Olá mundo.'}]}
            subtitle=app._render_transcription(data,'srt')
            self.assertIn('00:00:01,250 --> 00:00:02,300',subtitle)
            self.assertIn('Olá mundo.',subtitle)
            with self.assertRaisesRegex(RuntimeError,'timestamps reais'):
                app._render_transcription({'text':'oi','segments':[]},'srt')
            with self.assertRaisesRegex(RuntimeError,'extensão'):
                app._formats_from_options(p,'output.json','txt')

    def test_asr_start_does_not_provision_on_precondition_failure(self):
        with mock.patch.object(app,'ensure_key',side_effect=RuntimeError('sem SSH')):
            with mock.patch.object(app,'run') as r:
                with self.assertRaisesRegex(RuntimeError,'sem SSH'):
                    app.do_start(app.parser_make().parse_args(['start','transcribe','--asr-model','whisper']))
                r.assert_not_called()

    def test_transcribe_refuses_unknown_file_before_connection(self):
        state={'mode':'transcribe','model':app.ASR_MODELS['whisper'], 'port':18001,'token':'abc'}
        with mock.patch.object(app,'state_read',return_value=state):
            with self.assertRaisesRegex(RuntimeError,'não encontrado'):
                app.do_transcribe(app.parser_make().parse_args(['transcribe','--file','/tmp/naoexiste.m4a']))

    def test_streams_file_and_writes_three_formats(self):
        sent=[]
        class Remote(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server_seen_path=self.path
                assert self.path=='/_colab_ai/transcribe'
                assert self.headers.get('X-Api-Key')=='abc'
                assert self.headers.get('X-Colab-Ai-Extension')=='.wav'
                size=int(self.headers.get('Content-Length'))
                data=self.rfile.read(size)
                sent.append(data)
                payload={'model':'test','engine':'whisper','language':'pt','duration_seconds':2.0,
                         'text':'Olá.', 'segments':[{'start':0.0,'end':1.2,'text':'Olá.'}]}
                raw=json.dumps(payload).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*a):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Remote)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                infile=Path(d)/'teste.wav';infile.write_bytes(b'RIFF'+b'123'*4000)
                state={'mode':'transcribe','model':'test','port':server.server_port,'token':'abc'}
                a=app.parser_make().parse_args(['transcribe','--file',str(infile),'--format','all'])
                with mock.patch.object(app,'state_read',return_value=state):
                    with contextlib.redirect_stdout(io.StringIO()):app.do_transcribe(a)
                self.assertEqual(len(sent),1)
                self.assertEqual(sent[0],infile.read_bytes())
                self.assertEqual((Path(d)/'teste-transcricao.txt').read_text().strip(),'Olá.')
                self.assertIn('00:00:00,000',(Path(d)/'teste-transcricao.srt').read_text())
                self.assertEqual(json.loads((Path(d)/'teste-transcricao.json').read_text())['language'],'pt')
                with mock.patch.object(app,'state_read',return_value=state):
                    with self.assertRaisesRegex(RuntimeError,'já existe'):
                        app.do_transcribe(a)
                self.assertEqual(len(sent),1)
        finally:
            server.shutdown();server.server_close()

    def test_asr_proxy_access_control(self):
        proxy=load('asr_proxy_v026',ROOT/'remote/proxy.py')
        proxy.ASR_MODE=True
        handler=proxy.make_handler('token')
        srv=ThreadingHTTPServer(('127.0.0.1',0),handler)
        worker=threading.Thread(target=srv.serve_forever,daemon=True);worker.start()
        import urllib.request, urllib.error
        try:
            request=urllib.request.Request(f'http://127.0.0.1:{srv.server_port}/_colab_ai/transcribe',
                                           data=b'abc',method='POST',headers={'Content-Length':'3'})
            with self.assertRaises(urllib.error.HTTPError) as captured:
                urllib.request.urlopen(request,timeout=2)
            self.assertEqual(captured.exception.code,401)
        finally:
            srv.shutdown();srv.server_close()

if __name__=='__main__':unittest.main()

class RemoteASRLogicTests(unittest.TestCase):
    def setUp(self):
        # Import isolado com configuração simulada: não necessita GPU nem Colab.
        from pathlib import Path
        original=Path.read_text
        def fake_read(path,*args,**kwargs):
            if str(path)=='/content/colab-ai-config.json':
                return json.dumps({'asr_variant':'qwen','model':'Qwen/Qwen3-ASR-1.7B'})
            return original(path,*args,**kwargs)
        with mock.patch.object(Path,'read_text',new=fake_read):
            self.worker=load('worker_logic_026',ROOT/'remote/transcribe_api.py')

    def test_align_groups_only_real_timestamps(self):
        from types import SimpleNamespace as E
        items=[E(text='Olá',start_time=1.0,end_time=1.3),
               E(text='mundo.',start_time=1.3,end_time=1.7),
               E(text='Hoje',start_time=4,end_time=4.2)]
        result=self.worker.align_to_segments(items)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['start'],1.0)
        self.assertEqual(result[-1]['end'],4.2)

    def test_qwen_infer_uses_official_transcribe_signature(self):
        from types import SimpleNamespace as E
        items=[E(text='Olá',start_time=0.1,end_time=0.3),E(text='mundo.',start_time=0.3,end_time=1.0)]
        response=E(text='Olá mundo.',language='Portuguese',time_stamps=E(items=items))
        model=mock.Mock(transcribe=mock.Mock(return_value=[response]))
        payload=self.worker.infer(model,Path('/tmp/fake.wav'),1.1,'pt')
        self.assertEqual(payload['engine'],'qwen')
        self.assertEqual(payload['text'],'Olá mundo.')
        self.assertTrue(payload['timestamps'])
        model.transcribe.assert_called_once_with(audio='/tmp/fake.wav',language='Portuguese',return_time_stamps=True)

    def test_whisper_infer_segments(self):
        from types import SimpleNamespace as E
        self.worker.VARIANT='whisper'
        model=mock.Mock()
        model.transcribe.return_value=(iter([E(text='Bom dia',start=0,end=0.85)]),E(language='pt'))
        payload=self.worker.infer(model,Path('/tmp/fake.mp3'),1.0,'pt')
        self.assertEqual(payload['segments'][0]['text'],'Bom dia')
        self.assertEqual(payload['language'],'pt')
        self.assertEqual(payload['text'],'Bom dia')
