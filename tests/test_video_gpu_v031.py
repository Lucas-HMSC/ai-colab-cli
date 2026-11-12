"""Testes v0.3.1: dimensões nativas, formato vertical, perfil GPU e regressões sem Colab."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ai_colab as cli


def imported_remote(name, location):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'remote' / location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VideoV031CliTests(unittest.TestCase):
    def test_version_and_portrait_default(self):
        self.assertEqual(cli.VERSION, '0.3.1')
        args = cli.parser_make().parse_args(['video', 'generate', '--prompt', 'uma cidade', '--output', '/tmp/a.mp4'])
        self.assertEqual(args.aspect, '9:16')
        self.assertEqual(args.quality, 'high')
        self.assertEqual(args.offload, 'auto')
        self.assertEqual(cli.video_dimensions('hunyuan15', args.aspect, None, False), (480, 848))
        self.assertEqual(cli.video_dimensions('wan22-5b', args.aspect, None, False), (704, 1280))

    def test_dimensions_custom_hunyuan_and_native_wan(self):
        self.assertEqual(cli.video_dimensions('hunyuan15', '9:16', '448x768', False), (448, 768))
        self.assertEqual(cli.video_dimensions('hunyuan15', '16:9', None, False), (848, 480))
        with self.assertRaisesRegex(RuntimeError, 'Wan'):
            cli.video_dimensions('wan22-5b', '9:16', '480x848', False)
        with self.assertRaisesRegex(RuntimeError, 'limite'):
            cli.video_dimensions('hunyuan15', '9:16', '720x1280', False)
        with self.assertRaisesRegex(RuntimeError, 'conflita'):
            cli.video_dimensions('hunyuan15', '16:9', '480x848', False)
        with self.assertRaisesRegex(RuntimeError, 'múltiplas'):
            cli.video_dimensions('hunyuan15', '9:16', '479x848', False)
        with self.assertRaisesRegex(RuntimeError, 'LARGURAxALTURA'):
            cli.video_dimensions('hunyuan15', '9:16', 'abc', False)

    def test_quality_steps_and_step_distilled(self):
        for quality, steps in [('draft', 20), ('standard', 32), ('high', 50)]:
            self.assertEqual(cli.video_steps('wan22-5b', False, None, quality), steps)
        self.assertEqual(cli.video_steps('wan22-5b', False, 12, 'high'), 12)
        self.assertEqual(cli.video_steps('hunyuan15', True, None, 'draft'), 12)
        with self.assertRaisesRegex(RuntimeError, '12'):
            cli.video_steps('hunyuan15', True, 32, 'high')

    def test_new_gpu_command_does_not_open_ssh(self):
        args = cli.parser_make().parse_args(['gpu','status','--json','--deep'])
        state={'token':'test', 'port':18001}
        payload={'phase':'modelo pronto','metrics':{'gpu':{'name':'NVIDIA L4','utilization_pct':84,'memory_total_mib':23000,'memory_used_mib':12000, 'temperature_c':62},
                                                  'torch':{'cuda_available':True,'device':'NVIDIA L4'}}}
        with mock.patch.object(cli,'state_read',return_value=state), \
             mock.patch.object(cli,'runtime_status',return_value=(True,payload)) as status, \
             mock.patch.object(cli,'run') as subproc, \
             mock.patch('builtins.print') as printing:
            cli.do_gpu_status(args)
        status.assert_called_once_with(state, timeout=15, metrics=True, deep=True)
        subproc.assert_not_called()
        self.assertTrue(printing.called)

    def test_gpu_status_without_session(self):
        args=cli.parser_make().parse_args(['gpu','status'])
        with mock.patch.object(cli,'state_read',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'Nenhuma sessão'):
                cli.do_gpu_status(args)

    def test_video_request_has_dimensions_and_no_gpu_provision(self):
        with tempfile.TemporaryDirectory() as temp:
            args = cli.parser_make().parse_args(['video','generate','--prompt','sol',
                '--output',str(Path(temp)/'v.mp4'),'--quality','draft','--size','448x768'])
            state = {'mode':'video','port':18001,'token':'token','video_variant':'hunyuan15',
                     'model':'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v'}
            def request(_, path, data=None, timeout=30):
                if data is not None:
                    self.assertEqual(data['aspect'], '9:16')
                    self.assertEqual((data['width'],data['height']), (448,768))
                    self.assertEqual(data['steps'], 20)
                    self.assertEqual(data['offload'], 'auto')
                    return {'id':'a'*32,'status':'queued'}
                return {'status':'completed'}
            with mock.patch.object(cli,'state_read',return_value=state), \
                 mock.patch.object(cli,'_video_request',side_effect=request), \
                 mock.patch.object(cli,'do_video_fetch'), \
                 mock.patch.object(cli.time,'sleep',return_value=None), \
                 mock.patch.object(cli,'run') as subprocess_run:
                cli.do_video_generate(args)
                subprocess_run.assert_not_called()


class RemoteV031Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        config = Path(self.tmp.name)/'config.json'
        config.write_text(json.dumps({'model':'test','video_variant':'hunyuan15'}))
        env = mock.patch.dict(os.environ, {'COLAB_AI_VIDEO_CONFIG':str(config),
                                          'COLAB_AI_VIDEO_ROOT':str(Path(self.tmp.name)/'jobs')})
        env.start()
        self.addCleanup(env.stop)
        self.api = imported_remote('test_v031_api', 'video_api.py')

    def test_remote_request_defaults(self):
        data = self.api.check_request({'prompt':'pôr do sol'})
        self.assertEqual(data['aspect'],'9:16')
        self.assertEqual((data['width'],data['height']),(480,848))
        self.assertEqual(data['offload'],'auto')
        self.assertEqual(data['image_fit'],'cover')
        with self.assertRaises(ValueError):
            self.api.check_request({'prompt':'a','width':704,'height':1280,'aspect':'9:16'})
        with self.assertRaisesRegex(ValueError,'Offload'):
            self.api.check_request({'prompt':'a','offload':'unsafe'})

    def test_remote_wan_native_size(self):
        self.api.VARIANT='wan22-5b'
        data = self.api.check_request({'prompt':'a'})
        self.assertEqual((data['width'],data['height']),(704,1280))
        with self.assertRaisesRegex(ValueError,'Wan'):
            self.api.check_request({'prompt':'a','width':480,'height':848})

    def test_job_status_gpu_peak(self):
        output = Path(self.tmp.name)/'output.mp4'
        log = Path(self.tmp.name)/'log.txt'
        log.write_text('Processo iniciado\n')
        job={'id':'a'*32,'status':'running','model':'test','created_at':1000,'updated_at':1000,
             'output':output,'log':log, 'generation':{'width':480, 'height':848},
             'gpu':{'utilization_pct':80,'memory_used_mib':15000}, 'gpu_samples':7,'peak_vram_mib':18000}
        with mock.patch.object(self.api.time,'time',return_value=1010):
            result = self.api._public_status(job)
        self.assertEqual(result['elapsed_seconds'],10)
        self.assertEqual(result['peak_vram_mib'],18000)
        self.assertEqual(result['generation']['height'],848)

    def test_backend_offload_choice_and_input_image_fit(self):
        backend = imported_remote('test_v031_generate', 'video_generate.py')
        self.assertTrue(backend.offload_choice('auto',24))
        self.assertFalse(backend.offload_choice('auto',96))
        self.assertTrue(backend.offload_choice('on',96))
        with self.assertRaisesRegex(RuntimeError,'80 GiB'):
            backend.offload_choice('off',24)
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow não instalado localmente')
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder)/'input.png'
            dest=Path(folder)/'output.png'
            Image.new('RGB',(640,360),color=(0,128,255)).save(src)
            backend.image_prepare(src,480,848,'cover',dest)
            with Image.open(dest) as out:
                self.assertEqual(out.size,(480,848))
            backend.image_prepare(src,480,848,'contain',dest)
            with Image.open(dest) as out:
                self.assertEqual(out.size,(480,848))
            self.assertTrue(src.exists())

    def test_proxy_metrics_no_second_ssh(self):
        proxy = imported_remote('test_v031_proxy', 'proxy.py')
        def smi(cmd, **kwargs):
            if '--query-compute-apps=pid,process_name,used_gpu_memory' in cmd:
                return mock.Mock(returncode=0,stdout='1000, python, 12345\n')
            return mock.Mock(returncode=0, stdout='NVIDIA L4, 83, 23034, 12000, 11034, 62, 560.35\n')
        with mock.patch.object(proxy.subprocess,'run',side_effect=smi), \
             mock.patch.object(proxy.shutil,'disk_usage',return_value=mock.Mock(total=100,used=40,free=60)):
            metrics=proxy.collect_metrics(deep=False)
        self.assertEqual(metrics['gpu']['name'],'NVIDIA L4')
        self.assertEqual(metrics['gpu_processes'][0]['pid'],1000)
        self.assertEqual(metrics['gpu']['driver_version'],'560.35')


if __name__ == '__main__':
    unittest.main()
