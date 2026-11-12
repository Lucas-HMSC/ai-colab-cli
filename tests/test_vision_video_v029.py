"""v0.2.9 — orientação global do launcher e base sem GPU do módulo vídeo."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ai_colab_v029', ROOT / 'ai_colab.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class VisionGuidanceTests(unittest.TestCase):
    def test_default_uses_append_prompt_and_explicit_read(self):
        result = app.apply_global_vision_guidance(['--dangerously-skip-permissions', '-p', 'oi'], {})
        self.assertEqual(result.count('--append-system-prompt'), 1)
        self.assertEqual(result[result.index('--append-system-prompt') + 1], app.GLOBAL_VISION_GUIDANCE)
        self.assertIn('Read', app.GLOBAL_VISION_GUIDANCE)
        self.assertIn('API keys', app.GLOBAL_VISION_GUIDANCE)
        self.assertEqual(result[-2:], ['-p', 'oi'])

    def test_opt_out(self):
        base = ['--dangerously-skip-permissions', '-p', 'oi']
        for value in ('0', 'FALSE', 'no', 'off'):
            self.assertEqual(app.apply_global_vision_guidance(base, {'COLAB_AI_VISION_GUIDANCE': value}), base)

    def test_existing_prompt_is_merged_not_duplicated(self):
        result = app.apply_global_vision_guidance(['--append-system-prompt', 'Use teste', '-p', 'oi'], {})
        self.assertEqual(result.count('--append-system-prompt'), 1)
        self.assertIn('Use teste', result[1])
        self.assertIn(app.GLOBAL_VISION_GUIDANCE, result[1])
        self.assertEqual(result[-2:], ['-p', 'oi'])

    def test_existing_equals_syntax_is_merged(self):
        result = app.apply_global_vision_guidance(['--append-system-prompt=Especial'], {})
        self.assertEqual(len(result), 1)
        self.assertIn('Especial', result[0])
        self.assertIn(app.GLOBAL_VISION_GUIDANCE, result[0])

    def test_invalid_existing_prompt_not_silently_replaced(self):
        self.assertEqual(app.apply_global_vision_guidance(['--append-system-prompt'], {}),
                         ['--append-system-prompt'])

    def test_full_launcher_env_and_arg_precedence(self):
        state = {'mode':'code', 'model':'Qwen/Qwen3.8-27B-FP8', 'port':18001,
                 'token':'secret', 'context':262144}
        with mock.patch.object(app, 'state_read', return_value=state), \
             mock.patch.object(app, 'local_api', return_value=(True, 'ok')), \
             mock.patch.object(app.shutil, 'which', return_value='/opt/bin/claude'), \
             mock.patch.object(app, 'watchdog_start'), \
             mock.patch.dict(os.environ, {'COLAB_AI_VISION_GUIDANCE':'1', 'COLAB_AI_SKIP_PERMISSIONS':'1'}), \
             mock.patch.object(app.os, 'execvpe') as execvpe:
            app.do_claude(app.parser_make().parse_args(['claude', '--', '-p', 'catalogue']))
            executable, argv, env = execvpe.call_args.args
            self.assertEqual(executable, '/opt/bin/claude')
            self.assertEqual(argv[0:3], ['/opt/bin/claude', '--model', state['model']])
            self.assertEqual(argv.count('--append-system-prompt'), 1)
            self.assertEqual(argv.count('--dangerously-skip-permissions'), 1)
            self.assertEqual(env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'], '262144')
            self.assertEqual(argv[-2:], ['-p', 'catalogue'])
            self.assertIn('Read', argv[argv.index('--append-system-prompt') + 1])

    def test_full_launcher_opt_out(self):
        state = {'mode':'code', 'model':'Qwen/Qwen3.8-27B-FP8', 'port':18001,
                 'token':'secret', 'context':262144}
        with mock.patch.object(app, 'state_read', return_value=state), \
             mock.patch.object(app, 'local_api', return_value=(True, 'ok')), \
             mock.patch.object(app.shutil, 'which', return_value='/opt/bin/claude'), \
             mock.patch.object(app, 'watchdog_start'), \
             mock.patch.dict(os.environ, {'COLAB_AI_VISION_GUIDANCE':'0'}), \
             mock.patch.object(app.os, 'execvpe') as execvpe:
            app.do_claude(app.parser_make().parse_args(['claude']))
            self.assertNotIn('--append-system-prompt', execvpe.call_args.args[1])


class VideoCatalogTests(unittest.TestCase):
    def test_candidates_cover_major_workflows(self):
        self.assertIn('hunyuan15', app.VIDEO_MODELS)
        self.assertIn('wan22-5b', app.VIDEO_MODELS)
        self.assertIn('wan22-14b', app.VIDEO_MODELS)
        self.assertIn('ltx25', app.VIDEO_MODELS)
        self.assertIn('audio-video', app.VIDEO_MODELS['ltx25']['tasks'])
        self.assertEqual(app.VERSION, '0.3.0')

    def test_models_without_gpu(self):
        args = app.parser_make().parse_args(['video', 'models'])
        text = io.StringIO()
        with mock.patch.object(app, 'run') as run, \
             mock.patch.object(app, 'ensure_key') as key, \
             contextlib.redirect_stdout(text):
            args.func(args)
        self.assertIn('implementados', text.getvalue())
        self.assertIn('LTX-2.5', text.getvalue())
        run.assert_not_called()
        key.assert_not_called()

    def test_json_catalog_schema(self):
        args = app.parser_make().parse_args(['video','models','--json'])
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            args.func(args)
        result = json.loads(text.getvalue())
        self.assertEqual(result['stage'], 'hunyuan15_wan22-5b_available')
        self.assertIn('gpu', result['models']['hunyuan15'])

    def test_start_video_available_but_no_provision_in_parser(self):
        args = app.parser_make().parse_args(['start','video','--gpu','G4', '--video-model','wan22-5b'])
        self.assertEqual(args.mode, 'video')
        self.assertEqual(args.video_model, 'wan22-5b')
        self.assertEqual(args.gpu, 'G4')
