"""Regressão: versões publicadas de Colab CLI não aceitam `ssh -- CMD`."""
import contextlib
import importlib.util
import io
from pathlib import Path
from unittest import TestCase, mock
from subprocess import CompletedProcess

spec = importlib.util.spec_from_file_location(
    "colab_ai_logs", Path(__file__).resolve().parents[1] / "ai_colab.py"
)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class LogDownloadTests(TestCase):
    state = {"session": "colab-ai-code", "auth": "oauth2"}

    def test_logs_download_not_ssh_and_keep_only_requested_lines(self):
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            self.assertEqual(cmd[2], "download")
            self.assertEqual(cmd[4], "colab-ai-code")
            self.assertEqual(cmd[5], "/content/colab-ai-vllm.log")
            Path(cmd[6]).write_text("linha 1\nlinha 2\nlinha 3\n")
            return CompletedProcess(cmd, 0, "")

        with mock.patch.object(app, "state_read", return_value=self.state), \
             mock.patch.object(app, "cli_base", return_value=["/bin/colab", "--auth=oauth2"]), \
             mock.patch.object(app, "remote_session_exists", return_value=True), \
             mock.patch.object(app, "ensure_key") as ssh_key, \
             mock.patch.object(app, "run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()) as capture:
            app.do_logs(app.parser_make().parse_args(["logs", "--lines", "2"]))
        ssh_key.assert_not_called()
        self.assertEqual(len(seen), 1)
        self.assertEqual(capture.getvalue().splitlines()[-2:], ["linha 2", "linha 3"])

    def test_logs_never_creates_missing_session(self):
        with mock.patch.object(app, "state_read", return_value=self.state), \
             mock.patch.object(app, "cli_base", return_value=["/bin/colab", "--auth=oauth2"]), \
             mock.patch.object(app, "remote_session_exists", return_value=False), \
             mock.patch.object(app, "run") as runner:
            with self.assertRaisesRegex(RuntimeError, "não está ativa"):
                app.do_logs(app.parser_make().parse_args(["logs"]))
            runner.assert_not_called()

    def test_logs_download_error_is_reported(self):
        with mock.patch.object(app, "state_read", return_value=self.state), \
             mock.patch.object(app, "cli_base", return_value=["/bin/colab", "--auth=oauth2"]), \
             mock.patch.object(app, "remote_session_exists", return_value=True), \
             mock.patch.object(app, "run", return_value=CompletedProcess([], 1, "remote missing")):
            with self.assertRaisesRegex(RuntimeError, "remote missing"):
                app.do_logs(app.parser_make().parse_args(["logs"]))

    def test_rejects_invalid_line_count_without_remote_call(self):
        with mock.patch.object(app, "state_read", return_value=self.state), \
             mock.patch.object(app, "cli_base") as cli:
            with self.assertRaisesRegex(RuntimeError, "--lines"):
                app.do_logs(app.parser_make().parse_args(["logs", "--lines", "-5"]))
            cli.assert_not_called()
