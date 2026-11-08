"""Regressões para falha de ssh-keygen -y e preflight antes da GPU."""
import importlib.util
from pathlib import Path
from unittest import TestCase, mock
import tempfile
import os
from subprocess import CompletedProcess

spec=importlib.util.spec_from_file_location("colab_ai_ssh_reg",Path(__file__).resolve().parents[1]/"ai_colab.py")
app=importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

class SshIdentityTests(TestCase):
    def test_dedicated_key_and_permissions_without_using_personal_ssh(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def fake_keygen(cmd, **kwargs):
                calls.append(cmd)
                if '-t' in cmd:
                    Path(cmd[-1]).write_text('simulated key')
                    Path(cmd[-1]+'.pub').write_text('ssh-ed25519 AAAATEST colab-ai-cli\n')
                    return CompletedProcess(cmd,0,'','')
                return CompletedProcess(cmd,0,'ssh-ed25519 AAAATEST colab-ai-cli\n','')
            with mock.patch.object(app,"STATE_DIR",Path(d)), \
                 mock.patch.object(app.shutil,"which",return_value="/mock/ssh-keygen"), \
                 mock.patch.object(app.subprocess,"run",side_effect=fake_keygen):
                key=app.ensure_key()
                self.assertEqual(key,Path(d)/"ssh"/"id_ed25519")
                self.assertTrue(key.with_suffix(".pub").exists())
                self.assertEqual(key.stat().st_mode & 0o777,0o600)
                self.assertEqual(key.parent.stat().st_mode & 0o777,0o700)
                self.assertEqual(app.ensure_key(),key)
                self.assertEqual(sum('-t' in c for c in calls),1)

    def test_existing_damaged_dedicated_key_refused_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            identity=Path(d)/"ssh"/"id_ed25519"
            identity.parent.mkdir()
            identity.write_text("bad key")
            with mock.patch.object(app,"STATE_DIR",Path(d)), \
                 mock.patch.object(app.shutil,"which",return_value="/mock/ssh-keygen"), \
                 mock.patch.object(app.subprocess,"run",return_value=CompletedProcess([],255,'','Invalid format')):
                with self.assertRaisesRegex(RuntimeError,"inválida"):
                    app.ensure_key()
            self.assertEqual(identity.read_text(),"bad key")

    def test_ssh_check_no_gpu(self):
        with mock.patch.object(app,"ensure_key",return_value=Path("/tmp/dedicated")), \
             mock.patch.object(app,"run") as runner:
            self.assertEqual(app.main(["ssh-check"]),0)
            runner.assert_not_called()

    def test_start_validates_key_before_colab_creation(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(app,"STATE_DIR",Path(d)), \
             mock.patch.object(app,"STATE_FILE",Path(d)/"state.json"), \
             mock.patch.object(app,"ensure_key",side_effect=RuntimeError("identidade inválida")), \
             mock.patch.object(app,"run") as runner:
            self.assertEqual(app.main(["start","code","--gpu","G4"]),1)
            runner.assert_not_called()

