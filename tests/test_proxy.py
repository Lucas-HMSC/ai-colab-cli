import importlib.util
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
import unittest
import urllib.request
import urllib.error

SRC = Path(__file__).resolve().parents[1] / 'remote' / 'proxy.py'
spec = importlib.util.spec_from_file_location('colab_proxy', SRC)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)

class ProxySecurityTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), proxy.make_handler('secret-test'))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown(); self.server.server_close()

    def test_rejects_wrong_token(self):
        with self.assertRaises(urllib.error.HTTPError) as ex:
            urllib.request.urlopen(self.url+'/v1/models', timeout=3)
        self.assertEqual(ex.exception.code,401)

    def test_rejects_unlisted_path(self):
        req = urllib.request.Request(self.url+'/private',headers={'X-Api-Key':'secret-test'})
        with self.assertRaises(urllib.error.HTTPError) as ex:
            urllib.request.urlopen(req,timeout=3)
        self.assertEqual(ex.exception.code,404)

    def test_post_without_body_refused(self):
        req=urllib.request.Request(self.url+'/v1/messages', data=b'', method='POST',headers={'X-Api-Key':'secret-test'})
        # Auth accepted, upstream isn't expected to be active; ensure status isn't 401.
        with self.assertRaises(urllib.error.HTTPError) as ex:
            urllib.request.urlopen(req,timeout=3)
        self.assertNotEqual(ex.exception.code,401)
