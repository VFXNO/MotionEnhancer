"""Downloader safety checks: mocked downloads, never touches runtime models."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import fetch_model


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'ui_detect.onnx'
        self.data = b'test-model'
        self.sha = hashlib.sha256(self.data).hexdigest()

    def run_main(self, accept=True):
        argv = ['fetch_model.py', '--out', str(self.out)] + (['--accept-agpl'] if accept else [])
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fetch_model.main()

    def test_requires_licence_acknowledgement(self):
        with patch.object(fetch_model.urllib.request, 'urlopen') as network, self.assertRaises(SystemExit):
            self.run_main(False)
        network.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_refuses_different_existing_model(self):
        self.out.write_bytes(b'user-model')
        with patch.object(fetch_model.urllib.request, 'urlopen') as network, self.assertRaises(SystemExit):
            self.run_main()
        network.assert_not_called()
        self.assertEqual(self.out.read_bytes(), b'user-model')

    def test_existing_matching_model_is_unchanged(self):
        self.out.write_bytes(self.data)
        with patch.object(fetch_model, 'SHA256', self.sha), patch.object(fetch_model.urllib.request, 'urlopen') as network:
            self.run_main()
        network.assert_not_called()
        self.assertEqual(self.out.read_bytes(), self.data)

    def test_bad_download_hash_writes_nothing(self):
        with patch.object(fetch_model.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad')), self.assertRaises(RuntimeError):
            self.run_main()
        self.assertEqual(list(self.out.parent.iterdir()), [])

    def test_bad_upstream_licence_writes_nothing(self):
        with patch.object(fetch_model, 'SHA256', self.sha), patch.object(fetch_model.urllib.request, 'urlopen',
                side_effect=[io.BytesIO(self.data), io.BytesIO(b'MIT')]), self.assertRaises(RuntimeError):
            self.run_main()
        self.assertEqual(list(self.out.parent.iterdir()), [])

    def test_verified_download_retains_licence_and_source(self):
        licence = b'GNU AFFERO GENERAL PUBLIC LICENSE Version 3'
        with patch.object(fetch_model, 'SHA256', self.sha), patch.object(fetch_model.urllib.request, 'urlopen',
                side_effect=[io.BytesIO(self.data), io.BytesIO(licence)]):
            self.run_main()
        self.assertEqual(self.out.read_bytes(), self.data)
        self.assertEqual(self.out.with_suffix('.LICENSE.txt').read_bytes(), licence)
        self.assertIn(self.sha, self.out.with_suffix('.source.json').read_text())

    def test_does_not_replace_existing_provenance(self):
        existing = self.out.with_suffix('.source.json')
        existing.write_bytes(b'user-provenance')
        with patch.object(fetch_model.urllib.request, 'urlopen') as network, self.assertRaises(SystemExit):
            self.run_main()
        network.assert_not_called()
        self.assertEqual(existing.read_bytes(), b'user-provenance')


if __name__ == '__main__':
    unittest.main()
