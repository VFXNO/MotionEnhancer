"""CPU/export/GUI regressions for the independent AI-versus-DIS switch."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mefi import export
from mefi.bundle import copy_export
from mefi.model import ARCH, REFINED_ARCH, PyramidFlow


class SwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_bypass_removes_operators_and_keeps_jointly_trained_backbone_and_mask(self):
        for preset in ('small', 'medium', 'large'):
            net = PyramidFlow(preset, refiner=True)
            with torch.no_grad():
                net.fine[-1].refiner.head.bias.fill_(0.5)
            before = {key: value.clone() for key, value in net.state_dict().items()}
            bypass = net.without_refiner()
            self.assertEqual(bypass.arch, ARCH)
            self.assertFalse(any('refiner' in key for key in bypass.state_dict()))
            self.assertTrue(all(torch.equal(value, before[key]) for key, value in bypass.state_dict().items()))
            self.assertTrue(all(torch.equal(value, before[key]) for key, value in net.state_dict().items()))
            for h, w in ((64, 96), (65, 99)):
                with torch.inference_mode():
                    a, b = torch.rand(1, 3, h, w), torch.rand(1, 3, h, w)
                    flow, mask = net(a, b)
                    plain, plain_mask = bypass(a, b)
                self.assertTrue(torch.equal(mask, plain_mask))
                self.assertGreater(float((flow - plain).abs().mean()), 0.9)
                self.assertFalse(torch.equal(flow, plain))

    def make_export(self, folder, refined=True):
        ck = folder / 'checkpoint.pt'
        net = PyramidFlow('small', refiner=refined)
        torch.save({'arch': net.arch, 'preset': 'small', 'model': net.state_dict(), 'psnr': 31.5}, ck)
        source = ck.read_bytes()
        calls = []

        def fake_export(model, frames, out, **kwargs):
            calls.append((model.net.arch, sum(p.numel() for p in model.parameters())))
            Path(out).write_bytes(b'onnx-fixture')

        with patch.object(torch.onnx, 'export', side_effect=fake_export), patch('onnx.load'), \
                patch('onnx.checker.check_model'), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(export.main([str(ck), '--out', str(folder / 'custom.onnx'),
                                          '--width', '128', '--height', '80']), 0)
        self.assertEqual(ck.read_bytes(), source)
        return calls, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_refined_export_emits_both_graphs_and_versioned_pair_metadata(self):
        with tempfile.TemporaryDirectory(prefix='switch-export-') as temporary:
            folder = Path(temporary)
            calls, events = self.make_export(folder)
            self.assertEqual(calls, [(ARCH, 294516), (REFINED_ARCH, 295240)])
            primary = json.loads((folder / 'custom.json').read_text())
            bypass = json.loads((folder / 'custom.dis.json').read_text())
            self.assertTrue(primary['integrated_refiner'])
            self.assertFalse(bypass['integrated_refiner'])
            self.assertTrue(primary['bundle_id'])
            self.assertEqual(primary['bundle_id'], bypass['bundle_id'])
            self.assertEqual((primary['psnr'], bypass['psnr']), (31.5, None))
            self.assertEqual(events[-1]['dis_out'], str(folder / 'custom.dis.onnx'))

    def test_plain_export_stays_single_and_does_not_delete_unrelated_companions(self):
        with tempfile.TemporaryDirectory(prefix='switch-export-') as temporary:
            folder = Path(temporary)
            stale = folder / 'custom.dis.onnx'
            stale.write_bytes(b'keep-user-file')
            calls, events = self.make_export(folder, refined=False)
            self.assertEqual(calls, [(ARCH, 294516)])
            self.assertIsNone(events[-1]['dis_out'])
            self.assertEqual(stale.read_bytes(), b'keep-user-file')
            self.assertFalse(json.loads((folder / 'custom.json').read_text())['integrated_refiner'])

    def test_copy_pair_can_rename_primary_and_companion_without_mutating_source(self):
        with tempfile.TemporaryDirectory(prefix='switch-copy-') as temporary:
            folder = Path(temporary)
            self.make_export(folder)
            originals = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
            destination = folder / 'runtime/models/ai_flow.onnx'
            copy_export(folder / 'custom.onnx', destination)
            for suffix in ('.onnx', '.json', '.dis.onnx', '.dis.json'):
                self.assertTrue(destination.with_suffix(suffix).is_file())
            for suffix in ('.json', '.dis.json'):
                meta = json.loads(destination.with_suffix(suffix).read_text())
                self.assertEqual((meta['refined_model'], meta['dis_model']), ('ai_flow.onnx', 'ai_flow.dis.onnx'))
            for name, data in originals.items():
                self.assertEqual((folder / name).read_bytes(), data)
            copy_export(destination, destination)  # Same-file copy is a no-op.

    def test_mixed_export_pair_is_not_copied_over_existing_models(self):
        with tempfile.TemporaryDirectory(prefix='switch-copy-') as temporary:
            folder = Path(temporary)
            self.make_export(folder)
            companion = folder / 'custom.dis.json'
            metadata = json.loads(companion.read_text())
            metadata['bundle_id'] = 'wrong-export'
            companion.write_text(json.dumps(metadata))
            target = folder / 'existing.onnx'
            target.write_bytes(b'keep-existing-model')
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                copy_export(folder / 'custom.onnx', target)
            self.assertEqual(target.read_bytes(), b'keep-existing-model')

    def test_plain_export_copies_without_optional_refiner_files(self):
        with tempfile.TemporaryDirectory(prefix='switch-copy-') as temporary:
            folder = Path(temporary)
            self.make_export(folder, refined=False)
            target = folder / 'runtime/ai_flow.onnx'
            copy_export(folder / 'custom.onnx', target)
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_suffix('.dis.onnx').exists())

    def test_trainer_gui_reports_and_copies_complete_refined_bundle(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import app as ui
        application = ui.QApplication.instance() or ui.QApplication([])
        with tempfile.TemporaryDirectory(prefix='switch-gui-') as temporary, \
                patch.object(ui, 'load_settings', return_value={}), patch.object(ui, 'save_settings'):
            folder = Path(temporary)
            _, events = self.make_export(folder)
            window = ui.Trainer()
            try:
                page = window.export
                page.copy_all.setChecked(True)
                with patch.object(ui, 'program_builds', return_value=[str(folder / 'runtime')]):
                    page.on_job(events[-1])
                self.assertTrue((folder / 'runtime/models/ai_flow.dis.onnx').is_file())
                self.assertIn('checked = AI, unchecked = DIS', page.status.text())
            finally:
                window.close()
                window.deleteLater()
                application.processEvents()


if __name__ == '__main__':
    unittest.main()
