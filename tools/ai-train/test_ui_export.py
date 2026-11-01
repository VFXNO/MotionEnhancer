"""Generated fixtures only: no training, user checkpoints/datasets or activation."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from mefi import export
from mefi.bundle import copy_export
from mefi.export_ui import RUNTIME_FORMAT, RuntimeTemporal, load_ui_checkpoint
from mefi.model import Exported, PyramidFlow
from mefi.ui_model import FORMAT, HEAD_ARCH, TemporalUi


def fixture(refiner=True, mode='frozen'):
    torch.manual_seed(47)
    model = TemporalUi(PyramidFlow('small', refiner=refiner), mode=mode, state_channels=4, hidden=8)
    with torch.no_grad():
        # Nonzero flow/refinement and deliberately positive candidates exercise
        # real arithmetic. These are synthetic values, NOT learned weights.
        model.backbone.fine[-1].head.bias.uniform_(-.01, .01)
        if refiner:
            model.backbone.fine[-1].refiner.head.bias.fill_(.02)
        model.head[-1].bias.fill_(2.)
    return {'format': 'mefi-ui-training-1', 'arch': 'pair-ui-training-v1',
            'head_format': FORMAT, 'head_arch': HEAD_ARCH, 'kind': 'weights', 'deployable': False,
            'flow_arch': model.backbone.arch, 'preset': 'small', 'state_channels': 4, 'hidden': 8,
            'model': model.state_dict(), 'config': {'mode': 'joint' if mode == 'scratch' else 'frozen'},
            'ui_updates': 1, 'flow_updates': int(mode == 'scratch')}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_both_training_modes_load_strictly_without_original_checkpoint_or_data(self):
        for mode in ('scratch', 'frozen'):
            ck = fixture(mode=mode)
            before = copy.deepcopy(ck['model'])
            model = load_ui_checkpoint(ck)
            self.assertEqual(model.mode, mode)
            self.assertTrue(all(torch.equal(t, model.state_dict()[k]) for k, t in before.items()))
            self.assertTrue(all(torch.equal(t, ck['model'][k]) for k, t in before.items()))

    def test_invalid_versions_untrained_and_nonfinite_weights_are_rejected(self):
        ck = fixture()
        for key, value in [('format', 'research'), ('head_arch', 'other'), ('deployable', True),
                           ('ui_updates', 0), ('state_channels', 0), ('flow_arch', 'other')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_ui_checkpoint({**ck, key: value})
        for value in (float('nan'), 1e10):
            changed = copy.deepcopy(ck)
            next(iter(changed['model'].values())).flatten()[0] = value
            with self.assertRaisesRegex(ValueError, 'nonfinite|overflowing'):
                load_ui_checkpoint(changed)

    def test_runtime_flow_parity_state_feedback_and_fail_closed_history(self):
        for refined in (False, True):
            ck = fixture(refined)
            model = load_ui_checkpoint(ck)
            runtime = RuntimeTemporal(model).eval()
            reference = Exported(copy.deepcopy(model.backbone)).eval()
            frames = torch.rand(1, 6, 64, 128)
            state = torch.full((1, 4, 32, 64), .25)
            with torch.inference_mode():
                flow, mask, allow, current = runtime(frames, state, torch.ones(1, 1, 1, 1))
                baseline = reference(frames)
                startup = runtime(frames, state, torch.zeros(1, 1, 1, 1))
                poisoned = state.clone()
                poisoned[0, 0, 0, 0] = float('nan')
                invalid = runtime(frames, poisoned, torch.ones(1, 1, 1, 1))
                following = runtime(frames, current, torch.ones(1, 1, 1, 1))
            self.assertTrue(torch.equal(flow, baseline[0]))
            self.assertTrue(torch.equal(mask, baseline[1]))
            self.assertEqual(tuple(allow.shape), (1, 1, 64, 128))
            self.assertEqual(tuple(current.shape), tuple(state.shape))
            self.assertTrue(bool((allow > .5).all()))
            self.assertEqual(int(torch.count_nonzero(startup[2])), 0)
            self.assertEqual(int(torch.count_nonzero(invalid[2])), 0)
            self.assertTrue(all(bool(torch.isfinite(t).all()) for t in following))


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_onnx_pair_and_renamed_copy_preserve_source_and_flow_installation(self):
        import onnx
        with tempfile.TemporaryDirectory(prefix='ui-export-generated-') as temp:
            root = Path(temp)
            checkpoint = root / 'best_ui.pt'
            torch.save(fixture(), checkpoint)
            original = checkpoint.read_bytes()
            flow = root / 'ai_flow.onnx'
            flow.write_bytes(b'preexisting flow model must remain unchanged')
            (root / 'ai_flow.json').write_text('{"format":"mefi-pyramid-flow-1"}')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = export.main([str(checkpoint), '--out', str(flow), '--width', '128', '--height', '64'])
            self.assertEqual(code, 0, output.getvalue())
            events = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
            done = [e for e in events if e.get('event') == 'done'][0]
            self.assertEqual(done['runtime_name'], 'ai_flow_ui.onnx')
            self.assertEqual(done['format'], RUNTIME_FORMAT)
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(flow.read_bytes(), b'preexisting flow model must remain unchanged')
            primary = root / 'ai_flow_ui.onnx'
            bypass = primary.with_suffix('.dis.onnx')
            full, plain = onnx.load(primary), onnx.load(bypass)
            self.assertEqual(len(full.graph.input), 3)
            self.assertEqual(len(full.graph.output), 4)
            self.assertLess(len(plain.graph.node), len(full.graph.node))
            self.assertFalse(any('refiner' in t.name for t in plain.graph.initializer))
            target = root / 'renamed' / 'ui.onnx'
            copy_export(primary, target)
            a = json.loads(target.with_suffix('.json').read_text())
            b = json.loads(target.with_suffix('.dis.json').read_text())
            self.assertEqual(a['bundle_id'], b['bundle_id'])
            self.assertEqual(a['dis_model'], 'ui.dis.onnx')
            self.assertEqual(a['state_channels'], 4)

    def test_export_refuses_another_model_format_without_writes(self):
        with tempfile.TemporaryDirectory(prefix='ui-export-protection-') as temp:
            root = Path(temp)
            checkpoint = root / 'best_ui.pt'
            torch.save(fixture(False), checkpoint)
            out = root / 'custom.onnx'
            out.write_bytes(b'keep')
            out.with_suffix('.json').write_text('{"format":"mefi-pyramid-flow-1"}')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = export.main([str(checkpoint), '--out', str(out), '--width', '64', '--height', '32'])
            self.assertEqual(code, 1)
            self.assertIn('different model format', output.getvalue())
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})


class InterfaceTests(unittest.TestCase):
    def test_ui_checkpoint_routes_to_separate_export_and_copy_target(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import app
        application = app.QApplication.instance() or app.QApplication([])
        owner = SimpleNamespace(settings={}, store=Mock())
        with tempfile.TemporaryDirectory(prefix='ui-export-interface-') as temp, \
                patch.object(app, 'program_builds', return_value=[temp]):
            root = Path(temp)
            source = root / 'best_ui.pt'
            source.write_bytes(b'GUI selection only')
            page = app.ExportPage(owner)
            try:
                page.set_checkpoint(str(source))
                self.assertEqual(Path(page.out.text()).name, 'ai_flow_ui.onnx')
                with patch.object(page.job, 'start') as start:
                    page.export()
                self.assertEqual(start.call_args.args[0], 'mefi.export')
                with patch('mefi.bundle.copy_export') as copy_bundle:
                    page.on_job({'event': 'done', 'format': RUNTIME_FORMAT, 'runtime_name': 'ai_flow_ui.onnx',
                                 'out': str(root / 'external.onnx'), 'width': 128, 'height': 64, 'preset': 'small'})
                self.assertEqual(Path(copy_bundle.call_args.args[1]), root / 'models' / 'ai_flow_ui.onnx')
                self.assertIn('Trained temporal UI', page.status.text())
                page.out.setText(str(root / 'models' / 'ai_flow_ui.onnx'))
                page.set_checkpoint(str(root / 'best.pt'))
                self.assertEqual(Path(page.out.text()).name, 'ai_flow.onnx')
            finally:
                page.deleteLater()
                application.processEvents()


if __name__ == '__main__':
    unittest.main()
