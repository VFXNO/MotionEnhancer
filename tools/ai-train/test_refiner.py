"""CPU/refiner safety tests; all generated data/runs are temporary.

Run after test_quality.py. GPU/export/native measurements are opt-in via
benchmark_refiner.py; no teacher or flow-label generation is used anywhere.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_quality as quality
from mefi import train
from mefi.model import (ARCH, REFINED_ARCH, BLOCK_INPUT, FlowRefiner, PyramidFlow,
                        initialize_from, load_checkpoint)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_shapes_presets_odd_pyramids_and_unchanged_mask(self):
        for preset in ('small', 'medium', 'large'):
            torch.manual_seed(5)
            base = PyramidFlow(preset)
            refined = PyramidFlow(preset, refiner=True)
            refined.load_state_dict(base.state_dict(), strict=False)
            for h, w in ((64, 96), (65, 99), (80, 144)):
                with self.subTest(preset=preset, size=(h, w)), torch.inference_mode():
                    a, b = torch.rand(2, 3, h, w), torch.rand(2, 3, h, w)
                    expected = base(a, b)
                    flow, mask = refined(a, b)
                    self.assertEqual(flow.shape, (2, 4, h, w))
                    self.assertEqual(mask.shape, (2, 1, h, w))
                    self.assertTrue(torch.equal(flow, expected[0]))
                    self.assertTrue(torch.equal(mask, expected[1]))
                    levels = refined(a, b, all_levels=True)
                    self.assertEqual(len(levels), len(base(a, b, all_levels=True)))
                    self.assertEqual(levels[-1][2], 0)

    def test_correction_is_bounded_and_has_only_four_vector_channels(self):
        refiner = FlowRefiner(32)
        with torch.no_grad():
            refiner.head.weight.fill_(100)
            refiner.head.bias.fill_(100)
            correction = refiner(torch.rand(1, BLOCK_INPUT, 65, 99),
                                 torch.rand(1, 32, 33, 50), torch.rand(1, 5, 65, 99))
        self.assertEqual(correction.shape, (1, 4, 65, 99))
        self.assertTrue(torch.isfinite(correction).all())
        self.assertLessEqual(float(correction.abs().max()), FlowRefiner.LIMIT)

    def test_existing_loss_trains_refiner_without_extra_targets(self):
        net = PyramidFlow('small', refiner=True)
        a = torch.rand(2, 3, 64, 96)
        batch = torch.cat((a, a.roll(2, -1), a.roll(4, -1)), 1)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss, _, image = train.losses(net, batch, occlusion=0)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(image.shape, a.shape)
            loss.backward()
            head = net.fine[-1].refiner.head
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
            opt.step()
        self.assertGreater(float(net.fine[-1].refiner.mix[0].weight.grad.abs().sum()), 0)

    def test_explicit_legacy_warm_start_is_exact_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='refiner-initialize-') as scratch:
            source = Path(scratch) / 'base.pt'
            base = PyramidFlow('small')
            with torch.no_grad():
                for block in [base.coarse, *base.fine]:
                    block.head.weight.normal_(0, 0.002)
            torch.save({'arch': ARCH, 'preset': 'small', 'model': base.state_dict()}, source)
            before = source.read_bytes()
            refined = PyramidFlow('small', refiner=True)
            initialize_from(refined, source)
            a, b = torch.rand(1, 3, 64, 96), torch.rand(1, 3, 64, 96)
            with torch.inference_mode():
                expected, actual = base(a, b), refined(a, b)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'same model size'):
                initialize_from(PyramidFlow('medium', refiner=True), source)

    def test_versioned_roundtrip_and_refiner_cannot_be_silently_discarded(self):
        with tempfile.TemporaryDirectory(prefix='refiner-checkpoint-') as scratch:
            path = Path(scratch) / 'refined.pt'
            net = PyramidFlow('small', refiner=True)
            state = {'arch': REFINED_ARCH, 'preset': 'small', 'model': net.state_dict()}
            torch.save(state, path)
            _, loaded = load_checkpoint(path)
            self.assertEqual(loaded.arch, REFINED_ARCH)
            self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in net.state_dict().items()))
            with self.assertRaisesRegex(ValueError, 'discard'):
                initialize_from(PyramidFlow('small'), path)
            state['model'].pop('fine.3.refiner.head.weight')
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path)


class RefinedTrainingTests(quality.TrainingTests):
    """Repeat the existing training/checkpoint regressions with the new head."""

    def setUp(self):
        super().setUp()
        self.args += ['--refiner']

    def test_resume_restores_refiner_even_without_the_flag(self):
        self.assertEqual(self.run_training()[0], 0)
        self.args.remove('--refiner')
        result, events = self.run_training(['--resume', '--epochs', '2'])
        self.assertEqual(result, 0)
        config = next(e['config'] for e in events if e['event'] == 'resumed')
        self.assertTrue(config['refiner'])
        self.assertEqual(torch.load(self.run / 'last.pt', map_location='cpu')['arch'], REFINED_ARCH)

    def test_legacy_continue_does_not_silently_add_a_refiner(self):
        self.args.remove('--refiner')
        self.assertEqual(self.run_training()[0], 0)
        result, events = self.run_training(['--resume', '--refiner', '--epochs', '2'])
        self.assertEqual(result, 0)
        self.assertFalse(next(e['config'] for e in events if e['event'] == 'resumed')['refiner'])
        self.assertEqual(torch.load(self.run / 'last.pt', map_location='cpu')['arch'], ARCH)

    def test_new_run_initializes_weights_but_not_old_progress(self):
        self.args.remove('--refiner')
        self.assertEqual(self.run_training()[0], 0)
        source = self.run / 'best.pt'
        before = source.read_bytes()
        target = self.root / 'refined-new-run'
        result, events = self.run_training(['--run', str(target), '--refiner', '--init-from', str(source)])
        self.assertEqual(result, 0)
        self.assertTrue(any(e['event'] == 'initialized' for e in events))
        saved = torch.load(target / 'last.pt', map_location='cpu')
        self.assertEqual((saved['arch'], saved['step'], saved['next_epoch']), (REFINED_ARCH, 1, 1))
        self.assertEqual(source.read_bytes(), before)

    def test_resume_refuses_best_from_another_architecture_without_writing(self):
        self.assertEqual(self.run_training()[0], 0)
        path = self.run / 'best.pt'
        wrong = torch.load(path, map_location='cpu')
        wrong['arch'] = ARCH
        torch.save(wrong, path)
        before = {name: (self.run / name).read_bytes() for name in ('best.pt', 'last.pt', 'config.json')}
        result, events = self.run_training(['--resume', '--epochs', '2'])
        self.assertEqual(result, 1)
        self.assertIn('different model', events[-1]['message'])
        for name, contents in before.items():
            self.assertEqual((self.run / name).read_bytes(), contents)


class InterfaceTests(unittest.TestCase):
    def test_refiner_and_initial_weights_are_new_run_options_and_resume_restores_them(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import app as ui
        application = ui.QApplication.instance() or ui.QApplication([])
        with tempfile.TemporaryDirectory(prefix='refiner-ui-') as scratch, \
                patch.object(ui, 'load_settings', return_value={}), patch.object(ui, 'save_settings'):
            window = ui.Trainer()
            try:
                data = Path(scratch) / 'data'
                data.mkdir()
                (data / 'index.json').write_text('[]')
                page = window.train
                page.add_dataset(str(data))
                page.run.setText(str(Path(scratch) / 'run'))
                page.refiner.setChecked(True)
                page.init_from.setText('initial.pt')
                with patch.object(page.job, 'start') as start:
                    page.start()
                args = start.call_args.args[1]
                self.assertIn('--refiner', args)
                self.assertIn('--init-from', args)
                self.assertFalse(page.refiner.isEnabled())
                page.finished(0)
                page.resume.setChecked(True)
                with patch.object(page.job, 'start') as start:
                    page.start()
                args = start.call_args.args[1]
                self.assertIn('--resume', args)
                self.assertNotIn('--init-from', args)
                self.assertNotIn('--refiner', args)
                config = {'preset': 'small', 'refiner': False, 'batch': 4, 'crop': 64,
                          'lr': 0.001, 'data': [str(data)]}
                page.on_job({'event': 'resumed', 'config': config, 'epoch': 0, 'batch': 1,
                             'step': 1, 'steps': 4, 'best': 20})
                self.assertFalse(page.refiner.isChecked())
                self.assertEqual(page.init_from.text(), '')
            finally:
                window.close()
                window.deleteLater()
                application.processEvents()


if __name__ == '__main__':
    unittest.main()
