"""CPU/generated-fixture tests only. Optimizer updates are forbidden throughout.

No user datasets/checkpoints, training jobs, GPU work, exports or settings writes.
Loop/resume tests replace the objective/validation and block optimizer steps.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import cv2
import numpy as np
import torch

from mefi import train_ui
from mefi.checkpoint import rng_state
from mefi.model import ARCH, PyramidFlow, load_checkpoint
from mefi.train import losses as original_flow_losses
from mefi.ui_data import Labels, UiFrames, batch_at_step, head_labels, sha256
from mefi.ui_model import TemporalUi


class Fixture:
    def __init__(self, root):
        self.root, self.samples = root, []
        rng = np.random.default_rng(123)
        for split, count in (('train', 2), ('val', 1)):
            for index in range(count):
                frames = []
                for k in range(3):
                    frames.append(self.image(f'{split}{index}-frame{k}.png', rng.integers(0, 256, (64, 128, 3), np.uint8)))
                coverage = np.zeros((64, 128), np.uint8)
                coverage[8:24, 16:48] = 255
                self.samples.append({'id': f'{split}{index}', 'source': f'ui-{split}', 'split': split,
                                     'frame_ids': [index * 3, index * 3 + 1, index * 3 + 2], 'height': 64, 'width': 128,
                                     'frames': frames, 'coverage': self.image(f'{split}{index}-coverage.png', coverage),
                                     'valid': self.image(f'{split}{index}-valid.png', np.full((64, 128), 255, np.uint8))})
        # Explicit test allow-list entries, deliberately nonexistent. Neither
        # preflight nor training/validation may access or decode these paths.
        ref = lambda name, digest: {'path': name, 'sha256': digest * 64}
        self.samples.append({'id': 'locked', 'source': 'ui-test', 'split': 'test', 'frame_ids': [0, 1, 2],
                             'height': 64, 'width': 128, 'frames': [ref(f'test-frame{k}.png', c) for k, c in enumerate('abc')],
                             'coverage': ref('test-coverage.png', 'd'), 'valid': ref('test-valid.png', 'e')})
        self.path = root / 'labels.json'
        self.write()

    def image(self, name, image):
        path = self.root / name
        self.assert_write(path, image)
        return {'path': name, 'sha256': sha256(path)}

    @staticmethod
    def assert_write(path, image):
        ok, encoded = cv2.imencode('.jpg' if path.suffix.lower() == '.jpg' else '.png', image)
        if not ok:
            raise AssertionError('Could not create generated test fixture')
        path.write_bytes(encoded.tobytes())

    def write(self):
        self.path.write_text(json.dumps({'format': 'me-ui-labels-1', 'label_version': 'generated-test-1',
                                         'samples': self.samples}), encoding='utf-8')


class CpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        def forbidden_step(_optimizer, closure=None):
            raise AssertionError('Optimizer updates are forbidden in this suite')
        cls.no_updates = patch.object(torch.optim.AdamW, 'step', forbidden_step)
        cls.no_updates.start()

    @classmethod
    def tearDownClass(cls):
        cls.no_updates.stop()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory(prefix='ui-training-generated-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)

    def args(self, extra=()):
        return train_ui.parser().parse_args(['--run', str(self.root / 'run'), '--labels', str(self.fixture.path),
                                             '--data', str(self.root / 'flow-placeholder'), '--preset', 'small',
                                             '--epochs', '1', '--ui-crop', '64', '--crop', '64',
                                             '--max-steps', '100', '--max-minutes', '10', *extra])

    def test_manifest_and_causal_dense_shapes_never_access_test_files(self):
        labels = Labels(self.fixture.path)
        self.assertEqual(labels.identity['counts'], {'train': 2, 'val': 1, 'test': 1})
        frames, target, valid = UiFrames(labels, 'train', 64)[(0, 99)]
        self.assertEqual(frames.shape, (9, 64, 128))
        self.assertEqual(target.shape, (1, 32, 64))
        self.assertTrue(bool(valid.all()))
        with self.assertRaisesRegex(ValueError, 'never loads'):
            UiFrames(labels, 'test')

    def test_split_source_and_copied_frame_leakage_are_rejected(self):
        self.fixture.samples[2]['source'] = 'ui-train'
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'source leakage'):
            Labels(self.fixture.path)
        self.fixture.samples[2]['source'] = 'ui-val'
        self.fixture.samples[2]['frames'][0] = self.fixture.samples[0]['frames'][0].copy()
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'frame-copy leakage'):
            Labels(self.fixture.path)

    def test_chronology_and_path_escape_are_rejected(self):
        self.fixture.samples[0]['frame_ids'] = [0, 2, 4]
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            Labels(self.fixture.path)
        self.fixture.samples[0]['frame_ids'] = [0, 1, 2]
        self.fixture.samples[0]['frames'][0]['path'] = '../escape.png'
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            Labels(self.fixture.path)

    def test_file_tampering_before_and_after_preflight_is_rejected(self):
        labels = Labels(self.fixture.path)
        path = self.root / self.fixture.samples[0]['frames'][0]['path']
        path.write_bytes(path.read_bytes() + b'tamper')
        with self.assertRaisesRegex(ValueError, 'changed after preflight'):
            UiFrames(labels, 'train', 64)[0]
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            Labels(self.fixture.path)

    def test_unknown_cells_ignored_and_no_automatic_resizing(self):
        target, valid = torch.ones(1, 4, 4), torch.ones(1, 4, 4)
        valid[:, 0, 0] = 0
        coverage, known = head_labels(target, valid)
        self.assertEqual(float(known[0, 0, 0]), 0)
        self.assertEqual(float(coverage[0, 0, 0]), 0)
        self.assertEqual(float(known.sum()), 3)
        with self.assertRaisesRegex(ValueError, 'no automatic'):
            UiFrames(Labels(self.fixture.path), 'train', 128)

    def test_nonbinary_valid_and_empty_known_samples_fail(self):
        item = self.fixture.samples[0]
        for value, message in ((127, 'only'), (0, 'no fully known')):
            item['valid'] = self.fixture.image('bad-valid.png', np.full((64, 128), value, np.uint8))
            self.fixture.write()
            with self.assertRaisesRegex(ValueError, message):
                UiFrames(Labels(self.fixture.path), 'train', 64)[0]

    def test_ui_sampling_is_deterministic_and_independent_of_global_rng(self):
        data = UiFrames(Labels(self.fixture.path), 'train', 64)
        before = torch.get_rng_state().clone()
        a, b = batch_at_step(data, 3, 1, 22), batch_at_step(data, 3, 1, 22)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_joint_task_source_guard_preserves_original_flow_splitter(self):
        labels = Labels(self.fixture.path)
        labels.require_flow_disjoint([('unused', 'source:different-flow-recording', {})])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            labels.require_flow_disjoint([('unused', 'source:ui-val', {})])
        with self.assertRaisesRegex(ValueError, 'source-aware'):
            labels.require_flow_disjoint([('unused', 'legacy-name', {})])

    def test_locked_capture_guard_and_invalid_manifest_are_fail_closed(self):
        import mefi.ui_data as data
        with patch.object(data, 'LOCKED_CAPTURE', self.root):
            with self.assertRaisesRegex(ValueError, 'test-only'):
                Labels(self.fixture.path)
        self.fixture.path.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'me-ui-labels-1'):
            Labels(self.fixture.path)

    def test_new_joint_settings_support_scratch_without_any_checkpoint(self):
        config = train_ui.settings(self.args())
        self.assertEqual(config['mode'], 'joint')
        self.assertIsNone(config['init_from'])
        model = train_ui.new_model(config)
        self.assertEqual(model.mode, 'scratch')
        self.assertEqual([g['name'] for g in model.parameter_groups()], ['flow', 'ui'])

    def test_missing_labels_caps_and_frozen_source_fail_before_run(self):
        for args, message in ((['--run', str(self.root / 'never')], 'labels'),
                              (['--run', str(self.root / 'never'), '--labels', str(self.fixture.path), '--data', 'unused'], 'max_steps'),
                              (['--run', str(self.root / 'never'), '--mode', 'frozen', '--labels', str(self.fixture.path),
                                '--max-steps', '1', '--max-minutes', '1'], 'requires --init-from')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                train_ui.settings(train_ui.parser().parse_args(args))
        self.assertFalse((self.root / 'never').exists())

    def test_pinned_frozen_source_is_not_changed_and_restores_its_architecture(self):
        source, base = self.root / 'source.pt', PyramidFlow('small', refiner=True)
        torch.save({'arch': base.arch, 'preset': base.preset, 'model': base.state_dict()}, source)
        before = source.read_bytes()
        args = train_ui.parser().parse_args(['--run', str(self.root / 'never'), '--mode', 'frozen',
                                             '--labels', str(self.fixture.path), '--init-from', str(source),
                                             '--init-sha256', sha256(source), '--max-steps', '1', '--max-minutes', '1'])
        config = train_ui.settings(args)
        model = train_ui.new_model(config)
        self.assertEqual((config['preset'], config['refiner'], model.mode), ('small', True, 'frozen'))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertEqual(source.read_bytes(), before)
        config['init_sha256'] = '0' * 64
        with patch.object(torch, 'load', side_effect=AssertionError('Must reject before loading')):
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                train_ui.new_model(config)

    def test_joint_original_flow_loss_and_separate_clipping_are_exact(self):
        model = TemporalUi.from_scratch('small', refiner=True)
        baseline = copy.deepcopy(model.backbone)
        flow = torch.rand(1, 9, 64, 128)
        ui_batch = batch_at_step(UiFrames(Labels(self.fixture.path), 'train', 64), 0, 1, 22)
        expected = original_flow_losses(baseline, flow, occlusion=0)[0]
        actual, parts = train_ui.objective(model, ui_batch, flow)
        self.assertTrue(torch.equal(parts['flow_loss'], expected.detach()))
        expected.backward()
        actual.backward()
        # Large UI gradients must not alter the backbone's clipping scale.
        for parameter in model.head.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(10000)
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
        train_ui.clip_groups(model)
        for left, right in zip(baseline.parameters(), model.backbone.parameters()):
            self.assertTrue(left.grad is None and right.grad is None or
                            left.grad is not None and right.grad is not None and torch.equal(left.grad, right.grad))

    def test_ui_validation_reports_per_source_without_loading_test(self):
        model = TemporalUi(PyramidFlow('small'), mode='frozen')
        metrics = train_ui.validate_ui(model, UiFrames(Labels(self.fixture.path), 'val'), torch.device('cpu'),
                                       dict(positive_weight=1., dice_weight=.5))
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertEqual(set(metrics['by_source']), {'ui-val'})
        self.assertGreater(metrics['total']['known'], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_validation_budget_stop_happens_before_inference(self):
        model = TemporalUi(PyramidFlow('small'), mode='frozen')
        data = UiFrames(Labels(self.fixture.path), 'val')
        with patch.object(model, 'training_logits', side_effect=AssertionError('No inference after stop')):
            with self.assertRaisesRegex(train_ui.JobStopped, 'cap'):
                train_ui.validate_ui(model, data, torch.device('cpu'), {}, stop_check=lambda: 'cap')

    def test_versioned_ui_weights_are_rejected_by_legacy_runtime_export_loader(self):
        config = train_ui.settings(self.args())
        model = train_ui.new_model(config)
        payload = train_ui.weight_payload(model, config, {}, 0, float('inf'), None)
        path = self.root / 'ui.pt'
        torch.save(payload, path)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'pair-ui-training-v1'):
            load_checkpoint(path)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(payload['deployable'])

    def test_existing_run_is_rejected_without_loading_or_replacing_anything(self):
        run = self.root / 'run'
        run.mkdir()
        (run / 'best.pt').write_bytes(b'protected old flow checkpoint')
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(torch, 'load', side_effect=AssertionError('No source load')):
            self.assertEqual(train_ui.main(['--run', str(run)]), 1)
        self.assertIn('existing runs/checkpoints are preserved', output.getvalue())
        self.assertEqual((run / 'best.pt').read_bytes(), b'protected old flow checkpoint')

    def frozen_job_args(self):
        source, base = self.root / 'initial-generated.pt', PyramidFlow('small')
        torch.save({'arch': ARCH, 'preset': 'small', 'model': base.state_dict()}, source)
        return ['--run', str(self.root / 'run'), '--mode', 'frozen', '--labels', str(self.fixture.path),
                '--init-from', str(source), '--init-sha256', sha256(source), '--epochs', '1',
                '--ui-crop', '64', '--ui-batch', '1', '--max-steps', '100', '--max-minutes', '10', '--save-every', '1']

    @contextlib.contextmanager
    def mocked_loop(self):
        self.loop_initial_weights = None
        self.loop_flow_batches = []
        def fake_objective(model, *_args, **_kwargs):
            if self.loop_initial_weights is None:
                self.loop_initial_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if len(_args) > 1 and _args[1] is not None:
                self.loop_flow_batches.append(_args[1].detach().clone())
            loss = model.head[-1].weight.sum() * 0 + 1
            return loss, {'ui_loss': loss.detach(), 'flow_loss': loss.detach() * 0}
        metrics = dict(loss=1., total=dict(iou=.2), by_source={'generated-validation': {}})
        # AdamW.step is separately forbidden for the whole test class. This
        # replaces the step dispatch itself, never performs a weight update.
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(train_ui, 'objective', side_effect=fake_objective) as objective, \
                patch.object(train_ui, 'validate_ui', return_value=metrics) as validate, \
                patch.object(torch.amp.GradScaler, 'step', return_value=None), \
                warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore', UserWarning)
            yield objective, validate

    def test_mocked_resume_handles_pending_validation_without_retraining_batch(self):
        args = self.frozen_job_args()
        real_save = train_ui.save_checkpoint
        def interrupt(payload, path):
            real_save(payload, path)
            if path.name == 'last_ui.pt' and payload['step'] == 2:
                raise RuntimeError('simulated interruption before validation')
        with self.mocked_loop() as (objective, _), patch.object(train_ui, 'save_checkpoint', side_effect=interrupt):
            self.assertEqual(train_ui.main(args), 1)
            self.assertEqual(objective.call_count, 2)
        saved = torch.load(self.root / 'run/last_ui.pt', map_location='cpu', weights_only=True)
        self.assertEqual((saved['step'], saved['validated_epochs']), (2, 0))
        with self.mocked_loop() as (objective, validate):
            self.assertEqual(train_ui.main(['--run', str(self.root / 'run'), '--resume']), 0)
            self.assertEqual(objective.call_count, 0, 'Consumed batches must not be trained again')
            self.assertEqual(validate.call_count, 1, 'Interrupted validation must not be skipped')
        saved = torch.load(self.root / 'run/last_ui.pt', map_location='cpu', weights_only=True)
        self.assertEqual((saved['step'], saved['validated_epochs']), (2, 1))
        before = {p.name: p.read_bytes() for p in (self.root / 'run').iterdir()}
        with self.mocked_loop() as (objective, validate):
            self.assertEqual(train_ui.main(['--run', str(self.root / 'run'), '--resume']), 0)
            self.assertEqual(objective.call_count + validate.call_count, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / 'run').iterdir()})

    def test_mocked_joint_route_preserves_the_original_flow_batch_and_augments(self):
        from mefi.data import EpochBatches, Triplets, load_index, split
        folder = self.root / 'generated-flow'
        folder.mkdir()
        rng = np.random.default_rng(444)
        index = []
        for k in range(12):
            name = f'sample{k}'
            sample = folder / 'triplets' / name
            sample.mkdir(parents=True)
            for i in range(3):
                Fixture.assert_write(sample / f'im{i}.jpg', rng.integers(0, 256, (64, 128, 3), np.uint8))
            index.append(dict(dir=name, video='generated', source='different-flow-source',
                              frame=k * 3, stride=1, width=128, height=64, motion90=4))
        (folder / 'index.json').write_text(json.dumps(index), encoding='utf-8')
        items, _ = load_index([str(folder)])
        train, _ = split(items, .1)
        original = Triplets(train, 64, train=True, pan=0.)
        sampler = EpochBatches(len(original), 4, 1)
        expected = torch.stack([original[key] for key in next(iter(sampler))])
        args = ['--run', str(self.root / 'run'), '--mode', 'joint', '--labels', str(self.fixture.path),
                '--data', str(folder), '--preset', 'small', '--batch', '4', '--crop', '64',
                '--ui-crop', '64', '--ui-batch', '1', '--epochs', '1', '--workers', '0',
                '--max-steps', '1', '--max-minutes', '10']
        with self.mocked_loop():
            self.assertEqual(train_ui.main(args), 0)
            self.assertEqual(len(self.loop_flow_batches), 1)
            self.assertTrue(torch.equal(expected, self.loop_flow_batches[0]))
            initial = self.loop_initial_weights
        saved = torch.load(self.root / 'run/last_ui.pt', map_location='cpu', weights_only=True)
        self.assertEqual((saved['config']['mode'], saved['ui_updates'], saved['flow_updates']), ('joint', 1, 1))
        self.assertTrue(all(torch.equal(value, saved['model'][key]) for key, value in initial.items()))

    def test_mocked_job_cap_does_not_automatically_extend_or_change_settings(self):
        args = self.frozen_job_args() + ['--max-steps', '1']
        with self.mocked_loop() as (objective, validate):
            self.assertEqual(train_ui.main(args), 0)
            self.assertEqual((objective.call_count, validate.call_count), (1, 0))
        saved = torch.load(self.root / 'run/last_ui.pt', map_location='cpu', weights_only=True)
        before = (self.root / 'run/last_ui.pt').read_bytes()
        with self.mocked_loop() as (objective, _):
            self.assertEqual(train_ui.main(['--run', str(self.root / 'run'), '--resume']), 0)
            self.assertEqual(objective.call_count, 0)
            self.assertEqual(train_ui.main(['--run', str(self.root / 'run'), '--resume', '--ui-lr', '.01']), 1)
        self.assertEqual((self.root / 'run/last_ui.pt').read_bytes(), before)
        self.assertEqual((saved['step'], saved['ui_updates'], saved['flow_updates']), (1, 1, 0))

    def test_mocked_loop_keeps_every_weight_unchanged_and_resumes_reject_changed_labels(self):
        args = self.frozen_job_args() + ['--max-steps', '1']
        with self.mocked_loop():
            self.assertEqual(train_ui.main(args), 0)
            initial = self.loop_initial_weights
        saved = torch.load(self.root / 'run/last_ui.pt', map_location='cpu', weights_only=True)
        self.assertTrue(all(torch.equal(value, saved['model'][key]) for key, value in initial.items()),
                        'Even mocked orchestration tests must perform ZERO optimizer/weight updates')
        before = {p.name: p.read_bytes() for p in (self.root / 'run').iterdir()}
        manifest = json.loads(self.fixture.path.read_text(encoding='utf-8'))
        manifest['label_version'] = 'changed-fixture-version'
        self.fixture.path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.mocked_loop() as (objective, _):
            self.assertEqual(train_ui.main(['--run', str(self.root / 'run'), '--resume']), 1)
            self.assertEqual(objective.call_count, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / 'run').iterdir()})


class InterfaceTests(unittest.TestCase):
    def test_ui_and_flow_routes_are_explicit_and_no_job_is_launched(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import app as ui
        application = ui.QApplication.instance() or ui.QApplication([])
        with tempfile.TemporaryDirectory(prefix='ui-training-interface-') as temp, \
                patch.object(ui, 'load_settings', return_value={}), patch.object(ui, 'save_settings'), \
                patch.object(ui.Trainer, 'detect_device'):
            window = ui.Trainer()
            try:
                root, page = Path(temp), window.train
                data = root / 'flow'
                data.mkdir()
                (data / 'index.json').write_text('[]', encoding='utf-8')
                labels = root / 'labels.json'
                labels.write_text('{}', encoding='utf-8')  # GUI chooses path; runner validates content.
                page.add_dataset(str(data))
                page.run.setText(str(root / 'new-ui-run'))
                page.ui_mode.setCurrentIndex(page.ui_mode.findData('joint'))
                with patch.object(page.job, 'start') as start:
                    page.start()
                    start.assert_not_called()
                page.ui_labels.setText(str(labels))
                with patch.object(page.job, 'start') as start:
                    page.start()
                module, args = start.call_args.args
                self.assertEqual(module, 'mefi.train_ui')
                self.assertIn('--data', args)
                self.assertIn('--max-steps', args)
                self.assertNotIn('--init-from', args, 'Scratch must not need an initial checkpoint')
                self.assertFalse(page.ui_mode.isEnabled())
                page.on_job({'event': 'ui_done', 'step': 1, 'steps': 1, 'stopped': True,
                             'checkpoint': str(root / 'new-ui-run/last_ui.pt')})
                self.assertEqual(window.export.checkpoint.text(), str(root / 'new-ui-run/last_ui.pt'))
                self.assertEqual(os.path.basename(window.export.out.text()), 'ai_flow_ui.onnx')
                page.finished(0)
                page.ui_mode.setCurrentIndex(page.ui_mode.findData('frozen'))
                page.datasets.clear()
                with patch.object(page.job, 'start') as start:
                    page.start()
                    start.assert_not_called()
                initial = root / 'generated-source.pt'
                initial.write_bytes(b'GUI-only file; no checkpoint loader is called')
                page.init_from.setText(str(initial))
                # The GUI now obtains this through its read-only background
                # worker. Supply that generated-file event without launching a job.
                from mefi.checkpoint_hash import fingerprint
                page.initial_hash_timer.stop()
                page._initial_hash_pending = False
                page.on_initial_hash({'event': 'checkpoint_hash', 'request': page._initial_hash_request,
                                      **fingerprint(initial)})
                with patch.object(page.job, 'start') as start:
                    page.start()
                module, args = start.call_args.args
                self.assertEqual(module, 'mefi.train_ui')
                self.assertEqual(args[args.index('--mode') + 1], 'frozen')
                self.assertIn('--init-sha256', args)
                self.assertNotIn('--data', args)
                page.finished(0)
                page.init_from.clear()
                page.ui_init_sha.clear()
                page.add_dataset(str(data))
                page.ui_mode.setCurrentIndex(page.ui_mode.findData('off'))
                with patch.object(page.job, 'start') as start:
                    page.start()
                self.assertEqual(start.call_args.args[0], 'mefi.train')
                self.assertNotIn('--labels', start.call_args.args[1])
                page.finished(0)
                # Continue must detect UI format by its distinct filename,
                # even if the selector currently says Off. No checkpoint is
                # loaded here and the fake subprocess launch is still mocked.
                ui_run = root / 'new-ui-run'
                ui_run.mkdir()
                (ui_run / 'last_ui.pt').write_bytes(b'GUI routing marker, not a checkpoint')
                page.resume.setChecked(True)
                page.datasets.clear()
                with patch.object(page.job, 'start') as start:
                    page.start()
                module, args = start.call_args.args
                self.assertEqual(module, 'mefi.train_ui')
                self.assertIn('--resume', args)
                self.assertNotIn('--max-steps', args, 'Continue cannot silently extend the saved cap')
                self.assertNotIn('--labels', args, 'Continue restores the original manifest')
                page.finished(0)
            finally:
                window.close()
                window.deleteLater()
                application.processEvents()


if __name__ == '__main__':
    unittest.main()
