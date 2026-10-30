r"""Trainer data/validation regressions. Run with the trainer's Python:

    tools\ai-train\.venv\Scripts\python.exe tools\ai-train\test_quality.py

All generated datasets and checkpoints live in temporary test folders.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2
import torch

from mefi.data import FullFrames, Triplets, load_index, split, validation_identity
from mefi import checkpoint, dataset, train


def item(video, frame, stride=1, moving=True):
    return (f'{video}/{frame}/{stride}', video,
            {'frame': frame, 'stride': stride, 'motion90': 8 if moving else 0, 'video': video})


class ValidationTests(unittest.TestCase):
    def test_split_never_shares_source_frames(self):
        items = [item('clip', frame * 3, stride=20) for frame in range(30)]
        train, val = split(items)
        self.assertTrue(train and val)
        boundary = min(i[2]['frame'] for i in val)
        self.assertTrue(all(i[2]['frame'] + 2 * i[2]['stride'] < boundary for i in train))

    def test_split_preserves_normal_temporal_holdout(self):
        items = [item('clip', frame * 6) for frame in range(100)]
        train, val = split(items)
        self.assertEqual((len(train), len(val)), (88, 10))
        self.assertLess(max(i[2]['frame'] + 2 for i in train), min(i[2]['frame'] for i in val))

    def test_invalid_split_fraction_is_rejected(self):
        for fraction in (0, -0.1, 1, 1.5, float('nan')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split([item('clip', f) for f in range(20)], fraction)

    def test_all_moving_validation_fills_budget(self):
        chosen = FullFrames([item('clip', f) for f in range(200)], limit=96)
        self.assertEqual(len(chosen), 96)
        self.assertEqual(len({i[0] for i in chosen.items}), 96)

    def test_validation_balances_videos_and_spans_each_one(self):
        items = [item('big', f) for f in range(1000)] + [item('small', f) for f in range(100)]
        chosen = FullFrames(items, limit=20)
        for video in ('big', 'small'):
            frames = [i[2]['frame'] for i in chosen.items if i[1] == video]
            self.assertEqual(len(frames), 10)
            self.assertEqual(min(frames), 0)
            self.assertEqual(max(frames), 999 if video == 'big' else 99)

    def test_validation_fills_short_groups_without_duplicates(self):
        items = [item('big', f) for f in range(100)] + [item('small', 0)]
        chosen = FullFrames(items, limit=20)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(sum(i[1] == 'small' for i in chosen.items), 1)
        self.assertEqual(len({i[0] for i in chosen.items}), 20)

    def test_validation_reserves_stills_without_dropping_moving_samples(self):
        items = [item('move', f) for f in range(100)] + [item('still', f, moving=False) for f in range(100)]
        chosen = FullFrames(items, limit=20)
        self.assertEqual(sum(i[1] == 'move' for i in chosen.items), 15)
        self.assertEqual(sum(i[1] == 'still' for i in chosen.items), 5)
        self.assertEqual(chosen.items, FullFrames(items[::-1], limit=20).items)

    def test_validation_identity_tracks_selected_sources_and_motion(self):
        items = [item('clip', f) for f in range(30)]
        identity = validation_identity(FullFrames(items))
        self.assertEqual(identity, validation_identity(FullFrames(items[::-1])))
        items[0][2]['motion90'] = 0
        self.assertNotEqual(identity, validation_identity(FullFrames(items)))


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='trainer-quality-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def make_index(self, folder, source='source-1'):
        folder.mkdir()
        entries = []
        for f in range(20):
            directory = f'sample_{f}'
            path = folder / 'triplets' / directory
            path.mkdir(parents=True)
            for k in range(3):
                (path / f'im{k}.jpg').write_bytes(b'test')
            entries.append({'dir': directory, 'video': 'clip', 'source': source,
                            'frame': f * 6, 'stride': 1, 'motion90': 4})
        (folder / 'index.json').write_text(json.dumps(entries), encoding='utf-8')
        return entries

    def args(self, videos):
        return SimpleNamespace(out=str(self.root / 'dataset'), videos=videos, height=32,
                               strides=[1], gap=3, max_per_video=2, duplicate=0.1,
                               cut=255, asymmetry=4, min_motion=1, max_motion=40, quality=95)

    def build(self, args, frames=None, motion=(4.0, 8.0)):
        if frames is None:
            frames = [np.full((32, 48, 3), i * 5, np.uint8) for i in range(9)]
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(dataset, 'probe', return_value=(48, 32, 30, len(frames))), \
                patch.object(dataset, 'frames', side_effect=lambda *a: iter(frames)), \
                patch.object(dataset, 'MotionMeter') as meter:
            meter.return_value.return_value = motion
            return dataset.build(args)

    def test_loading_same_dataset_twice_does_not_reweight_training(self):
        folder = self.root / 'one'
        self.make_index(folder)
        items, missing = load_index([str(folder), os.path.join(str(folder), '.')])
        self.assertEqual((len(items), missing), (20, 0))

    def test_same_source_in_two_datasets_has_one_temporal_split(self):
        a, b = self.root / 'one', self.root / 'two'
        self.make_index(a)
        self.make_index(b)
        items, _ = load_index([str(a), str(b)])
        # Rebuilt/copied samples from one source cannot land on both sides.
        self.assertEqual(len(items), 20)
        train, val = split(items)
        self.assertTrue(train and val)
        self.assertTrue({i[2]['frame'] for i in train}.isdisjoint(i[2]['frame'] for i in val))

    def test_same_stem_videos_get_distinct_source_ids(self):
        args = self.args([str(self.root / 'clip.mp4'), str(self.root / 'clip.webm')])
        self.assertEqual(self.build(args), 0)
        index_path = Path(args.out) / 'index.json'
        entries = json.loads(index_path.read_text())
        self.assertEqual(len(entries), 4)
        self.assertEqual(len({e['source'] for e in entries}), 2)
        self.assertEqual(len({e['dir'] for e in entries}), 4)
        before = index_path.read_bytes()
        self.assertEqual(self.build(args), 0)
        self.assertEqual(index_path.read_bytes(), before)

    def test_legacy_dataset_is_not_silently_duplicated_or_relabelled(self):
        args = self.args([str(self.root / 'clip.mp4')])
        entries = self.make_index(Path(args.out))
        for entry in entries:
            entry.pop('source')
        path = Path(args.out) / 'index.json'
        path.write_text(json.dumps(entries), encoding='utf-8')
        before = path.read_bytes()
        self.assertEqual(self.build(args), 1)
        self.assertEqual(path.read_bytes(), before)
        loaded, missing = load_index([args.out])
        self.assertEqual((len(loaded), missing), (20, 0))

    def test_strong_return_flash_is_kept_even_without_endpoint_motion(self):
        args = self.args([str(self.root / 'flash.mp4')])
        args.cut = 45
        frames = [np.full((32, 48, 3), value, np.uint8) for value in (8, 250, 12)]
        self.assertEqual(self.build(args, frames, motion=(0, 0)), 0)
        entries = json.loads((Path(args.out) / 'index.json').read_text())
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]['flash'])
        self.assertEqual(entries[0]['motion90'], 0)
        loaded, missing = load_index([args.out])
        self.assertEqual((len(loaded), missing), (1, 0))
        batch = Triplets(loaded, crop=64, train=False)[0]
        self.assertGreater(float(batch[3:6].mean()), 0.97)
        self.assertLess(float(batch[:3].mean()), 0.04)

    def test_return_flash_over_endpoint_cut_threshold_is_still_kept(self):
        args = self.args([str(self.root / 'flash.mp4')])
        args.cut = 45
        frames = [np.full((32, 48, 3), value, np.uint8) for value in (0, 255, 50)]
        self.assertEqual(self.build(args, frames, motion=(0, 0)), 0)
        entries = json.loads((Path(args.out) / 'index.json').read_text())
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]['flash'])

    def test_non_flash_scene_cut_is_still_rejected(self):
        args = self.args([str(self.root / 'cut.mp4')])
        args.cut = 45
        frames = [np.full((32, 48, 3), value, np.uint8) for value in (0, 10, 180)]
        self.assertEqual(self.build(args, frames), 0)
        self.assertEqual(json.loads((Path(args.out) / 'index.json').read_text()), [])

    def test_non_flash_still_is_still_filtered_by_minimum_motion(self):
        args = self.args([str(self.root / 'still.mp4')])
        args.cut = 45
        frames = [np.full((32, 48, 3), value, np.uint8) for value in (8, 16, 24)]
        self.assertEqual(self.build(args, frames, motion=(0, 0)), 0)
        self.assertEqual(json.loads((Path(args.out) / 'index.json').read_text()), [])

    def test_mild_linear_brightness_change_is_not_a_return_flash(self):
        args = self.args([str(self.root / 'fade.mp4')])
        args.min_motion, args.cut = 0, 45
        frames = [np.full((32, 48, 3), value, np.uint8) for value in (8, 16, 24)]
        self.assertEqual(self.build(args, frames), 0)
        entries = json.loads((Path(args.out) / 'index.json').read_text())
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0]['flash'])


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='trainer-training-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.data = self.root / 'data'
        self.run = self.root / 'run'
        self.data.mkdir()
        rng = np.random.default_rng(5)
        texture = cv2.GaussianBlur(rng.integers(0, 256, (64, 128, 3), dtype=np.uint8), (5, 5), 0)
        entries = []
        for f in range(20):
            folder = self.data / 'triplets' / str(f)
            folder.mkdir(parents=True)
            for k in range(3):
                self.assertTrue(cv2.imwrite(str(folder / f'im{k}.jpg'), np.roll(texture, f + 2 * k, axis=1)))
            entries.append({'dir': str(f), 'video': 'test', 'source': 'test-source', 'frame': f * 6,
                            'stride': 1, 'height': 64, 'width': 128, 'motion90': 4})
        (self.data / 'index.json').write_text(json.dumps(entries), encoding='utf-8')
        self.args = ['--data', str(self.data), '--run', str(self.run), '--epochs', '1',
                     '--batch', '32', '--workers', '0', '--crop', '64', '--preset', 'small']

    def run_training(self, extra=()):
        output = io.StringIO()
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with contextlib.redirect_stdout(output), patch.object(torch.cuda, 'is_available', return_value=False):
                result = train.main(self.args + list(extra))
        finally:
            torch.set_num_threads(threads)
        return result, [json.loads(line) for line in output.getvalue().splitlines()]

    def assert_state_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(first, second)

    def stop_after(self, steps):
        original = train.losses
        count = 0

        def loss_and_stop(*args, **kwargs):
            nonlocal count
            result = original(*args, **kwargs)
            count += 1
            if count == steps:
                (self.run / 'STOP').touch()
            return result

        return patch.object(train, 'losses', side_effect=loss_and_stop)

    def test_small_dataset_really_trains_and_keeps_matching_validation_on_resume(self):
        result, events = self.run_training()
        self.assertEqual(result, 0)
        data = next(e for e in events if e['event'] == 'data')
        self.assertEqual((data['train'], data['val'], data['steps_per_epoch']), (16, 2, 1))
        checkpoint = torch.load(self.run / 'last.pt', map_location='cpu')
        self.assertTrue(checkpoint['opt']['state'], 'No optimizer step was taken')
        self.assertTrue(any(torch.count_nonzero(v) for k, v in checkpoint['model'].items() if 'head.weight' in k))
        self.assertEqual(checkpoint['validation'], data['validation'])
        self.assertTrue(np.isfinite(checkpoint['best']))
        before = (self.run / 'best.pt').read_bytes()
        result, events = self.run_training(['--resume'])
        self.assertEqual(result, 0)
        self.assertTrue(any(e['event'] == 'resumed' for e in events))
        self.assertEqual((self.run / 'best.pt').read_bytes(), before)

    def test_new_run_refuses_to_overwrite_checkpoints(self):
        self.run.mkdir()
        last = self.run / 'last.pt'
        last.write_bytes(b'preserve this checkpoint')
        result, events = self.run_training()
        self.assertEqual(result, 1)
        self.assertEqual(events[-1]['event'], 'error')
        self.assertEqual(last.read_bytes(), b'preserve this checkpoint')
        self.assertFalse((self.run / 'config.json').exists())

    def test_seeded_fresh_runs_reproduce_cpu_augmentation_and_weights(self):
        self.assertEqual(self.run_training()[0], 0)
        first = torch.load(self.run / 'last.pt', map_location='cpu')
        self.run = self.root / 'repeat'
        self.args[self.args.index('--run') + 1] = str(self.run)
        self.assertEqual(self.run_training()[0], 0)
        second = torch.load(self.run / 'last.pt', map_location='cpu')
        self.assertEqual(first['validation'], second['validation'])
        for name in first['model']:
            self.assertTrue(torch.equal(first['model'][name], second['model'][name]), name)

    def test_resume_requires_a_checkpoint(self):
        result, events = self.run_training(['--resume'])
        self.assertEqual(result, 1)
        self.assertIn('no last.pt', events[-1]['message'])

    def test_flash_only_dataset_trains_and_scores_without_motion(self):
        path = self.data / 'index.json'
        entries = json.loads(path.read_text())
        for entry in entries:
            entry.update(flash=True, motion90=0)
            folder = self.data / 'triplets' / entry['dir']
            for k, value in enumerate((8, 250, 8)):
                self.assertTrue(cv2.imwrite(str(folder / f'im{k}.jpg'), np.full((64, 128, 3), value, np.uint8)))
        path.write_text(json.dumps(entries), encoding='utf-8')
        result, events = self.run_training()
        self.assertEqual(result, 0)
        scored = next(e for e in events if e['event'] == 'epoch')
        self.assertEqual(scored['scored'], 'all')
        self.assertEqual(scored['scored_count'], 2)
        self.assertEqual(scored['moving'], 0)
        self.assertTrue(np.isfinite(scored['psnr']))
        checkpoint = torch.load(self.run / 'last.pt', map_location='cpu')
        self.assertTrue(checkpoint['opt']['state'])

    def test_resume_rejects_incomplete_checkpoint_without_overwriting(self):
        self.run.mkdir()
        last = self.run / 'last.pt'
        torch.save({'arch': 'pyramid-v2', 'preset': 'small', 'epochs': 1, 'best': 99}, last)
        before = last.read_bytes()
        result, events = self.run_training(['--resume'])
        self.assertEqual(result, 1)
        self.assertIn('incomplete', events[-1]['message'])
        self.assertEqual(last.read_bytes(), before)
        self.assertFalse((self.run / 'config.json').exists())

    def test_mid_epoch_resume_matches_uninterrupted_training_with_workers(self):
        # The prefetched worker must not change either sample order or random
        # augmentation when a new process resumes in the middle of an epoch.
        for workers in (0, 1):
            with self.subTest(workers=workers):
                options = ['--epochs', '2', '--batch', '4', '--workers', str(workers), '--save-every', '1']
                full = self.root / f'full-{workers}'
                self.assertEqual(self.run_training(options + ['--run', str(full)])[0], 0)
                expected = torch.load(full / 'last.pt', map_location='cpu')
                self.run = self.root / f'partial-{workers}'
                options += ['--run', str(self.run)]
                with self.stop_after(2):
                    result, events = self.run_training(options)
                self.assertEqual(result, 0)
                self.assertTrue(events[-1]['stopped'])
                paused = torch.load(self.run / 'last.pt', map_location='cpu')
                self.assertEqual((paused['next_epoch'], paused['next_batch'], paused['step']), (0, 2, 2))
                self.assertEqual(self.run_training(options + ['--resume'])[0], 0)
                actual = torch.load(self.run / 'last.pt', map_location='cpu')
                self.assertEqual((actual['step'], actual['next_epoch'], actual['next_batch']), (8, 2, 0))
                for key in ('model', 'opt', 'sched', 'scaler', 'rng', 'schedule'):
                    self.assert_state_equal(expected[key], actual[key])

    def test_periodic_checkpoint_recovers_after_interruption_before_validation(self):
        options = ['--epochs', '2', '--batch', '4', '--save-every', '1']
        full = self.root / 'full'
        self.assertEqual(self.run_training(options + ['--run', str(full)])[0], 0)
        expected = torch.load(full / 'last.pt', map_location='cpu')
        original = train.save_checkpoint

        def interrupt(state, path):
            original(state, path)
            if Path(path).name == 'last.pt' and state['step'] == 4:
                raise RuntimeError('simulated interruption after last batch, before validation')

        with patch.object(train, 'save_checkpoint', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                self.run_training(options)
        saved = torch.load(self.run / 'last.pt', map_location='cpu')
        self.assertEqual((saved['next_epoch'], saved['next_batch']), (0, 4))
        self.assertFalse((self.run / 'best.pt').exists())
        self.assertEqual(self.run_training(options + ['--resume'])[0], 0)
        actual = torch.load(self.run / 'last.pt', map_location='cpu')
        for key in ('model', 'opt', 'sched', 'scaler', 'rng'):
            self.assert_state_equal(expected[key], actual[key])

    def test_finished_run_can_extend_and_restores_saved_settings_and_lr(self):
        self.assertEqual(self.run_training()[0], 0)
        previous = torch.load(self.run / 'last.pt', map_location='cpu')
        saved_lr = previous['opt']['param_groups'][0]['lr']
        result, events = self.run_training(['--resume', '--epochs', '2', '--preset', 'large', '--batch', '4',
                                          '--crop', '128', '--lr', '0.005', '--seed', '42',
                                          '--data', str(self.root / 'not-the-original-dataset')])
        self.assertEqual(result, 0)
        resumed = next(e for e in events if e['event'] == 'resumed')
        self.assertEqual(resumed['lr'], saved_lr)
        self.assertEqual((resumed['config']['preset'], resumed['config']['batch'], resumed['config']['crop'],
                          resumed['config']['lr'], resumed['config']['seed'], resumed['config']['data']),
                         ('small', 32, 64, 0.001, 1, [str(self.data)]))
        saved = torch.load(self.run / 'last.pt', map_location='cpu')
        self.assertEqual((saved['step'], saved['next_epoch'], saved['epochs']), (2, 2, 2))
        self.assertLessEqual(saved['opt']['param_groups'][0]['lr'], saved_lr)

    def test_completed_resume_is_a_read_only_noop_with_actionable_message(self):
        self.assertEqual(self.run_training()[0], 0)
        before = {name: (self.run / name).read_bytes() for name in ('last.pt', 'best.pt', 'config.json')}
        result, events = self.run_training(['--resume'])
        self.assertEqual(result, 0)
        self.assertTrue(events[-1]['already_complete'])
        self.assertIn('Total epochs', events[-1]['message'])
        for name, content in before.items():
            self.assertEqual((self.run / name).read_bytes(), content)

    def test_legacy_resume_rescores_and_backs_up_instead_of_comparing_old_scores(self):
        options = ['--epochs', '2', '--batch', '4']
        with self.stop_after(2):
            self.assertEqual(self.run_training(options)[0], 0)
        last, best = self.run / 'last.pt', self.run / 'best.pt'
        modern = torch.load(last, map_location='cpu')
        # This is the complete legacy format, including optimizer/scheduler.
        old = {key: modern[key] for key in ('arch', 'preset', 'model', 'opt', 'sched', 'epoch',
                                           'epochs', 'best', 'train_height')}
        old['best'] = 999.0  # Cannot ever be compared with the current subset.
        torch.save(old, last)
        old_best = torch.load(best, map_location='cpu')
        del old_best['validation']
        old_best['psnr'] = 999.0
        torch.save(old_best, best)
        originals = {'last.pt': last.read_bytes(), 'best.pt': best.read_bytes()}
        result, events = self.run_training(options + ['--resume'])
        self.assertEqual(result, 0)
        resumed = next(e for e in events if e['event'] == 'resumed')
        self.assertEqual((resumed['epoch'], resumed['batch'], resumed['step']), (0, 2, 2))
        self.assertTrue(any(e['event'] == 'rescored' for e in events))
        for name, content in originals.items():
            self.assertEqual((self.run / f'{name}.before-resume-001').read_bytes(), content)
        saved = torch.load(last, map_location='cpu')
        self.assertEqual(saved['step'], 8)
        self.assertLess(saved['best'], 100)
        self.assertEqual(saved['validation'], torch.load(best, map_location='cpu')['validation'])
        self.assertTrue(saved['resume_version'])

    def test_changed_training_samples_are_rejected_before_writing(self):
        with self.stop_after(1):
            self.assertEqual(self.run_training(['--epochs', '2', '--batch', '4'])[0], 0)
        before = {name: (self.run / name).read_bytes() for name in ('last.pt', 'best.pt', 'config.json')}
        index = self.data / 'index.json'
        entries = json.loads(index.read_text())
        index.write_text(json.dumps(entries[1:]), encoding='utf-8')
        result, events = self.run_training(['--resume', '--epochs', '2'])
        self.assertEqual(result, 1)
        self.assertIn('training samples changed', events[-1]['message'])
        for name, content in before.items():
            self.assertEqual((self.run / name).read_bytes(), content)

    def test_stop_token_keeps_initialization_stop_and_ignores_stale_requests(self):
        self.run.mkdir()
        stop = self.run / 'STOP'
        stop.write_text('current-start', encoding='utf-8')
        options = ['--epochs', '2', '--batch', '4', '--stop-token', 'current-start']
        result, events = self.run_training(options)
        self.assertEqual(result, 0)
        self.assertTrue(events[-1]['stopped'])
        self.assertEqual(events[-1]['step'], 1)
        # The old STOP is still there, but belongs to the previous launch.
        result, events = self.run_training(options + ['--resume', '--stop-token', 'new-start'])
        self.assertEqual(result, 0)
        self.assertFalse(events[-1]['stopped'])
        self.assertEqual(events[-1]['step'], 8)

    def test_atomic_checkpoint_failure_preserves_existing_file(self):
        self.run.mkdir()
        path = self.run / 'last.pt'
        original = b'last usable checkpoint'
        path.write_bytes(original)

        def fail(state, handle):
            handle.write(b'incomplete replacement')
            raise OSError('simulated full disk')

        with patch.object(torch, 'save', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'simulated full disk'):
                checkpoint.save_checkpoint({'step': 1}, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.run.iterdir()), [path])


class InterfaceTests(unittest.TestCase):
    def test_resume_progress_settings_stop_path_and_completed_message(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import app as ui
        application = ui.QApplication.instance() or ui.QApplication([])
        with tempfile.TemporaryDirectory(prefix='trainer-resume-interface-') as scratch, \
                patch.object(ui, 'load_settings', return_value={}), patch.object(ui, 'save_settings'):
            window = ui.Trainer()
            try:
                page = window.train
                run = Path(scratch) / 'original-run'
                page.run.setText(str(run))
                page.resume.setChecked(True)
                with patch.object(page.job, 'start') as start:
                    page.start()
                arguments = start.call_args.args[1]
                self.assertIn('--resume', arguments)
                self.assertNotIn('--data', arguments, 'Resume can restore a dataset not selected in the UI')
                self.assertFalse(page.run.isEnabled())
                page.run.setText(str(Path(scratch) / 'different-run'))
                page.stop()
                self.assertEqual((run / 'STOP').read_text(), page.stop_token)
                self.assertFalse((Path(scratch) / 'different-run').exists())
                config = {'preset': 'large', 'batch': 16, 'crop': 480, 'lr': 0.0007,
                          'data': [str(Path(scratch) / 'saved-dataset')]}
                page.on_job({'event': 'resumed', 'epoch': 2, 'batch': 3, 'step': 23, 'steps': 60,
                             'best': 29.4, 'config': config})
                self.assertEqual((page.bar.value(), page.bar.maximum()), (23, 60))
                self.assertEqual((page.batch.value(), page.crop.value(), page.preset.currentData()), (16, 480, 'large'))
                self.assertEqual(page.datasets.item(0).text(), config['data'][0])
                page.on_job({'event': 'done', 'best': 29.4, 'stopped': True, 'step': 23, 'steps': 60})
                self.assertIn('Paused at step 23', page.status.text())
                self.assertEqual(window.export.checkpoint.text(), str(run / 'best.pt'))
                page.on_job({'event': 'done', 'best': 29.4, 'already_complete': True,
                             'message': 'Increase Total epochs above 6 to continue training.'})
                page.finished(0)
                self.assertIn('Total epochs above 6', page.status.text())
                self.assertTrue(page.run.isEnabled())
            finally:
                window.close()
                window.deleteLater()
                application.processEvents()

    def test_scored_counts_source_counts_and_actionable_errors(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import app as ui
        application = ui.QApplication.instance() or ui.QApplication([])
        with tempfile.TemporaryDirectory(prefix='trainer-interface-') as scratch, \
                patch.object(ui, 'load_settings', return_value={}), patch.object(ui, 'save_settings'):
            window = ui.Trainer()
            try:
                window.train.on_job({'event': 'data', 'train': 400, 'val': 96, 'val_available': 120,
                                     'val_moving': 72, 'val_videos': 9, 'videos': 9, 'steps_per_epoch': 25})
                self.assertIn('96 scored', window.train.status.text())
                self.assertIn('120 held out', window.train.status.text())
                message = 'Choose a new dataset folder; existing data is unchanged.'
                window.dataset.on_job({'event': 'error', 'message': message})
                window.dataset.finished(1)
                self.assertIn(message, window.dataset.status.text())
                window.dataset.current = 'test'
                window.dataset.on_job({'event': 'progress', 'frame': 2, 'frames': 30, 'kept': 1, 'flashes': 1,
                                       'rejected': {'duplicate': 0, 'cut': 0, 'still': 0, 'fast': 0}})
                self.assertIn('1 flashes', window.dataset.status.text())
                window.train.on_job({'event': 'epoch', 'epoch': 0, 'epochs': 1, 'psnr': 25, 'blend_psnr': 24,
                                     'best': 25, 'scored': 'all', 'scored_count': 2, 'moving': 0, 'psnr_all': 25,
                                     'preview': str(Path(scratch) / 'none.png'), 'improved': True})
                self.assertIn('2 validation clips (all)', window.train.status.text())
                self.assertIn('all clips', window.train.m_quality_caption.text())
                folder = Path(scratch)
                (folder / 'index.json').write_text(json.dumps([{'video': 'same', 'source': 'one'},
                                                              {'video': 'same', 'source': 'two'}]))
                window.dataset.out.setText(str(folder))
                self.assertIn('2 video(s)', window.dataset.summary.text())
            finally:
                window.close()
                window.deleteLater()
                application.processEvents()


if __name__ == '__main__':
    unittest.main()
