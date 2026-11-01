"""Generated/mock-only automatic HUD tests. No user footage, jobs or downloads."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

from mefi import ui_auto_label as automatic
from mefi.ui_labeling import Annotation, Project
from mefi.ui_data import Labels, UiFrames, sha256
from test_ui_labeling import GuiTests as LabelGuiBase


def hud_frames(scene=20, *, changed=False, absent=False):
    frames = []
    for index in range(3):
        frame = np.full((64, 128, 3), scene + index, np.uint8)
        frame[4:60, 4:96] = 60  # Opaque HUD background, not the surrounding scene.
        if not absent:
            frame[24:32, 32:48] = 240
            if changed:
                frame[24:32, 52:58] = 240  # New shape outside the source foreground.
        frames.append(frame)
    return frames


def reviewed(project, sample):
    _, annotation = project.load_sample(sample)
    annotation.coverage[24:32, 32:48] = 255
    annotation.valid[:] = 255
    project.save_annotation(sample, annotation, reviewed=True)


def fixture(root):
    project = Project.create(root)
    seed = project.add_sample(hud_frames(), 'game-train-a', 'train', [0, 1, 2], {})
    reviewed(project, seed)
    target = project.add_sample(hud_frames(80, changed=True), 'game-train-a', 'train', [3, 4, 5], {})
    other = project.add_sample(hud_frames(100, changed=True), 'game-train-b', 'train', [0, 1, 2], {})
    val = project.add_sample(hud_frames(140), 'game-validation', 'val', [0, 1, 2], {})
    reviewed(project, val)
    return project, seed, target, other, val


def request_for(project, seed, targets, *, cross_source=False):
    return dict(action='auto_labels', project_root=str(project.root), project_sha256=project.disk_hash,
                template_id=seed['id'], template_revision=seed['revision'], target_ids=[s['id'] for s in targets],
                tolerance=12, padding=16, cross_source=cross_source, time_limit_minutes=30)


def prepare(project, seed, targets, folder, **kwargs):
    request = request_for(project, seed, targets, **kwargs)
    info = automatic.prepare_auto_labels(project.root, folder, **{key: value for key, value in request.items()
                                                                 if key not in ('action', 'project_root', 'time_limit_minutes')})
    return info, request


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class AutoLabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generated-auto-hud-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project, self.seed, self.target, self.other, self.val = fixture(self.root / 'project')

    def teacher(self):
        frames, annotation = self.project.load_sample(self.seed)
        return automatic.HudTeacher(frames[2], annotation)

    def accept(self, info, request, folder=None):
        return automatic.accept_auto_labels(self.project, folder or self.root / 'stage', info,
                                             project_sha256=request['project_sha256'], target_ids=request['target_ids'])

    def test_local_segmentation_adapts_new_shapes_without_filling_rectangles_or_scene(self):
        annotation, stats = self.teacher().label(hud_frames(80, changed=True))
        self.assertTrue(stats['eligible'])
        self.assertEqual((annotation.coverage[26, 54], annotation.valid[26, 54]), (255, 255), 'New glyph must be detected')
        self.assertEqual((annotation.coverage[26, 50], annotation.valid[26, 50]), (0, 255), 'Gap is background, not a filled HUD box')
        self.assertEqual(annotation.valid[0, 120], 0, 'Outside seed neighborhoods stays unknown')
        self.assertGreater(stats['positive_cells'], 4)
        self.assertGreater(stats['negative_cells'], 16)

    def test_temporal_disagreement_and_unseen_colors_remain_unknown(self):
        frames = hud_frames(80, changed=True)
        frames[0][26, 54] = 60
        for frame in frames:
            frame[26, 56] = (150, 20, 70)
        annotation, stats = self.teacher().label(frames)
        self.assertEqual(annotation.valid[26, 54], 0)
        self.assertEqual(annotation.valid[26, 56], 0)
        self.assertTrue(stats['eligible'])

    def test_absent_hud_and_whole_white_scene_do_not_get_automatic_background_labels(self):
        for frames in (hud_frames(80, absent=True), [np.full((64, 128, 3), 240, np.uint8) for _ in range(3)]):
            with self.subTest(mean=float(frames[0].mean())):
                annotation, stats = self.teacher().label(frames)
                self.assertFalse(stats['eligible'])
                self.assertFalse(annotation.valid.any())

    def test_template_needs_both_classes_and_valid_parameters(self):
        frames, annotation = self.project.load_sample(self.seed)
        annotation.valid[annotation.coverage == 0] = 0
        with self.assertRaisesRegex(ValueError, 'background'):
            automatic.HudTeacher(frames[2], annotation)
        for kwargs in ({'tolerance': 33}, {'tolerance': True}, {'padding': -1}, {'padding': 65}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                automatic.HudTeacher(frames[2], annotation, **kwargs)

    def test_batch_saved_once_without_per_sample_review_and_preserves_seed_validation(self):
        seed_before, val_before = copy.deepcopy(self.seed), copy.deepcopy(self.val)
        before = snapshot(self.project.root)
        info, request = prepare(self.project, self.seed, [self.target, self.other], self.root / 'stage', cross_source=True)
        self.assertEqual(snapshot(self.project.root), before, 'Worker must only stage results')
        with patch.object(self.project, 'persist', wraps=self.project.persist) as persist:
            changed = self.accept(info, request)
        persist.assert_called_once()
        self.assertEqual(len(changed), 2)
        self.assertEqual(self.seed, seed_before)
        self.assertEqual(self.val, val_before)
        for sample in changed:
            self.assertFalse(sample['reviewed'])
            self.assertTrue(sample['auto_label']['eligible'])
            self.assertTrue(sample['auto_label']['spot_check'])
            self.assertEqual(self.project.load_sample(sample)[1].coverage[26, 54], 255)
        self.assertEqual(len(Project.open(self.project.root).samples), 4)

    def test_no_validation_teachers_targets_manual_drafts_or_unconfirmed_recordings(self):
        for seed, targets, message in ((self.val, [self.target], 'TRAINING template'),
                                        (self.seed, [self.val], 'validation'),
                                        (self.seed, [self.other], 'same HUD/game layout')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                prepare(self.project, seed, targets, self.root / 'stage')
        _, annotation = self.project.load_sample(self.target)
        annotation.stroke((20, 20), (22, 20), 2, 'ui')
        self.project.save_annotation(self.target, annotation)
        with self.assertRaisesRegex(ValueError, 'manual drafts'):
            prepare(self.project, self.seed, [self.target], self.root / 'stage')

    def test_stale_template_and_concurrent_edits_are_refused_without_label_changes(self):
        info, request = prepare(self.project, self.seed, [self.target], self.root / 'stage')
        self.project.document['label_version'] = 'external-revision'
        self.project.persist()
        before = snapshot(self.project.root)
        with self.assertRaisesRegex(ValueError, 'Project changed'):
            self.accept(info, request)
        self.assertEqual(snapshot(self.project.root), before)
        request = request_for(self.project, self.seed, [self.target])
        request['template_revision'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'pinned revision'):
            automatic.prepare_auto_labels(self.project.root, self.root / 'next', **{k: v for k, v in request.items()
                                                                                 if k not in ('action', 'project_root', 'time_limit_minutes')})

    def test_last_mask_tampering_and_eligibility_forgery_refuse_entire_batch(self):
        info, request = prepare(self.project, self.seed, [self.target, self.other], self.root / 'stage', cross_source=True)
        before = snapshot(self.project.root)
        bad = copy.deepcopy(info)
        bad['records'][-1]['auto_label']['eligible'] = False
        with self.assertRaisesRegex(ValueError, 'eligibility/counts'):
            self.accept(bad, request)
        ref = info['records'][-1]['annotation']['valid']
        (self.root / 'stage' / ref['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Project file changed'):
            self.accept(info, request)
        self.assertEqual(snapshot(self.project.root), before)

    def test_publish_failure_rolls_back_new_files_metadata_and_review_flags(self):
        info, request = prepare(self.project, self.seed, [self.target, self.other], self.root / 'stage', cross_source=True)
        before, samples = snapshot(self.project.root), copy.deepcopy(self.project.samples)
        with patch.object(self.project, 'persist', side_effect=ValueError('injected publication failure')):
            with self.assertRaisesRegex(ValueError, 'publication failure'):
                self.accept(info, request)
        self.assertEqual(snapshot(self.project.root), before)
        self.assertEqual(self.project.samples, samples)

    def test_changed_source_pixels_and_forged_validation_template_refuse_acceptance(self):
        info, request = prepare(self.project, self.seed, [self.target], self.root / 'stage')
        bad = copy.deepcopy(info)
        bad['records'][0]['auto_label']['template_id'] = self.val['id']
        with self.assertRaisesRegex(ValueError, 'template identity'):
            self.accept(bad, request)
        (self.project.root / self.target['frames'][0]['path']).write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'Project file changed'):
            self.accept(info, request)
        self.assertIsNone(self.target['annotation'])

    def test_encoded_staging_limit_leaves_existing_annotations_untouched(self):
        with patch.object(automatic, 'MAX_STAGED_BYTES', 1):
            info, request = prepare(self.project, self.seed, [self.target], self.root / 'stage')
        self.assertFalse(info['records'])
        self.assertTrue(info['warnings'])
        self.assertEqual(self.accept(info, request), [])
        self.assertIsNone(self.target['annotation'])

    def test_1000_windows_can_be_labeled_without_1000_manual_save_calls(self):
        # Reuse immutable generated frame refs to keep this scaling test small.
        targets = []
        for index in range(1000):
            sample = copy.deepcopy(self.target)
            sample.update(id=uuid.uuid4().hex, frame_ids=[6 + index * 3 + k for k in range(3)])
            targets.append(sample)
        self.project.samples.extend(targets)
        self.project.persist()
        folder = self.root / 'large-stage'
        info, request = prepare(self.project, self.seed, targets, folder)
        self.assertEqual(len(info['records']), 1000)
        self.assertEqual(sum(r['auto_label']['eligible'] for r in info['records']), 1000)
        self.assertLessEqual(sum(r['auto_label']['spot_check'] for r in info['records']), 8)
        with patch.object(self.project, 'save_annotation', side_effect=AssertionError('No per-window saves')):
            with patch.object(self.project, 'persist', wraps=self.project.persist) as persist:
                self.accept(info, request, folder)
        persist.assert_called_once()
        self.assertTrue(all(s['auto_label']['eligible'] and not s['reviewed'] for s in targets))

    def exported(self):
        info, request = prepare(self.project, self.seed, [self.target], self.root / 'stage')
        self.accept(info, request)
        return self.project.export(self.root / 'mixed-export', include_auto=True)

    def test_machine_export_requires_explicit_loader_optin_and_keeps_manual_validation(self):
        path = self.exported()
        doc = json.loads(path.read_text())
        self.assertEqual(doc['format'], 'me-ui-labels-2')
        self.assertEqual([s['label_origin'] for s in doc['samples']], ['human-reviewed', 'machine-generated', 'human-reviewed'])
        with self.assertRaisesRegex(ValueError, 'allow-auto-ui-labels'):
            Labels(path)
        labels = Labels(path, allow_auto_labels=True)
        self.assertEqual(labels.identity['counts'], {'train': 2, 'val': 1, 'test': 0})
        self.assertEqual(labels.identity['label_origins'], {'human-reviewed': 2, 'machine-generated': 1})
        frames, target, known = UiFrames(labels, 'train')[1]
        self.assertEqual(tuple(frames.shape), (9, 64, 128))
        self.assertEqual(float(target[0, 13, 27]), 1.)
        self.assertEqual(float(known[0, 13, 27]), 1.)
        self.assertEqual(float(known[0, 0, 60]), 0.)
        cropped = UiFrames(labels, 'train', 64)
        for seed in range(20):
            a, b = cropped[(1, seed)], cropped[(1, seed)]
            self.assertTrue(bool(((a[1] >= .5) & (a[2] > 0)).any()))
            self.assertTrue(all(bool((x == y).all()) for x, y in zip(a, b)), 'Automatic crop selection must be deterministic')
        normal = self.project.export(self.root / 'human-export')
        self.assertEqual(json.loads(normal.read_text())['format'], 'me-ui-labels-1')
        self.assertEqual(Labels(normal).identity['counts']['train'], 1)

    def test_machine_validation_template_leakage_downgrading_and_review_lies_are_rejected(self):
        path = self.exported()
        original = json.loads(path.read_text())
        for case in ('validation', 'template-leak', 'downgrade', 'review-lie', 'missing-origin'):
            doc = copy.deepcopy(original)
            machine = next(s for s in doc['samples'] if s['label_origin'] == 'machine-generated')
            if case == 'validation':
                machine.update(split='val', source='another-val')
            elif case == 'template-leak':
                machine['auto_label']['template_source'] = self.val['source']
            elif case == 'downgrade':
                doc['format'] = 'me-ui-labels-1'
            elif case == 'review-lie':
                machine['reviewed'] = True
            else:
                del machine['label_origin']
            path.write_text(json.dumps(doc))
            with self.subTest(case=case), self.assertRaises(ValueError):
                Labels(path, allow_auto_labels=True)

    def test_sparse_automatic_crop_falls_back_to_known_positive_cells_only_for_machine_data(self):
        path = self.exported()
        doc = json.loads(path.read_text())
        machine = next(s for s in doc['samples'] if s['label_origin'] == 'machine-generated')
        machine.update(height=128, width=256)
        def image(name, array):
            ok, data = cv2.imencode('.png', array)
            self.assertTrue(ok)
            full = path.parent / name
            full.write_bytes(data.tobytes())
            return {'path': name, 'sha256': sha256(full)}
        machine['frames'] = [image(f'large-frame{k}.png', np.full((128, 256, 3), 10 + k, np.uint8)) for k in range(3)]
        target, valid = np.zeros((128, 256), np.uint8), np.zeros((128, 256), np.uint8)
        target[116:124, 240:248], valid[112:128, 232:256] = 255, 255
        machine['coverage'], machine['valid'] = image('large-coverage.png', target), image('large-valid.png', valid)
        path.write_text(json.dumps(doc))
        dataset = UiFrames(Labels(path, allow_auto_labels=True), 'train', 64)
        for seed in range(30):
            a, b = dataset[(1, seed)], dataset[(1, seed)]
            self.assertEqual(tuple(a[0].shape), (9, 64, 128))
            self.assertTrue(bool(((a[1] >= .5) & (a[2] > 0)).any()))
            self.assertTrue(all(bool((x == y).all()) for x, y in zip(a, b)))

    def test_template_sources_also_participate_in_joint_flow_disjointness(self):
        path = self.exported()
        doc = json.loads(path.read_text())
        machine = next(s for s in doc['samples'] if s['label_origin'] == 'machine-generated')
        machine['auto_label']['template_source'] = 'additional-training-template'
        path.write_text(json.dumps(doc))
        labels = Labels(path, allow_auto_labels=True)
        with self.assertRaisesRegex(ValueError, 'UI and flow recordings'):
            labels.require_flow_disjoint([('generated', 'source:additional-training-template', None)])

    def test_editing_machine_annotation_removes_automatic_export_eligibility(self):
        self.exported()
        _, annotation = self.project.load_sample(self.target)
        annotation.stroke((54, 26), (54, 26), 0, 'unknown')
        self.project.save_annotation(self.target, annotation, reviewed=False)
        self.assertIsNone(self.target['auto_label'])
        path = self.project.export(self.root / 'after-edit', include_auto=True)
        self.assertEqual(Labels(path).identity['counts']['train'], 1)
        self.project.save_annotation(self.target, annotation, reviewed=True)
        path = self.project.export(self.root / 'after-review')
        self.assertEqual(Labels(path).identity['counts']['train'], 2)

    def test_changed_teacher_requires_regeneration_before_machine_export(self):
        self.exported()
        reviewed(self.project, self.seed)
        with self.assertRaisesRegex(ValueError, 'template changed'):
            self.project.export(self.root / 'stale-teacher-export', include_auto=True)
        self.assertFalse((self.root / 'stale-teacher-export').exists())

    def test_training_permission_defaults_off_and_resume_cannot_silently_change_it(self):
        from mefi import train_ui
        args = train_ui.parser().parse_args(['--run', str(self.root / 'run'), '--labels', str(self.root / 'labels.json'),
                                             '--data', str(self.root / 'flow'), '--max-steps', '2', '--max-minutes', '1'])
        config = train_ui.settings(args)
        self.assertFalse(config['allow_auto_ui_labels'])
        args.allow_auto_ui_labels = True
        config = train_ui.settings(args)
        self.assertTrue(config['allow_auto_ui_labels'])
        required = ('model opt sched scaler rng identities steps_per_epoch schedule elapsed_seconds best initial_checkpoint '
                    'flow_arch preset state_channels hidden').split()
        checkpoint = dict.fromkeys(required)
        checkpoint.update(format=train_ui.TRAIN_FORMAT, arch=train_ui.TRAIN_ARCH, kind='resume', resume_version=1,
                          head_format=train_ui.FORMAT, head_arch=train_ui.HEAD_ARCH, deployable=False,
                          step=0, ui_updates=0, flow_updates=0, validated_epochs=0, config=config,
                          experiment_id=config['experiment_id'])
        resume = train_ui.parser().parse_args(['--run', str(self.root / 'run'), '--resume'])
        self.assertTrue(train_ui.settings(resume, checkpoint)['allow_auto_ui_labels'])
        config.pop('allow_auto_ui_labels')  # Old human-only checkpoint remains compatible.
        self.assertFalse(train_ui.settings(resume, checkpoint).get('allow_auto_ui_labels', False))
        resume.allow_auto_ui_labels = True
        with self.assertRaisesRegex(ValueError, 'label trust'):
            train_ui.settings(resume, checkpoint)


class AutoGuiTests(LabelGuiBase):
    def install_fixture(self):
        project, seed, target, other, val = fixture(self.root / 'automatic-project')
        self.page.set_project(project)
        self.page.select_sample(0)
        self.page.pin_hud_template()
        return project, seed, target, other, val

    def test_batch_button_targets_only_compatible_training_windows_with_explicit_confirmation(self):
        from PySide6.QtWidgets import QMessageBox
        project, seed, target, other, val = self.install_fixture()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.No), patch.object(self.page.job, 'start') as launch:
            self.page.auto_label_all()
        launch.assert_not_called()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes), patch.object(self.page.job, 'start'):
            self.page.auto_label_all()
        self.assertEqual(self.page.request['target_ids'], [target['id']])
        self.assertEqual(self.page.deadline.interval(), 30 * 60000)
        with patch.object(self.page.job, 'kill'):
            self.page.cancel_worker()
        self.page.worker_finished(1)
        self.assertIsNone(target['annotation'])
        self.assertTrue(val['reviewed'])

    def test_real_generated_worker_saves_machine_masks_and_exports_without_per_window_save(self):
        project, seed, target, other, val = self.install_fixture()
        self.run_generated_worker(request_for(project, seed, [target, other], cross_source=True))
        self.assertFalse(target['reviewed'])
        self.assertTrue(target['auto_label']['eligible'])
        self.assertTrue(other['auto_label']['eligible'])
        self.assertIn('No per-window Save', self.page.status.text())
        self.assertFalse(self.page.include_auto_export.isChecked())
        self.page.include_auto_export.setChecked(True)
        folder = self.root / 'gui-export'
        with patch.object(self.ui.QFileDialog, 'getExistingDirectory', return_value=str(folder)):
            self.page.export_dataset()
        self.assertTrue(self.page.exported_manifest)
        self.assertEqual(Labels(folder / 'labels.json', allow_auto_labels=True).identity['counts']['train'], 3)
        self.page.use_manifest()
        self.assertFalse(self.window.train.ui_allow_auto.isChecked(), 'Selecting a manifest must not grant training permission')

    def test_uncertain_and_audit_navigation_never_marks_samples_reviewed(self):
        project, seed, target, other, val = self.install_fixture()
        self.run_generated_worker(request_for(project, seed, [target, other], cross_source=True))
        self.page.next_auto_review()
        self.assertIs(self.page.sample, target)
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertFalse(self.page.dirty)
        self.page.reviewed.setChecked(True)
        self.assertTrue(self.page.save_current())
        self.assertIsNone(target['auto_label'])
        self.page.next_auto_review()
        self.assertIs(self.page.sample, other)
        self.assertFalse(other['reviewed'])

    def test_cancel_even_after_worker_staging_accepts_no_machine_labels(self):
        from mefi.ui_label_worker import main
        project, seed, target, other, val = self.install_fixture()
        request = request_for(project, seed, [target])
        with patch.object(self.page.job, 'start'):
            self.page.start_worker(request)
        folder = Path(self.page.worker_temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--request', str(folder / 'request.json')]), 0)
        self.page.worker_done = True
        with patch.object(self.page.job, 'kill'):
            self.page.cancel_worker()
        self.page.worker_finished(0)
        self.assertIsNone(target['annotation'])
        self.assertFalse(folder.exists())

    def test_train_checkbox_emits_explicit_flag_only_for_new_ui_runs(self):
        page = self.window.train
        labels = self.root / 'labels.json'
        labels.write_text('{}')
        data = self.root / 'flow'
        data.mkdir()
        (data / 'index.json').write_text('[]')
        page.add_dataset(str(data))
        page.ui_labels.setText(str(labels))
        page.run.setText(str(self.root / 'no-run-will-start'))
        page.ui_mode.setCurrentIndex(page.ui_mode.findData('joint'))
        self.assertFalse(page.ui_allow_auto.isChecked())
        with patch.object(page.job, 'start') as launch:
            page.start_ui()
        self.assertNotIn('--allow-auto-ui-labels', launch.call_args.args[1])
        page.finished(1)
        page.ui_allow_auto.setChecked(True)
        with patch.object(page.job, 'start') as launch:
            page.start_ui()
        self.assertIn('--allow-auto-ui-labels', launch.call_args.args[1])
        page.finished(1)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(AutoLabelTests))
    suite.addTests(AutoGuiTests(name) for name in AutoGuiTests.__dict__ if name.startswith('test_'))
    return suite


if __name__ == '__main__':
    unittest.main()
