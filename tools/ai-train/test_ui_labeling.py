"""Generated local fixtures only; no user footage/models, training or downloads."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from mefi.ui_labeling import (Annotation, Project, file_hash, image_window,
                              prepare_frames, source_id, video_window)
from mefi.ui_label_assist import inspect_model, probability_map, suggest
from mefi.ui_label_batch import prepare_video_batch, window_positions


def generated_video(root, name='generated.avi', count=24):
    path = root / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (128, 64))
    if not writer.isOpened():
        raise AssertionError('Generated MJPG fixture encoder unavailable')
    for index in range(count):
        writer.write(np.full((64, 128, 3), index * 8, np.uint8))
    writer.release()
    return path


def toy_model(path, *, outputs=1, external=False):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    weights = numpy_helper.from_array(np.array([1., 0., 0.], np.float32).reshape(1, 3, 1, 1), 'weights')
    if external:
        onnx.external_data_helper.set_external_data(weights, 'must-not-be-read.bin')
        weights.ClearField('raw_data')
    x = helper.make_tensor_value_info('rgb', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('ui', TensorProto.FLOAT, [1, 1, 32, 32])
    nodes = [helper.make_node('Conv', ['rgb', 'weights'], ['ui'], kernel_shape=[1, 1])]
    out = [y]
    if outputs == 2:
        nodes.append(helper.make_node('Identity', ['ui'], ['second']))
        out.append(helper.make_tensor_value_info('second', TensorProto.FLOAT, [1, 1, 32, 32]))
    model = helper.make_model(helper.make_graph(nodes, 'generated_rgb_selector', [x], out, [weights]),
                              opset_imports=[helper.make_opsetid('', 11)], ir_version=7)
    path.write_bytes(model.SerializeToString())


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generated-ui-labels-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(19)

    def frames(self):
        return [self.rng.integers(0, 256, (64, 128, 3), np.uint8) for _ in range(3)]

    def project(self):
        project = Project.create(self.root / 'project')
        a = project.add_sample(self.frames(), 'recording-a', 'train', [0, 1, 2], {})
        b = project.add_sample(self.frames(), 'recording-b', 'val', [0, 1, 2], {})
        return project, a, b

    def review(self, project, sample):
        _, annotation = project.load_sample(sample)
        annotation.valid[:] = 255
        annotation.stroke((16, 16), (22, 16), 2, 'ui')
        project.save_annotation(sample, annotation, reviewed=True)
        return annotation

    def test_annotation_is_unknown_until_manually_painted(self):
        annotation = Annotation(64, 128)
        self.assertFalse(annotation.valid.any())
        annotation.checkpoint()
        annotation.stroke((10, 10), (20, 10), 2, 'ui', 160)
        self.assertEqual((annotation.coverage[10, 15], annotation.valid[10, 15]), (160, 255))
        annotation.stroke((15, 10), (15, 10), 1, 'unknown')
        self.assertEqual((annotation.coverage[10, 15], annotation.valid[10, 15]), (0, 0))
        annotation.stroke((15, 10), (15, 10), 1, 'background')
        self.assertEqual((annotation.coverage[10, 15], annotation.valid[10, 15]), (0, 255))

    def test_brush_undo_and_redo_restore_both_masks(self):
        annotation = Annotation(64, 128)
        annotation.checkpoint()
        annotation.stroke((10, 10), (20, 10), 2, 'ui')
        expected = annotation.coverage.copy()
        self.assertTrue(annotation.undo())
        self.assertFalse(annotation.valid.any())
        self.assertTrue(annotation.undo(True))
        self.assertTrue(np.array_equal(annotation.coverage, expected))

    def test_zero_radius_brush_edits_exactly_one_pixel(self):
        annotation = Annotation(64, 128)
        annotation.stroke((10, 10), (10, 10), 0, 'ui', 128)
        self.assertEqual(int((annotation.valid != 0).sum()), 1)
        self.assertEqual(annotation.coverage[10, 10], 128)

    def test_ai_draft_preserves_manual_pixels_and_ignores_uncertainty(self):
        annotation = Annotation(64, 128)
        annotation.stroke((4, 4), (4, 4), 1, 'background')
        probability = np.full((64, 128), .5, np.float32)
        probability[4, 4], probability[10, 10], probability[12, 12] = .99, .95, .05
        self.assertEqual(annotation.apply_suggestion(probability), 2)
        self.assertEqual(annotation.coverage[4, 4], 0, 'Manual background must not be replaced')
        self.assertEqual((annotation.coverage[10, 10], annotation.valid[10, 10]), (255, 255))
        self.assertEqual((annotation.coverage[12, 12], annotation.valid[12, 12]), (0, 255))
        self.assertEqual(annotation.valid[20, 20], 0, 'Uncertain pixels stay unknown')

    def test_ai_draft_rejects_nonfinite_out_of_range_and_wrong_shape(self):
        annotation = Annotation(64, 128)
        for probability in (np.full((64, 128), np.nan), np.full((64, 128), 2.), np.zeros((32, 32))):
            with self.subTest(shape=probability.shape), self.assertRaises(ValueError):
                annotation.apply_suggestion(probability)
        self.assertFalse(annotation.valid.any())

    def test_new_projects_never_overwrite_existing_directories(self):
        folder = self.root / 'existing'
        folder.mkdir()
        (folder / 'protected.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'never replaced'):
            Project.create(folder)
        self.assertEqual((folder / 'protected.txt').read_text(), 'keep')

    def test_source_and_exact_frame_copy_leakage_are_rejected(self):
        project, sample, _ = self.project()
        with self.assertRaisesRegex(ValueError, 'another split'):
            project.add_sample(self.frames(), 'recording-a', 'val', [3, 4, 5], {})
        frames, _ = project.load_sample(sample)
        with self.assertRaisesRegex(ValueError, 'Identical frames'):
            project.add_sample(frames, 'renamed-recording', 'val', [3, 4, 5], {})
        self.assertEqual(len(Project.open(project.root).samples), 2)

    def test_duplicate_window_and_nonconsecutive_ids_are_rejected(self):
        project, sample, _ = self.project()
        with self.assertRaisesRegex(ValueError, 'already in the project'):
            project.add_sample(self.frames(), sample['source'], 'train', [0, 1, 2], {})
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            project.add_sample(self.frames(), 'c', 'train', [0, 2, 4], {})

    def test_renamed_video_cannot_change_split_even_for_different_frames(self):
        project = Project.create(self.root / 'project')
        origin = {'kind': 'video', 'sha256': 'a' * 64}
        project.add_sample(self.frames(), 'first-path', 'train', [0, 1, 2], origin)
        with self.assertRaisesRegex(ValueError, 'renamed copy'):
            project.add_sample(self.frames(), 'different-path', 'val', [9, 10, 11], origin)

    def test_unreviewed_and_empty_annotations_cannot_form_a_dataset(self):
        project, sample, _ = self.project()
        with self.assertRaisesRegex(ValueError, 'Review at least'):
            project.export(self.root / 'export')
        _, annotation = project.load_sample(sample)
        with self.assertRaisesRegex(ValueError, 'fully known'):
            project.save_annotation(sample, annotation, reviewed=True)
        project.save_annotation(sample, annotation, reviewed=False)
        self.assertFalse(Project.open(project.root).samples[0]['reviewed'])

    def test_reviewed_export_matches_strict_training_loader(self):
        from mefi.ui_data import Labels, UiFrames
        project, a, b = self.project()
        self.review(project, a)
        self.review(project, b)
        path = project.export(self.root / 'dataset')
        labels = Labels(path)
        self.assertEqual(labels.identity['counts'], {'train': 1, 'val': 1, 'test': 0})
        frames, target, known = UiFrames(labels, 'train', 64)[0]
        self.assertEqual(tuple(frames.shape), (9, 64, 128))
        self.assertEqual(tuple(target.shape), (1, 32, 64))
        self.assertTrue(bool(known.all()))
        self.assertTrue((path.parent / 'provenance.json').is_file())

    def test_export_is_immutable_and_draft_samples_are_excluded(self):
        project, a, b = self.project()
        self.review(project, a)
        self.review(project, b)
        project.add_sample(self.frames(), 'c', 'train', [0, 1, 2], {})
        path = project.export(self.root / 'dataset')
        before = {str(p): p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'existing files'):
            project.export(path.parent)
        _, annotation = project.load_sample(a)
        annotation.coverage[:] = 0
        project.save_annotation(a, annotation, reviewed=False)
        self.assertEqual(before, {str(p): p.read_bytes() for p in path.parent.rglob('*') if p.is_file()})
        self.assertEqual(len(json.loads(path.read_text())['samples']), 2)

    def test_export_cannot_live_inside_editable_project(self):
        project, a, b = self.project()
        self.review(project, a)
        self.review(project, b)
        with self.assertRaisesRegex(ValueError, 'not inside'):
            project.export(project.root / 'nested')

    def test_project_detects_tampered_frames_before_export(self):
        project, a, b = self.project()
        self.review(project, a)
        self.review(project, b)
        (project.root / a['frames'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Project file changed'):
            project.export(self.root / 'refused')
        self.assertFalse((self.root / 'refused').exists())

    def test_project_refuses_external_edits_and_escaping_file_references(self):
        project, a, _ = self.project()
        path = project.root / 'project.json'
        path.write_text(path.read_text() + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'another editor'):
            project.persist()
        doc = json.loads(path.read_text())
        doc['samples'][0]['frames'][0]['path'] = '../../escape.png'
        path.write_text(json.dumps(doc), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'relative paths'):
            Project.open(project.root)

    def test_project_rejects_traversal_sample_ids_and_nonboolean_review(self):
        project, a, _ = self.project()
        path = project.root / 'project.json'
        original = json.loads(path.read_text())
        document = json.loads(path.read_text())
        document['samples'][0]['id'] = '../../escape'
        path.write_text(json.dumps(document), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Invalid/duplicate'):
            Project.open(project.root)
        original['samples'][0]['reviewed'] = 'yes'
        path.write_text(json.dumps(original), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'explicit booleans'):
            Project.open(project.root)

    def test_relative_reference_cannot_reenter_project_via_parent(self):
        project, a, _ = self.project()
        path = project.root / 'project.json'
        document = json.loads(path.read_text())
        ref = document['samples'][0]['frames'][0]
        ref['path'] = '../project/' + ref['path']
        path.write_text(json.dumps(document), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'relative paths'):
            Project.open(project.root)

    def test_explicit_resize_changes_only_prepared_copies(self):
        frames = [self.rng.integers(0, 256, (72, 128, 3), np.uint8) for _ in range(3)]
        original = frames[0].copy()
        with self.assertRaisesRegex(ValueError, 'explicit resizing'):
            prepare_frames(frames)
        result = prepare_frames(frames, (128, 64))
        self.assertEqual(result[0].shape, (64, 128, 3))
        self.assertTrue(np.array_equal(frames[0], original))

    def test_image_import_and_locked_capture_guard(self):
        from mefi import ui_labeling
        paths = []
        for i, frame in enumerate(self.frames()):
            path = self.root / f'frame{i}.png'
            ok, data = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.assertTrue(ok)
            path.write_bytes(data.tobytes())
            paths.append(path)
        result, info = image_window(paths, 0)
        self.assertEqual(result[0].shape, (64, 128, 3))
        self.assertEqual(info['sha256'][0], file_hash(paths[0]))
        with patch.object(ui_labeling, 'LOCKED_CAPTURE', self.root):
            with self.assertRaisesRegex(ValueError, 'locked regression'):
                image_window(paths, 0)

    def test_generated_video_decodes_exact_selected_ordinals_and_preserves_source(self):
        path = self.root / 'generated.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (128, 64))
        self.assertTrue(writer.isOpened(), 'Generated MJPG fixture encoder unavailable')
        for index in range(6):
            writer.write(np.full((64, 128, 3), index * 30, np.uint8))
        writer.release()
        before = path.read_bytes()
        frames, info = video_window(path, 4)
        self.assertEqual(info['frame_ids'], [2, 3, 4])
        self.assertEqual(info['source_id'], source_id(path))
        for frame, expected in zip(frames, (60, 90, 120)):
            self.assertLess(abs(float(frame.mean()) - expected), 3)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'ended before'):
            video_window(path, 8)

    def test_ai_output_semantics_and_coordinate_restore(self):
        probability = probability_map(np.zeros((1, 1, 8, 8), np.float32), 'logits', (64, 128))
        self.assertEqual(probability.shape, (64, 128))
        self.assertTrue(np.all(probability == .5))
        for raw in (np.full((1, 1, 8, 8), 2.), np.full((1, 1, 8, 8), np.nan)):
            with self.assertRaises(ValueError):
                probability_map(raw, 'probabilities', (64, 128))

    def test_local_generated_onnx_inference_uses_cpu_and_preserves_originals(self):
        path = self.root / 'generated.onnx'
        toy_model(path)
        before = path.read_bytes()
        frame = np.zeros((64, 128, 3), np.uint8)
        frame[:, :64, 0] = 255
        probability, info = suggest(frame, path, file_hash(path))
        self.assertEqual(probability.shape, (64, 128))
        self.assertGreater(float(probability[32, 10]), .99)
        self.assertLess(float(probability[32, 110]), .01)
        self.assertEqual(info['backend'], 'OpenCV DNN CPU')
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(np.all(frame[:, :64, 0] == 255))

    def test_ai_model_requires_exact_pin_dense_contract_and_embedded_weights(self):
        path = self.root / 'generated.onnx'
        toy_model(path)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            inspect_model(path, '0' * 64)
        toy_model(path, outputs=2)
        with self.assertRaisesRegex(ValueError, 'one RGB input'):
            inspect_model(path, file_hash(path))
        toy_model(path, external=True)
        with self.assertRaisesRegex(ValueError, 'External ONNX'):
            inspect_model(path, file_hash(path))

    def test_worker_accepts_only_generated_requested_images_and_no_training(self):
        from mefi.ui_label_worker import main
        paths = []
        for i, frame in enumerate(self.frames()):
            path = self.root / f'{i}.png'
            ok, data = cv2.imencode('.png', frame)
            self.assertTrue(ok)
            path.write_bytes(data.tobytes())
            paths.append(str(path))
        request = self.root / 'request.json'
        request.write_text(json.dumps(dict(action='images', paths=paths, first=0, resize=None)))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--request', str(request)]), 0)
        with np.load(self.root / 'result.npz', allow_pickle=False) as data:
            self.assertEqual(set(data.files), {'frame0', 'frame1', 'frame2'})
        self.assertEqual(json.loads((self.root / 'info.json').read_text())['kind'], 'explicit-images')
        before = (self.root / 'result.npz').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--request', str(request)]), 1)
        self.assertEqual((self.root / 'result.npz').read_bytes(), before)


class BulkLabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generated-bulk-labels-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def staged(self, **kwargs):
        path = generated_video(self.root)
        folder = self.root / 'stage'
        folder.mkdir()
        info = prepare_video_batch(path, folder, count=4, **kwargs)
        return path, folder, info

    def test_positions_cover_range_without_overlapping_triplets(self):
        self.assertEqual(window_positions(2, 23, 4), [2, 9, 16, 23])
        self.assertEqual(window_positions(2, 6, 20), [2, 6])
        self.assertEqual(window_positions(2, 2, 20), [2])
        for count in range(1, 101):
            positions = window_positions(7, 400, count)
            self.assertEqual(positions[0], 7)
            self.assertTrue(all(b - a >= 3 for a, b in zip(positions, positions[1:])))
            if count > 1:
                self.assertEqual(positions[-1], 400)
        for args in ((1, 20, 2), (3, 2, 2), (2, 20, 0), (2, 20, 101), (2, 20, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                window_positions(*args)

    def test_sequential_batch_retains_exact_ordinals_hashes_and_original(self):
        path = generated_video(self.root)
        before = path.read_bytes()
        progress = []
        info = prepare_video_batch(path, self.root / 'stage', count=4,
                                   progress=lambda *args: progress.append(args))
        self.assertEqual(info['prepared'], 4)
        self.assertEqual(info['last'], 23)
        self.assertFalse(info['warnings'])
        self.assertTrue(progress)
        for record, current in zip(info['samples'], (2, 9, 16, 23)):
            self.assertEqual(record['frame_ids'], [current - 2, current - 1, current])
            self.assertEqual(record['provenance']['sha256'], file_hash(path))
            for ref, ordinal in zip(record['frames'], record['frame_ids']):
                image = cv2.imread(str(self.root / 'stage' / ref['path']))
                self.assertEqual(file_hash(self.root / 'stage' / ref['path']), ref['sha256'])
                self.assertLess(abs(float(image.mean()) - ordinal * 8), 3)
        self.assertEqual(path.read_bytes(), before)

    def test_explicit_range_and_early_eof_warn_without_inventing_frames(self):
        path, folder, info = self.staged(first=5, last=14)
        self.assertEqual([r['frame_ids'][2] for r in info['samples']], [5, 8, 11, 14])
        info = prepare_video_batch(path, self.root / 'past-end', count=20, last=100)
        self.assertTrue(info['warnings'])
        self.assertTrue(all(r['frame_ids'][2] < 24 for r in info['samples']))
        with self.assertRaisesRegex(ValueError, 'No complete window'):
            prepare_video_batch(path, self.root / 'empty', count=2, first=100, last=110)

    def test_missing_metadata_requests_explicit_end_and_releases_decoder(self):
        from unittest.mock import Mock
        path = self.root / 'video.marker'
        path.write_bytes(b'not decoded; mocked reader')
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = float('nan')
        with patch('mefi.ui_label_batch.cv2.VideoCapture', return_value=cap):
            with self.assertRaisesRegex(ValueError, 'explicit last'):
                prepare_video_batch(path, self.root / 'stage')
        cap.read.assert_not_called()
        cap.release.assert_called_once()

    def test_one_window_does_not_decode_unused_remainder_of_video(self):
        from unittest.mock import Mock
        path = self.root / 'video.marker'
        path.write_bytes(b'mocked long recording')
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 100000
        cap.read.side_effect = [(True, np.zeros((64, 128, 3), np.uint8)) for _ in range(3)]
        with patch('mefi.ui_label_batch.cv2.VideoCapture', return_value=cap):
            info = prepare_video_batch(path, self.root / 'stage', count=1)
        self.assertEqual(info['prepared'], 1)
        self.assertEqual(cap.read.call_count, 3)
        cap.release.assert_called_once()

    def test_changed_source_and_locked_regression_capture_are_refused(self):
        from mefi import ui_label_batch, ui_labeling
        path = generated_video(self.root)
        stamp = path.stat()
        def hash_then_change_stamp(value):
            digest = file_hash(value)
            os.utime(value, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000000))
            return digest
        with patch.object(ui_label_batch, 'file_hash', side_effect=hash_then_change_stamp):
            with self.assertRaisesRegex(ValueError, 'Source video changed'):
                prepare_video_batch(path, self.root / 'stage', count=4)
        with patch.object(ui_labeling, 'LOCKED_CAPTURE', self.root):
            with self.assertRaisesRegex(ValueError, 'locked regression'):
                prepare_video_batch(path, self.root / 'locked', count=4)
        self.assertFalse((self.root / 'locked').exists())

    def test_staging_memory_and_disk_budget_are_bounded(self):
        from mefi import ui_label_batch
        path = generated_video(self.root)
        with patch.object(ui_label_batch, 'MAX_BATCH_BYTES', 3 * 64 * 128 * 3 + 32):
            info = prepare_video_batch(path, self.root / 'stage', count=4)
        self.assertEqual(info['prepared'], 1)
        self.assertIn('512 MiB', info['warnings'][0])
        self.assertEqual(len(list((self.root / 'stage').rglob('*.png'))), 3)
        with patch.object(ui_label_batch, 'MAX_BATCH_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'smaller prepared size'):
                prepare_video_batch(path, self.root / 'too-big', count=4)
        self.assertFalse((self.root / 'too-big').exists())

    def test_project_commits_batch_once_skips_duplicates_and_leaves_unknown(self):
        _, folder, info = self.staged()
        project = Project.create(self.root / 'project')
        with patch.object(project, 'persist', wraps=project.persist) as persist:
            added = project.add_prepared_samples(folder, info['samples'], 'recording', 'train')
        persist.assert_called_once()
        self.assertEqual(len(added), 4)
        self.assertEqual(len(Project.open(project.root).samples), 4)
        for sample in added:
            _, annotation = project.load_sample(sample)
            self.assertFalse(sample['reviewed'])
            self.assertFalse(annotation.valid.any())
        before = (project.root / 'project.json').read_bytes()
        self.assertEqual(project.add_prepared_samples(folder, info['samples'], 'recording', 'train'), [])
        self.assertEqual((project.root / 'project.json').read_bytes(), before)

    def test_last_window_tampering_refuses_entire_batch_before_writing(self):
        _, folder, info = self.staged()
        project = Project.create(self.root / 'project')
        before = (project.root / 'project.json').read_bytes()
        (folder / info['samples'][-1]['frames'][-1]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Project file changed'):
            project.add_prepared_samples(folder, info['samples'], 'recording', 'train')
        self.assertFalse(project.samples)
        self.assertEqual((project.root / 'project.json').read_bytes(), before)
        self.assertFalse((project.root / 'frames').exists())

    def test_late_frame_leakage_refuses_all_windows(self):
        _, folder, info = self.staged()
        project = Project.create(self.root / 'project')
        record = info['samples'][-1]
        frames = [cv2.cvtColor(cv2.imread(str(folder / ref['path'])), cv2.COLOR_BGR2RGB) for ref in record['frames']]
        project.add_sample(frames, 'separate-recording', 'val', record['frame_ids'], {})
        before = (project.root / 'project.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Identical frames'):
            project.add_prepared_samples(folder, info['samples'], 'new-recording', 'train')
        self.assertEqual(len(project.samples), 1)
        self.assertEqual((project.root / 'project.json').read_bytes(), before)
        self.assertEqual(len(list((project.root / 'frames').rglob('*.png'))), 3)

    def test_source_and_renamed_video_cannot_cross_splits(self):
        _, folder, info = self.staged()
        project = Project.create(self.root / 'project')
        project.add_prepared_samples(folder, info['samples'][:1], 'recording', 'train')
        with self.assertRaisesRegex(ValueError, 'another split'):
            project.add_prepared_samples(folder, info['samples'], 'recording', 'val')
        with self.assertRaisesRegex(ValueError, 'renamed copy'):
            project.add_prepared_samples(folder, info['samples'][1:], 'renamed', 'val')
        self.assertEqual(len(project.samples), 1)

    def test_failed_publication_rolls_back_only_new_files_and_samples(self):
        _, folder, info = self.staged()
        project = Project.create(self.root / 'project')
        project.add_prepared_samples(folder, info['samples'][:1], 'recording', 'train')
        before = {str(p.relative_to(project.root)): p.read_bytes() for p in project.root.rglob('*') if p.is_file()}
        with patch.object(project, 'persist', side_effect=ValueError('injected publish failure')):
            with self.assertRaisesRegex(ValueError, 'publish failure'):
                project.add_prepared_samples(folder, info['samples'][1:], 'recording', 'train')
        self.assertEqual(len(project.samples), 1)
        self.assertEqual(before, {str(p.relative_to(project.root)): p.read_bytes() for p in project.root.rglob('*') if p.is_file()})

    def test_external_project_edit_and_bad_batch_references_are_refused(self):
        _, folder, info = self.staged()
        project = Project.create(self.root / 'project')
        bad = copy.deepcopy(info['samples'])
        bad[-1]['frames'][0]['path'] = '../escape.png'
        with self.assertRaisesRegex(ValueError, 'relative paths'):
            project.add_prepared_samples(folder, bad, 'recording', 'train')
        path = project.root / 'project.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'another editor'):
            project.add_prepared_samples(folder, info['samples'], 'recording', 'train')
        self.assertFalse(project.samples)
        self.assertFalse((project.root / 'frames').exists())

    def test_hud_reuse_only_copies_matching_foreground_preserves_known_and_undoes(self):
        source = Annotation(64, 128)
        source.valid[:] = 255
        source.coverage[10:14, 20:24] = 160
        reference = np.full((64, 128, 3), 250, np.uint8)
        frames = [reference.copy() for _ in range(3)]
        frames[0][10, 20] = 0
        frames[1][10, 21] = 230
        frames[2][10, 22] = 238  # Exactly the inclusive 12-level tolerance.
        target = Annotation(64, 128)
        target.valid[11, 20] = 255  # Manual background must win.
        target.valid[11, 21], target.coverage[11, 21] = 255, 255
        before = (target.valid.copy(), target.coverage.copy())
        self.assertEqual(target.reuse_hud(source, reference, frames), 12)
        self.assertEqual((target.valid[10, 20], target.valid[10, 21]), (0, 0))
        self.assertEqual(target.coverage[10, 22], 160)
        self.assertEqual(target.coverage[11, 20], 0)
        self.assertEqual(target.coverage[11, 21], 255)
        self.assertEqual(target.valid[30, 30], 0, 'Template background must never be copied')
        self.assertTrue(np.all(source.valid == 255))
        self.assertTrue(target.undo())
        self.assertTrue(np.array_equal(target.valid, before[0]))
        self.assertTrue(np.array_equal(target.coverage, before[1]))
        self.assertTrue(target.undo(True))
        self.assertEqual(target.coverage[10, 22], 160)
        for tolerance in (-1, 33, True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                target.reuse_hud(source, reference, frames, tolerance)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import app
        cls.ui = app
        cls.application = app.QApplication.instance() or app.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generated-label-interface-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = patch.object(self.ui, 'load_settings', return_value={'dataset': str(self.root/'no-dataset')})
        self.writes = patch.object(self.ui, 'save_settings')
        self.device = patch.object(self.ui.Trainer, 'detect_device')
        def no_jobs(*args, **kwargs):
            raise AssertionError('No unmocked jobs may start in GUI tests')
        self.jobs = patch.object(self.ui.Job, 'start', no_jobs)
        for context in (self.settings, self.writes, self.device, self.jobs):
            context.start()
            self.addCleanup(context.stop)
        self.window = self.ui.Trainer()
        self.page = self.window.label_ui
        project = Project.create(self.root/'project')
        rng = np.random.default_rng(1)
        self.sample = project.add_sample([rng.integers(0, 256, (64, 128, 3), np.uint8) for _ in range(3)],
                                         'generated-recording', 'train', [0, 1, 2], {})
        self.page.set_project(project)
        self.page.select_sample(0)
        self.addCleanup(self.close_window)

    def close_window(self):
        self.page.dirty = False
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()

    def test_navigation_preserves_old_indices_and_adds_label_page(self):
        self.assertIs(self.window.pages.widget(1), self.window.train)
        self.assertIs(self.window.pages.widget(2), self.window.export)
        self.assertIs(self.window.pages.widget(3), self.page)
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertIsNone(self.page.worker_temp)

    def test_canvas_paints_only_current_frame_and_invalidates_review(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest
        canvas = self.page.canvas
        canvas.set_zoom(1)
        self.page.reviewed.setChecked(True)
        QTest.mousePress(canvas, self.ui.Qt.LeftButton, pos=QPoint(20, 20))
        QTest.mouseRelease(canvas, self.ui.Qt.LeftButton, pos=QPoint(20, 20))
        self.assertEqual(canvas.annotation.coverage[20, 20], 255)
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertTrue(self.page.dirty)
        self.page.undo()
        self.assertFalse(canvas.annotation.valid.any())
        self.page.frame_view.setCurrentIndex(0)
        QTest.mousePress(canvas, self.ui.Qt.LeftButton, pos=QPoint(20, 20))
        QTest.mouseRelease(canvas, self.ui.Qt.LeftButton, pos=QPoint(20, 20))
        self.assertFalse(canvas.annotation.valid.any(), 'Previous frames must be view-only')

    def test_suggestions_are_separate_until_applied_and_never_auto_reviewed(self):
        probability = np.full((64, 128), .95, np.float32)
        self.page.set_suggestion(probability, {'kind': 'generated-test'})
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertFalse(self.page.dirty)
        self.page.apply_draft()
        self.assertTrue(self.page.canvas.annotation.valid.all())
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertTrue(self.page.save_current())
        self.assertFalse(self.sample['reviewed'])
        self.assertEqual(self.sample['guidance'][0]['positive_threshold'], .85)

    def test_manual_review_save_and_later_edit_require_fresh_review(self):
        annotation = self.page.canvas.annotation
        annotation.stroke((20, 20), (22, 20), 2, 'ui')
        self.page.edited()
        self.page.reviewed.setChecked(True)
        self.assertTrue(self.page.save_current())
        self.assertTrue(self.sample['reviewed'])
        self.page.edited()
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertTrue(self.page.save_current())
        self.assertFalse(self.sample['reviewed'])

    def test_use_manifest_only_populates_training_path_without_launching_or_changing_mode(self):
        self.page.exported_manifest = self.root/'generated-labels.json'
        mode = self.window.train.ui_mode.currentData()
        with patch.object(self.window.train.job, 'start') as launch:
            self.page.use_manifest()
            launch.assert_not_called()
        self.assertEqual(self.window.train.ui_labels.text(), str(self.page.exported_manifest))
        self.assertEqual(self.window.train.ui_mode.currentData(), mode)

    def test_explicit_ai_request_is_pinned_and_cancellable_without_accepting_results(self):
        path = self.root/'model.onnx'
        path.write_bytes(b'GUI test marker; no inference will run')
        self.page.model.setText(str(path))
        self.page.model_hash = file_hash(path)
        with patch.object(self.page.job, 'start') as launch:
            self.page.suggest()
        self.assertEqual(launch.call_args.args[0], 'mefi.ui_label_worker')
        request = json.loads(Path(self.page.worker_temp.name, 'request.json').read_text())
        self.assertEqual(request['sha256'], file_hash(path))
        self.assertEqual(request['action'], 'suggest')
        self.assertFalse(self.page.controls.isEnabled())
        with patch.object(self.page.job, 'kill') as kill:
            self.page.timeout()
            kill.assert_called_once()
        self.page.worker_finished(1)
        self.assertTrue(self.page.controls.isEnabled())
        self.assertIsNone(self.page.worker_temp)
        self.assertIsNone(self.page.canvas.suggestion)
        self.assertIn('five-minute', self.page.status.text())

    def test_unsaved_cancel_does_not_switch_samples_or_discard_work(self):
        from PySide6.QtWidgets import QMessageBox
        self.page.dirty = True
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Cancel):
            self.assertFalse(self.page.unsaved_ok())
            self.assertFalse(self.page.prepare_close())
        self.assertTrue(self.page.dirty)

    def test_save_during_sample_switch_keeps_list_and_canvas_on_same_sample(self):
        from PySide6.QtWidgets import QMessageBox
        rng = np.random.default_rng(77)
        second = self.page.project.add_sample([rng.integers(0, 256, (64, 128, 3), np.uint8) for _ in range(3)],
                                              'other-recording', 'val', [0, 1, 2], {})
        self.page.refresh_samples(0)
        self.page.canvas.annotation.stroke((20, 20), (20, 20), 0, 'ui')
        self.page.edited()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Save):
            self.page.samples.setCurrentRow(1)
        self.assertIs(self.page.sample, second)
        self.assertEqual(self.page.samples.currentRow(), 1)
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.page.canvas.annotation.stroke((20, 20), (20, 20), 0, 'ui')
        self.page.edited()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Discard):
            self.assertTrue(self.page.unsaved_ok())
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertFalse(self.page.dirty)

    def run_generated_worker(self, request, frame=None):
        import sys
        from PySide6.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        self.page.job.proc.finished.connect(loop.quit)
        def launch(module, args):
            self.assertEqual(module, 'mefi.ui_label_worker', 'Only the label worker is permitted')
            self.page.job.proc.start(sys.executable, ['-m', module, *args])
        with patch.object(self.page.job, 'start', side_effect=launch):
            self.page.start_worker(request, frame)
        timer.start(15000)
        loop.exec()
        timer.stop()
        self.application.processEvents()
        self.assertFalse(self.page.job.running(), 'Generated labeling worker did not finish in 15 s')
        self.assertIsNone(self.page.worker_temp)

    def test_generated_image_worker_roundtrip_imports_an_unknown_draft(self):
        rng = np.random.default_rng(5)
        paths = []
        for index in range(3):
            path = self.root / f'generated-frame{index}.png'
            ok, data = cv2.imencode('.png', rng.integers(0, 256, (64, 128, 3), np.uint8))
            self.assertTrue(ok)
            path.write_bytes(data.tobytes())
            paths.append(str(path))
        self.run_generated_worker(dict(action='images', paths=paths, first=0, resize=None,
                                       source='generated-validation', split='val'))
        self.assertEqual(len(self.page.project.samples), 2)
        self.assertEqual(self.page.sample['split'], 'val')
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertFalse(self.page.sample['reviewed'])

    def test_batch_button_builds_explicit_request_without_changing_existing_samples(self):
        path = generated_video(self.root)
        self.page.video.setText(str(path))
        self.page.recording.setText('selected-recording')
        self.page.batch_count.setValue(4)
        self.page.current.setValue(5)
        self.page.batch_last.setValue(20)
        with patch.object(self.page.job, 'start'):
            self.page.import_video_batch()
        request = json.loads(Path(self.page.worker_temp.name, 'request.json').read_text())
        self.assertEqual(request['action'], 'video_batch')
        self.assertEqual((request['first'], request['last'], request['count']), (5, 20, 4))
        self.assertEqual(len(self.page.project.samples), 1)
        with patch.object(self.page.job, 'kill'):
            self.page.cancel_worker()
        self.page.worker_finished(1)
        self.assertEqual(len(self.page.project.samples), 1)
        self.assertIsNone(self.page.worker_temp)

    def test_generated_batch_worker_roundtrip_imports_many_unknown_windows_and_skips_duplicates(self):
        path = generated_video(self.root)
        before = path.read_bytes()
        request = dict(action='video_batch', path=str(path), first=2, last=None, count=4,
                       resize=None, source='generated-batch', split='train')
        self.run_generated_worker(request)
        self.assertEqual(len(self.page.project.samples), 5)
        self.assertEqual(self.page.sample['frame_ids'], [0, 1, 2])
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertIn('Imported 4', self.page.status.text())
        self.assertEqual(path.read_bytes(), before)
        self.run_generated_worker(request)
        self.assertEqual(len(self.page.project.samples), 5)
        self.assertIn('skipped 4 duplicates', self.page.status.text())

    def test_cancel_after_staging_accepts_no_windows(self):
        from mefi.ui_label_worker import main
        path = generated_video(self.root)
        request = dict(action='video_batch', path=str(path), first=2, last=23, count=4,
                       resize=None, source='generated-batch', split='train')
        with patch.object(self.page.job, 'start'):
            self.page.start_worker(request)
        folder = Path(self.page.worker_temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--request', str(folder / 'request.json')]), 0)
        self.page.worker_done = True
        with patch.object(self.page.job, 'kill'):
            self.page.cancel_worker()
        self.page.worker_finished(0)
        self.assertEqual(len(self.page.project.samples), 1)
        self.assertFalse(folder.exists())
        self.assertIn('cancelled', self.page.status.text())

    def pin_generated_hud(self):
        self.page.canvas.annotation.valid[:] = 255
        self.page.canvas.annotation.coverage[10:14, 20:24] = 160
        self.page.edited()
        self.page.reviewed.setChecked(True)
        self.assertTrue(self.page.save_current())
        self.page.pin_hud_template()
        self.assertIsNotNone(self.page.hud_template)
        return self.page.canvas.frames[2].copy()

    def test_hud_template_requires_saved_review_and_does_not_automatically_apply(self):
        self.page.pin_hud_template()
        self.assertIsNone(self.page.hud_template)
        reference = self.pin_generated_hud()
        project = self.page.project
        sample = project.add_sample([reference.copy() for _ in range(3)], self.sample['source'], 'train', [3, 4, 5], {})
        self.page.refresh_samples(0)
        self.assertFalse(self.page.auto_hud.isChecked())
        self.page.select_sample(1)
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.page.apply_hud_template()
        self.assertEqual(int((self.page.canvas.annotation.valid > 0).sum()), 16)
        self.assertEqual(self.page.canvas.annotation.coverage[10, 20], 160)
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertTrue(self.page.dirty)
        self.assertFalse(project.load_sample(sample)[1].valid.any(), 'Draft should be unsaved')
        self.assertTrue(self.sample['reviewed'])

    def test_auto_reuse_save_next_requires_review_and_only_saves_explicitly(self):
        reference = self.pin_generated_hud()
        project = self.page.project
        for first in (3, 6):
            project.add_sample([reference.copy() for _ in range(3)], self.sample['source'], 'train',
                               [first, first + 1, first + 2], {})
        self.page.auto_hud.setChecked(True)
        self.page.refresh_samples(0)
        self.page.select_sample(1)
        self.assertTrue(self.page.dirty)
        self.assertFalse(self.page.reviewed.isChecked())
        self.page.save_next()
        self.assertIs(self.page.sample, project.samples[1])
        self.assertFalse(project.samples[1]['reviewed'])
        self.page.reviewed.setChecked(True)
        self.page.save_next()
        self.assertTrue(project.samples[1]['reviewed'])
        self.assertIs(self.page.sample, project.samples[2])
        self.assertEqual(self.page.samples.currentRow(), 2)
        self.assertTrue(self.page.dirty)
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertFalse(project.load_sample(project.samples[2])[1].valid.any())
        self.assertTrue(project.samples[1]['guidance'][0]['review_required'])
        self.assertEqual(project.samples[1]['guidance'][0]['revision'], self.sample['revision'])

    def test_hud_template_never_crosses_recording_or_size_and_resets_on_project_change(self):
        reference = self.pin_generated_hud()
        project = self.page.project
        project.add_sample([reference.copy() for _ in range(3)], 'different-recording', 'train', [3, 4, 5], {})
        project.add_sample([np.zeros((64, 64, 3), np.uint8) for _ in range(3)], self.sample['source'], 'train', [6, 7, 8], {})
        self.page.auto_hud.setChecked(True)
        self.page.refresh_samples(0)
        for row in (1, 2):
            self.page.select_sample(row)
            self.page.apply_hud_template()
            self.assertFalse(self.page.canvas.annotation.valid.any())
            self.assertFalse(self.page.dirty)
        self.page.set_project(Project.create(self.root / 'another-project'))
        self.assertIsNone(self.page.hud_template)
        self.assertFalse(self.page.auto_hud.isChecked())

    def test_generated_model_worker_roundtrip_is_only_a_suggestion(self):
        path = self.root / 'generated.onnx'
        toy_model(path)
        self.run_generated_worker(dict(action='suggest', model=str(path), sha256=file_hash(path),
                                       normalization='RGB 0..1', output_kind='probabilities'), self.page.canvas.frames[2])
        self.assertIsNotNone(self.page.canvas.suggestion)
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertFalse(self.page.dirty)

    def test_gui_exports_reviewed_copies_and_never_starts_training(self):
        project = self.page.project
        rng = np.random.default_rng(99)
        project.add_sample([rng.integers(0, 256, (64, 128, 3), np.uint8) for _ in range(3)],
                           'other-recording', 'val', [0, 1, 2], {})
        for sample in project.samples:
            _, annotation = project.load_sample(sample)
            annotation.valid[:] = 255
            project.save_annotation(sample, annotation, reviewed=True)
        destination = self.root / 'dataset-export'
        with patch.object(self.ui.QFileDialog, 'getExistingDirectory', return_value=str(destination)), \
                patch.object(self.window.train.job, 'start') as training:
            self.page.export_dataset()
            training.assert_not_called()
        self.assertEqual(self.page.exported_manifest, destination / 'labels.json')
        self.assertTrue(self.page.use_button.isEnabled())
        self.assertEqual(len(json.loads((destination/'labels.json').read_text())['samples']), 2)


if __name__ == '__main__':
    unittest.main()
