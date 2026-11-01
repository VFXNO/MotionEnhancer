"""Generated/mock-only EfficientSAM adapter tests; network/training forbidden.

Authorized real pinned-model checks on generated images are run separately and
recorded explicitly; this suite does not need a download or installed model.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mefi import ui_sam as sam
from mefi.ui_labeling import Annotation, digest_bytes
from test_ui_labeling import GuiTests as LabelGuiBase


class SamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generated-sam-adapter-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.network = patch.object(sam.urllib.request, 'urlopen', side_effect=AssertionError('No real network in tests'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_prompt_scaling_padding_and_negative_semantics(self):
        coords, labels = sam.prompt_arrays((64, 128), [[32, 16], [64, 32]], [1, 0])
        self.assertEqual(coords.shape, (1, 1, 6, 2))
        self.assertEqual(labels.shape, (1, 1, 6))
        np.testing.assert_array_equal(coords[0, 0, :2], [[256, 256], [512, 512]])
        np.testing.assert_array_equal(labels[0, 0], [1, 0, -1, -1, -1, -1])

    def test_box_uses_two_corner_labels_not_a_solid_label(self):
        coords, labels = sam.prompt_arrays((64, 128), box=[8, 4, 120, 60])
        np.testing.assert_array_equal(labels[0, 0], [2, 3, -1, -1, -1, -1])
        np.testing.assert_array_equal(coords[0, 0, :2], [[64, 64], [960, 960]])

    def test_invalid_and_outside_prompts_fail_before_inference(self):
        options = [dict(), dict(points=[[1, 1]], labels=[0]), dict(points=[[1, 1]] * 7, labels=[1] * 7),
                   dict(points=[[128, 2]], labels=[1]), dict(points=[[-1, 0]], labels=[1]),
                   dict(points=[[np.nan, 0]], labels=[1]), dict(points=[[4, 4]], labels=[]),
                   dict(points=[[4, 4]], labels=[2]),
                   dict(box=[2, 2, 2, 8]), dict(box=[-1, 0, 8, 8]), dict(box=[0, 0, 128, 64])]
        for option in options:
            with self.subTest(option=option), self.assertRaises(ValueError):
                sam.prompt_arrays((64, 128), **option)

    def test_mask_iou_dedup_and_stable_quality_order(self):
        a = np.zeros((64, 128), bool)
        a[8:20, 8:20] = True
        b = np.zeros_like(a)
        b[30:40, 60:70] = True
        candidates = [{'mask': a, 'score': .8}, {'mask': a.copy(), 'score': .9}, {'mask': b, 'score': .8}]
        result = sam.retain_candidates(candidates)
        self.assertEqual([x['score'] for x in result], [.9, .8])
        self.assertEqual(sam.mask_iou(a, a), 1.)
        self.assertEqual(sam.mask_iou(a, b), 0.)
        self.assertEqual(len(sam.retain_candidates(candidates, limit=1)), 1)

    def test_selected_mask_admission_leaves_background_unknown_and_preserves_manual(self):
        annotation = Annotation(64, 128)
        annotation.stroke((10, 10), (10, 10), 0, 'background')
        annotation.stroke((11, 10), (11, 10), 0, 'ui', 128)
        mask = np.zeros((64, 128), bool)
        mask[10:14, 10:14] = True
        self.assertEqual(annotation.apply_foreground(mask), 14)
        self.assertEqual(annotation.coverage[10, 10], 0)
        self.assertEqual(annotation.coverage[10, 11], 128)
        self.assertEqual(annotation.valid[30, 30], 0)
        self.assertTrue(annotation.undo())
        self.assertEqual(annotation.valid[12, 12], 0)
        self.assertEqual(annotation.valid[10, 10], 255)
        for invalid in (mask.astype(np.uint8), mask[:32]):
            with self.assertRaises(ValueError):
                annotation.apply_foreground(invalid)

    def test_empty_or_already_known_mask_does_not_create_undo_entries(self):
        annotation = Annotation(64, 128)
        self.assertEqual(annotation.apply_foreground(np.zeros((64, 128), bool)), 0)
        self.assertFalse(annotation.undo_stack)

    def test_download_is_verified_exclusively_published_and_reused_offline(self):
        raw = b'generated pinned artifact'
        path = self.root / 'artifact.onnx'
        events = []
        with patch.object(sam.urllib.request, 'urlopen', return_value=io.BytesIO(raw)) as network:
            sam.download_file('https://example.invalid/generated', path, digest_bytes(raw), len(raw),
                              progress=lambda done, total: events.append((done, total)))
            network.assert_called_once()
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(events[-1], (len(raw), len(raw)))
        sam.download_file('must-not-fetch', path, digest_bytes(raw), len(raw))
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_bad_or_truncated_download_never_publishes_or_leaves_partials(self):
        for raw in (b'short', b'too many bytes', b'00000000'):
            path = self.root / 'bad.onnx'
            with patch.object(sam.urllib.request, 'urlopen', return_value=io.BytesIO(raw)):
                with self.assertRaises(ValueError):
                    sam.download_file('https://example.invalid/generated', path, digest_bytes(b'expected'), 8)
            self.assertFalse(path.exists())
            self.assertFalse(list(self.root.glob('*.partial')))

    def test_download_never_overwrites_existing_mismatched_model(self):
        path = self.root / 'existing.onnx'
        path.write_bytes(b'keep existing')
        with self.assertRaisesRegex(ValueError, 'not overwritten'):
            sam.download_file('no network', path, '0' * 64, 13)
        self.assertEqual(path.read_bytes(), b'keep existing')

    def test_model_bytes_are_pinned_before_opencv_parses_them(self):
        path = self.root / 'generated.onnx'
        path.write_bytes(b'bad')
        with patch.object(sam.cv2.dnn, 'readNetFromONNX') as load:
            with self.assertRaisesRegex(ValueError, 'file size'):
                sam.EfficientSam(path)
            with patch.object(sam, 'MODEL_BYTES', 3), self.assertRaisesRegex(ValueError, 'checksum'):
                sam.EfficientSam(path)
            load.assert_not_called()

    def fake_model(self):
        model = sam.EfficientSam.__new__(sam.EfficientSam)
        model.path = self.root / 'mock-only-not-loaded.onnx'
        model.shape, model.blob = None, None
        from unittest.mock import Mock
        model.net = Mock()
        return model

    def test_preprocess_rgb_scale_and_source_immutability(self):
        model = self.fake_model()
        frame = np.zeros((64, 128, 3), np.uint8)
        frame[:, :, 0] = 255
        original = frame.copy()
        model.set_frame(frame)
        self.assertEqual(model.blob.shape, (1, 3, 1024, 1024))
        self.assertEqual(model.blob.dtype, np.float32)
        self.assertTrue((model.blob[0, 0] == 1).all())
        self.assertFalse(model.blob[0, 1:].any())
        np.testing.assert_array_equal(frame, original)

    def test_predict_restores_binary_masks_and_enforces_clicks(self):
        model = self.fake_model()
        model.set_frame(np.zeros((64, 128, 3), np.uint8))
        logits = np.full((1, 1, 3, 1024, 1024), -1, np.float32)
        logits[0, 0, 0, 256:768, 256:768] = 1
        logits[0, 0, 1] = 1  # Violates the negative click, despite a higher score.
        model.net.forward.return_value = logits, np.array([[[.8, .99, .1]]], np.float32)
        result = model.predict([[64, 32], [2, 2]], [1, 0])
        self.assertEqual(len(result), 1)
        mask = result[0]['mask']
        self.assertEqual(mask.shape, (64, 128))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[32, 64])
        self.assertFalse(mask[2, 2])
        labels = [call.args[0] for call in model.net.setInput.call_args_list if call.args[1] == 'batched_point_labels'][0]
        self.assertEqual(labels[0, 0, 1], 0, 'Negative prompt must not be padding -1')

    def test_invalid_model_outputs_fail_closed(self):
        model = self.fake_model()
        model.set_frame(np.zeros((64, 128, 3), np.uint8))
        for outputs in ((np.zeros((1, 1, 3, 32, 32)), np.zeros((1, 1, 3))),
                        (np.zeros((1, 1, 3, 1024, 1024)), np.full((1, 1, 3), np.nan))):
            model.net.forward.return_value = outputs
            with self.assertRaisesRegex(ValueError, 'invalid mask/quality'):
                model.predict([[10, 10]], [1])

    def test_auto_grid_is_bounded_deduplicated_and_reports_progress(self):
        model = self.fake_model()
        model.shape = (64, 128)
        mask = np.zeros(model.shape, bool)
        mask[4:20, 4:20] = True
        item = dict(mask=mask, score=.9, area=int(mask.sum()), prompt={})
        progress = []
        with patch.object(model, 'predict', return_value=[item]) as predict:
            masks, info = model.propose(grid=3, progress=lambda *values: progress.append(values))
        self.assertEqual(predict.call_count, 9)
        self.assertEqual(len(masks), 1)
        self.assertEqual(progress[-1], (9, 9, 1))
        self.assertEqual(info['grid'], 3)
        self.assertIn('not automatic UI', info['role'])
        with self.assertRaises(ValueError):
            model.propose(grid=30)

    def test_auto_filter_rejects_empty_whole_frame_and_low_quality_masks(self):
        model = self.fake_model()
        model.shape = (64, 128)
        items = [dict(mask=np.ones(model.shape, bool), score=.99, area=64 * 128, prompt={}),
                 dict(mask=np.zeros(model.shape, bool), score=.99, area=0, prompt={}),
                 dict(mask=np.eye(64, 128, dtype=bool), score=.49, area=64, prompt={})]
        with patch.object(model, 'predict', return_value=items):
            masks, info = model.propose(grid=3)
        self.assertEqual(masks.shape, (0, 64, 128))
        self.assertFalse(info['candidates'])

    def test_missing_frame_and_missing_model_are_actionable_failures(self):
        with self.assertRaisesRegex(ValueError, 'not downloaded'):
            sam.EfficientSam(self.root / 'not-downloaded.onnx')
        with self.assertRaisesRegex(ValueError, 'Select a frame'):
            self.fake_model().propose(grid=3)

    def test_bad_frame_contract_is_rejected(self):
        model = self.fake_model()
        for frame in (np.zeros((64, 128), np.uint8), np.zeros((64, 128, 4), np.uint8),
                      np.zeros((64, 128, 3), np.float32)):
            with self.assertRaises(ValueError):
                model.set_frame(frame)

    def test_worker_smart_and_auto_preserve_distinct_request_semantics(self):
        from mefi.ui_label_worker import main
        for action in ('sam_smart', 'sam_auto'):
            folder = self.root / action
            folder.mkdir()
            request = dict(action=action, points=[[10, 10]], labels=[1], box=None, grid=3)
            path = folder / 'request.json'
            path.write_text(json.dumps(request))
            np.savez_compressed(folder / 'input.npz', frame=np.zeros((64, 128, 3), np.uint8))
            masks = np.zeros((1, 64, 128), bool)
            with patch.object(sam, 'EfficientSam') as factory, contextlib.redirect_stdout(io.StringIO()):
                factory.return_value.propose.return_value = masks, {'candidates': [{'score': .9}]}
                self.assertEqual(main(['--request', str(path)]), 0)
                arguments = factory.return_value.propose.call_args.kwargs
                if action == 'sam_smart':
                    self.assertEqual(arguments['points'], [[10, 10]])
                    self.assertEqual(arguments['labels'], [1])
                    self.assertNotIn('grid', arguments)
                else:
                    self.assertEqual(arguments['grid'], 3)
                    self.assertNotIn('points', arguments)
            with np.load(folder / 'result.npz', allow_pickle=False) as data:
                self.assertEqual(data['masks'].dtype, np.bool_)


class SamGuiTests(LabelGuiBase):
    # Reuse fixture setup/cleanup only, not the inherited test methods. The
    # loader below selects methods from this class explicitly.
    def test_sam_clicks_and_box_never_paint_labels(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest
        page = self.page
        canvas = page.canvas
        canvas.set_zoom(1)
        page.interaction.setCurrentIndex(1)
        QTest.mouseClick(canvas, self.ui.Qt.LeftButton, pos=QPoint(30, 20))
        QTest.mouseClick(canvas, self.ui.Qt.RightButton, pos=QPoint(60, 30))
        self.assertEqual(canvas.points, [[30, 20], [60, 30]])
        self.assertEqual(canvas.point_labels, [1, 0])
        self.assertFalse(page.dirty)
        self.assertFalse(canvas.annotation.valid.any())
        page.interaction.setCurrentIndex(2)
        QTest.mousePress(canvas, self.ui.Qt.LeftButton, pos=QPoint(10, 10))
        QTest.mouseRelease(canvas, self.ui.Qt.LeftButton, pos=QPoint(40, 40))
        self.assertEqual(canvas.box, [10, 10, 40, 40])
        self.assertFalse(canvas.annotation.valid.any())

    def candidate(self):
        masks = np.zeros((2, 64, 128), bool)
        masks[0, 10:20, 10:20] = True
        masks[1, 30:40, 30:40] = True
        info = {'kind': 'generated candidates', 'candidates': [{'score': .9, 'prompt': {}}, {'score': .8, 'prompt': {}}]}
        self.page.set_candidates(masks, info)
        return masks

    def test_candidates_start_unchecked_and_preview_is_not_acceptance(self):
        self.candidate()
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertFalse(self.page.dirty)
        self.assertEqual(self.page.candidates.item(0).checkState(), self.ui.Qt.Unchecked)
        self.page.accept_candidates()
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertIn('Check the candidate', self.page.status.text())
        self.assertFalse(self.page.apply_button.isEnabled())
        self.page.apply_draft()  # Must never turn the rest of the image into background.
        self.assertFalse(self.page.canvas.annotation.valid.any())

    def test_only_checked_masks_are_added_to_unknown_pixels_and_review_clears(self):
        self.candidate()
        self.page.canvas.annotation.stroke((12, 12), (12, 12), 0, 'background')
        self.page.reviewed.setChecked(True)
        self.page.candidates.item(0).setCheckState(self.ui.Qt.Checked)
        self.page.accept_candidates()
        annotation = self.page.canvas.annotation
        self.assertEqual(int((annotation.valid != 0).sum()), 100)
        self.assertEqual(annotation.coverage[12, 12], 0)
        self.assertEqual(annotation.coverage[15, 15], 255)
        self.assertEqual(annotation.valid[35, 35], 0)
        self.assertEqual(annotation.valid[60, 100], 0)
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertTrue(self.page.dirty)
        self.assertIn('foreground only', self.page.guidance[0]['admission'])
        self.page.undo()
        self.assertEqual(int((annotation.valid != 0).sum()), 1)

    def test_discard_or_prompt_change_does_not_mutate_labels(self):
        self.candidate()
        self.page.canvas.clear_prompts()
        self.assertIsNone(self.page.candidate_masks)
        self.assertIsNone(self.page.canvas.suggestion)
        self.assertFalse(self.page.canvas.annotation.valid.any())
        self.assertFalse(self.page.dirty)

    def test_smart_and_automatic_buttons_route_only_explicit_frame_to_worker(self):
        with patch.object(self.page, 'start_worker') as worker:
            self.page.smart_masks()
            worker.assert_not_called()
            self.page.canvas.points, self.page.canvas.point_labels = [[20, 20]], [1]
            self.page.smart_masks()
            self.assertEqual(worker.call_args.args[0]['action'], 'sam_smart')
            self.assertEqual(worker.call_args.args[0]['points'], [[20, 20]])
            self.assertEqual(worker.call_args.kwargs['frame'].shape, (64, 128, 3))
            self.page.auto_masks()
            self.assertEqual(worker.call_args.args[0], {'action': 'sam_auto', 'grid': 3})

    def test_download_requires_confirmation_and_never_starts_inference(self):
        from PySide6.QtWidgets import QMessageBox
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.No), patch.object(self.page, 'start_worker') as worker:
            self.page.download_sam()
            worker.assert_not_called()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes), patch.object(self.page, 'start_worker') as worker:
            self.page.download_sam()
            self.assertEqual(worker.call_args.args[0], {'action': 'sam_download'})

    def test_empty_scan_and_invalid_candidate_outputs_never_create_labels(self):
        self.page.set_candidates(np.zeros((0, 64, 128), bool), {'candidates': []})
        self.assertIsNone(self.page.canvas.suggestion)
        self.assertIn('No usable candidate', self.page.status.text())
        with self.assertRaises(ValueError):
            self.page.set_candidates(np.zeros((1, 64, 128), np.uint8), {'candidates': [{'score': .9}]})
        self.assertFalse(self.page.canvas.annotation.valid.any())

    def test_switching_samples_clears_candidates_and_prompt_coordinates(self):
        self.candidate()
        self.page.canvas.points, self.page.canvas.point_labels = [[15, 15]], [1]
        rng = np.random.default_rng(13)
        self.page.project.add_sample([rng.integers(0, 256, (64, 128, 3), np.uint8) for _ in range(3)],
                                     'generated-other-recording', 'val', [0, 1, 2], {})
        self.page.refresh_samples(0)
        self.page.select_sample(1)
        self.assertIsNone(self.page.candidate_masks)
        self.assertFalse(self.page.canvas.points)
        self.assertIsNone(self.page.canvas.suggestion)
        self.assertFalse(self.page.canvas.annotation.valid.any())

    def test_selecting_a_brush_returns_to_paint_without_accepting_candidates(self):
        self.candidate()
        self.page.interaction.setCurrentIndex(1)
        self.page.radius.setValue(1)
        self.assertEqual(self.page.interaction.currentData(), 'paint')
        self.assertFalse(self.page.canvas.annotation.valid.any())


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(SamTests))
    # Explicitly avoid rerunning the inherited baseline tests here.
    suite.addTests(SamGuiTests(name) for name in SamGuiTests.__dict__ if name.startswith('test_'))
    return suite


if __name__ == '__main__':
    unittest.main()
