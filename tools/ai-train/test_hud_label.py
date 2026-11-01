"""HUD labeling (mefi.hud_label, mefi.hud_label_gui) on generated videos only:
a static HUD over a moving scene. No user footage, models or downloads."""
import os
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from mefi import hud_label
from mefi.ui_data import Labels

W, H, FRAMES = 320, 192, 60


def hud_layer():
    """A HUD: outlined text, an icon frame and a bar. Returns colour, mask."""
    color = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    for img, value in ((color, (255, 255, 255)), (mask, 255)):
        cv2.putText(img, 'HP 120', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, value, 2, cv2.LINE_AA)
        cv2.rectangle(img, (250, 140), (300, 180), value, 2)
        cv2.rectangle(img, (20, 160), (150, 172), value, 2)
    return color, mask


def scene(k, seed=1):
    """A textured scene (game-like detail) panning 4 px per frame."""
    rng = np.random.default_rng(seed)
    big = cv2.resize(rng.integers(0, 255, (H // 2, (W + 4 * FRAMES) // 2, 3), dtype=np.uint8),
                     (W + 4 * FRAMES, H), interpolation=cv2.INTER_LINEAR)
    return big[:, 4 * k:4 * k + W].copy()


def write_video(path, with_hud=True, seed=1):
    color, mask = hud_layer()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (W, H))
    for k in range(FRAMES):
        frame = scene(k, seed)
        if with_hud:
            frame[mask > 0] = color[mask > 0]
        writer.write(frame)
    writer.release()
    return mask


class Suggestion(unittest.TestCase):
    def test_finds_the_static_hud_not_the_scene(self):
        color, mask = hud_layer()
        frames = []
        for k in range(24):
            f = scene(k)
            f[mask > 0] = color[mask > 0]
            frames.append(f)
        s = hud_label.suggest(frames) > 0
        hud = mask > 0
        self.assertGreater((s & hud).sum() / hud.sum(), 0.8)
        far = ~cv2.dilate(mask, np.ones((9, 9), np.uint8)).astype(bool)
        self.assertLess((s & far).sum() / far.sum(), 0.01)

    def test_still_scene_teaches_nothing(self):
        frames = [scene(0)] * 12
        self.assertEqual(int((hud_label.suggest(frames) > 0).sum()), 0)

    def test_positions_spread_over_the_video(self):
        positions = hud_label.label_positions(1000, 10, margin=24)
        self.assertEqual(len(positions), 10)
        self.assertGreaterEqual(positions[0], 24)
        self.assertLess(positions[-1], 999)
        self.assertEqual(positions, sorted(set(positions)))


class Projects(unittest.TestCase):
    def test_saved_project_is_a_valid_training_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / 'train_video.avi', tmp / 'val_video.avi'
            write_video(a, seed=1)
            write_video(b, seed=2)   # another recording: frames must differ
            project = hud_label.Project(tmp / 'labels')
            for video, split in ((a, 'train'), (b, 'val')):
                first, window = hud_label.read_window(video, 30, 24)
                coverage = hud_label.suggest(window)
                valid = np.full(coverage.shape, 255, np.uint8)
                project.save(video, 30, window[-3:], coverage, valid, split)
            labels = Labels(project.path)   # the trainer's own validator
            self.assertEqual(labels.identity['counts'], {'train': 1, 'val': 1, 'test': 0})
            item = labels.items['train'][0]
            self.assertEqual(item['frame_ids'], [28, 29, 30])
            self.assertEqual((item['width'], item['height']), (320, 192))
            # One recording, one split.
            first, window = hud_label.read_window(a, 40, 24)
            with self.assertRaises(ValueError):
                project.save(a, 40, window[-3:], coverage, valid, 'val')
            # Saving the same frame again replaces it.
            project.save(a, 30, window[-3:], coverage, valid, 'train')
            self.assertEqual(project.counts()['train'], 1)

    def test_resized_to_model_size_and_new_project_needs_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp / 'v.avi'
            write_video(video)
            (tmp / 'busy').mkdir()
            (tmp / 'busy' / 'something.txt').write_text('x')
            with self.assertRaises(ValueError):
                hud_label.Project(tmp / 'busy')
            project = hud_label.Project(tmp / 'labels')
            first, window = hud_label.read_window(video, 30, 24)
            cov = np.zeros((H, W), np.uint8)
            project.save(video, 30, window[-3:], cov, np.full((H, W), 255, np.uint8), 'train', size=(160, 96))
            s = project.samples()[0]
            self.assertEqual((s['width'], s['height']), (160, 96))
            self.assertEqual(project.coverage(s['id']).shape, (96, 160))


class Page(unittest.TestCase):
    def test_load_paint_undo_save(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtWidgets import QApplication
        from mefi.hud_label_gui import LabelPage
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp / 'game.avi'
            write_video(video)
            page = LabelPage(None)
            page.resize(1200, 800)
            page.open(str(tmp / 'labels'))
            page.set_video(str(video))
            page.count.setValue(3)
            page.load_frames()
            deadline = time.time() + 30
            while page.coverage is None and time.time() < deadline:
                app.processEvents()
                page.poll()
                time.sleep(0.05)
            self.assertIsNotNone(page.coverage, 'the first frame never loaded')
            self.assertGreater(int((page.coverage > 0).sum()), 0, 'no suggestion was painted')
            before = page.coverage.copy()
            page.set_mode('erase')
            page.begin_stroke()
            page.paint(QPointF(0, 0), QPointF(W, H))
            self.assertTrue(page.dirty)
            self.assertFalse(np.array_equal(before, page.coverage))
            page.undo_stroke()
            self.assertTrue(np.array_equal(before, page.coverage))
            page.set_mode('unknown')
            page.begin_stroke()
            page.paint(QPointF(100, 100), QPointF(100, 100))
            self.assertEqual(int(page.valid[100, 100]), 0)
            self.assertTrue(page.save())
            self.assertEqual(page.project.counts()['train'], 1)
            self.assertFalse(page.dirty)
            page.pool.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
