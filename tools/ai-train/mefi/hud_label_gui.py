"""Label UI page: paint the HUD on frames from your gameplay videos.

Pick a video and how many frames to label; the frames are spread over the
video. Each one opens pre-painted by mefi.hud_label.suggest (green), you fix
it with the brush - paint HUD, erase, or mark pixels you are unsure of as
unknown - and Save moves on to the next. The next frame is read and
suggested in the background while you paint the current one. Saved samples
go into the project's labels.json, which the Train page's UI training reads.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from . import hud_label

WINDOW = 24          # frames read before each labeled one, for the suggestion
UNDO_LIMIT = 30
HUD_COLOR = (60, 230, 90)       # RGB of painted HUD
UNKNOWN_COLOR = (255, 170, 40)  # RGB of unknown pixels


def _label(text, name=None, wrap=False):
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def _card(title):
    frame = QFrame()
    frame.setObjectName('card')
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 12)
    lay.setSpacing(8)
    lay.addWidget(_label(title, 'section'))
    return frame, lay


def _model_size():
    """The installed AI model's network size (label at the size the program
    runs the UI model at), from the newest program build's ai_flow.json."""
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[1]
    candidates = list(repo.glob('*/Release/models/ai_flow.json')) + [repo / 'models' / 'ai_flow.json']
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None
    try:
        meta = json.loads(max(candidates, key=lambda c: c.stat().st_mtime).read_text(encoding='utf-8'))
        return int(meta['width']), int(meta['height'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Canvas(QWidget):
    """The frame with its label on top; zoom with the wheel, pan with the
    right or middle button, paint with the left."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 320)
        self.image = None        # QImage of the frame
        self.overlay = None      # QImage view of page.overlay (RGBA)
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.cursor_pos = None
        self.panning = None
        self.painting = False
        self.last = None

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self.image = QImage(self.rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.fit()

    def set_overlay(self, rgba):
        h, w = rgba.shape[:2]
        self.overlay = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        self.update()

    def fit(self):
        if self.image is None:
            return
        sx = self.width() / self.image.width()
        sy = self.height() / self.image.height()
        self.zoom = min(sx, sy)
        self.offset = QPointF((self.width() - self.image.width() * self.zoom) / 2,
                              (self.height() - self.image.height() * self.zoom) / 2)
        self.update()

    def to_image(self, pos):
        return (pos - self.offset) / self.zoom

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#050506'))
        if self.image is None:
            p.setPen(QColor('#8a8a94'))
            p.drawText(self.rect(), Qt.AlignCenter, self.page.empty_text())
            return
        target = QRectF(self.offset.x(), self.offset.y(), self.image.width() * self.zoom,
                        self.image.height() * self.zoom)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 1)
        p.drawImage(target, self.image)
        if self.overlay is not None and self.page.show_overlay:
            p.setOpacity(self.page.opacity.value() / 100)
            p.drawImage(target, self.overlay)
            p.setOpacity(1)
        if self.cursor_pos is not None:
            radius = self.page.brush.value() * self.zoom
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self.cursor_pos, radius, radius)

    def wheelEvent(self, event):
        if self.image is None:
            return
        anchor = event.position()
        before = self.to_image(anchor)
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.zoom = min(16.0, max(0.05, self.zoom * factor))
        self.offset = anchor - before * self.zoom
        self.update()

    def mousePressEvent(self, event):
        if self.image is None:
            return
        self.setFocus()
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.panning = event.position() - self.offset
        elif event.button() == Qt.LeftButton:
            self.page.begin_stroke()
            self.painting = True
            self.last = self.to_image(event.position())
            self.page.paint(self.last, self.last)

    def mouseMoveEvent(self, event):
        self.cursor_pos = event.position()
        if self.panning is not None:
            self.offset = event.position() - self.panning
        elif self.painting:
            point = self.to_image(event.position())
            self.page.paint(self.last, point)
            self.last = point
        self.update()

    def mouseReleaseEvent(self, event):
        self.panning = None
        if self.painting:
            self.painting = False
            self.page.end_stroke()

    def leaveEvent(self, event):
        self.cursor_pos = None
        self.update()


class LabelPage(QWidget):
    """The Label UI page of the trainer (app.Trainer)."""

    def __init__(self, app, job_factory=None):
        super().__init__()
        self.setObjectName('page')
        self.setStyleSheet('QPushButton:checked { background: #f0f0f4; color: #0a0a0c; border-color: #f0f0f4; }')
        self.app = app
        self.settings = app.settings if app else {}
        self.project = None
        self.video = None
        self.info = None              # frame count, fps, width, height
        self.positions = []
        self.index = -1
        self.frames = None            # the labeled frame's window (BGR)
        self.coverage = self.valid = None
        self.overlay = None
        self.undo = []
        self.dirty = False
        self.show_overlay = True
        self.mode = 'paint'
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending = {}             # position -> future (window, suggestion)
        self.wanted = None
        self.label_size = _model_size()

        root = QVBoxLayout(self)
        root.setContentsMargins(34, 22, 34, 24)
        root.setSpacing(12)
        root.addWidget(_label('Label UI', 'title'))
        root.addWidget(_label('Paint the HUD on frames from your gameplay. Each frame opens pre-painted from how '
                              'its HUD stays fixed while the game moves; fix it with the brush and save. The labels '
                              'train and measure the optional UI model.', 'subtitle', wrap=True))
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        # The controls scroll instead of squeezing when the window is short.
        holder = QWidget()
        holder.setObjectName('page')
        side = QVBoxLayout(holder)
        side.setContentsMargins(0, 0, 6, 0)
        side.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_width = 300
        scroll.setFixedWidth(side_width + 14)
        body.addWidget(scroll)

        card, lay = _card('Project')
        row = QHBoxLayout()
        b = QPushButton('New…')
        b.clicked.connect(self.new_project)
        row.addWidget(b)
        b = QPushButton('Open…')
        b.clicked.connect(self.open_project)
        row.addWidget(b)
        lay.addLayout(row)
        self.project_text = _label('No project yet.', 'dim', wrap=True)
        lay.addWidget(self.project_text)
        self.use_button = QPushButton('Use for training')
        self.use_button.setToolTip('Puts this project\'s labels.json into the Train page\'s UI label field.')
        self.use_button.clicked.connect(self.use_for_training)
        lay.addWidget(self.use_button)
        card.setFixedWidth(side_width)
        side.addWidget(card)

        card, lay = _card('Video')
        b = QPushButton('Choose video…')
        b.clicked.connect(self.pick_video)
        lay.addWidget(b)
        self.video_text = _label('No video.', 'dim', wrap=True)
        lay.addWidget(self.video_text)
        row = QHBoxLayout()
        row.addWidget(_label('Split', 'dim'))
        self.split = QComboBox()
        self.split.addItem('Training', 'train')
        self.split.addItem('Validation', 'val')
        self.split.setToolTip('Validation needs a different video than training. A video stays in the split it '
                              'was first saved in.')
        row.addWidget(self.split, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(_label('Frames', 'dim'))
        self.count = QSpinBox()
        self.count.setRange(1, 500)
        self.count.setValue(int(self.settings.get('label_frames', 20)))
        row.addWidget(self.count)
        self.size = QComboBox()
        if self.label_size:
            self.size.addItem(f'{self.label_size[0]}×{self.label_size[1]} model', 'model')
        self.size.addItem('Original size', 'original')
        self.size.setToolTip('The size samples are saved at. The AI model size matches what the UI model sees.')
        row.addWidget(self.size, 1)
        lay.addLayout(row)
        self.load_button = QPushButton('Load frames')
        self.load_button.setObjectName('primary')
        self.load_button.clicked.connect(self.load_frames)
        lay.addWidget(self.load_button)
        card.setFixedWidth(side_width)
        side.addWidget(card)

        card, lay = _card('Frames')
        self.list = QListWidget()
        self.list.setMinimumHeight(150)
        self.list.currentRowChanged.connect(self.select)
        lay.addWidget(self.list, 1)
        card.setFixedWidth(side_width)
        side.addWidget(card, 1)

        card, lay = _card('Brush')
        row = QHBoxLayout()
        self.modes = QButtonGroup(self)
        for key, (text, mode, tip) in enumerate((('HUD', 'paint', 'Paint HUD (B)'),
                                                 ('Erase', 'erase', 'Not HUD (E)'),
                                                 ('Unsure', 'unknown', 'Unsure: training ignores these pixels (U)'))):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, m=mode: self.set_mode(m))
            self.modes.addButton(b, key)
            row.addWidget(b)
        self.modes.button(0).setChecked(True)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(_label('Size  [ ]', 'dim'))
        self.brush = QSlider(Qt.Horizontal)
        self.brush.setRange(2, 120)
        self.brush.setValue(int(self.settings.get('label_brush', 14)))
        row.addWidget(self.brush, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(_label('Overlay  H', 'dim'))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(55)
        self.opacity.valueChanged.connect(lambda _: self.canvas.update())
        row.addWidget(self.opacity, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        for text, fn, tip in (('Undo', self.undo_stroke, 'Ctrl+Z'), ('Clear', self.clear, 'Remove all HUD paint'),
                              ('Suggest', self.resuggest, 'Paint the suggestion again')):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        card.setFixedWidth(side_width)
        side.addWidget(card)

        row = QHBoxLayout()
        self.save_button = QPushButton('Save && next   S')
        self.save_button.setObjectName('primary')
        self.save_button.clicked.connect(self.save_and_next)
        row.addWidget(self.save_button, 1)
        b = QPushButton('Skip  N')
        b.clicked.connect(self.next_frame)
        row.addWidget(b)
        side.addLayout(row)
        self.status = _label('', 'dim', wrap=True)
        self.status.setFixedWidth(side_width)
        side.addWidget(self.status)

        self.canvas = Canvas(self)
        body.addWidget(self.canvas, 1)

        for keys, fn in (('B', lambda: self.set_mode('paint')), ('E', lambda: self.set_mode('erase')),
                         ('U', lambda: self.set_mode('unknown')), ('[', lambda: self.resize_brush(1 / 1.25)),
                         (']', lambda: self.resize_brush(1.25)), ('Ctrl+Z', self.undo_stroke),
                         ('S', self.save_and_next), ('N', self.next_frame), ('F', self.canvas.fit),
                         ('H', self.toggle_overlay)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(fn)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)
        last = self.settings.get('label_project')
        if last and (Path(last) / hud_label.PROJECT_FILE).is_file():
            try:
                self.open(last)
            except (OSError, ValueError):
                pass
        self.refresh()

    # ------------------------------------------------------------ state

    def empty_text(self):
        if not self.project:
            return 'Create or open a label project, then choose a video.'
        if not self.video:
            return 'Choose a video with the HUD you want to label.'
        if self.wanted is not None:
            return 'Reading frames and painting the suggestion…'
        return 'Press Load frames.'

    def store(self, values):
        if self.app:
            self.app.store(values)

    def refresh(self):
        if self.project:
            c = self.project.counts()
            self.project_text.setToolTip(str(self.project.folder))
            self.project_text.setText(f'{self.project.folder.name}\nTraining {c["train"]} · Validation {c["val"]}'
                                      + (f' · Test {c["test"]}' if c['test'] else '')
                                      + ('' if c['train'] and c['val'] else
                                         '\nTraining needs samples in both splits (different videos).'))
        else:
            self.project_text.setText('No project yet. Choose an empty folder for a new one.')
        self.use_button.setEnabled(bool(self.project and self.project.samples()))
        ready = self.project is not None and self.coverage is not None
        self.save_button.setEnabled(ready)
        self.load_button.setEnabled(self.project is not None and self.video is not None)
        if self.video and self.project:
            fixed = self.project.split_of(hud_label.source_of(self.video))
            if fixed in ('train', 'val'):
                self.split.setCurrentIndex(0 if fixed == 'train' else 1)
            self.split.setEnabled(fixed is None)

    def message(self, text):
        self.status.setText(text)

    # ------------------------------------------------------------ project

    def new_project(self):
        folder = QFileDialog.getExistingDirectory(self, 'Empty folder for the label project')
        if not folder:
            return
        try:
            self.project = hud_label.Project(folder)
            self.project._write()
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, 'Label UI', str(e))
            return
        self.store({'label_project': folder})
        self.refresh()
        self.mark_saved()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Label project', '', 'labels.json (labels.json)')
        if path:
            try:
                self.open(os.path.dirname(path))
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, 'Label UI', str(e))

    def open(self, folder):
        self.project = hud_label.Project(folder)
        self.store({'label_project': str(self.project.folder)})
        self.refresh()
        self.mark_saved()

    def use_for_training(self):
        if self.project and self.app and hasattr(self.app, 'train'):
            self.app.train.ui_labels.setText(str(self.project.path))
            self.message('The Train page now uses these labels (UI training: Flow + UI or UI only).')

    # ------------------------------------------------------------ video

    def pick_video(self):
        if not self.prepare_close():
            return
        start = os.path.dirname(self.settings.get('label_video', '')) or ''
        path, _ = QFileDialog.getOpenFileName(self, 'Gameplay video', start,
                                              'Videos (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)')
        if path:
            self.set_video(path)

    def set_video(self, path):
        try:
            self.info = hud_label.video_info(path)
        except ValueError as e:
            QMessageBox.warning(self, 'Label UI', str(e))
            return
        self.video = path
        frames, fps, w, h = self.info
        self.video_text.setText(f'{Path(path).name}\n{w}×{h}, {frames} frames, {frames / fps / 60:.1f} min')
        self.store({'label_video': path})
        self.positions, self.index = [], -1
        self.list.clear()
        self.frames = self.coverage = self.valid = None
        self.canvas.image = None
        self.canvas.update()
        self.refresh()

    def load_frames(self):
        if not (self.project and self.video):
            return
        if not self.prepare_close():
            return
        self.positions = hud_label.label_positions(self.info[0], self.count.value(), margin=WINDOW)
        self.store({'label_frames': self.count.value()})
        self.list.blockSignals(True)
        self.list.clear()
        for pos in self.positions:
            self.list.addItem(QListWidgetItem(''))
        self.list.blockSignals(False)
        self.mark_saved()
        self.list.setCurrentRow(0)

    def sample_id(self, position):
        return f'{Path(self.video).stem}-{position:07d}'

    def mark_saved(self):
        if not self.video:
            return
        fps = self.info[1] if self.info else 30.0
        for row, pos in enumerate(self.positions):
            saved = bool(self.project and self.project.sample(self.sample_id(pos)))
            seconds = pos / fps
            self.list.item(row).setText(f'{"✓" if saved else "•"}   frame {pos}   {int(seconds // 60)}:{seconds % 60:04.1f}')

    def request(self, position):
        if position in self.pending or position is None:
            return
        video = self.video

        def work():
            # Only the three saved frames are kept: a 1440p run of 24 is 265 MB.
            first, window = hud_label.read_window(video, position, WINDOW)
            return window[-3:], hud_label.suggest(window)
        self.pending[position] = self.pool.submit(work)

    def select(self, row):
        if row < 0 or row >= len(self.positions):
            return
        if row != self.index and not self.prepare_close():
            self.list.blockSignals(True)
            self.list.setCurrentRow(self.index)
            self.list.blockSignals(False)
            return
        self.index = row
        self.wanted = self.positions[row]
        self.frames = self.coverage = self.valid = None
        self.canvas.image = None
        self.canvas.update()
        self.request(self.wanted)
        if row + 1 < len(self.positions):
            self.request(self.positions[row + 1])
        self.message('Reading frames…')
        self.refresh()

    def poll(self):
        if self.wanted is None:
            return
        future = self.pending.get(self.wanted)
        if not future or not future.done():
            return
        position = self.wanted
        self.wanted = None
        try:
            window, suggestion = future.result()
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.message(f'Could not read frame {position}: {e}')
            del self.pending[position]
            return
        # Keep the next one's result, drop older ones (memory).
        keep = {p: f for p, f in self.pending.items() if p > position}
        self.pending = keep
        self.frames = window[-3:]
        saved = self.project.sample(self.sample_id(position)) if self.project else None
        frame = window[-1]
        h, w = frame.shape[:2]
        if saved:
            cov, val = self.project.coverage(saved['id']), self.project.valid(saved['id'])
            self.coverage = cv2.resize(cov, (w, h), interpolation=cv2.INTER_NEAREST)
            self.valid = cv2.resize(val, (w, h), interpolation=cv2.INTER_NEAREST)
            self.message('Saved label loaded. Change it and save again, or move on.')
        else:
            self.coverage = suggestion.copy()
            self.valid = np.full((h, w), 255, np.uint8)
            self.message('Pre-painted suggestion: green = HUD. Fix it, then Save (S).')
        self.suggestion = suggestion
        self.undo = []
        self.dirty = False   # only the user's own paint asks to be saved
        self.canvas.set_frame(frame)
        self.rebuild_overlay()
        self.refresh()

    # ------------------------------------------------------------ painting

    def set_mode(self, mode):
        self.mode = mode
        self.modes.button(('paint', 'erase', 'unknown').index(mode)).setChecked(True)

    def resize_brush(self, factor):
        self.brush.setValue(int(round(max(2, min(120, self.brush.value() * factor)))))
        self.store({'label_brush': self.brush.value()})
        self.canvas.update()

    def toggle_overlay(self):
        self.show_overlay = not self.show_overlay
        self.canvas.update()

    def rebuild_overlay(self, box=None):
        if self.coverage is None:
            return
        h, w = self.coverage.shape
        if self.overlay is None or self.overlay.shape[:2] != (h, w):
            self.overlay = np.zeros((h, w, 4), np.uint8)
            box = None
        y0, y1, x0, x1 = box if box else (0, h, 0, w)
        cov = self.coverage[y0:y1, x0:x1] > 127
        unknown = self.valid[y0:y1, x0:x1] < 128
        region = self.overlay[y0:y1, x0:x1]
        region[:] = 0
        region[cov] = (*HUD_COLOR, 255)
        region[unknown] = (*UNKNOWN_COLOR, 255)
        if box is None:
            self.canvas.set_overlay(self.overlay)
        else:
            self.canvas.update()

    def begin_stroke(self):
        if self.coverage is None:
            return
        self.undo.append((np.packbits(self.coverage > 127), np.packbits(self.valid > 127)))
        self.undo = self.undo[-UNDO_LIMIT:]

    def end_stroke(self):
        pass

    def paint(self, a, b):
        if self.coverage is None:
            return
        r = self.brush.value()
        p0 = (int(round(a.x())), int(round(a.y())))
        p1 = (int(round(b.x())), int(round(b.y())))
        if self.mode == 'paint':
            cv2.line(self.coverage, p0, p1, 255, 2 * r, cv2.LINE_8)
            cv2.line(self.valid, p0, p1, 255, 2 * r, cv2.LINE_8)
        elif self.mode == 'erase':
            cv2.line(self.coverage, p0, p1, 0, 2 * r, cv2.LINE_8)
            cv2.line(self.valid, p0, p1, 255, 2 * r, cv2.LINE_8)
        else:
            cv2.line(self.valid, p0, p1, 0, 2 * r, cv2.LINE_8)
            cv2.line(self.coverage, p0, p1, 0, 2 * r, cv2.LINE_8)
        h, w = self.coverage.shape
        x0, x1 = max(0, min(p0[0], p1[0]) - r - 1), min(w, max(p0[0], p1[0]) + r + 2)
        y0, y1 = max(0, min(p0[1], p1[1]) - r - 1), min(h, max(p0[1], p1[1]) + r + 2)
        if x0 < x1 and y0 < y1:
            self.rebuild_overlay((y0, y1, x0, x1))
        self.dirty = True

    def undo_stroke(self):
        if not self.undo or self.coverage is None:
            return
        cov, val = self.undo.pop()
        h, w = self.coverage.shape
        self.coverage = np.unpackbits(cov)[:h * w].reshape(h, w) * np.uint8(255)
        self.valid = np.unpackbits(val)[:h * w].reshape(h, w) * np.uint8(255)
        self.rebuild_overlay()
        self.dirty = True

    def clear(self):
        if self.coverage is None:
            return
        self.begin_stroke()
        self.coverage[:] = 0
        self.valid[:] = 255
        self.rebuild_overlay()
        self.dirty = True

    def resuggest(self):
        if self.coverage is None:
            return
        self.begin_stroke()
        self.coverage = self.suggestion.copy()
        self.valid[:] = 255
        self.rebuild_overlay()
        self.dirty = True

    # ------------------------------------------------------------ saving

    def save(self):
        if self.coverage is None or self.project is None:
            return False
        size = self.label_size if self.size.currentData() == 'model' else None
        try:
            self.project.save(self.video, self.positions[self.index], self.frames, self.coverage, self.valid,
                              self.split.currentData(), size)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, 'Label UI', str(e))
            return False
        self.dirty = False
        self.mark_saved()
        self.refresh()
        return True

    def save_and_next(self):
        if self.save():
            self.next_frame()

    def next_frame(self):
        if self.index + 1 < len(self.positions):
            self.list.setCurrentRow(self.index + 1)
        elif self.positions:
            self.message('That was the last frame. Load more frames or choose another video.')

    def prepare_close(self):
        """True when it is fine to leave the current frame (asks about
        unsaved paint)."""
        if not self.dirty or self.coverage is None:
            return True
        answer = QMessageBox.question(self, 'Label UI', 'Save the label on this frame first?',
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save()
        if answer == QMessageBox.Discard:
            self.dirty = False
            return True
        return False
