"""Motion Enhancer AI Trainer: dataset, training and export in one window.

    tools\\ai-train\\.venv\\Scripts\\python.exe tools\\ai-train\\app.py

Every long job (building a dataset, training, exporting) runs as its own
Python process (mefi.dataset / mefi.train / mefi.export) and reports JSON
lines that this window turns into progress, charts and previews; closing or
stopping never leaves the GPU in a half-finished state, and a crash in a job
does not take the window with it.
"""
import glob
import json
import math
import os
import shutil
import sys
import time

from PySide6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
                               QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
SETTINGS = os.path.join(HERE, 'settings.json')

# The program's palette (src/GuiApp.cpp).
BG, SIDEBAR, PANEL, PANEL_HI, FIELD = '#08080a', '#0c0c0e', '#111114', '#1c1c20', '#0d0d0f'
BORDER, BORDER_HI, TEXT, DIM = '#2a2a30', '#4a4a52', '#eeeef2', '#8a8a94'

STYLE = f"""
* {{ font-family: 'Segoe UI Variable Text', 'Segoe UI'; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#page {{ background: {BG}; }}
QWidget#sidebar {{ background: {SIDEBAR}; border-right: 1px solid #1e1e22; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-family: 'Segoe UI Variable Display', 'Segoe UI'; font-size: 28px; font-weight: 600; }}
QLabel#subtitle, QLabel#dim {{ color: {DIM}; }}
QLabel#section {{ font-family: 'Segoe UI Variable Display', 'Segoe UI'; font-size: 15px; font-weight: 600; }}
QLabel#metric {{ font-family: 'Segoe UI Variable Display', 'Segoe UI'; font-size: 22px; font-weight: 600; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QPlainTextEdit {{
    background: {FIELD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 5px 8px;
    selection-background-color: {PANEL_HI}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: #8a8a94; }}
QListWidget::item {{ padding: 4px; }}
QListWidget::item:selected {{ background: {PANEL_HI}; color: {TEXT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {PANEL_HI}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0px; }}
QPushButton {{ background: {PANEL_HI}; border: 1px solid {BORDER}; border-radius: 15px; padding: 6px 16px; }}
QPushButton:hover {{ background: #26262b; border-color: #6e6e78; }}
QPushButton:disabled {{ color: {DIM}; background: {PANEL}; }}
QPushButton#primary {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f4f4f6, stop:1 #c8c8d0);
    color: #0a0a0c; border: none; font-weight: 600; padding: 8px 22px; border-radius: 17px; }}
QPushButton#primary:hover {{ background: #ffffff; }}
QPushButton#primary:disabled {{ background: {PANEL_HI}; color: {DIM}; }}
QPushButton#nav {{ background: transparent; border: none; border-radius: 12px; text-align: left;
    padding: 12px 14px; font-size: 15px; color: {DIM}; }}
QPushButton#nav:hover {{ background: #18181b; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {PANEL_HI}; color: {TEXT}; border: 1px solid #3a3a42; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {BORDER_HI}; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background: #f0f0f4; border-color: #f0f0f4; }}
QProgressBar {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ border-radius: 3px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6e6e78, stop:1 #ffffff); }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: #2e2e34; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; }}
"""


def load_settings():
    try:
        with open(SETTINGS, encoding='utf-8') as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - first run
        return {}


def save_settings(values):
    with open(SETTINGS, 'w', encoding='utf-8') as f:
        json.dump(values, f, indent=1)


def program_builds():
    """Folders holding motion_enhancer.exe (its models/ is where the program
    looks for ai_flow.onnx)."""
    found = []
    for exe in glob.glob(os.path.join(REPO, '*', 'Release', 'motion_enhancer.exe')):
        found.append(os.path.dirname(exe))
    return sorted(found, key=lambda d: os.path.getmtime(os.path.join(d, 'motion_enhancer.exe')), reverse=True)


# ---------------------------------------------------------------- widgets

class Emblem(QWidget):
    """The program's emblem, animated: a spark round the ring, orbiting nodes."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 150)
        self.t = 0.0
        timer = QTimer(self)
        timer.timeout.connect(self.tick)
        timer.start(33)

    def tick(self):
        if self.window().isActiveWindow():
            self.t += 0.033
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2, 70
        breath = 0.5 + 0.5 * math.sin(self.t * 1.6)
        glow = QRadialGradient(QPointF(cx, cy), 70)
        glow.setColorAt(0, QColor(255, 255, 255, int(40 + 15 * breath)))
        glow.setColorAt(1, QColor(255, 255, 255, 0))
        p.setBrush(glow)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), 70, 70)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 215), 1.6))
        p.drawEllipse(QPointF(cx, cy), 30, 30)
        p.setPen(QPen(QColor(255, 255, 255, 80), 1.0))
        p.drawEllipse(QPointF(cx, cy), 38, 38)
        for a0 in (-140, -40, 40, 140, -90, 90):
            a = math.radians(a0 + self.t * 8)
            x0, y0 = cx + 38 * math.cos(a), cy + 38 * math.sin(a)
            x1, y1 = cx + 48 * math.cos(a), cy + 48 * math.sin(a)
            p.setPen(QPen(QColor(255, 255, 255, 80), 1.0))
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 230))
            p.drawEllipse(QPointF(x1, y1), 2.5, 2.5)
        head = self.t * 70
        for k in range(14):
            f = k / 14
            pen = QPen(QColor(255, 255, 255, int(220 * (1 - f) ** 2)), 2.6 - 1.8 * f)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(cx - 30, cy - 30, 60, 60), int(-(head - 4 * (k + 1)) * 16), int(-4.2 * 16))
        a = math.radians(head)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QPointF(cx + 30 * math.cos(a), cy + 30 * math.sin(a)), 2.4, 2.4)
        for i, (dy, length, w) in enumerate(((-9, 18, 1.6), (0, 26, 2.4), (9, 18, 1.6))):
            grad = QLinearGradient(QPointF(cx - length - 4, 0), QPointF(cx, 0))
            grad.setColorAt(0, QColor(255, 255, 255, 0))
            grad.setColorAt(1, QColor(255, 255, 255, 220))
            pen = QPen(QBrush(grad), w)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(cx - length - 4, cy + dy), QPointF(cx + 2, cy + dy))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QPointF(cx + 8, cy), 7, 7)
        p.setPen(QPen(QColor(255, 255, 255, int(80 + 80 * breath)), 1.2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx + 8, cy), 11 + 2 * breath, 11 + 2 * breath)
        p.setPen(QColor(TEXT))
        f = QFont('Segoe UI Variable Display', 9)
        f.setBold(True)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 3.5)
        p.setFont(f)
        p.drawText(QRectF(0, 124, self.width(), 18), Qt.AlignCenter, 'AI TRAINER')


class Chart(QWidget):
    """A small line chart: named series of (x, y), drawn in the theme."""

    def __init__(self, title, unit=''):
        super().__init__()
        self.title, self.unit = title, unit
        self.series = {}
        self.styles = {}
        self.setMinimumHeight(170)

    def set_series(self, name, color='#ffffff', dashed=False):
        self.series.setdefault(name, [])
        self.styles[name] = (color, dashed)

    def add(self, name, x, y):
        if y is None or not math.isfinite(y):
            return
        self.series[name].append((x, y))
        self.update()

    def clear(self):
        for k in self.series:
            self.series[k] = []
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(46, 28, -12, -22)
        p.setPen(QColor(TEXT))
        p.setFont(QFont('Segoe UI Variable Display', 10, QFont.DemiBold))
        p.drawText(QRectF(0, 2, self.width(), 20), Qt.AlignLeft | Qt.AlignVCenter, '  ' + self.title)
        pts = [pt for s in self.series.values() for pt in s]
        if not pts:
            p.setPen(QColor(DIM))
            p.setFont(QFont('Segoe UI', 9))
            p.drawText(r, Qt.AlignCenter, 'waiting for data')
            return
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        x0, x1 = min(xs), max(max(xs), min(xs) + 1e-6)
        y0, y1 = min(ys), max(ys)
        pad = (y1 - y0) * 0.1 or 0.5
        y0, y1 = y0 - pad, y1 + pad
        p.setFont(QFont('Segoe UI', 8))
        for i in range(5):
            y = r.top() + r.height() * i / 4
            p.setPen(QPen(QColor(255, 255, 255, 18), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(DIM))
            v = y1 - (y1 - y0) * i / 4
            p.drawText(QRectF(0, y - 8, 42, 16), Qt.AlignRight | Qt.AlignVCenter, f'{v:.3g}')
        mx = lambda x: r.left() + (x - x0) / (x1 - x0) * r.width()
        my = lambda y: r.bottom() - (y - y0) / (y1 - y0) * r.height()
        for name, s in self.series.items():
            if not s:
                continue
            color, dashed = self.styles[name]
            path = QPainterPath(QPointF(mx(s[0][0]), my(s[0][1])))
            for x, y in s[1:]:
                path.lineTo(mx(x), my(y))
            pen = QPen(QColor(color), 1.8)
            if dashed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawPath(path)
            p.setBrush(QColor(color))
            p.drawEllipse(QPointF(mx(s[-1][0]), my(s[-1][1])), 2.5, 2.5)
        p.setPen(QColor(DIM))
        legend = '   '.join(f'— {n}' for n in self.series)
        p.drawText(QRectF(r.left(), r.bottom() + 4, r.width(), 16), Qt.AlignRight, legend)


def card(title=None):
    frame = QFrame()
    frame.setObjectName('card')
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(18, 14, 18, 16)
    lay.setSpacing(10)
    if title:
        t = QLabel(title)
        t.setObjectName('section')
        lay.addWidget(t)
    return frame, lay


def label(text, name=None):
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    return w


def spin(value, lo, hi, step=1, decimals=None):
    if decimals is None:
        w = QSpinBox()
    else:
        w = QDoubleSpinBox()
        w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(value)
    w.setFixedWidth(110)
    return w


class Job:
    """A mefi.* module run as a process, its JSON lines passed to `on_event`."""

    def __init__(self, parent, on_event, on_finished):
        self.proc = QProcess(parent)
        self.proc.setWorkingDirectory(HERE)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUTF8', '1')
        env.insert('PYTHONUNBUFFERED', '1')
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.SeparateChannels)
        self.buffer = b''
        self.on_event, self.on_finished = on_event, on_finished
        self.proc.readyReadStandardOutput.connect(self.read)
        self.proc.readyReadStandardError.connect(self.read_err)
        self.proc.finished.connect(lambda code, _status: self.on_finished(code))

    def start(self, module, args):
        self.proc.start(sys.executable, ['-m', module] + args)

    def running(self):
        return self.proc.state() != QProcess.NotRunning

    def kill(self):
        if self.running():
            self.proc.kill()

    def read(self):
        self.buffer += bytes(self.proc.readAllStandardOutput())
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            text = line.decode('utf-8', 'replace').strip()
            if not text:
                continue
            try:
                self.on_event(json.loads(text))
            except json.JSONDecodeError:
                self.on_event({'event': 'log', 'text': text})

    def read_err(self):
        text = bytes(self.proc.readAllStandardError()).decode('utf-8', 'replace').strip()
        for line in text.splitlines():
            if line.strip() and 'Warning' not in line:
                self.on_event({'event': 'log', 'text': line})


def path_row(edit, button_text, pick):
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    b = QPushButton(button_text)
    b.clicked.connect(pick)
    row.addWidget(b)
    return row


def scaled(path, width):
    img = QImage(path)
    if img.isNull():
        return QPixmap()
    return QPixmap.fromImage(img.scaledToWidth(width, Qt.SmoothTransformation))


# ---------------------------------------------------------------- pages

class Page(QWidget):
    def __init__(self, title, subtitle):
        super().__init__()
        self.setObjectName('page')
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(34, 22, 34, 24)
        self.root.setSpacing(14)
        self.root.addWidget(label(title, 'title'))
        sub = label(subtitle, 'subtitle')
        sub.setWordWrap(True)
        self.root.addWidget(sub)


class DatasetPage(Page):
    def __init__(self, app):
        super().__init__('Dataset', 'Turn your own videos into training triplets: three frames in a row, '
                                    'the middle one being what the model learns to rebuild. Repeated frames, '
                                    'scene cuts, stills and too-fast motion are left out.')
        self.app = app
        s = app.settings
        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        vids, vl = card('Videos')
        self.videos = QListWidget()
        self.videos.setSelectionMode(QListWidget.ExtendedSelection)
        for v in s.get('videos', []):
            self.videos.addItem(v)
        self.setAcceptDrops(True)
        vl.addWidget(self.videos, 1)
        row = QHBoxLayout()
        for text, fn in (('Add videos…', self.add_videos), ('Remove', self.remove), ('Clear', self.videos.clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        vl.addLayout(row)
        vl.addWidget(label('Tip: drop video files here. Use footage like what you will play: '
                           'the same games, camera speeds and HUDs.', 'dim'))
        left.addWidget(vids, 1)

        prog, pl = card('Build')
        self.status = label('Ready.', 'dim')
        self.bar = QProgressBar()
        self.total_bar = QProgressBar()
        pl.addWidget(self.status)
        pl.addWidget(label('This video', 'dim'))
        pl.addWidget(self.bar)
        pl.addWidget(label('All videos', 'dim'))
        pl.addWidget(self.total_bar)
        self.thumbs = QLabel()
        self.thumbs.setMinimumHeight(110)
        self.thumbs.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.thumbs)
        row = QHBoxLayout()
        self.build_button = QPushButton('BUILD DATASET')
        self.build_button.setObjectName('primary')
        self.build_button.clicked.connect(self.build)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self.job.kill())
        row.addWidget(self.build_button)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        pl.addLayout(row)
        left.addWidget(prog)
        body.addLayout(left, 3)

        sett, sl = card('Settings')
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.out = QLineEdit(s.get('dataset', os.path.join(HERE, 'datasets', 'default')))
        sl.addWidget(label('Dataset folder (added to if it exists)', 'dim'))
        sl.addLayout(path_row(self.out, 'Browse', self.pick_out))
        self.height_ = spin(s.get('height', 360), 144, 1080, 8)
        self.max_per = spin(s.get('max_per_video', 600), 10, 100000, 50)
        self.gap = spin(s.get('gap', 6), 1, 600)
        self.min_motion = spin(s.get('min_motion', 1.0), 0.0, 50.0, 0.5, 1)
        self.max_motion = spin(s.get('max_motion', 40.0), 2.0, 200.0, 2.0, 1)
        self.duplicate = spin(s.get('duplicate', 0.6), 0.0, 10.0, 0.1, 2)
        self.cut = spin(s.get('cut', 45.0), 5.0, 255.0, 1.0, 1)
        rows = [('Frame height (px)', self.height_, '360 matches 1440p capture'),
                ('Triplets per video, at most', self.max_per, ''),
                ('Frames between triplets', self.gap, 'spreads samples over the video'),
                ('Least motion (px)', self.min_motion, 'below: a still, skipped'),
                ('Most motion (px)', self.max_motion, 'above: too fast to learn from'),
                ('Repeated-frame threshold', self.duplicate, 'mean difference, levels'),
                ('Scene-cut threshold', self.cut, 'mean difference, levels')]
        for r, (name, w, hint) in enumerate(rows):
            grid.addWidget(label(name), r, 0)
            grid.addWidget(w, r, 1)
            grid.addWidget(label(hint, 'dim'), r, 2)
        sl.addLayout(grid)
        sl.addWidget(label('Frame gaps', 'dim'))
        gaps = QHBoxLayout()
        self.strides = []
        for k, text in ((1, '1 (next frame)'), (2, '2 (every other)'), (3, '3')):
            c = QCheckBox(text)
            c.setChecked(k in s.get('strides', [1, 2]))
            self.strides.append((k, c))
            gaps.addWidget(c)
        gaps.addStretch(1)
        sl.addLayout(gaps)
        self.summary = label('', 'dim')
        sl.addWidget(self.summary)
        sl.addStretch(1)
        body.addWidget(sett, 2)
        self.root.addLayout(body, 1)
        self.job = Job(self, self.on_job, self.finished)
        self.out.textChanged.connect(self.refresh_summary)
        self.refresh_summary()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self.videos.addItem(url.toLocalFile())

    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Add videos', '', 'Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.ts);;All files (*)')
        for f in files:
            self.videos.addItem(f)

    def remove(self):
        for item in self.videos.selectedItems():
            self.videos.takeItem(self.videos.row(item))

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, 'Dataset folder', self.out.text())
        if d:
            self.out.setText(d)

    def refresh_summary(self):
        index = os.path.join(self.out.text(), 'index.json')
        try:
            with open(index, encoding='utf-8') as f:
                items = json.load(f)
            vids = len({e['video'] for e in items})
            self.summary.setText(f'This folder holds {len(items)} triplets from {vids} video(s).')
        except Exception:  # noqa: BLE001
            self.summary.setText('New dataset.')

    def values(self):
        return {'videos': [self.videos.item(i).text() for i in range(self.videos.count())],
                'dataset': self.out.text(), 'height': self.height_.value(),
                'max_per_video': self.max_per.value(), 'gap': self.gap.value(),
                'min_motion': self.min_motion.value(), 'max_motion': self.max_motion.value(),
                'duplicate': self.duplicate.value(), 'cut': self.cut.value(),
                'strides': [k for k, c in self.strides if c.isChecked()] or [1]}

    def build(self):
        v = self.values()
        if not v['videos']:
            self.status.setText('Add at least one video.')
            return
        self.app.store(v)
        args = ['--out', v['dataset'], '--height', str(v['height']), '--max-per-video', str(v['max_per_video']),
                '--gap', str(v['gap']), '--min-motion', str(v['min_motion']), '--max-motion', str(v['max_motion']),
                '--duplicate', str(v['duplicate']), '--cut', str(v['cut']), '--strides'] + [str(k) for k in v['strides']]
        self.total = len(v['videos'])
        self.total_bar.setRange(0, self.total)
        self.total_bar.setValue(0)
        self.build_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText('Starting…')
        self.app.set_busy('Building dataset')
        self.job.start('mefi.dataset', args + v['videos'])

    def on_job(self, e):
        kind = e.get('event')
        if kind == 'video':
            self.current = os.path.basename(e['video'])
            self.bar.setRange(0, max(1, e.get('frames') or 1))
            self.bar.setValue(0)
            self.total_bar.setValue(e['index'])
        elif kind == 'progress':
            self.bar.setValue(min(e['frame'], self.bar.maximum()))
            r = e['rejected']
            self.status.setText(f"{self.current}: frame {e['frame']} / {e['frames']}, kept {e['kept']} "
                                f"(skipped: {r['duplicate']} repeated, {r['cut']} cuts, {r['still']} still, {r['fast']} fast)")
        elif kind == 'sample':
            ims = [QImage(os.path.join(e['path'], f'im{k}.jpg')) for k in range(3)]
            if all(not i.isNull() for i in ims):
                h = 100
                ims = [i.scaledToHeight(h, Qt.SmoothTransformation) for i in ims]
                strip = QImage(sum(i.width() for i in ims) + 8, h, QImage.Format_RGB32)
                strip.fill(QColor(PANEL))
                p = QPainter(strip)
                x = 0
                for i in ims:
                    p.drawImage(x, 0, i)
                    x += i.width() + 4
                p.end()
                self.thumbs.setPixmap(QPixmap.fromImage(strip))
        elif kind == 'video_done':
            self.total_bar.setValue(e['index'] + 1)
        elif kind == 'done':
            self.status.setText(f"Done: {e['kept']} new triplets, {e['total']} in the dataset ({e['seconds']} s).")
            self.app.train.add_dataset(self.out.text())
        elif kind == 'error':
            self.status.setText(f"Error: {e.get('message')}")

    def finished(self, code):
        self.build_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.app.set_busy(None)
        self.refresh_summary()
        if code != 0 and not self.status.text().startswith('Done'):
            self.status.setText('Stopped.')


PRESET_TEXT = {'small': 'Small — about 0.7 ms at 640×352', 'medium': 'Medium — about 0.9 ms (recommended)',
               'large': 'Large — about 1.0 ms'}


class TrainPage(Page):
    def __init__(self, app):
        super().__init__('Train', 'Train the pyramid model on your triplets with ROCm on the GPU. '
                                  'Quality is measured on whole videos it never trains on.')
        self.app = app
        s = app.settings
        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        data, dl = card('Data')
        self.datasets = QListWidget()
        self.datasets.setMaximumHeight(90)
        for d in s.get('train_datasets', [s.get('dataset', '')]):
            if d:
                self.datasets.addItem(d)
        dl.addWidget(self.datasets)
        row = QHBoxLayout()
        b = QPushButton('Add dataset…')
        b.clicked.connect(self.pick_dataset)
        row.addWidget(b)
        b = QPushButton('Remove')
        b.clicked.connect(lambda: [self.datasets.takeItem(self.datasets.row(i)) for i in self.datasets.selectedItems()])
        row.addWidget(b)
        row.addStretch(1)
        dl.addLayout(row)
        left.addWidget(data)

        model, ml = card('Model')
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.preset = QComboBox()
        for k, t in PRESET_TEXT.items():
            self.preset.addItem(t, k)
        self.preset.setCurrentIndex(list(PRESET_TEXT).index(s.get('preset', 'medium')))
        self.epochs = spin(s.get('epochs', 60), 1, 2000, 5)
        self.batch = spin(s.get('batch', 16), 1, 128, 2)
        self.lr = spin(s.get('lr', 3e-4), 1e-5, 1e-2, 1e-4, 5)
        self.crop = spin(s.get('crop', 256), 64, 1024, 32)
        self.run = QLineEdit(s.get('run', os.path.join(HERE, 'runs', 'run1')))
        self.resume = QCheckBox('Continue this run from its last checkpoint')
        grid.addWidget(label('Size'), 0, 0)
        grid.addWidget(self.preset, 0, 1, 1, 2)
        for r, (name, w, hint) in enumerate((('Epochs', self.epochs, ''), ('Batch', self.batch, 'lower if memory runs out'),
                                             ('Learning rate', self.lr, ''), ('Crop (px)', self.crop, 'training patch size')), 1):
            grid.addWidget(label(name), r, 0)
            grid.addWidget(w, r, 1)
            grid.addWidget(label(hint, 'dim'), r, 2)
        ml.addLayout(grid)
        ml.addWidget(label('Run folder (checkpoints and previews)', 'dim'))
        ml.addLayout(path_row(self.run, 'Browse', self.pick_run))
        ml.addWidget(self.resume)
        row = QHBoxLayout()
        self.start_button = QPushButton('START TRAINING')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        ml.addLayout(row)
        self.bar = QProgressBar()
        ml.addWidget(self.bar)
        self.status = label('Ready.', 'dim')
        self.status.setWordWrap(True)
        ml.addWidget(self.status)
        left.addWidget(model)
        left.addStretch(1)
        body.addLayout(left, 2)

        right = QVBoxLayout()
        metrics, mtl = card()
        mrow = QHBoxLayout()
        self.m_psnr = label('–', 'metric')
        self.m_gain = label('–', 'metric')
        self.m_eta = label('–', 'metric')
        for name, w in (('Best quality (dB)', self.m_psnr), ('Over a plain blend', self.m_gain), ('Time left', self.m_eta)):
            col = QVBoxLayout()
            col.addWidget(w)
            col.addWidget(label(name, 'dim'))
            mrow.addLayout(col)
        mtl.addLayout(mrow)
        right.addWidget(metrics)
        charts, cl = card()
        self.loss_chart = Chart('Training loss')
        self.loss_chart.set_series('loss', '#ffffff')
        self.psnr_chart = Chart('Validation quality (dB)')
        self.psnr_chart.set_series('model', '#ffffff')
        self.psnr_chart.set_series('blend', '#6e6e78', dashed=True)
        cl.addWidget(self.loss_chart)
        cl.addWidget(self.psnr_chart)
        right.addWidget(charts)
        prev, pvl = card('Preview')
        pvl.addWidget(label('Top: frame 0, true middle, frame 2.  Bottom: the model\'s middle, its motion, the error ×3.', 'dim'))
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(200)
        pvl.addWidget(self.preview, 1)
        right.addWidget(prev, 1)
        body.addLayout(right, 3)
        self.root.addLayout(body, 1)
        self.job = Job(self, self.on_job, self.finished)

    def add_dataset(self, d):
        if d and not any(self.datasets.item(i).text() == d for i in range(self.datasets.count())):
            self.datasets.addItem(d)

    def pick_dataset(self):
        d = QFileDialog.getExistingDirectory(self, 'Dataset folder', os.path.join(HERE, 'datasets'))
        if d:
            self.add_dataset(d)

    def pick_run(self):
        d = QFileDialog.getExistingDirectory(self, 'Run folder', os.path.join(HERE, 'runs'))
        if d:
            self.run.setText(d)

    def start(self):
        sets = [self.datasets.item(i).text() for i in range(self.datasets.count())]
        sets = [d for d in sets if os.path.exists(os.path.join(d, 'index.json'))]
        if not sets:
            self.status.setText('Add a dataset folder (build one on the Dataset page first).')
            return
        v = {'train_datasets': sets, 'preset': self.preset.currentData(), 'epochs': self.epochs.value(),
             'batch': self.batch.value(), 'lr': self.lr.value(), 'crop': self.crop.value(), 'run': self.run.text()}
        self.app.store(v)
        args = ['--data'] + sets + ['--run', v['run'], '--preset', v['preset'], '--epochs', str(v['epochs']),
                                    '--batch', str(v['batch']), '--lr', str(v['lr']), '--crop', str(v['crop'])]
        if self.resume.isChecked():
            args.append('--resume')
        else:
            self.loss_chart.clear()
            self.psnr_chart.clear()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText('Starting…')
        self.app.set_busy('Training')
        self.job.start('mefi.train', args)

    def stop(self):
        # Graceful: the trainer finishes the step and keeps its checkpoints.
        os.makedirs(self.run.text(), exist_ok=True)
        open(os.path.join(self.run.text(), 'STOP'), 'w').close()
        self.status.setText('Stopping after this step…')

    def on_job(self, e):
        kind = e.get('event')
        if kind == 'device':
            self.app.set_device(e['name'])
        elif kind == 'data':
            self.status.setText(f"{e['train']} training and {e['val']} validation triplets from {e['videos']} video(s); "
                                f"{e['steps_per_epoch']} steps per epoch.")
        elif kind == 'step':
            self.bar.setRange(0, e['steps'])
            self.bar.setValue(e['step'])
            self.loss_chart.add('loss', e['step'], e['loss'])
            eta = e.get('eta', 0)
            self.m_eta.setText(f'{eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}' if eta >= 3600 else f'{eta // 60}:{eta % 60:02d}')
        elif kind == 'epoch':
            self.psnr_chart.add('model', e['epoch'] + 1, e['psnr'])
            self.psnr_chart.add('blend', e['epoch'] + 1, e['blend_psnr'])
            self.m_psnr.setText(f"{e['best']:.2f}")
            self.m_gain.setText(f"{e['best'] - e['blend_psnr']:+.2f} dB")
            pix = scaled(e['preview'], max(300, self.preview.width() - 10))
            if not pix.isNull():
                self.preview.setPixmap(pix)
            self.status.setText(f"Epoch {e['epoch'] + 1} / {e['epochs']}: {e['psnr']:.2f} dB"
                                + ('  (best so far, saved)' if e['improved'] else ''))
        elif kind == 'done':
            self.status.setText(f"Finished: best {e['best']:.2f} dB. Export it on the Export page.")
            self.app.export.set_checkpoint(os.path.join(self.run.text(), 'best.pt'))
        elif kind == 'error':
            self.status.setText(f"Error: {e.get('message')}")
        elif kind == 'log' and ('Error' in e['text'] or 'error' in e['text']):
            self.status.setText(e['text'][-300:])

    def finished(self, code):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.app.set_busy(None)


RESOLUTIONS = [('640×352 — a quarter of 1440p (recommended)', 640, 352),
               ('480×272 — a quarter of 1080p', 480, 272),
               ('960×544 — 3/8 of 1440p, slower', 960, 544),
               ('1280×720 — half of 1440p, slowest', 1280, 720)]


class ExportPage(Page):
    def __init__(self, app):
        super().__init__('Export', 'Write the trained model as the ONNX file Motion Enhancer loads '
                                   '(models\\ai_flow.onnx next to the program), then measure its speed there.')
        self.app = app
        s = app.settings
        c, l = card('Model')
        self.checkpoint = QLineEdit(s.get('checkpoint', os.path.join(HERE, 'runs', 'run1', 'best.pt')))
        l.addWidget(label('Checkpoint', 'dim'))
        l.addLayout(path_row(self.checkpoint, 'Browse', self.pick_checkpoint))
        self.resolution = QComboBox()
        for text, w, h in RESOLUTIONS:
            self.resolution.addItem(text, (w, h))
        l.addWidget(label('Network resolution (each captured frame is scaled to it)', 'dim'))
        l.addWidget(self.resolution)
        builds = program_builds()
        default = os.path.join(builds[0], 'models', 'ai_flow.onnx') if builds else os.path.join(HERE, 'ai_flow.onnx')
        self.out = QLineEdit(s.get('export', default))
        l.addWidget(label('Output', 'dim'))
        l.addLayout(path_row(self.out, 'Browse', self.pick_out))
        self.copy_all = QCheckBox('Also copy it to every other Motion Enhancer build in this repository')
        self.copy_all.setChecked(True)
        l.addWidget(self.copy_all)
        row = QHBoxLayout()
        self.export_button = QPushButton('EXPORT')
        self.export_button.setObjectName('primary')
        self.export_button.clicked.connect(self.export)
        self.bench_button = QPushButton('Measure speed in Motion Enhancer')
        self.bench_button.clicked.connect(self.bench)
        row.addWidget(self.export_button)
        row.addWidget(self.bench_button)
        row.addStretch(1)
        l.addLayout(row)
        self.status = label('Ready.', 'dim')
        self.status.setWordWrap(True)
        l.addWidget(self.status)
        self.root.addWidget(c)
        r, rl = card('Result')
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        rl.addWidget(self.result)
        self.root.addWidget(r, 1)
        self.job = Job(self, self.on_job, self.finished)
        self.bench_proc = None

    def set_checkpoint(self, path):
        self.checkpoint.setText(path)

    def pick_checkpoint(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Checkpoint', os.path.join(HERE, 'runs'), 'Checkpoints (*.pt)')
        if f:
            self.checkpoint.setText(f)

    def pick_out(self):
        f, _ = QFileDialog.getSaveFileName(self, 'Output', self.out.text(), 'ONNX (*.onnx)')
        if f:
            self.out.setText(f)

    def export(self):
        if not os.path.exists(self.checkpoint.text()):
            self.status.setText('Choose a checkpoint (best.pt from a training run).')
            return
        w, h = self.resolution.currentData()
        self.app.store({'checkpoint': self.checkpoint.text(), 'export': self.out.text()})
        self.export_button.setEnabled(False)
        self.status.setText('Exporting…')
        self.job.start('mefi.export', [self.checkpoint.text(), '--out', self.out.text(), '--width', str(w), '--height', str(h)])

    def on_job(self, e):
        if e.get('event') == 'done':
            lines = [f"Written: {e['out']}", f"Resolution {e['width']}×{e['height']}, preset {e['preset']}, "
                     f"validation {e['psnr']:.2f} dB" if e.get('psnr') else '']
            if self.copy_all.isChecked():
                src = e['out']
                for build in program_builds():
                    dst = os.path.join(build, 'models', 'ai_flow.onnx')
                    if os.path.abspath(dst) == os.path.abspath(src):
                        continue
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    shutil.copy2(os.path.splitext(src)[0] + '.json', os.path.splitext(dst)[0] + '.json')
                    lines.append(f'Copied to {dst}')
            self.result.setPlainText('\n'.join(l for l in lines if l))
            self.status.setText('Exported. Select "AI pyramid" as the flow engine in Motion Enhancer.')
        elif e.get('event') == 'error':
            self.status.setText(f"Error: {e.get('message')}")
        elif e.get('event') == 'log':
            self.result.appendPlainText(e['text'])

    def finished(self, code):
        self.export_button.setEnabled(True)
        if code != 0 and not self.status.text().startswith('Exported'):
            self.status.setText('Export failed; see the result box.')

    def bench(self):
        builds = program_builds()
        model = self.out.text()
        if not builds or not os.path.exists(model):
            self.status.setText('Export first; the program must be built.')
            return
        self.result.appendPlainText(f'\nMeasuring with {builds[0]}\\motion_enhancer.exe …')
        self.bench_proc = QProcess(self)
        self.bench_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.bench_proc.readyReadStandardOutput.connect(
            lambda: self.result.appendPlainText(bytes(self.bench_proc.readAllStandardOutput()).decode('utf-8', 'replace').rstrip()))
        self.bench_proc.start(os.path.join(builds[0], 'motion_enhancer.exe'), ['--ai-flow-test', model])


# ---------------------------------------------------------------- window

class Trainer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle('Motion Enhancer — AI Trainer')
        self.resize(1320, 860)
        root = QWidget()
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        side = QWidget()
        side.setObjectName('sidebar')
        side.setFixedWidth(232)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(16, 18, 16, 16)
        sl.setSpacing(6)
        sl.addWidget(Emblem(), 0, Qt.AlignHCenter)
        sub = label('Pyramid flow · ROCm', 'dim')
        sub.setAlignment(Qt.AlignHCenter)
        sl.addWidget(sub)
        sl.addSpacing(18)
        self.pages = QStackedWidget()
        self.dataset = DatasetPage(self)
        self.train = TrainPage(self)
        self.export = ExportPage(self)
        self.nav = []
        for i, (text, page) in enumerate((('▤   Dataset', self.dataset), ('◈   Train', self.train), ('⇪   Export', self.export))):
            self.pages.addWidget(page)
            b = QPushButton(text)
            b.setObjectName('nav')
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=i: self.show_page(k))
            sl.addWidget(b)
            self.nav.append(b)
        sl.addStretch(1)
        status, stl = card()
        self.busy = label('Idle', 'section')
        self.device = label('GPU: checking…', 'dim')
        self.device.setWordWrap(True)
        stl.addWidget(self.busy)
        stl.addWidget(self.device)
        sl.addWidget(status)
        lay.addWidget(side)
        lay.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self.show_page(self.settings.get('page', 0))
        QTimer.singleShot(200, self.detect_device)

    def show_page(self, k):
        self.pages.setCurrentIndex(k)
        for i, b in enumerate(self.nav):
            b.setChecked(i == k)
        self.store({'page': k})

    def store(self, values):
        self.settings.update(values)
        save_settings(self.settings)

    def set_busy(self, text):
        self.busy.setText(text or 'Idle')

    def set_device(self, name):
        self.device.setText(f'GPU: {name}')

    def detect_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.set_device(f'{torch.cuda.get_device_name(0)} (ROCm {torch.version.hip or ""})')
            else:
                self.set_device('No ROCm GPU found - training would run on the CPU')
        except Exception as e:  # noqa: BLE001
            self.set_device(f'PyTorch unavailable: {e}')

    def closeEvent(self, e):
        for job in (self.dataset.job, self.train.job, self.export.job):
            job.kill()
        self.store(self.dataset.values())
        e.accept()


def dark_title_bar(window):
    """Windows 11: a dark caption to match."""
    try:
        import ctypes
        hwnd = int(window.winId())
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), 4)
        color = ctypes.c_int(0x000e0c0c)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(color), 4)
    except Exception:  # noqa: BLE001 - not Windows 11
        pass


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = Trainer()
    dark_title_bar(w)
    if '--snapshot' in sys.argv:
        # Offline check: render each page to PNGs and exit (no screen read).
        out = sys.argv[sys.argv.index('--snapshot') + 1]
        w.show()
        for k in range(3):
            w.show_page(k)
            app.processEvents()
            w.grab().save(f'{out}_{k}.png')
        return 0
    w.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
