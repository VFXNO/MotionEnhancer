"""Review-first UI annotation, bulk window imports and optional AI guidance."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)

from .ui_label_assist import NORMALIZATIONS
from .ui_labeling import Project, file_hash, permitted_source, read_image, source_id
from .ui_sam import MODEL_BYTES, model_path


def image_qt(array):
    array = np.ascontiguousarray(array)
    fmt = QImage.Format_RGBA8888 if array.shape[2] == 4 else QImage.Format_RGB888
    return QImage(array.data, array.shape[1], array.shape[0], array.strides[0], fmt).copy()


def text_label(text):
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def number(value, minimum, maximum, step=1):
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


class LabelCanvas(QWidget):
    changed = Signal()
    prompts_changed = Signal()

    def __init__(self):
        super().__init__()
        self.frames, self.annotation, self.suggestion = None, None, None
        self.frame_index, self.zoom, self.radius, self.coverage = 2, 1., 3, 255
        self.mode, self.previous = 'ui', None
        self.show_labels, self.show_suggestion = True, True
        self.interaction, self.points, self.point_labels, self.box = 'paint', [], [], None
        self.box_start, self.box_end = None, None
        self.empty_text = 'Create/open a label project, then import a three-frame window.'
        self.setMinimumSize(320, 200)
        self.setMouseTracking(True)

    def set_sample(self, frames, annotation):
        self.frames, self.annotation, self.suggestion = frames, annotation, None
        self.previous = None
        self.clear_prompts()
        self.set_zoom(self.zoom)

    def clear_prompts(self):
        self.points, self.point_labels, self.box = [], [], None
        self.box_start, self.box_end = None, None
        self.prompts_changed.emit()
        self.update()

    def set_zoom(self, zoom):
        self.zoom = min(16., max(.125, zoom))
        if self.frames is not None:
            h, w = self.frames[0].shape[:2]
            self.setFixedSize(max(1, round(w * self.zoom)), max(1, round(h * self.zoom)))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#08080a'))
        if self.frames is None:
            painter.setPen(QColor('#aaaab2'))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
            return
        painter.drawImage(self.rect(), image_qt(self.frames[self.frame_index]))
        if self.frame_index != 2:
            return
        if self.show_labels:
            overlay = np.zeros((*self.annotation.valid.shape, 4), np.uint8)
            overlay[self.annotation.valid == 0] = (50, 100, 220, 40)
            background = (self.annotation.valid == 255) & (self.annotation.coverage == 0)
            overlay[background] = (20, 180, 240, 45)
            ui = (self.annotation.valid == 255) & (self.annotation.coverage > 0)
            overlay[ui] = (40, 240, 90, 120)
            painter.drawImage(self.rect(), image_qt(overlay))
        if self.show_suggestion and self.suggestion is not None:
            overlay = np.zeros((*self.suggestion.shape, 4), np.uint8)
            overlay[:, :, :3] = (245, 80, 215)
            overlay[:, :, 3] = np.rint(self.suggestion * 140).astype(np.uint8)
            painter.drawImage(self.rect(), image_qt(overlay))
        if self.interaction != 'paint':
            painter.setBrush(Qt.NoBrush)
            for (x, y), label in zip(self.points, self.point_labels):
                painter.setPen(QPen(QColor('#55ff88') if label else QColor('#ff6060'), 2))
                painter.drawEllipse(QRectF(x * self.zoom - 4, y * self.zoom - 4, 8, 8))
            box = self.box
            if self.box_start is not None and self.box_end is not None:
                box = [*self.box_start, *self.box_end]
            if box is not None:
                painter.setPen(QPen(QColor('#ffe070'), 2, Qt.DashLine))
                painter.drawRect(QRectF(box[0] * self.zoom, box[1] * self.zoom,
                                        (box[2] - box[0]) * self.zoom, (box[3] - box[1]) * self.zoom).normalized())

    def coordinate(self, position):
        h, w = self.annotation.valid.shape
        return (min(w - 1, max(0, int(position.x() / self.zoom))),
                min(h - 1, max(0, int(position.y() / self.zoom))))

    def mousePressEvent(self, event):
        if self.annotation is None or self.frame_index != 2:
            return
        if self.interaction == 'sam-point' and event.button() in (Qt.LeftButton, Qt.RightButton):
            if len(self.points) < 6:
                self.points.append(list(self.coordinate(event.position())))
                self.point_labels.append(1 if event.button() == Qt.LeftButton else 0)
                self.box = None
                self.prompts_changed.emit()
                self.update()
            return
        if self.interaction == 'sam-box' and event.button() == Qt.LeftButton:
            self.box_start = self.box_end = self.coordinate(event.position())
            self.points, self.point_labels, self.box = [], [], None
            self.update()
            return
        if event.button() != Qt.LeftButton or self.interaction != 'paint':
            return
        self.annotation.checkpoint()
        self.previous = self.coordinate(event.position())
        self.paint_to(self.previous)

    def mouseMoveEvent(self, event):
        if self.box_start is not None and event.buttons() & Qt.LeftButton:
            self.box_end = self.coordinate(event.position())
            self.update()
            return
        if self.previous is not None and event.buttons() & Qt.LeftButton:
            self.paint_to(self.coordinate(event.position()))

    def paint_to(self, point):
        self.annotation.stroke(self.previous, point, self.radius, self.mode, self.coverage)
        self.previous = point
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.box_start is not None and event.button() == Qt.LeftButton:
            end = self.coordinate(event.position())
            x0, y0 = self.box_start
            x1, y1 = end
            self.box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)] if x0 != x1 and y0 != y1 else None
            self.box_start = self.box_end = None
            self.prompts_changed.emit()
            self.update()
        self.previous = None


class ImageWindowDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Choose three consecutive frames in chronological order')
        self.resize(720, 280)
        layout = QVBoxLayout(self)
        self.paths = []
        for name in ('Previous frame t-2', 'Previous frame t-1', 'Current frame t (label this one)'):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            field = QLineEdit()
            row.addWidget(field, 1)
            button = QPushButton('Browse')
            button.clicked.connect(lambda _=False, edit=field: self.pick(edit))
            row.addWidget(button)
            layout.addLayout(row)
            self.paths.append(field)
        self.confirm = QCheckBox('These are consecutive frames from the same recording, in the order shown.')
        layout.addWidget(self.confirm)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def pick(self, field):
        path, _ = QFileDialog.getOpenFileName(self, 'Frame image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if path:
            field.setText(path)


class LabelPage(QWidget):
    def __init__(self, app, job_factory):
        super().__init__()
        self.setObjectName('page')
        self.app, self.project, self.sample = app, None, None
        self.dirty, self.guidance, self.exported_manifest = False, None, None
        self.worker_temp, self.request, self.worker_done, self.worker_error = None, None, False, ''
        self.candidate_masks, self.candidate_info, self.suggestion_kind = None, None, 'dense'
        self.hud_template = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        title = QLabel('Label UI')
        title.setObjectName('title')
        root.addWidget(title)
        root.addWidget(text_label('Batch-label training windows from one reviewed HUD example, or edit with brushes / EfficientSAM. '
                                 'Machine labels are not verified truth. No training or game-model activation happens here.'))
        body = QHBoxLayout()
        root.addLayout(body, 1)
        self.controls = QWidget()
        self.controls.setObjectName('labelControls')
        self.controls.setStyleSheet('QWidget#labelControls { background: #08080a; }')
        left = QVBoxLayout(self.controls)
        left.setContentsMargins(0, 0, 8, 0)
        row = QHBoxLayout()
        self.button(row, 'New project', self.new_project)
        self.button(row, 'Open project', self.open_project)
        left.addLayout(row)
        self.project_name = text_label('No project open. Choose a separate new/empty folder, not an existing dataset or run.')
        left.addWidget(self.project_name)
        left.addWidget(text_label('1. Import windows from your recording'))
        self.video = QLineEdit()
        self.video.setPlaceholderText('Choose your own recording; no folder scanning')
        row = QHBoxLayout()
        row.addWidget(self.video, 1)
        self.button(row, 'Video…', self.pick_video)
        left.addLayout(row)
        form = QFormLayout()
        self.current = number(2, 2, 10000000)
        self.recording = QLineEdit()
        self.recording.setPlaceholderText('Recording identity (filled for a selected video)')
        self.split = QComboBox()
        self.split.addItem('Training', 'train')
        self.split.addItem('Validation — different recording', 'val')
        form.addRow('Current frame (zero-based)', self.current)
        form.addRow('Original recording ID', self.recording)
        form.addRow('Recording split', self.split)
        self.video_minutes = number(30, 5, 120, 5)
        self.video_minutes.setToolTip('Time allowed for either video-import button. AI suggestions still have a five-minute limit. Cancel remains available.')
        form.addRow('Video time limit (minutes)', self.video_minutes)
        self.resize_frames = QCheckBox('Resize prepared copies\n(original files stay unchanged)')
        self.resize_width, self.resize_height = number(832, 64, 4096, 16), number(480, 64, 4096, 16)
        dims = QHBoxLayout()
        dims.addWidget(self.resize_width)
        dims.addWidget(QLabel('×'))
        dims.addWidget(self.resize_height)
        form.addRow(self.resize_frames)
        form.addRow('Prepared width × height', dims)
        left.addLayout(form)
        row = QHBoxLayout()
        self.button(row, 'Read video window', self.import_video)
        self.button(row, 'Choose 3 images…', self.import_images)
        left.addLayout(row)
        form = QFormLayout()
        self.batch_count = number(20, 1, 100)
        self.batch_last = number(0, 0, 10000000)
        self.batch_last.setSpecialValueText('End of video (metadata)')
        form.addRow('Windows across video', self.batch_count)
        form.addRow('Last current frame (0 = end)', self.batch_last)
        left.addLayout(form)
        self.button(left, 'Sample many windows from video', self.import_video_batch)
        left.addWidget(text_label('Bulk: evenly spaced windows from Current frame through Last/end. '
                                 'Each sample keeps 3 consecutive frames; label only the last. '
                                 'Up to 100 windows / 512 MiB per import; duplicates are skipped.'))
        left.addWidget(text_label('Images: current frame number defines the last of the three chosen files. '
                                 'Keep one recording ID/split for all related or renamed footage. '
                                 'Joint flow+UI training needs different recordings for the two tasks.'))
        self.samples = QListWidget()
        self.samples.setMinimumHeight(90)
        self.samples.setMaximumHeight(160)
        self.samples.currentRowChanged.connect(self.select_sample)
        left.addWidget(self.samples)
        left.addWidget(text_label('2. Paint actual shapes, not rectangles around text'))
        self.tool = QComboBox()
        for name, key in (('UI pixels — green', 'ui'), ('Checked background — cyan', 'background'),
                          ('Unknown / erase label — blue', 'unknown')):
            self.tool.addItem(name, key)
        self.radius, self.brush_coverage = number(3, 0, 256), number(255, 1, 255)
        form = QFormLayout()
        form.addRow('Brush', self.tool)
        form.addRow('Radius (0 = single pixel)', self.radius)
        form.addRow('UI coverage (edge softness)', self.brush_coverage)
        left.addLayout(form)
        self.tool.currentIndexChanged.connect(self.brush_changed)
        self.radius.valueChanged.connect(self.brush_changed)
        self.brush_coverage.valueChanged.connect(self.brush_changed)
        row = QHBoxLayout()
        self.button(row, 'Undo', lambda: self.undo())
        self.button(row, 'Redo', lambda: self.undo(True))
        left.addLayout(row)
        self.button(left, 'Mark remaining unknown as checked background…', self.background_remainder)
        self.button(left, 'Reset this sample to unknown…', self.reset_unknown)
        left.addWidget(text_label('Reuse an unchanged HUD instead of repainting it'))
        self.button(left, 'Use saved reviewed sample as HUD template', self.pin_hud_template)
        self.hud_status = text_label('No HUD template selected. Review and save one good example first.')
        left.addWidget(self.hud_status)
        form = QFormLayout()
        self.hud_tolerance = number(12, 0, 32)
        form.addRow('RGB change tolerance (0–32)', self.hud_tolerance)
        left.addLayout(form)
        self.auto_hud = QCheckBox('Reuse template when opening unreviewed windows')
        left.addWidget(self.auto_hud)
        self.button(left, 'Apply HUD template to this window', self.apply_hud_template)
        left.addWidget(text_label('Same recording and size only. Copies UI pixels only where colors match '
                                 'the template in all 3 frames; never copies background or review. '
                                 'Changing text, transparency or moving HUD may need correction.'))
        left.addWidget(text_label('Batch auto-labeling — no per-window Save needed'))
        self.same_hud = QCheckBox('Other training recordings use this same\ngame, HUD layout and prepared size')
        left.addWidget(self.same_hud)
        form = QFormLayout()
        self.auto_padding, self.auto_minutes = number(16, 0, 64), number(30, 5, 120, 5)
        form.addRow('HUD neighborhood (pixels)', self.auto_padding)
        form.addRow('Auto-label limit (minutes)', self.auto_minutes)
        left.addLayout(form)
        self.button(left, 'Auto-label all compatible training windows…', self.auto_label_all)
        self.button(left, 'Next uncertain / recommended spot-check', self.next_auto_review)
        self.auto_status = text_label('One reviewed TRAINING template with UI and nearby checked background is needed. '
                                      'Local colors + temporal checks can adapt changing shapes inside seeded HUD neighborhoods. '
                                      'Ambiguous pixels stay unknown; this is not a general HUD detector.')
        left.addWidget(self.auto_status)
        left.addWidget(text_label('3. AI selection / automatic candidates (CPU)'))
        self.sam_status = text_label('')
        left.addWidget(self.sam_status)
        self.sam_download = self.button(left, 'Download EfficientSAM-Ti (48.3 MB)…', self.download_sam)
        self.update_sam_status()
        self.button(left, 'Generate masks for clicks / box', self.smart_masks)
        self.sam_grid = QComboBox()
        self.sam_grid.addItem('Quick scan — 3 × 3 points', 3)
        self.sam_grid.addItem('Denser scan — 5 × 5 points (slower)', 5)
        left.addWidget(self.sam_grid)
        self.button(left, 'Scan current frame for candidate masks', self.auto_masks)
        left.addWidget(text_label('Smart: choose AI clicks or AI box above the image, then select the UI shape. '
                                 'Automatic: scan this frame, preview masks, check the ones that are UI. '
                                 'Scenery can also be proposed. Nothing is accepted automatically.'))
        self.advanced = QWidget()
        advanced = QVBoxLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        self.button(left, 'Other dense model / external probability PNG…',
                    lambda: self.advanced.setVisible(not self.advanced.isVisible()))
        left.addWidget(self.advanced)
        self.advanced.setVisible(False)
        self.model = QLineEdit()
        self.model.setReadOnly(True)
        self.model.setPlaceholderText('No model bundled; select a compatible local ONNX')
        self.model_hash = None
        row = QHBoxLayout()
        row.addWidget(self.model, 1)
        self.button(row, 'Model…', self.pick_model)
        advanced.addLayout(row)
        self.normalization, self.output_kind = QComboBox(), QComboBox()
        self.normalization.addItems(NORMALIZATIONS)
        self.output_kind.addItems(('probabilities', 'logits'))
        form = QFormLayout()
        form.addRow('Model input normalization', self.normalization)
        form.addRow('Dense model output type', self.output_kind)
        advanced.addLayout(form)
        advanced.addWidget(text_label('Compatible model: fixed FP32 [1,3,H,W] RGB input → [1,1,H,W] UI map. '
                                 'Generic object segmenters do not automatically know what is HUD. '
                                 'Box/OCR/prompt models need a separate adapter.'))
        self.button(advanced, 'Suggest current frame with local AI', self.suggest)
        self.button(advanced, 'Import external AI probability PNG…', self.import_probability)
        self.apply_button = self.button(advanced, 'Apply AI draft only to unknown pixels', self.apply_draft)
        self.apply_button.setEnabled(False)
        advanced.addWidget(text_label('Magenta = unaccepted AI suggestion. Applying a dense-model draft uses ≥0.85 UI / ≤0.15 background; '
                                 'uncertain pixels stay unknown and manual labels are preserved. Review every sample.'))
        left.addWidget(text_label('4. Review, save and export'))
        self.reviewed = QCheckBox('I reviewed the current-frame labels;\nunsure areas remain unknown.')
        self.reviewed.setToolTip('AI confidence is not review. Correct scenery/text mistakes before checking this.')
        self.reviewed.toggled.connect(self.review_changed)
        left.addWidget(self.reviewed)
        self.button(left, 'Save annotation', self.save_current)
        self.button(left, 'Save reviewed and next unreviewed', self.save_next)
        self.include_auto_export = QCheckBox('Include eligible automatic TRAINING labels\n(unreviewed; validation stays human-reviewed)')
        left.addWidget(self.include_auto_export)
        self.button(left, 'Export training dataset…', self.export_dataset)
        self.use_button = self.button(left, 'Use exported manifest on Train page', self.use_manifest)
        self.use_button.setEnabled(False)
        left.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(355)
        scroll.setMaximumWidth(500)
        body.addWidget(scroll, 2)
        right = QVBoxLayout()
        toolbar = QHBoxLayout()
        self.frame_view = QComboBox()
        self.frame_view.addItems(('Previous t-2 (view only)', 'Previous t-1 (view only)', 'Current t (paint here)'))
        self.frame_view.setCurrentIndex(2)
        self.frame_view.currentIndexChanged.connect(self.view_changed)
        toolbar.addWidget(self.frame_view)
        self.button(toolbar, '−', lambda: self.zoom(.5))
        self.button(toolbar, '+', lambda: self.zoom(2))
        self.button(toolbar, '1:1', lambda: self.canvas.set_zoom(1))
        self.button(toolbar, 'Fit', self.fit)
        right.addLayout(toolbar)
        self.canvas = LabelCanvas()
        self.canvas.changed.connect(self.edited)
        self.canvas.prompts_changed.connect(self.prompts_changed)
        selection = QHBoxLayout()
        self.interaction = QComboBox()
        self.interaction.addItem('Paint labels', 'paint')
        self.interaction.addItem('AI clicks: left + / right −', 'sam-point')
        self.interaction.addItem('AI box: drag around a shape', 'sam-box')
        self.interaction.currentIndexChanged.connect(self.interaction_changed)
        selection.addWidget(self.interaction)
        self.button(selection, 'Clear clicks / box', self.canvas.clear_prompts)
        right.addLayout(selection)
        self.viewport = QScrollArea()
        self.viewport.viewport().setObjectName('labelViewport')
        self.viewport.viewport().setStyleSheet('QWidget#labelViewport { background-color: #08080a; }')
        palette = self.viewport.viewport().palette()
        palette.setColor(QPalette.Window, QColor('#08080a'))
        self.viewport.viewport().setPalette(palette)
        self.viewport.viewport().setAutoFillBackground(True)
        self.viewport.setWidget(self.canvas)
        self.viewport.setAlignment(Qt.AlignCenter)
        right.addWidget(self.viewport, 1)
        self.candidate_panel = QWidget()
        candidate_layout = QVBoxLayout(self.candidate_panel)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_layout.addWidget(text_label('AI candidate masks — click to preview; check only actual UI (score = predicted shape quality, NOT UI certainty)'))
        self.candidates = QListWidget()
        self.candidates.setMaximumHeight(130)
        self.candidates.currentRowChanged.connect(self.preview_candidate)
        candidate_layout.addWidget(self.candidates)
        candidate_buttons = QHBoxLayout()
        self.button(candidate_buttons, 'Add checked masks as UI', self.accept_candidates)
        self.button(candidate_buttons, 'Discard candidate masks', self.clear_candidates)
        candidate_layout.addLayout(candidate_buttons)
        right.addWidget(self.candidate_panel)
        self.candidate_panel.setVisible(False)
        row = QHBoxLayout()
        self.labels_visible, self.ai_visible = QCheckBox('Show labels'), QCheckBox('Show AI suggestion')
        for check in (self.labels_visible, self.ai_visible):
            check.setChecked(True)
            row.addWidget(check)
            check.toggled.connect(self.overlays_changed)
        self.cancel_button = self.button(row, 'Cancel operation', self.cancel_worker)
        self.cancel_button.setEnabled(False)
        right.addLayout(row)
        self.status = text_label('Unknown pixels are ignored during learning. Save reviewed samples in both training and validation recordings.')
        right.addWidget(self.status)
        body.addLayout(right, 3)
        self.job = job_factory(self, self.on_job, self.worker_finished)
        self.job.proc.errorOccurred.connect(self.worker_process_error)
        self.deadline = QTimer(self)
        self.deadline.setSingleShot(True)
        self.deadline.timeout.connect(self.timeout)

    @staticmethod
    def button(layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(lambda _=False: callback())
        layout.addWidget(button)
        return button

    def error(self, error):
        self.status.setText(f'Label UI: {error}')

    def update_sam_status(self):
        path = model_path()
        ready = path.is_file() and path.stat().st_size == MODEL_BYTES
        self.sam_status.setText('EfficientSAM-Ti available — verified on each run.\nOffline CPU; Apache-2.0; suggestions need review.'
                                if ready else 'EfficientSAM-Ti not downloaded. Manual labeling still works.')
        self.sam_status.setToolTip(str(path))
        self.sam_download.setText('Verify cached EfficientSAM-Ti…' if ready else 'Download EfficientSAM-Ti (48.3 MB)…')

    def download_sam(self):
        answer = QMessageBox.question(self, 'Download offline labeling model',
                                      'Verify cached EfficientSAM-Ti and download any missing pinned files '
                                      '(model: 48.3 MB) from the OpenCV model repository? '
                                      'Apache-2.0 license and SHA256 verification included. '
                                      'Saved separately under your local app-data labeling-model cache. '
                                      'This does not install packages, process recordings, train or change game models.',
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            try:
                self.start_worker({'action': 'sam_download'})
            except Exception as error:
                self.error(error)

    def interaction_changed(self, *_args):
        self.canvas.previous = None
        self.canvas.box_start = self.canvas.box_end = None
        self.canvas.interaction = self.interaction.currentData()
        if self.canvas.interaction != 'paint':
            self.frame_view.setCurrentIndex(2)
        self.canvas.update()
        if self.canvas.interaction == 'sam-point':
            if self.canvas.box is not None:
                self.canvas.clear_prompts()
            self.status.setText('Click inside the UI shape (+); right-click scenery to exclude (−). Up to 6 points. Then Generate masks for clicks / box.')
        elif self.canvas.interaction == 'sam-box':
            if self.canvas.points:
                self.canvas.clear_prompts()
            self.status.setText('Drag a box tightly around one UI shape. The box is only an AI prompt, never a filled label. Then Generate masks for clicks / box.')

    def prompts_changed(self):
        self.clear_candidates()
        if self.sample is not None:
            self.status.setText(f'Selection: {len(self.canvas.points)}/6 clicks' if self.canvas.box is None
                                else 'Box selection ready. Generate masks for clicks / box; no label has changed.')

    def smart_masks(self):
        try:
            if self.sample is None:
                raise ValueError('Import/select a frame window before using AI selection')
            if self.canvas.box is None and not self.canvas.points:
                self.interaction.setCurrentIndex(1)
                raise ValueError('Click inside the UI on the image, or choose AI box and drag a selection, then generate masks')
            self.frame_view.setCurrentIndex(2)
            self.start_worker({'action': 'sam_smart', 'points': self.canvas.points.copy(),
                               'labels': self.canvas.point_labels.copy(), 'box': self.canvas.box}, frame=self.canvas.frames[2])
        except Exception as error:
            self.error(error)

    def auto_masks(self):
        try:
            if self.sample is None:
                raise ValueError('Import/select a frame window before scanning')
            self.frame_view.setCurrentIndex(2)
            self.canvas.clear_prompts()
            self.interaction.setCurrentIndex(0)
            self.start_worker({'action': 'sam_auto', 'grid': self.sam_grid.currentData()}, frame=self.canvas.frames[2])
        except Exception as error:
            self.error(error)

    def clear_candidates(self):
        self.candidate_masks, self.candidate_info = None, None
        self.candidates.clear()
        self.candidate_panel.setVisible(False)
        if self.suggestion_kind == 'sam':
            self.canvas.suggestion = None
            self.canvas.update()
        self.suggestion_kind = 'dense'

    def set_candidates(self, masks, info):
        records = info.get('candidates', [])
        if (self.sample is None or masks.ndim != 3 or masks.dtype != np.bool_
                or masks.shape[1:] != self.canvas.annotation.valid.shape or len(masks) > 48
                or len(records) != len(masks) or any(not np.isfinite(item['score']) for item in records)):
            raise ValueError('Invalid AI candidate masks; no labels changed')
        self.clear_candidates()
        self.canvas.suggestion = None
        self.apply_button.setEnabled(False)
        self.candidate_masks, self.candidate_info, self.suggestion_kind = masks, info, 'sam'
        for index, (mask, record) in enumerate(zip(masks, records)):
            item = QListWidgetItem(f"Mask {index + 1} · shape score {record['score']:.3f} · {100 * mask.mean():.2f}% of image")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.candidates.addItem(item)
        self.candidate_panel.setVisible(bool(len(masks)))
        if len(masks):
            self.candidates.setCurrentRow(0)
            self.status.setText(f'{len(masks)} candidate masks. Click to preview; check actual UI only, then Add checked masks as UI. No labels have changed.')
        else:
            self.status.setText('No usable candidate masks. Try a tighter AI box/click, or paint manually. No labels changed; the frame remains unknown.')

    def preview_candidate(self, row):
        if self.candidate_masks is None or not 0 <= row < len(self.candidate_masks):
            return
        self.canvas.suggestion = self.candidate_masks[row].astype(np.float32)
        self.suggestion_kind = 'sam'
        self.frame_view.setCurrentIndex(2)
        self.canvas.update()

    def accept_candidates(self):
        if self.sample is None or self.candidate_masks is None:
            return
        selected = [index for index in range(len(self.candidate_masks))
                    if self.candidates.item(index).checkState() == Qt.Checked]
        if not selected:
            self.error('Check the candidate masks that are actual UI first; merely previewing does not accept one')
            return
        mask = np.any(self.candidate_masks[selected], axis=0)
        count = self.canvas.annotation.apply_foreground(mask)
        if count:
            info = {key: value for key, value in self.candidate_info.items() if key != 'candidates'}
            info.update(selected_candidates=[self.candidate_info['candidates'][i] for i in selected],
                        accepted_draft_pixels=count, admission='selected foreground only; outside stays unknown', review_required=True)
            self.guidance = [*(self.guidance or []), info]
            self.edited()
        self.clear_candidates()
        self.interaction.setCurrentIndex(0)
        self.canvas.update()
        self.status.setText(f'Added {count} unknown pixels as draft UI. Outside the masks stays unknown; existing manual labels were preserved. Correct edges and review before saving.')

    def brush_changed(self, *_args):
        self.canvas.mode, self.canvas.radius = self.tool.currentData(), self.radius.value()
        self.canvas.coverage = self.brush_coverage.value()
        self.interaction.setCurrentIndex(0)

    def view_changed(self, index):
        self.canvas.frame_index = index
        self.canvas.previous = None
        self.canvas.update()

    def overlays_changed(self):
        self.canvas.show_labels, self.canvas.show_suggestion = self.labels_visible.isChecked(), self.ai_visible.isChecked()
        self.canvas.update()

    def zoom(self, factor):
        self.canvas.set_zoom(self.canvas.zoom * factor)

    def fit(self):
        if self.canvas.frames is not None:
            h, w = self.canvas.frames[0].shape[:2]
            self.canvas.set_zoom(min(self.viewport.viewport().width() / w, self.viewport.viewport().height() / h))

    def edited(self):
        self.dirty = True
        self.reviewed.blockSignals(True)
        self.reviewed.setChecked(False)
        self.reviewed.blockSignals(False)

    def review_changed(self):
        if self.sample is not None:
            self.dirty = True

    def unsaved_ok(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, 'Unsaved annotation', 'Save this annotation before leaving it?',
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save_current()
        if answer == QMessageBox.Discard:
            try:
                if self.sample is not None:
                    frames, annotation = self.project.load_sample(self.sample)
                    self.canvas.set_sample(frames, annotation)
                    self.guidance = self.sample.get('guidance')
                    self.reviewed.blockSignals(True)
                    self.reviewed.setChecked(self.sample.get('reviewed', False))
                    self.reviewed.blockSignals(False)
                    self.apply_button.setEnabled(False)
                self.dirty = False
                return True
            except Exception as error:
                self.error(error)
        return False

    def new_project(self):
        if not self.unsaved_ok():
            return
        folder = QFileDialog.getExistingDirectory(self, 'Choose a NEW or EMPTY labeling project folder')
        if folder:
            try:
                self.set_project(Project.create(folder))
            except Exception as error:
                self.error(error)

    def open_project(self):
        if not self.unsaved_ok():
            return
        folder = QFileDialog.getExistingDirectory(self, 'Open a UI labeling project (contains project.json)')
        if folder:
            try:
                self.set_project(Project.open(folder))
            except Exception as error:
                self.error(error)

    def set_project(self, project):
        self.project, self.sample, self.dirty = project, None, False
        self.guidance, self.exported_manifest = None, None
        self.hud_template = None
        self.auto_hud.setChecked(False)
        self.same_hud.setChecked(False)
        self.include_auto_export.setChecked(False)
        self.hud_status.setText('No HUD template selected. Review and save one good example first.')
        automatic = [s for s in project.samples if s.get('auto_label') and not s.get('reviewed')]
        eligible = sum(s['auto_label'].get('eligible', False) for s in automatic)
        self.auto_status.setText(f'Project contains {len(automatic)} machine annotations ({eligible} eligible training masks). '
                                 'Pin one reviewed TRAINING HUD example with nearby checked background to generate or regenerate a batch.')
        self.use_button.setEnabled(False)
        self.project_name.setText(str(project.root))
        self.canvas.frames, self.canvas.annotation, self.canvas.suggestion = None, None, None
        self.canvas.empty_text = 'Project ready. Select a saved window on the left,\nor import a video window / three images to start labeling.'
        self.canvas.clear_prompts()
        self.canvas.update()
        self.reviewed.blockSignals(True)
        self.reviewed.setChecked(False)
        self.reviewed.blockSignals(False)
        self.apply_button.setEnabled(False)
        self.refresh_samples()
        self.status.setText('Project open. Select your own recording and a training/validation split. No source is read until you request an import.')

    def refresh_samples(self, selected=None):
        self.samples.blockSignals(True)
        self.samples.clear()
        if self.project:
            for item in self.project.samples:
                state = 'reviewed' if item.get('reviewed') else 'draft'
                if not item.get('reviewed') and item.get('auto_label'):
                    state = 'auto training' if item['auto_label']['eligible'] else 'auto: needs review'
                    if item['auto_label'].get('spot_check'):
                        state += ' · spot-check'
                self.samples.addItem(f"{item['split']} · frame {item['frame_ids'][2]} · {state} · {item['source'][:14]}")
        if selected is not None:
            self.samples.setCurrentRow(selected)
        self.samples.blockSignals(False)

    def select_sample(self, row):
        if not self.project or row < 0 or row >= len(self.project.samples):
            return
        old = self.project.samples.index(self.sample) if self.sample else -1
        if not self.unsaved_ok():
            self.samples.blockSignals(True)
            self.samples.setCurrentRow(old)
            self.samples.blockSignals(False)
            return
        try:
            sample = self.project.samples[row]
            frames, annotation = self.project.load_sample(sample)
            self.sample, self.dirty = sample, False
            self.samples.blockSignals(True)
            self.samples.setCurrentRow(row)
            self.samples.blockSignals(False)
            self.guidance = sample.get('guidance')
            self.canvas.set_sample(frames, annotation)
            self.frame_view.setCurrentIndex(2)
            self.reviewed.blockSignals(True)
            self.reviewed.setChecked(sample.get('reviewed', False))
            self.reviewed.blockSignals(False)
            self.apply_button.setEnabled(False)
            self.fit()
            self.status.setText('Paint only the current frame. Green = UI, cyan = checked background, blue = unknown/ignored. Zoom in for text edges.')
            if sample.get('auto_label'):
                meta = sample['auto_label']
                self.status.setText(f"Machine-generated labels, NOT human-reviewed. {meta['stats']['reason']}. "
                                    'Inspect/correct a spot-check, or export eligible training labels with the explicit automatic-label option.')
            if self.auto_hud.isChecked() and self.hud_matches():
                self.apply_hud_template()
        except Exception as error:
            self.samples.blockSignals(True)
            self.samples.setCurrentRow(old)
            self.samples.blockSignals(False)
            self.error(error)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select your recording', '', 'Videos (*.mp4 *.mkv *.avi *.mov *.webm)')
        if path:
            self.video.setText(path)
            self.recording.setText(source_id(path))

    def resize_setting(self):
        return [self.resize_width.value(), self.resize_height.value()] if self.resize_frames.isChecked() else None

    def require_import(self):
        if not self.project:
            raise ValueError('Create or open a separate labeling project first')
        if not self.recording.text().strip():
            raise ValueError('Enter the original recording ID (keep it the same for all related windows)')
        if not self.unsaved_ok():
            return False
        return True

    def import_video(self):
        try:
            if not self.require_import():
                return
            path = permitted_source(self.video.text().strip())
            self.start_worker({'action': 'video', 'path': str(path), 'current': self.current.value(),
                               'resize': self.resize_setting(), 'source': self.recording.text().strip(),
                               'split': self.split.currentData(), 'time_limit_minutes': self.video_minutes.value()})
        except Exception as error:
            self.error(error)

    def import_images(self):
        try:
            if not self.project:
                raise ValueError('Create or open a labeling project first')
            dialog = ImageWindowDialog(self)
            if dialog.exec() != QDialog.Accepted:
                return
            if not dialog.confirm.isChecked():
                raise ValueError('Confirm the chronology and common recording before importing')
            paths = [str(permitted_source(edit.text().strip())) for edit in dialog.paths]
            if not self.recording.text().strip():
                self.recording.setText(source_id(Path(paths[0]).parent))
            if not self.require_import():
                return
            self.start_worker({'action': 'images', 'paths': paths, 'first': self.current.value() - 2,
                               'resize': self.resize_setting(), 'source': self.recording.text().strip(),
                               'split': self.split.currentData()})
        except Exception as error:
            self.error(error)

    def import_video_batch(self):
        try:
            if not self.require_import():
                return
            path = permitted_source(self.video.text().strip())
            last = self.batch_last.value() or None
            if last is not None and last < self.current.value():
                raise ValueError('Last current frame must be at or after Current frame')
            self.start_worker({'action': 'video_batch', 'path': str(path), 'first': self.current.value(),
                               'last': last, 'count': self.batch_count.value(), 'resize': self.resize_setting(),
                               'source': self.recording.text().strip(), 'split': self.split.currentData(),
                               'time_limit_minutes': self.video_minutes.value()})
        except Exception as error:
            self.error(error)

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Compatible LOCAL dense UI segmenter (no download)', '', 'ONNX (*.onnx)')
        if path:
            try:
                if Path(path).stat().st_size > 512 * 1024 * 1024:
                    raise ValueError('Model exceeds the 512 MiB limit')
                self.model_hash = file_hash(path)
                self.model.setText(path)
                self.status.setText('Local model pinned. Select its documented normalization/output type, then request a suggestion. No inference has run yet.')
            except Exception as error:
                self.error(error)

    def suggest(self):
        try:
            if self.sample is None or not self.model_hash:
                raise ValueError('Select a sample and a compatible local ONNX model first')
            self.start_worker({'action': 'suggest', 'model': self.model.text(), 'sha256': self.model_hash,
                               'normalization': self.normalization.currentText(), 'output_kind': self.output_kind.currentText()},
                              frame=self.canvas.frames[2])
        except Exception as error:
            self.error(error)

    def start_worker(self, request, frame=None):
        if self.job.running():
            raise ValueError('Wait for the current labeling operation or cancel it')
        if any(job.running() for job in (self.app.dataset.job, self.app.train.job, self.app.export.job)):
            raise ValueError('Finish/stop the existing trainer job before preparing labels')
        video = request['action'] in ('video', 'video_batch')
        automatic = request['action'] == 'auto_labels'
        minutes = request.get('time_limit_minutes', 30) if video or automatic else 5
        if type(minutes) is not int or not 5 <= minutes <= 120:
            raise ValueError('Video time limit must be 5..120 whole minutes')
        base = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'Temp/opencode'
        base.mkdir(parents=True, exist_ok=True)
        self.worker_temp = tempfile.TemporaryDirectory(prefix='me-ui-label-operation-', dir=base)
        folder = Path(self.worker_temp.name)
        self.request, self.worker_done, self.worker_error = request, False, ''
        if request['action'] in ('sam_smart', 'sam_auto'):
            self.clear_candidates()
        (folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
        if frame is not None:
            np.savez_compressed(folder / 'input.npz', frame=frame)
        self.controls.setEnabled(False)
        self.canvas.setEnabled(False)
        self.candidate_panel.setEnabled(False)
        self.interaction.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText(f'Importing video with a {minutes}-minute time limit. Only selected triplets become color images; '
                            'sequential decoding still visits intervening frames. Cancel stops the import; originals are read-only.'
                            if video else 'Running only this requested local operation. Cancel stops its subprocess; originals are read-only.')
        if automatic:
            self.status.setText(f'Batch auto-labeling training windows with a {minutes}-minute limit. '
                                'Machine labels are saved together only on success; validation and manual labels stay unchanged.')
        self.deadline.start(minutes * 60000)
        self.job.start('mefi.ui_label_worker', ['--request', str(folder / 'request.json')])

    def on_job(self, event):
        if event.get('event') == 'label_progress':
            self.status.setText(f"Decoding up to selected frame {event['total']}: frame {event['frame']}. No training or flow teacher.")
        elif event.get('event') == 'batch_progress':
            self.status.setText(f"Sampling video: frame {event['frame']}/{event['total']} · {event['done']}/{event['windows']} windows staged. Cancel accepts no windows.")
        elif event.get('event') == 'auto_label_progress':
            self.status.setText(f"Auto-labeling {event['done']}/{event['total']} · {event['eligible']} eligible training masks. "
                                'Other results need review; cancel discards this operation, not existing labels.')
        elif event.get('event') == 'label_done':
            self.worker_done = True
        elif event.get('event') == 'sam_progress':
            self.status.setText(f"AI scan {event['done']}/{event['total']} · {event['candidates']} candidates so far. CPU only; cancel discards this operation's results.")
        elif event.get('event') == 'download_progress':
            self.status.setText(f"Downloading labeling model: {event['received'] / 1000000:.1f}/{event['total'] / 1000000:.1f} MB. Verification required before use.")
        elif event.get('event') == 'error':
            self.worker_error = event['message']
            self.error(event['message'])

    def cancel_worker(self):
        self.worker_error = 'Operation cancelled; no result accepted'
        self.job.kill()

    def worker_process_error(self, error):
        if error == self.job.proc.ProcessError.FailedToStart and self.worker_temp:
            self.worker_error = 'Could not start the local labeling subprocess: ' + self.job.proc.errorString()
            self.worker_finished(1)

    def timeout(self):
        if self.request and self.request['action'] == 'auto_labels':
            minutes = self.deadline.interval() // 60000
            self.worker_error = f'The {minutes}-minute auto-label limit was reached; no results accepted. Increase Auto-label limit (minutes) and retry.'
        elif self.request and self.request['action'] in ('video', 'video_batch'):
            minutes = self.deadline.interval() // 60000
            self.worker_error = (f'The {minutes}-minute video import limit was reached; no result accepted. '
                                 'Increase Video time limit (minutes), or choose an earlier frame/range, then retry.')
        else:
            self.worker_error = 'The five-minute operation limit was reached; no result accepted'
        self.job.kill()

    def worker_finished(self, code):
        self.deadline.stop()
        try:
            if code or self.worker_error or not self.worker_done:
                raise ValueError(self.worker_error or self.job.failure())
            folder = Path(self.worker_temp.name)
            info = json.loads((folder / 'info.json').read_text(encoding='utf-8'))
            with np.load(folder / 'result.npz', allow_pickle=False) as data:
                if self.request['action'] == 'suggest':
                    self.set_suggestion(data['probability'].copy(), info)
                elif self.request['action'] == 'sam_download':
                    self.update_sam_status()
                    self.status.setText('EfficientSAM-Ti ready and SHA256 verified. Select AI clicks/box, or Scan current frame. No inference or training started.')
                elif self.request['action'] in ('sam_smart', 'sam_auto'):
                    self.set_candidates(data['masks'].copy(), info)
                elif self.request['action'] == 'auto_labels':
                    from .ui_auto_label import accept_auto_labels
                    changed = accept_auto_labels(self.project, folder, info, project_sha256=self.request['project_sha256'],
                                                 target_ids=self.request['target_ids'])
                    selected = self.project.samples.index(self.sample) if self.sample else None
                    self.dirty = False
                    self.refresh_samples(selected)
                    if selected is not None:
                        self.select_sample(selected)
                    eligible = sum(s['auto_label']['eligible'] for s in changed)
                    audits = sum(s['auto_label'].get('spot_check', False) for s in changed)
                    message = (f'Saved {len(changed)} automatic annotations: {eligible} eligible for training, '
                               f'{len(changed) - eligible} need review, {audits} spot-checks recommended. '
                               'No per-window Save is needed for eligible machine training labels. '
                               'Validation still needs reviewed examples from another recording. ' + ' '.join(info.get('warnings', [])))
                    self.auto_status.setText(message)
                    self.status.setText(message)
                elif self.request['action'] == 'video_batch':
                    records = info.get('samples', [])
                    if (info.get('kind') != 'video-batch' or info.get('prepared') != len(records)
                            or not 1 <= len(records) <= self.request['count']
                            or any(record['provenance'].get('path') != self.request['path'] for record in records)):
                        raise ValueError('Invalid prepared video batch; no windows accepted')
                    added = self.project.add_prepared_samples(folder, records, self.request['source'], self.request['split'])
                    skipped = len(records) - len(added)
                    if added:
                        row = self.project.samples.index(added[0])
                        self.dirty = False
                        self.refresh_samples(row)
                        self.select_sample(row)
                    detail = f' {self.status.text()}' if self.dirty else ''
                    self.status.setText(f'Imported {len(added)} unknown windows; skipped {skipped} duplicates. '
                                        'Review the current frame of each sample. '
                                        + ' '.join(info.get('warnings', [])) + detail)
                else:
                    frames = [data[f'frame{i}'].copy() for i in range(3)]
                    first = self.request['first'] if self.request['action'] == 'images' else self.request['current'] - 2
                    sample = self.project.add_sample(frames, self.request['source'], self.request['split'],
                                                     [first, first + 1, first + 2], info)
                    row = self.project.samples.index(sample)
                    self.dirty = False  # Any previous edit was explicitly saved/discarded before the import.
                    self.refresh_samples(row)
                    self.select_sample(row)
        except Exception as error:
            self.error(error)
        finally:
            self.controls.setEnabled(True)
            self.canvas.setEnabled(True)
            self.candidate_panel.setEnabled(True)
            self.interaction.setEnabled(True)
            self.cancel_button.setEnabled(False)
            if self.worker_temp:
                self.worker_temp.cleanup()
                self.worker_temp = None
            self.request = None

    def set_suggestion(self, probability, info):
        if (self.sample is None or probability.shape != self.canvas.annotation.valid.shape
                or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1):
            raise ValueError('Suggestion must be a finite [0,1] map matching the prepared current frame')
        self.clear_candidates()
        self.suggestion_kind = 'dense'
        self.canvas.suggestion = probability
        self.suggestion_info = info
        self.apply_button.setEnabled(True)
        self.canvas.update()
        self.status.setText('Magenta is only an AI suggestion; no label changed. Apply a draft if useful, then correct mistakes and explicitly review the sample.')

    def import_probability(self):
        if self.sample is None:
            self.error('Select a sample first')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'External AI probability map, not boxes', '', 'Grayscale PNG (*.png)')
        if path:
            try:
                raw = permitted_source(path).read_bytes()
                image = read_image(raw)
                if image.ndim != 2:
                    raise ValueError('Choose a single-channel 8-bit probability PNG, 0..255, matching this current frame')
                from .ui_labeling import digest_bytes
                self.set_suggestion(image.astype(np.float32) / 255,
                                    {'kind': 'external-probability-png', 'path': str(Path(path).resolve()),
                                     'sha256': digest_bytes(raw), 'role': 'unreviewed external AI suggestion'})
            except Exception as error:
                self.error(error)

    def apply_draft(self):
        if self.canvas.suggestion is None or self.sample is None or self.suggestion_kind != 'dense':
            return
        try:
            count = self.canvas.annotation.apply_suggestion(self.canvas.suggestion)
            previous = self.guidance or []
            self.guidance = [*previous, {**self.suggestion_info, 'positive_threshold': .85, 'negative_threshold': .15,
                                         'accepted_draft_pixels': count, 'review_required': True}]
            self.edited()
            self.canvas.update()
            self.status.setText(f'AI draft changed {count} previously unknown pixels. Inspect/correct it; this sample is now unreviewed.')
        except Exception as error:
            self.error(error)

    def undo(self, redo=False):
        if self.sample is not None and self.canvas.annotation.undo(redo):
            self.edited()
            self.canvas.update()

    def pin_hud_template(self):
        try:
            if self.sample is None or self.dirty or not self.sample.get('reviewed'):
                raise ValueError('Review and save this sample first, then use its saved HUD as a template')
            frames, annotation = self.project.load_sample(self.sample)
            count = int(((annotation.valid == 255) & (annotation.coverage > 0)).sum())
            if not count:
                raise ValueError('This reviewed sample has no UI pixels to reuse')
            self.hud_template = {'sample_id': self.sample['id'], 'source': self.sample['source'],
                                 'height': self.sample['height'], 'width': self.sample['width'],
                                 'reference': frames[2], 'annotation': annotation,
                                 'split': self.sample['split'], 'revision': self.sample.get('revision'),
                                 'provenance': {'kind': 'reviewed-hud-template', 'sample_id': self.sample['id'],
                                                'revision': self.sample.get('revision'), 'frame_ids': self.sample['frame_ids'].copy(),
                                                'annotation': self.sample['annotation'].copy(), 'frames': self.sample['frames'].copy()}}
            self.hud_status.setText(f"HUD template: reviewed frame {self.sample['frame_ids'][2]} · {count:,} UI pixels. "
                                    'Per-window reuse stays in this recording; batch labeling can use other confirmed same-layout recordings.')
            self.same_hud.setChecked(False)
            self.status.setText('Saved reviewed HUD pinned in memory. Use Auto-label all compatible training windows for a saved batch, '
                                'or the older per-window draft reuse. No other sample has changed.')
        except Exception as error:
            self.error(error)

    def hud_matches(self):
        template, sample = self.hud_template, self.sample
        return bool(template and sample and not sample.get('reviewed') and not sample.get('auto_label') and template['sample_id'] != sample['id']
                    and (template['source'], template['height'], template['width'])
                    == (sample['source'], sample['height'], sample['width']))

    def apply_hud_template(self):
        try:
            if not self.hud_matches():
                raise ValueError('Choose a different unreviewed window from the same recording and prepared size as the HUD template')
            template, tolerance = self.hud_template, self.hud_tolerance.value()
            count = self.canvas.annotation.reuse_hud(template['annotation'], template['reference'], self.canvas.frames, tolerance)
            if count:
                self.guidance = [*(self.guidance or []), {**template['provenance'], 'rgb_tolerance': tolerance,
                                                         'admission': 'foreground only; all three target frames color-matched; unknown pixels only',
                                                         'accepted_draft_pixels': count, 'review_required': True}]
                self.edited()
                self.frame_view.setCurrentIndex(2)
                self.canvas.update()
            self.status.setText(f'Reused {count:,} matching HUD pixels as an unsaved draft. '
                                'Background and existing labels were not copied/changed. Check missing or changed text, then review and save.')
        except Exception as error:
            self.error(error)

    def background_remainder(self):
        if self.sample is None:
            return
        if QMessageBox.question(self, 'Review the remaining background',
                                'Have you checked ALL remaining unknown pixels for missed UI? '
                                'This marks them as background. Missing UI would become a wrong training label.',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            annotation = self.canvas.annotation
            annotation.checkpoint()
            annotation.coverage[annotation.valid == 0] = 0
            annotation.valid[:] = 255
            self.edited()
            self.canvas.update()

    def auto_label_all(self):
        try:
            if not self.project or not self.hud_template or self.hud_template['split'] != 'train':
                raise ValueError('Pin one saved reviewed TRAINING HUD example first; validation may never teach automatic labels')
            if not self.unsaved_ok():
                return
            from .ui_auto_label import HudTeacher, MAX_SAMPLES
            template = self.hud_template
            HudTeacher(template['reference'], template['annotation'], tolerance=self.hud_tolerance.value(), padding=self.auto_padding.value())
            targets = [s for s in self.project.samples if s['split'] == 'train' and not s.get('reviewed')
                       and (not s.get('annotation') or s.get('auto_label'))
                       and (s['height'], s['width']) == (template['height'], template['width'])
                       and (s['source'] == template['source'] or self.same_hud.isChecked())]
            if not targets:
                raise ValueError('No untouched/machine training windows match. Import windows first; for another recording confirm the same game/HUD layout')
            if len(targets) > MAX_SAMPLES:
                raise ValueError(f'At most {MAX_SAMPLES} windows per automatic-label operation')
            if QMessageBox.question(self, 'Generate automatic TRAINING labels',
                                    f'Generate and save machine labels for {len(targets)} training windows? '
                                    'Reviewed labels, manual drafts and validation remain unchanged. '
                                    'This uses local HUD colors and temporal consistency, not a semantic HUD classifier. '
                                    'It can miss changed/transparent UI or confuse matching scenery. '
                                    'Uncertain pixels stay unknown. Check recommended spot-checks and use reviewed validation. '
                                    'No download, upload or training will start.',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            self.start_worker({'action': 'auto_labels', 'project_root': str(self.project.root),
                               'project_sha256': self.project.disk_hash, 'template_id': template['sample_id'],
                               'template_revision': template['revision'], 'target_ids': [s['id'] for s in targets],
                               'tolerance': self.hud_tolerance.value(), 'padding': self.auto_padding.value(),
                               'cross_source': self.same_hud.isChecked(), 'time_limit_minutes': self.auto_minutes.value()})
        except Exception as error:
            self.error(error)

    def next_auto_review(self):
        if not self.project:
            return
        count = len(self.project.samples)
        current = self.project.samples.index(self.sample) if self.sample else -1
        for offset in range(1, count + 1):
            row = (current + offset) % count
            sample = self.project.samples[row]
            meta = sample.get('auto_label') or {}
            if not sample.get('reviewed') and (not meta or not meta.get('eligible') or meta.get('spot_check')):
                self.select_sample(row)
                return
        self.status.setText('No uncertain/recommended audit windows remain. Eligible automatic labels are still machine-generated, not human truth.')

    def reset_unknown(self):
        if self.sample is not None and QMessageBox.question(self, 'Reset current annotation',
                                                           'Clear the current draft to unknown? You can undo this change.',
                                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            annotation = self.canvas.annotation
            annotation.checkpoint()
            annotation.coverage[:] = 0
            annotation.valid[:] = 0
            self.edited()
            self.canvas.update()

    def save_current(self):
        if self.sample is None:
            self.error('Import/select a sample first')
            return False
        try:
            self.project.save_annotation(self.sample, self.canvas.annotation,
                                         reviewed=self.reviewed.isChecked(), guidance=self.guidance)
            self.dirty = False
            self.refresh_samples(self.project.samples.index(self.sample))
            self.status.setText('Saved reviewed annotation.' if self.reviewed.isChecked() else 'Saved draft; it will NOT be exported until you review it.')
            return True
        except Exception as error:
            self.error(error)
            return False

    def export_dataset(self):
        if not self.project or not self.unsaved_ok():
            return
        folder = QFileDialog.getExistingDirectory(self, 'Choose a NEW/EMPTY export folder (reviewed + optionally eligible machine training labels)')
        if folder:
            try:
                manifest = self.project.export(folder, include_auto=self.include_auto_export.isChecked())
                self.exported_manifest = manifest
                self.use_button.setEnabled(True)
                self.status.setText(f'Exported immutable dataset and all PNGs/hashes/JSON: {manifest}. '
                                    'Machine training labels, if included, remain marked unreviewed and require Allow machine-generated UI training labels on Train. No training started.')
            except Exception as error:
                self.error(error)

    def save_next(self):
        if self.sample is None or not self.reviewed.isChecked():
            self.error('Inspect/correct this sample and check the review checkbox before Save reviewed and next')
            return
        if not self.save_current():
            return
        current, count = self.project.samples.index(self.sample), len(self.project.samples)
        for offset in range(1, count):
            row = (current + offset) % count
            if not self.project.samples[row].get('reviewed'):
                self.select_sample(row)
                return
        self.status.setText('Saved. All current project samples are reviewed; import more varied footage or export a new dataset snapshot.')

    def use_manifest(self):
        if self.exported_manifest:
            self.app.train.ui_labels.setText(str(self.exported_manifest))
            self.app.show_page(1)
            self.app.train.status.setText('UI labels selected. If the export includes automatic labels, explicitly enable Allow machine-generated UI training labels. '
                                          'Choose UI only or Flow + UI and a new run folder. Training has not started.')

    def prepare_close(self):
        if not self.unsaved_ok():
            return False
        self.deadline.stop()
        self.job.kill()
        if self.job.running() and not self.job.proc.waitForFinished(3000):
            self.error('The labeling subprocess has not stopped yet; close again after it finishes')
            return False
        if self.worker_temp:
            self.worker_temp.cleanup()
            self.worker_temp = None
        return True
