"""Local, review-first UI annotation projects and immutable training manifests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import cv2
import numpy as np


PROJECT_FORMAT = 'me-ui-label-project-1'
LABEL_FORMAT = 'me-ui-labels-1'
REPO = Path(__file__).resolve().parents[3]
LOCKED_CAPTURE = REPO / 'build-project/Release/captures/20261101-124214'
MAX_PIXELS = 4096 * 4096


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_id(path):
    # Same canonical path identity as the original flow dataset builder.
    return digest_bytes(os.path.normcase(os.path.realpath(path)).encode('utf-8'))


def permitted_source(path):
    path = Path(path).resolve()
    if path.is_relative_to(LOCKED_CAPTURE.resolve()):
        raise ValueError('The locked regression capture cannot be imported into a labeling project')
    if not path.is_file():
        raise ValueError('Select an existing source file')
    return path


def new_folder(path):
    path = Path(path).resolve()
    protected = [REPO / name for name in ('backup', 'third_party', 'build-project', 'bin')]
    protected += [REPO / 'tools/ai-train' / name for name in ('.venv', 'runs')]
    protected.append(Path('C:/MotionEnhancer'))
    if any(path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Choose a separate labeling/export folder, outside builds, runs, SDKs, backups and installations')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Choose a new or empty folder; existing files are never replaced')
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.labels-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def relative_file(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Invalid project file reference')
    name, expected = ref['path'], ref['sha256']
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or Path(name).drive or '..' in Path(name).parts):
        raise ValueError('Project files must use relative paths')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('Invalid project path/hash')
    raw = path.read_bytes()
    if digest_bytes(raw) != expected:
        raise ValueError(f'Project file changed: {name}')
    return path, raw


def read_image(raw):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or any(d > 4096 for d in image.shape[:2]):
        raise ValueError('Expected an 8-bit image, at most 4096 x 4096 pixels')
    return image


def write_png(root, name, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Could not encode annotation PNG')
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('Annotation path escapes the project')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encoded.tobytes()
    with path.open('xb') as handle:
        handle.write(raw)
    return {'path': name, 'sha256': digest_bytes(raw)}


def prepare_frames(frames, resize=None):
    if len(frames) != 3 or any(im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 for im in frames):
        raise ValueError('Select three 8-bit RGB frames')
    if any(im.shape != frames[0].shape for im in frames):
        raise ValueError('All three frames must have identical dimensions')
    if resize is not None:
        w, h = resize
    else:
        h, w = frames[0].shape[:2]
    if any(type(d) is not int or d < 64 or d > 4096 or d % 16 for d in (w, h)) or w * h > MAX_PIXELS:
        raise ValueError('Prepared dimensions must be multiples of 16 and at least 64; enable explicit resizing if needed')
    if resize is None:
        return [np.ascontiguousarray(im) for im in frames]
    return [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if w <= im.shape[1] and h <= im.shape[0] else cv2.INTER_LINEAR)
            for im in frames]


def image_window(paths, first, resize=None):
    if len(paths) != 3 or type(first) is not int or first < 0:
        raise ValueError('Choose three chronological images and a nonnegative first frame number')
    paths = [permitted_source(p) for p in paths]
    if len(set(paths)) != 3:
        raise ValueError('Choose three distinct frame files')
    raw = [p.read_bytes() for p in paths]
    frames = [read_image(value) for value in raw]
    if any(im.ndim != 3 or im.shape[2] != 3 for im in frames):
        raise ValueError('Frame files must have three color channels')
    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in frames]
    return prepare_frames(rgb, resize), {'kind': 'explicit-images', 'paths': [str(p) for p in paths],
                                        'sha256': [digest_bytes(value) for value in raw],
                                        'original_shape': list(rgb[0].shape), 'resize': resize,
                                        'chronology': 'user-confirmed consecutive frame order'}


def video_window(path, current, resize=None, progress=None):
    path = permitted_source(path)
    if type(current) is not int or current < 2:
        raise ValueError('The current frame number must be at least 2 (zero-based)')
    before = path.stat()
    digest = file_hash(path)
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError('The installed video decoder could not open this file')
        # Decode from the beginning instead of trusting approximate keyframe
        # seeking. Frame IDs are exact ordinals in this decoder's output stream.
        for index in range(current + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video ended before the selected three-frame window')
            if index >= current - 2:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if progress and index % 120 == 0:
                progress(index, current)
    finally:
        cap.release()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source video changed during decoding')
    return prepare_frames(frames, resize), {'kind': 'video', 'path': str(path), 'sha256': digest,
                                          'source_id': source_id(path), 'original_shape': list(frames[0].shape),
                                          'resize': resize, 'frame_ids': [current - 2, current - 1, current]}


class Annotation:
    """Unknown by default; AI suggestions never overwrite manually known pixels."""
    def __init__(self, height, width, coverage=None, valid=None):
        self.coverage = np.zeros((height, width), np.uint8) if coverage is None else coverage.copy()
        self.valid = np.zeros((height, width), np.uint8) if valid is None else valid.copy()
        self.undo_stack, self.redo_stack = [], []

    def checkpoint(self):
        self.undo_stack.append((self.coverage.copy(), self.valid.copy()))
        budget = max(1, min(20, (64 * 1024 * 1024) // (self.coverage.nbytes * 2)))
        self.undo_stack = self.undo_stack[-budget:]
        self.redo_stack.clear()

    def undo(self, redo=False):
        source, target = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if not source:
            return False
        target.append((self.coverage.copy(), self.valid.copy()))
        self.coverage, self.valid = source.pop()
        return True

    def stroke(self, start, end, radius, mode, coverage=255):
        if mode not in ('ui', 'background', 'unknown') or not 0 <= radius <= 256 or not 1 <= coverage <= 255:
            raise ValueError('Invalid brush settings')
        mask = np.zeros_like(self.valid)
        cv2.line(mask, tuple(start), tuple(end), 255, 2 * radius + 1, cv2.LINE_8)
        cv2.circle(mask, tuple(end), radius, 255, -1)
        selected = mask != 0
        self.coverage[selected] = coverage if mode == 'ui' else 0
        self.valid[selected] = 0 if mode == 'unknown' else 255

    def apply_suggestion(self, probability, low=.15, high=.85):
        if probability.shape != self.coverage.shape or not np.isfinite(probability).all() or not 0 <= low < high <= 1:
            raise ValueError('Invalid AI probability map or confidence band')
        if probability.min() < 0 or probability.max() > 1:
            raise ValueError('AI probability map must be in [0,1]')
        self.checkpoint()
        certain = ((probability <= low) | (probability >= high)) & (self.valid == 0)
        # Model confidence is not compositing coverage. Confident predictions
        # become binary draft labels; antialiased coverage is edited manually.
        self.coverage[certain] = np.where(probability[certain] >= high, 255, 0).astype(np.uint8)
        self.valid[certain] = 255
        return int(certain.sum())

    def apply_foreground(self, mask):
        """Object mask admission never labels the rest of a frame background."""
        if mask.shape != self.valid.shape or mask.dtype != np.bool_:
            raise ValueError('Expected a boolean candidate mask in prepared-frame coordinates')
        selected = mask & (self.valid == 0)
        count = int(selected.sum())
        if count:
            self.checkpoint()
            self.coverage[selected] = 255
            self.valid[selected] = 255
        return count

    def reuse_hud(self, template, reference, frames, tolerance=12):
        """Copy only matching, stable foreground into unknown pixels as a draft.

        A same-coordinate color match is a conservative editing aid, not proof
        of UI identity. Never transfer background or a template's review flag.
        """
        shape = (*self.valid.shape, 3)
        if (type(tolerance) is not int or not 0 <= tolerance <= 32
                or template.valid.shape != self.valid.shape or template.coverage.shape != self.valid.shape
                or len(frames) != 3
                or any(im.shape != shape or im.dtype != np.uint8 for im in [reference, *frames])):
            raise ValueError('HUD reuse needs matching prepared dimensions and an RGB tolerance of 0..32')
        selected = (template.valid == 255) & (template.coverage > 0) & (self.valid == 0)
        reference = reference.astype(np.int16)
        for frame in frames:
            selected &= np.max(np.abs(frame.astype(np.int16) - reference), axis=2) <= tolerance
        count = int(selected.sum())
        if count:
            self.checkpoint()
            self.coverage[selected] = template.coverage[selected]
            self.valid[selected] = 255
        return count


class Project:
    def __init__(self, root, document):
        self.root, self.document = root, document
        raw = (root / 'project.json').read_bytes()
        if json.loads(raw) != document:
            raise ValueError('Project changed while opening; reopen it')
        self.disk_hash = digest_bytes(raw)

    @classmethod
    def create(cls, root):
        root = new_folder(root)
        doc = {'format': PROJECT_FORMAT, 'id': uuid.uuid4().hex, 'samples': [], 'label_version': 'reviewed-ui-v1'}
        atomic_json(root / 'project.json', doc)
        return cls(root, doc)

    @classmethod
    def open(cls, root):
        root = Path(root).resolve()
        doc = json.loads((root / 'project.json').read_text(encoding='utf-8'))
        if (not isinstance(doc, dict) or doc.get('format') != PROJECT_FORMAT or not isinstance(doc.get('samples'), list)
                or not isinstance(doc.get('id'), str) or not isinstance(doc.get('label_version'), str)
                or not doc['label_version']):
            raise ValueError('Not a UI labeling project')
        project = cls(root, doc)
        sources, hashes, ids = {}, {}, set()
        for sample in doc['samples']:
            if (not isinstance(sample, dict) or not isinstance(sample.get('id'), str) or sample['id'] in ids
                    or len(sample['id']) != 32 or any(c not in '0123456789abcdef' for c in sample['id'])):
                raise ValueError('Invalid/duplicate project sample')
            if type(sample.get('reviewed')) is not bool:
                raise ValueError('Project review flags must be explicit booleans')
            ids.add(sample['id'])
            project.check_identity(sample['source'], sample['split'], sample['frames'], sources, hashes)
            frames, annotation = project.load_sample(sample)
            if sample.get('reviewed') and not annotation.valid.any():
                raise ValueError('A reviewed sample has no known pixels')
        return project

    @property
    def samples(self):
        return self.document['samples']

    def persist(self):
        # Refuse external/concurrent edits rather than silently overwriting them.
        if digest_bytes((self.root / 'project.json').read_bytes()) != self.disk_hash:
            raise ValueError('Project changed in another editor; reopen it instead of overwriting changes')
        atomic_json(self.root / 'project.json', self.document)
        self.disk_hash = digest_bytes((self.root / 'project.json').read_bytes())

    @staticmethod
    def check_identity(source, split, frames, sources, hashes):
        if not isinstance(source, str) or not source.strip() or split not in ('train', 'val'):
            raise ValueError('Choose a recording ID and train/validation split; locked test data is not editable here')
        if source in sources and sources[source] != split:
            raise ValueError('This recording already belongs to another split; use a different recording for validation')
        sources[source] = split
        for ref in frames:
            digest = ref['sha256']
            if digest in hashes and hashes[digest] != split:
                raise ValueError('Identical frames cannot cross training/validation splits')
            hashes[digest] = split

    def add_sample(self, frames, source, split, frame_ids, provenance):
        if not isinstance(source, str):
            raise ValueError('A recording ID is required')
        source = source.strip()
        frames = prepare_frames(frames)
        if (len(frame_ids) != 3 or any(type(i) is not int or i < 0 for i in frame_ids)
                or frame_ids != list(range(frame_ids[0], frame_ids[0] + 3))):
            raise ValueError('Three consecutive chronological frame IDs are required')
        name = uuid.uuid4().hex
        refs = []
        for index, frame in enumerate(frames):
            # Encode before writing so leakage can be refused without creating
            # a half-imported sample in the project.
            ok, data = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError('Could not prepare frame PNG')
            refs.append((f'frames/{name}/{index}.png', data.tobytes()))
        new_refs = [{'path': p, 'sha256': digest_bytes(raw)} for p, raw in refs]
        sources, hashes = {}, {}
        for sample in self.samples:
            self.check_identity(sample['source'], sample['split'], sample['frames'], sources, hashes)
            if sample['source'] == source and sample['frame_ids'] == frame_ids:
                raise ValueError('This recording/frame window is already in the project')
            if (provenance.get('kind') == 'video' and sample['provenance'].get('kind') == 'video'
                    and provenance['sha256'] == sample['provenance']['sha256'] and sample['split'] != split):
                raise ValueError('A renamed copy of the same video cannot cross training/validation splits')
        self.check_identity(source, split, new_refs, sources, hashes)
        for path, raw in refs:
            full = self.root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            with full.open('xb') as handle:
                handle.write(raw)
        h, w = frames[0].shape[:2]
        sample = {'id': name, 'source': source.strip(), 'split': split, 'frame_ids': frame_ids,
                  'height': h, 'width': w, 'frames': new_refs, 'reviewed': False,
                  'annotation': None, 'provenance': provenance}
        self.samples.append(sample)
        self.persist()
        return sample

    def load_sample(self, sample):
        h, w = sample['height'], sample['width']
        if any(type(d) is not int or d < 64 or d > 4096 or d % 16 for d in (h, w)):
            raise ValueError('Invalid prepared project dimensions')
        ids = sample.get('frame_ids')
        if (not isinstance(ids, list) or len(ids) != 3 or any(type(i) is not int or i < 0 for i in ids)
                or ids != list(range(ids[0], ids[0] + 3)) or len(sample['frames']) != 3):
            raise ValueError('Project samples require three consecutive chronological frames')
        frames = [read_image(relative_file(self.root, ref)[1]) for ref in sample['frames']]
        if any(im.shape != (h, w, 3) for im in frames):
            raise ValueError('Project frame dimensions changed')
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        coverage = valid = None
        if sample.get('annotation'):
            coverage, valid = [read_image(relative_file(self.root, sample['annotation'][key])[1]) for key in ('coverage', 'valid')]
            if coverage.shape != (h, w) or valid.shape != (h, w) or not np.isin(valid, [0, 255]).all():
                raise ValueError('Invalid project coverage/known-pixel masks')
        return frames, Annotation(h, w, coverage, valid)

    def add_prepared_samples(self, root, records, source, split):
        """Validate a bounded staged batch, then publish all new samples together.

        Duplicate ordinals are skipped. Leakage, changed staged files or an
        external project edit refuse the whole batch, not just its final window.
        """
        root = Path(root).resolve()
        if not isinstance(source, str) or not isinstance(records, list) or not 1 <= len(records) <= 100:
            raise ValueError('Choose a recording ID and a batch of 1..100 windows')
        source = source.strip()
        sources, hashes, windows, video_splits = {}, {}, set(), {}
        for sample in self.samples:
            self.check_identity(sample['source'], sample['split'], sample['frames'], sources, hashes)
            windows.add((sample['source'], tuple(sample['frame_ids'])))
            origin = sample.get('provenance', {})
            if origin.get('kind') == 'video':
                video_splits[origin['sha256']] = sample['split']
        # Check the chosen split even when every incoming ordinal is a duplicate.
        self.check_identity(source, split, [], sources, hashes)
        prepared = []
        for record in records:
            h, w, ids, refs = record['height'], record['width'], record['frame_ids'], record['frames']
            origin = record['provenance']
            if (any(type(d) is not int or d < 64 or d > 4096 or d % 16 for d in (h, w))
                    or not isinstance(ids, list) or len(ids) != 3 or any(type(i) is not int or i < 0 for i in ids)
                    or ids != list(range(ids[0], ids[0] + 3)) or len(refs) != 3):
                raise ValueError('Invalid prepared batch dimensions or frame ordinals')
            for ref in refs:
                _, raw = relative_file(root, ref)
                if read_image(raw).shape != (h, w, 3):
                    raise ValueError('Prepared batch frame dimensions changed')
            if origin.get('kind') != 'video' or origin.get('frame_ids') != ids:
                raise ValueError('Prepared batch needs matching video provenance')
            digest = origin.get('sha256')
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError('Prepared video checksum is missing')
            if digest in video_splits and video_splits[digest] != split:
                raise ValueError('A renamed copy of the same video cannot cross training/validation splits')
            self.check_identity(source, split, refs, sources, hashes)
            key = (source, tuple(ids))
            if key in windows:
                continue
            windows.add(key)
            name = uuid.uuid4().hex
            new_refs = [{'path': f'frames/{name}/{i}.png', 'sha256': ref['sha256']} for i, ref in enumerate(refs)]
            sample = {'id': name, 'source': source, 'split': split, 'frame_ids': ids.copy(),
                      'height': h, 'width': w, 'frames': new_refs, 'reviewed': False,
                      'annotation': None, 'provenance': copy.deepcopy(origin)}
            prepared.append((sample, refs))
        if digest_bytes((self.root / 'project.json').read_bytes()) != self.disk_hash:
            raise ValueError('Project changed in another editor; reopen it instead of overwriting changes')
        created, before = [], len(self.samples)
        try:
            for sample, refs in prepared:
                for new, ref in zip(sample['frames'], refs):
                    _, raw = relative_file(root, ref)
                    path = self.root / new['path']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as handle:
                        created.append(path)
                        handle.write(raw)
            self.samples.extend(sample for sample, _ in prepared)
            if prepared:
                self.persist()
        except Exception:
            del self.samples[before:]
            # Only these newly allocated immutable files belong to this attempt.
            for path in reversed(created):
                path.unlink(missing_ok=True)
                if not any(path.parent.iterdir()):
                    path.parent.rmdir()
            raise
        return [sample for sample, _ in prepared]

    def save_annotation(self, sample, annotation, *, reviewed=False, guidance=None):
        if sample not in self.samples or type(reviewed) is not bool:
            raise ValueError('Sample does not belong to this project')
        h, w = sample['height'], sample['width']
        if (annotation.coverage.shape != (h, w) or annotation.valid.shape != (h, w)
                or annotation.coverage.dtype != np.uint8 or annotation.valid.dtype != np.uint8
                or not np.isin(annotation.valid, [0, 255]).all()):
            raise ValueError('Invalid annotation shape/type')
        if reviewed and not (annotation.valid.reshape(h // 2, 2, w // 2, 2).min(axis=(1, 3)) == 255).any():
            raise ValueError('Review needs at least one fully known 2 x 2 cell')
        revision = uuid.uuid4().hex
        prefix = f'annotations/{sample["id"]}/{revision}'
        refs = {key: write_png(self.root, prefix + '-' + key + '.png', array)
                for key, array in (('coverage', annotation.coverage), ('valid', annotation.valid))}
        sample.update(annotation=refs, reviewed=reviewed, guidance=copy.deepcopy(guidance), revision=revision)
        self.persist()

    def export(self, destination):
        if digest_bytes((self.root / 'project.json').read_bytes()) != self.disk_hash:
            raise ValueError('Project changed in another editor; reopen before exporting')
        if Path(destination).resolve().is_relative_to(self.root):
            raise ValueError('Export to a separate new folder, not inside the editable project')
        samples = [sample for sample in self.samples if sample.get('reviewed') is True]
        if not any(s['split'] == 'train' for s in samples) or not any(s['split'] == 'val' for s in samples):
            raise ValueError('Review at least one training and one validation sample from separate recordings first')
        # Validate all project references before creating an immutable export.
        sources, hashes = {}, {}
        for sample in samples:
            self.check_identity(sample['source'], sample['split'], sample['frames'], sources, hashes)
            _, annotation = self.load_sample(sample)
            h, w = annotation.valid.shape
            if not (annotation.valid.reshape(h // 2, 2, w // 2, 2).min(axis=(1, 3)) == 255).any():
                raise ValueError('A reviewed annotation has no fully known 2 x 2 cell')
        root = new_folder(destination)
        manifest = {'format': LABEL_FORMAT, 'label_version': self.document['label_version'], 'samples': []}
        for sample in samples:
            item = {key: copy.deepcopy(sample[key]) for key in ('id', 'source', 'split', 'height', 'width', 'frame_ids')}
            def copy_ref(ref):
                _, raw = relative_file(self.root, ref)
                path = root / ref['path']
                if not path.resolve().is_relative_to(root):
                    raise ValueError('Export file escapes the new dataset folder')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if path.read_bytes() != raw:
                        raise ValueError('Conflicting exported file reference')
                else:
                    with path.open('xb') as handle:
                        handle.write(raw)
                return ref.copy()
            item['frames'] = [copy_ref(ref) for ref in sample['frames']]
            for key in ('coverage', 'valid'):
                item[key] = copy_ref(sample['annotation'][key])
            manifest['samples'].append(item)
        # Preserve origin/model/review provenance separately from the strict
        # trainer contract. Nothing in this project is installed or activated.
        atomic_json(root / 'provenance.json', {'project_id': self.document['id'], 'samples': samples})
        atomic_json(root / 'labels.json', manifest)
        return root / 'labels.json'
