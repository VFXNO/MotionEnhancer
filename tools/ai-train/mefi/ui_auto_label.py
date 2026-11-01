"""Local, seed-guided HUD pseudo-labels. Never human truth or validation labels.

Learn local foreground/background colors from one reviewed training example.
Within small seed neighborhoods, require unchanged HUD anchors and agreement
across all three target frames. Abstain rather than label the rest background.
"""
import copy
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

from .ui_labeling import (Annotation, PROJECT_FORMAT, Project, digest_bytes,
                          read_image, relative_file)


FORMAT = 'me-ui-auto-label-1'
MAX_SAMPLES = 5000
MAX_STAGED_BYTES = 512 * 1024 * 1024
MIN_POSITIVE_CELLS = 4
MIN_NEGATIVE_CELLS = 16


def cell_counts(annotation):
    h, w = annotation.valid.shape
    known = annotation.valid.reshape(h // 2, 2, w // 2, 2).min(axis=(1, 3)) == 255
    positive = annotation.coverage.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)) >= 127.5
    return {'positive_cells': int((known & positive).sum()),
            'negative_cells': int((known & ~positive).sum()), 'known_pixels': int((annotation.valid == 255).sum())}


def palette(rgb):
    """Deterministic, bounded local color prototypes; no model/download/training."""
    if not len(rgb):
        return np.empty((0, 3), np.int16)
    quantized = rgb.astype(np.int32) // 16
    keys = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    selected = values[np.argsort(-counts, kind='stable')[:32]]
    return np.array([np.rint(rgb[keys == key].mean(axis=0)) for key in selected], np.int16)


def color_distance(rgb, colors):
    distance = np.full(rgb.shape[:2], 256, np.int16)
    rgb = rgb.astype(np.int16)
    for color in colors:
        np.minimum(distance, np.abs(rgb - color).max(axis=2), out=distance)
    return distance


class HudTeacher:
    def __init__(self, reference, annotation, *, tolerance=12, padding=16):
        if type(tolerance) is not int or not 0 <= tolerance <= 32 or type(padding) is not int or not 0 <= padding <= 64:
            raise ValueError('Choose RGB tolerance 0..32 and HUD neighborhood padding 0..64')
        h, w = annotation.valid.shape
        if reference.shape != (h, w, 3) or reference.dtype != np.uint8:
            raise ValueError('HUD template dimensions/type do not match its labels')
        counts = cell_counts(annotation)
        if counts['positive_cells'] < MIN_POSITIVE_CELLS or counts['negative_cells'] < MIN_NEGATIVE_CELLS:
            raise ValueError('The reviewed template needs at least 4 fully known UI and 16 background 2x2 cells')
        foreground = (annotation.valid == 255) & (annotation.coverage >= 128)
        background = (annotation.valid == 255) & (annotation.coverage == 0)
        neighborhood = cv2.dilate(foreground.astype(np.uint8), np.ones((2 * padding + 1,) * 2, np.uint8))
        count, regions, stats, _ = cv2.connectedComponentsWithStats(neighborhood, connectivity=8)
        if count > 65:
            raise ValueError('Too many separate HUD seed regions; increase neighborhood padding or simplify the template')
        self.shape, self.tolerance, self.regions = (h, w), tolerance, []
        for index in range(1, count):
            x, y, width, height, _ = stats[index]
            box = np.s_[y:y + height, x:x + width]
            region = regions[box] == index
            fg, bg = foreground[box] & region, background[box] & region
            colors, negatives = palette(reference[box][fg]), palette(reference[box][bg])
            if not len(colors) or not len(negatives):
                continue
            self.regions.append((box, region, fg, bg, colors, negatives, reference[box].copy(),
                                 annotation.coverage[box].copy(), annotation.valid[box].copy()))
        if not self.regions:
            raise ValueError('Paint checked background near the HUD shapes too; each usable seed region needs both classes')

    def label(self, frames):
        if len(frames) != 3 or any(im.shape != (*self.shape, 3) or im.dtype != np.uint8 for im in frames):
            raise ValueError('Automatic labeling requires the same prepared size and HUD layout as the template')
        annotation = Annotation(*self.shape)
        anchors, possible, active = 0, 0, 0
        for box, region, fg, bg, colors, negatives, reference, coverage, valid in self.regions:
            crops = [im[box] for im in frames]
            matching = np.ones(region.shape, bool)
            for crop in crops:
                matching &= np.abs(crop.astype(np.int16) - reference.astype(np.int16)).max(axis=2) <= self.tolerance
            matched, total = int((matching & fg).sum()), int(fg.sum())
            anchors, possible = anchors + matched, possible + total
            # A broad same-color area without the original HUD shape is not a
            # reason to hallucinate new glyphs or call scenery background.
            if matched / max(1, total) < .6 or int((matching & bg).sum()) / max(1, int(bg.sum())) < .6:
                continue
            active += 1
            positive, negative = region.copy(), region.copy()
            for crop in crops:
                df, db = color_distance(crop, colors), color_distance(crop, negatives)
                positive &= (df <= self.tolerance) & (db >= self.tolerance + 8)
                negative &= (db <= self.tolerance) & (df >= self.tolerance + 8)
            # Preserve explicitly known, unchanged foreground edge coverage.
            # New shapes are binary weak labels, not estimated antialiasing.
            exact = matching & region & (valid == 255) & (coverage > 0)
            target, known = annotation.coverage[box], annotation.valid[box]
            target[positive], known[positive | negative] = 255, 255
            target[exact], known[exact] = coverage[exact], 255
        counts = cell_counts(annotation)
        eligible = counts['positive_cells'] >= MIN_POSITIVE_CELLS and counts['negative_cells'] >= MIN_NEGATIVE_CELLS
        reason = ('matched HUD and both supervised classes' if eligible else
                  'HUD anchors changed/absent or too few confident UI/background cells')
        return annotation, {**counts, 'anchor_fraction': anchors / max(1, possible), 'active_regions': active,
                            'eligible': eligible, 'reason': reason}


def prepare_auto_labels(project_root, operation_root, *, project_sha256, template_id, template_revision,
                        target_ids, tolerance=12, padding=16, cross_source=False, progress=None):
    project_root, operation_root = Path(project_root).resolve(), Path(operation_root).resolve()
    raw = (project_root / 'project.json').read_bytes()
    if digest_bytes(raw) != project_sha256:
        raise ValueError('Project changed before automatic labeling; reopen it')
    document = json.loads(raw)
    if document.get('format') != PROJECT_FORMAT:
        raise ValueError('Not an editable UI labeling project')
    project = Project(project_root, document)
    if (not isinstance(target_ids, list) or not 1 <= len(target_ids) <= MAX_SAMPLES
            or any(not isinstance(i, str) for i in target_ids) or len(set(target_ids)) != len(target_ids)
            or type(cross_source) is not bool):
        raise ValueError(f'Choose 1..{MAX_SAMPLES} unique training windows')
    lookup = {sample['id']: sample for sample in project.samples}
    template = lookup[template_id]
    if template['split'] != 'train' or template.get('reviewed') is not True or template.get('revision') != template_revision:
        raise ValueError('Automatic labeling needs a saved, reviewed TRAINING template at its pinned revision')
    frames, annotation = project.load_sample(template)
    teacher = HudTeacher(frames[2], annotation, tolerance=tolerance, padding=padding)
    targets = [lookup[name] for name in target_ids]
    for sample in targets:
        if (sample['split'] != 'train' or sample.get('reviewed') or sample['id'] == template_id
                or (sample.get('annotation') and not sample.get('auto_label'))):
            raise ValueError('Automatic labeling may not replace reviewed labels, manual drafts or validation')
        if sample['source'] != template['source'] and not cross_source:
            raise ValueError('Confirm the same HUD/game layout before labeling another training recording')
    records, staged_bytes, warnings = [], 0, []
    for ordinal, sample in enumerate(targets):
        frames, _ = project.load_sample(sample)
        result, stats = teacher.label(frames)
        metadata = {'format': FORMAT, 'method': 'seeded-local-colors+three-frame-consistency-v1',
                    'template_id': template_id, 'template_revision': template_revision,
                    'template_source': template['source'], 'template_split': 'train',
                    'template_annotation': copy.deepcopy(template['annotation']),
                    'template_frames': copy.deepcopy(template['frames']),
                    'rgb_tolerance': tolerance, 'padding': padding, 'stats': stats,
                    'eligible': stats['eligible'], 'human_reviewed': False,
                    'limitations': 'Heuristic pseudo-labels; no semantic HUD guarantee. Unmatched/ambiguous pixels stay unknown.'}
        encoded = []
        for array in (result.coverage, result.valid):
            ok, data = cv2.imencode('.png', array)
            if not ok:
                raise ValueError('Could not encode automatic labels')
            encoded.append(data.tobytes())
        if staged_bytes + sum(map(len, encoded)) > MAX_STAGED_BYTES:
            warnings.append('512 MiB encoded-label staging limit reached; remaining windows were left unchanged.')
            break
        staged_bytes += sum(map(len, encoded))
        refs = {}
        for key, data in zip(('coverage', 'valid'), encoded):
            name = f'auto/{ordinal}/{key}.png'
            path = operation_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(data)
            refs[key] = {'path': name, 'sha256': digest_bytes(data)}
        records.append({'id': sample['id'], 'annotation': refs, 'auto_label': metadata})
        if progress:
            progress(ordinal + 1, len(targets), sum(r['auto_label']['eligible'] for r in records))
    if digest_bytes((project_root / 'project.json').read_bytes()) != project_sha256:
        raise ValueError('Project changed during automatic labeling; no results accepted')
    # A small deterministic audit queue is a recommendation, not a review flag.
    eligible = [r for r in records if r['auto_label']['eligible']]
    audit = {eligible[i * (len(eligible) - 1) // max(1, min(8, len(eligible)) - 1)]['id']
             for i in range(min(8, len(eligible)))}
    for record in records:
        record['auto_label']['spot_check'] = record['id'] in audit
    return {'kind': 'auto-label-batch', 'project_sha256': project_sha256, 'records': records, 'warnings': warnings}


def accept_auto_labels(project, operation_root, result, *, project_sha256, target_ids):
    """Preflight every staged mask, then atomically publish new revisions only."""
    operation_root = Path(operation_root).resolve()
    if (project.disk_hash != project_sha256 or result.get('project_sha256') != project_sha256
            or digest_bytes((project.root / 'project.json').read_bytes()) != project_sha256):
        raise ValueError('Project changed; automatic results refused without replacing labels')
    records, allowed = result.get('records'), set(target_ids)
    if result.get('kind') != 'auto-label-batch' or not isinstance(records, list) or len(records) > len(allowed):
        raise ValueError('Invalid automatic label result')
    lookup, changes, ids, verified = {s['id']: s for s in project.samples}, [], set(), set()
    for record in records:
        name, meta = record['id'], record['auto_label']
        if name not in allowed or name in ids:
            raise ValueError('Unexpected/duplicate automatic result sample')
        ids.add(name)
        sample = lookup[name]
        if (sample['split'] != 'train' or sample.get('reviewed')
                or (sample.get('annotation') and not sample.get('auto_label'))
                or meta.get('format') != FORMAT or meta.get('template_split') != 'train'
                or meta.get('human_reviewed') is not False or type(meta.get('eligible')) is not bool):
            raise ValueError('Automatic results cannot replace human labels or enter validation')
        template = lookup.get(meta.get('template_id'))
        if (not template or template['split'] != 'train' or template.get('reviewed') is not True
                or template.get('revision') != meta.get('template_revision')
                or template['source'] != meta.get('template_source')
                or template['frames'] != meta.get('template_frames')
                or template['annotation'] != meta.get('template_annotation')):
            raise ValueError('Automatic result template identity/revision is invalid')
        for ref in [*sample['frames'], *template['frames'], *template['annotation'].values()]:
            key = (ref['path'], ref['sha256'])
            if key not in verified:
                relative_file(project.root, ref)
                verified.add(key)
        arrays = [read_image(relative_file(operation_root, record['annotation'][key])[1]) for key in ('coverage', 'valid')]
        h, w = sample['height'], sample['width']
        if (any(im.shape != (h, w) for im in arrays) or not np.isin(arrays[1], [0, 255]).all()
                or np.any(arrays[0][arrays[1] == 0])):
            raise ValueError('Invalid automatic coverage/valid masks')
        counts = cell_counts(Annotation(h, w, *arrays))
        eligible = counts['positive_cells'] >= MIN_POSITIVE_CELLS and counts['negative_cells'] >= MIN_NEGATIVE_CELLS
        if meta['eligible'] != eligible or any(meta['stats'].get(k) != v for k, v in counts.items()):
            raise ValueError('Automatic label eligibility/counts do not match staged pixels')
        changes.append((sample, record))
    originals, created = [(s, copy.deepcopy(s)) for s, _ in changes], []
    try:
        for sample, record in changes:
            revision = uuid.uuid4().hex
            refs = {}
            for key in ('coverage', 'valid'):
                _, raw = relative_file(operation_root, record['annotation'][key])
                path = project.root / f'annotations/{sample["id"]}/{revision}-{key}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as handle:
                    created.append(path)
                    handle.write(raw)
                refs[key] = {'path': path.relative_to(project.root).as_posix(), 'sha256': digest_bytes(raw)}
            sample.update(annotation=refs, revision=revision, reviewed=False, auto_label=copy.deepcopy(record['auto_label']),
                          guidance=[{'kind': FORMAT, 'human_reviewed': False, 'template_id': record['auto_label']['template_id']}])
        if changes:
            project.persist()
    except Exception:
        for sample, original in originals:
            sample.clear()
            sample.update(original)
        for path in reversed(created):
            path.unlink(missing_ok=True)
            if not any(path.parent.iterdir()):
                path.parent.rmdir()
        raise
    return [s for s, _ in changes]
