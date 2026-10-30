"""Copy an exported runtime model (and its optional DIS-bypass companion)."""
import json
import os
from pathlib import Path
import shutil
import tempfile


def _copy(source, destination):
    fd, name = tempfile.mkstemp(prefix='.' + destination.name, suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, name)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.remove(name)


def _metadata(destination, metadata):
    fd, name = tempfile.mkstemp(prefix='.' + destination.name, suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=1)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.remove(name)


def copy_export(source, destination):
    """Copy all four files for a refined export, even when renaming it.

    Metadata is published last. Incomplete updates fail runtime bundle checks
    instead of mixing checkpoints. Stale companions of a legacy single export
    are ignored, not deleted (they might be user work).
    """
    source, destination = Path(source), Path(destination)
    if os.path.normcase(os.path.abspath(source)) == os.path.normcase(os.path.abspath(destination)):
        return
    primary = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    paired = primary.get('integrated_refiner', primary.get('arch') == 'pyramid-refine-v1')
    if paired:
        companion = source.with_suffix('.dis.onnx')
        secondary = json.loads(companion.with_suffix('.json').read_text(encoding='utf-8'))
        if (not companion.is_file() or not primary.get('bundle_id')
                or primary['bundle_id'] != secondary.get('bundle_id')
                or secondary.get('integrated_refiner') is not False
                or secondary.get('arch') != 'pyramid-v2'
                or any(primary.get(key) != secondary.get(key) for key in ('width', 'height', 'preset'))):
            raise ValueError('AI/DIS export pair is missing or mismatched; export it again')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if paired:
        target = destination.with_suffix('.dis.onnx')
        for metadata in (primary, secondary):
            metadata.update(dis_model=target.name, refined_model=destination.name)
        _copy(companion, target)
        _metadata(target.with_suffix('.json'), secondary)
    _copy(source, destination)
    _metadata(destination.with_suffix('.json'), primary)
    stale = Path(str(destination) + '.data')
    if stale.exists():
        stale.unlink()
