"""Single requested labeling operation in a disposable local subprocess."""
import argparse
import json
from pathlib import Path

import numpy as np

from .ui_labeling import image_window, video_window


def emit(**event):
    print(json.dumps(event), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding='utf-8'))
        root = args.request.parent
        if any((root / name).exists() for name in ('result.npz', 'info.json')):
            raise ValueError('Worker result files already exist; use a fresh operation directory')
        resize = tuple(request['resize']) if request.get('resize') else None
        if request['action'] == 'images':
            frames, info = image_window(request['paths'], request['first'], resize)
            arrays = {f'frame{i}': im for i, im in enumerate(frames)}
        elif request['action'] == 'video':
            frames, info = video_window(request['path'], request['current'], resize,
                                        progress=lambda index, end: emit(event='label_progress', frame=index, total=end))
            arrays = {f'frame{i}': im for i, im in enumerate(frames)}
        elif request['action'] == 'video_batch':
            from .ui_label_batch import prepare_video_batch
            info = prepare_video_batch(request['path'], root, count=request['count'], first=request['first'],
                                       last=request.get('last'), resize=resize,
                                       progress=lambda index, end, done, total:
                                       emit(event='batch_progress', frame=index, total=end, done=done, windows=total))
            arrays = {}
        elif request['action'] == 'auto_labels':
            from .ui_auto_label import prepare_auto_labels
            info = prepare_auto_labels(request['project_root'], root, project_sha256=request['project_sha256'],
                                       template_id=request['template_id'], template_revision=request['template_revision'],
                                       target_ids=request['target_ids'], tolerance=request['tolerance'], padding=request['padding'],
                                       cross_source=request['cross_source'], progress=lambda done, total, eligible:
                                       emit(event='auto_label_progress', done=done, total=total, eligible=eligible))
            arrays = {}
        elif request['action'] == 'suggest':
            from .ui_label_assist import suggest
            with np.load(root / 'input.npz', allow_pickle=False) as data:
                frame = data['frame']
            probability, info = suggest(frame, request['model'], request['sha256'],
                                        normalization=request['normalization'], output_kind=request['output_kind'])
            arrays = {'probability': probability}
        elif request['action'] == 'sam_download':
            from .ui_sam import download_model, MODEL_SHA256
            path = download_model(progress=lambda done, total: emit(event='download_progress', received=done, total=total))
            info, arrays = {'model': str(path), 'sha256': MODEL_SHA256}, {}
        elif request['action'] in ('sam_smart', 'sam_auto'):
            from .ui_sam import EfficientSam
            with np.load(root / 'input.npz', allow_pickle=False) as data:
                frame = data['frame']
            model = EfficientSam()
            model.set_frame(frame)
            if request['action'] == 'sam_auto':
                masks, info = model.propose(grid=request['grid'], progress=lambda done, total, count:
                                            emit(event='sam_progress', done=done, total=total, candidates=count))
            else:
                masks, info = model.propose(points=request.get('points'), labels=request.get('labels'), box=request.get('box'))
            arrays = {'masks': masks}
        else:
            raise ValueError('Unknown labeling action')
        with (root / 'result.npz').open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        with (root / 'info.json').open('x', encoding='utf-8') as handle:
            json.dump(info, handle, indent=2)
        emit(event='label_done', action=request['action'])
        return 0
    except Exception as error:  # A failed model/decoder must not crash the editor.
        emit(event='error', message=str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
