"""Read-only native detector/gating checks on local screenshots; no capture or training.

Requires Pillow only for drawing diagnostic boxes. All outputs go to a NEW folder.
The model is selected by environment override, not installed over any runtime model.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MOTION_ENHANCER_UI_MODEL=str(args.model.resolve()))
    exe = str(args.executable.resolve())

    def run(command, log):
        result = subprocess.run([exe, *command, '--adapter', 'dgpu'], env=env, capture_output=True,
                                text=True, timeout=120)
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Native check failed ({result.returncode}); see {log}')

    run(['--ui-regions-self-test'], args.out / 'native-gating.log')
    reports = []
    for index, image in enumerate(args.images):
        dest = args.out / f'{index:03d}-{image.stem}'
        report = dest.with_suffix('.json')
        run(['--ui-detect-test', str(image.resolve()), str(report.resolve())], dest.with_suffix('.log'))
        data = json.loads(report.read_text())
        preview = Image.open(image).convert('RGB')
        draw = ImageDraw.Draw(preview)
        for region in data['regions']:
            draw.rectangle(tuple(region[k] for k in ('left', 'top', 'right', 'bottom')), outline='red', width=3)
            draw.text((region['left'], region['top']), f"{region['confidence']:.2f}", fill='yellow')
        preview.save(dest.with_suffix('.boxes.png'))
        samples = sorted(data['worker_ms'])
        reports.append({'input': str(image.resolve()), 'regions': len(data['regions']),
                        'worker_mean_ms': sum(samples) / len(samples),
                        'worker_p95_ms': samples[int(0.95 * (len(samples) - 1))],
                        'worker_max_ms': max(samples)})
    (args.out / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))
    print('Boxes are diagnostic allow-list areas, NOT the final pixel masks. Accuracy requires visual review.')


if __name__ == '__main__':
    main()
