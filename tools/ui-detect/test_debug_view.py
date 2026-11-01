"""Opt-in native UI debug-view regressions. Outputs must be a NEW folder.

Uses deterministic synthetic masks only for diagnostic fixtures; no training,
production model files, capture sessions or installed-program changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    exe = args.executable.resolve()

    def protected():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (exe.parent / 'models').iterdir() if p.is_file()}

    before = protected()
    env = dict(os.environ, MOTION_ENHANCER_UI_DEBUG_DUMP=str(args.out.resolve()))
    env.pop('MOTION_ENHANCER_DEBUG_VIEW', None)
    result = subprocess.run([str(exe), '--ui-regions-self-test', '--adapter', 'dgpu'], env=env,
                            capture_output=True, text=True, timeout=120)
    (args.out / 'native.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    assert result.returncode == 0, f'Native UI debug test failed: see {args.out / "native.log"}'
    assert 'UI debug: pair snapshot' in result.stdout
    assert 'stale boxes suppressed: PASS' in result.stdout
    # Verify the saved native GPU outputs as well as the C++ pixel assertions.
    active = Image.open(args.out / 'regions-and-mask.png').convert('RGB')
    assert active.getpixel((128, 120)) == (0, 217, 255)
    assert active.getpixel((170, 120)) == (26, 255, 51)
    assert active.getpixel((300, 120)) == (72, 72, 72)
    headers = []
    for name in ('no-result', 'zero-regions', 'ai-off', 'mask-off', 'no-pair'):
        image = Image.open(args.out / f'{name}.png').convert('RGB')
        assert image.getpixel((128, 120)) == (72, 72, 72), f'Stale box in {name}'
        header = image.crop((6, 36, 198, 52)).tobytes()
        assert header not in headers, f'Indistinguishable status: {name}'
        headers.append(header)
    # Ordinary rendering is unchanged by going through debug view 7. These
    # are local generated images, not additional production datasets.
    frame0, frame1 = args.out / 'a.png', args.out / 'b.png'
    for path, color in ((frame0, (40, 80, 120)), (frame1, (80, 120, 160))):
        Image.new('RGB', (512, 240), color).save(path)
    normal_hashes = []
    for view in (0, 7, 0):
        dest = args.out / f'offline-{view}-{len(normal_hashes)}.png'
        runenv = dict(os.environ, MOTION_ENHANCER_DEBUG_VIEW=str(view))
        runenv.pop('MOTION_ENHANCER_UI_DEBUG_DUMP', None)
        run = subprocess.run([str(exe), '--gpu-offline', str(frame0.resolve()), str(frame1.resolve()),
                              str(dest.resolve()), '--adapter', 'dgpu'], env=runenv,
                             capture_output=True, text=True, timeout=120)
        dest.with_suffix('.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        assert run.returncode == 0, f'Offline view {view} failed'
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if view == 0:
            normal_hashes.append(digest)
        else:
            assert digest != normal_hashes[0], 'Offline debug view did not change'
    assert normal_hashes[0] == normal_hashes[1], 'Normal output changed across debug runs'
    assert protected() == before, 'Runtime models changed'
    summary = {'native_ui_regions_and_debug': 'PASS', 'status_images_distinct': 'PASS',
               'offline_view_7': 'PASS', 'normal_output_repeatable': 'PASS', 'runtime_models_unchanged': True}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
