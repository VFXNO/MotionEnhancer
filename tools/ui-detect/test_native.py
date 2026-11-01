"""Opt-in native regression/CLI checks; new output folder, no model replacements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--frame0', required=True, type=Path)
    parser.add_argument('--frame1', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    exe = args.executable.resolve()
    models = exe.parent / 'models'

    def protected():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models.glob('ai_flow*')}

    before = protected()
    results = {}

    def run(label, flags, expected=0, model=None):
        env = dict(os.environ)
        if model:
            env['MOTION_ENHANCER_UI_MODEL'] = str(model)
        result = subprocess.run([str(exe), *map(str, flags), '--adapter', 'dgpu'], env=env,
                                capture_output=True, text=True, timeout=120)
        (args.out / (label + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == expected, f'{label}: exit {result.returncode}, expected {expected}'
        results[label] = 'PASS'
        return result.stdout + result.stderr

    for flag in ('--d3d12-self-test', '--projection-self-test', '--ai-blend-mask-self-test',
                 '--ai-refinement-self-test', '--ui-regions-self-test'):
        run(flag[2:], [flag])
    pair = ['--gpu-offline', args.frame0.resolve(), args.frame1.resolve()]
    missing = args.out.resolve() / 'intentionally-missing.onnx'
    log = run('missing-detector-fails-closed', [*pair, args.out / 'must-not-exist.png', '--ai-ui-regions'],
              expected=1, model=missing)
    assert 'UI detector' in log and not (args.out / 'must-not-exist.png').exists()
    log = run('parent-mask-off-skips-missing-detector', [*pair, args.out / 'mask-off.png',
              '--ai-ui-regions', '--no-ui-mask'], model=missing)
    assert 'UI detector:' not in log
    log = run('real-detector-native-flow-integration', [*pair, args.out / 'gated.png', '--ai-ui-regions'],
              model=args.model.resolve())
    assert 'AI UI regions:' in log
    assert protected() == before, 'Protected flow model files changed'
    (args.out / 'summary.json').write_text(json.dumps({'results': results, 'flow_hashes_before_after': before}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
