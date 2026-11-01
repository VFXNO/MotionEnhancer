"""Check repeated icon detections on one supplied local image; no screen capture.

Same-input stability is NOT temporal/game-HUD accuracy. The test itself does not
change runtime thresholds, models or defaults; current-proposal continuity runs
normally. No screen capture, training, model activation or stale-box reuse.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=24)
    args = parser.parse_args()
    if not 4 <= args.runs <= 500:
        parser.error('--runs must be 4..500')
    args.out.mkdir(parents=True, exist_ok=False)
    exe = args.executable.resolve()
    image = args.image.resolve()
    models = exe.parent / 'models'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {p.name: sha(p) for p in models.iterdir() if p.is_file()}
    input_sha = sha(image)
    reference, results = {}, []
    # Reverse backend order in the second fresh-process round as well as
    # repeating within each detector instance. Diagnostics are not timed runs.
    for round_id, choices in enumerate((('icons', 'matrix'), ('matrix', 'icons'))):
        for choice in choices:
            label = f'round{round_id}-{choice}'
            output = (args.out / (label + '.json')).resolve()
            env = {k: v for k, v in os.environ.items() if not k.startswith('MOTION_ENHANCER_UI_')}
            env.update(MOTION_ENHANCER_UI_TEST_CHECKSUM='1', MOTION_ENHANCER_UI_TEST_RUNS=str(args.runs))
            result = subprocess.run([str(exe), '--ui-detect-test', str(image), str(output),
                                     '--ui-detector', choice, '--adapter', 'dgpu'], env=env,
                                    capture_output=True, timeout=120)
            log = (result.stdout + result.stderr).decode('utf-8', errors='replace')
            (args.out / (label + '.log')).write_text(log, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'{label} failed; see its log')
            data = json.loads(output.read_text())
            checksums = data['diagnostic_checksums']
            if len(checksums) != args.runs or len(set(checksums)) != 1 or checksums[0] == '0':
                raise AssertionError(f'{label}: identical input produced inconsistent raw proposals')
            if data['detector'] != 'OmniParser V1 640x640' or data['cpu_fallback'] is not False:
                raise AssertionError('Only the existing GPU icon detector is allowed')
            if choice == 'matrix' and (not data['verified_wmma_dispatches'] or
                    data['verified_wmma_dispatches'] != data['cooperative_matrix_dispatches']):
                raise AssertionError('Expected ISA-verified matrix dispatches')
            evidence = {'checksum': checksums[0], 'regions': data['regions'], 'backend': data['backend']}
            history = data['region_history']
            if len(history) != args.runs or any(s['regions'] != data['regions'] for s in history):
                raise AssertionError(f'{label}: identical input produced inconsistent region history')
            if choice in reference and reference[choice] != evidence:
                raise AssertionError(f'{choice}: fresh detector instances produced different boxes/output')
            reference[choice] = evidence
            results.append({'round': round_id, 'choice': choice, 'calls': len(checksums),
                            'regions': len(data['regions']), 'raw_checksum': checksums[0]})
    if before != {p.name: sha(p) for p in models.iterdir() if p.is_file()} or input_sha != sha(image):
        raise AssertionError('Input or runtime models changed')
    report = {'same_input_consistent_within_and_across_instances': True, 'results': results,
              'model_sha256': before, 'image_sha256': input_sha, 'runtime_models_unchanged': True,
              'temporal_accuracy_established': False,
              'note': 'Same-input determinism only; not a guarantee against misses, false positives or moving-scene flicker.'}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
