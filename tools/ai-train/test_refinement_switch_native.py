"""Opt-in paired ONNX/DirectML, runtime-routing and Win32 checkbox tests.

--checkpoint must be a refined checkpoint; --cases a folder of numbered
folders containing a.png, b.png and truth.png (benchmark_refiner.py output).
Output MUST be a new folder. Does not train, install or replace user models.
"""
import argparse
import contextlib
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from mefi import export
from mefi.bundle import copy_export
from mefi.model import Exported, REFINED_ARCH, load_checkpoint


def gui_check(executable, env, metadata):
    """Only close the window we launch, never another user's application."""
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetDlgItem.argtypes, user.GetDlgItem.restype = [wintypes.HWND, ctypes.c_int], wintypes.HWND
    user.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user.SendMessageW.restype = ctypes.c_ssize_t
    user.IsWindowEnabled.argtypes = [wintypes.HWND]
    user.IsWindowEnabled.restype = wintypes.BOOL
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback, wintypes.LPARAM]
    user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    process = subprocess.Popen([executable, '--gui'], env=env)
    window = None
    original = metadata.read_bytes()
    try:
        deadline = time.monotonic() + 20
        while window is None and time.monotonic() < deadline:
            @callback
            def find(handle, _):
                nonlocal window
                pid = wintypes.DWORD()
                user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
                if pid.value == process.pid and user.GetDlgItem(handle, 120):
                    window = handle
                    return False
                return True
            user.EnumWindows(find, 0)
            if process.poll() is not None:
                raise AssertionError('Test GUI exited before controls were ready')
            if window is None:
                time.sleep(0.05)
        assert window is not None, 'Test GUI control creation timed out'
        combo, checkbox = user.GetDlgItem(window, 113), user.GetDlgItem(window, 120)
        send = user.SendMessageW
        assert not user.IsWindowEnabled(checkbox)  # Initial engine is DIS.
        send(combo, 0x14E, 4, 0)  # CB_SETCURSEL: AI pyramid.
        send(window, 0x111, 113 | (1 << 16), combo)  # CBN_SELCHANGE.
        assert user.IsWindowEnabled(checkbox)
        assert send(checkbox, 0xF0, 0, 0) == 0  # BM_GETCHECK, default off.
        send(window, 0x111, 120, checkbox)
        assert send(checkbox, 0xF0, 0, 0) == 1
        send(window, 0x111, 120, checkbox)
        assert send(checkbox, 0xF0, 0, 0) == 0
        # Metadata-only refresh must disable and uncheck an unavailable pair.
        send(window, 0x111, 120, checkbox)
        meta = json.loads(original)
        meta['bundle_id'] = 'intentionally-mismatched-test'
        metadata.write_text(json.dumps(meta), encoding='utf-8')
        send(window, 0x111, 118, 0)  # AI Model -> Refresh.
        assert not user.IsWindowEnabled(checkbox)
        assert send(checkbox, 0xF0, 0, 0) == 0
        metadata.write_bytes(original)
        send(window, 0x111, 118, 0)
        assert user.IsWindowEnabled(checkbox)
        return {'engine_gating': True, 'checked_unchecked': True, 'missing_pair_disabled': True,
                'refresh_reenables_valid_pair': True}
    finally:
        metadata.write_bytes(original)
        if window:
            user.PostMessageW(window, 0x10, 0, 0)  # WM_CLOSE, only our process.
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--executable', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=352)
    args = p.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    executable = str(Path(args.executable).resolve())
    cases = Path(args.cases).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    original_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    ck, net = load_checkpoint(checkpoint)
    assert net.arch == REFINED_ARCH
    model = root / 'export.onnx'
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert export.main([str(checkpoint), '--out', str(model), '--width', str(args.width), '--height', str(args.height)]) == 0
    (root / 'export.log').write_text(output.getvalue(), encoding='utf-8')
    # Exercise the same renamed-bundle copy used by the trainer GUI.
    model = root / 'models/ai_flow.onnx'
    copy_export(root / 'export.onnx', model)
    import onnx
    full, bypass = onnx.load(model), onnx.load(model.with_suffix('.dis.onnx'))
    assert len(bypass.graph.node) < len(full.graph.node)
    assert not any('refiner' in value.name for value in bypass.graph.initializer)
    env = os.environ.copy()
    env['MOTION_ENHANCER_AI_FLOW_MODEL'] = str(model)
    for name in ('MOTION_ENHANCER_BENCH_ITERS', 'MOTION_ENHANCER_BENCH_SERIAL',
                 'MOTION_ENHANCER_AI_TEST_INPUT', 'MOTION_ENHANCER_AI_TEST_OUTPUT'):
        env.pop(name, None)
    report = {'checkpoint_sha256': original_sha, 'torch': str(torch.__version__), 'hip': torch.version.hip,
              'gpu': torch.cuda.get_device_name(0), 'onnx_nodes': {'ai': len(full.graph.node), 'dis': len(bypass.graph.node)},
              'routing': [], 'numerical': [], 'native_quality': []}

    def run(name, arguments, expected=0):
        done = subprocess.run([executable, *arguments, '--adapter', 'dgpu'], cwd=root, env=env,
                              capture_output=True, timeout=180)
        log = done.stdout.decode('utf-8', errors='replace') + done.stderr.decode('utf-8', errors='replace')
        (root / f'{name}.log').write_text(log, encoding='utf-8')
        assert (done.returncode == 0) == (expected == 0), log[-3000:]
        return log

    log = run('self-test', ['--ai-refinement-self-test'])
    assert 'AI graph, zero DIS patch passes' in log and 'bypass graph + DIS patch passes' in log
    report['self_test'] = True
    # Native tensor dumps validate each graph, not just ONNX checker/load.
    w, h = args.width, args.height
    images = [cv2.imread(str(cases / '0' / filename)) for filename in ('a.png', 'b.png')]
    frames = np.concatenate([cv2.cvtColor(cv2.resize(im, (w, h)), cv2.COLOR_BGR2RGB)
        .astype(np.float32).transpose(2, 0, 1) / 255 for im in images], axis=0)[None].copy()
    input_file = root / 'frames.f32'
    frames.tofile(input_file)
    for integrated in (False, True):
        prefix = root / ('ai' if integrated else 'dis')
        graph = model if integrated else model.with_suffix('.dis.onnx')
        env['MOTION_ENHANCER_AI_TEST_INPUT'] = str(input_file)
        env['MOTION_ENHANCER_AI_TEST_OUTPUT'] = str(prefix)
        log = run(prefix.name + '-numerical', ['--ai-flow-test', str(graph)])
        assert 'output finite and mask range: PASS' in log
        reference_net = net if integrated else net.without_refiner()
        with torch.inference_mode():
            reference = Exported(reference_net.cuda().eval())(torch.from_numpy(frames).cuda())
            values = [x.cpu().numpy() for x in reference]
        actual = [np.fromfile(str(prefix) + f'.{name}.f32', np.float32).reshape(1, channels, h, w)
                  for name, channels in (('flow', 4), ('mask', 1))]
        errors = [np.abs(a - b) for a, b in zip(values, actual)]
        row = {'mode': prefix.name, 'flow_mae': float(errors[0].mean()), 'flow_p99': float(np.percentile(errors[0], 99)),
               'mask_mae': float(errors[1].mean())}
        assert row['flow_mae'] <= 0.1 and row['flow_p99'] <= 0.5 and row['mask_mae'] <= 0.005, row
        report['numerical'].append(row)
        reference_net.cpu()
        del reference
        torch.cuda.empty_cache()
    env.pop('MOTION_ENHANCER_AI_TEST_INPUT')
    env.pop('MOTION_ENHANCER_AI_TEST_OUTPUT')
    metadata = model.with_suffix('.json')
    report['gui'] = gui_check(executable, env, metadata)
    for integrated in (False, True):
        for mask in (False, True):
            mode = f'{"ai" if integrated else "dis"}-mask-{int(mask)}'
            for folder in sorted(cases.iterdir()):
                if not folder.is_dir():
                    continue
                result = root / f'{mode}-{folder.name}.png'
                command = ['--gpu-offline', str(folder / 'a.png'), str(folder / 'b.png'), str(result),
                           '--flow-engine', 'ai', '--no-ui-mask']
                if integrated:
                    command.append('--ai-refinement')
                if mask:
                    command.append('--ai-blend-mask')
                log = run(result.stem, command)
                assert 'Flow estimator: AI pyramid flow (ONNX)' in log
                if integrated:
                    assert 'AI refinement selected: integrated AI; DIS patch passes: 0' in log
                else:
                    assert 'AI refinement selected: DIS;' in log and 'DIS patch passes: 0' not in log
                truth = cv2.imread(str(folder / 'truth.png')).astype(np.float64)
                prediction = cv2.imread(str(result)).astype(np.float64)
                assert prediction.shape == truth.shape and np.isfinite(prediction).all()
                psnr = -10 * np.log10(max(float((((prediction - truth) / 255) ** 2).mean()), 1e-10))
                report['native_quality'].append({'mode': mode, 'case': folder.name, 'psnr': float(psnr)})
    # Both modes reject a stale/missing companion, rather than secretly using
    # the integrated graph when unchecked or pure DIS when checked.
    companion_metadata = model.with_suffix('.dis.json')
    before = companion_metadata.read_bytes()
    wrong = json.loads(before)
    wrong['bundle_id'] = 'test-mismatched'
    companion_metadata.write_text(json.dumps(wrong))
    folder = cases / '0'
    command = ['--gpu-offline', str(folder / 'a.png'), str(folder / 'b.png'), str(root / 'invalid.png'), '--flow-engine', 'ai']
    try:
        for integrated in (False, True):
            log = run('reject-mixed-' + str(integrated), command + (['--ai-refinement'] if integrated else []), expected=1)
            assert 'pair is missing or mismatched' in log
            report['routing'].append({'mixed_pair_rejected': True, 'integrated': integrated})
    finally:
        companion_metadata.write_bytes(before)
    companion_graph = model.with_suffix('.dis.onnx')
    hidden = model.with_suffix('.dis.onnx.test-hidden')
    companion_graph.rename(hidden)
    try:
        for integrated in (False, True):
            log = run('reject-missing-' + str(integrated), command + (['--ai-refinement'] if integrated else []), expected=1)
            assert 'pair is missing or mismatched' in log
            report['routing'].append({'missing_pair_rejected': True, 'integrated': integrated})
    finally:
        hidden.rename(companion_graph)
    # Explicit legacy export still works unchecked; checking cannot invent
    # a trained refiner or silently switch the whole estimator to DIS.
    legacy_checkpoint = root / 'legacy.pt'
    legacy_net = net.without_refiner()
    torch.save({'arch': legacy_net.arch, 'preset': legacy_net.preset, 'model': legacy_net.state_dict()}, legacy_checkpoint)
    legacy_model = root / 'legacy.onnx'
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert export.main([str(legacy_checkpoint), '--out', str(legacy_model), '--width', str(w), '--height', str(h)]) == 0
    (root / 'legacy-export.log').write_text(output.getvalue(), encoding='utf-8')
    env['MOTION_ENHANCER_AI_FLOW_MODEL'] = str(legacy_model)
    log = run('legacy-dis', command)
    assert 'AI refinement selected: DIS;' in log and 'Flow estimator: AI pyramid flow (ONNX)' in log
    log = run('legacy-ai-rejected', command + ['--ai-refinement'], expected=1)
    assert 'has no integrated refiner' in log
    report['routing'].append({'legacy_dis_passed': True, 'legacy_ai_rejected': True})
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_sha
    report['checkpoint_unchanged'] = True
    (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'self_test_pairs': 40, 'numerical': report['numerical'],
                      'gui': report['gui'], 'native_quality_cases': len(report['native_quality']), 'out': str(root)}), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
