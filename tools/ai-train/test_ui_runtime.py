"""Opt-in generated ONNX/DirectML + Win32 integration tests, not a training job.

--out must be a new directory. Uses only synthetic weights/images/tensors;
does not replace installed models, use datasets or close existing applications.
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
import subprocess
import time

import numpy as np
import torch

from mefi import export
from mefi.export_ui import RuntimeTemporal, load_ui_checkpoint
from test_ui_export import fixture


def gui_check(executable, env, metadata):
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetDlgItem.argtypes, user.GetDlgItem.restype = [wintypes.HWND, ctypes.c_int], wintypes.HWND
    user.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user.SendMessageW.restype = ctypes.c_ssize_t
    user.IsWindowEnabled.argtypes, user.IsWindowEnabled.restype = [wintypes.HWND], wintypes.BOOL
    user.IsWindowVisible.argtypes, user.IsWindowVisible.restype = [wintypes.HWND], wintypes.BOOL
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback, wintypes.LPARAM]
    user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    process = subprocess.Popen([str(executable), '--gui'], env=env)
    window = None
    saved = metadata.read_bytes()
    try:
        deadline = time.monotonic() + 20
        while window is None and time.monotonic() < deadline:
            @callback
            def find(handle, _):
                nonlocal window
                pid = wintypes.DWORD()
                user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
                if pid.value == process.pid and user.IsWindowVisible(handle) and user.GetDlgItem(handle, 124):
                    window = handle
                    return False
                return True
            user.EnumWindows(find, 0)
            if process.poll() is not None:
                raise AssertionError('Test GUI exited early')
            time.sleep(.05)
        assert window, 'Test GUI did not become ready'
        combo, box, mask = (user.GetDlgItem(window, control) for control in (113, 124, 122))
        send = user.SendMessageW
        assert not user.IsWindowEnabled(box)  # Default engine remains DIS.
        send(combo, 0x14E, 4, 0)
        send(window, 0x111, 113 | (1 << 16), combo)
        assert user.IsWindowEnabled(box)
        assert send(box, 0xF0, 0, 0) == 0  # Default remains off.
        send(window, 0x111, 124, box)
        assert send(box, 0xF0, 0, 0) == 1
        send(window, 0x111, 122, mask)
        assert not user.IsWindowEnabled(box) and send(box, 0xF0, 0, 0) == 0
        send(window, 0x111, 122, mask)
        assert user.IsWindowEnabled(box)
        metadata.write_text('{"format":"invalid-test-format"}', encoding='utf-8')
        send(window, 0x111, 118, 0)
        assert not user.IsWindowEnabled(box)
        metadata.write_bytes(saved)
        send(window, 0x111, 118, 0)
        assert user.IsWindowEnabled(box)
        return {'default_off': True, 'engine_gating': True, 'mask_gating': True, 'refresh': True}
    finally:
        metadata.write_bytes(saved)
        if window:
            user.PostMessageW(window, 0x10, 0, 0)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()  # Only the test process we launched.
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, executable = Path(args.out).resolve(), Path(args.executable).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint = root / 'synthetic-ui.pt'
    ck = fixture()
    torch.save(ck, checkpoint)
    original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = root / 'ai_flow_ui.onnx'
    with contextlib.redirect_stdout(io.StringIO()) as log:
        assert export.main([str(checkpoint), '--out', str(model), '--width', '128', '--height', '64']) == 0
    (root / 'export.log').write_text(log.getvalue(), encoding='utf-8')
    env = os.environ.copy()
    env['MOTION_ENHANCER_AI_UI_MODEL'] = str(model)
    for key in ('MOTION_ENHANCER_AI_TEST_INPUT', 'MOTION_ENHANCER_AI_TEST_OUTPUT'):
        env.pop(key, None)

    def run(name, command, success=True):
        result = subprocess.run([str(executable), *command, '--adapter', 'dgpu'], cwd=root, env=env,
                                capture_output=True, timeout=180)
        text = result.stdout.decode('utf-8', errors='replace') + result.stderr.decode('utf-8', errors='replace')
        (root / (name + '.log')).write_text(text, encoding='utf-8')
        assert (result.returncode == 0) == success, text[-5000:]
        return text

    report = {'synthetic_weights': True, 'numerical': []}
    assert 'native evidence: PASS' in run('state-pipeline', ['--ai-temporal-ui-self-test'])
    frames = torch.rand(1, 6, 64, 128)
    frames.numpy().tofile(root / 'frames.f32')
    env['MOTION_ENHANCER_AI_TEST_INPUT'] = str(root / 'frames.f32')
    for integrated in (False, True):
        graph = model if integrated else model.with_suffix('.dis.onnx')
        prefix = root / ('integrated' if integrated else 'dis')
        env['MOTION_ENHANCER_AI_TEST_OUTPUT'] = str(prefix)
        assert 'mask range: PASS' in run(prefix.name, ['--ai-flow-test', str(graph)])
        net = load_ui_checkpoint(ck)
        if not integrated:
            net.tap.backbone = net.backbone.without_refiner()
        with torch.inference_mode():
            expected = RuntimeTemporal(net)(frames, torch.full((1, 4, 32, 64), .25), torch.ones(1, 1, 1, 1))
        row = {'integrated': integrated}
        for name, reference in zip(('flow', 'mask', 'ui_allow', 'next_state'), expected):
            actual = np.fromfile(str(prefix) + f'.{name}.f32', np.float32).reshape(reference.shape)
            error = np.abs(actual - reference.numpy())
            row[name + '_mae'] = float(error.mean())
            assert np.isfinite(actual).all() and float(error.max()) < (.1 if name == 'flow' else .02), row
        report['numerical'].append(row)
    env.pop('MOTION_ENHANCER_AI_TEST_INPUT')
    env.pop('MOTION_ENHANCER_AI_TEST_OUTPUT')
    metadata = model.with_suffix('.dis.json')
    saved = metadata.read_bytes()
    wrong = json.loads(saved)
    wrong['bundle_id'] = 'generated-mismatch'
    metadata.write_text(json.dumps(wrong), encoding='utf-8')
    try:
        assert 'missing or mismatched' in run('mixed-pair-rejected', ['--ai-temporal-ui-self-test'], False)
    finally:
        metadata.write_bytes(saved)
    report['gui'] = gui_check(executable, env, model.with_suffix('.json'))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original
    report['checkpoint_unchanged'] = True
    report['passed'] = True
    (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
