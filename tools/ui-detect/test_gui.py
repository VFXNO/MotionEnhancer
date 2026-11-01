"""Check only an owned test GUI, without capture or changing any model files."""
import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.SetProcessDPIAware()
    user.GetDlgItem.argtypes = [wintypes.HWND, ctypes.c_int]
    user.GetDlgItem.restype = wintypes.HWND
    user.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user.SendMessageW.restype = ctypes.c_ssize_t
    user.IsWindowEnabled.argtypes = [wintypes.HWND]
    user.IsWindowEnabled.restype = wintypes.BOOL
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.IsWindowVisible.restype = wintypes.BOOL
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback, wintypes.LPARAM]
    process = subprocess.Popen([str(args.executable.resolve()), '--gui'])
    window = None
    try:
        deadline = time.monotonic() + 20
        while window is None and time.monotonic() < deadline:
            @callback
            def find(handle, _):
                nonlocal window
                pid = wintypes.DWORD()
                user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
                if pid.value == process.pid and user.IsWindowVisible(handle) and user.GetDlgItem(handle, 121):
                    window = handle
                    return False
                return True
            user.EnumWindows(find, 0)
            if process.poll() is not None:
                raise AssertionError('Owned GUI exited before controls were ready')
            if window is None:
                time.sleep(0.05)
        assert window, 'Owned GUI creation timed out'
        roi, parent, refinement, combo = [user.GetDlgItem(window, i) for i in (121, 122, 120, 113)]
        detector = user.GetDlgItem(window, 123)
        send = user.SendMessageW
        assert send(roi, 0xF0, 0, 0) == 0, 'AI UI areas must default off'
        assert send(parent, 0xF0, 0, 0) == 1
        assert user.IsWindowEnabled(roi), 'Must work with initial DIS engine'
        assert detector and send(detector, 0x146, 0, 0) == 2, 'Only DirectML/matrix icons may remain'
        for index in range(2):
            label = ctypes.create_unicode_buffer(128)
            send(detector, 0x148, index, ctypes.addressof(label))
            assert 'icons' in label.value.lower() and 'text' not in label.value.lower()
        assert send(detector, 0x147, 0, 0) == 0, 'DirectML icons must remain default'
        assert not user.IsWindowEnabled(detector), 'Default-off detector must disable selector'
        send(window, 0x111, 121, roi)
        assert send(roi, 0xF0, 0, 0) == 1
        assert user.IsWindowEnabled(detector)
        for selection in (1, 0):
            send(detector, 0x14E, selection, 0)
            send(window, 0x111, 123 | (1 << 16), detector)
            assert send(detector, 0x147, 0, 0) == selection
            assert send(refinement, 0xF0, 0, 0) == 0
        assert send(refinement, 0xF0, 0, 0) == 0, 'UI detector must not enable AI flow refinement'
        send(window, 0x111, 122, parent)
        assert not user.IsWindowEnabled(roi), 'Parent UI mask disables region detection'
        assert not user.IsWindowEnabled(detector)
        send(window, 0x111, 122, parent)
        assert user.IsWindowEnabled(roi)
        for engine in (4, 3):
            send(combo, 0x14E, engine, 0)
            send(window, 0x111, 113 | (1 << 16), combo)
            assert user.IsWindowEnabled(roi), 'UI detection must be independent of motion engine'
            assert send(roi, 0xF0, 0, 0) == 1
        send(window, 0x111, 121, roi)
        assert send(roi, 0xF0, 0, 0) == 0
        assert not user.IsWindowEnabled(detector)
        a, b = wintypes.RECT(), wintypes.RECT()
        user.GetWindowRect(roi, ctypes.byref(a))
        user.GetWindowRect(refinement, ctypes.byref(b))
        assert a.left >= b.right, 'UI detector checkbox must not overlap refinement control'
        user.GetWindowRect(detector, ctypes.byref(b))
        assert b.left >= a.right, 'Detector selector must not overlap checkbox'
        if args.screenshot:
            # PrintWindow captures ONLY the window launched by this test. It does
            # not grab a desktop rectangle that could contain other applications.
            from PIL import Image
            gdi = ctypes.WinDLL('gdi32', use_last_error=True)
            user.GetWindowDC.argtypes, user.GetWindowDC.restype = [wintypes.HWND], wintypes.HDC
            user.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
            user.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
            gdi.CreateCompatibleDC.argtypes, gdi.CreateCompatibleDC.restype = [wintypes.HDC], wintypes.HDC
            gdi.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
            gdi.CreateCompatibleBitmap.restype = wintypes.HBITMAP
            gdi.SelectObject.argtypes, gdi.SelectObject.restype = [wintypes.HDC, wintypes.HANDLE], wintypes.HANDLE
            gdi.GetBitmapBits.argtypes = [wintypes.HBITMAP, wintypes.LONG, ctypes.c_void_p]
            gdi.DeleteObject.argtypes = [wintypes.HANDLE]
            gdi.DeleteDC.argtypes = [wintypes.HDC]
            rc = wintypes.RECT()
            user.GetWindowRect(window, ctypes.byref(rc))
            w, h = rc.right - rc.left, rc.bottom - rc.top
            dc = user.GetWindowDC(window)
            memory = gdi.CreateCompatibleDC(dc)
            bitmap = gdi.CreateCompatibleBitmap(dc, w, h)
            old = gdi.SelectObject(memory, bitmap)
            try:
                assert user.PrintWindow(window, memory, 2), 'Owned-window PrintWindow failed'
                pixels = ctypes.create_string_buffer(w * h * 4)
                assert gdi.GetBitmapBits(bitmap, len(pixels), pixels) == len(pixels)
                Image.frombuffer('RGB', (w, h), pixels.raw, 'raw', 'BGRX', 0, 1).save(args.screenshot)
            finally:
                gdi.SelectObject(memory, old)
                gdi.DeleteObject(bitmap)
                gdi.DeleteDC(memory)
                user.ReleaseDC(window, dc)
        print('GUI: default-off, icons only (DirectML/matrix), native-mask dependency, AI/DIS independence, refinement unchanged, toggles and layout: PASS')
    finally:
        if window:
            user.PostMessageW(window, 0x10, 0, 0)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__':
    main()
