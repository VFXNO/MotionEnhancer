"""Check embedded Explorer icons and off-screen GUI/splash window icons.

Windows/Python stdlib only. Does not install anything or modify shortcuts.
Usage: python tools/test_app_icon.py [path/to/motion_enhancer.exe]
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import os
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
KERNEL = c.WinDLL("kernel32", use_last_error=True)
USER = c.WinDLL("user32", use_last_error=True)
GDI = c.WinDLL("gdi32", use_last_error=True)
SHELL = c.WinDLL("shell32", use_last_error=True)


def api(dll, name, result, *args):
    function = getattr(dll, name)
    function.restype, function.argtypes = result, args
    return function


load_library = api(KERNEL, "LoadLibraryExW", w.HMODULE, w.LPCWSTR, w.HANDLE, w.DWORD)
free_library = api(KERNEL, "FreeLibrary", w.BOOL, w.HMODULE)
find_resource = api(KERNEL, "FindResourceW", w.HANDLE, w.HMODULE, c.c_void_p, c.c_void_p)
load_resource = api(KERNEL, "LoadResource", w.HANDLE, w.HMODULE, w.HANDLE)
lock_resource = api(KERNEL, "LockResource", c.c_void_p, w.HANDLE)
resource_size = api(KERNEL, "SizeofResource", w.DWORD, w.HMODULE, w.HANDLE)
extract_icon = api(SHELL, "ExtractIconExW", w.UINT, w.LPCWSTR, c.c_int,
                   c.POINTER(w.HICON), c.POINTER(w.HICON), w.UINT)
destroy_icon = api(USER, "DestroyIcon", w.BOOL, w.HICON)
load_icon = api(USER, "LoadIconW", w.HICON, w.HINSTANCE, c.c_void_p)
load_image = api(USER, "LoadImageW", w.HANDLE, w.HINSTANCE, w.LPCWSTR, w.UINT, c.c_int, c.c_int, w.UINT)
get_metric = api(USER, "GetSystemMetrics", c.c_int, c.c_int)
get_class_icon = api(USER, "GetClassLongPtrW", c.c_size_t, w.HWND, c.c_int)
get_class_name = api(USER, "GetClassNameW", c.c_int, w.HWND, w.LPWSTR, c.c_int)
get_pid = api(USER, "GetWindowThreadProcessId", w.DWORD, w.HWND, c.POINTER(w.DWORD))
delete_object = api(GDI, "DeleteObject", w.BOOL, w.HANDLE)
bitmap_bits = api(GDI, "GetBitmapBits", w.LONG, w.HBITMAP, w.LONG, w.LPVOID)


class IconInfo(c.Structure):
    _fields_ = [("icon", w.BOOL), ("x", w.DWORD), ("y", w.DWORD),
                ("mask", w.HBITMAP), ("color", w.HBITMAP)]


get_icon_info = api(USER, "GetIconInfo", w.BOOL, w.HICON, c.POINTER(IconInfo))


def pixels(icon, size):
    info = IconInfo()
    assert icon and get_icon_info(icon, c.byref(info)), "Missing/unreadable icon"
    try:
        assert info.color, "Expected a full-color icon"
        data = c.create_string_buffer(size[0] * size[1] * 4)
        assert bitmap_bits(info.color, len(data), data) == len(data)
        return data.raw
    finally:
        if info.mask: delete_object(info.mask)
        if info.color: delete_object(info.color)


def verify(executable):
    source = (ROOT / "packaging/assets/installer.ico").read_bytes()
    count = struct.unpack_from("<HHH", source)[2]
    module = load_library(str(executable), None, 0x22)  # Resource-only; never executes the EXE.
    assert module, "Cannot open executable resources"
    try:
        def resource(kind, number):
            handle = find_resource(module, number, kind)
            assert handle, f"Missing resource type {kind}, ID {number}"
            return c.string_at(lock_resource(load_resource(module, handle)), resource_size(module, handle))
        group = resource(14, 101)  # RT_GROUP_ICON / IDI_MOTION_ENHANCER
        assert struct.unpack_from("<HHH", group) == (0, 1, count)
        sizes = []
        for index in range(count):
            expected = struct.unpack_from("<BBBBHHII", source, 6 + index * 16)
            actual = struct.unpack_from("<BBBBHHIH", group, 6 + index * 14)
            assert actual[:7] == expected[:7]
            assert resource(3, actual[7]) == source[expected[7]:expected[7] + expected[6]]
            sizes.append(expected[0] or 256)
        assert sizes == [16, 24, 32, 48, 64, 128, 256]
        print("Embedded icon: all seven sizes match the app/setup emblem", flush=True)
    finally:
        free_library(module)

    expected_icons = []
    try:
        for small in (False, True):
            size = (get_metric(49 if small else 11), get_metric(50 if small else 12))
            icon = load_image(None, str(ROOT / "packaging/assets/installer.ico"), 1, *size, 0x10)
            expected_icons.append((icon, size, pixels(icon, size)))
        large, small = w.HICON(), w.HICON()
        assert extract_icon(str(executable), 0, c.byref(large), c.byref(small), 1) > 0
        try:
            for icon, (_, size, expected) in zip((large, small), expected_icons):
                assert pixels(icon, size) == expected, "Explorer-extracted icon differs from the emblem"
        finally:
            if large: destroy_icon(large)
            if small: destroy_icon(small)
        print("Explorer/shortcut large and small icon extraction: PASS", flush=True)

        env = {k: v for k, v in os.environ.items() if not k.startswith("MOTION_ENHANCER_")}
        env["MOTION_ENHANCER_SPLASH_TEST"] = "1"  # Existing off-screen, automatic-exit hook.
        process = subprocess.Popen([str(executable), "--gui"], cwd=executable.parent,
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        seen = set()
        try:
            callback_type = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
            enum_windows = api(USER, "EnumWindows", w.BOOL, callback_type, w.LPARAM)
            deadline = time.monotonic() + 20
            while process.poll() is None and time.monotonic() < deadline:
                found = []
                callback = callback_type(lambda window, _: found.append(window) or True)
                enum_windows(callback, 0)
                for window in found:
                    pid = w.DWORD()
                    get_pid(window, c.byref(pid))
                    if pid.value != process.pid: continue
                    name = c.create_unicode_buffer(128)
                    get_class_name(window, name, len(name))
                    if name.value not in ("MotionEnhancerGuiClass", "MotionEnhancerSplashClass"): continue
                    if name.value in seen: continue
                    for field, (_, size, expected) in zip((-14, -34), expected_icons):
                        icon = get_class_icon(window, field)  # GCLP_HICON / GCLP_HICONSM
                        assert icon != load_icon(None, 32512), "Generic Windows icon is still in use"
                        assert pixels(icon, size) == expected, f"Wrong icon on {name.value}"
                    seen.add(name.value)
                time.sleep(0.02)
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr
            assert seen == {"MotionEnhancerGuiClass", "MotionEnhancerSplashClass"}, seen
            print("Off-screen GUI/splash title-bar and taskbar class icons: PASS", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()  # Only the verification process this script launched.
                process.wait(timeout=10)
    finally:
        for icon, _, _ in expected_icons:
            if icon: destroy_icon(icon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", nargs="?", type=Path, default=ROOT / "build-project/Release/motion_enhancer.exe")
    verify(parser.parse_args().executable.resolve())
