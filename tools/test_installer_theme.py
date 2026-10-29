"""Exercise the actual installer UI, audio switch, install and uninstall.

Windows/Python stdlib only. Full tests refuse an existing user installation;
--preview-only checks UI/music without installing or changing registration.
Screenshots capture only the test installer's window.
Usage: python tools/test_installer_theme.py [--screenshots <directory>]
"""
import argparse
from array import array
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import winreg
import wave
import zlib

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "dist" / "MotionEnhancer-1.0.0-Setup-x64.exe"
KEYS = (r"Software\MotionEnhancer",
        r"Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer")
USER = c.WinDLL("user32", use_last_error=True)
GDI = c.WinDLL("gdi32", use_last_error=True)
ENUM = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)


def api(dll, name, result, *args):
    func = getattr(dll, name)
    func.restype, func.argtypes = result, args
    return func


enum_windows = api(USER, "EnumWindows", w.BOOL, ENUM, w.LPARAM)
enum_children = api(USER, "EnumChildWindows", w.BOOL, w.HWND, ENUM, w.LPARAM)
get_pid = api(USER, "GetWindowThreadProcessId", w.DWORD, w.HWND, c.POINTER(w.DWORD))
get_item = api(USER, "GetDlgItem", w.HWND, w.HWND, c.c_int)
get_text = api(USER, "GetWindowTextW", c.c_int, w.HWND, w.LPWSTR, c.c_int)
get_prop = api(USER, "GetPropW", w.HANDLE, w.HWND, w.LPCWSTR)
post = api(USER, "PostMessageW", w.BOOL, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
send = api(USER, "SendMessageW", w.LPARAM, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
get_rect = api(USER, "GetWindowRect", w.BOOL, w.HWND, c.POINTER(w.RECT))
get_dc = api(USER, "GetWindowDC", w.HDC, w.HWND)
release_dc = api(USER, "ReleaseDC", c.c_int, w.HWND, w.HDC)
create_dc = api(GDI, "CreateCompatibleDC", w.HDC, w.HDC)
create_bitmap = api(GDI, "CreateCompatibleBitmap", w.HBITMAP, w.HDC, c.c_int, c.c_int)
select = api(GDI, "SelectObject", w.HANDLE, w.HDC, w.HANDLE)
delete_object = api(GDI, "DeleteObject", w.BOOL, w.HANDLE)
delete_dc = api(GDI, "DeleteDC", w.BOOL, w.HDC)
dibits = api(GDI, "GetDIBits", c.c_int, w.HDC, w.HBITMAP, w.UINT, w.UINT, w.LPVOID, w.LPVOID, w.UINT)
print_window = api(USER, "PrintWindow", w.BOOL, w.HWND, w.HDC, w.UINT)
is_visible = api(USER, "IsWindowVisible", w.BOOL, w.HWND)
is_enabled = api(USER, "IsWindowEnabled", w.BOOL, w.HWND)


def text(window):
    buf = c.create_unicode_buffer(2048)
    # GetWindowText cannot read a foreign Edit control's contents. WM_GETTEXT
    # is one of the standard messages Windows marshals across processes.
    send(window, 0x000D, len(buf), c.addressof(buf))
    return buf.value


def windows(parent=None):
    found = []
    callback = ENUM(lambda window, _: found.append(window) or True)
    if parent: enum_children(parent, callback, 0)
    else: enum_windows(callback, 0)
    return found


def process_id(window):
    pid = w.DWORD()
    get_pid(window, c.byref(pid))
    return pid.value


def wait_for(check, description, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value: return value
        time.sleep(0.05)
    raise AssertionError("Timed out: " + description)


def registered():
    for key in KEYS:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0,
                                winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
                return True
        except FileNotFoundError:
            pass
    return False


def registration_snapshot():
    result = {}
    for key in KEYS:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0,
                                winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as handle:
                values = []
                for index in range(winreg.QueryInfoKey(handle)[1]):
                    values.append(winreg.EnumValue(handle, index))
                result[key] = sorted(values)
        except FileNotFoundError:
            pass
    return result


def screenshot(window, destination):
    rect = w.RECT()
    get_rect(window, c.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    dc = get_dc(window)
    memory = create_dc(dc)
    bitmap = create_bitmap(dc, width, height)
    old = select(memory, bitmap)
    try:
        assert print_window(window, memory, 2), "PrintWindow failed"
        select(memory, old)
        header = c.create_string_buffer(struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0,
                                                    width * height * 4, 0, 0, 0, 0))
        pixels = c.create_string_buffer(width * height * 4)
        assert dibits(memory, bitmap, 0, height, pixels, header, 0) == height
        raw = pixels.raw
        rgb = bytearray(width * height * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = raw[2::4], raw[1::4], raw[0::4]
        # Catch accidental white default wizard surfaces, not just a correct title.
        assert sum(v < 60 for v in rgb[0::3]) / (width * height) > 0.80, "Theme is not predominantly dark"

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        rows = b"".join(b"\0" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        destination.write_bytes(png + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    finally:
        select(memory, old)
        delete_object(bitmap)
        delete_dc(memory)
        release_dc(window, dc)


def root_window(process, music=True):
    def find():
        for window in windows():
            if process_id(window) != process.pid: continue
            if not get_item(window, 1037): continue
            if music and not get_prop(window, "MotionEnhancer.MusicPlaying"): continue
            return window
        if process.poll() is not None: raise AssertionError(f"Installer exited early: {process.returncode}")
    return wait_for(find, "installer window")


def command(window, number):
    button = get_item(window, number)
    assert button and is_enabled(button), f"Disabled/missing button {number}"
    post(window, 0x0111, number, button)  # WM_COMMAND / BN_CLICKED


def page(window, heading, name, screenshots):
    wait_for(lambda: text(get_item(window, 1037)) == heading, name)
    time.sleep(0.15)  # Let native controls finish painting after the page callback.
    screenshot(window, screenshots / (name + ".png"))
    print(name + ": dark UI PASS", flush=True)


def close_cancelled(process, window):
    command(window, 2)
    def confirmation():
        return next((h for h in windows() if process_id(h) == process.pid and h != window and get_item(h, 6)), None)
    modal = wait_for(confirmation, "cancel confirmation")
    post(modal, 0x0111, 6, get_item(modal, 6))
    process.wait(timeout=20)


def main(screenshots, preview_only=False):
    if not preview_only:
        assert not registered(), "Existing installation: use --preview-only to leave it untouched"
    original_registration = registration_snapshot()
    assert SETUP.is_file(), "Build the installer first"
    with wave.open(str(ROOT / "packaging/assets/setup-music.wav"), "rb") as track:
        assert (track.getnchannels(), track.getsampwidth(), track.getframerate()) == (2, 2, 22050)
        assert track.getnframes() == 22050 * 32
        pcm = array("h", track.readframes(track.getnframes()))
        assert 0 < max(map(abs, pcm)) <= round(32767 * 0.161), "Soundtrack peak must remain quiet/unclipped"
    print("Original 32-second stereo soundtrack and quiet peak limit: PASS", flush=True)
    scratch = Path(tempfile.mkdtemp(prefix="installer-theme-", dir=Path(os.environ["LOCALAPPDATA"]) / "Temp" / "opencode"))
    screenshots = screenshots or scratch / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    install = scratch / "installed app"
    # NSIS /D= and _?= consume the unquoted remainder of the command line.
    process = subprocess.Popen(f'"{SETUP}" /NOMUSIC /D={install}')
    root = root_window(process)
    try:
        page(root, "A smoother experience.", "welcome", screenshots)
        assert get_prop(root, "MotionEnhancer.MusicPlaying") == 2
        command(root, 1200)
        wait_for(lambda: get_prop(root, "MotionEnhancer.MusicPlaying") == 1, "music enabled")
        command(root, 1200)
        wait_for(lambda: get_prop(root, "MotionEnhancer.MusicPlaying") == 2, "music muted")
        print("/NOMUSIC and playback/mute controls: PASS", flush=True)
        command(root, 1)
        page(root, "Built on open technology.", "license", screenshots)
        command(root, 1)
        page(root, "Everything you need.", "components", screenshots)
        command(root, 1)
        page(root, "Find its home.", "directory", screenshots)
        current_page = next(h for h in windows(root) if is_visible(h) and get_prop(h, "MotionEnhancer.LayoutDone"))
        destination = text(get_item(current_page, 1019))
        assert destination == str(install), f"Destination field mismatch: {destination!r} != {str(install)!r}"
        assert is_visible(get_item(current_page, 1019)) and is_visible(get_item(current_page, 1001)), "Path and Browse must remain visible"
        command(root, 3)
        page(root, "Everything you need.", "components-back", screenshots)
        command(root, 1)
        page(root, "Find its home.", "directory-again", screenshots)
        if preview_only:
            close_cancelled(process, root)
            process = subprocess.Popen([str(SETUP)])
            root = root_window(process)
            wait_for(lambda: get_prop(root, "MotionEnhancer.MusicPlaying") == 1, "default music")
            page(root, "A smoother experience.", "welcome-music", screenshots)
            close_cancelled(process, root)
            assert registration_snapshot() == original_registration, "Preview must leave installation registration untouched"
            print("All preview pages, music default/toggle/cancel and unchanged registration: PASS", flush=True)
            return
        command(root, 1)
        # Installation uses NSIS's actual extraction progress, not a fake timer.
        wait_for(lambda: text(get_item(root, 1037)) in ("Bringing motion to life.", "You're ready to flow."), "installation")
        if text(get_item(root, 1037)) == "Bringing motion to life.":
            screenshot(root, screenshots / "installing.png")
        page(root, "You're ready to flow.", "finish", screenshots)
        command(root, 1)
        assert process.wait(timeout=20) == 0
        manifest = json.loads((install / "payload-sha256.json").read_text(encoding="utf-8-sig"))
        for entry in manifest:
            assert hashlib.sha256((install / entry["file"]).read_bytes()).hexdigest() == entry["sha256"], entry["file"]
        print(f"Interactive installation: {len(manifest)} payload hashes PASS", flush=True)
        native = subprocess.run([str(install / "motion_enhancer.exe"), "--d3d12-self-test"],
                                cwd=scratch, capture_output=True, text=True, timeout=120)
        assert native.returncode == 0, native.stdout + native.stderr
        # Uninstall must leave real user-created files alone.
        for file in (install / "captures" / "keep.txt", install / "tools/ai-train/datasets/keep.txt"):
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("user data")
        process = subprocess.Popen(f'"{install / "Uninstall.exe"}" _?={install}')
        root = root_window(process, music=False)
        page(root, "Make room for what's next.", "uninstall", screenshots)
        assert not get_prop(root, "MotionEnhancer.MusicPlaying"), "Uninstaller should not start music"
        command(root, 1)
        wait_for(lambda: not (install / "motion_enhancer.exe").exists(), "uninstallation")
        wait_for(lambda: is_enabled(get_item(root, 1)), "uninstaller close button")
        page(root, "Application removed.", "uninstall-finished", screenshots)
        command(root, 1)
        assert process.wait(timeout=20) == 0
        assert (install / "models/ai_flow.onnx").exists()
        assert (install / "captures/keep.txt").read_text() == "user data"
        assert (install / "tools/ai-train/datasets/keep.txt").read_text() == "user data"
        assert not registered()
        print("Uninstall theme and data preservation: PASS", flush=True)
        # Music defaults on, but canceling must release the audio and plugin DLL.
        process = subprocess.Popen([str(SETUP)])
        root = root_window(process)
        wait_for(lambda: get_prop(root, "MotionEnhancer.MusicPlaying") == 1, "default music")
        page(root, "A smoother experience.", "welcome-music", screenshots)
        close_cancelled(process, root)
        assert not registered()
        print("Default soundtrack and cancel exit: PASS", flush=True)
        silent = scratch / "silent app"
        process = subprocess.Popen(f'"{SETUP}" /S /D={silent}')
        deadline = time.monotonic() + 120
        while process.poll() is None:
            assert time.monotonic() < deadline, "Silent install timed out"
            assert not any(process_id(h) == process.pid and get_prop(h, "MotionEnhancer.MusicPlaying")
                           for h in windows()), "Silent installation must not initialize the music/UI"
            time.sleep(0.01)
        assert process.returncode == 0
        assert (silent / "motion_enhancer.exe").exists()
        assert not (silent / "setup-music.wav").exists() and not (silent / "MotionTheme.dll").exists()
        ai = subprocess.run([str(silent / "motion_enhancer.exe"), "--ai-blend-mask-self-test"],
                            cwd=scratch, capture_output=True, text=True, timeout=120)
        assert ai.returncode == 0, ai.stdout + ai.stderr
        subprocess.run(f'"{silent / "Uninstall.exe"}" /S _?={silent}', timeout=60, check=True)
        assert not registered()
        print("Silent install/uninstall, no skin/audio initialization, installed AI model: PASS", flush=True)
        print("Installer-theme verification PASS", flush=True)
    finally:
        if process.poll() is None:
            # Only this explicitly spawned verification process, never a user's app.
            process.terminate()
            process.wait(timeout=15)
        if registered() and not preview_only:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEYS[0], 0,
                                winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                remaining = Path(winreg.QueryValueEx(key, "InstallDir")[0])
            assert remaining in (install, scratch / "silent app"), "Refusing to remove an unrelated installation"
            subprocess.run(f'"{remaining / "Uninstall.exe"}" /S _?={remaining}', timeout=60, check=True)
        if preview_only or not registered(): shutil.rmtree(scratch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--preview-only", action="store_true", help="Never install/uninstall; safe with an existing installation")
    args = parser.parse_args()
    main(args.screenshots, args.preview_only)
