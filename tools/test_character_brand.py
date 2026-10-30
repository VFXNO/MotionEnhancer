"""Check baked character artwork and optionally the real, off-screen Windows GUI.

python tools/test_character_brand.py
python tools/test_character_brand.py --exe path/to/motion_enhancer.exe --artifacts folder
Uses NumPy and Pillow; native checks never capture the desktop or start capture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "packaging/assets"


class CharacterAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((ASSETS / "character-atlas.json").read_text())
        cls.atlas = np.asarray(Image.open(ASSETS / "character-atlas.png").convert("RGBA"))

    def frames(self):
        m = self.metadata
        for index in range(m["frames"]):
            x = index % m["columns"] * (m["width"] + 2 * m["padding"]) + m["padding"]
            y = index // m["columns"] * (m["height"] + 2 * m["padding"]) + m["padding"]
            yield self.atlas[y:y + m["height"], x:x + m["width"]]

    def test_metadata_and_generated_header(self):
        m = self.metadata
        self.assertEqual(m["fps"], 24)
        self.assertEqual(m["fps"], m["source_fps"], "Use every native source frame")
        self.assertEqual(hashlib.sha256((ASSETS / "character-atlas.png").read_bytes()).hexdigest(), m["atlas_sha256"])
        header = (ROOT / "include/BrandCharacterAtlas.h").read_text()
        for symbol, key in (("Width", "width"), ("Height", "height"), ("Padding", "padding"),
                            ("Columns", "columns"), ("Frames", "frames"), ("Fps", "fps")):
            self.assertRegex(header, rf"\b{symbol} = {m[key]}\b")
        self.assertEqual(m["frames"] / m["fps"], m["period_seconds"])
        rows = (m["frames"] + m["columns"] - 1) // m["columns"]
        self.assertEqual(self.atlas.shape, (rows * (m["height"] + 2 * m["padding"]),
                                          m["columns"] * (m["width"] + 2 * m["padding"]), 4))

    def test_transparent_edges_and_padding(self):
        for frame in self.frames():
            self.assertFalse(frame[[0, -1], :, 3].any())
            self.assertFalse(frame[:, [0, -1], 3].any())
            self.assertEqual(int(frame[..., 3].max()), 255)
            self.assertGreater(np.count_nonzero(frame[..., 3]), frame.shape[0] * frame.shape[1] * .5)
        self.assertFalse(self.atlas[self.atlas[..., 3] == 0].any(), "Hidden colour on transparent pixels")
        m = self.metadata
        for step, axis in ((m["width"] + 2 * m["padding"], 1), (m["height"] + 2 * m["padding"], 0)):
            for start in range(0, self.atlas.shape[axis], step):
                for pos in (start, start + 1, start + step - 2, start + step - 1):
                    self.assertFalse(np.take(self.atlas, pos, axis=axis).any())

    def test_animation_and_soft_loop_join(self):
        frames = [f[..., :3].astype(np.float32) * (f[..., 3:].astype(np.float32) / 255) for f in self.frames()]
        changes = [float(np.abs(a - b).mean()) for a, b in zip(frames, frames[1:])]
        seam = float(np.abs(frames[-1] - frames[0]).mean())
        self.assertGreater(float(np.mean(changes)), .2, "Animation is stationary")
        self.assertLessEqual(seam, max(changes), "Loop boundary is harsher than ordinary frame motion")
        print(f"Loop join mean change: {seam:.3f}/255; ordinary maximum: {max(changes):.3f}/255", flush=True)


def native_checks(executable, artifacts):
    executable = executable.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    # Check the exact bytes embedded in the executable, not merely file presence.
    import ctypes as c
    from ctypes import wintypes as w
    kernel = c.WinDLL("kernel32", use_last_error=True)
    def api(name, result, *args):
        function = getattr(kernel, name)
        function.restype, function.argtypes = result, args
        return function
    load = api("LoadLibraryExW", w.HMODULE, w.LPCWSTR, w.HANDLE, w.DWORD)
    find = api("FindResourceW", w.HANDLE, w.HMODULE, c.c_void_p, c.c_void_p)
    resource = api("LoadResource", w.HANDLE, w.HMODULE, w.HANDLE)
    lock = api("LockResource", c.c_void_p, w.HANDLE)
    size = api("SizeofResource", w.DWORD, w.HMODULE, w.HANDLE)
    free = api("FreeLibrary", w.BOOL, w.HMODULE)
    module = load(str(executable), None, 0x22)
    assert module, "Cannot open preview executable resources"
    try:
        handle = find(module, 103, 10)
        assert handle, "Missing character resource"
        assert c.string_at(lock(resource(module, handle)), size(module, handle)) == (ASSETS / "character-atlas.png").read_bytes()
    finally:
        free(module)
    print("Embedded character atlas: PASS", flush=True)

    env = {k: v for k, v in os.environ.items() if not k.startswith("MOTION_ENHANCER_")}
    def run(extra):
        result = subprocess.run([str(executable), "--gui"], cwd=executable.parent, env=env | extra,
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return result
    images = {}
    fps = json.loads((ASSETS / "character-atlas.json").read_text())["fps"]
    held_time, next_frame_time = .5 / fps, 1.25 / fps
    for page, seconds in ((0, 0), (0, held_time), (0, next_frame_time), (0, 3), (0, 9), (1, 0), (2, 0)):
        output = artifacts / f"gui-page{page}-t{seconds}.png"
        run({"MOTION_ENHANCER_GUI_SNAPSHOT": str(output), "MOTION_ENHANCER_GUI_PAGE": str(page),
             "MOTION_ENHANCER_GUI_TIME": str(seconds)})
        images[page, seconds] = np.asarray(Image.open(output).convert("RGB"))
    character = np.s_[370:682, :231]
    np.testing.assert_array_equal(images[0, 0][character], images[0, held_time][character])
    assert np.abs(images[0, 0][character].astype(float) - images[0, next_frame_time][character]).mean() > .1
    np.testing.assert_array_equal(images[0, 0][character], images[0, 9][character])
    assert np.abs(images[0, 0][character].astype(float) - images[0, 3][character]).mean() > 1
    for page in (1, 2):
        np.testing.assert_array_equal(images[0, 0][character], images[page, 0][character])
    # Compare the native portrait against the actual colour-managed sprite.
    atlas = Image.open(ASSETS / "character-atlas.png").convert("RGBA")
    tile = atlas.crop((2, 2, 218, 314))
    expected = Image.new("RGB", tile.size, (12, 12, 14))
    expected.paste(tile, (0, 0), tile)
    actual = images[0, 0][370:682, 8:224]
    error = np.abs(actual.astype(float) - np.asarray(expected).astype(float)).mean()
    assert error < 2, f"Rendered character is absent, displaced or incorrectly decoded: {error}"
    next_tile = atlas.crop((222, 2, 438, 314))
    expected_next = Image.new("RGB", next_tile.size, (12, 12, 14))
    expected_next.paste(next_tile, (0, 0), next_tile)
    next_actual = images[0, next_frame_time][370:682, 8:224]
    next_error = np.abs(next_actual.astype(float) - np.asarray(expected_next).astype(float)).mean()
    assert next_error < 2, f"Native {fps} FPS frame selection is incorrect: {next_error}"
    print(f"Three native pages, {fps} FPS cadence, 9s loop, resource decode/compositing (MAE {error:.3f}): PASS", flush=True)
    splash = artifacts / "splash"
    splash.mkdir(exist_ok=True)
    run({"MOTION_ENHANCER_SPLASH_FRAMES": str(splash / "frame")})
    preview = np.asarray(Image.open(splash / "frame_60.png").convert("RGB"))
    assert preview.shape == (360, 580, 3)
    assert (preview[60:230, 390:510] > 100).any(), "Splash cameo is missing"
    result = run({"MOTION_ENHANCER_SPLASH_TEST": "1"})
    assert re.search(r"window visible 1, layered 0", result.stdout), result.stdout
    print("Native splash rendering and off-screen fade/handoff: PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CharacterAssets))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.exe:
        artifacts = args.artifacts or Path(tempfile.mkdtemp(prefix="mefi-character-"))
        native_checks(args.exe, artifacts.resolve())
