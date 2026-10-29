"""Generate original installer music and a matching emblem icon (stdlib only).

Run with Python 3: python tools/make_installer_assets.py
The generated assets are checked in; normal packaging does not need Python.
"""
from array import array
import math
from pathlib import Path
import struct
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "packaging" / "assets"


def icon():
    images = []
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixels = bytearray()
        # Supersampling keeps the orbit/rays readable at small icon sizes.
        for y in reversed(range(size)):
            for x in range(size):
                total = 0.0
                for yy in (0.125, 0.375, 0.625, 0.875):
                    for xx in (0.125, 0.375, 0.625, 0.875):
                        dx, dy = (x + xx) / size - 0.5, (y + yy) / size - 0.5
                        r = math.hypot(dx, dy)
                        value = 0.0
                        if abs(r - 0.31) < 0.009: value = 0.85
                        if abs(r - 0.39) < 0.005: value = max(value, 0.4)
                        if math.hypot(dx - 0.065, dy) < 0.065: value = 1.0
                        for line_y, line_x in ((-0.08, -0.19), (0, -0.23), (0.08, -0.19)):
                            if line_x < dx < 0.03 and abs(dy - line_y) < 0.008: value = 0.85
                        for k in range(6):
                            a = k * math.pi / 3
                            if math.hypot(dx - 0.43 * math.cos(a), dy - 0.43 * math.sin(a)) < 0.02:
                                value = 0.9
                        total += value
                v = int(12 + 230 * total / 16)
                pixels.extend((v + 2, v, v, 255))  # BGRA, on the app's near-black surface
        mask = bytes(((size + 31) // 32) * 4 * size)
        header = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0,
                             len(pixels) + len(mask), 0, 0, 0, 0)
        images.append((size, header + pixels + mask))
    offset = 6 + len(images) * 16
    directory, contents = bytearray(), bytearray()
    for size, data in images:
        directory.extend(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset))
        contents.extend(data)
        offset += len(data)
    (ASSETS / "installer.ico").write_bytes(struct.pack("<HHH", 0, 1, len(images)) + directory + contents)


def music():
    # Original 32-second ambient cue: warm Dm9 / Bbmaj7 / Fmaj9 / Cadd9 pads
    # with a sparse bell motif. No recordings, samples or external tracks.
    rate, seconds = 22050, 32
    length = rate * seconds
    left, right = array("f", [0.0]) * length, array("f", [0.0]) * length
    chords = ((50, 57, 60, 64, 69), (46, 53, 57, 62, 65),
              (41, 53, 57, 60, 67), (48, 55, 62, 64, 67))

    def frequency(note):
        # Integral cycles per loop eliminate a waveform discontinuity.
        return round(440 * 2 ** ((note - 69) / 12) * seconds) / seconds

    for chord_index, chord in enumerate(chords):
        for voice, note in enumerate(chord):
            f = frequency(note)
            phase = voice * 0.73
            pan = 0.25 + voice * 0.125
            for j in range(rate * 10):
                t = j / rate
                env = min(1, t / 2) * min(1, (10 - t) / 2)
                env = 0.5 - 0.5 * math.cos(math.pi * env)
                value = env * (math.sin(2 * math.pi * f * t + phase)
                               + 0.18 * math.sin(2 * math.pi * 2 * f * t + phase)) * 0.12
                index = (chord_index * 8 * rate + j) % length
                left[index] += value * math.sqrt(1 - pan)
                right[index] += value * math.sqrt(pan)
    motif = (74, 69, 76, 72, 74, 77, 72, 69, 72, 79, 76, 72, 74, 76, 67, 69)
    for beat, note in enumerate(motif):
        f = frequency(note)
        for j in range(rate * 3):
            t = j / rate
            env = (1 - math.exp(-t * 35)) * math.exp(-t * 2) * max(0, 1 - (t / 3) ** 4)
            value = 0.12 * env * (math.sin(2 * math.pi * f * t) + 0.12 * math.sin(2 * math.pi * 3 * f * t))
            index = (beat * 2 * rate + j) % length
            left[index] += value * (0.65 if beat % 2 else 0.35)
            right[index] += value * (0.35 if beat % 2 else 0.65)
            # Circular stereo delay also joins seamlessly at the loop boundary.
            delayed = (index + int(rate * 0.75)) % length
            left[delayed] += value * 0.12
            right[delayed] += value * 0.18
    peak = max(max(map(abs, left)), max(map(abs, right)))
    gain = 0.16 * 32767 / peak  # Quiet by construction; never changes system volume.
    pcm = array("h")
    for l, r in zip(left, right):
        pcm.append(round(l * gain))
        pcm.append(round(r * gain))
    if sys.byteorder != "little": pcm.byteswap()
    with wave.open(str(ASSETS / "setup-music.wav"), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    icon()
    music()
    print(f"Generated original installer icon and ambient music in {ASSETS}")
