"""Generate original installer music and the shared app/setup emblem (stdlib only).

Run with Python 3: python tools/make_installer_assets.py
The generated assets are checked in; normal packaging does not need Python.
"""
from array import array
import argparse
import math
from pathlib import Path
import struct
import sys
import wave
import xml.etree.ElementTree as ET
import zlib

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "packaging" / "assets"


def logo_shapes(size=256):
    """Read the deliberately small SVG subset used by our single master logo."""
    shapes = []
    svg = ET.parse(ASSETS / "logo.svg").getroot()
    if svg.attrib["viewBox"] != "0 0 100 100":
        raise ValueError("Logo must use its 100-unit viewBox")
    for element in svg:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "title" or size < float(element.attrib.get("data-min-size", 0)): continue
        fill = element.attrib.get("fill", "none")
        stroke = element.attrib.get("stroke", "none")
        width = max(float(element.attrib.get("stroke-width", 0)),
                    float(element.attrib.get("data-min-stroke-px", 0)) * 100 / size)
        if tag == "circle":
            x, y, r = (float(element.attrib[k]) for k in ("cx", "cy", "r"))
            r = max(r, float(element.attrib.get("data-min-radius-px", 0)) * 100 / size)
            if fill != "none":
                shapes.append((tag, (x, y, r), tuple(bytes.fromhex(fill[1:])), (x - r, y - r, x + r, y + r)))
            if stroke != "none":
                radius = r + width / 2
                shapes.append(("ring", (x, y, r, width / 2), tuple(bytes.fromhex(stroke[1:])),
                               (x - radius, y - radius, x + radius, y + radius)))
        elif tag == "polygon":
            points = [tuple(map(float, p.split(","))) for p in element.attrib["points"].split()]
            xs, ys = zip(*points)
            shapes.append((tag, points, tuple(bytes.fromhex(fill[1:])), (min(xs), min(ys), max(xs), max(ys))))
        elif tag in ("line", "polyline"):
            if tag == "line":
                points = [(float(element.attrib[x]), float(element.attrib[y])) for x, y in (("x1", "y1"), ("x2", "y2"))]
            else:
                points = [tuple(map(float, p.split(","))) for p in element.attrib["points"].split()]
            color, radius = tuple(bytes.fromhex(stroke[1:])), width / 2
            for (ax, ay), (bx, by) in zip(points, points[1:]):
                shapes.append(("stroke", (ax, ay, bx, by, radius), color,
                               (min(ax, bx) - radius, min(ay, by) - radius, max(ax, bx) + radius, max(ay, by) + radius)))
        else:
            raise ValueError("Unsupported logo element: " + tag)
    return shapes


def raster_logo(size, shapes):
    """Straight-alpha RGBA, supersampled without ever baking in a background."""
    rgba = bytearray()
    for y in range(size):
        for x in range(size):
            red = green = blue = covered = 0
            for yy in (0.125, 0.375, 0.625, 0.875):
                for xx in (0.125, 0.375, 0.625, 0.875):
                    sx, sy = (x + xx) * 100 / size, (y + yy) * 100 / size
                    ink = None
                    for tag, data, color, (x0, y0, x1, y1) in shapes:
                        if not (x0 <= sx <= x1 and y0 <= sy <= y1): continue
                        if tag == "circle":
                            cx, cy, r = data
                            inside = (sx - cx) ** 2 + (sy - cy) ** 2 <= r * r
                        elif tag == "ring":
                            cx, cy, r, half_width = data
                            inside = abs(math.hypot(sx - cx, sy - cy) - r) <= half_width
                        elif tag == "stroke":
                            ax, ay, bx, by, radius = data
                            dx, dy = bx - ax, by - ay
                            along = max(0, min(1, ((sx - ax) * dx + (sy - ay) * dy) / (dx * dx + dy * dy)))
                            inside = (sx - ax - along * dx) ** 2 + (sy - ay - along * dy) ** 2 <= radius * radius
                        else:
                            inside = False
                            ax, ay = data[-1]
                            for bx, by in data:
                                if (ay > sy) != (by > sy) and sx < (bx - ax) * (sy - ay) / (by - ay) + ax:
                                    inside = not inside
                                ax, ay = bx, by
                        if inside: ink = color
                    if ink is not None:
                        covered += 1
                        red += ink[0]
                        green += ink[1]
                        blue += ink[2]
            rgba.extend((round(red / covered), round(green / covered), round(blue / covered),
                         round(255 * covered / 16)) if covered else (0, 0, 0, 0))
    return rgba


def write_png(path, size, rgba):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + rgba[y * size * 4:(y + 1) * size * 4] for y in range(size))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def icon():
    images = []
    for size in (16, 24, 32, 48, 64, 128, 256):
        rgba = raster_logo(size, logo_shapes(size))
        if size == 256: write_png(ASSETS / "logo.png", size, rgba)
        pixels = bytearray()
        stride = ((size + 31) // 32) * 4
        mask = bytearray(stride * size)
        for row, y in enumerate(reversed(range(size))):
            for x in range(size):
                offset = (y * size + x) * 4
                r, g, b, a = rgba[offset:offset + 4]
                pixels.extend((b, g, r, a))
                # The legacy AND mask agrees with zero alpha, including corners.
                if not a: mask[row * stride + x // 8] |= 0x80 >> (x % 8)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icons-only", action="store_true", help="Regenerate logo PNG/ICO without touching the music")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    icon()
    if not args.icons_only: music()
    print(f"Generated shared transparent logo/icon{' and ambient music' if not args.icons_only else ''} in {ASSETS}")
