# Usage: python docs/tools/make_readme_art.py docs/assets
# The GUI screenshots and GIFs come from the app itself: MOTION_ENHANCER_GUI_SNAPSHOT,
# MOTION_ENHANCER_GUI_TIME and MOTION_ENHANCER_SPLASH_FRAMES (see README, Graphical Interface).
# Renders the README's hero banner and section dividers in the app's style:
# near-black, white light, and the shared app/setup orbital emblem.
# Drawn at 3x and scaled down for smooth edges.
import random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = sys.argv[1]
S = 3
W, H = 1280, 420
FONTS = r'C:\Windows\Fonts'


def font(name, size):
    return ImageFont.truetype(f'{FONTS}\\{name}', size * S)


def rgba(a, v=255):
    return (v, v, v, int(max(0, min(255, a))))


def bezier(p0, p1, p2, p3, n=200):
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
        pts.append((x * S, y * S))
    return pts


class Over:
    """ImageDraw on a fresh transparent layer, composited (blended) onto the
    image when the block ends - plain ImageDraw replaces pixels."""
    def __init__(self, img):
        self.img = img
        self.layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    def __enter__(self):
        return ImageDraw.Draw(self.layer)
    def __exit__(self, *args):
        self.img.alpha_composite(self.layer)


def emblem(layer, cx, cy, k):
    """Use the master artwork rather than maintaining another logo drawing."""
    source = Path(__file__).resolve().parents[2] / 'packaging/assets/logo.png'
    size = round(104 * k * S)
    with Image.open(source) as logo:
        badge = logo.convert('RGBA').resize((size, size), Image.Resampling.LANCZOS)
    layer.alpha_composite(badge, (round(cx * S - size / 2), round(cy * S - size / 2)))


def logo_preview():
    """Actual ICO frames on light, dark and checkerboard surfaces."""
    image = Image.new('RGB', (600, 356), (8, 8, 10))
    d = ImageDraw.Draw(image)
    f = ImageFont.truetype(f'{FONTS}\\segoeui.ttf', 14)
    source = Path(__file__).resolve().parents[2] / 'packaging/assets/installer.ico'
    with Image.open(source) as ico:
        for panel, (background, label) in enumerate((((12, 12, 14), 'Dark'),
                ((243, 243, 246), 'Light'), ((38, 38, 44), 'Transparency'))):
            x = panel * 200
            d.rectangle((x, 0, x + 199, 225), fill=background)
            if panel == 2:
                for y in range(0, 226, 12):
                    for xx in range(x, x + 200, 12):
                        if ((xx - x) // 12 + y // 12) % 2:
                            d.rectangle((xx, y, min(x + 199, xx + 11), min(225, y + 11)), fill=(55, 55, 63))
            frame = ico.ico.getimage((128, 128)).convert('RGBA')
            image.paste(frame, (x + 36, 36), frame)
            d.text((x + 100, 191), label, font=f, fill=(50, 50, 58) if panel == 1 else (225, 225, 232), anchor='mm')
        d.text((22, 248), 'Native icon sizes: 16 / 24 / 32 / 48 / 64 px', font=f, fill=(175, 175, 186))
        x = 22
        for size in (16, 24, 32, 48, 64):
            frame = ico.ico.getimage((size, size)).convert('RGBA')
            image.paste(frame, (x, 310 - size // 2), frame)
            x += size + 30
    return image


def tracked(d, x, y, text, fnt, track, fill):
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += d.textlength(ch, font=fnt) + track * S
    return x


def width_tracked(d, text, fnt, track):
    return sum(d.textlength(c, font=fnt) for c in text) + track * S * (len(text) - 1)


def banner():
    img = Image.new('RGBA', (W * S, H * S), (8, 8, 10, 255))
    # vertical sheen and a soft light behind the emblem
    light = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(light)
    ld.ellipse([(330 - 260) * S, (210 - 220) * S, (330 + 260) * S, (210 + 220) * S], fill=(255, 255, 255, 26))
    light = light.filter(ImageFilter.GaussianBlur(70 * S))
    img.alpha_composite(light)
    # flowing lines, like the sidebar's
    lines = Image.new('RGBA', img.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(lines)
    for i in range(7):
        o = 22 * i
        dl.line(bezier((-40, 420 - o), (360, 120 - o), (780, 520 - o * 1.5), (1320, 90 + o)),
                fill=rgba(60 - 7 * i), width=S)
    for i in range(4):
        o = 26 * i
        dl.line(bezier((-40, 40 + o), (420, 260 + o), (900, -60 + o), (1320, 300 - o)), fill=rgba(28 - 6 * i),
                width=S)
    img.alpha_composite(lines)
    # stars
    random.seed(7)
    with Over(img) as d:
     for _ in range(90):
        x, y = random.uniform(0, W), random.uniform(0, H)
        r = random.choice((0.6, 0.8, 1.0, 1.3))
        d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=rgba(random.randint(40, 170)))
    # emblem
    emblem(img, 330, 210, 2.3)
    d = ImageDraw.Draw(img)
    # title and tagline
    title = font('seguisb.ttf', 52)
    x0 = 560
    tracked(d, x0 * S, 132 * S, 'MOTION', title, 9, (255, 255, 255, 255))
    tracked(d, x0 * S, 196 * S, 'ENHANCER', title, 9, (255, 255, 255, 255))
    tag = font('segoeui.ttf', 21)
    d.text((x0 * S + 2 * S, 274 * S), 'Real-time frame generation for any window', font=tag, fill=(170, 170, 180, 255))
    # chips
    chip = font('segoeui.ttf', 15)
    x = x0 + 2
    for label in ('D3D12', 'DIS optical flow', 'AMD FidelityFX', 'DirectML'):
        w = d.textlength(label, font=chip) / S + 26
        d.rounded_rectangle([x * S, 318 * S, (x + w) * S, 348 * S], radius=15 * S, fill=(20, 20, 24, 255),
                            outline=(70, 70, 80, 255), width=S)
        d.text(((x + 13) * S, 324 * S), label, font=chip, fill=(225, 225, 232, 255))
        x += w + 10
    # a thin lit edge at the bottom
    with Over(img) as d:
     for i in range(W):
        f = 1 - abs(i - W / 2) / (W / 2)
        d.line([(i * S, (H - 1) * S), ((i + 1) * S, (H - 1) * S)], fill=rgba(140 * f ** 2), width=S)
    return img.resize((W, H), Image.LANCZOS).convert('RGB')


def divider(text, path):
    """A section header strip: a small lit marker, the title, a fading rule."""
    w, h = 1280, 64
    img = Image.new('RGBA', (w * S, h * S), (8, 8, 10, 255))
    glow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle([10 * S, 18 * S, 18 * S, 46 * S], radius=4 * S, fill=rgba(140))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(6 * S)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([12 * S, 20 * S, 16 * S, 44 * S], radius=2 * S, fill=rgba(255))
    f = font('seguisb.ttf', 24)
    end = tracked(d, 34 * S, 15 * S, text.upper(), f, 4, (255, 255, 255, 255)) / S
    with Over(img) as d:
     for i in range(int(end + 20), w - 10):
        a = 70 * (1 - (i - end - 20) / (w - end - 30))
        d.line([(i * S, 33 * S), ((i + 1) * S, 33 * S)], fill=rgba(a), width=S)
    img.resize((w, h), Image.LANCZOS).convert('RGB').save(path)


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    banner().save(os.path.join(OUT, 'banner.png'))
    logo_preview().save(os.path.join(OUT, 'logo-preview.png'))
    for name, text in (('highlights', 'Highlights'), ('interface', 'Interface'), ('quick-start', 'Quick start'),
                       ('how-it-works', 'How it works'), ('documentation', 'Technical documentation')):
        divider(text, os.path.join(OUT, f'section-{name}.png'))
    print('ok')
