# Usage: python docs/tools/make_readme_art.py docs/assets
# The GUI screenshots and GIFs come from the app itself: MOTION_ENHANCER_GUI_SNAPSHOT,
# MOTION_ENHANCER_GUI_TIME and MOTION_ENHANCER_SPLASH_FRAMES (see README, Graphical Interface).
# Renders the README's hero banner and section dividers in the app's style:
# near-black, white light, the emblem (as GuiApp.cpp paintEmblem draws it).
# Drawn at 3x and scaled down for smooth edges.
import math, random, sys
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
    """The emblem at scale k (1 = the sidebar's size)."""
    sc = lambda v: v * k * S
    # glow
    glow = Image.new('RGBA', layer.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    r = sc(60)
    gd.ellipse([cx * S - r, cy * S - r, cx * S + r, cy * S + r], fill=rgba(70))
    glow = glow.filter(ImageFilter.GaussianBlur(sc(26)))
    layer.alpha_composite(glow)
    # rings
    with Over(layer) as d:
      for rad, a, w in ((30, 220, 1.7), (38, 90, 1.0)):
        rr = sc(rad)
        d.ellipse([cx * S - rr, cy * S - rr, cx * S + rr, cy * S + rr], outline=rgba(a), width=max(1, round(sc(w))))
    # spark with a fading tail
    head = -38.0
    for i in range(18):
      with Over(layer) as d:
        f = i / 18
        a0 = math.radians(head - 4 * (i + 1))
        a1 = math.radians(head - 4 * i)
        rr = sc(30)
        d.line([(cx * S + rr * math.cos(a0), cy * S + rr * math.sin(a0)),
                (cx * S + rr * math.cos(a1), cy * S + rr * math.sin(a1))],
               fill=rgba(230 * (1 - f) ** 2), width=max(1, round(sc(2.8 - 1.9 * f))))
    d = None
    ax = cx + 30 * k * math.cos(math.radians(head))
    ay = cy + 30 * k * math.sin(math.radians(head))
    for rr, a in ((6, 50), (2.6, 255)):
      with Over(layer) as d:
        d.ellipse([ax * S - sc(rr), ay * S - sc(rr), ax * S + sc(rr), ay * S + sc(rr)], fill=rgba(a))
    # nodes and traces
    with Over(layer) as d:
     for a0 in (-140, -40, 40, 140, -90, 90):
        a = math.radians(a0 + 12)
        x0, y0 = cx + 38 * k * math.cos(a), cy + 38 * k * math.sin(a)
        x1, y1 = cx + 48 * k * math.cos(a), cy + 48 * k * math.sin(a)
        d.line([(x0 * S, y0 * S), (x1 * S, y1 * S)], fill=rgba(90), width=max(1, round(sc(1.0))))
        d.ellipse([x1 * S - sc(2.5), y1 * S - sc(2.5), x1 * S + sc(2.5), y1 * S + sc(2.5)], fill=rgba(235))
    # trails
    with Over(layer) as d:
     for i, (dy, length, w) in enumerate(((-9, 18, 1.6), (0, 26, 2.4), (9, 18, 1.6))):
        n = 40
        for j in range(n):
            f0, f1 = j / n, (j + 1) / n
            x0 = cx - (length + 4) * k + (length + 6) * k * f0
            x1 = cx - (length + 4) * k + (length + 6) * k * f1
            d.line([(x0 * S, (cy + dy * k) * S), (x1 * S, (cy + dy * k) * S)], fill=rgba(220 * f1),
                   width=max(1, round(sc(w))))
    # core and halo
    core = Image.new('RGBA', layer.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    rr = sc(16)
    cd.ellipse([(cx + 8 * k) * S - rr, cy * S - rr, (cx + 8 * k) * S + rr, cy * S + rr], fill=rgba(120))
    core = core.filter(ImageFilter.GaussianBlur(sc(8)))
    layer.alpha_composite(core)
    with Over(layer) as d:
     rr = sc(7)
     d.ellipse([(cx + 8 * k) * S - rr, cy * S - rr, (cx + 8 * k) * S + rr, cy * S + rr], fill=rgba(255))
     rr = sc(12)
     d.ellipse([(cx + 8 * k) * S - rr, cy * S - rr, (cx + 8 * k) * S + rr, cy * S + rr], outline=rgba(150),
              width=max(1, round(sc(1.2))))


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
    for name, text in (('highlights', 'Highlights'), ('interface', 'Interface'), ('quick-start', 'Quick start'),
                       ('how-it-works', 'How it works'), ('documentation', 'Technical documentation')):
        divider(text, os.path.join(OUT, f'section-{name}.png'))
    print('ok')
