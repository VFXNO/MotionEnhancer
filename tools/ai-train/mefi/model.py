"""The AI pyramid flow model (Motion Enhancer's "AI pyramid" engine).

Coarse to fine over a pyramid of the network resolution (level 0 the
network resolution itself, each level half the one below). The number of
levels follows the resolution (levels_for): enough that the top level is
4-8 px on its short side, so any motion is a pixel or two there - 7 levels
at 640x352 (a quarter of 1440p), 8 at 1280x720, 7 for a 256 px training
crop.
At every level a block sees both frames at that level, both frames warped
towards each other by the flow so far, that flow and the mask so far, and
predicts a residual to the flow and the mask - the IFNet idea (RIFE, Huang
et al. 2022), for flow in both directions instead of to a fixed middle
time, because the program's GPU synthesis (projection, occlusion handling,
any time t) works from the two directional fields.

Every level above level 3 shares one small block: a level's flow is in
that level's own pixels, so the same refinement applies at every scale -
which is also what lets one set of weights run any number of levels - and
with the user's own (often small) datasets fewer weights generalise better.
Levels 3 to 0 each have their own, wider block.

Sizes need not be powers of two: each level is ceil(previous / 2), and the
flow is carried up by the exact per-axis ratio between levels.

Outputs, at the network resolution:
  flow  [N, 4, H, W]  forward flow F01 (x, y) on frame0's grid, then
                      backward flow F10 (x, y) on frame1's grid, in pixels
                      of the network resolution
  mask  [N, 1, H, W]  how much of the middle frame comes from frame0
                      (sigmoid), used by training's synthesis at t = 0.5
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = 'pyramid-v2'
REFINED_ARCH = 'pyramid-refine-v1'
# The first two names share weights (the 8-level model was this one with a
# fixed level count). The refined architecture has additional versioned keys.
COMPATIBLE = (ARCH, 'pyramid8-v1', REFINED_ARCH)
FINE = 4        # levels 0..3 have their own blocks; all above share one


def levels_for(h, w):
    """Levels for a network resolution: the top level 4-8 px on its short
    side (at least FINE + 1, at most 10)."""
    short = min(h, w)
    return max(FINE + 1, min(10, int(math.floor(math.log2(max(short, 8) / 4.0))) + 1))


def warp(image, flow):
    """Backward warp: sample `image` at p + flow(p), flow in pixels.

    The sampling grid is built in FP32 whatever the network runs in: FP16
    resolves only 0.5 between 512 and 1024, so at x = 640 an FP16 grid put
    every sample on a half-pixel grid (a quarter pixel of error on average,
    a pixel at 1440p), in mixed-precision training and in the exported FP16
    model alike."""
    n, _, h, w = image.shape
    flow = flow.float()
    gy, gx = torch.meshgrid(torch.arange(h, device=image.device, dtype=torch.float32),
                            torch.arange(w, device=image.device, dtype=torch.float32), indexing='ij')
    x = (gx.unsqueeze(0) + flow[:, 0]) * (2.0 / max(w - 1, 1)) - 1.0
    y = (gy.unsqueeze(0) + flow[:, 1]) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((x, y), dim=-1)
    out = F.grid_sample(image.float(), grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out.to(image.dtype)


def level_sizes(h, w, levels=None):
    """(h, w) of every level, level 0 first: each ceil(previous / 2)."""
    if levels is None:
        levels = levels_for(h, w)
    sizes = [(h, w)]
    for _ in range(levels - 1):
        ph, pw = sizes[-1]
        sizes.append((max(1, math.ceil(ph / 2)), max(1, math.ceil(pw / 2))))
    return sizes


def resize(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def image_pyramid(image, sizes):
    """The image at every level (level 0 first), each from the one below."""
    out = [image]
    for size in sizes[1:]:
        out.append(resize(out[-1], size))
    return out


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.1, inplace=True))


class Block(nn.Module):
    """One refinement: a half-resolution body, pixel-shuffled back up."""

    def __init__(self, cin, width, depth):
        super().__init__()
        self.stem = nn.Sequential(conv(cin, width // 2, 2), conv(width // 2, width))
        self.body = nn.Sequential(*[conv(width, width) for _ in range(depth)])
        # 4 flow + 1 mask, x4 for the 2x2 pixel shuffle. Starts at zero:
        # an untrained model is "no motion, an even blend". With random
        # starting outputs, the top level's noise was doubled at each of the
        # seven levels below (up to ~128 px of random flow at level 0), and
        # training spent its first 19 epochs below a plain blend undoing it.
        self.head = nn.Conv2d(width, 5 * 4, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        f = self.stem(x)
        f = self.body(f) + f
        # An odd level is one pixel short of what the shuffle gives.
        return F.pixel_shuffle(self.head(f), 2)[:, :, :h, :w]


class FlowRefiner(nn.Module):
    """Local vector correction using the finest block's existing features.

    No second encoder, image generator, extra warp or extra scored pyramid
    level. Pixel shuffle exposes the half-resolution body's features at full
    resolution; a narrow, depthwise spatial filter then corrects the four flow
    channels only. The existing blend mask is unchanged. The correction starts
    at zero and is bounded to +/-2 network pixels per component, so this head
    cannot replace the coarse pyramid's large-displacement search.
    """

    LIMIT = 2.0

    def __init__(self, width):
        super().__init__()
        # mix is deliberately 1x1: only the inexpensive depthwise layer works
        # spatially at full resolution.
        self.mix = nn.Sequential(nn.Conv2d(BLOCK_INPUT + width // 4 + 5, 16, 1), nn.LeakyReLU(0.1))
        self.local = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.LeakyReLU(0.1))
        self.head = nn.Conv2d(16, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, features, residual):
        detail = F.pixel_shuffle(features, 2)[:, :, :x.shape[-2], :x.shape[-1]]
        context = torch.cat((x, detail, residual), 1)
        return torch.tanh(self.head(self.local(self.mix(context)))) * self.LIMIT


class RefinedBlock(Block):
    """The existing finest block plus a shared-feature, flow-only head."""

    def __init__(self, cin, width, depth):
        super().__init__(cin, width, depth)
        self.refiner = FlowRefiner(width)

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.stem(x)
        features = self.body(features) + features
        residual = F.pixel_shuffle(self.head(features), 2)[:, :, :h, :w]
        correction = self.refiner(x, features, residual)
        # The caller scales the block's four flow outputs by FLOW_GAIN.
        return torch.cat((residual[:, :4] + correction / FLOW_GAIN, residual[:, 4:5]), 1)


# The blocks' flow outputs are scaled up: motion is tens of pixels while a
# freshly initialised layer outputs about one. Without the gain, training
# spent its first dozen epochs only growing the output (on a small test set
# it first beat a plain blend at epoch 14; with it, at epoch 1, and it
# reached the same final quality). Part of the model: changing it changes
# what a checkpoint means.
FLOW_GAIN = 4.0

# Preset: (shared coarse block width, depth), (widths of levels 3, 2, 1, 0),
# depth of the fine blocks.
PRESETS = {
    'small': ((48, 1), (64, 48, 40, 32), 2),
    'medium': ((64, 2), (96, 64, 48, 40), 3),
    'large': ((96, 2), (128, 96, 64, 48), 4),
}

# Every block's input: both frames, both warped, the flow (4) and the mask.
BLOCK_INPUT = 6 + 6 + 4 + 1


class PyramidFlow(nn.Module):
    def __init__(self, preset='medium', refiner=False):
        super().__init__()
        (coarse_width, coarse_depth), fine_widths, fine_depth = PRESETS[preset]
        self.preset = preset
        self.arch = REFINED_ARCH if refiner else ARCH
        self.coarse = Block(BLOCK_INPUT, coarse_width, coarse_depth)
        self.fine = nn.ModuleList([(RefinedBlock if refiner and i == FINE - 1 else Block)(
            BLOCK_INPUT, w, fine_depth) for i, w in enumerate(fine_widths)])

    def block(self, level):
        """The block of a level: its own for levels 0..3, the shared one above."""
        return self.coarse if level >= FINE else self.fine[FINE - 1 - level]

    def without_refiner(self):
        """Export-only backbone copy: omit the head AND its computation.

        These are the jointly trained backbone weights, not a separately
        trained baseline. Never mutate the checkpoint or a training model.
        """
        net = PyramidFlow(self.preset)
        weights = {key: value for key, value in self.state_dict().items()
                   if not key.startswith(f'fine.{FINE - 1}.refiner.')}
        net.load_state_dict(weights, strict=True)
        return net

    def forward(self, frame0, frame1, all_levels=False):
        """frame0, frame1: [N, 3, H, W] in 0..1. Any H, W (multiples of 16
        keep every level's sampling exact)."""
        n, _, h, w = frame0.shape
        sizes = level_sizes(h, w)
        p0 = image_pyramid(frame0, sizes)
        p1 = image_pyramid(frame1, sizes)
        flow = None
        mask = None
        results = []
        for level in range(len(sizes) - 1, -1, -1):
            f0, f1 = p0[level], p1[level]
            lh, lw = sizes[level]
            if flow is None:
                flow = torch.zeros(n, 4, lh, lw, device=frame0.device, dtype=frame0.dtype)
                mask = torch.zeros(n, 1, lh, lw, device=frame0.device, dtype=frame0.dtype)
                w1, w0 = f1, f0
            else:
                # Up one level: the per-axis size ratio, in pixels as well.
                ph, pw = flow.shape[-2:]
                ratio = torch.tensor([lw / pw, lh / ph, lw / pw, lh / ph], device=flow.device,
                                     dtype=flow.dtype).view(1, 4, 1, 1)
                flow = resize(flow, (lh, lw)) * ratio
                mask = resize(mask, (lh, lw))
                w1 = warp(f1, flow[:, 0:2])   # frame1 seen from frame0 (F01)
                w0 = warp(f0, flow[:, 2:4])   # frame0 seen from frame1 (F10)
            out = self.block(level)(torch.cat((f0, f1, w1, w0, flow, mask), 1))
            flow = flow + out[:, :4] * FLOW_GAIN
            mask = mask + out[:, 4:5]
            if all_levels:
                results.append((flow, mask, level))
        if all_levels:
            return results
        return flow, torch.sigmoid(mask)


def load_checkpoint(path, map_location='cpu'):
    """A checkpoint and its model, refusing ones of another architecture
    with a readable message."""
    ck = torch.load(path, map_location=map_location)
    arch = ck.get('arch', 'pyramid3-v0')
    if arch not in COMPATIBLE:
        raise ValueError(f'{path} is a {arch} checkpoint; this trainer makes {ARCH} models. '
                         'Train a new run (the model changed to a pyramid of up to 10 levels).')
    net = PyramidFlow(ck['preset'], refiner=arch == REFINED_ARCH)
    net.load_state_dict(ck['model'])
    return ck, net


def initialize_from(net, path):
    """Explicit weights-only start of a NEW run, never an architecture resume.

    A legacy model may seed its refined variant without losing any learned
    weights. Only the new, initially zero-output refiner may be missing.
    Optimizer/schedule/validation state is intentionally not imported.
    """
    ck, source = load_checkpoint(path)
    if source.preset != net.preset:
        raise ValueError(f"initial checkpoint is {source.preset}; select the same model size")
    if source.arch == REFINED_ARCH and net.arch != REFINED_ARCH:
        raise ValueError('cannot discard a trained integrated refiner; enable --refiner')
    if not all(bool(torch.isfinite(value).all()) for value in ck['model'].values()):
        raise ValueError('initial checkpoint contains nonfinite weights')
    missing = set(net.state_dict()) - set(ck['model'])
    if any(not name.startswith('fine.3.refiner.') for name in missing):
        raise ValueError('initial checkpoint is missing backbone weights')
    net.load_state_dict(ck['model'], strict=not missing)
    return ck


class Exported(nn.Module):
    """What the program runs: one [1, 6, H, W] input (frame0 RGB, frame1
    RGB, 0..1, planar), the network in FP16 (warps in FP32), FP32 outputs."""

    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, frames):
        x = frames.half()
        flow, mask = self.net(x[:, 0:3], x[:, 3:6])
        return flow.float(), mask.float()
