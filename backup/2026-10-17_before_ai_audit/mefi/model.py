"""The AI pyramid flow model (Motion Enhancer's "AI pyramid" engine).

Coarse to fine over an 8-level pyramid of the network resolution (level 0
the network resolution itself - a quarter of the captured frame, 640x352 for
1440p - each level half the one below, down to 5x3 px at level 7, so a
motion of a few hundred pixels at 1440p is a pixel or two at the top).
At every level a block sees both frames at that level, both frames warped
towards each other by the flow so far, that flow and the mask so far, and
predicts a residual to the flow and the mask - the IFNet idea (RIFE, Huang
et al. 2022), for flow in both directions instead of to a fixed middle
time, because the program's GPU synthesis (projection, occlusion handling,
any time t) works from the two directional fields.

The four coarsest levels share one small block: a level's flow is in that
level's own pixels, so the same refinement applies at every scale, and with
the user's own (often small) datasets fewer weights generalise better. The
four finest levels each have their own, wider block.

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

ARCH = 'pyramid8-v1'
LEVELS = 8
SHARED = 4      # the coarsest levels that share one block


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


def level_sizes(h, w, levels=LEVELS):
    """(h, w) of every level, level 0 first: each ceil(previous / 2)."""
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
        # 4 flow + 1 mask, x4 for the 2x2 pixel shuffle.
        self.head = nn.Conv2d(width, 5 * 4, 3, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f = self.stem(x)
        f = self.body(f) + f
        # An odd level is one pixel short of what the shuffle gives.
        return F.pixel_shuffle(self.head(f), 2)[:, :, :h, :w]


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
    def __init__(self, preset='medium'):
        super().__init__()
        (coarse_width, coarse_depth), fine_widths, fine_depth = PRESETS[preset]
        self.preset = preset
        self.coarse = Block(BLOCK_INPUT, coarse_width, coarse_depth)
        self.fine = nn.ModuleList([Block(BLOCK_INPUT, w, fine_depth) for w in fine_widths])

    def block(self, level):
        """The block of a level: the shared one above level 3."""
        return self.coarse if level >= LEVELS - SHARED else self.fine[LEVELS - SHARED - 1 - level]

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
        for level in range(LEVELS - 1, -1, -1):
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
    if arch != ARCH:
        raise ValueError(f'{path} is a {arch} checkpoint; this trainer makes {ARCH} models. '
                         'Train a new run (the model changed to an 8-level pyramid).')
    net = PyramidFlow(ck['preset'])
    net.load_state_dict(ck['model'])
    return ck, net


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
