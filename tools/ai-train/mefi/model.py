"""The AI pyramid flow model (Motion Enhancer's "AI pyramid" engine).

Coarse to fine over three scales of the network resolution (1/4, 1/2, 1/1;
at run time the network resolution is a quarter of the captured frame, so
640x352 for 1440p). Every block sees both frames at its scale, both frames
warped towards each other by the flow so far, and that flow, and predicts a
residual to it - the IFNet idea (RIFE, Huang et al. 2022), here for flow in
both directions instead of to a fixed middle time, because the program's
GPU synthesis (projection, occlusion handling, any time t) works from the
two directional fields.

Outputs, at the network resolution:
  flow  [N, 4, H, W]  forward flow F01 (x, y) on frame0's grid, then
                      backward flow F10 (x, y) on frame1's grid, in pixels
                      of the network resolution
  mask  [N, 1, H, W]  how much of the middle frame comes from frame0
                      (sigmoid), used by training's synthesis at t = 0.5

Kept small on purpose: about 30 layers, so DirectML's per-layer cost stays
low and the whole network runs in about a millisecond (see tools/ai-train
README).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(image, flow):
    """Backward warp: sample `image` at p + flow(p). flow in pixels."""
    n, _, h, w = image.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=image.device, dtype=flow.dtype),
                            torch.arange(w, device=image.device, dtype=flow.dtype), indexing='ij')
    x = (gx.unsqueeze(0) + flow[:, 0]) * (2.0 / max(w - 1, 1)) - 1.0
    y = (gy.unsqueeze(0) + flow[:, 1]) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((x, y), dim=-1)
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.1, inplace=True))


class Block(nn.Module):
    """One pyramid level: half-resolution body, 2x up to the level."""

    def __init__(self, cin, width, depth):
        super().__init__()
        self.stem = nn.Sequential(conv(cin, width // 2, 2), conv(width // 2, width))
        self.body = nn.Sequential(*[conv(width, width) for _ in range(depth)])
        # Pixel shuffle back to the level: 4 flow + 1 mask, x4 for the 2x2.
        self.head = nn.Conv2d(width, 5 * 4, 3, 1, 1)

    def forward(self, x):
        f = self.stem(x)
        f = self.body(f) + f
        return F.pixel_shuffle(self.head(f), 2)


# The blocks' flow outputs are scaled up: motion is tens of pixels while a
# freshly initialised layer outputs about one. Without the gain, training
# spent its first dozen epochs only growing the output (on a small test set
# it first beat a plain blend at epoch 14; with it, at epoch 1, and it
# reached the same final quality). Part of the model: changing it changes
# what a checkpoint means.
FLOW_GAIN = 4.0


# Preset widths: (channels at 1/4, 1/2, 1/1; residual convs per block).
PRESETS = {
    'small': ((64, 48, 32), 3),
    'medium': ((96, 64, 40), 4),
    'large': ((128, 96, 48), 5),
}


class PyramidFlow(nn.Module):
    def __init__(self, preset='medium'):
        super().__init__()
        widths, depth = PRESETS[preset]
        self.preset = preset
        # Level 0 sees the two frames only; the finer ones also the two
        # warped frames, the flow so far and the mask logit.
        self.blocks = nn.ModuleList([
            Block(6, widths[0], depth),
            Block(6 + 6 + 4 + 1, widths[1], depth),
            Block(6 + 6 + 4 + 1, widths[2], depth),
        ])
        self.scales = (4, 2, 1)

    def forward(self, frame0, frame1, all_levels=False):
        """frame0, frame1: [N, 3, H, W] in 0..1. H and W multiples of 16."""
        flow = None
        mask = None
        results = []
        for block, scale in zip(self.blocks, self.scales):
            if scale > 1:
                f0 = F.interpolate(frame0, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                f1 = F.interpolate(frame1, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
            else:
                f0, f1 = frame0, frame1
            if flow is None:
                out = block(torch.cat((f0, f1), 1))
                flow = out[:, :4] * FLOW_GAIN
                mask = out[:, 4:5]
            else:
                # Up to this level: twice the size, twice the pixels.
                flow = F.interpolate(flow, scale_factor=2.0, mode='bilinear', align_corners=False) * 2.0
                mask = F.interpolate(mask, scale_factor=2.0, mode='bilinear', align_corners=False)
                w1 = warp(f1, flow[:, 0:2])   # frame1 seen from frame0 (F01)
                w0 = warp(f0, flow[:, 2:4])   # frame0 seen from frame1 (F10)
                out = block(torch.cat((f0, f1, w1, w0, flow, mask), 1))
                flow = flow + out[:, :4] * FLOW_GAIN
                mask = mask + out[:, 4:5]
            if all_levels:
                results.append((flow, mask, scale))
        if all_levels:
            return results
        return flow, torch.sigmoid(mask)


class Exported(nn.Module):
    """What the program runs: one [1, 6, H, W] input (frame0 RGB, frame1
    RGB, 0..1, planar), the network in FP16, FP32 outputs."""

    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, frames):
        x = frames.half()
        flow, mask = self.net(x[:, 0:3], x[:, 3:6])
        return flow.float(), mask.float()
