"""Opt-in temporal UI model shared by the trainer and isolated research tools.

Scratch mode trains flow through its original objective. UI features are detached
in BOTH modes: UI supervision must never change flow/refinement/blend gradients.
The independently initialized projection/head predicts dense UI candidate logits.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import (FLOW_GAIN, PRESETS, PyramidFlow, RefinedBlock,
                    image_pyramid, level_sizes, resize, warp)


FORMAT = 'me-ui-pair-temporal-1'
HEAD_ARCH = 'pair-ui-dw-v1'
MODES = ('frozen', 'scratch')


class FeatureTap(nn.Module):
    """Read the existing finest body once; retain the original flow operations.

    The small forward adapter mirrors PyramidFlow.forward; parity tests guard
    against upstream changes. No hooks, second encoder, or duplicate finest
    block execution is introduced. This is not a per-frame descriptor encoder.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, frame0, frame1):
        n, _, h, w = frame0.shape
        sizes = level_sizes(h, w)
        p0, p1 = image_pyramid(frame0, sizes), image_pyramid(frame1, sizes)
        flow = mask = feature = None
        for level in range(len(sizes) - 1, -1, -1):
            f0, f1 = p0[level], p1[level]
            lh, lw = sizes[level]
            if flow is None:
                flow = torch.zeros(n, 4, lh, lw, device=frame0.device, dtype=frame0.dtype)
                mask = torch.zeros(n, 1, lh, lw, device=frame0.device, dtype=frame0.dtype)
                w1, w0 = f1, f0
            else:
                ph, pw = flow.shape[-2:]
                ratio = torch.tensor([lw / pw, lh / ph, lw / pw, lh / ph],
                                     device=flow.device, dtype=flow.dtype).view(1, 4, 1, 1)
                flow = resize(flow, (lh, lw)) * ratio
                mask = resize(mask, (lh, lw))
                w1, w0 = warp(f1, flow[:, 0:2]), warp(f0, flow[:, 2:4])
            x = torch.cat((f0, f1, w1, w0, flow, mask), 1)
            block = self.backbone.block(level)
            if level:
                out = block(x)
            else:
                stem = block.stem(x)
                feature = block.body(stem) + stem
                out = F.pixel_shuffle(block.head(feature), 2)[:, :, :lh, :lw]
                if isinstance(block, RefinedBlock):
                    correction = block.refiner(x, feature, out)
                    out = torch.cat((out[:, :4] + correction / FLOW_GAIN, out[:, 4:5]), 1)
            flow = flow + out[:, :4] * FLOW_GAIN
            mask = mask + out[:, 4:5]
        return flow, torch.sigmoid(mask), feature


class SpatialBlock(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.Conv2d(width, width, 3, padding=1, groups=width),
                         nn.LeakyReLU(.1), nn.Conv2d(width, width, 1), nn.LeakyReLU(.1))


class TemporalUi(nn.Module):
    def __init__(self, backbone, *, mode='frozen', state_channels=8, hidden=16):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'UI mode must be one of {MODES}')
        if not isinstance(backbone, PyramidFlow):
            raise TypeError('UI prototype requires the existing PyramidFlow architecture')
        if not 1 <= state_channels <= 32 or not 1 <= hidden <= 64:
            raise ValueError('State channels must be 1..32 and hidden width 1..64')
        # Never mutate the caller's model, requires_grad flags, mode or weights.
        self.tap = FeatureTap(copy.deepcopy(backbone))
        self.mode, self.state_channels, self.hidden = mode, state_channels, hidden
        channels = PRESETS[backbone.preset][1][-1]
        self.project = nn.Conv2d(channels, state_channels, 1)
        self.head = nn.Sequential(nn.Conv2d(2 * state_channels, hidden, 1), nn.LeakyReLU(.1),
                                  SpatialBlock(hidden), SpatialBlock(hidden), nn.Conv2d(hidden, 1, 1))
        self.backbone.requires_grad_(mode == 'scratch')
        self.ui_updates = 0
        self.flow_updates = 0
        self.train(True)

    @property
    def backbone(self):
        return self.tap.backbone

    @classmethod
    def from_scratch(cls, preset='medium', refiner=False, **kwargs):
        """New random flow + UI weights; no checkpoint/pretraining is required."""
        return cls(PyramidFlow(preset, refiner=refiner), mode='scratch', **kwargs)

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'frozen':
            self.backbone.eval()
        return self

    def parameter_groups(self, flow_lr=1e-3, ui_lr=3e-4):
        """Explicit optimizer groups, but no optimizer or training job is run."""
        groups = [{'name': 'ui', 'params': list(self.project.parameters()) + list(self.head.parameters()),
                   'lr': ui_lr}]
        if self.mode == 'scratch':
            groups.insert(0, {'name': 'flow', 'params': list(self.backbone.parameters()), 'lr': flow_lr})
        return groups

    def _logits(self, current_feature, previous_state):
        # Also detached in scratch mode. Only original flow losses train flow.
        current_state = self.project(current_feature.detach())
        return self.head(torch.cat((previous_state, current_state), 1)), current_state

    def forward(self, frames, previous_state, history_valid):
        """Streaming graph. Host must additionally verify pair/generation IDs."""
        if frames.ndim != 4 or frames.shape[1] != 6:
            raise ValueError('frames must be [N,6,H,W]')
        shape = (frames.shape[0], self.state_channels, (frames.shape[2] + 1) // 2,
                 (frames.shape[3] + 1) // 2)
        if tuple(previous_state.shape) != shape or tuple(history_valid.shape) != (frames.shape[0], 1, 1, 1):
            raise ValueError('History/state shape does not match frames/head')
        flow, blend, feature = self.tap(frames[:, :3], frames[:, 3:6])
        valid = (history_valid == 1) & torch.isfinite(previous_state).all(dim=(1, 2, 3), keepdim=True)
        # A bad/disabled history must not poison convolutions through NaN * 0.
        previous = torch.where(valid, previous_state, torch.zeros_like(previous_state))
        logits, current = self._logits(feature, previous)
        valid = valid & torch.isfinite(current).all(dim=(1, 2, 3), keepdim=True)
        valid = valid & torch.isfinite(logits).all(dim=(1, 2, 3), keepdim=True)
        probability = torch.sigmoid(logits)
        allow = torch.where(valid, probability, torch.zeros_like(probability))
        return flow, blend, allow, current

    def training_logits(self, frames):
        """Three actual consecutive RGB frames [N,9,H,W], not flow truth.

        Both backbone evaluations are no-grad. The shared projection gets
        gradients from BOTH pairs; no stale detached projection cache is used
        during a training step. No future frame or pseudo-label is generated.
        """
        if frames.ndim != 4 or frames.shape[1] != 9:
            raise ValueError('UI supervision needs three RGB frames [N,9,H,W]')
        with torch.no_grad():
            previous = self.tap(frames[:, :3], frames[:, 3:6])[2]
            current = self.tap(frames[:, 3:6], frames[:, 6:9])[2]
        return self._logits(current, self.project(previous.detach()))[0]


class ExportedTap(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.tap = FeatureTap(copy.deepcopy(backbone)).half().eval()

    def forward(self, frames):
        x = frames.half()
        return tuple(t.float() for t in self.tap(x[:, :3], x[:, 3:6]))


class ExportedTemporal(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = copy.deepcopy(model).half().eval()

    def forward(self, frames, previous_state, history_valid):
        return tuple(t.float() for t in self.net(frames.half(), previous_state.half(), history_valid))


def candidate_loss(logits, target, valid, *, positive_weight=1.0, dice_weight=.5):
    """Dense candidate coverage; unknown pixels ignored, empty positives use BCE."""
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError('UI logits, dense target and valid-pixel mask must have identical shapes')
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('UI tensors must be [N,1,H,W]')
    if not math.isfinite(positive_weight) or not math.isfinite(dice_weight) or not positive_weight > 0 or not dice_weight >= 0:
        raise ValueError('Invalid UI loss weights')
    if not bool(torch.isfinite(valid).all()) or not bool(((valid == 0) | (valid == 1)).all()):
        raise ValueError('Valid-pixel mask must contain finite zeros/ones')
    known = valid.bool()
    if not bool(torch.isfinite(target[known]).all()) or not bool(((target[known] >= 0) & (target[known] <= 1)).all()):
        raise ValueError('Known UI targets must be finite coverage values in [0,1]')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite UI logits')
    target = torch.where(known, target, torch.zeros_like(target)).float()
    weight, logits = valid.float(), logits.float()
    positive = logits.new_tensor(positive_weight)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=positive)
           * weight).sum() / weight.sum().clamp_min(1)
    probability, truth = torch.sigmoid(logits) * weight, target * weight
    axes = (1, 2, 3)
    truth_sum = truth.sum(axes)
    dice = 1 - (2 * (probability * truth).sum(axes) + 1e-6) / (
        probability.sum(axes) + truth_sum + 1e-6)
    foreground = (truth_sum > 0).float()
    dice = (dice * foreground).sum() / foreground.sum().clamp_min(1)
    return bce + dice_weight * dice
