"""Offline reference history ownership. Not a live GPU queue/fence implementation."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Pair:
    generation: int
    previous: int
    current: int
    width: int
    height: int
    variant: str

    def __post_init__(self):
        if self.generation < 0 or self.previous < 0 or self.current <= self.previous:
            raise ValueError('Invalid capture generation or frame order')
        if self.width < 1 or self.height < 1 or not self.variant:
            raise ValueError('Invalid size/model variant')

    @property
    def context(self):
        return self.generation, self.width, self.height, self.variant


class History:
    """Keep two owned pair states, including the predecessor of a replacement.

    Call reset on cuts, capture gaps, errors or invalidations; pixel similarity
    does not prove continuity. Never select history using capture slots alone.
    """
    def __init__(self, channels=8):
        if not 1 <= channels <= 32:
            raise ValueError('Invalid state channel count')
        self.channels = channels
        self.reset()

    def reset(self):
        self.context = None
        self.entries = []
        self.pending = None

    def prepare(self, pair, *, device='cpu', dtype=torch.float32, discontinuity=False):
        if discontinuity or self.context != pair.context:
            self.reset()
            self.context = pair.context
        if self.entries and pair.current <= self.entries[0][0].current:
            self.reset()
            raise ValueError('Out-of-order/replayed pair cannot consume history')
        self.pending = pair
        shape = (1, self.channels, (pair.height + 1) // 2, (pair.width + 1) // 2)
        for prior, state in self.entries:
            if prior.current == pair.previous:
                return state.to(device=device, dtype=dtype).clone(), torch.ones(1, 1, 1, 1, device=device)
        # Missing history cannot leave an older unrelated candidate valid.
        self.entries = []
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(1, 1, 1, 1, device=device)

    def commit(self, pair, state):
        shape = (1, self.channels, (pair.height + 1) // 2, (pair.width + 1) // 2)
        if self.pending != pair or self.context != pair.context or tuple(state.shape) != shape or not bool(torch.isfinite(state).all()):
            self.reset()
            raise ValueError('Invalid/nonfinite state or unprepared capture context')
        if self.entries and pair.current <= self.entries[0][0].current:
            self.reset()
            raise ValueError('Out-of-order/replayed state cannot be committed')
        predecessor = [entry for entry in self.entries if entry[0].current == pair.previous]
        self.entries = [(pair, state.detach().clone())] + predecessor[:1]
        self.pending = None
