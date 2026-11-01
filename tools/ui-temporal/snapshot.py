"""Versioned weights-only RESEARCH snapshots, never application checkpoints.

These snapshots preserve scratch/frozen origin and UI weights; optimizer/RNG
resume is deliberately not claimed until a labeled training runner is approved.
"""
import hashlib
from pathlib import Path

import torch

from model import FORMAT, HEAD_ARCH, TemporalUi
from mefi.model import ARCH, REFINED_ARCH, PyramidFlow


def save(path, model):
    path = Path(path)
    payload = {'format': FORMAT, 'head_arch': HEAD_ARCH, 'kind': 'research-weights-only',
               'mode': model.mode, 'preset': model.backbone.preset, 'flow_arch': model.backbone.arch,
               'state_channels': model.state_channels, 'hidden': model.hidden,
               'ui_updates': model.ui_updates, 'flow_updates': model.flow_updates,
               'model': model.state_dict(), 'deployable': False}
    # Exclusive creation: neither an old run nor a user checkpoint is replaced.
    with path.open('xb') as handle:
        try:
            torch.save(payload, handle)
        except BaseException:
            handle.close()
            path.unlink()  # only the incomplete file created above
            raise
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, sha256):
    path = Path(path)
    if len(sha256) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != sha256.lower():
        raise ValueError('Research snapshot SHA256 mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('format'), payload.get('head_arch'), payload.get('kind')) != (
            FORMAT, HEAD_ARCH, 'research-weights-only') or payload.get('deployable') is not False:
        raise ValueError('Not a compatible research UI snapshot; legacy flow checkpoints are not converted')
    if payload.get('flow_arch') not in (ARCH, REFINED_ARCH):
        raise ValueError('Unsupported flow architecture in research snapshot')
    model = TemporalUi(PyramidFlow(payload['preset'], refiner=payload['flow_arch'] == REFINED_ARCH),
                       mode=payload['mode'], state_channels=payload['state_channels'], hidden=payload['hidden'])
    state = payload['model']
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Nonfinite research weights')
    model.load_state_dict(state, strict=True)
    for name in ('ui_updates', 'flow_updates'):
        value = payload[name]
        if not isinstance(value, int) or value < 0:
            raise ValueError('Invalid update count')
        setattr(model, name, value)
    return model
