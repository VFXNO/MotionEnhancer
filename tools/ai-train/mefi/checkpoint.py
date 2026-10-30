"""Durable trainer checkpoints and the state needed for real continuation."""
import json
import math
import os
import random
import shutil
import tempfile

import numpy as np
import torch


RESUME_VERSION = 1


def atomic_write(path, writer):
    """Never truncate the last usable checkpoint if saving fails."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix='.' + os.path.basename(path) + '-', suffix='.tmp', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_checkpoint(state, path):
    atomic_write(path, lambda handle: torch.save(state, handle))


def save_config(config, path):
    atomic_write(path, lambda handle: handle.write(json.dumps(config, indent=1).encode('utf-8')))


def backup_checkpoint(path):
    """Preserve originals when upgrading legacy state or rescoring best.pt."""
    if not os.path.isfile(path):
        return None
    index = 1
    while True:
        backup = f'{path}.before-resume-{index:03d}'
        try:
            with open(backup, 'xb') as output, open(path, 'rb') as source:
                shutil.copyfileobj(source, output)
            return backup
        except FileExistsError:
            index += 1


def rng_state(device):
    numpy = np.random.get_state()
    state = {'python': random.getstate(), 'torch': torch.get_rng_state(),
             # Plain Python values keep torch.load(weights_only=True) usable.
             'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:])}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda' and state.get('cuda'):
        if len(state['cuda']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def lr_factor(step, plan):
    if plan['kind'] == 'tail':
        progress = min(1.0, max(0.0, (step - plan['start']) / max(1, plan['steps'] - plan['start'])))
        return plan['floor'] + (plan['factor'] - plan['floor']) * 0.5 * (1 + math.cos(math.pi * progress))
    warm = min(1.0, (step + 1) / max(1, plan['warmup']))
    return warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / plan['steps']))))
