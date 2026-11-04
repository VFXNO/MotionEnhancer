"""Trains the AI pyramid flow model on triplet datasets (mefi.dataset).

    python -m mefi.train --data datasets/mygame --run runs/mygame --preset medium --epochs 60

Every one of the 8 pyramid levels is scored by how well its flow rebuilds the middle frame
of each triplet at t = 0.5, plus how well each direction's flow alone maps
one end frame onto the other, plus a little smoothness of the flow where the
picture is flat. No motion labels are needed.

Progress goes to stdout as JSON lines (the trainer app reads them). Writing
a file named STOP into the run folder ends the run after the current step,
keeping the checkpoints. --resume continues from the run's last checkpoint.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import (RESUME_VERSION, backup_checkpoint, lr_factor, restore_rng,
                         rng_state, save_checkpoint, save_config)
from .data import EpochBatches, FullFrames, Triplets, is_moving, load_index, sample_identity, split, validation_identity
from .model import (ARCH, COMPATIBLE, REFINED_ARCH, PyramidFlow, image_pyramid,
                    initialize_from, level_sizes, warp)


def emit(**event):
    print(json.dumps(event), flush=True)


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()


def census(a, b, size=7):
    """Ternary census difference: compares local structure rather than
    brightness, so a flow is not rewarded for matching shading alone."""
    def transform(img):
        g = img.mean(1, keepdim=True)
        patches = F.unfold(g, size, padding=size // 2)
        centre = g.flatten(2)
        t = patches - centre
        return t / torch.sqrt(0.81 + t * t)
    ta, tb = transform(a), transform(b)
    d = (ta - tb) ** 2
    return (d / (0.1 + d)).mean()


def smoothness(flow, image, weight=10.0):
    """First-order smoothness, relaxed across picture edges."""
    dx_f = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy_f = flow[:, :, 1:] - flow[:, :, :-1]
    dx_i = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(1, keepdim=True)
    dy_i = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(1, keepdim=True)
    return (dx_f.abs() * torch.exp(-weight * dx_i)).mean() + (dy_f.abs() * torch.exp(-weight * dy_i)).mean()


def visible(f01, f10):
    """Where each field's points are seen in the other frame (UnFlow's
    forward-backward check, Meister et al. 2018): F01 and F10 taken there
    cancel out to within a few percent of their size plus half a pixel.
    Occluded points - hidden in the other frame - have no correspondence,
    and scoring them pushed the model to bend the motion at a moving
    object's edge instead of leaving the occluded band to the program's
    occlusion handling. Returns (frame0's mask, frame1's mask), 1 = visible,
    and the two round-trip errors."""
    f01, f10 = f01.float(), f10.float()
    back0 = warp(f10, f01)          # F10 where F01 lands
    back1 = warp(f01, f10)
    trip0 = f01 + back0
    trip1 = f10 + back1
    def mask(trip, a, b):
        bound = 0.01 * ((a * a).sum(1, keepdim=True) + (b * b).sum(1, keepdim=True)) + 0.5
        return ((trip * trip).sum(1, keepdim=True) < bound).float()
    return mask(trip0, f01, back0).detach(), mask(trip1, f10, back1).detach(), trip0, trip1


def masked_charbonnier(x, mask, eps=1e-3):
    c = torch.sqrt(x.float() ** 2 + eps * eps)
    return (c * mask).sum() / (mask.sum() * x.shape[1] + 1.0)


def middle(f0, f2, flow, mask):
    """The middle frame from the two directional flows (Super SloMo's
    linear-motion approximation at t = 0.5) and the blend mask."""
    f01, f10 = flow[:, 0:2], flow[:, 2:4]
    ft0 = -0.25 * f01 + 0.25 * f10
    ft1 = 0.25 * f01 - 0.25 * f10
    return mask * warp(f0, ft0) + (1.0 - mask) * warp(f2, ft1)


# Every level's loss counts the same. Halving it per level left the coarse
# levels - the only ones that can find a large motion - almost untrained,
# and training sat at a plain blend for longer (synthetic test, 15 epochs:
# 33.0 / 32.9 dB on two seeds with equal weights and a learning rate of
# 1e-3, against 32.6 / 31.7 before).
LEVEL_WEIGHT = 1.0
# Levels smaller than this are not scored (census compares 7x7 patches).
# Scoring every level measured no better (32.7 against 33.0 dB).
MIN_SCORED = 8
DIRECT_WEIGHT = 0.25


def losses(net, batch, occlusion=1.0):
    """`occlusion` (0..1) fades in the occlusion masking: with it on from the
    start, the untrained model's two directions disagreed everywhere, the
    check marked almost every point hidden, the main motion signal vanished
    and training settled on a plain blend (29.99 dB against 32.64 without,
    on the synthetic test, after 15 epochs). UnFlow, likewise, adds it only
    after training without it."""
    f0, f1, f2 = batch[:, 0:3], batch[:, 3:6], batch[:, 6:9]
    levels = net(f0, f2, all_levels=True)
    # The frames at every level, exactly as the network built its own.
    sizes = level_sizes(*f0.shape[-2:])
    p0, p1, p2 = image_pyramid(f0, sizes), image_pyramid(f1, sizes), image_pyramid(f2, sizes)
    total = 0.0
    parts = {}
    final = None
    for flow, mask_logit, level in levels:
        a, m, b = p0[level], p1[level], p2[level]
        if min(a.shape[-2:]) < MIN_SCORED:
            continue
        mask = torch.sigmoid(mask_logit)
        pred = middle(a, b, flow, mask)
        rec = charbonnier(pred - m) + census(pred, m)
        # Each direction mapping one end frame onto the other, where the
        # point is visible in both, and the two directions agreeing there -
        # what the program's occlusion check (round trip) relies on.
        v0, v1, trip0, trip1 = visible(flow[:, 0:2], flow[:, 2:4])
        e0, e1 = warp(b, flow[:, 0:2]) - a, warp(a, flow[:, 2:4]) - b
        plain = charbonnier(e0) + charbonnier(e1)
        if occlusion > 0.0:
            masked = masked_charbonnier(e0, v0) + masked_charbonnier(e1, v1)
            direct = (1.0 - occlusion) * plain + occlusion * masked
            consistent = occlusion * (masked_charbonnier(trip0, v0) + masked_charbonnier(trip1, v1))
        else:
            direct = plain
            consistent = torch.zeros((), device=flow.device)
        # A term pulling the two directions to be mirror images everywhere
        # was tried: it drove training to near-zero motion (which is
        # trivially consistent) and a plain blend's score.
        smooth = smoothness(flow[:, 0:2], a) + smoothness(flow[:, 2:4], b)
        weight = LEVEL_WEIGHT ** level
        total = total + weight * (rec + DIRECT_WEIGHT * direct + 0.02 * consistent + 0.02 * smooth)
        if level == 0:
            # Kept on the GPU: reading them back every step made the CPU wait
            # for the GPU each step; they are read when progress is reported.
            parts = {'rec': rec.detach(), 'direct': direct.detach(), 'smooth': smooth.detach(),
                     'visible': (0.5 * (v0.mean() + v1.mean())).detach()}
            final = pred
    return total, parts, final


@torch.no_grad()
def validate(net, dataset, device, epoch):
    """PSNR of the rebuilt middle frame and of a plain blend, over the
    moving validation triplets (the headline) and over all of them, and the
    preview of this epoch's sample (a different moving one each epoch)."""
    net.eval()
    scores = {'moving': ([], []), 'all': ([], [])}
    show = dataset.preview_order[epoch % len(dataset.preview_order)]
    preview = None
    for k in range(len(dataset)):
        batch, moving = dataset[k]
        batch = batch.unsqueeze(0).to(device)
        f0, f1, f2 = batch[:, 0:3], batch[:, 3:6], batch[:, 6:9]
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            flow, mask = net(f0, f2)
        flow, mask = flow.float(), mask.float()
        pred = middle(f0, f2, flow, mask).clamp(0, 1)
        blend = 0.5 * (f0 + f2)
        p = float(10 * torch.log10(1.0 / ((pred - f1) ** 2).mean().clamp_min(1e-10)))
        b = float(10 * torch.log10(1.0 / ((blend - f1) ** 2).mean().clamp_min(1e-10)))
        for group in (('moving', 'all') if bool(moving) else ('all',)):
            scores[group][0].append(p)
            scores[group][1].append(b)
        if k == show:
            v0, v1, _, _ = visible(flow[:, 0:2], flow[:, 2:4])
            preview = (f0[0], f1[0], f2[0], pred[0], blend[0], flow[0, 0:2], flow[0, 2:4], mask[0], v0[0])
    net.train()
    mean = lambda xs: float(np.mean(xs)) if xs else float('nan')
    out = {g: (mean(s[0]), mean(s[1]), len(s[0])) for g, s in scores.items()}
    return out, preview


def flow_image(flow):
    """Flow as colour: hue = direction, brightness = speed."""
    import cv2
    f = flow.permute(1, 2, 0).cpu().numpy()
    mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
    hsv = np.zeros(f.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 255 / max(np.percentile(mag, 99), 1.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def save_preview(path, preview):
    """Three rows: frame0, the true middle, frame2 / the model's middle, a
    plain blend, the model's error x3 / its forward and backward motion, and
    its occlusion view (white: the blend takes frame0, black: frame2; red:
    hidden in frame2 by the forward-backward check)."""
    import cv2
    f0, f1, f2, pred, blend, f01, f10, mask, vis = preview
    to = lambda t: (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    error = (np.abs(to(pred).astype(int) - to(f1).astype(int)) * 3).clip(0, 255).astype(np.uint8)
    m = (mask[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    occ = np.stack([m, m, m], -1)
    hidden = vis[0].cpu().numpy() < 0.5
    occ[hidden] = (occ[hidden] * 0.4 + np.array([255, 40, 40]) * 0.6).astype(np.uint8)
    rows = [np.concatenate([to(f0), to(f1), to(f2)], 1),
            np.concatenate([to(pred), to(blend), error], 1),
            np.concatenate([flow_image(f01), flow_image(f10), occ], 1)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def headline(scores):
    group = 'moving' if scores['moving'][2] else 'all'
    if not math.isfinite(scores[group][0]):
        raise ValueError('validation is not finite; existing checkpoints have been kept')
    return group, scores[group]


def read_resume(path, args):
    """CPU loading also keeps RNG byte tensors usable on either device."""
    ck = torch.load(path, map_location='cpu')
    if not isinstance(ck, dict) or not all(k in ck for k in ('model', 'opt', 'sched', 'epoch', 'preset')):
        raise ValueError('last.pt is incomplete: it needs model, optimizer, scheduler and epoch state')
    if ck.get('arch') not in COMPATIBLE:
        raise ValueError(f"this checkpoint is {ck.get('arch', 'older')}, not a compatible {ARCH} model")
    if ck.get('resume_version', 0) > RESUME_VERSION:
        raise ValueError('this checkpoint needs a newer trainer version')
    if ck.get('resume_version') and not all(k in ck for k in
            ('config', 'rng', 'scaler', 'step', 'next_epoch', 'next_batch', 'training', 'steps_per_epoch', 'schedule')):
        raise ValueError('last.pt is incomplete: continuation state is missing')
    config = ck.get('config')
    if not config:
        path = os.path.join(args.run, 'config.json')
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as handle:
                config = json.load(handle)
    config = config or {}
    # Continue means the same experiment, not a silent change of batch size,
    # crop, seed, initial LR or model because the UI still has other defaults.
    for name in ('data', 'batch', 'crop', 'lr', 'val', 'pan', 'warmup', 'occlusion_start', 'seed'):
        if name in config:
            setattr(args, name, config[name])
    args.preset = ck['preset']
    args.refiner = ck['arch'] == REFINED_ARCH
    args.init_from = None
    return ck


def rescore_resume(net, ck, previous, dataset, device, validation, train_height):
    """Rebase legacy/changed metrics instead of discarding learned weights or
    comparing old and new scores. The old best is evaluated, not forgotten."""
    best, candidate = -1.0, None
    try:
        for state in ([previous] if previous else []) + [ck]:
            if (state.get('arch') not in COMPATIBLE or state.get('preset') != ck['preset']
                    or (state.get('arch') == REFINED_ARCH) != (ck['arch'] == REFINED_ARCH)):
                raise ValueError('best.pt belongs to a different model; existing checkpoints have been kept')
            net.load_state_dict(state['model'])
            scores, _ = validate(net, dataset, device, max(0, state['epoch']))
            _, (score, _, _) = headline(scores)
            if score > best:
                best = score
                candidate = {'arch': net.arch, 'preset': ck['preset'], 'model': state['model'],
                             'psnr': score, 'epoch': state['epoch'], 'train_height': train_height,
                             'validation': validation}
    finally:
        net.load_state_dict(ck['model'])
    return best, candidate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs='+', help='dataset folder(s); restored from the checkpoint on resume')
    p.add_argument('--run', required=True, help='run folder (checkpoints, previews, log)')
    p.add_argument('--preset', default='medium', choices=['small', 'medium', 'large'])
    p.add_argument('--refiner', action='store_true', help='integrated lightweight flow-only refinement (new run)')
    p.add_argument('--epochs', type=int, default=60, help='total target epochs, including epochs already trained')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val', type=float, default=0.1, help='last fraction of each video held out')
    p.add_argument('--pan', type=float, default=0.0,
                   help='largest synthetic camera pan, frame0 to frame2, as a fraction of the crop (0 = off)')
    p.add_argument('--warmup', type=int, default=-1, help='learning-rate warmup steps (-1: 5%% of the run, at most 500)')
    # Off by default: it measured no better (32.4-32.5 dB against 33.0
    # without, synthetic test) and fading it in at 20% stalled training.
    p.add_argument('--occlusion-start', type=float, default=2.0,
                   help='fraction of the run where occlusion masking starts fading in (over 0.2); 2 = never')
    start = p.add_mutually_exclusive_group()
    start.add_argument('--resume', action='store_true')
    start.add_argument('--init-from', help='weights-only initialization in a new run; preserves the source checkpoint')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save-every', type=int, default=100, help='checkpoint interval in steps (0: only stop/epoch end)')
    p.add_argument('--stop-token', help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    last_path = os.path.join(args.run, 'last.pt')
    best_path = os.path.join(args.run, 'best.pt')
    if not args.resume and any(os.path.exists(os.path.join(args.run, name)) for name in ('best.pt', 'last.pt')):
        emit(event='error', message='This run already has checkpoints. Choose a new run folder or Continue this run; '
             'existing checkpoints have not been changed.')
        return 1
    if args.resume and not os.path.exists(last_path):
        emit(event='error', message='Cannot continue: this run has no last.pt checkpoint. Choose the original '
             'run folder, or start a new run without Continue this run.')
        return 1
    ck = None
    if args.resume:
        try:
            ck = read_resume(last_path, args)
        except Exception as error:
            emit(event='error', message=f'Cannot continue: {error}')
            return 1
    if not args.data:
        emit(event='error', message='Choose a dataset folder (--data). This checkpoint has no saved dataset settings.')
        return 1
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.save_every < 0 or not 0 < args.val < 1:
        p.error('epochs/batch must be positive, workers/save-every nonnegative, and val between 0 and 1')
    os.makedirs(args.run, exist_ok=True)
    stop_file = os.path.join(args.run, 'STOP')
    if not args.stop_token and os.path.exists(stop_file):
        os.remove(stop_file)

    def stop_requested():
        if not os.path.isfile(stop_file):
            return False
        if not args.stop_token:
            return True
        with open(stop_file, encoding='utf-8') as handle:
            return handle.read().strip() == args.stop_token

    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit(event='device', name=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
         torch=torch.__version__)

    items, missing = load_index(args.data)
    if missing:
        # The index lists triplets whose images are gone (a deleted
        # `triplets` folder): training uses the rest, and the dataset tab's
        # Build makes the missing ones again from the same videos.
        emit(event='warning', message=f'{missing} triplets in the index have no images on disk '
                                       f'and are skipped; rebuild the dataset to restore them')
    train_items, val_items = split(items, args.val)
    if not train_items or not val_items:
        hint = ' - their images are missing: rebuild the dataset from its videos' if missing else ''
        emit(event='error', message=f'need triplets for training and validation (have {len(items)}{hint})')
        return 1
    train_set = Triplets(train_items, args.crop, train=True, pan=args.pan)
    # The frame height the dataset was built at: the export compares it with
    # the network resolution it is asked for.
    train_height = int(np.median([e['height'] for _, _, e in train_items]))
    val_set = FullFrames(val_items)
    validation = validation_identity(val_set)
    training = sample_identity(train_items)
    batches = EpochBatches(len(train_set), args.batch, args.seed)
    per_epoch = batches.steps
    train_loader = DataLoader(train_set, batch_sampler=batches, num_workers=args.workers,
                              persistent_workers=args.workers > 0, pin_memory=device.type == 'cuda',
                              generator=torch.Generator().manual_seed(args.seed))
    emit(event='data', train=len(train_items), val=len(val_set), val_available=len(val_items),
         val_moving=sum(1 for i in val_set.items if is_moving(i)),
         val_videos=len({v for _, v, _ in val_set.items}), validation=validation,
         videos=len({v for _, v, _ in items}), steps_per_epoch=per_epoch)
    if len(train_set) < args.batch:
        emit(event='warning', message=f'Only {len(train_set)} training triplets: using one batch of that size '
             f'instead of dropping every step. Add more varied motion footage for a useful model.')

    net = PyramidFlow(args.preset, refiner=args.refiner)
    if args.init_from:
        try:
            source = initialize_from(net, args.init_from)
        except Exception as error:
            emit(event='error', message=f'Cannot initialize new run: {error}')
            return 1
        emit(event='initialized', checkpoint=os.path.abspath(args.init_from), source_arch=source['arch'],
             arch=net.arch, message='Weights only; optimizer, learning rate and validation start a new experiment.')
    net = net.to(device)
    emit(event='model', arch=net.arch, preset=args.preset, parameters=sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * per_epoch
    warmup = args.warmup if args.warmup >= 0 else min(500, steps // 20)
    plan = {'kind': 'cosine', 'steps': steps, 'warmup': warmup}
    start_epoch, start_batch, step, best = 0, 0, 0, -1.0
    if ck:
        step = int(ck.get('step', ck['sched'].get('last_epoch', 0)))
        start_epoch = int(ck.get('next_epoch', step // per_epoch))
        start_batch = int(ck.get('next_batch', step % per_epoch))
        if (step < 0 or start_epoch < 0 or not 0 <= start_batch <= per_epoch
                or step != start_epoch * per_epoch + start_batch):
            emit(event='error', message='Cannot continue: checkpoint batch position is invalid; files are unchanged.')
            return 1
        if ck.get('training') and (ck['training'] != training or ck.get('steps_per_epoch') != per_epoch):
            emit(event='error', message='Cannot continue exactly: the training samples changed. Restore the original '
                 'dataset for this run; existing checkpoints are unchanged.')
            return 1
        original_steps = int(ck.get('epochs', args.epochs)) * per_epoch
        original_warmup = args.warmup if args.warmup >= 0 else min(500, original_steps // 20)
        plan = ck.get('schedule', {'kind': 'cosine', 'steps': max(1, original_steps), 'warmup': original_warmup})
        if steps != original_steps:
            group = ck['opt']['param_groups'][0]
            factor = group['lr'] / group.get('initial_lr', args.lr)
            # Extend/retarget from the saved LR, not the high initial LR or a
            # newly evaluated cosine that jumps when its duration changes.
            plan = {'kind': 'tail', 'start': int(ck['sched']['last_epoch']),
                    'steps': int(ck['sched']['last_epoch']) + max(1, steps - step),
                    'factor': factor, 'floor': min(0.02, factor)}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, plan))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    if ck:
        try:
            net.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            if scaler.is_enabled() and ck.get('scaler'):
                scaler.load_state_dict(ck['scaler'])
            best = float(ck.get('best', -1.0))
            if not all(torch.isfinite(value).all() for value in net.parameters()):
                raise ValueError('the checkpoint contains nonfinite model weights')
            emit(event='resumed', epoch=start_epoch, batch=start_batch, step=step, steps=steps,
                 lr=opt.param_groups[0]['lr'], best=best, config=vars(args))
            if start_epoch >= args.epochs:
                emit(event='done', best=best, stopped=False, already_complete=True, epoch=start_epoch,
                     message=f'Already completed {start_epoch} epochs. Increase Total epochs above '
                     f'{start_epoch} to continue training.')
                return 0
            previous = torch.load(best_path, map_location='cpu') if os.path.isfile(best_path) else None
            if previous is not None and (previous.get('arch') not in COMPATIBLE
                    or previous.get('preset') != ck['preset']
                    or (previous.get('arch') == REFINED_ARCH) != args.refiner):
                raise ValueError('best.pt belongs to a different model; existing checkpoints have been kept')
            changed = ck.get('validation') != validation or (
                previous is not None and previous.get('validation') != validation)
            if previous is None and best >= 0:
                changed = True
            if changed:
                emit(event='warning', message='Validation changed or this is a legacy checkpoint. '
                     'Re-scoring the saved best and last weights on the current hold-out before continuing; '
                     'original checkpoints will be backed up.')
                best, rescored = rescore_resume(net, ck, previous, val_set, device, validation, train_height)
            else:
                rescored = None
                if previous is not None:
                    best = max(best, float(previous['psnr']))
            if ck.get('rng'):
                restore_rng(ck['rng'], device)
            else:
                emit(event='warning', message='Legacy checkpoint restored model/optimizer/scheduler. '
                     'Its batch position was inferred from the scheduler; old random/AMP states were not '
                     'recorded, so historical augmentation cannot be reproduced exactly.')
            if changed or not ck.get('resume_version'):
                backup_checkpoint(last_path)
                backup_checkpoint(best_path)
            if rescored is not None:
                save_checkpoint(rescored, best_path)
                emit(event='rescored', best=best, validation=validation)
        except Exception as error:
            emit(event='error', message=f'Cannot continue: {error}')
            return 1
    save_config(vars(args), os.path.join(args.run, 'config.json'))
    started = time.time()
    first_step = step
    stopped = False
    skipped = ck.get('skipped', 0) if ck else 0
    in_a_row = ck.get('in_a_row', 0) if ck else 0

    def write_last(epoch, next_epoch, next_batch, running, count):
        if not all(torch.isfinite(value).all() for value in net.parameters()):
            raise ValueError('Training produced nonfinite weights; the last usable checkpoint has been kept.')
        save_checkpoint({'resume_version': RESUME_VERSION, 'arch': net.arch, 'model': net.state_dict(),
                         'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                         'rng': rng_state(device), 'config': vars(args), 'schedule': plan,
                         'preset': args.preset, 'epoch': epoch, 'next_epoch': next_epoch, 'next_batch': next_batch,
                         'step': step, 'steps_per_epoch': per_epoch, 'training': training,
                         'loss_sum': float(running), 'loss_count': count, 'skipped': skipped, 'in_a_row': in_a_row,
                         'best': best, 'epochs': args.epochs, 'train_height': train_height,
                         'validation': validation}, last_path)

    if args.batch < 4:
        emit(event='warning', message=f'batch {args.batch} is very small: each step sees one or two clips, so '
                                      f'training is noisy and slow and can blow up; 8-16 is normal')
    for epoch in range(start_epoch, args.epochs):
        net.train()
        batches.epoch = epoch
        batches.start_batch = start_batch if epoch == start_epoch else 0
        partial = ck is not None and epoch == start_epoch and start_batch > 0
        running = torch.tensor(ck.get('loss_sum', 0.0) if partial else 0.0, device=device)
        count = ck.get('loss_count', 0) if partial else 0
        consumed = batches.start_batch
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            # Occlusion masking fades in between 40% and 60% of the run.
            occlusion = min(1.0, max(0.0, (step / steps - args.occlusion_start) / 0.2))
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                loss, parts, _ = losses(net, batch, occlusion)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(norm):
                scaler.step(opt)
                in_a_row = 0
            else:
                # A step that is not a number is not taken. At batch 1 the
                # motion blew up (smoothness 2650 against 0.05), the loss
                # overflowed FP16 136 times in a row, the loss scale halved
                # each time down to 4e-40, and then every weight became NaN;
                # validation scored NaN, which never beats the best, and the
                # run went on for 56 more epochs with its first one as best.
                skipped += 1
                in_a_row += 1
                if in_a_row >= 50:
                    emit(event='error', message=f'training diverged in epoch {epoch + 1} ({in_a_row} steps in a row '
                                                f'were not numbers); best.pt is kept. Use a larger batch (8-16) '
                                                f'or a lower learning rate.')
                    return 1
            scaler.update()
            sched.step()
            step += 1
            consumed += 1
            if torch.isfinite(loss):
                running += loss.detach().float()
                count += 1
            if step % 20 == 0:
                elapsed = time.time() - started
                done = step - first_step
                remaining = (steps - step) * elapsed / max(1, done)
                emit(event='step', step=step, steps=steps, epoch=epoch, loss=float(running) / max(1, count),
                     lr=sched.get_last_lr()[0], eta=round(remaining), skipped=skipped,
                     **{k: float(v) for k, v in parts.items()})
            stopped = stop_requested()
            if stopped or (args.save_every and step % args.save_every == 0):
                write_last(epoch, epoch, consumed, running, count)
                emit(event='checkpoint', step=step, epoch=epoch, batch=consumed, path=last_path)
            if stopped:
                break
        if not all(torch.isfinite(p).all() for p in net.parameters()):
            # Broken beyond what skipping steps prevents: stop, and keep the
            # last good checkpoints (best.pt, last.pt) as they are.
            emit(event='error', message=f'training diverged in epoch {epoch + 1} (the weights are no longer '
                                        f'numbers); best.pt is kept. Use a larger batch or a lower learning rate.')
            return 1
        scores, preview = validate(net, val_set, device, epoch)
        preview_path = os.path.join(args.run, 'preview.png')
        save_preview(preview_path, preview)
        # Keep the motion-only headline, but score all triplets when none
        # moves, including flash-only datasets with identical endpoints.
        try:
            group, (psnr, blend, _) = headline(scores)
        except ValueError as error:
            emit(event='error', message=str(error))
            return 1
        improved = psnr > best
        if improved:
            best = psnr
            save_checkpoint({'arch': net.arch, 'model': net.state_dict(), 'preset': args.preset, 'psnr': psnr,
                             'epoch': epoch, 'train_height': train_height, 'validation': validation}, best_path)
        complete = consumed == per_epoch
        write_last(epoch, epoch + 1 if complete else epoch, 0 if complete else consumed,
                   0.0 if complete else running, 0 if complete else count)
        emit(event='epoch', epoch=epoch, epochs=args.epochs, loss=float(running) / max(1, count), psnr=psnr,
             blend_psnr=blend, best=best, improved=improved, preview=preview_path, scored=group,
             scored_count=scores[group][2], complete=complete, step=step, steps=steps, batch=consumed,
             moving=scores['moving'][2], psnr_all=scores['all'][0], blend_all=scores['all'][1])
        if stopped:
            break
    emit(event='done', best=best, stopped=stopped, step=step, steps=steps, seconds=round(time.time() - started))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
