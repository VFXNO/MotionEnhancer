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

from .data import FullFrames, Triplets, is_moving, load_index, split
from .model import ARCH, COMPATIBLE, PyramidFlow, image_pyramid, level_sizes, warp


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs='+', required=True, help='dataset folder(s)')
    p.add_argument('--run', required=True, help='run folder (checkpoints, previews, log)')
    p.add_argument('--preset', default='medium', choices=['small', 'medium', 'large'])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val', type=float, default=0.1, help='fraction of videos held out')
    p.add_argument('--pan', type=float, default=0.0,
                   help='largest synthetic camera pan, frame0 to frame2, as a fraction of the crop (0 = off)')
    p.add_argument('--warmup', type=int, default=-1, help='learning-rate warmup steps (-1: 5%% of the run, at most 500)')
    # Off by default: it measured no better (32.4-32.5 dB against 33.0
    # without, synthetic test) and fading it in at 20% stalled training.
    p.add_argument('--occlusion-start', type=float, default=2.0,
                   help='fraction of the run where occlusion masking starts fading in (over 0.2); 2 = never')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args(argv)

    os.makedirs(args.run, exist_ok=True)
    random.seed(args.seed)
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
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, persistent_workers=args.workers > 0, pin_memory=True)
    emit(event='data', train=len(train_items), val=len(val_items),
         val_moving=sum(1 for i in val_items if is_moving(i)),
         videos=len({v for _, v, _ in items}), steps_per_epoch=len(train_loader))

    net = PyramidFlow(args.preset).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * len(train_loader))
    warmup = args.warmup if args.warmup >= 0 else min(500, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start_epoch, best = 0, -1.0
    last_path = os.path.join(args.run, 'last.pt')
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        if ck.get('arch') not in COMPATIBLE or ck.get('preset') != args.preset:
            emit(event='error', message=f"cannot continue: that run is a {ck.get('arch', 'older')} "
                                        f"{ck.get('preset')} model, not a {ARCH} {args.preset} one. Start a new run.")
            return 1
        if ck.get('epochs', args.epochs) != args.epochs:
            emit(event='error', message=f"cannot continue with a different epoch count "
                                        f"(the run was started for {ck['epochs']}); keep it at {ck['epochs']}.")
            return 1
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_epoch, best = ck['epoch'] + 1, ck.get('best', -1.0)
        emit(event='resumed', epoch=start_epoch)
    with open(os.path.join(args.run, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=1)

    stop_file = os.path.join(args.run, 'STOP')
    if os.path.exists(stop_file):
        os.remove(stop_file)
    started = time.time()
    step = start_epoch * len(train_loader)
    stopped = False
    skipped = 0
    in_a_row = 0
    if args.batch < 4:
        emit(event='warning', message=f'batch {args.batch} is very small: each step sees one or two clips, so '
                                      f'training is noisy and slow and can blow up; 8-16 is normal')
    for epoch in range(start_epoch, args.epochs):
        net.train()
        running, count = torch.zeros((), device=device), 0
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
            if torch.isfinite(loss):
                running += loss.detach().float()
                count += 1
            if step % 20 == 0:
                elapsed = time.time() - started
                done = step - start_epoch * len(train_loader)
                remaining = (steps - step) * elapsed / max(1, done)
                emit(event='step', step=step, steps=steps, epoch=epoch, loss=float(running) / max(1, count),
                     lr=sched.get_last_lr()[0], eta=round(remaining), skipped=skipped,
                     **{k: float(v) for k, v in parts.items()})
            if os.path.exists(stop_file):
                stopped = True
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
        # The headline is the moving triplets (a still one is rebuilt by a
        # plain blend already); all of them when none moves.
        group = 'moving' if scores['moving'][2] else 'all'
        psnr, blend, _ = scores[group]
        if not math.isfinite(psnr):
            emit(event='error', message=f'validation is not a number in epoch {epoch + 1}: training diverged; '
                                        f'best.pt is kept. Use a larger batch or a lower learning rate.')
            return 1
        improved = psnr > best
        if improved:
            best = psnr
            torch.save({'arch': ARCH, 'model': net.state_dict(), 'preset': args.preset, 'psnr': psnr,
                        'epoch': epoch, 'train_height': train_height}, os.path.join(args.run, 'best.pt'))
        torch.save({'arch': ARCH, 'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'preset': args.preset, 'epoch': epoch, 'best': best, 'epochs': args.epochs,
                    'train_height': train_height}, last_path)
        emit(event='epoch', epoch=epoch, epochs=args.epochs, loss=float(running) / max(1, count), psnr=psnr,
             blend_psnr=blend, best=best, improved=improved, preview=preview_path, scored=group,
             moving=scores['moving'][2], psnr_all=scores['all'][0], blend_all=scores['all'][1])
        if stopped:
            break
    emit(event='done', best=best, stopped=stopped, seconds=round(time.time() - started))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
