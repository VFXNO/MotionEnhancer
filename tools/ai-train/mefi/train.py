"""Trains the AI pyramid flow model on triplet datasets (mefi.dataset).

    python -m mefi.train --data datasets/mygame --run runs/mygame --preset medium --epochs 60

Every pyramid level is scored by how well its flow rebuilds the middle frame
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

from .data import FullFrames, Triplets, load_index, split
from .model import PyramidFlow, warp


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


def middle(f0, f2, flow, mask):
    """The middle frame from the two directional flows (Super SloMo's
    linear-motion approximation at t = 0.5) and the blend mask."""
    f01, f10 = flow[:, 0:2], flow[:, 2:4]
    ft0 = -0.25 * f01 + 0.25 * f10
    ft1 = 0.25 * f01 - 0.25 * f10
    return mask * warp(f0, ft0) + (1.0 - mask) * warp(f2, ft1)


def losses(net, batch):
    f0, f1, f2 = batch[:, 0:3], batch[:, 3:6], batch[:, 6:9]
    levels = net(f0, f2, all_levels=True)
    total = 0.0
    parts = {}
    for flow, mask_logit, scale in levels:
        if scale > 1:
            a = F.interpolate(f0, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
            b = F.interpolate(f2, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
            m = F.interpolate(f1, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
        else:
            a, b, m = f0, f2, f1
        mask = torch.sigmoid(mask_logit)
        pred = middle(a, b, flow, mask)
        rec = charbonnier(pred - m) + census(pred, m)
        direct = charbonnier(warp(b, flow[:, 0:2]) - a) + charbonnier(warp(a, flow[:, 2:4]) - b)
        smooth = smoothness(flow[:, 0:2], a) + smoothness(flow[:, 2:4], b)
        weight = 1.0 / scale
        total = total + weight * (rec + 0.25 * direct + 0.02 * smooth)
        if scale == 1:
            parts = {'rec': float(rec), 'direct': float(direct), 'smooth': float(smooth)}
            final = pred
    return total, parts, final


@torch.no_grad()
def validate(net, loader, device):
    net.eval()
    psnrs, blends = [], []
    preview = None
    for batch in loader:
        batch = batch.to(device)
        f0, f1, f2 = batch[:, 0:3], batch[:, 3:6], batch[:, 6:9]
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            flow, mask = net(f0, f2)
        pred = middle(f0.float(), f2.float(), flow.float(), mask.float()).clamp(0, 1)
        mse = ((pred - f1) ** 2).mean(dim=(1, 2, 3))
        psnrs += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).tolist()
        blend = 0.5 * (f0 + f2)
        bmse = ((blend - f1) ** 2).mean(dim=(1, 2, 3))
        blends += (10 * torch.log10(1.0 / bmse.clamp_min(1e-10))).tolist()
        if preview is None:
            preview = (f0[0], f1[0], pred[0], f2[0], flow[0, 0:2].float())
    net.train()
    return float(np.mean(psnrs)), float(np.mean(blends)), preview


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
    import cv2
    f0, f1, pred, f2, flow = preview
    to = lambda t: (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    top = np.concatenate([to(f0), to(f1), to(f2)], 1)
    bottom = np.concatenate([to(pred), flow_image(flow), (np.abs(to(pred).astype(int) - to(f1).astype(int)) * 3).clip(0, 255).astype(np.uint8)], 1)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate([top, bottom], 0), cv2.COLOR_RGB2BGR))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs='+', required=True, help='dataset folder(s)')
    p.add_argument('--run', required=True, help='run folder (checkpoints, previews, log)')
    p.add_argument('--preset', default='medium', choices=['small', 'medium', 'large'])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val', type=float, default=0.1, help='fraction of videos held out')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args(argv)

    os.makedirs(args.run, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit(event='device', name=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
         torch=torch.__version__)

    items = load_index(args.data)
    train_items, val_items = split(items, args.val)
    if not train_items or not val_items:
        emit(event='error', message=f'need triplets for training and validation (have {len(items)})')
        return 1
    train_set = Triplets(train_items, args.crop, train=True)
    val_set = FullFrames(val_items)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, persistent_workers=args.workers > 0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    emit(event='data', train=len(train_items), val=len(val_items),
         videos=len({v for _, v, _ in items}), steps_per_epoch=len(train_loader))

    net = PyramidFlow(args.preset).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * len(train_loader))
    warmup = min(500, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start_epoch, best = 0, -1.0
    last_path = os.path.join(args.run, 'last.pt')
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
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
    for epoch in range(start_epoch, args.epochs):
        net.train()
        running, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                loss, parts, _ = losses(net, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += float(loss)
            count += 1
            if step % 20 == 0:
                elapsed = time.time() - started
                done = step - start_epoch * len(train_loader)
                remaining = (steps - step) * elapsed / max(1, done)
                emit(event='step', step=step, steps=steps, epoch=epoch, loss=running / count,
                     lr=sched.get_last_lr()[0], eta=round(remaining), **parts)
            if os.path.exists(stop_file):
                stopped = True
                break
        psnr, blend, preview = validate(net, val_loader, device)
        preview_path = os.path.join(args.run, 'preview.png')
        save_preview(preview_path, preview)
        improved = psnr > best
        if improved:
            best = psnr
            torch.save({'model': net.state_dict(), 'preset': args.preset, 'psnr': psnr, 'epoch': epoch},
                       os.path.join(args.run, 'best.pt'))
        torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'preset': args.preset, 'epoch': epoch, 'best': best}, last_path)
        emit(event='epoch', epoch=epoch, epochs=args.epochs, loss=running / max(1, count), psnr=psnr,
             blend_psnr=blend, best=best, improved=improved, preview=preview_path)
        if stopped:
            break
    emit(event='done', best=best, stopped=stopped, seconds=round(time.time() - started))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
