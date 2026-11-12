"""Gradient descent for a picture: soft coloured blobs (2D Gaussians) start as noise and learn one photo.

Each blob is 9 numbers: position (2), log-size (2), angle (1), colour (3), strength (1). Every step the blobs are
painted, compared with the photo (L1 + SSIM), and Adam nudges every number to make the mistake smaller.
Every few steps new small blobs are added where the error is largest. The film is rendered from the run itself,
frame by frame, while it trains: every number on screen is the real one.

Runs on an Apple-silicon GPU (MPS), NVIDIA (CUDA) or the CPU. Pure PyTorch, no CUDA-only libraries."""
from __future__ import annotations

import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import torch.nn.functional as F

from . import core
from .core import Hud, Layout

BUCKETS = (5, 9, 17, 33, 65)          # patch sizes in px; a blob uses the smallest patch that covers 3 sigma


def device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Blobs:
    def __init__(self, nmax: int, n0: int, dev: torch.device, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.dev, self.nmax, self.n = dev, nmax, n0
        r = lambda *s: torch.rand(*s, generator=g)
        self.pos = r(nmax, 2).to(dev)
        self.lsig = (math.log(0.010) + r(nmax, 2) * (math.log(0.035) - math.log(0.010))).to(dev)
        self.rot = (r(nmax) * math.pi).to(dev)
        self.col = torch.logit(r(nmax, 3).clamp(0.03, 0.97)).to(dev)
        self.amp = torch.full((nmax,), 0.0).to(dev)
        self.params = [self.pos, self.lsig, self.rot, self.col, self.amp]
        for p in self.params:
            p.requires_grad_(True)

    LS_MIN, LS_MAX, AMP_MAX = math.log(0.0006), math.log(0.05), 3.0


def render(b: Blobs, W: int, H: int, view=(0.0, 0.0, 1.0, 1.0), bg=None, hard: float = 0.0, shrink: float = 1.0,
           chunk_px: int = 3_000_000) -> torch.Tensor:
    """Paint the first b.n blobs into an HxW image of the normalised window view=(u0, v0, u1, v1).
    hard>0 draws each blob as a crisp disc (display only, used for the 'it is all blobs' zoom)."""
    dev, n = b.dev, b.nmax          # always work on all nmax blobs (fixed shapes keep MPS kernels cached); dead ones get weight 0
    alive = (torch.arange(n, device=dev) < b.n)
    u0, v0, u1, v1 = view
    sxp, syp = W / (u1 - u0), H / (v1 - v0)                       # px per normalised unit
    cx = (b.pos[:n, 0] - u0) * sxp
    cy = (b.pos[:n, 1] - v0) * syp
    sig = torch.exp(b.lsig[:n]) * shrink
    sx, sy = sig[:, 0] * sxp, sig[:, 1] * sxp                     # sizes measured in x-units so blobs stay round
    c, s = torch.cos(b.rot[:n]), torch.sin(b.rot[:n])
    ix, iy = 1 / (sx * sx), 1 / (sy * sy)
    A = c * c * ix + s * s * iy
    Bc = c * s * (ix - iy)
    C = s * s * ix + c * c * iy
    a = Blobs.AMP_MAX * torch.sigmoid(b.amp[:n])
    rgb = torch.sigmoid(b.col[:n])
    rad = 3 * torch.maximum(sx, sy).detach()
    vis = alive & (cx + rad > 0) & (cx - rad < W) & (cy + rad > 0) & (cy - rad < H)
    lim = BUCKETS[-1] // 2
    acc = _splat(vis & (rad <= lim), cx, cy, A, Bc, C, a, rgb, rad, W, H, hard, chunk_px)
    # blobs too big for the largest patch are painted on a coarser grid (1/f resolution) and upsampled:
    # big blobs are smooth, so nothing is lost, and nothing is cut off into squares
    f = 2
    while True:
        m = vis & (rad > lim * f / 2) & ((rad <= lim * f) | (f >= 64))
        if bool(m.any()):
            Wf, Hf = -(-W // f), -(-H // f)
            sub = _splat(m, cx / f, cy / f, A * f * f, Bc * f * f, C * f * f, a, rgb, rad / f, Wf, Hf, hard, chunk_px)
            up = F.interpolate(sub.reshape(Hf, Wf, 4).permute(2, 0, 1).unsqueeze(0), size=(Hf * f, Wf * f),
                               mode="bilinear", align_corners=False)[0, :, :H, :W]
            acc = acc + up.permute(1, 2, 0).reshape(H * W, 4)
        if f >= 64 or not bool((vis & (rad > lim * f)).any()):
            break
        f *= 2
    ws = acc[:, 3:4]
    cov = 1 - torch.exp(-ws)
    bgc = bg if bg is not None else torch.ones(3, device=dev)
    return (bgc * (1 - cov) + acc[:, :3] / (ws + 1e-6) * cov).reshape(H, W, 3)


def _splat(vis, cx, cy, A, Bc, C, a, rgb, rad, W, H, hard, chunk_px) -> torch.Tensor:
    dev = cx.device
    acc = torch.zeros(H * W, 4, device=dev)
    lo = 0.0
    for P in BUCKETS:
        hi = P / 2 if P != BUCKETS[-1] else 1e9
        sel = torch.nonzero(vis & (rad > lo) & (rad <= hi)).squeeze(1)
        lo = hi
        if sel.numel() == 0:
            continue
        o = torch.arange(P, device=dev) - P // 2
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
        # MPS recompiles kernels for every new tensor shape, so pad the selection to a multiple of Q blobs
        # (padding rows point at blob 0 with weight 0): shapes repeat from step to step and stay cached.
        Q = 1024
        m = sel.numel()
        padn = (-m) % Q
        valid = torch.ones(m + padn, device=dev)
        if padn:
            sel = torch.cat([sel, torch.zeros(padn, dtype=sel.dtype, device=dev)])
            valid[m:] = 0
        step = max(Q, (chunk_px // (P * P)) // Q * Q)
        for k in range(0, sel.numel(), step):
            i = sel[k:k + step]
            vk = valid[k:k + step]
            px = torch.floor(cx[i]).detach().long().unsqueeze(1) + ox
            py = torch.floor(cy[i]).detach().long().unsqueeze(1) + oy
            dx = px.float() + 0.5 - cx[i].unsqueeze(1)
            dy = py.float() + 0.5 - cy[i].unsqueeze(1)
            q = A[i, None] * dx * dx + 2 * Bc[i, None] * dx * dy + C[i, None] * dy * dy
            fall = torch.exp(-0.5 * q)
            if hard > 0:
                fall = fall * (1 - hard) + hard * (q < 1.0).float()
            inside = ((px >= 0) & (px < W) & (py >= 0) & (py < H)).float()
            w = (a[i] * vk)[:, None] * fall * inside
            idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
            val = torch.cat([w.unsqueeze(2) * rgb[i].unsqueeze(1), w.unsqueeze(2)], 2).reshape(-1, 4)
            acc = acc.index_add(0, idx, val)
    return acc


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.permute(2, 0, 1).unsqueeze(0); y = y.permute(2, 0, 1).unsqueeze(0)
    g = torch.exp(-(torch.arange(11, device=x.device) - 5.0) ** 2 / 4.5); g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda t: F.conv2d(t, k, padding=5, groups=3)
    mx, my = f(x), f(y)
    vx, vy, vxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    return (((2 * mx * my + 1e-4) * (2 * vxy + 9e-4)) / ((mx * mx + my * my + 1e-4) * (vx + vy + 9e-4))).mean()


STAGES = [(0.12, "Finding the light"), (0.30, "Finding the shape"), (0.60, "Finding the details"), (1.01, "Finding the texture")]


def stage_name(f: float) -> str:
    for edge, name in STAGES:
        if f < edge:
            return name
    return STAGES[-1][1]


def run(image: str | None, out: str, fmt: str = "16:9", seconds: float = 24, fps: int = 30, steps: int = 1500,
        nmax: int = 20000, n0: int = 800, train_width: int = 320, aspect: float = 0.8, focus=(0.5, 0.42),
        zoom_at=(0.5, 0.38), accent: str = "#e9c46a", sound: bool = True, title: str = "GRADIENT DESCENT",
        subtitle: str = "for a picture", dev: str | None = None, seed: int = 7, log=print) -> dict:
    d = device(dev)
    torch.manual_seed(seed)
    lay = Layout(fmt, aspect, accent)
    PW, PH = lay.out_size
    TW = train_width; TH = int(round(TW / aspect))
    ref_full = core.load_image(image, max(PW, TW * 2), aspect, focus)
    ref_panel = core.resize(ref_full, PW, PH)
    tgt = {w: torch.from_numpy(core.resize(ref_full, w, int(round(w / aspect))).astype(np.float32) / 255).to(d)
           for w in (TW // 4, TW // 2, TW)}
    t_full = tgt[TW]
    border = torch.cat([t_full[0], t_full[-1], t_full[:, 0], t_full[:, -1]])
    bg = border.mean(0)
    b = Blobs(nmax, n0, d, seed)
    opt = torch.optim.Adam([{"params": [b.pos], "lr": 0.003}, {"params": [b.lsig], "lr": 0.02},
                            {"params": [b.rot], "lr": 0.02}, {"params": [b.col], "lr": 0.03},
                            {"params": [b.amp], "lr": 0.03}], betas=(0.9, 0.99))
    base_lr = [g["lr"] for g in opt.param_groups]

    # film timeline
    T_INTRO, T_HOLD, T_ZOOM, T_TITLE = 1.6, 2.6, 3.4, 3.2
    t_train = max(4.0, seconds - T_INTRO - T_HOLD - T_ZOOM - T_TITLE)
    n_train_frames = int(t_train * fps)
    plan = core.schedule(steps, n_train_frames, k=3.2)
    want: dict[int, int] = {}
    for st in plan:
        want[st] = want.get(st, 0) + 1
    caps = [(0.2, T_INTRO + 2.2, "It starts as noise. It has never seen this picture."),
            (T_INTRO + t_train * 0.35, T_INTRO + t_train * 0.62, "Every step, it fixes its biggest mistake.")]
    curve, events, err_track = [], [], []
    t0 = time.time()

    def n_target(f: float) -> int:
        pts = [(0.0, n0), (0.12, n0 * 3), (0.30, int(nmax * 0.25)), (0.75, nmax), (1.0, nmax)]
        for (f0, a0), (f1, a1) in zip(pts, pts[1:]):
            if f <= f1:
                return int(a0 + (a1 - a0) * (f - f0) / (f1 - f0))
        return nmax

    @torch.no_grad()
    def eval_loss() -> float:
        r = render(b, TW // 2, TH // 2 if False else int(round(TW / 2 / aspect)), bg=bg)
        return (r - tgt[TW // 2]).abs().mean().item()

    @torch.no_grad()
    def panel(view=(0, 0, 1, 1), hard=0.0, shrink=1.0) -> np.ndarray:
        return core.to_u8(render(b, PW, PH, view, bg=bg, hard=hard, shrink=shrink).cpu().numpy())

    with core.Film(out, lay.W, lay.H, fps) as film:
        def emit(step: int, count: int):
            loss = eval_loss()
            curve.append(loss)
            f = step / steps
            img = panel()
            for _ in range(count):
                tt = film.seconds
                cap, ca = core.caption_at(tt, caps)
                hud = Hud(title=title, stage="Everything is random" if step == 0 else stage_name(f),
                          stats=[("STEP", core.fmt_int(step)), ("LOSS", f"{loss:.4f}"), ("BLOBS", core.fmt_int(b.n))],
                          caption=cap, caption_alpha=ca, curve=curve, curve_total=len(want) + 1,
                          left_label="REFERENCE", right_label="CODE  ·  GRADIENT DESCENT")
                film.add(lay.frame(ref_panel, img, hud))
            err_track.append(loss)

        # intro: hold the noise
        loss0 = eval_loss()
        hud0 = Hud(title=title, stage="Everything is random", stats=[("STEP", "0"), ("LOSS", f"{loss0:.4f}"), ("BLOBS", core.fmt_int(b.n))])
        img0 = panel()
        for i in range(int(T_INTRO * fps)):
            cap, ca = core.caption_at(film.seconds, caps)
            hud0.caption, hud0.caption_alpha = cap, ca
            film.add(lay.frame(ref_panel, img0, hud0))
        if 0 in want:
            emit(0, want.pop(0))
        for step in range(1, steps + 1):
            f = step / steps
            W = TW // 4 if f < 0.10 else TW // 2 if f < 0.28 else TW
            H = int(round(W / aspect))
            decay = 1.0 if f < 0.55 else 0.15 ** ((f - 0.55) / 0.45)
            for gp, lr in zip(opt.param_groups, base_lr):
                gp["lr"] = lr * decay
            img = render(b, W, H, bg=bg)
            tg = tgt[W]
            loss = (img - tg).abs().mean()
            if W >= TW // 2:
                loss = 0.8 * loss + 0.2 * (1 - ssim(img, tg))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                b.lsig.clamp_(Blobs.LS_MIN, Blobs.LS_MAX); b.pos.clamp_(-0.1, 1.1)
                if step >= 40 and step % 20 == 0 and b.n < nmax:
                    k = min(n_target(f), nmax) - b.n
                    if k > 0:
                        err = (img.detach() - tg).abs().mean(2).reshape(-1) ** 1.5 + 1e-6
                        ind = torch.multinomial(err.cpu(), k, replacement=True).to(d)
                        n = b.n
                        b.pos[n:n + k, 0] = ((ind % W).float() + torch.rand(k, device=d)) / W
                        b.pos[n:n + k, 1] = ((ind // W).float() + torch.rand(k, device=d)) / H
                        b.lsig[n:n + k] = math.log(1.3 / W) + torch.randn(k, 2, device=d) * 0.2
                        b.rot[n:n + k] = torch.rand(k, device=d) * math.pi
                        b.col[n:n + k] = torch.logit(img.detach().reshape(-1, 3)[ind].clamp(0.02, 0.98))  # starts as what is already painted
                        b.amp[n:n + k] = -1.2
                        b.n = n + k
                        events.append(film.seconds)
            if step in want:
                emit(step, want.pop(step))
            if step % 200 == 0:
                log(f"  step {step:5d}/{steps}  W {W}  blobs {b.n:6d}  loss {curve[-1] if curve else 0:.4f}  {time.time() - t0:6.1f}s")
        train_s = time.time() - t0
        with torch.no_grad():
            final = render(b, TW, TH, bg=bg)
            mse = ((final - t_full) ** 2).mean().item()
        psnr = -10 * math.log10(max(mse, 1e-10))
        stats = [("STEPS", core.fmt_int(steps)), ("BLOBS", core.fmt_int(b.n)), ("PSNR", f"{psnr:.1f} dB")]
        final_img = panel()
        # hold: which one is the photo
        for i in range(int(T_HOLD * fps)):
            u = i / (T_HOLD * fps)
            film.add(lay.frame(ref_panel, final_img, Hud(title=title, stage="Converged", stats=stats, curve=curve,
                     caption="Which one is the photo?", caption_alpha=min(1, u * 4, (1 - u) * 4),
                     show_labels=not (0.25 < u < 0.8))))
        # zoom proof: soft blobs -> hard discs
        zx, zy = zoom_at
        nZ = int(T_ZOOM * fps)
        for i in range(nZ):
            u = i / max(1, nZ - 1)
            z = 1 + 6 * core.ease(min(1, u / 0.45))
            hw, hh = 0.5 / z, 0.5 / z
            cxv = 0.5 + (zx - 0.5) * core.ease(min(1, u / 0.45)); cyv = 0.5 + (zy - 0.5) * core.ease(min(1, u / 0.45))
            view = (cxv - hw, cyv - hh, cxv + hw, cyv + hh)
            hard = core.ease((u - 0.55) / 0.3)
            img = panel(view, hard=hard, shrink=1 - 0.3 * hard)
            rx0, ry0 = int(view[0] * ref_full.shape[1]), int(view[1] * ref_full.shape[0])
            rx1, ry1 = int(view[2] * ref_full.shape[1]), int(view[3] * ref_full.shape[0])
            rcrop = core.resize(ref_full[max(0, ry0):ry1, max(0, rx0):rx1], PW, PH)
            cap = "Zoom in." if u < 0.5 else "It is all blobs."
            film.add(lay.frame(rcrop, img, Hud(title=title, stage="Proof", stats=stats, caption=cap,
                     caption_alpha=min(1, (u % 0.5) * 8, (0.5 - u % 0.5) * 8))))
        nT = int(T_TITLE * fps)
        lines = [f"{b.n:,} blobs  ·  {steps:,} steps of gradient descent  ·  {train_s / 60:.1f} min on {d.type.upper()}",
                 "No image model. Every frame is the real training run."]
        for i in range(nT):
            film.add(lay.title_card(title, subtitle, lines, alpha=min(1, i / (0.5 * fps), (nT - i) / (0.4 * fps))))
        dur = film.seconds
    if sound:
        e = np.array(err_track) if err_track else np.array([1.0, 0.0])
        e = (e - e.min()) / max(1e-9, e.max() - e.min())
        lead = int(T_INTRO / dur * 200); tail = 200 - lead - len(e) * 0
        env = np.concatenate([np.ones(lead), np.interp(np.linspace(0, 1, 140), np.linspace(0, 1, len(e)), e), np.zeros(60)])
        from . import sound as snd
        wav = out[:-4] + ".wav"
        snd.write_wav(wav, snd.score(dur, env, events=events[::3], swell_at=dur - T_TITLE - T_ZOOM))
        core.mux_audio(out, wav)
        os.remove(wav)
    return {"out": out, "blobs": b.n, "steps": steps, "psnr": round(psnr, 2), "train_seconds": round(train_s, 1),
            "device": d.type, "seconds": round(dur, 2)}
