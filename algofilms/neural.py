"""Neural race: small neural networks race to memorise one photo, live, side by side.

Each network is a function f(x, y) -> colour with no image inside it. Three designs are trained on the same photo:
  * plain ReLU MLP          - famously stays blurry ("spectral bias")
  * ReLU + Fourier features - random sine/cosine inputs let it learn fine detail
  * SIREN                   - sine activations everywhere
The finale zooms into the photo: the photo turns into pixels, the network keeps painting smooth detail, because it
is a function, not a grid.

PyTorch on MPS / CUDA / CPU."""
from __future__ import annotations

import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from . import core
from .core import FORMATS, INK, DIM, Hud
from .splat import device


class ReLUNet(nn.Module):
    def __init__(self, width=256, depth=4, fourier: int = 0, scale: float = 10.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("Bm", torch.randn(2, fourier, generator=g) * scale if fourier else torch.zeros(2, 0))
        d_in = 2 * fourier if fourier else 2
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)
        self.fourier = fourier

    def forward(self, xy):
        if self.fourier:
            p = 2 * math.pi * xy @ self.Bm
            xy = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return torch.sigmoid(self.net(xy))


class Sine(nn.Module):
    def __init__(self, d_in, d_out, w0=30.0, first=False):
        super().__init__()
        self.l, self.w0 = nn.Linear(d_in, d_out), w0
        with torch.no_grad():
            b = 1 / d_in if first else math.sqrt(6 / d_in) / w0
            self.l.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.w0 * self.l(x))


class Siren(nn.Module):
    def __init__(self, width=256, depth=4, w0=30.0):
        super().__init__()
        layers = [Sine(2, width, w0, first=True)] + [Sine(width, width, w0) for _ in range(depth - 1)]
        self.net = nn.Sequential(*layers, nn.Linear(width, 3))

    def forward(self, xy):
        return self.net(xy * 2 - 1) * 0.5 + 0.5


ARCHS = {   # name, builder, learning rate
    "relu": ("PLAIN RELU", lambda: ReLUNet(fourier=0), 1e-3),
    "fourier": ("FOURIER FEATURES", lambda: ReLUNet(fourier=128, scale=12.0), 1e-3),
    "siren": ("SIREN", lambda: Siren(), 1e-4),
}


def grid(w, h, view=(0, 0, 1, 1), dev="cpu"):
    u0, v0, u1, v1 = view
    ys = (torch.arange(h, device=dev) + 0.5) / h * (v1 - v0) + v0
    xs = (torch.arange(w, device=dev) + 0.5) / w * (u1 - u0) + u0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], -1).reshape(-1, 2)


@torch.no_grad()
def paint(net, w, h, view=(0, 0, 1, 1), dev="cpu", chunk=200_000) -> np.ndarray:
    xy = grid(w, h, view, dev)
    out = torch.cat([net(xy[i:i + chunk]) for i in range(0, len(xy), chunk)])
    return core.to_u8(out.reshape(h, w, 3).clamp(0, 1).cpu().numpy())


def boxes(fmt: str, aspect: float, n: int):
    """2x2 grid (reference + 3 nets) for every format; 1x4 row in 16:9."""
    W, H = FORMATS[fmt]
    if fmt == "16:9":
        top, bot = int(H * 0.16), int(H * 0.2)
        ph = H - top - bot; pw = int(ph * aspect)
        gap = 28
        while n * pw + (n - 1) * gap > W * 0.94:
            pw -= 4; ph = int(pw / aspect)
        x0 = (W - n * pw - (n - 1) * gap) // 2
        y0 = (H - ph) // 2 - int(H * 0.02)
        return [(x0 + i * (pw + gap), y0, pw, ph) for i in range(n)]
    top, bot = int(H * (0.12 if fmt == "9:16" else 0.13)), int(H * (0.16 if fmt == "9:16" else 0.12))
    gap = 26
    cols = 2; rows = math.ceil(n / cols)
    ph = (H - top - bot - (rows - 1) * gap) // rows; pw = int(ph * aspect)
    while cols * pw + gap > W * 0.92:
        pw -= 4; ph = int(pw / aspect)
    x0 = (W - cols * pw - gap) // 2; y0 = top + ((H - top - bot) - rows * ph - (rows - 1) * gap) // 2
    return [(x0 + (i % cols) * (pw + gap), y0 + (i // cols) * (ph + gap), pw, ph) for i in range(n)]


def compose(fmt, bx, panels, labels, subs, hud: Hud, accent, highlight=None) -> np.ndarray:
    W, H = FORMATS[fmt]
    canvas = np.zeros((H, W, 3), np.uint8)
    for (x, y, w, h), p in zip(bx, panels):
        canvas[y:y + h, x:x + w] = p if p.shape[:2] == (h, w) else core.resize(p, w, h)
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
    s = W / 1920 if fmt == "16:9" else W / 1080 * 0.9
    lf = core.font("mono", int(13 * s * 1.35)); vf = core.font("mono", int(20 * s))
    acc = core.hex_rgb(accent)
    for i, ((x, y, w, h), lab, sub) in enumerate(zip(bx, labels, subs)):
        d.text((x, y - int(24 * s)), core.spaced(lab), font=lf, fill=acc if highlight == i else DIM)
        if sub:
            d.text((x + w, y + h + int(10 * s)), sub, font=vf, fill=INK, anchor="ra")
    tf = core.font("sans", int(26 * s)); sf = core.font("mono", int(14 * s * 1.2))
    d.text((W // 2, int(H * 0.035)), hud.title, font=tf, fill=INK, anchor="mt")
    d.text((W // 2, int(H * 0.035) + int(40 * s)), core.spaced(hud.stage), font=sf, fill=acc, anchor="mt")
    if hud.stats:
        txt = "     ".join(f"{k} {v}" for k, v in hud.stats)
        d.text((W // 2, H - int(H * 0.035) - int(70 * s)), txt, font=core.font("mono", int(18 * s)), fill=DIM, anchor="ms")
    if hud.caption and hud.caption_alpha > 0:
        d.text((W // 2, H - int(H * 0.035)), hud.caption, font=core.font("serif", int(40 * s)),
               fill=tuple(int(c * hud.caption_alpha) for c in INK), anchor="ms")
    return np.asarray(im)


def run(image: str | None, out: str, fmt: str = "16:9", seconds: float = 22, fps: int = 30, steps: int = 2000,
        archs=("relu", "fourier", "siren"), train_width: int = 256, aspect: float = 0.8, focus=(0.5, 0.42),
        zoom_at=(0.44, 0.33), accent: str = "#e9c46a", sound: bool = True, dev: str | None = None, lr: float = 1e-3,
        batch: int = 16384, log=print) -> dict:
    d = device(dev)
    torch.manual_seed(0)
    TW, TH = train_width, int(round(train_width / aspect))
    target_np = core.load_image(image, TW, aspect, focus)
    target = torch.from_numpy(target_np).to(d).reshape(-1, 3)
    xy = grid(TW, TH, dev=d)
    nets = [ARCHS[a][1]().to(d) for a in archs]
    opts = [torch.optim.Adam(n.parameters(), lr=ARCHS[a][2] * lr / 1e-3) for n, a in zip(nets, archs)]
    names = ["THE PHOTO"] + [ARCHS[a][0] for a in archs]
    nparams = [sum(p.numel() for p in n.parameters()) for n in nets]
    bx = boxes(fmt, aspect, 1 + len(nets))
    PW, PH = bx[0][2], bx[0][3]
    ref_big = core.load_image(image, TW, aspect, focus)          # the photo at its real (training) resolution
    ref_panel = core.resize(ref_big, PW, PH)
    T_INTRO, T_ZOOM, T_TITLE = 1.6, 4.5, 3.0
    t_train = max(4.0, seconds - T_INTRO - T_ZOOM - T_TITLE)
    plan = core.schedule(steps, int(t_train * fps), k=3.0)
    want: dict[int, int] = {}
    for st in plan:
        want[st] = want.get(st, 0) + 1
    psnr = [0.0] * len(nets)
    curve = []
    events = []
    t0 = time.time()

    @torch.no_grad()
    def measure():
        for i, n in enumerate(nets):
            mse = ((n(xy) - target) ** 2).mean().item()
            psnr[i] = -10 * math.log10(max(mse, 1e-10))

    cache = {}

    def frame(step, cap="", ca=0.0, stage="Training"):
        if cache.get("step") != step:
            cache["step"] = step
            cache["panels"] = [ref_panel] + [paint(n, PW, PH, dev=d) for n in nets]
        subs = [""] + [f"{p:5.1f} dB" for p in psnr]
        best = 1 + int(np.argmax(psnr)) if step > 0 else None
        hud = Hud(title="NEURAL RACE", stage=stage, caption=cap, caption_alpha=ca,
                  stats=[("STEP", f"{step:,}"), ("INPUT", "(x, y)"), ("OUTPUT", "colour")])
        return compose(fmt, bx, cache["panels"], names, subs, hud, accent, best)

    with core.Film(out, *FORMATS[fmt], fps) as film:
        measure()
        nI = int(T_INTRO * fps)
        for i in range(nI):
            u = i / nI
            film.add(frame(0, "Three networks. One photo. No pixels inside them.", min(1, u * 4, (1 - u) * 4), "Untrained"))
        caps = [(T_INTRO + t_train * 0.15, T_INTRO + t_train * 0.45, "Each one only learns: (x, y) in, colour out."),
                (T_INTRO + t_train * 0.6, T_INTRO + t_train * 0.95, "Same data, same steps. Different maths.")]
        if 0 in want:
            film.add(frame(0), want.pop(0))
        gen = torch.Generator().manual_seed(1)
        for step in range(1, steps + 1):
            sel = torch.randint(0, len(xy), (batch,), generator=gen).to(d)     # random mini-batch of pixels
            bxy, btg = xy[sel], target[sel]
            for n, o in zip(nets, opts):
                o.zero_grad(set_to_none=True)
                loss = ((n(bxy) - btg) ** 2).mean()
                loss.backward()
                o.step()
            if step in want:
                measure()
                curve.append(list(psnr))
                cap, ca = core.caption_at(film.seconds, caps)
                film.add(frame(step, cap, ca), want.pop(step))
                if len(curve) % 10 == 0:
                    events.append(film.seconds)
            if step % 250 == 0:
                log(f"  step {step}/{steps}  PSNR " + "  ".join(f"{a}={p:.1f}" for a, p in zip(archs, psnr)) + f"  {time.time() - t0:.0f}s")
        train_s = time.time() - t0
        # zoom finale: the photo becomes pixels, the networks keep painting
        zx, zy = zoom_at
        nZ = int(T_ZOOM * fps)
        for i in range(nZ):
            u = core.ease(min(1, i / (nZ * 0.6)))
            z = 1 + 9 * u
            hw = 0.5 / z
            cx = 0.5 + (zx - 0.5) * u; cy = 0.5 + (zy - 0.5) * u
            view = (cx - hw, cy - hw * 1.0, cx + hw, cy + hw * 1.0)
            # photo: crop its real pixels and enlarge with NEAREST, so the pixel grid shows
            x0, y0 = int(view[0] * TW), int(view[1] * TH); x1, y1 = max(x0 + 1, int(math.ceil(view[2] * TW))), max(y0 + 1, int(math.ceil(view[3] * TH)))
            crop = ref_big[max(0, y0):y1, max(0, x0):x1]
            ph = core.resize(crop, PW, PH, smooth=False)
            panels = [ph] + [paint(n, PW, PH, view, dev=d) for n in nets]
            hud = Hud(title="NEURAL RACE", stage=f"Zoom {z:.0f}x",
                      caption="The photo is a grid. The networks are functions." if i > nZ * 0.35 else "Zoom in.",
                      caption_alpha=1.0, stats=[("STEP", f"{steps:,}"), ("ZOOM", f"{z:.1f}x")])
            film.add(compose(fmt, bx, panels, names, [""] + [f"{p:5.1f} dB" for p in psnr], hud, accent))
        nT = int(T_TITLE * fps)
        lay = core.Layout(fmt, aspect, accent)
        lns = ["  ·  ".join(f"{ARCHS[a][0].title()} {p:.1f} dB" for a, p in zip(archs, psnr)),
               f"{steps:,} steps each, {train_s / 60:.1f} min on {d.type.upper()}  ·  ~{nparams[-1] // 1000}k weights per network"]
        for i in range(nT):
            film.add(lay.title_card("NEURAL RACE", "who learns a face fastest", lns, alpha=min(1, i / (0.5 * fps), (nT - i) / (0.4 * fps))))
        dur = film.seconds
    if sound:
        from . import sound as snd
        best = np.array([max(c) for c in curve]) if curve else np.array([10.0, 30.0])
        e = 1 - (best - best.min()) / max(1e-9, best.max() - best.min())
        lead = int(T_INTRO / dur * 200)
        env = np.concatenate([np.ones(lead), np.interp(np.linspace(0, 1, 140), np.linspace(0, 1, len(e)), e), np.zeros(60)])
        wav = out[:-4] + ".wav"
        snd.write_wav(wav, snd.score(dur, env, events=events, chord="min9", swell_at=dur - T_TITLE - T_ZOOM))
        core.mux_audio(out, wav); os.remove(wav)
    return {"out": out, "psnr": {a: round(p, 2) for a, p in zip(archs, psnr)}, "train_seconds": round(train_s, 1),
            "device": d.type, "seconds": round(dur, 2)}
