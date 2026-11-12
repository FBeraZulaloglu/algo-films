"""Lab: new images from a trained blob painting, without any generative model.

  styles  - the same photo as triangle vector art (PNG + SVG), string art, and a low-blob "painting"
  edits   - a trained blob painting re-coloured (duotone palettes) and exploded / reassembled
  poses   - blobs trained on photo A keep training on photo B; interpolating the two parameter sets gives
            in-between frames that were never photographed

Usage:
  python -m algofilms.lab styles photo.jpg --out out/lab
  python -m algofilms.lab edits  photo.jpg --out out/lab
  python -m algofilms.lab poses  a.jpg b.jpg --out out/lab"""
from __future__ import annotations

import argparse
import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
from PIL import Image

from . import core


# ----------------------------------------------------------------------------------------------- blob fitting
def fit(target: np.ndarray, steps: int = 1200, nmax: int = 15000, n0: int = 800, blobs=None, dev=None, log=print,
        densify: bool = True, lr_scale: float = 1.0):
    """Fits blobs to target (float HxWx3). If `blobs` is given, training continues from them (same blob identities)."""
    import torch
    from .splat import Blobs, device, render, ssim
    d = device(dev)
    TH, TW = target.shape[:2]
    aspect = TW / TH
    tgt = {w: torch.from_numpy(core.resize(target, w, int(round(w / aspect))).astype(np.float32) / 255).to(d)
           for w in (TW // 4, TW // 2, TW)}
    t_full = tgt[TW]
    bg = torch.cat([t_full[0], t_full[-1], t_full[:, 0], t_full[:, -1]]).mean(0)
    fresh = blobs is None
    b = blobs or Blobs(nmax, n0, d)
    nmax = b.nmax
    opt = torch.optim.Adam([{"params": [b.pos], "lr": 0.003 * lr_scale}, {"params": [b.lsig], "lr": 0.02 * lr_scale},
                            {"params": [b.rot], "lr": 0.02 * lr_scale}, {"params": [b.col], "lr": 0.03 * lr_scale},
                            {"params": [b.amp], "lr": 0.03 * lr_scale}], betas=(0.9, 0.99))
    base = [g["lr"] for g in opt.param_groups]
    t0 = time.time()
    for step in range(1, steps + 1):
        f = step / steps
        W = TW if not fresh else (TW // 4 if f < 0.10 else TW // 2 if f < 0.28 else TW)
        H = tgt[W].shape[0]
        decay = 1.0 if f < 0.55 else 0.15 ** ((f - 0.55) / 0.45)
        for gp, lr in zip(opt.param_groups, base):
            gp["lr"] = lr * decay
        img = render(b, W, H, bg=bg)
        loss = (img - tgt[W]).abs().mean()
        if W >= TW // 2:
            loss = 0.8 * loss + 0.2 * (1 - ssim(img, tgt[W]))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            b.lsig.clamp_(Blobs.LS_MIN, Blobs.LS_MAX); b.pos.clamp_(-0.1, 1.1)
            if densify and fresh and step >= 40 and step % 20 == 0 and b.n < nmax:
                pts = [(0.0, n0), (0.12, n0 * 3), (0.30, int(nmax * 0.25)), (0.75, nmax), (1.0, nmax)]
                want = nmax
                for (f0, a0), (f1, a1) in zip(pts, pts[1:]):
                    if f <= f1:
                        want = int(a0 + (a1 - a0) * (f - f0) / (f1 - f0)); break
                k = min(want, nmax) - b.n
                if k > 0:
                    err = (img.detach() - tgt[W]).abs().mean(2).reshape(-1) ** 1.5 + 1e-6
                    ind = torch.multinomial(err.cpu(), k, replacement=True).to(d)
                    n = b.n
                    b.pos[n:n + k, 0] = ((ind % W).float() + torch.rand(k, device=d)) / W
                    b.pos[n:n + k, 1] = ((ind // W).float() + torch.rand(k, device=d)) / H
                    b.lsig[n:n + k] = math.log(1.3 / W) + torch.randn(k, 2, device=d) * 0.2
                    b.rot[n:n + k] = torch.rand(k, device=d) * math.pi
                    b.col[n:n + k] = torch.logit(img.detach().reshape(-1, 3)[ind].clamp(0.02, 0.98))
                    b.amp[n:n + k] = -1.2
                    b.n = n + k
        if step % 200 == 0:
            log(f"  step {step}/{steps}  blobs {b.n}  loss {loss.item():.4f}  {time.time() - t0:.0f}s")
    return b, bg


def state(b):
    return [p.detach().clone() for p in b.params]


def set_state(b, s):
    import torch
    with torch.no_grad():
        for p, v in zip(b.params, s):
            p.copy_(v)


def lerp_state(sa, sb, t: float):
    """Blend two parameter sets of the same blobs. Angles take the short way round (they are mod pi)."""
    import torch
    out = [a + (c - a) * t for a, c in zip(sa, sb)]
    dr = torch.remainder(sb[2] - sa[2] + math.pi / 2, math.pi) - math.pi / 2
    out[2] = sa[2] + dr * t
    return out


def shot(b, bg, W: int, H: int, **kw) -> np.ndarray:
    import torch
    from .splat import render
    with torch.no_grad():
        return core.to_u8(render(b, W, H, bg=bg, **kw).cpu().numpy())


def save(img: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path, quality=93)
    return path


# ----------------------------------------------------------------------------------------------- 1 styles
def triangles_svg(shapes, base, W: int, H: int, path: str):
    rgb = lambda c: "#%02x%02x%02x" % tuple(int(np.clip(v, 0, 1) * 255 + 0.5) for v in c)
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n')
        f.write(f'<rect width="{W}" height="{H}" fill="{rgb(base)}"/>\n')
        for tri, col, a in shapes:
            pts = " ".join(f"{x * W:.1f},{y * H:.1f}" for x, y in tri)
            f.write(f'<polygon points="{pts}" fill="{rgb(col)}" fill-opacity="{a:.2f}"/>\n')
        f.write("</svg>\n")


def styles(image: str, out: str, focus=(0.5, 0.45), aspect: float = 0.8, log=print) -> dict:
    from . import shapes as shp, strings as st
    res = {}
    W, H = 1080, int(1080 / aspect)
    # triangles: 150 (bold) and 500 (detailed), PNG + SVG
    tgt = core.load_image(image, 200, aspect, focus)
    for n in (150, 500):
        sh, curve, base = shp.evolve(tgt, n, log=lambda s: None)
        res[f"triangles_{n}"] = save(shp.draw(sh, base, W, H), f"{out}/1-triangles-{n}.jpg")
        triangles_svg(sh, base, W, H, f"{out}/1-triangles-{n}.svg")
        log(f"  triangles {n}: mse {curve[-1]:.5f}")
    # string art still
    g_rgb = core.load_image(image, 400, 1.0, (focus[0], focus[1] - 0.06))
    g = g_rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.percentile(g, [2, 98]); g = np.clip((g - lo) / max(1e-6, hi - lo), 0, 1)
    seq, curve, P = st.solve(g, 260, 4000)
    S = 1400
    board = st.Board(S, ink=0.30 * (3000 / len(seq)) ** 0.5)
    Pf = P * S / 400
    for a, b_ in zip(seq, seq[1:]):
        board.add_line(Pf[a], Pf[b_])
    img = board.image()
    yy, xx = np.mgrid[:S, :S]
    img[((xx - S / 2) ** 2 + (yy - S / 2) ** 2) > (S / 2) ** 2] = 0
    res["strings"] = save(img, f"{out}/1-string-art.jpg")
    with open(f"{out}/1-string-art_nails.txt", "w") as f:
        f.write(" ".join(map(str, seq)) + "\n")
    log(f"  string art: {len(seq) - 1} lines")
    # low-blob paintings
    tgt = core.load_image(image, 320, aspect, focus)
    for n, steps in ((300, 500), (1500, 700)):
        b, bg = fit(tgt, steps=steps, nmax=n, n0=min(n, 300), log=lambda s: None)
        res[f"blobs_{n}"] = save(shot(b, bg, W, H), f"{out}/1-blobs-{n}.jpg")
        res[f"discs_{n}"] = save(shot(b, bg, W, H, hard=1.0, shrink=0.8), f"{out}/1-discs-{n}.jpg")
        log(f"  blobs {n} done")
    return res


# ----------------------------------------------------------------------------------------------- 2 edits
PALETTES = {
    "red-black": [(8, 8, 10), (120, 14, 28), (215, 38, 61), (245, 236, 224)],
    "gold-night": [(6, 10, 24), (38, 52, 96), (201, 152, 62), (250, 240, 214)],
    "acid": [(10, 10, 10), (26, 60, 52), (190, 255, 60), (250, 255, 235)],
    "riso": [(24, 30, 90), (240, 70, 90), (255, 190, 60), (252, 246, 232)],
}


def recolour(b, bg, name: str):
    """Maps every blob's brightness onto a palette ramp. Only the colour numbers change; shapes stay learned."""
    import torch
    ramp = torch.tensor(PALETTES[name], dtype=torch.float32, device=b.dev) / 255
    def remap(rgb):
        l = (rgb * torch.tensor([0.299, 0.587, 0.114], device=b.dev)).sum(-1).clamp(0, 1) * (len(ramp) - 1)
        i = l.floor().long().clamp(0, len(ramp) - 2); f = (l - i).unsqueeze(-1)
        return ramp[i] * (1 - f) + ramp[i + 1] * f
    s = state(b)
    with torch.no_grad():
        b.col.copy_(torch.logit(remap(torch.sigmoid(b.col)).clamp(0.004, 0.996)))
    return s, remap(bg.unsqueeze(0))[0]


def edits(image: str, out: str, focus=(0.5, 0.45), aspect: float = 0.8, steps: int = 1200, nmax: int = 15000,
          fps: int = 30, log=print, trained=None) -> dict:
    import torch
    res = {}
    W, H = 1080, int(1080 / aspect)
    if trained is None:
        tgt = core.load_image(image, 320, aspect, focus)
        b, bg = fit(tgt, steps=steps, nmax=nmax, log=log)
    else:
        b, bg = trained
    res["base"] = save(shot(b, bg, W, H), f"{out}/2-base.jpg")
    for name in PALETTES:
        s, bg2 = recolour(b, bg, name)
        res[name] = save(shot(b, bg2, W, H), f"{out}/2-palette-{name}.jpg")
        set_state(b, s)
    # explode and reassemble: each blob flies out along its own direction, spins, and comes home
    s0 = state(b)
    g = torch.Generator().manual_seed(4)
    ang = torch.rand(b.nmax, generator=g) * 2 * math.pi
    dist = (0.15 + 0.55 * torch.rand(b.nmax, generator=g))
    off = (torch.stack([torch.cos(ang), torch.sin(ang)], 1) * dist[:, None]).to(b.dev)
    spin = ((torch.rand(b.nmax, generator=g) - 0.5) * 6).to(b.dev)
    delay = (torch.rand(b.nmax, generator=g) * 0.35).to(b.dev)
    VW, VH = 720, int(720 / aspect)
    path = f"{out}/2-explode.mp4"
    n = int(5.0 * fps)
    with core.Film(path, VW, VH, fps) as film:
        for i in range(n):
            u = i / (n - 1)
            k = math.sin(math.pi * u) ** 1.5                    # 0 -> 1 -> 0
            with torch.no_grad():
                e = (k * 1.35 - delay).clamp(0, 1); e = e * e * (3 - 2 * e)
                b.pos.copy_(s0[0] + off * e[:, None]); b.rot.copy_(s0[2] + spin * e)
                b.lsig.copy_(s0[1] + 0.5 * e[:, None])
            film.add(shot(b, bg, VW, VH))
    set_state(b, s0)
    res["explode"] = path
    return res


# ----------------------------------------------------------------------------------------------- 3 poses
def poses(image_a: str, image_b: str, out: str, focus_a=(0.5, 0.45), focus_b=(0.5, 0.45), aspect: float = 0.8,
          steps: int = 1200, steps_b: int = 700, nmax: int = 15000, fps: int = 30, log=print):
    """Train on A, then keep training the same blobs on B. Returns stills at 0/25/50/75/100% and a loop video."""
    res = {}
    W, H = 1080, int(1080 / aspect)
    ta = core.load_image(image_a, 320, aspect, focus_a); tb = core.load_image(image_b, 320, aspect, focus_b)
    log("  fitting pose A ...")
    b, bg_a = fit(ta, steps=steps, nmax=nmax, log=log)
    sa = state(b)
    log("  continuing the same blobs on pose B ...")
    b, bg_b = fit(tb, steps=steps_b, blobs=b, log=log, lr_scale=0.6)
    sb = state(b)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        set_state(b, lerp_state(sa, sb, t))
        res[f"t{int(t * 100):03d}"] = save(shot(b, bg_a + (bg_b - bg_a) * t, W, H), f"{out}/3-pose-{int(t * 100):03d}.jpg")
    VW, VH = 720, int(720 / aspect)
    path = f"{out}/3-pose-morph.mp4"
    n = int(3.0 * fps)
    with core.Film(path, VW, VH, fps) as film:
        hold = int(0.7 * fps)
        seq = [0.0] * hold + [core.ease(i / (n - 1)) for i in range(n)] + [1.0] * hold + [core.ease(1 - i / (n - 1)) for i in range(n)]
        for t in seq:
            set_state(b, lerp_state(sa, sb, t))
            film.add(shot(b, bg_a + (bg_b - bg_a) * t, VW, VH))
    res["morph"] = path
    set_state(b, sa)
    return res, (b, bg_a)


def _pair(s):
    a, c = s.split(","); return float(a), float(c)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="algofilms.lab")
    ap.add_argument("cmd", choices=["styles", "edits", "poses", "all"])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="out/lab")
    ap.add_argument("--focus", type=_pair, default=(0.5, 0.45))
    ap.add_argument("--focus-b", type=_pair, default=None)
    a = ap.parse_args(argv)
    r = {}
    if a.cmd in ("styles", "all"):
        r["styles"] = styles(a.images[0], a.out, a.focus)
    trained = None
    if a.cmd in ("poses", "all"):
        r["poses"], trained = poses(a.images[0], a.images[1], a.out, a.focus, a.focus_b or a.focus)
    if a.cmd in ("edits", "all"):
        r["edits"] = edits(a.images[0], a.out, a.focus, trained=trained)
    import json
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
