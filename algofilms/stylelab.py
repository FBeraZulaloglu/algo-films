"""Style lab: one photo, many looks, all classic algorithms (no image model), plus edits that are only possible
once the photo has been turned into blobs (move, recolour and reshape parts of it by changing numbers).

  python -m algofilms.stylelab photo.jpg [pose_b.jpg] [extra.jpg ...] --out out/stylelab

Writes: a contact sheet of every style, the single stills, blob edits, and a 9:16 reel that wipes through them."""
from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import core

PAPER = np.array([238, 232, 220], np.float32)
INKC = np.array([20, 18, 22], np.float32)


def lum(img):
    return (img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255


# ----------------------------------------------------------------------------------------------- styles
def duotone(img, dark=(10, 8, 12), light=(215, 38, 61), paper=(246, 238, 226)):
    l = lum(img)[..., None]
    d, m, p = (np.array(c, np.float32) for c in (dark, light, paper))
    out = np.where(l < 0.5, d + (m - d) * (l / 0.5), m + (p - m) * ((l - 0.5) / 0.5))
    return out.astype(np.uint8)


def _dots(l, cell, angle):
    h, w = l.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    c, s = math.cos(angle), math.sin(angle)
    u, v = xx * c + yy * s, -xx * s + yy * c
    du, dv = (u % cell) - cell / 2, (v % cell) - cell / 2
    r = cell * 0.72 * np.sqrt(np.clip(1 - l, 0, 1))
    return np.clip((r - np.hypot(du, dv)) + 0.5, 0, 1)          # soft edge = anti-aliasing


def halftone(img, cell=11):
    l = cv2.GaussianBlur(lum(img), (0, 0), cell / 2.5)
    l = np.clip((l - 0.02) / 0.75, 0, 1) ** 0.55          # lift shadows so dark clothes keep texture
    a = _dots(l, cell, math.radians(45))[..., None]
    return (PAPER * (1 - a) + INKC * a).astype(np.uint8)


def riso(img, cell=9):
    f = img.astype(np.float32) / 255
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), cell / 2.5)
    a = _dots(np.clip(blur(lum(img)) / 0.7, 0, 1) ** 0.45, cell, math.radians(75))[..., None]          # blue: shadows
    b = _dots(blur(1 - np.clip(f[..., 0] - f[..., 2] * 0.6, 0, 1)), cell, math.radians(15))[..., None]  # red: warm skin
    blue, red = np.array([40, 60, 150], np.float32) / 255, np.array([238, 72, 80], np.float32) / 255
    out = (PAPER / 255) * (1 - a * (1 - blue)) * (1 - b * 0.75 * (1 - red))
    return core.to_u8(out)


def sketch(img):
    g = lum(img) * 255
    inv = cv2.GaussianBlur(255 - g, (0, 0), 14)
    d = np.clip(g * 255 / np.maximum(255 - inv, 1), 0, 255) / 255
    d = d ** 1.6
    return (PAPER * d[..., None] + INKC * (1 - d[..., None])).astype(np.uint8)


def toon(img, levels=5):
    x = img.copy()
    for _ in range(4):
        x = cv2.bilateralFilter(x, 9, 60, 9)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = (np.floor(lab[..., 0] / 255 * levels) + 0.5) / levels * 255
    x = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    g = cv2.medianBlur((lum(img) * 255).astype(np.uint8), 7)
    e = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 6)
    e = cv2.GaussianBlur(e, (0, 0), 0.8).astype(np.float32)[..., None] / 255
    return (x * e).astype(np.uint8)


def kuwahara(img, r=11):
    """Oil-paint look: each pixel takes the mean of whichever neighbouring quadrant is the calmest."""
    f = img.astype(np.float32)
    g = lum(img)
    k = (r + 1, r + 1)
    best_v, out = None, None
    for ax, ay in ((0, 0), (r, 0), (0, r), (r, r)):                # four quadrant boxes around the pixel
        m = cv2.boxFilter(f, -1, k, anchor=(ax, ay), borderType=cv2.BORDER_REFLECT)
        gm = cv2.boxFilter(g, -1, k, anchor=(ax, ay), borderType=cv2.BORDER_REFLECT)
        v = cv2.boxFilter(g * g, -1, k, anchor=(ax, ay), borderType=cv2.BORDER_REFLECT) - gm * gm
        if out is None:
            best_v, out = v, m
        else:
            sel = v < best_v
            out = np.where(sel[..., None], m, out); best_v = np.where(sel, v, best_v)
    return np.clip(out, 0, 255).astype(np.uint8)


def pixel_art(img, width=64, colours=14):
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, int(round(width * h / w))), interpolation=cv2.INTER_AREA)
    z = small.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(1)
    _, lab, cen = cv2.kmeans(z, colours, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    q = cen[lab.ravel()].reshape(small.shape).astype(np.uint8)
    return cv2.resize(q, (w, h), interpolation=cv2.INTER_NEAREST)


def glitch(img, seed=3):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], int(w * 0.012), 1)
    out[..., 2] = np.roll(img[..., 2], -int(w * 0.012), 1)
    for _ in range(14):
        y = rng.integers(0, h - 10); hh = rng.integers(6, max(8, h // 22)); dx = rng.integers(-w // 9, w // 9)
        out[y:y + hh] = np.roll(out[y:y + hh], dx, 1)
    out = out.astype(np.float32)
    out[::3] *= 0.72
    return np.clip(out, 0, 255).astype(np.uint8)


def ascii_art(img, cols=90):
    h, w = img.shape[:2]
    cw = w / cols; ch = cw * 1.9
    rows = int(h / ch)
    small = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)
    l = lum(small); l = np.clip((l - l.min()) / max(1e-6, np.percentile(l, 99) - l.min()), 0, 1) ** 0.6
    ramp = " .,:;i1tfLCG08@"
    f = core.font("mono", int(ch * 0.95))
    im = Image.new("RGB", (w, h), (6, 6, 8)); d = ImageDraw.Draw(im)
    for r in range(rows):
        for c in range(cols):
            chx = ramp[min(len(ramp) - 1, int(l[r, c] * len(ramp)))]
            if chx != " ":
                col = tuple(int(v) for v in np.clip(small[r, c].astype(np.float32) * 1.6 + 70, 0, 255))
                d.text((c * cw, r * ch), chx, font=f, fill=col)
    return np.asarray(im)


def engrave(img, period=9):
    """Banknote look: wavy horizontal lines whose thickness follows the darkness."""
    l = cv2.GaussianBlur(lum(img), (0, 0), 2.0)
    h, w = l.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    y2 = yy + 5 * np.sin(xx / 46.0) + (l - 0.5) * 10          # lines bend around the face
    dist = np.abs((y2 % period) - period / 2)
    a = np.clip(((1 - l) ** 1.1 * period / 2 * 1.05 - dist) + 0.5, 0, 1)[..., None]
    return (PAPER * (1 - a) + np.array([24, 44, 36], np.float32) * a).astype(np.uint8)


def thermal(img):
    g = cv2.GaussianBlur((lum(img) * 255).astype(np.uint8), (0, 0), 3)
    return cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def lowpoly(img, n=350):
    from . import shapes
    h, w = img.shape[:2]
    small = cv2.resize(img, (200, int(200 * h / w)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sh, _, base = shapes.evolve(small, n, log=lambda s: None)
    return shapes.draw(sh, base, w, h)


def neon_edges(img):
    g = cv2.GaussianBlur((lum(img) * 255).astype(np.uint8), (0, 0), 1.6)
    e = cv2.Canny(g, 14, 44).astype(np.float32) / 255
    glow = cv2.GaussianBlur(e, (0, 0), 6) * 1.6 + cv2.GaussianBlur(e, (0, 0), 1.2)
    h = img.shape[0]
    grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    col = np.array([255, 60, 120], np.float32) * (1 - grad) + np.array([60, 200, 255], np.float32) * grad
    out = np.array([8, 6, 16], np.float32) + col * np.clip(glow, 0, 1)[..., None] + 255 * e[..., None] * 0.6
    return np.clip(out, 0, 255).astype(np.uint8)


STYLES = [
    ("ORIGINAL", lambda x: x), ("DUOTONE", duotone), ("HALFTONE", halftone), ("RISOGRAPH", riso),
    ("PENCIL", sketch), ("TOON", toon), ("OIL PAINT", kuwahara), ("PIXEL ART", pixel_art),
    ("ENGRAVING", engrave), ("NEON LINES", neon_edges), ("LOW POLY", lowpoly), ("ASCII", ascii_art),
    ("THERMAL", thermal), ("GLITCH", glitch),
]


def sheet(items, path, cols=4, tw=420, label=True):
    th = int(tw * items[0][1].shape[0] / items[0][1].shape[1])
    rows = math.ceil(len(items) / cols)
    pad = 34 if label else 0
    im = Image.new("RGB", (cols * tw, rows * (th + pad)), (0, 0, 0)); d = ImageDraw.Draw(im)
    f = core.font("mono", 18)
    for i, (name, a) in enumerate(items):
        x, y = (i % cols) * tw, (i // cols) * (th + pad)
        im.paste(Image.fromarray(a).resize((tw, th), Image.LANCZOS), (x, y))
        if label:
            d.text((x + 8, y + th + 7), core.spaced(name), font=f, fill=(200, 196, 188))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, quality=90)
    return path


# ----------------------------------------------------------------------------------------------- blob edits
def blob_edits(b, bg, W, H):
    """Edits made by changing blob numbers. Returns [(name, image)]. The blobs are restored afterwards."""
    import torch
    from .lab import set_state, shot, state
    s0 = state(b)
    n = b.n
    out = [("BLOBS (LEARNED)", shot(b, bg, W, H))]
    rgb = torch.sigmoid(s0[3]); l = (rgb * torch.tensor([0.299, 0.587, 0.114], device=b.dev)).sum(1)
    pos = s0[0]
    is_bg = ((rgb - bg).abs().max(1).values < 0.16) & (l > 0.35)
    is_shirt = (l < 0.16) & (pos[:, 1] > 0.66)

    def logit(c):
        return torch.logit(torch.tensor(c, device=b.dev).clamp(0.004, 0.996))

    with torch.no_grad():
        # 1 background swap: only blobs that look like the wall change colour (keeps their shading)
        shade = (l / l[is_bg].mean()).clamp(0.5, 1.3)[:, None]
        newc = (torch.tensor([0.78, 0.10, 0.18], device=b.dev) * shade).clamp(0.004, 0.996)
        b.col[is_bg] = torch.logit(newc[is_bg])
    red_bg = torch.tensor([0.78, 0.10, 0.18], device=b.dev)
    out.append(("BACKGROUND SWAPPED", shot(b, red_bg, W, H))); set_state(b, s0)
    with torch.no_grad():
        # 2 shirt recolour: dark blobs in the lower part get a new hue, brightness kept
        k = (l / 0.10).clamp(0.3, 2.2)[:, None]
        newc = (torch.tensor([0.10, 0.34, 0.30], device=b.dev) * k).clamp(0.004, 0.996)
        b.col[is_shirt] = torch.logit(newc[is_shirt])
    out.append(("SHIRT RECOLOURED", shot(b, bg, W, H))); set_state(b, s0)
    # 3 pointillist: every blob becomes a small crisp dot
    out.append(("POINTILLIST", shot(b, bg, W, H, hard=1.0, shrink=0.7))); set_state(b, s0)
    with torch.no_grad():
        # 4 swirl: rotate positions around the face, more near the centre
        c = torch.tensor([0.5, 0.42], device=b.dev)
        d = pos - c; r = d.norm(dim=1)
        a = 1.4 * torch.exp(-(r / 0.22) ** 2)
        ca, sa = torch.cos(a), torch.sin(a)
        b.pos.copy_(c + torch.stack([d[:, 0] * ca - d[:, 1] * sa, d[:, 0] * sa + d[:, 1] * ca], 1))
        b.rot.copy_(s0[2] + a)
    out.append(("SWIRL", shot(b, bg, W, H))); set_state(b, s0)
    with torch.no_grad():
        # 5 melt: blobs slide down, the lower in the face the further
        g = torch.Generator().manual_seed(2)
        k = (torch.rand(b.nmax, generator=g).to(b.dev) ** 2) * (pos[:, 1].clamp(0, 1)) * 0.5
        b.pos[:, 1] = pos[:, 1] + k * (~is_bg).float()
        b.lsig[:, 1] = s0[1][:, 1] + k * 3 * (~is_bg).float()
        b.rot.copy_(torch.where(k > 0.02, torch.full_like(s0[2], math.pi / 2), s0[2]))
    out.append(("MELT", shot(b, bg, W, H))); set_state(b, s0)
    with torch.no_grad():
        # 6 big brush: fewer, larger strokes (keep every 6th blob and enlarge it)
        keep = (torch.arange(b.nmax, device=b.dev) % 6 == 0)
        b.amp[~keep] = -20
        b.lsig.copy_(s0[1] + 0.9)
    out.append(("BIG BRUSH", shot(b, bg, W, H))); set_state(b, s0)
    return out


# ----------------------------------------------------------------------------------------------- reel
def reel(frames_named, path, accent="#d7263d", fps=30, hold=1.25, wipe=0.45, title="ONE PHOTO", subtitle="every look is code"):
    """9:16 reel: a 4:5 panel that wipes from one look to the next with a slow push-in."""
    W, H = core.FORMATS["9:16"]
    pw = 1000; ph = int(pw * frames_named[0][1].shape[0] / frames_named[0][1].shape[1])
    x0, y0 = (W - pw) // 2, (H - ph) // 2 - 20
    acc = core.hex_rgb(accent)
    imgs = [(n, cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)) for n, a in frames_named]
    tf, sf, lf, cf = core.font("sans", 30), core.font("mono", 20), core.font("title", 64), core.font("mono", 22)
    nH, nW = int(hold * fps), int(wipe * fps)
    total = len(imgs) * (nH + nW)

    def zoom(a, k):
        z = 1 + 0.05 * k
        m = cv2.resize(a, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
        oy, ox = (m.shape[0] - ph) // 2, (m.shape[1] - pw) // 2
        return m[oy:oy + ph, ox:ox + pw]

    with core.Film(path, W, H, fps) as film:
        fi = 0
        for i, (name, a) in enumerate(imgs):
            nxt = imgs[(i + 1) % len(imgs)]
            for j in range(nH + nW):
                k = fi / total
                cur = zoom(a, k)
                edge = None
                if j >= nH and i < len(imgs) - 1:
                    u = core.ease((j - nH) / nW)
                    cut = int(pw * u)
                    cur = cur.copy(); cur[:, :cut] = zoom(nxt[1], k)[:, :cut]
                    edge = cut
                    label = nxt[0] if u > 0.5 else name
                else:
                    label = name
                canvas = np.zeros((H, W, 3), np.uint8)
                canvas[y0:y0 + ph, x0:x0 + pw] = cur
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                if edge is not None:
                    d.line([(x0 + edge, y0), (x0 + edge, y0 + ph)], fill=acc, width=4)
                d.text((W // 2, 110), title, font=tf, fill=core.INK, anchor="mm")
                d.text((W // 2, 158), core.spaced(subtitle), font=sf, fill=acc, anchor="mm")
                d.text((W // 2, y0 + ph + 90), label, font=lf, fill=core.INK, anchor="mm")
                d.text((W // 2, y0 + ph + 160), f"{i + 1:02d} / {len(imgs):02d}", font=cf, fill=core.DIM, anchor="mm")
                film.add(np.asarray(im)); fi += 1
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="algofilms.stylelab")
    ap.add_argument("images", nargs="+", help="main photo, optional second pose, optional extra photos")
    ap.add_argument("--out", default="out/stylelab")
    ap.add_argument("--focus", default="0.5,0.45")
    ap.add_argument("--no-blobs", action="store_true")
    a = ap.parse_args(argv)
    fx, fy = map(float, a.focus.split(","))
    W, H = 1080, 1350
    load = lambda p: core.to_u8(core.load_image(p, W, 0.8, (fx, fy)))
    main_img = load(a.images[0])
    named = []
    os.makedirs(a.out, exist_ok=True)
    for name, fn in STYLES:
        named.append((name, fn(main_img)))
        Image.fromarray(named[-1][1]).save(f"{a.out}/style-{len(named) - 1:02d}-{name.lower().replace(' ', '-')}.jpg", quality=92)
        print("  style", name, flush=True)
    sheet(named, f"{a.out}/sheet-styles.jpg", cols=5, tw=380)
    for k, p in enumerate(a.images[1:], 2):
        img = load(p)
        picks = [STYLES[i] for i in (0, 1, 3, 5, 8, 9, 10, 4)]
        sheet([(n, fn(img)) for n, fn in picks], f"{a.out}/sheet-photo{k}.jpg", cols=4, tw=380)
        print("  sheet for photo", k, flush=True)
    reel_items = [named[i] for i in (0, 1, 5, 6, 8, 3, 9, 10, 7, 11)]
    if not a.no_blobs:
        from . import lab
        tgt = core.load_image(a.images[0], 320, 0.8, (fx, fy))
        b, bg = lab.fit(tgt, steps=1200, nmax=15000, log=lambda s: print(s, flush=True))
        ed = blob_edits(b, bg, W, H)
        for i, (n, im) in enumerate(ed):
            Image.fromarray(im).save(f"{a.out}/blob-{i:02d}-{n.lower().replace(' ', '-').replace('(', '').replace(')', '')}.jpg", quality=92)
        sheet(ed, f"{a.out}/sheet-blob-edits.jpg", cols=4, tw=420)
        reel_items += ed[1:]
        if len(a.images) > 1:      # in-between expressions
            sa = lab.state(b)
            tb = core.load_image(a.images[1], 320, 0.8, (fx, fy))
            b, bg_b = lab.fit(tb, steps=700, blobs=b, lr_scale=0.6, log=lambda s: print(s, flush=True))
            sb = lab.state(b)
            mids = []
            for t in (0.0, 0.33, 0.66, 1.0):
                lab.set_state(b, lab.lerp_state(sa, sb, t))
                mids.append((f"EXPRESSION {int(t * 100)}%", lab.shot(b, bg + (bg_b - bg) * t, W, H)))
            sheet(mids, f"{a.out}/sheet-expressions.jpg", cols=4, tw=420)
            reel_items += mids[1:]
    reel(reel_items, f"{a.out}/style-reel-9x16.mp4")
    print("done", a.out)


if __name__ == "__main__":
    main()
