"""Pixel morph: every pixel of one picture flies to a new place and together they become another picture.
Nothing is added, nothing is recoloured: it is a pure rearrangement (a permutation) of the source pixels.

The assignment is solved by matching brightness ranks, then improved with thousands of vectorised pair swaps that
lower colour error + travel distance. The film flies each pixel along a curved path with a staggered start.

numpy + Pillow only."""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw

from . import core
from .core import Hud, Layout


def text_image(text: str, w: int, h: int, accent=(233, 196, 106)) -> np.ndarray:
    """A simple poster to morph into: light serif text on a dark vertical gradient."""
    im = Image.new("RGB", (w * 4, h * 4))
    g = np.linspace(0, 1, h * 4)[:, None, None]
    base = (np.array([12, 12, 16]) * (1 - g) + np.array([46, 30, 22]) * g).astype(np.uint8)
    im = Image.fromarray(np.broadcast_to(base, (h * 4, w * 4, 3)).copy())
    d = ImageDraw.Draw(im)
    lines = text.split("\\n") if "\\n" in text else text.split("\n")
    size = int(min(w * 4 * 0.9 / max(len(l) for l in lines) * 1.6, h * 4 * 0.7 / len(lines)))
    f = core.font("title", size)
    total = len(lines) * size * 1.05
    for i, ln in enumerate(lines):
        d.text((w * 2, h * 2 - total / 2 + (i + 0.5) * size * 1.05), ln, font=f, fill=accent, anchor="mm")
    return np.asarray(im.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255


def assign(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src, dst: (N,3) colours. Brightness-rank matching. Returns perm: destination slot t takes source pixel perm[t]."""
    N = len(src)
    lum = lambda c: c @ np.array([0.299, 0.587, 0.114])
    perm = np.empty(N, int)
    perm[np.argsort(lum(dst), kind="stable")] = np.argsort(lum(src), kind="stable")
    return perm


def refine(perm: np.ndarray, src_col: np.ndarray, dst_col: np.ndarray, src_xy: np.ndarray, dst_xy: np.ndarray,
           lam: float = 0.06, rounds: int = 600, seed: int = 3) -> tuple[np.ndarray, list[float]]:
    """Pair-swap descent. cost(slot t <- pixel p) = |colour(p) - target(t)|^2 + lam * |xy(p) - xy(t)|^2."""
    rng = np.random.default_rng(seed)
    N = len(perm)
    W = np.array([1.0, 1.25, 0.8])                       # perceptual-ish weights (green matters most)

    def cost(p, t):
        dc = ((src_col[p] - dst_col[t]) ** 2 * W).sum(-1)
        dd = ((src_xy[p] - dst_xy[t]) ** 2).sum(-1)
        return dc + lam * dd

    hist = []
    order = np.argsort(dst_xy[:, 1] * 10 + dst_xy[:, 0])
    for r in range(rounds):
        if r % 2 == 0:
            sh = rng.permutation(N)                       # random global pairs (fix colours)
        else:
            k = rng.integers(1, 40)                       # nearby slots in scan order (fix local structure)
            sh = np.roll(order, k)
        a, b = sh[: N // 2 * 2: 2], sh[1: N // 2 * 2: 2]
        before = cost(perm[a], a) + cost(perm[b], b)
        after = cost(perm[b], a) + cost(perm[a], b)
        good = after < before
        pa = perm[a[good]].copy()
        perm[a[good]] = perm[b[good]]; perm[b[good]] = pa
        if r % 20 == 0:
            hist.append(float(cost(perm, np.arange(N)).mean()))
    return perm, hist


def run(source: str | None, out: str, target: str | None = None, text: str = "HELLO", fmt: str = "9:16",
        seconds: float = 16, fps: int = 30, grid: int = 140, aspect: float = 0.8, accent: str = "#e9c46a",
        sound: bool = True, focus=(0.5, 0.42), back: bool = True, log=print) -> dict:
    gw = grid; gh = int(round(gw / aspect))
    A = core.load_image(source, gw, aspect, focus)
    B = core.load_image(target, gw, aspect, (0.5, 0.5)) if target else text_image(text, gw, gh, core.hex_rgb(accent))
    A = core.resize(A, gw, gh).astype(np.float32) / 255; B = core.resize(B, gw, gh).astype(np.float32) / 255
    N = gw * gh
    yy, xx = np.mgrid[:gh, :gw]
    xy = np.stack([xx.reshape(-1) / gw, yy.reshape(-1) / gw], 1)
    cA, cB = A.reshape(-1, 3), B.reshape(-1, 3)
    log(f"  {N:,} pixels, solving the rearrangement ...")
    perm = assign(cA, cB)
    perm, hist = refine(perm, cA, cB, xy, xy)
    err = float(np.abs(cA[perm] - cB).mean())
    lay = Layout(fmt, aspect, accent, reference=True)
    PW, PH = lay.out_size
    cell = PW / gw
    k = int(math.ceil(cell)) + 1
    src_px = xy[perm] * cell * gw
    dst_px = xy * cell * gw
    col = (cA[perm] * 255).astype(np.uint8)
    rng = np.random.default_rng(5)
    dist = np.hypot(*(dst_px - src_px).T)
    delay = 0.55 * (src_px[:, 1] / PH) * 0.6 + 0.4 * rng.random(N) * 0.4    # top rows leave first, with jitter
    delay = (delay - delay.min()) / (delay.max() - delay.min()) * 0.45
    perp = np.stack([-(dst_px - src_px)[:, 1], (dst_px - src_px)[:, 0]], 1) / (dist[:, None] + 1e-6)
    swirl = rng.uniform(-0.35, 0.35, N)[:, None] * dist[:, None]
    oy, ox = np.mgrid[:k, :k]

    def paint(u: float) -> np.ndarray:
        img = np.zeros((PH + k, PW + k, 3), np.uint8)
        local = np.clip((u - delay) / 0.55, 0, 1)
        e = local * local * (3 - 2 * local)
        p = src_px + (dst_px - src_px) * e[:, None] + perp * swirl * np.sin(np.pi * e)[:, None]
        order = np.argsort(local, kind="stable")       # moving pixels drawn on top
        x0 = np.clip(p[order, 0].astype(int), 0, PW); y0 = np.clip(p[order, 1].astype(int), 0, PH)
        c = col[order]
        kk = int(math.ceil(cell))
        for dy in range(kk):
            for dx in range(kk):
                img[y0 + dy, x0 + dx] = c
        return img[:PH, :PW]

    T_A, T_M, T_B, T_TITLE = 2.0, seconds * 0.32, 2.4, 2.8
    T_BACK = seconds * 0.22 if back else 0
    target_ref = core.resize(B, PW, PH)
    events = []
    with core.Film(out, lay.W, lay.H, fps) as film:
        def hud(stage, cap="", ca=0.0, moved=0.0):
            return Hud(title="PIXEL MORPH", stage=stage, caption=cap, caption_alpha=ca,
                       stats=[("PIXELS", f"{N:,}"), ("MOVED", f"{moved * 100:3.0f}%"), ("ADDED", "0")],
                       left_label="TARGET", right_label="THE SAME PIXELS")
        start = paint(0)
        for i in range(int(T_A * fps)):
            u = i / (T_A * fps)
            film.add(lay.frame(target_ref, start, hud("The original", f"Every pixel of this picture ...", min(1, u * 4, (1 - u) * 6), 0)))
        nM = int(T_M * fps)
        for i in range(nM):
            u = i / (nM - 1)
            if i % 3 == 0:
                events.append(film.seconds)
            moved = float(np.clip((u - delay) / 0.55, 0, 1).mean())
            film.add(lay.frame(target_ref, paint(u), hud("Rearranging", "", 0, moved)))
        end = paint(1.0)
        for i in range(int(T_B * fps)):
            u = i / (T_B * fps)
            film.add(lay.frame(target_ref, end, hud("Rearranged", "... moved. Nothing added, nothing recoloured.", min(1, u * 4, (1 - u) * 4), 1)))
        if back:
            nK = int(T_BACK * fps)
            for i in range(nK):
                u = 1 - i / (nK - 1)
                film.add(lay.frame(target_ref, paint(u), hud("And back", "", 0, u)))
        nT = int(T_TITLE * fps)
        lns = [f"{N:,} pixels  ·  one permutation  ·  mean colour error {err * 255:.0f}/255",
               "Rank matching + pair-swap descent. No image model."]
        for i in range(nT):
            film.add(lay.title_card("PIXEL MORPH", "the same pixels, moved", lns, alpha=min(1, i / (0.5 * fps), (nT - i) / (0.4 * fps))))
        dur = film.seconds
    if sound:
        from . import sound as snd
        tl = np.linspace(0, dur, 300)
        env = np.where(tl < T_A, 0.9, np.where(tl < T_A + T_M, 0.9 - 0.9 * (tl - T_A) / T_M, 0.0))
        wav = out[:-4] + ".wav"
        snd.write_wav(wav, snd.score(dur, env, events=events, chord="maj9"))
        core.mux_audio(out, wav); os.remove(wav)
    return {"out": out, "pixels": N, "mean_error": round(err, 4), "seconds": round(dur, 2)}
