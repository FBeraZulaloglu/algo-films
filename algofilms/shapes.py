"""Shape evolution: a photo rebuilt from a few hundred translucent triangles, one at a time, by trial and error.

Each new triangle starts as a batch of random guesses; the best one is then mutated (hill climbing) and kept only if
it lowers the error. Its colour is solved exactly (least squares) for the pixels it covers. The film adds the shapes
one by one and redraws them at film resolution, so the result is sharp vector art at any size.

numpy + OpenCV."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from . import core
from .core import Hud, Layout


def _mask(tri: np.ndarray, w: int, h: int):
    """tri in px (3,2). Returns bbox (x0, y0, x1, y1) and a boolean mask inside it, or None."""
    x0, y0 = np.floor(tri.min(0)).astype(int); x1, y1 = np.ceil(tri.max(0)).astype(int) + 1
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round((tri - [x0, y0]) * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
    if not m.any():
        return None
    return (x0, y0, x1, y1), m.astype(bool)


def _score(tri, cur, tgt, alpha):
    """Error change if this triangle is added with its best colour. Returns (delta, colour)."""
    h, w = cur.shape[:2]
    r = _mask(tri, w, h)
    if r is None:
        return 0.0, None
    (x0, y0, x1, y1), m = r
    c = cur[y0:y1, x0:x1][m]; t = tgt[y0:y1, x0:x1][m]
    col = np.clip(((t - (1 - alpha) * c) / alpha).mean(0), 0, 1)        # least-squares colour for this alpha
    new = (1 - alpha) * c + alpha * col
    return float(((new - t) ** 2).sum() - ((c - t) ** 2).sum()), col


def evolve(tgt: np.ndarray, n_shapes: int = 300, guesses: int = 80, mutations: int = 120, alpha: float = 0.55,
           seed: int = 1, log=print):
    rng = np.random.default_rng(seed)
    h, w = tgt.shape[:2]
    cur = np.broadcast_to(tgt.reshape(-1, 3).mean(0), tgt.shape).copy()
    shapes, curve = [], []
    total = lambda: float(((cur - tgt) ** 2).mean())
    curve.append(total())
    for k in range(n_shapes):
        size = max(4.0, 0.5 * min(w, h) * (1 - k / n_shapes) ** 1.5)    # big shapes first, small later
        best, bcol, bd = None, None, 0.0
        err = ((cur - tgt) ** 2).sum(2).reshape(-1)
        for _ in range(guesses):
            i = rng.choice(len(err), p=err / err.sum()) if rng.random() < 0.7 else rng.integers(len(err))
            c = np.array([i % w, i // w], float)
            tri = c + rng.normal(0, size / 2, (3, 2))
            dlt, col = _score(tri, cur, tgt, alpha)
            if col is not None and dlt < bd:
                best, bcol, bd = tri, col, dlt
        if best is None:
            continue
        for _ in range(mutations):
            tri = best.copy()
            tri[rng.integers(3)] += rng.normal(0, max(1.0, size / 6), 2)
            dlt, col = _score(tri, cur, tgt, alpha)
            if col is not None and dlt < bd:
                best, bcol, bd = tri, col, dlt
        (x0, y0, x1, y1), m = _mask(best, w, h)
        reg = cur[y0:y1, x0:x1]
        reg[m] = (1 - alpha) * reg[m] + alpha * bcol
        shapes.append((best / [w, h], bcol, alpha))
        curve.append(total())
        if (k + 1) % 100 == 0:
            log(f"  {k + 1}/{n_shapes} shapes  mse {curve[-1]:.5f}")
    return shapes, curve, tgt.reshape(-1, 3).mean(0)


def draw(shapes, base, W: int, H: int, ss: int = 2) -> np.ndarray:
    img = np.empty((H * ss, W * ss, 3), np.float32); img[:] = base
    for tri, col, a in shapes:
        p = np.round(tri * [W * ss, H * ss] * 16).astype(np.int32)
        layer = img.copy()
        cv2.fillPoly(layer, [p], tuple(float(c) for c in col), lineType=cv2.LINE_AA, shift=4)
        img = cv2.addWeighted(layer, a, img, 1 - a, 0)
    return cv2.resize(core.to_u8(img), (W, H), interpolation=cv2.INTER_AREA)


def run(image: str | None, out: str, fmt: str = "9:16", seconds: float = 18, fps: int = 30, shapes: int = 300,
        work_width: int = 160, aspect: float = 0.8, accent: str = "#e9c46a", sound: bool = True, focus=(0.5, 0.42),
        log=print) -> dict:
    tgt = core.load_image(image, work_width, aspect, focus)
    log(f"  evolving {shapes} triangles ...")
    sh, curve, base = evolve(tgt, shapes, log=log)
    lay = Layout(fmt, aspect, accent)
    PW, PH = lay.out_size
    ref = core.resize(core.load_image(image, PW, aspect, focus), PW, PH)
    T_INTRO, T_HOLD, T_TITLE = 1.4, 2.6, 3.0
    t_ev = max(4.0, seconds - T_INTRO - T_HOLD - T_TITLE)
    plan = core.schedule(len(sh), int(t_ev * fps), k=2.2)
    caps = [(0.2, T_INTRO + 1.8, "Start with one flat colour."),
            (T_INTRO + t_ev * 0.3, T_INTRO + t_ev * 0.65, "Each triangle is a random guess, kept only if it helps.")]
    # incremental drawing at film res
    ss = 2
    img = np.empty((PH * ss, PW * ss, 3), np.float32); img[:] = base
    events = []

    def add_shape(k):
        nonlocal img
        tri, col, a = sh[k]
        p = np.round(tri * [PW * ss, PH * ss] * 16).astype(np.int32)
        x0, y0 = np.maximum(0, p.min(0) // 16 - 2); x1, y1 = p.max(0) // 16 + 3
        x1, y1 = min(PW * ss, x1), min(PH * ss, y1)
        if x1 <= x0 or y1 <= y0:
            return
        reg = img[y0:y1, x0:x1]; layer = reg.copy()
        cv2.fillPoly(layer, [p - [x0 * 16, y0 * 16]], tuple(float(c) for c in col), lineType=cv2.LINE_AA, shift=4)
        img[y0:y1, x0:x1] = cv2.addWeighted(layer, a, reg, 1 - a, 0)

    def view():
        return cv2.resize(core.to_u8(img), (PW, PH), interpolation=cv2.INTER_AREA)

    def hud(k, stage, cap="", ca=0.0, labels=True):
        return Hud(title="SHAPE EVOLUTION", stage=stage, caption=cap, caption_alpha=ca, show_labels=labels,
                   stats=[("TRIANGLES", f"{k:,}"), ("ERROR", f"{curve[k] * 1000:.2f}")],
                   curve=curve[:k + 1] if k > 0 else None, curve_total=len(curve), right_label="TRIANGLES")

    with core.Film(out, lay.W, lay.H, fps) as film:
        v = view()
        for i in range(int(T_INTRO * fps)):
            cap, ca = core.caption_at(film.seconds, caps)
            film.add(lay.frame(ref, v, hud(0, "One colour", cap, ca)))
        done = 0
        for st in plan:
            while done < st:
                add_shape(done); done += 1
                if done < 120 or done % 8 == 0:
                    events.append(film.seconds)
            cap, ca = core.caption_at(film.seconds, caps)
            f = done / len(sh)
            stage = "Big blocks" if f < 0.15 else "The silhouette" if f < 0.45 else "The face" if f < 0.8 else "Small corrections"
            film.add(lay.frame(ref, view(), hud(done, stage, cap, ca)))
        v = view()
        for i in range(int(T_HOLD * fps)):
            u = i / (T_HOLD * fps)
            film.add(lay.frame(ref, v, hud(done, "Finished", f"{done} triangles.", min(1, u * 4, (1 - u) * 4))))
        nT = int(T_TITLE * fps)
        lns = [f"{done} translucent triangles  ·  random guesses + hill climbing", "No image model. Every shape was earned by lowering the error."]
        for i in range(nT):
            film.add(lay.title_card("EVOLUTION", "a face from triangles", lns, alpha=min(1, i / (0.5 * fps), (nT - i) / (0.4 * fps))))
        dur = film.seconds
    if sound:
        from . import sound as snd
        e = np.array(curve); e = (e - e.min()) / max(1e-9, e.max() - e.min())
        lead = int(T_INTRO / dur * 200)
        env = np.concatenate([np.ones(lead), np.interp(np.linspace(0, 1, 140), np.linspace(0, 1, len(e)), e ** 0.5), np.zeros(60)])
        wav = out[:-4] + ".wav"
        snd.write_wav(wav, snd.score(dur, env, events=events, chord="sus"))
        core.mux_audio(out, wav); os.remove(wav)
    return {"out": out, "shapes": done, "mse": round(curve[-1], 5), "seconds": round(dur, 2)}
