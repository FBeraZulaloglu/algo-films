"""Fourier epicycles: a chain of spinning circles draws a word, a logo or the outline of a photo with one pen.

The outline is traced into one closed path, resampled, and turned into circles with the discrete Fourier transform
(each frequency = one circle: its size is the amplitude, its start angle the phase). Part 1 draws the shape with all
circles; part 2 shows how the drawing sharpens as the number of circles grows.

numpy + OpenCV + Pillow."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import core
from .core import Hud, Layout


# ----------------------------------------------------------------------------------------------- path extraction
def _contours_from_mask(mask: np.ndarray, min_len: int = 20) -> list[np.ndarray]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :].astype(np.float64) for c in cs if len(c) >= min_len]


def text_contours(text: str, size: int = 900) -> list[np.ndarray]:
    f = core.font("title", 400)
    probe = Image.new("L", (10, 10)); bb = ImageDraw.Draw(probe).textbbox((0, 0), text, font=f)
    w, h = bb[2] - bb[0] + 80, bb[3] - bb[1] + 80
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((40 - bb[0], 40 - bb[1]), text, font=f, fill=255)
    return _contours_from_mask(np.asarray(im) > 127)


def image_contours(path: str | None, size: int = 700, mode: str = "auto", focus=(0.5, 0.42)) -> list[np.ndarray]:
    src = core.load_image(path, size, 1.0, focus)
    g = (core.to_u8(src) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    if mode == "silhouette":            # dark/opaque logo on light background
        return _contours_from_mask(g < 128, 30)
    g = cv2.GaussianBlur(g, (5, 5), 1.4)
    v = np.median(g)
    edges = cv2.Canny(g, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cs = sorted([c[:, 0, :].astype(np.float64) for c in cs if len(c) >= 40], key=len, reverse=True)
    return cs[:60]


def chain(contours: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Joins contours into one closed path (greedy nearest neighbour). Returns points + a 'pen up' flag per point
    (True on the connecting jumps, drawn faintly)."""
    rest = [c for c in contours]
    path, jump = [rest.pop(0)], [np.zeros(len(contours[0]), bool)]
    while rest:
        end = path[-1][-1]
        best, bi, bk, rev = 1e18, 0, 0, False
        for i, c in enumerate(rest):
            d = np.hypot(*(c - end).T)
            k = int(np.argmin(d))
            if d[k] < best:
                best, bi, bk = d[k], i, k
        c = rest.pop(bi)
        c = np.roll(c, -bk, 0)                            # start the closed contour at its nearest point
        c = np.vstack([c, c[:1]])
        bridge = np.linspace(end, c[0], max(2, int(best / 3)))[1:-1]
        path += [bridge, c]; jump += [np.ones(len(bridge), bool), np.zeros(len(c), bool)]
    first = path[0][0]; end = path[-1][-1]
    bridge = np.linspace(end, first, max(2, int(np.hypot(*(first - end)) / 3)))[1:-1]
    path.append(bridge); jump.append(np.ones(len(bridge), bool))
    return np.vstack(path), np.concatenate(jump)


def resample(pts: np.ndarray, flags: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], m, endpoint=False)
    x = np.interp(t, s, pts[:, 0]); y = np.interp(t, s, pts[:, 1])
    fl = flags[np.clip(np.searchsorted(s, t), 0, len(flags) - 1)]
    return x + 1j * y, fl


# ----------------------------------------------------------------------------------------------- fourier
def fourier(z: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns frequencies and coefficients sorted by amplitude (largest first, the k=0 centre term first of all)."""
    m = len(z)
    C = np.fft.fft(z) / m
    freqs = np.fft.fftfreq(m, 1 / m).astype(int)
    keep = np.abs(freqs) <= k_max
    freqs, C = freqs[keep], C[keep]
    order = np.argsort(-np.abs(C) * (freqs != 0))
    order = np.concatenate([np.where(freqs == 0)[0], [o for o in order if freqs[o] != 0]])
    return freqs[order], C[order]


def evaluate(freqs, C, t: np.ndarray, n: int | None = None) -> np.ndarray:
    n = n or len(C)
    return (C[:n, None] * np.exp(2j * np.pi * freqs[:n, None] * t[None, :])).sum(0)


# ----------------------------------------------------------------------------------------------- film
def run(out: str, text: str | None = None, image: str | None = None, fmt: str = "9:16", seconds: float = 20, fps: int = 30,
        circles: int = 300, points: int = 4096, accent: str = "#e9c46a", sound: bool = True, mode: str = "auto",
        log=print) -> dict:
    cs = text_contours(text) if text else image_contours(image, mode=mode)
    if not cs:
        raise SystemExit("No outline found in the input.")
    pts, fl = chain(cs)
    z, jump = resample(pts, fl, points)
    z = z - z.mean()
    freqs, C = fourier(z, circles)
    n = len(C)
    log(f"  {len(cs)} outlines -> one path of {points} points -> {n} circles")
    lay = Layout(fmt, 1.0 if not text else 1.6 if fmt == "16:9" else 1.0, accent, reference=False)
    PW, PH = lay.out_size
    ext = max(np.ptp(z.real), np.ptp(z.imag) * PW / PH)
    sc = 0.86 * PW / ext
    to_px = lambda w: np.stack([PW / 2 + w.real * sc, PH / 2 + w.imag * sc], -1)
    acc = core.hex_rgb(accent)
    SS = 2                                                     # supersample for smooth thin lines
    T_DRAW, T_GROW, T_TITLE = seconds * 0.55, seconds * 0.30, max(2.5, seconds * 0.15)
    full_t = np.linspace(0, 1, points * 2, endpoint=False)
    full_px = to_px(evaluate(freqs, C, full_t))
    full_jump = np.repeat(jump, 2)

    def canvas():
        return np.zeros((PH * SS, PW * SS, 3), np.uint8)

    def polyline(img, P, jumpmask, col, faint, th=1):
        P = (P * SS * 16).astype(np.int32)
        if len(P) < 2:
            return
        # split into drawn / jump runs
        start = 0
        for i in range(1, len(P) + 1):
            if i == len(P) or jumpmask[i] != jumpmask[start]:
                seg = P[max(0, start - 1):i]
                c = faint if jumpmask[start] else col
                if len(seg) >= 2:
                    cv2.polylines(img, [seg], False, c, th * SS, cv2.LINE_AA, shift=4)
                start = i

    def down(img):
        return cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)

    events = []
    with core.Film(out, lay.W, lay.H, fps) as film:
        nD = int(T_DRAW * fps)
        for i in range(nD):
            u = core.ease(i / (nD - 1)) if i < nD - 1 else 1.0
            img = canvas()
            k = int(u * len(full_t))
            polyline(img, full_px[:k + 1], full_jump[:k + 1], acc, (70, 66, 50), 2)
            # circles chain at time u
            terms = C * np.exp(2j * np.pi * freqs * u)
            centers = np.concatenate([[0], np.cumsum(terms)])
            cp = to_px(centers)
            show = min(n, 120)
            for j in range(1, show):
                r = abs(terms[j]) * sc
                if r * SS < 1.5:
                    continue
                cv2.circle(img, tuple((cp[j] * SS * 16).astype(int)), int(r * SS * 16), (55, 55, 60), 1, cv2.LINE_AA, shift=4)
            cv2.polylines(img, [(cp[:show + 1] * SS * 16).astype(np.int32)], False, (200, 196, 188), SS, cv2.LINE_AA, shift=4)
            tip = cp[-1]
            cv2.circle(img, tuple((tip * SS * 16).astype(int)), 5 * SS * 16, acc, -1, cv2.LINE_AA, shift=4)
            hud = Hud(title="EPICYCLES", stage="Drawing with circles", stats=[("CIRCLES", f"{n:,}"), ("TURN", f"{u * 100:4.0f}%")],
                      caption="Every circle spins at its own speed." if 0.1 < i / nD < 0.5 else "", caption_alpha=1.0,
                      show_labels=False)
            film.add(lay.frame(None, down(img), hud))
        final = canvas()
        polyline(final, full_px, full_jump, acc, (70, 66, 50), 2)
        nG = int(T_GROW * fps)
        steps_n = sorted(set(int(round(2 ** (1 + (math.log2(n) - 1) * x))) for x in np.linspace(0, 1, 12)))
        for i in range(nG):
            u = i / max(1, nG - 1)
            m = steps_n[min(len(steps_n) - 1, int(u * len(steps_n)))]
            if not events or events[-1][1] != m:
                events.append((film.seconds, m))
            img = canvas()
            P = to_px(evaluate(freqs, C, full_t, m))
            polyline(img, P, full_jump, (230, 226, 216), (60, 58, 52), 2)
            hud = Hud(title="EPICYCLES", stage="How many circles do you need?", stats=[("CIRCLES", f"{m:,}"), ("OF", f"{n:,}")],
                      caption=f"{m} circles", caption_alpha=1.0, show_labels=False)
            film.add(lay.frame(None, down(img), hud))
        nT = int(T_TITLE * fps)
        lns = [f"{n:,} circles  ·  one pen  ·  the discrete Fourier transform", "No image model. Pure maths, drawn in code."]
        for i in range(nT):
            film.add(lay.title_card("EPICYCLES", "everything is circles", lns, alpha=min(1, i / (0.5 * fps), (nT - i) / (0.4 * fps))))
        dur = film.seconds
    if sound:
        from . import sound as snd
        env = np.concatenate([np.linspace(1, 0.5, int(T_DRAW / dur * 200)), np.linspace(0.5, 0, int(T_GROW / dur * 200)), np.zeros(40)])
        wav = out[:-4] + ".wav"
        snd.write_wav(wav, snd.score(dur, env, events=[e[0] for e in events], chord="maj9"))
        core.mux_audio(out, wav); os.remove(wav)
    return {"out": out, "circles": n, "outlines": len(cs), "seconds": round(dur, 2)}
