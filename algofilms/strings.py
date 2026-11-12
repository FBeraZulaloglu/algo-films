"""String art: one continuous thread wound around nails on a circle redraws a photo.

Greedy solver: from the current nail, pick the next nail whose straight line covers the most remaining darkness,
lighten those pixels, repeat. The film shows the thread being wound line by line. The nail sequence is written out
as a text file, so the piece can be built for real on a board.

numpy + Pillow only."""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw

from . import core
from .core import Hud, Layout


def nails_xy(n: int, size: int, frame: str = "circle") -> np.ndarray:
    if frame == "circle":
        a = np.arange(n) / n * 2 * math.pi - math.pi / 2
        r = size / 2 - 1
        return np.stack([size / 2 + r * np.cos(a), size / 2 + r * np.sin(a)], 1)
    per = n // 4; s = size - 1; t = np.arange(per) / per * s
    return np.concatenate([np.stack([t, np.zeros(per)], 1), np.stack([np.full(per, s), t], 1),
                           np.stack([s - t, np.full(per, s)], 1), np.stack([np.zeros(per), s - t], 1)])


def _line_idx(p, q, size):
    L = int(np.hypot(*(q - p))) + 1
    xs = np.linspace(p[0], q[0], L).round().astype(np.int32).clip(0, size - 1)
    ys = np.linspace(p[1], q[1], L).round().astype(np.int32).clip(0, size - 1)
    return ys * size + xs


def solve(gray: np.ndarray, n_nails: int = 240, n_lines: int = 3000, strength: float = 0.18, min_gap: int = 12,
          frame: str = "circle") -> tuple[list[int], list[float], np.ndarray]:
    """gray: float HxW (square) in [0,1], 1 = white. Returns nail sequence, residual curve, nail coords."""
    size = gray.shape[0]
    P = nails_xy(n_nails, size, frame)
    dark = (1 - gray).astype(np.float32).reshape(-1)
    if frame == "circle":
        yy, xx = np.mgrid[:size, :size]
        dark[((xx - size / 2) ** 2 + (yy - size / 2) ** 2).reshape(-1) > (size / 2) ** 2] = 0
    # per nail: every candidate line, concatenated, with offsets for reduceat
    cand, cat, offs, lens = [], [], [], []
    for i in range(n_nails):
        js = [j for j in range(n_nails) if min((i - j) % n_nails, (j - i) % n_nails) >= min_gap]
        lines = [_line_idx(P[i], P[j], size) for j in js]
        cand.append(np.array(js)); cat.append(np.concatenate(lines))
        o = np.cumsum([0] + [len(l) for l in lines[:-1]]); offs.append(o); lens.append(np.array([len(l) for l in lines]))
    seq, used, curve = [0], set(), []
    cur = 0
    total0 = dark.sum()
    for _ in range(n_lines):
        sc = np.add.reduceat(dark[cat[cur]], offs[cur]) / lens[cur]
        for k, j in enumerate(cand[cur]):
            if (min(cur, j), max(cur, j)) in used:
                sc[k] = -1
        k = int(np.argmax(sc))
        if sc[k] <= 0:
            break
        nxt = int(cand[cur][k])
        idx = cat[cur][offs[cur][k]:offs[cur][k] + lens[cur][k]]
        dark[idx] = np.maximum(0, dark[idx] - strength)
        used.add((min(cur, nxt), max(cur, nxt)))
        seq.append(nxt); cur = nxt
        curve.append(float(dark.sum() / total0))
    return seq, curve, P


class Board:
    """Accumulates thread darkness at film resolution with sub-pixel bilinear splats (anti-aliased, additive)."""

    def __init__(self, size: int, ink: float = 0.10):
        self.size, self.ink = size, ink
        self.acc = np.zeros(size * size, np.float64)

    def add_line(self, p, q):
        L = int(np.hypot(*(q - p)) * 1.5) + 2
        t = np.linspace(0, 1, L)
        x = p[0] + (q[0] - p[0]) * t; y = p[1] + (q[1] - p[1]) * t
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = x - x0; fy = y - y0
        w = self.ink / 1.5
        s = self.size
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi = np.clip(x0 + dx, 0, s - 1); yi = np.clip(y0 + dy, 0, s - 1)
            self.acc += np.bincount(yi * s + xi, weights=ww * w, minlength=s * s)

    def image(self, paper=(236, 231, 220), thread=(18, 16, 15)) -> np.ndarray:
        a = (1 - np.exp(-self.acc.reshape(self.size, self.size) * 1.0))[..., None]
        return (np.array(paper) * (1 - a) + np.array(thread) * a).astype(np.uint8)


def run(image: str | None, out: str, fmt: str = "9:16", seconds: float = 22, fps: int = 30, nails: int = 240,
        lines: int = 3500, solve_size: int = 400, accent: str = "#e63946", sound: bool = True, focus=(0.5, 0.38),
        frame: str = "circle", diameter_cm: float = 50.0, sequence_out: str | None = None, log=print) -> dict:
    gray_rgb = core.load_image(image, solve_size, 1.0, focus)
    gray = (gray_rgb @ np.array([0.299, 0.587, 0.114], np.float32))
    # contrast stretch so the solver spends thread on the face, not on mid-grey background
    lo, hi = np.percentile(gray, [2, 98]); gray = np.clip((gray - lo) / max(1e-6, hi - lo), 0, 1)
    log(f"  solving {lines} lines on {nails} nails ...")
    seq, curve, P = solve(gray, nails, lines, frame=frame)
    lay = Layout(fmt, 1.0, accent)
    PW, PH = lay.out_size
    scale = PW / solve_size
    Pf = P * scale
    board = Board(PW, ink=0.30 * (3000 / max(1000, len(seq))) ** 0.5)
    ref = core.resize(np.repeat(gray[..., None], 3, 2), PW, PH)
    if frame == "circle":
        yy, xx = np.mgrid[:PH, :PW]
        mask = ((xx - PW / 2) ** 2 + (yy - PH / 2) ** 2) <= (PW / 2) ** 2
        ref = np.where(mask[..., None], ref, 0).astype(np.uint8)
    else:
        mask = np.ones((PH, PW), bool)
    seg_len = [np.hypot(*(P[b] - P[a])) / solve_size * diameter_cm / 100 for a, b in zip(seq, seq[1:])]
    cum_m = np.concatenate([[0], np.cumsum(seg_len)])
    T_INTRO, T_HOLD, T_TITLE = 1.4, 2.6, 3.0
    t_wind = max(4.0, seconds - T_INTRO - T_HOLD - T_TITLE)
    nF = int(t_wind * fps)
    plan = core.schedule(len(seq) - 1, nF, k=3.0)
    caps = [(0.2, T_INTRO + 2.0, f"One thread. {nails} nails. No pen."),
            (T_INTRO + t_wind * 0.45, T_INTRO + t_wind * 0.75, "Every line goes where it is darkest.")]
    events = []

    def draw(drawn: int, highlight: int = 0) -> np.ndarray:
        img = board.image()
        img = np.where(mask[..., None], img, 0).astype(np.uint8)
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        acc = core.hex_rgb(accent)
        for k in range(max(1, drawn - highlight + 1), drawn + 1):
            a, b_ = Pf[seq[k - 1]], Pf[seq[k]]
            d.line([tuple(a), tuple(b_)], fill=acc, width=max(1, int(PW / 600)))
        r = max(1.2, PW / 500)
        for x, y in Pf:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(120, 116, 108))
        if drawn > 0:
            x, y = Pf[seq[drawn]]
            d.ellipse([x - 3 * r, y - 3 * r, x + 3 * r, y + 3 * r], fill=acc)
        return np.asarray(im)

    def hud(drawn: int, stage: str, cap="", ca=0.0, labels=True) -> Hud:
        return Hud(title="STRING ART", stage=stage, caption=cap, caption_alpha=ca, show_labels=labels,
                   stats=[("LINES", f"{drawn:,}"), ("NAILS", str(nails)), ("THREAD", f"{cum_m[drawn]:.0f} m")],
                   curve=curve[:max(1, drawn)] if drawn > 1 else None, curve_total=len(curve),
                   left_label="REFERENCE", right_label="ONE THREAD")

    with core.Film(out, lay.W, lay.H, fps) as film:
        img = draw(0)
        for i in range(int(T_INTRO * fps)):
            cap, ca = core.caption_at(film.seconds, caps)
            film.add(lay.frame(ref, img, hud(0, "An empty board", cap, ca)))
        drawn = 0
        for st in plan:
            while drawn < st:
                drawn += 1
                board.add_line(Pf[seq[drawn - 1]], Pf[seq[drawn]])
                if drawn < 400 or drawn % 25 == 0:
                    events.append(film.seconds)
            cap, ca = core.caption_at(film.seconds, caps)
            f = drawn / max(1, len(seq) - 1)
            stage = "First lines" if f < 0.05 else "Blocking the shadows" if f < 0.3 else "Building the face" if f < 0.75 else "Fine detail"
            film.add(lay.frame(ref, draw(drawn, highlight=3 if f < 0.5 else 1), hud(drawn, stage, cap, ca)))
        final = draw(drawn, 0)
        for i in range(int(T_HOLD * fps)):
            u = i / (T_HOLD * fps)
            film.add(lay.frame(ref, final, hud(drawn, "Finished", "Which one is the photo?", min(1, u * 4, (1 - u) * 4), not (0.25 < u < 0.8))))
        nT = int(T_TITLE * fps)
        lns = [f"{drawn:,} straight lines  ·  {nails} nails  ·  {cum_m[drawn]:.0f} m of thread on a {diameter_cm:.0f} cm board",
               "Greedy solver, no image model. The nail order is saved: you can build it."]
        for i in range(nT):
            film.add(lay.title_card("STRING ART", "one thread, one face", lns, alpha=min(1, i / (0.5 * fps), (nT - i) / (0.4 * fps))))
        dur = film.seconds
    seq_path = sequence_out or out[:-4] + "_nails.txt"
    with open(seq_path, "w") as f:
        f.write(f"# {nails} nails on a {frame}, numbered clockwise from the top. {len(seq) - 1} lines, {cum_m[-1]:.1f} m of thread.\n")
        f.write(" ".join(map(str, seq)) + "\n")
    if sound:
        from . import sound as snd
        lead = int(T_INTRO / dur * 200)
        e = np.array(curve); e = (e - e.min()) / max(1e-9, e.max() - e.min())
        env = np.concatenate([np.ones(lead), np.interp(np.linspace(0, 1, 140), np.linspace(0, 1, len(e)), e), np.zeros(60)])
        wav = out[:-4] + ".wav"
        snd.write_wav(wav, snd.score(dur, env, events=events[:600:2], chord="sus"))
        core.mux_audio(out, wav); os.remove(wav)
    return {"out": out, "lines": len(seq) - 1, "thread_m": round(float(cum_m[-1]), 1), "sequence": seq_path, "seconds": round(dur, 2)}
