"""Camera tour: a virtual camera wanders through one large still image (a wide interior, a painting, a map),
visiting points of interest on a smooth path, then pulls back to reveal the whole picture.

Pure geometry: a spline path, log-space zoom, a slight keystone in the direction of travel (it reads as the camera
turning rather than a flat slide), and motion blur from sub-frame sampling. No depth model, no video model.

  python -m algofilms.tour wide.png -o tour.mp4 --stops "0.86,0.72,0.15,0.8; 0.55,0.42,0.22; 0.33,0.50,0.13,1.0"

Each stop is x,y,width[,hold]: centre in 0..1 of the image, window width as a fraction of the image width,
and an optional hold in seconds."""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from PIL import Image

from . import core


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


class Path:
    """Stops -> camera state (cx, cy, w) at any time. Zoom is interpolated in log space."""

    def __init__(self, stops, speed: float = 0.20, min_move: float = 1.4, ease: float = 0.65, schedule=None):
        self.P = np.array([[s[0], s[1], math.log(s[2])] for s in stops], np.float64)
        self.hold = [s[3] if len(s) > 3 else 0.0 for s in stops]
        self.ease = ease
        self.seg = []
        t = 0.0
        self.times = []           # (t_start_hold, t_end_hold) per stop
        for i in range(len(stops)):
            self.times.append((t, t + self.hold[i])); t += self.hold[i]
            if i < len(stops) - 1:
                d = math.hypot(*(self.P[i + 1][:2] - self.P[i][:2])) + 0.35 * abs(self.P[i + 1][2] - self.P[i][2])
                dur = max(min_move, d / speed)
                self.seg.append((t, dur)); t += dur
        self.duration = t
        if schedule:              # explicit (arrive, depart) seconds per stop, e.g. to land on narration cues
            self.times = [tuple(x) for x in schedule]
            self.seg = [(schedule[i][1], max(0.05, schedule[i + 1][0] - schedule[i][1])) for i in range(len(stops) - 1)]
            self.duration = schedule[-1][1]

    def at(self, t: float):
        t = min(max(t, 0.0), self.duration)
        n = len(self.P)
        for i in range(n):
            a, b = self.times[i]
            if a <= t <= b or i == n - 1:
                return self._out(self.P[i])
            s, dur = self.seg[i]
            if t < s + dur:
                u = (t - s) / dur
                u = self.ease * (u * u * (3 - 2 * u)) + (1 - self.ease) * u
                p0, p1, p2, p3 = self.P[max(0, i - 1)], self.P[i], self.P[i + 1], self.P[min(n - 1, i + 2)]
                p = _catmull(p0, p1, p2, p3, u)
                lo, hi = min(p1[2], p2[2]), max(p1[2], p2[2])
                p[2] = min(max(p[2], lo - 0.08), hi + 0.08)     # no zoom overshoot
                return self._out(p)
        return self._out(self.P[-1])

    @staticmethod
    def _out(p):
        return float(p[0]), float(p[1]), float(math.exp(p[2]))


def run(image: str, out: str, stops, fmt: str = "9:16", fps: int = 30, speed: float = 0.20, blur_samples: int = 4,
        keystone: float = 0.045, remove_badge: bool = True, depth: str | None = None, parallax: float = 0.40,
        sway: float = 0.012, schedule=None, log=print) -> dict:
    W, H = core.FORMATS[fmt]
    src = np.asarray(Image.open(image).convert("RGB"))
    ih, iw = src.shape[:2]
    if remove_badge:        # small generator badge in the top-left corner: paint it out
        x1, y1 = int(iw * 0.058), int(ih * 0.085)
        m = np.zeros((y1, x1), np.uint8); m[int(ih * 0.008):, int(iw * 0.004):] = 255
        src = src.copy(); src[:y1, :x1] = cv2.inpaint(np.ascontiguousarray(src[:y1, :x1]), m, 9, cv2.INPAINT_TELEA)
    # pyramid so that strong zoom-outs are sampled from a pre-filtered level (no shimmer)
    pyr = [src]
    while pyr[-1].shape[1] > 900:
        pyr.append(cv2.pyrDown(pyr[-1]))
    bg = cv2.GaussianBlur(cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA), (0, 0), 40)
    bg = (bg.astype(np.float32) * 0.35).astype(np.uint8)
    path = Path(stops, speed, schedule=schedule)
    D = None
    if depth:                                    # disparity map in 0..1 (1 = near), e.g. from Depth Anything
        D = np.load(depth).astype(np.float32) if depth.endswith(".npy") else np.asarray(Image.open(depth).convert("L"), np.float32) / 255
        D = cv2.dilate(D, np.ones((5, 5), np.uint8))               # near things keep their edges when they slide
        D = cv2.GaussianBlur(D, (0, 0), 2.0)
        qx, qy = np.meshgrid((np.arange(W, dtype=np.float32) + 0.5) / W - 0.5, (np.arange(H, dtype=np.float32) + 0.5) / H - 0.5)
        dh, dw = D.shape
    aspect = W / H
    max_w = (ih * aspect) / iw                    # widest window that still fits the image height

    def focus_depth(t):
        """Depth the camera is looking at, averaged over +-0.6 s so it glides instead of jumping."""
        v = []
        for k in np.linspace(-0.6, 0.6, 7):
            cx, cy, w = path.at(t + k)
            x = int(min(max(cx, 0), 1) * (dw - 1)); y = int(min(max(cy, 0), 1) * (dh - 1))
            v.append(float(D[max(0, y - 6):y + 7, max(0, x - 6):x + 7].mean()))
        return sum(v) / len(v)

    def frame_parallax(t):
        """Every pixel slides by an amount that depends on its depth relative to what the camera looks at:
        near things sweep past faster than far ones, as they do for a camera that really moves."""
        cx, cy, w = path.at(t)
        ww = w * iw; wh = ww / aspect
        if w <= max_w:
            cx = min(max(cx, w / 2), 1 - w / 2); hh = wh / ih; cy = min(max(cy, hh / 2), 1 - hh / 2)
        d0 = focus_depth(t)
        g = parallax * min(1.0, max_w / w) ** 0.5            # less effect once pulled far back
        ox = sway * iw * math.sin(t * 0.55); oy = sway * 0.4 * iw * math.sin(t * 0.37 + 1.0)   # slow drift, no shake
        sx = cx * iw + qx * ww; sy = cy * ih + qy * wh       # first guess: no parallax
        for _ in range(3):                                   # fixed-point: depth is read where the pixel comes from
            d = cv2.remap(D, sx * (dw / iw), sy * (dh / ih), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            m = 1 + g * (d - d0)
            sx = cx * iw + qx * ww / m - ox * (d - d0) * g * 2.5
            sy = cy * ih + qy * wh / m - oy * (d - d0) * g * 2.5
        lvl = int(min(len(pyr) - 1, max(0, math.floor(math.log2(max(1.0, ww / W))))))
        sc = 1 / (2 ** lvl)
        img = cv2.remap(pyr[lvl], sx * sc, sy * sc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if w > max_w * 1.001:
            inside = ((sx >= 0) & (sx < iw) & (sy >= 0) & (sy < ih)).astype(np.float32)
            a = cv2.GaussianBlur(inside, (0, 0), 1.2)[..., None]
            img = (img * a + bg * (1 - a)).astype(np.uint8)
        return img

    def frame_at(t, vx):
        if D is not None:
            return frame_parallax(t)
        cx, cy, w = path.at(t)
        ww = w * iw; wh = ww / aspect
        if w <= max_w:                            # keep the window inside the picture
            cx = min(max(cx, w / 2), 1 - w / 2)
            hh = wh / ih
            cy = min(max(cy, hh / 2), 1 - hh / 2)
        x0, y0 = cx * iw - ww / 2, cy * ih - wh / 2
        lvl = int(min(len(pyr) - 1, max(0, math.floor(math.log2(max(1.0, ww / W))))))
        s = 1 / (2 ** lvl)
        srcq = np.float32([[x0, y0], [x0 + ww, y0], [x0 + ww, y0 + wh], [x0, y0 + wh]]) * s
        k = float(np.clip(vx * keystone, -keystone, keystone))       # leading edge grows a little
        l, r = max(0.0, -k), max(0.0, k)
        dst = np.float32([[0, -H * l], [W, -H * r], [W, H * (1 + r)], [0, H * (1 + l)]])
        M = cv2.getPerspectiveTransform(srcq, dst)
        img = cv2.warpPerspective(pyr[lvl], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if w > max_w * 1.001:                     # pulled back beyond the picture: sit it on a soft backdrop
            mask = cv2.warpPerspective(np.full(pyr[lvl].shape[:2], 255, np.uint8), M, (W, H), flags=cv2.INTER_LINEAR)
            a = (cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255)[..., None]
            img = (img * a + bg * (1 - a)).astype(np.uint8)
        return img

    n = int(round(path.duration * fps))
    dt = 1 / fps
    with core.Film(out, W, H, fps) as film:
        for i in range(n):
            t = i * dt
            c0, c1 = path.at(t - dt / 2), path.at(t + dt / 2)
            vx = (c1[0] - c0[0]) / dt / max(1e-6, c1[2])            # window-widths per second
            speed_now = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / max(1e-6, c1[2]) + abs(math.log(c1[2] / c0[2]))
            k = blur_samples if speed_now > 0.0015 else 1
            if k == 1:
                film.add(frame_at(t, vx))
            else:
                acc = np.zeros((H, W, 3), np.float32)
                for j in range(k):
                    acc += frame_at(t + (j / k - 0.5) * dt * 0.5, vx)   # 180-degree shutter
                film.add((acc / k).astype(np.uint8))
            if i % 150 == 0:
                log(f"  frame {i}/{n}")
        dur = film.seconds
    return {"out": out, "seconds": round(dur, 2), "stops": len(stops), "source": [iw, ih]}


def parse_stops(s: str):
    out = []
    for part in s.split(";"):
        v = [float(x) for x in part.strip().split(",") if x.strip()]
        if len(v) >= 3:
            out.append(tuple(v))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--stops", required=True); ap.add_argument("--format", default="9:16")
    ap.add_argument("--speed", type=float, default=0.20); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--keep-badge", action="store_true")
    ap.add_argument("--depth", default=None, help="disparity map (.npy or image, bright = near) for real parallax")
    ap.add_argument("--parallax", type=float, default=0.40)
    a = ap.parse_args()
    print(run(a.image, a.out, parse_stops(a.stops), a.format, a.fps, a.speed, remove_badge=not a.keep_badge, depth=a.depth, parallax=a.parallax))
