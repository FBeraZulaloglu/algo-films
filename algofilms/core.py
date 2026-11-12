"""Shared pieces for every film: image loading, the ffmpeg frame pipe, layouts, the HUD and timing curves.

Everything here is plain numpy + Pillow + ffmpeg. No network, no model downloads, no paid APIs.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------------------------- formats
FORMATS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}

INK = (243, 239, 230)
DIM = (141, 138, 132)
FAINT = (60, 58, 55)
BG = (0, 0, 0)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ----------------------------------------------------------------------------------------------- images
def sample_image_path() -> str | None:
    """A public-domain portrait that ships with matplotlib (Grace Hopper, U.S. Navy photo)."""
    try:
        import matplotlib
        p = os.path.join(matplotlib.get_data_path(), "sample_data", "grace_hopper.jpg")
        return p if os.path.exists(p) else None
    except Exception:
        return None


def load_image(path: str | None, width: int, aspect: float = 0.8, focus: tuple[float, float] = (0.5, 0.45)) -> np.ndarray:
    """Centre-crop to `aspect` (w/h) around `focus` and resize to `width`. Returns float32 HxWx3 in [0, 1]."""
    path = path or sample_image_path()
    if path is None:
        raise SystemExit("No image given and the matplotlib sample photo was not found. Pass a path.")
    im = Image.open(path).convert("RGB")
    W, H = im.size
    if W / H > aspect:
        cw, ch = int(H * aspect), H
    else:
        cw, ch = W, int(W / aspect)
    x0 = int(np.clip(focus[0] * W - cw / 2, 0, W - cw))
    y0 = int(np.clip(focus[1] * H - ch / 2, 0, H - ch))
    im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((width, int(round(width / aspect))), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def resize(a: np.ndarray, w: int, h: int, smooth: bool = True) -> np.ndarray:
    im = Image.fromarray(a if a.dtype == np.uint8 else to_u8(a))
    return np.asarray(im.resize((w, h), Image.LANCZOS if smooth else Image.NEAREST))


# ----------------------------------------------------------------------------------------------- fonts
_FONT_CANDIDATES = {
    "mono": ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf"],
    "sans": ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/SFNS.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"],
    "serif": ["/System/Library/Fonts/NewYorkItalic.ttf", "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", "C:/Windows/Fonts/georgiai.ttf"],
    "title": ["/System/Library/Fonts/NewYork.ttf", "/System/Library/Fonts/Supplemental/Georgia.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "C:/Windows/Fonts/georgia.ttf"],
}
_font_cache: dict = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _font_cache:
        f = None
        for p in _FONT_CANDIDATES[kind]:
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, size)
                    break
                except OSError:
                    continue
        _font_cache[key] = f or ImageFont.load_default(size)
    return _font_cache[key]


def spaced(text: str, n: int = 1) -> str:
    """Letter-spaced small caps look for HUD labels."""
    return (" " * n).join(text.upper())


# ----------------------------------------------------------------------------------------------- timing
def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def slow_start(u: float, k: float = 4.0) -> float:
    """Maps film progress u in [0,1] to process progress. Early chaos gets more screen time."""
    u = min(1.0, max(0.0, u))
    return u if k == 0 else (math.exp(k * u) - 1) / (math.exp(k) - 1)


def schedule(total_steps: int, frames: int, k: float = 4.0) -> list[int]:
    """Which process step to show on each of `frames` frames (non-decreasing, starts at 0, ends at total)."""
    return [int(round(slow_start(i / max(1, frames - 1), k) * total_steps)) for i in range(frames)]


# ----------------------------------------------------------------------------------------------- film writer
class Film:
    """Pipes raw RGB frames into ffmpeg. Use as a context manager: `with Film(path, w, h, fps) as f: f.add(frame)`."""

    def __init__(self, path: str, w: int, h: int, fps: int = 30, crf: int = 18):
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg is required (brew install ffmpeg / apt install ffmpeg).")
        self.path, self.w, self.h, self.fps, self.frames = path, w, h, fps, 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def add(self, frame: np.ndarray, repeat: int = 1):
        if frame.dtype != np.uint8:
            frame = to_u8(frame)
        assert frame.shape == (self.h, self.w, 3), (frame.shape, (self.h, self.w))
        b = np.ascontiguousarray(frame).tobytes()
        for _ in range(repeat):
            self.p.stdin.write(b)
            self.frames += 1

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def mux_audio(video: str, wav: str, out: str | None = None) -> str:
    out = out or video
    tmp = out + ".tmp.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", tmp], check=True)
    os.replace(tmp, out)
    return out


def to_gif(video: str, gif: str, width: int = 480, fps: int = 12, start: float = 0, dur: float | None = None):
    """Small looping GIF for READMEs (palette-optimised)."""
    t = ["-ss", str(start)] + (["-t", str(dur)] if dur else [])
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *t, "-i", video, "-vf", vf, gif], check=True)


# ----------------------------------------------------------------------------------------------- layout + HUD
@dataclass
class Hud:
    title: str = ""                      # e.g. "GRADIENT DESCENT"
    stage: str = ""                      # e.g. "Finding the eyes"
    stats: list = field(default_factory=list)   # [("STEP", "1200"), ("LOSS", "0.0412")]
    caption: str = ""                    # big serif italic line
    caption_alpha: float = 1.0
    curve: list | None = None            # values to draw as a sparkline (log scale)
    curve_total: int | None = None       # x-extent of the sparkline (curve grows into it)
    left_label: str = "REFERENCE"
    right_label: str = "CODE"
    show_labels: bool = True
    footer: str = ""


class Layout:
    """Places a reference image and the generated image on a black canvas, then draws the HUD.

    16:9 -> side by side, 9:16 -> stacked, 1:1 / 4:5 -> generated image only (reference as a small inset)."""

    def __init__(self, fmt: str, aspect: float, accent: str = "#e9c46a", reference: bool = True):
        self.W, self.H = FORMATS[fmt]
        self.fmt, self.aspect, self.accent, self.ref_on = fmt, aspect, hex_rgb(accent), reference
        W, H = self.W, self.H
        if fmt == "16:9" and reference:
            ph = int(H * 0.72); pw = int(ph * aspect)
            if 2 * pw + 60 > W * 0.70:
                pw = int(W * 0.70 / 2 - 30); ph = int(pw / aspect)
            gap = 48; x0 = max(int(W * 0.20), (W - 2 * pw - gap) // 2 + int(W * 0.04)); y0 = int(H * 0.115)
            self.ref_box = (x0, y0, pw, ph); self.out_box = (x0 + pw + gap, y0, pw, ph)
        elif fmt == "9:16" and reference:
            ph = int(H * 0.36); pw = int(ph * aspect)
            if pw > W * 0.86:
                pw = int(W * 0.86); ph = int(pw / aspect)
            x0 = (W - pw) // 2; gap = 40; y0 = int(H * 0.12)
            self.ref_box = (x0, y0, pw, ph); self.out_box = (x0, y0 + ph + gap, pw, ph)
        else:
            ph = int(H * 0.72); pw = int(ph * aspect)
            if pw > W * 0.9:
                pw = int(W * 0.9); ph = int(pw / aspect)
            x0 = (W - pw) // 2; y0 = int((H - ph) * 0.40)
            self.out_box = (x0, y0, pw, ph)
            iw = int(pw * 0.22); ih = int(iw / aspect)
            self.ref_box = (x0 + pw - iw - 16, y0 + 16, iw, ih) if reference else None

    @property
    def out_size(self) -> tuple[int, int]:
        return self.out_box[2], self.out_box[3]

    def frame(self, ref: np.ndarray | None, out: np.ndarray, hud: Hud, ref_alpha: float = 1.0) -> np.ndarray:
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        x, y, w, h = self.out_box
        o = out if out.shape[:2] == (h, w) else resize(out, w, h)
        canvas[y:y + h, x:x + w] = o if o.dtype == np.uint8 else to_u8(o)
        if ref is not None and self.ref_box is not None and ref_alpha > 0:
            rx, ry, rw, rh = self.ref_box
            r = (ref if ref.shape[:2] == (rh, rw) else resize(ref, rw, rh)).astype(np.float32)
            canvas[ry:ry + rh, rx:rx + rw] = (canvas[ry:ry + rh, rx:rx + rw] * (1 - ref_alpha) + r * ref_alpha).astype(np.uint8)
        return self.draw_hud(canvas, hud, ref_alpha)

    def draw_hud(self, canvas: np.ndarray, hud: Hud, ref_alpha: float = 1.0) -> np.ndarray:
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        W, H = self.W, self.H
        s = W / 1920 if self.fmt == "16:9" else W / 1080 * 0.9
        x, y, w, h = self.out_box
        lab = font("mono", int(13 * s * 1.35))
        if hud.show_labels:
            if self.ref_box is not None and self.fmt in ("16:9", "9:16") and ref_alpha > 0.02:
                rx, ry, rw, rh = self.ref_box
                d.text((rx, ry - int(26 * s)), spaced(hud.left_label), font=lab, fill=tuple(int(c * ref_alpha) for c in DIM))
            d.text((x, y - int(26 * s)), spaced(hud.right_label), font=lab, fill=DIM)
        # title + stage, top left (16:9) or top centre
        if hud.title or hud.stage:
            tf = font("sans", int((24 if self.fmt == "16:9" else 26) * s)); sf = font("mono", int(14 * s * 1.2))
            if self.fmt == "16:9":
                tx, ty = int(40 * s), y
                d.text((tx, ty), hud.title, font=tf, fill=INK)
                d.multiline_text((tx, ty + int(44 * s)), _wrap(spaced(hud.stage), 12), font=sf, fill=self.accent, spacing=6)
            else:
                ty = int(H * 0.035)
                d.text((W // 2, ty), hud.title, font=tf, fill=INK, anchor="mt")
                d.text((W // 2, ty + int(40 * s)), spaced(hud.stage), font=sf, fill=self.accent, anchor="mt")
        # stats
        if hud.stats:
            kf = font("mono", int(12 * s * 1.25)); vf = font("mono", int(28 * s))
            if self.fmt == "16:9":
                sx = int(40 * s); sy = y + int(170 * s)
                for k, v in hud.stats:
                    d.text((sx, sy), spaced(k), font=kf, fill=DIM)
                    d.text((sx, sy + int(22 * s)), v, font=vf, fill=INK)
                    sy += int(84 * s)
            else:
                n = len(hud.stats); colw = w / n; sy = y + h + int(26 * s)
                for i, (k, v) in enumerate(hud.stats):
                    cx = int(x + colw * (i + 0.5))
                    d.text((cx, sy), spaced(k), font=kf, fill=DIM, anchor="mt")
                    d.text((cx, sy + int(24 * s)), v, font=vf, fill=INK, anchor="mt")
        # loss curve
        if hud.curve and len(hud.curve) > 1 and self.fmt in ("16:9", "9:16"):
            if self.fmt == "16:9":
                cx0, cy0, cw, ch = x - (self.out_box[0] - self.ref_box[0] if self.ref_box else 0), y + h + int(30 * s), 0, int(56 * s)
                cw = (x + w) - cx0
            else:
                cx0, cw, ch = x, w, int(46 * s); cy0 = y + h + int((110 if hud.stats else 30) * s)
            v = np.log(np.maximum(np.asarray(hud.curve, np.float64), 1e-9))
            lo, hi = v.min(), v.max() if v.max() > v.min() else v.min() + 1
            total = max(hud.curve_total or len(v), len(v))
            pts = [(cx0 + cw * i / max(1, total - 1), cy0 + ch * (1 - (vi - lo) / (hi - lo))) for i, vi in enumerate(v)]
            d.line([(cx0, cy0 + ch), (cx0 + cw, cy0 + ch)], fill=FAINT, width=1)
            d.line(pts, fill=INK, width=max(1, int(2 * s)), joint="curve")
            d.ellipse([pts[-1][0] - 4 * s, pts[-1][1] - 4 * s, pts[-1][0] + 4 * s, pts[-1][1] + 4 * s], fill=self.accent)
        if hud.caption and hud.caption_alpha > 0:
            cf = font("serif", int(40 * s))
            col = tuple(int(c * hud.caption_alpha) for c in INK)
            if self.fmt == "16:9":
                cy = int(H * 0.04)
            elif self.fmt == "9:16":   # below stats + curve
                cy = y + h + int(((110 if hud.stats else 30) + 46 + 40) * s)
                cy = min(cy, H - int(60 * s))
            else:                      # square / 4:5: below the stats, no curve
                cy = y + h + int((92 if hud.stats else 24) * s)
            d.text((W // 2, cy), hud.caption, font=cf, fill=col, anchor="mt")
        if hud.footer:
            d.text((W // 2, H - int(30 * s)), hud.footer, font=font("mono", int(12 * s * 1.2)), fill=DIM, anchor="ms")
        return np.asarray(im)

    def title_card(self, title: str, subtitle: str, lines: list[str], alpha: float = 1.0) -> np.ndarray:
        im = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(im)
        s = self.W / 1920 if self.fmt == "16:9" else self.W / 1080 * 0.85
        a = lambda c: tuple(int(v * alpha) for v in c)
        cy = self.H // 2 - int(90 * s)
        d.text((self.W // 2, cy), spaced(title, 1), font=font("title", int(84 * s)), fill=a(INK), anchor="mm")
        d.text((self.W // 2, cy + int(90 * s)), subtitle, font=font("serif", int(44 * s)), fill=a(self.accent), anchor="mm")
        lf = font("sans", int((20 if self.fmt == "16:9" else 25) * s))
        out_lines = []
        for ln in lines:                       # wrap to 86% of the width
            cur = ""
            for wd in ln.split(" "):
                t = (cur + " " + wd).strip()
                if d.textlength(t, font=lf) > self.W * 0.86 and cur:
                    out_lines.append(cur); cur = wd
                else:
                    cur = t
            out_lines.append(cur)
        for i, ln in enumerate(out_lines):
            d.text((self.W // 2, cy + int((170 + 38 * i) * s)), ln, font=lf, fill=a(DIM), anchor="mm")
        return np.asarray(im)


def _wrap(text: str, n: int) -> str:
    words, lines, cur = text.split("  "), [], ""
    for wd in words:
        if len(cur) + len(wd) + 2 > n * 2 and cur:
            lines.append(cur); cur = wd
        else:
            cur = (cur + "  " + wd) if cur else wd
    if cur:
        lines.append(cur)
    return "\n".join(l.strip() for l in lines)


def caption_at(t: float, caps: list[tuple[float, float, str]]) -> tuple[str, float]:
    """caps = [(start, end, text)]; returns text + fade alpha at time t."""
    for a, b, txt in caps:
        if a <= t <= b:
            return txt, min(1.0, (t - a) / 0.4, (b - t) / 0.4)
    return "", 0.0


def fmt_int(n: int) -> str:
    return f"{n:,}"
