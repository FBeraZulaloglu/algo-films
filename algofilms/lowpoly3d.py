"""Low-poly 3D portrait: a photo becomes a triangle mesh with relief, plus a per-frame vertex cache (PC2) in which
the triangles fly in from a cloud, lock into the flat picture, then inflate into 3D. Blender only has to light
and film it (see blender/lowpoly_scene.py).

No depth model: the relief is a hand-made guess (distance from the silhouette edge + a bulge at the face).

  python -m algofilms.lowpoly3d photo.jpg --out out/lowpoly3d"""
from __future__ import annotations

import argparse
import math
import os
import struct

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

from . import core


def subject_mask(img: np.ndarray) -> np.ndarray:
    """Rough person mask for a portrait on a plain wall: far from the border colour, largest blob, filled."""
    f = cv2.GaussianBlur(img, (0, 0), 3).astype(np.float32)
    h, w = f.shape[:2]
    lab = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    top = np.concatenate([lab[:int(h * 0.25), :int(w * 0.12)].reshape(-1, 3), lab[:int(h * 0.25), -int(w * 0.12):].reshape(-1, 3)])
    ref = np.median(top, 0)
    dl = np.abs(lab[..., 0] - ref[0]); dc = np.linalg.norm(lab[..., 1:] - ref[1:], axis=2)
    m = ((dc > 9.0) | (dl > 62.0)).astype(np.uint8)        # colour differs, or much darker: a soft wall shadow is neither
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lbl == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # fill holes
    ff = m.copy(); cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    return ((m == 1) | (ff == 0)).astype(np.float32)


def relief(mask: np.ndarray, face=(0.5, 0.40), face_r=0.20) -> np.ndarray:
    h, w = mask.shape
    dist = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    dome = np.clip(dist / (0.16 * w), 0, 1)
    dome = np.sin(dome * math.pi / 2) ** 0.8                         # round shoulders, flat middle
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    bulge = np.exp(-(((xx / w - face[0]) / face_r) ** 2 + ((yy / w - face[1] * h / w) / (face_r * 1.25)) ** 2))
    z = (0.55 * dome + 0.75 * bulge * mask) * mask
    return cv2.GaussianBlur(z, (0, 0), w / 90)


def triangulate(img: np.ndarray, n_points: int = 2600, seed: int = 2):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), 2).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    p = (mag ** 1.2 + mag.mean() * 0.25).reshape(-1); p /= p.sum()
    idx = rng.choice(h * w, n_points * 3, p=p)
    cand = np.stack([idx % w, idx // w], 1).astype(np.float32)
    # greedy thinning: keep points at least r apart (r smaller where there is more detail)
    grid = {}
    keep = []
    r0 = math.sqrt(h * w / n_points) * 0.55
    for x, y in cand:
        key = (int(x // r0), int(y // r0))
        if key in grid:
            continue
        grid[key] = 1; keep.append((x, y))
        if len(keep) >= n_points:
            break
    k = 14
    border = [(x, 0) for x in np.linspace(0, w - 1, k)] + [(x, h - 1) for x in np.linspace(0, w - 1, k)] + \
             [(0, y) for y in np.linspace(0, h - 1, int(k * h / w))[1:-1]] + [(w - 1, y) for y in np.linspace(0, h - 1, int(k * h / w))[1:-1]]
    pts = np.array(keep + border, np.float32)
    tri = Delaunay(pts).simplices
    blur = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32)
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cen = (a + b + c) / 3
    samp = [cen, (a + cen) / 2, (b + cen) / 2, (c + cen) / 2]
    col = np.mean([blur[np.clip(s[:, 1].astype(int), 0, h - 1), np.clip(s[:, 0].astype(int), 0, w - 1)] for s in samp], 0) / 255
    return pts, tri, col.astype(np.float32)


def preview(pts, tri, col, w, h, path):
    img = np.zeros((h * 2, w * 2, 3), np.uint8)
    for t, c in zip(tri, col):
        cv2.fillPoly(img, [np.round(pts[t] * 2 * 16).astype(np.int32)], tuple(int(v * 255) for v in c), cv2.LINE_AA, shift=4)
    Image.fromarray(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)).save(path, quality=92)


def write_pc2(path: str, frames: np.ndarray, start: float = 1.0):
    """frames: (F, N, 3) float32. Blender's Mesh Cache modifier reads this."""
    F, N, _ = frames.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<12sllffl", b"POINTCACHE2\0", 1, N, start, 1.0, F))
        f.write(frames.astype("<f4").tobytes())


def build(image: str, out: str, focus=(0.5, 0.45), aspect: float = 0.8, n_points: int = 2600, fps: int = 24,
          seconds: float = 10.0, depth: float = 0.55, seed: int = 4, log=print) -> dict:
    os.makedirs(out, exist_ok=True)
    W = 900; H = int(W / aspect)
    img = core.to_u8(core.load_image(image, W, aspect, focus))
    mask = subject_mask(img)
    z = relief(mask)
    Image.fromarray((np.clip(z / z.max(), 0, 1) * 255).astype(np.uint8)).save(f"{out}/relief.png")
    pts, tri, col = triangulate(img, n_points)
    preview(pts, tri, col, W, H, f"{out}/lowpoly-2d.jpg")
    T = len(tri)
    # world units: picture is 2 wide, 2/aspect tall, centred, facing -Y (Blender camera looks along +Y)
    sx = 2.0 / W
    pz = z[np.clip(pts[:, 1].astype(int), 0, H - 1), np.clip(pts[:, 0].astype(int), 0, W - 1)] * depth
    P = np.stack([(pts[:, 0] - W / 2) * sx, -pz, (H / 2 - pts[:, 1]) * sx], 1).astype(np.float32)       # x, y(depth toward camera = -y), z(up)
    V3 = P[tri]                                     # (T, 3, 3) final relief positions, one private vertex per corner
    V0 = V3.copy(); V0[:, :, 1] = 0                 # flat picture
    rng = np.random.default_rng(seed)
    cen = V0.mean(1)
    # scatter: every triangle gets its own offset, spin axis and start delay
    dirs = rng.normal(size=(T, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    off = dirs * rng.uniform(1.2, 3.6, (T, 1)); off[:, 1] = -np.abs(off[:, 1]) * 0.9 - 0.2     # toward the camera side
    axis = rng.normal(size=(T, 3)); axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = rng.uniform(2.0, 7.0, T) * rng.choice([-1, 1], T)
    is_bg = mask[np.clip(((H / 2 - cen[:, 2] / sx)).astype(int), 0, H - 1), np.clip((cen[:, 0] / sx + W / 2).astype(int), 0, W - 1)] < 0.5
    r = np.hypot(cen[:, 0], cen[:, 2] - 0.35)
    delay = 0.62 * (r / r.max()) + 0.10 * rng.random(T)          # the face lands first, the edges last
    rng_bg = rng.random(T).astype(np.float32)

    def rot(v, ax, a):                               # Rodrigues, v (T,3,3) about per-triangle axis
        c, s = np.cos(a)[:, None, None], np.sin(a)[:, None, None]
        k = ax[:, None, :]
        return v * c + np.cross(np.broadcast_to(k, v.shape), v) * s + k * (v * k).sum(2, keepdims=True) * (1 - c)

    F = int(seconds * fps)
    t_in0, t_in1, t_3d0, t_3d1 = 0.9, 5.0, 5.2, 6.6
    frames = np.empty((F, T * 3, 3), np.float32)
    for f in range(F):
        t = f / fps
        u = np.clip((t - t_in0) / (t_in1 - t_in0), 0, 1)
        e = np.clip((u * 1.72 - delay) / 0.72, 0, 1)
        e = 1 - (1 - e) ** 3                          # ease out: fast arrival, soft landing
        k = 1 - e
        drift = 0.06 * math.sin(t * 0.9)             # the cloud breathes before it starts
        inflate = core.ease((t - t_3d0) / (t_3d1 - t_3d0))
        base = V0 + (V3 - V0) * inflate
        local = (base - cen[:, None, :]) * (1 - 0.45 * k[:, None, None])
        local = rot(local, axis, ang * k)
        pos = cen[:, None, :] + local + (off * (k + drift * (k > 0.98))[:, None])[:, None, :]
        # when the picture inflates, the wall behind lets go: its triangles shrink and drift back, leaving a bust
        g = (inflate * is_bg)[:, None, None]
        c3 = pos.mean(1, keepdims=True)
        pos = c3 + (pos - c3) * (1 - g) + np.array([0, 2.5, 0], np.float32) * g * (0.4 + 0.6 * rng_bg)[:, None, None]
        frames[f] = pos.reshape(-1, 3)
    write_pc2(f"{out}/assemble.pc2", frames)
    np.savez(f"{out}/mesh.npz", verts=frames[-1], colors=np.repeat(col, 3, 0), fps=fps, frames=F, aspect=aspect)
    log(f"  {T} triangles, {F} frames -> {out}/assemble.pc2 ({os.path.getsize(out + '/assemble.pc2') / 1e6:.0f} MB)")
    return {"triangles": T, "frames": F, "out": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--out", default="out/lowpoly3d")
    ap.add_argument("--points", type=int, default=2600); ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--focus", default="0.5,0.45")
    a = ap.parse_args()
    print(build(a.image, a.out, tuple(map(float, a.focus.split(","))), n_points=a.points, seconds=a.seconds))
