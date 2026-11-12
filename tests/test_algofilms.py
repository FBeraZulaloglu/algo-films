"""Fast checks for every tool: the maths does what it claims, and each film renders end to end at tiny sizes.

Run:  python -m pytest -q"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algofilms import core, sound  # noqa: E402


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"]


# ----------------------------------------------------------------------------------------------- core
def test_schedule_is_monotone_and_complete():
    s = core.schedule(1000, 90, k=3)
    assert s[0] == 0 and s[-1] == 1000 and all(b >= a for a, b in zip(s, s[1:]))
    assert s[45] < 500          # slow start: half the film shows less than half the steps


def test_load_image_crops_to_aspect():
    im = core.load_image(None, 200, 0.8)
    assert im.shape == (250, 200, 3) and im.dtype == np.float32 and 0 <= im.min() and im.max() <= 1


def test_film_writer_and_layouts(tmp_path):
    for fmt in core.FORMATS:
        lay = core.Layout(fmt, 0.8)
        w, h = lay.out_size
        f = lay.frame(np.zeros((h, w, 3), np.uint8), np.ones((h, w, 3), np.float32) * 0.5,
                      core.Hud(title="T", stage="s", stats=[("A", "1")], caption="c", curve=[1, 0.5, 0.2]))
        assert f.shape == (core.FORMATS[fmt][1], core.FORMATS[fmt][0], 3)
    p = str(tmp_path / "x.mp4")
    with core.Film(p, 64, 48, 10) as film:
        film.add(np.zeros((48, 64, 3), np.uint8), repeat=5)
    st = probe(p)[0]
    assert st["width"] == 64 and int(st["nb_frames"]) == 5


def test_sound_is_bounded_stereo():
    x = sound.score(2.0, np.linspace(1, 0, 50), events=[0.5, 1.0])
    assert x.shape == (int(2.0 * sound.SR), 2) and np.abs(x).max() <= 1.0 and np.abs(x).max() > 0.1


# ----------------------------------------------------------------------------------------------- splat
def test_splat_gradient_descent_lowers_loss():
    torch = pytest.importorskip("torch")
    from algofilms.splat import Blobs, render
    dev = torch.device("cpu")
    tgt = torch.zeros(40, 32, 3); tgt[10:30, 8:24] = torch.tensor([0.9, 0.2, 0.1])
    b = Blobs(400, 400, dev)
    opt = torch.optim.Adam(b.params, lr=0.03)
    losses = []
    for _ in range(40):
        img = render(b, 32, 40, bg=torch.zeros(3))
        loss = (img - tgt).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.6


def test_splat_big_blobs_are_not_cut_into_squares():
    torch = pytest.importorskip("torch")
    from algofilms.splat import Blobs, render
    b = Blobs(1, 1, torch.device("cpu"))
    with torch.no_grad():
        b.pos[0] = torch.tensor([0.5, 0.5]); b.lsig[0] = math.log(0.2); b.amp[0] = 5; b.col[0] = 5
    img = render(b, 400, 400, bg=torch.zeros(3))[..., 0]
    # a blob with sigma = 80 px must still be bright 150 px from its centre (a 65 px patch would cut it to zero)
    assert img[200, 350] > 0.1 and abs(img[200, 350] - img[350, 200]) < 0.02


# ----------------------------------------------------------------------------------------------- strings
def test_string_solver_lowers_residual():
    from algofilms.strings import solve
    g = np.ones((120, 120), np.float32); g[30:90, 50:70] = 0
    seq, curve, P = solve(g, n_nails=60, n_lines=80, min_gap=5)
    assert len(seq) > 20 and curve[-1] < curve[0] and len(set(seq)) > 3


# ----------------------------------------------------------------------------------------------- epicycles
def test_fourier_reconstructs_a_circle_exactly():
    from algofilms.epicycles import evaluate, fourier
    t = np.linspace(0, 1, 256, endpoint=False)
    z = 3 * np.exp(2j * np.pi * t) + 0.5 * np.exp(-2j * np.pi * 3 * t)
    f, C = fourier(z, 10)
    assert np.allclose(evaluate(f, C, t, 3), z, atol=1e-9)
    assert abs(abs(C[1]) - 3) < 1e-9          # largest circle first (after the centre term)


def test_text_becomes_one_closed_path():
    from algofilms.epicycles import chain, resample, text_contours
    pts, fl = chain(text_contours("AB"))
    z, jump = resample(pts, fl, 512)
    assert len(z) == 512 and jump.any() and not jump.all()


# ----------------------------------------------------------------------------------------------- morph
def test_pixel_morph_is_a_permutation_and_improves():
    from algofilms.pixelmorph import assign, refine
    rng = np.random.default_rng(0)
    n = 30 * 30
    src, dst = rng.random((n, 3)), rng.random((n, 3))
    yy, xx = np.mgrid[:30, :30]; xy = np.stack([xx.ravel(), yy.ravel()], 1) / 30
    p0 = assign(src, dst)
    p, hist = refine(p0.copy(), src, dst, xy, xy, rounds=200)
    assert sorted(p) == list(range(n)) and hist[-1] < hist[0]


# ----------------------------------------------------------------------------------------------- shapes
def test_shape_evolution_lowers_error():
    from algofilms.shapes import evolve
    t = np.zeros((40, 32, 3), np.float32); t[:, 16:] = [0.9, 0.8, 0.1]
    sh, curve, base = evolve(t, n_shapes=15, guesses=20, mutations=20, log=lambda s: None)
    assert curve[-1] < curve[0] * 0.5 and len(sh) > 5


# ----------------------------------------------------------------------------------------------- neural
def test_neural_nets_learn():
    torch = pytest.importorskip("torch")
    from algofilms.neural import ARCHS, grid
    xy = grid(16, 20); tgt = (xy[:, :1] > 0.5).float().repeat(1, 3)
    for name in ARCHS:
        torch.manual_seed(0)
        net = ARCHS[name][1](); opt = torch.optim.Adam(net.parameters(), lr=ARCHS[name][2])
        l0 = ((net(xy) - tgt) ** 2).mean().item()
        for _ in range(60):
            opt.zero_grad(); loss = ((net(xy) - tgt) ** 2).mean(); loss.backward(); opt.step()
        assert loss.item() < l0, name


# ----------------------------------------------------------------------------------------------- end to end
E2E = [
    ["strings", "--lines", "200", "--nails", "80", "--seconds", "8"],
    ["epicycles", "--text", "Hi", "--circles", "40", "--seconds", "8"],
    ["morph", "--grid", "40", "--seconds", "8", "--text", "OK"],
    ["shapes", "--shapes", "20", "--seconds", "8"],
    ["splat", "--steps", "30", "--blobs", "1500", "--train-width", "96", "--seconds", "12", "--device", "cpu"],
    ["neural", "--steps", "20", "--seconds", "11", "--device", "cpu"],
]


@pytest.mark.parametrize("args", E2E, ids=[a[0] for a in E2E])
def test_cli_renders_film(tmp_path, args):
    out = str(tmp_path / f"{args[0]}.mp4")
    extra = [] if args[0] == "strings" else ["--no-sound"]
    r = subprocess.run([sys.executable, "-m", "algofilms", *args, "-o", out, "--fps", "12", *extra],
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    kinds = {s["codec_type"] for s in probe(out)}
    assert "video" in kinds
    if args[0] == "strings":
        assert "audio" in kinds and os.path.exists(out[:-4] + "_nails.txt")
