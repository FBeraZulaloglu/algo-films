"""algo-films command line.

  python -m algofilms splat     photo.jpg -o splat.mp4 --format 9:16
  python -m algofilms strings   photo.jpg -o strings.mp4
  python -m algofilms epicycles --text "HELLO" -o epi.mp4
  python -m algofilms morph     photo.jpg --text "HELLO" -o morph.mp4
  python -m algofilms neural    photo.jpg -o race.mp4 --format 16:9
  python -m algofilms shapes    photo.jpg -o shapes.mp4

Leave out the photo to use the public-domain sample portrait that ships with matplotlib."""
from __future__ import annotations

import argparse
import json
import time


def _pair(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="algofilms", description="Process films rendered in code. No AI credits.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, fmt="9:16", seconds=20.0, image=True):
        if image:
            p.add_argument("image", nargs="?", default=None, help="input photo (default: bundled sample portrait)")
        p.add_argument("-o", "--out", required=True, help="output .mp4")
        p.add_argument("--format", default=fmt, choices=["16:9", "9:16", "1:1", "4:5"])
        p.add_argument("--seconds", type=float, default=seconds)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--accent", default="#e9c46a", help="accent colour (hex)")
        p.add_argument("--no-sound", action="store_true", help="skip the generated soundtrack")
        p.add_argument("--focus", type=_pair, default=(0.5, 0.42), help="crop centre as x,y in 0..1")
        return p

    p = common(sub.add_parser("splat", help="gradient descent: blobs learn a photo"), "16:9", 24)
    p.add_argument("--steps", type=int, default=1500); p.add_argument("--blobs", type=int, default=20000)
    p.add_argument("--train-width", type=int, default=320); p.add_argument("--zoom-at", type=_pair, default=(0.5, 0.38))
    p.add_argument("--title", default="GRADIENT DESCENT"); p.add_argument("--subtitle", default="for a picture")
    p.add_argument("--device", default=None)

    p = common(sub.add_parser("strings", help="string art: one thread, many nails"), "9:16", 22)
    p.add_argument("--nails", type=int, default=240); p.add_argument("--lines", type=int, default=3500)
    p.add_argument("--frame", default="circle", choices=["circle", "square"])
    p.add_argument("--diameter-cm", type=float, default=50.0)

    p = common(sub.add_parser("epicycles", help="Fourier circles draw a word or an outline"), "9:16", 20, image=False)
    p.add_argument("--text", default=None); p.add_argument("--image", default=None)
    p.add_argument("--circles", type=int, default=300); p.add_argument("--mode", default="auto", choices=["auto", "silhouette"])

    p = common(sub.add_parser("morph", help="every pixel of one picture rearranged into another"), "9:16", 16)
    p.add_argument("--target", default=None, help="target image (default: --text poster)")
    p.add_argument("--text", default="HELLO"); p.add_argument("--grid", type=int, default=140)
    p.add_argument("--no-back", action="store_true")

    p = common(sub.add_parser("neural", help="three neural networks race to learn a photo"), "16:9", 22)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--zoom-at", type=_pair, default=(0.44, 0.33))
    p.add_argument("--archs", default="relu,fourier,siren"); p.add_argument("--device", default=None)

    p = common(sub.add_parser("shapes", help="a photo from translucent triangles, by trial and error"), "9:16", 18)
    p.add_argument("--shapes", type=int, default=300)

    a = ap.parse_args(argv)
    snd = not a.no_sound
    t0 = time.time()
    if a.cmd == "splat":
        from . import splat
        r = splat.run(a.image, a.out, a.format, a.seconds, a.fps, a.steps, a.blobs, train_width=a.train_width,
                      focus=a.focus, zoom_at=a.zoom_at, accent=a.accent, sound=snd, title=a.title, subtitle=a.subtitle, dev=a.device)
    elif a.cmd == "strings":
        from . import strings
        r = strings.run(a.image, a.out, a.format, a.seconds, a.fps, a.nails, a.lines, accent=a.accent, sound=snd,
                        focus=a.focus, frame=a.frame, diameter_cm=a.diameter_cm)
    elif a.cmd == "epicycles":
        from . import epicycles
        r = epicycles.run(a.out, text=a.text, image=a.image, fmt=a.format, seconds=a.seconds, fps=a.fps,
                          circles=a.circles, accent=a.accent, sound=snd, mode=a.mode)
    elif a.cmd == "morph":
        from . import pixelmorph
        r = pixelmorph.run(a.image, a.out, a.target, a.text, a.format, a.seconds, a.fps, a.grid, accent=a.accent,
                           sound=snd, focus=a.focus, back=not a.no_back)
    elif a.cmd == "neural":
        from . import neural
        r = neural.run(a.image, a.out, a.format, a.seconds, a.fps, a.steps, tuple(a.archs.split(",")), focus=a.focus,
                       zoom_at=a.zoom_at, accent=a.accent, sound=snd, dev=a.device)
    else:
        from . import shapes
        r = shapes.run(a.image, a.out, a.format, a.seconds, a.fps, a.shapes, accent=a.accent, sound=snd, focus=a.focus)
    r["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
