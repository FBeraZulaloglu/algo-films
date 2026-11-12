#!/usr/bin/env bash
# Renders every demo film one after another (never in parallel: heavy jobs share one GPU / 16 GB of RAM),
# then cuts the small README GIFs. Uses the public-domain sample portrait bundled with matplotlib.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
mkdir -p out docs/media
run() { echo "== $1"; "$PY" -m algofilms "$@"; }
run splat     -o out/splat.mp4     --format 16:9 --seconds 26
run neural    -o out/neural.mp4    --format 16:9 --seconds 24
run strings   -o out/strings.mp4   --format 9:16 --seconds 22
run shapes    -o out/shapes.mp4    --format 9:16 --seconds 18
run morph     -o out/morph.mp4     --format 9:16 --seconds 16 --text "HELLO"
run epicycles -o out/epicycles.mp4 --format 9:16 --seconds 20 --text "Hello"
"$PY" - <<'PY'
from algofilms.core import to_gif
to_gif("out/splat.mp4", "docs/media/splat.gif", 640, 10, 1.0, 15)
to_gif("out/neural.mp4", "docs/media/neural.gif", 640, 10, 1.0, 20)
to_gif("out/strings.mp4", "docs/media/strings.gif", 300, 10, 0.5, 15)
to_gif("out/shapes.mp4", "docs/media/shapes.gif", 300, 10, 0.5, 13)
to_gif("out/morph.mp4", "docs/media/morph.gif", 300, 12, 1.0, 9)
to_gif("out/epicycles.mp4", "docs/media/epicycles.gif", 300, 12, 0.5, 16)
PY
ls -la out docs/media
