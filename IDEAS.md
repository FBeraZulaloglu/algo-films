# More process films (no AI credits)

The six tools in this repo share one recipe. Any algorithm that fits it can become a film:

1. **A visible start state that is obviously not the answer** (noise, an empty board, one flat colour).
2. **A real process that measurably gets closer** (a loss, an error, a residual), so the numbers on screen are honest.
3. **A payoff you have to wait for** (side by side "which one is the photo?", a zoom that proves what it is made of).
4. **One sentence of truth on the title card** (how many steps, blobs, lines; what was *not* learned).

Ideas that fit the recipe and run on a laptop, ordered by how quickly they could be built on top of `algofilms.core`:

| Idea | The process | The payoff shot |
|---|---|---|
| **Halftone / stippling by Voronoi relaxation** | Lloyd's algorithm moves 20k dots until their density matches the photo's darkness | Zoom out: dots become a face; zoom in: it was only dots |
| **TSP art** | Stipple dots, then a travelling-salesman tour joins them into one unbroken line | "One line. Never lifts the pen." The line traced end to end |
| **Physarum / slime mould painting** | 200k agents follow and deposit trails, seeded by the photo's brightness | The photo emerges as veins, then dissolves back |
| **Reaction-diffusion portrait** | Gray-Scott equations where feed/kill rates come from the photo | Coral patterns that resolve into a face |
| **Boids that flock into a logo** | Flocking rules plus a weak pull to target points | Chaos -> murmuration -> logo, then they scatter |
| **Sorting algorithms on image columns** | Shuffle the photo's pixel columns, then sort them with bubble/quick/merge sort side by side | A race: which sort rebuilds the photo first |
| **Maze solving** | Generate a maze whose walls follow the photo's edges, then BFS/A* floods it | The solved path draws the portrait |
| **Simulated annealing jigsaw** | The photo cut into tiles and shuffled; annealing swaps tiles to match neighbours | "Nobody told it what the picture was" (no reference used) |
| **Cellular automata reveal** | Conway's Life seeded by the photo; or a rule that grows toward the target | Gliders and noise that settle into an image |
| **Mosaic from your own frames** | Each tile of a portrait is replaced by the closest frame from a video you already made | Zoom from the face into one of your old videos |
| **Gradient descent on music** | A spectrogram of noise optimised toward a real recording (Griffin-Lim), audible every step | You hear a melody emerge from hiss |
| **3D point cloud from one photo, without a model** | Shape-from-shading on a face lit from one side | The camera turns (honest: a guess, not a scan) |
| **Particle text for creator intros** | Your channel name written by 30k particles that follow a curl-noise field | A branded 3-second sting, rendered free |

## Production notes

- Put a hold on the noise at the start (1-2 s) and a hold on the result at the end. The middle can be fast.
- Map film time to steps with a slow start (`core.schedule(..., k=3)`): the first steps change the most.
- Keep labels honest: show the real step, loss and count, and say what is estimated or hand-made.
- Vertical (9:16) works best stacked: reference on top, process below, numbers between them.
- The generated soundtrack (`algofilms.sound`) is a bed that follows the error curve. For a release, lay a real track.
