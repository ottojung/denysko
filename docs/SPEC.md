# SPEC: Single-Letter → Desmos Polynomial Converter

Version 1.1 · Status: draft · Scope frozen: **one letter per run, nothing else**

---

## 1. Mission

Given exactly one uppercase letter, emit a small set of restricted `y = f(x)` polynomial
curves that, pasted into Desmos, render the **outline** of that letter so it is recognizable
at a glance.

One letter in → one letter drawn. This tool will not become a text renderer.

## 2. Non-goals (explicit)

- Multi-character input, words, spaces, kerning.
- Parametric curves (`x = f(t)`), implicit relations, inequalities, trig/exotics.
- Font selection, styling, bold/italic variants.
- Filling the interior of the glyph.
- Interactivity or GUI.

If a feature requires more than one glyph, it is out of scope.

## 3. Interface

```
desmos-letter <letter> [--seed N] [--max-curves K]
```

- `<letter>`: exactly one character from `A`–`Z`. Anything else → exit code 2 and no
  stdout.
- Successful output: **stdout only**, one Desmos expression per line. The tool writes no
  files.
- Diagnostics and validation failures go to stderr, never stdout.

### Expression contract (every emitted line)

1. Form `y=<poly>\ \left\{a\le x\le b\right\}` — a domain restriction is
   **mandatory**; an unrestricted line is invalid output.
2. Standard power basis only (`x^k`, decimal coefficients). No scientific notation, no
   functions, no variables other than `x`.
3. **Minimum sufficient degree**: each curve uses the smallest degree sufficient for the
   boundary points assigned to it. Degrees are tried in ascending order and the first
   sufficient degree is used — never pad degrees "just in case." Sufficiency is defined
   geometrically in §6.
4. At most `K` expressions total (default **12**) — keeps results usable in Desmos free
   tier.
5. Domain endpoints are finite and within `[−5, 105]`; coefficients are finite and
   `|c| < 10^9`.

## 4. Canonical glyph geometry

The input geometry must not depend on fonts installed on the host machine.

1. Use Matplotlib's bundled `DejaVuSans.ttf` as the canonical font, at `font_size = 100`.
2. Construct the filled glyph path for the requested letter.
3. Let its filled bounding box have width `w` and height `h`. Apply one **uniform** scale
   `s = 100 / max(w, h)` so aspect ratio is preserved, then translate so the lower-left
   corner of the filled bounding box is `(0, 0)`.
4. The resulting glyph therefore lies inside `[0, 100] × [0, 100]` and touches the left
   and bottom edges. In this project, `y = 0` is the normalized drawing baseline; it is
   not a promise to preserve the font's typographic baseline metric.
5. The y-axis points up everywhere downstream.

Normalization happens exactly once. Rasterization, fitting, validation, and serialization
all use these normalized coordinates.

Exact-output reproducibility is required for a fixed dependency lockfile. A change to the
canonical font bytes or locked numeric/rendering dependencies may intentionally change the
pinned reference output.

## 5. Target point cloud `P`

`P` represents the **boundary of the filled glyph**, not its interior pixels and not a
centerline/skeleton.

To construct it:

1. Rasterize the normalized filled glyph deterministically onto a fixed `512 × 512`
   boolean sampling grid covering `[0, 100] × [0, 100]`. No antialiasing threshold may
   depend on display/backend state.
2. A filled grid sample is a boundary sample if at least one of its four orthogonal
   neighbours is outside the filled glyph (samples beyond the grid count as outside).
3. Map boundary sample centres back to normalized coordinates; those coordinates form
   `P`.

This includes both the exterior boundary and boundaries of holes automatically. For
example, `O` contributes an outer loop and an inner loop.

The same `(letter, dependency lockfile)` must produce the same `P` on every run.

## 6. Geometric distance and fitting

Vertical residual `|c(x_p) - y_p|` is **not** the distance metric in this project. It
would make vertical and near-vertical glyph segments fundamentally unrepresentable by a
small number of `y = f(x)` curves.

For a restricted polynomial curve

```
C = {(x, c(x)) : a ≤ x ≤ b}
```

and a target point `p = (p_x, p_y)`, define its distance to the curve as ordinary
Euclidean distance to the nearest point on the restricted graph:

```
d(p, C) = min_{a ≤ x ≤ b} sqrt((x - p_x)^2 + (c(x) - p_y)^2)
```

This is the metric used by fitting, degree selection, fitness, and validation. In
particular, a sufficiently steep polynomial over a narrow x-domain can approximate a
vertical stroke because horizontal error counts correctly as geometric error.

### Assigned points and minimum sufficient degree

The search may assign an inlier subset `Q_i ⊆ P` to each output curve `C_i`; up to the
allowed global outlier fraction may remain unassigned.

For a fixed assignment and domain, degree `d` is sufficient iff there exists a degree-`d`
polynomial whose every assigned point satisfies

```
d(p, C_i) ≤ τ    for all p in Q_i
```

where `τ` is the coverage tolerance from §7. The fitter must test degrees from 0 upward
and use the first sufficient degree.

### Search objective

Search may be stochastic but **must be reproducible**: the same `(letter, seed, locked
environment)` produces byte-for-byte identical stdout. The seed has a fixed default.

The primary objective is boundary coverage. Fitness must be monotone in the useful
direction: moving an uncovered target point closer to the nearest curve, while changing
nothing else, must never make the solution worse.

A recommended lexicographic objective is:

1. maximize fraction of `P` within `τ` of some curve;
2. minimize a robust high-percentile target→curve distance;
3. minimize curve count;
4. minimize total polynomial degree.

The implementation may use a different objective only if it preserves those semantics.

## 7. Built-in validation gate (the tool grades itself)

Validation uses `τ = 2` normalized units by default. Numerical distance calculations may
be approximate internally, but the validator must be conservative enough that its
classification error is at most **0.05 normalized units** around a threshold.

Before emitting anything, all checks must pass:

| # | Check | Threshold (defaults) |
|---|-------|----------------------|
| V1 | **Boundary coverage:** fraction of `P` whose Euclidean distance to the nearest restricted curve is ≤ `τ` | ≥ **95 %** |
| V2 | **Curve adherence:** fraction of each curve's graph lying within Euclidean distance `τ` of the glyph boundary | ≥ **95 %** per curve |
| V3 | **Confinement:** every curve on its own restricted domain stays inside the glyph bbox grown by a margin | margin = **5 units** |
| V4 | **Round-trip:** every emitted line parses back to the same restricted polynomial via the project's parser | **100 %** |

For V2, sample each restricted graph densely enough in arc length that adjacent validation
samples are at most `0.5` normalized units apart. Endpoints are always included. Distance
from a graph sample to the glyph boundary is Euclidean distance to the nearest point in
`P`.

V2 exists to reject a curve that happens to cover useful boundary points but draws a long
spurious excursion through empty space. V3 remains a hard safety bound against numerical
explosions.

Any failure ⇒ exit code 1, print the failed check(s) to stderr, and print **nothing** to
stdout. No silent bad output, ever.

## 8. Acceptance criteria

- All 26 letters pass V1–V4 with defaults.
- The results are visually recognizable as outlines without large spurious strokes.
- Three reference letters are pinned in CI with exact-output regression under the locked
  environment and default seed:
  - `A` — diagonals, vertical-ish geometry, and crossbar;
  - `O` — exterior boundary plus closed hole;
  - `R` — mixed straight/curved geometry plus hole.
- Include at least one regression assertion demonstrating the reason for the Euclidean
  metric: a near-vertical synthetic target that fails under same-x vertical residual but
  passes geometric-distance fitting.
- End-to-end runtime ≤ **30 s** per letter on a laptop-class machine.

## 9. Dependencies

`uv`, Python ≥ 3.11, `numpy`, `matplotlib`. Nothing else.

The project must carry a `uv.lock`; seeded exact-output tests are interpreted relative to
that locked environment.
