# SPEC: Single-Letter → Desmos Polynomial Converter

Version 2.0 · Status: draft · Scope frozen: **one letter per run, nothing else**

Version 2.0 rework: emitted curves are **unbounded polynomials**. Domain
restrictions (`\left\{...\right\}`) are no longer part of the output language;
the relevant portion of each curve is derived from the glyph's vertical band
instead (§6).

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

1. Form `y=<poly>` — a plain unbounded polynomial. **No domain restriction** is emitted;
   an expression containing `\left\{...\right\}` (or any other restriction) is invalid
   output.
2. Standard power basis only (`x^k`, decimal coefficients). No scientific notation, no
   functions, no variables other than `x`.
3. **Minimum sufficient degree**: each curve uses the smallest degree sufficient for the
   boundary points assigned to it. Degrees are tried in ascending order and the first
   sufficient degree is used — never pad degrees "just in case." Sufficiency is defined
   geometrically in §6.
4. At most `K` expressions total (default **12**) — keeps results usable in Desmos free
   tier.
5. Coefficients are finite and `|c| < 10^9`.

### Unbounded-curve semantics

A polynomial is allowed to travel arbitrarily far from the glyph once it is vertically
above or below it. The governing rule is:

> Whenever the polynomial is inside the glyph's vertical range, it must be tracing the
> glyph boundary. Once it leaves that vertical range, it should leave steeply, continue
> monotonically toward infinity, and never re-enter.

Formal treatment: `trace(P)` in §6, tail behaviour in §6.4, validation in §7.

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

Normalizations and all downstream geometry (including the vertical band `ymin`/`ymax`
used by the trace and tail rules) are always derived from the actual normalized glyph
boundary points; global constants such as `[0, 100]` or `[-5, 105]` must never be
assumed in their place.

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

## 6. Geometric distance, trace, and fitting

Vertical residual `|c(x_p) - y_p|` is **not** the distance metric in this project. It
would make vertical and near-vertical glyph segments fundamentally unrepresentable by a
small number of `y = f(x)` curves.

For an unbounded polynomial graph and a target point `p = (p_x, p_y)`, distance is
ordinary Euclidean distance to the nearest point of the **trace** (defined below):

```
d(p, P) = min_{x in trace(P)} sqrt((x - p_x)^2 + (P(x) - p_y)^2)
```

In particular, a sufficiently steep polynomial can hug a vertical stroke because
horizontal error counts correctly as geometric error.

### 6.1 The trace of a polynomial

Let the normalized glyph boundary have vertical extent `[ymin, ymax]` (`ymin ≈ 0`, but
always computed from the actual boundary). For a polynomial `P`:

```
trace(P) = { x : ymin <= P(x) <= ymax }
```

This is the part of the graph that lives inside the letter's vertical band. Everything
outside that y-range is ignored for boundary-distance purposes:

- `P(x) > ymax` or `P(x) < ymin` contributes **zero** surface-distance penalty,
  regardless of how far from the letter the graph travels.
- Boundary coverage (§6.6) also counts only distances to trace samples.

### 6.2 One finite trace interval

Because `degree ≤ 5`, `trace(P)` is computed analytically enough from the real roots of

```
P(x) - ymin    and    P(x) - ymax
```

Collect the real roots, sort them, and classify the intervals between them.

A usable curve must have exactly **one finite non-empty trace interval** `[l, r]`.
Curves whose in-band set is unbounded, empty, or split into multiple separated intervals
are rejected. This guarantees that after the polynomial leaves the letter's vertical
band it never later returns. Disconnected glyphs are not special-cased; they are
outside the current problem.

### 6.3 Tail behaviour

For a curve with trace interval `[l, r]`:

- **Left tail** (`x < l`) and **right tail** (`x > r`): the polynomial must remain
  outside `[ymin, ymax]`, be monotone away from the band, and never turn around to
  re-enter.
- A sufficient monotonicity rule (the one used): no real root of `P'(x)` exists below
  `l`, and none exists above `r`.
- The exits must be steep: with hardcoded `MIN_TAIL_SLOPE = 8`,
  `|P'(l)| ≥ MIN_TAIL_SLOPE` and `|P'(r)| ≥ MIN_TAIL_SLOPE`.

Nothing after the tails have left the band is rewarded or penalized: what happens
vertically beyond the glyph range, including intersections between curves far above or
below the glyph, does not matter.

### 6.4 Surface adherence

The graph is sampled only across its trace interval `[l, r]`. Each sampled point's
distance to the glyph boundary `𝒫` is Euclidean. A curve is surface-valid iff at least
95 % of its trace samples lie within `τ` of the boundary. This is the central visual
constraint.

### 6.5 Search: exploration vs feasibility

Search may be stochastic but **must be reproducible**: the same `(letter, seed, locked
environment)` produces byte-for-byte identical stdout. The seed has a fixed default.

Exploration and feasibility are scored separately — one lexicographic tuple must not do
both jobs. For every candidate the search computes `newly_covered`, `surface_penalty`
(`Σ max(0, d - τ)²` over trace samples), `tail_penalty` (wrong trace component count,
unbounded/empty trace, derivative roots outside the trace interval, exit-slope deficit
below `MIN_TAIL_SLOPE`; a simple deterministic scalar), `degree`, and
`mean_surface_distance`. The exploratory hill-climbing score is

```
(newly_covered, -surface_penalty, -tail_penalty, -degree, -mean_surface_distance)
```

larger being better, and it may visit infeasible states. During every hill climb the
search additionally keeps the **best feasible candidate encountered so far** (exactly
one finite trace interval, tails monotone, exit slopes ≥ `MIN_TAIL_SLOPE`, ≥ 95 % of the
trace within `τ`, at least `MIN_NEW_POINTS` newly covered points), compared by
`(newly_covered, -degree, -mean_surface_distance)`, and returns it even if later
exploratory mutations drift into infeasibility.

Seeding biases toward the local contour without contour tracing: for each restart, pick
`p1` among uncovered boundary points, sample up to 8 candidate partners `p2` from the
distance bands 3–15 (expanded to 25 when necessary), score each straight segment
`p1 → p2` at 33 sample points by mean boundary distance, and take the best-scoring,
more distant `p2` on ties. From `(p1, p2)` up to five initial polynomials are built —
the ordinary line plus four cubics through `(xl, target_left)`, `(xr, target_right)`
with `xl/xr` padded 5 units outward and targets `(ymax+5 / ymin−5)` per side, fitted in
normalized `u = (x-50)/50` coordinates for stability. Refinement starts from the best
already-feasible seed if any, otherwise from the best exploration score; the two
boundary points are initialization constraints only and refinement may move away from
them.

### 6.6 Coverage and assigned points

Boundary coverage works as before: a glyph boundary point is covered if its Euclidean
distance to the sampled trace of any output polynomial is ≤ `τ`; the search maximizes
newly covered boundary points.

For degree selection and reduction, the search may assign an inlier subset
`Q_i ⊆ 𝒫` to each output curve; up to the allowed global outlier fraction may remain
unassigned. For a fixed assignment, degree `d` is sufficient iff there exists a
degree-`d` polynomial whose every assigned point satisfies `d(p, C_i) ≤ τ` for all
`p ∈ Q_i`. The fitter tests degrees from 0 upward and uses the first sufficient degree.
Reduction refines coefficients only — never degree — and accepts a lower degree only if
the curve remains fully feasible and still covers every assigned point.

## 7. Built-in validation gate (the tool grades itself)

Validation uses `τ = 2` normalized units by default. Numerical distance calculations may
be approximate internally, but the validator must be conservative enough that its
classification error is at most **0.05 normalized units** around a threshold.

Before emitting anything, all checks must pass:

| # | Check | Threshold (defaults) |
|---|-------|----------------------|
| V1 | **Boundary coverage:** fraction of `𝒫` whose Euclidean distance to the union of all polynomial trace portions is ≤ `τ` | ≥ **95 %** |
| V2 | **Surface adherence:** per curve, fraction of graph samples whose y lies inside `[ymin, ymax]` and within Euclidean distance `τ` of the glyph boundary | ≥ **95 %** per curve |
| V3 | **Tail behaviour:** exactly one finite trace interval; no re-entry into the vertical glyph range; no derivative root beyond either trace endpoint; exit-slope magnitude ≥ 8 at both ends | all curves |
| V4 | **Round-trip:** every emitted line parses back to the same polynomial via the project's parser; coefficients finite, `\|c\| < 10^9`, no scientific notation | **100 %** |

Samples vertically outside the glyph range are ignored by V1/V2; unbounded tails are
intentional and never penalized (the old bbox-confinement rule is gone).

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
