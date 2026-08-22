# SPEC: Single-Letter → Desmos Polynomial Converter

Version 2.0 · Status: draft · Scope frozen: **one letter per run, nothing else**

Version 2.0 rework: emitted curves are **unbounded polynomials**. Domain
restrictions (`\left\{...\right\}`) are no longer part of the output language;
the relevant portion of each curve is derived from the glyph's vertical band
instead (§6).

---

## 1. Mission

Given exactly one uppercase letter, emit a small set of unbounded `y = f(x)` polynomial
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

Formal treatment: `trace(P)` in §6, tail behaviour in §6.3, validation in §7.

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

The slope at the exact trace endpoints `l, r` is **not** the steepness gate:
a polynomial following the glyph surface may naturally cross `ymin`/`ymax`
with a shallow slope (an `A` leg reaches the bottom at slope ~2-3; a rounded
glyph may touch top/bottom tangentially). The rule is:

> follow the glyph faithfully until leaving its vertical band; then, once
> outside, bend rapidly toward vertical and never return.

For each trace endpoint the analysis determines whether the graph exited
above `ymax` or below `ymin`, then locates the first point farther outward
where it reaches

```
ymax + TAIL_VERTICAL_MARGIN      (exited above)
ymin - TAIL_VERTICAL_MARGIN      (exited below)
```

with `TAIL_VERTICAL_MARGIN = 5`, analytically via polynomial roots of
`P-(ymax+5)` / `P-(ymin-5)` (the nearest root in the tail direction; no such
root means the tail is invalid). Let that point be `t`. A valid tail requires:

- the horizontal run from the trace endpoint to `t` is at most
  `MAX_TAIL_X_RUN = 5` (right: `t - r ≤ 5`; left: `l - t ≤ 5`);
- `|P'(t)| ≥ MIN_TAIL_SLOPE = 8`;
- no real root of `P'(x)` beyond the trace endpoint (right: none above `r`;
  left: none below `l`);
- the derivative points away from the band immediately outside the trace:
  right tail exited above ⇒ `P' > 0`, exited below ⇒ `P' < 0`; left tail
  (moving toward `-∞`) exited above ⇒ `P' < 0`, exited below ⇒ `P' > 0`.

This guarantees that after leaving the glyph band the curve continues away
and never returns. Nothing after the tail has left the band is rewarded or
penalized: what happens vertically beyond the glyph range, including
intersections between curves far above or below the glyph, does not matter.

### 6.4 Surface adherence

The graph is sampled only across its trace interval `[l, r]`. Each sampled point's
distance to the glyph boundary `𝒫` is Euclidean. A curve is surface-valid iff at least
95 % of its trace samples lie within `τ` of the boundary. This is the central visual
constraint.

### 6.5 Search: exploration vs feasibility

Search may be stochastic but **must be reproducible**: the same `(letter, seed, locked
environment)` produces byte-for-byte identical stdout. The seed has a fixed default.

Exploration and feasibility are judged separately. Every candidate is measured once by a
single-pass analysis (convert `u → x`, roots of `P-ymin`, `P-ymax`, `P'` and the two
margin levels `P-(ymax+5)` / `P-(ymin-5)`, trace components, one sample pass over every
finite component, one distance pass) producing normalized metrics:

```
coverage_fraction      = newly_covered / max(1, n_uncovered)
bad_surface_fraction   = fraction of sampled in-band graph points with distance > τ
mean_surface_excess    = mean(max(0, distance - τ)) / τ
trace_penalty          = 2.0 * extra_component_fraction  (multiple finite components)
                         + 2.0 if any unbounded component; 2.0 if empty
extra_component_fraction = extra_arc / max(total_arc, 1e-9)
tail_penalty           = per-side: turns + direction + margin + x-run + slope
x_run_penalty          = max(0, x_run - MAX_TAIL_X_RUN) / MAX_TAIL_X_RUN
slope_deficit          = max(0, MIN_TAIL_SLOPE - |P'(t)|) / MIN_TAIL_SLOPE
```

Surface and coverage metrics score the union of **all** finite trace components, so a
spurious extra stroke cannot hide behind the widest component. `trace_penalty` is
continuous in the extra-component arc (shrinking a spurious component improves merit
gradually). A tail that never reaches the ±5 margin estimates its remaining vertical
fraction by probing at `end ± MAX_TAIL_X_RUN` (penalty `1.0 + remaining_fraction`) plus
a fixed wrong-direction term and a turn count; feasibility itself stays hard.

The **exploratory merit** is a single scalar, larger being better:

```
merit = coverage_fraction
        - 4.0 * bad_surface_fraction
        - 0.5 * mean_surface_excess
        - 2.0 * trace_penalty
        - 1.0 * tail_penalty
        - 0.005 * degree
```

It may visit infeasible states; severe surface violations dominate coverage, invalid
tails are expensive, and small temporary imperfections remain crossable. During every
hill climb the search maintains three states separately — the current state, the
**best exploratory state** (the highest-merit candidate seen anywhere, returned for
diagnostics when nothing is feasible), and the **best feasible candidate** — feasible
iff exactly one finite trace, ≥ 95 % trace adherence within `τ`, no derivative roots
outside the trace, both tails leave the band permanently and reach the ±5 vertical
margin within `MAX_TAIL_X_RUN` with `|P'| ≥ MIN_TAIL_SLOPE` there, and
`newly_covered ≥ MIN_NEW_POINTS` — compared by `(newly_covered, -degree,
-mean_surface_distance)`. Mutation is 80 % single-coefficient / 20 % degree; there is
no domain mutation.

Seeding biases toward the local contour without contour tracing: for each restart, pick
`p1` among uncovered boundary points, sample up to 8 candidate partners `p2` from the
distance bands 3–15 (expanded to 25 when necessary) whose x-separation from `p1` is at
least `MIN_X_SEPARATION`, score each straight segment `p1 → p2` at 33 sample points by
`(mean boundary distance, max boundary distance, -point distance)`, and take the
best-scoring, more distant `p2` on ties. From `(p1, p2)` exactly five initial
polynomials are built, all in normalized `u = (x-50)/50` coordinates:

1. the ordinary line `L(u)` through `p1, p2`;
2. four **degree-5 two-parameter bent seeds** `P(u) = L(u) + aQ(u) + bR(u)` with
   `Q(u) = (u-ul)² (u-ur)²` and `R(u) = Q(u)(u-m)`, `m = (ul+ur)/2`, anchored at the
   line's **provisional trace endpoints** `[l0, r0]` (with `ul = u(l0)`,
   `ur = u(r0)`) — one for each tail orientation (up/up, down/down, up/down,
   down/up).

The provisional endpoints are where the local straight stroke would naturally leave
the glyph's vertical band, derived analytically from intersections of the line with
`ymin`/`ymax`; for an unbounded (horizontal/nearly horizontal) line a finite working
window is defined instead around the seed-pair midpoint with half-width
`UNBOUNDED_SEED_HALF_WIDTH`, clamped to the padded glyph extents. Working-window
metadata is used only for seed construction; the resulting candidates must still
satisfy the ordinary global trace rules once analyzed.

Because both `Q` and `R` vanish with zero derivative at `ul` and `ur`, every bent seed
follows the provisional straight surface route all the way to its natural band exits
and bends only outside them: `P(ul) = L(ul)`, `P(ur) = L(ur)`, `P'(ul) = L'(ul)`,
`P'(ur) = L'(ur)`. The pair `(a, b)` is solved exactly so that `P` reaches the
requested tail levels (`up = ymax + TAIL_VERTICAL_MARGIN`,
`down = ymin - TAIL_VERTICAL_MARGIN`) at

```
xL = l0 - SEED_TAIL_X_RUN      xR = r0 + SEED_TAIL_X_RUN
SEED_TAIL_X_RUN = MAX_TAIL_X_RUN
```

(`[Q(uL), R(uL); Q(uR), R(uR)] · [a, b] = [TL - L(xL), TR - L(xR)]`; a singular or
ill-conditioned system skips that seed). Tail target positions are therefore relative
to the provisional trace exits, aligned with the actual tail rule — never global glyph
x-extents. The line itself remains a seed for naturally steep strokes and
lower-degree solutions.

Every generated seed is independently refined: each restart hill-climbs all five
seeds and compares their best feasible and best exploratory states. To keep total
work constant, `REFINE_STEPS` is interpreted as a **per-restart refinement budget**
rather than per-seed steps; it is split deterministically across the seeds
(`steps_per_seed = REFINE_STEPS // len(seeds)`, remainder distributed to the first
seeds). Rescue restarts keep their existing count under the same interpretation.
Bent-seed hills use structured mutations — 50 % coefficient / 25 % Q-direction /
25 % R-direction, where Q bends both tails together and R alters left-vs-right
asymmetry — with degree mutation suppressed during the first half of refinement so
the constructed quintic basin is refined before structural exploration resumes; the
plain line keeps the ordinary coefficient/degree behaviour throughout.

Search geometry is deliberately cheap: a deterministic evenly-spaced subset of at most
`SEARCH_BOUNDARY_MAX` boundary points and at most `SEARCH_GRAPH_MAX` trace samples
per component drive every hill-climb step; dense, full-boundary evaluation is reserved
for accepted curves and final validation.

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
| V3 | **Tail behaviour:** exactly one finite trace interval; no re-entry into the vertical glyph range; no derivative root beyond either trace endpoint; each tail reaches `ymin−5`/`ymax+5` within `MAX_TAIL_X_RUN` with slope magnitude ≥ 8 there | all curves |
| V4 | **Round-trip:** every emitted line parses back to the same polynomial via the project's parser; coefficients finite, `\|c\| < 10^9`, no scientific notation | **100 %** |

Samples vertically outside the glyph range are ignored by V1/V2; unbounded tails are
intentional and never penalized (the old bbox-confinement rule is gone).

Any failure ⇒ exit code 1, print the failed check(s) to stderr, and print **nothing** to
stdout. No silent bad output, ever.

## 8. Acceptance criteria

- All 26 letters pass V1–V4 with defaults.

  > This remains the long-term target; the current trace model has known
  > representability limitations documented in CHALLENGES.md.
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
