# Denysko implementation notes: challenges observed

Status: the pipeline is implemented per `docs/SPEC.md` around **unbounded
polynomials** (`y=<poly>`, no Desmos domain restriction). The default pytest
suite is fast (target < 1 s, enforced by a wall-clock deadline in
`tests/conftest.py`; currently ~0.6 s) and contains no stochastic real-letter
fitting. Manual acceptance runs of real letters are separate from the unit
suite; see the section on manual behavior below.

## Current unbounded trace semantics

A polynomial's useful part is its **trace**:

```
trace(P) = { x : ymin <= P(x) <= ymax }
```

where `[ymin, ymax]` is the normalized glyph's vertical band, always derived
from the actual boundary points. A valid curve must have exactly one finite
non-empty trace interval `[l, r]`; everything outside the band contributes no
surface penalty. There is no internal drawing interval and no Desmos domain
restriction — the trace is intrinsic to the polynomial.

## Tail semantics: post-exit steepness

The slope at the exact trace endpoints `l, r` is **not** the steepness gate.
A curve following the glyph surface may legitimately cross `ymin`/`ymax` with
a shallow slope (an `A` leg reaches the bottom at slope ~2-3; a rounded glyph
may touch top/bottom tangentially). The rule is: follow the glyph faithfully
until leaving its vertical band, then bend rapidly toward vertical and never
return.

For each trace endpoint the analysis:

- determines whether the graph exited above `ymax` or below `ymin`;
- finds the first point farther outward reaching
  `ymax + TAIL_VERTICAL_MARGIN` / `ymin - TAIL_VERTICAL_MARGIN`
  (`TAIL_VERTICAL_MARGIN = 5`) **analytically** via the nearest root of
  `P-(ymax+5)` / `P-(ymin-5)` in the tail direction (no such root ⇒ invalid);
- measures the horizontal run to that point (must be ≤ `MAX_TAIL_X_RUN = 5`);
- measures the slope there (must be `|P'| ≥ MIN_TAIL_SLOPE = 8`);
- requires permanent escape: no real derivative root beyond the trace
  endpoint, and the derivative points away from the band immediately outside.

This replaces the old `|P'(l)|, |P'(r)| ≥ 8` rule, which conflicted with
surface following and made the earlier "degree <= 5 cannot trace a long
diagonal" conclusion incorrect.

## The removed impossible horizontal-bar test

An earlier version of the test suite contained
`test_horizontal_segment_search_with_escaping_tails`, which demanded that an
isolated horizontal segment be representable. That is not the intended
geometry (a polynomial is not asked to abruptly stop following a surface and
escape through empty in-band space), so the test was deleted rather than
weakening the contract. The suite instead carries handcrafted validator tests
that follow a boundary and exit steeply after leaving it.

## Tail rule tests

The suite replaces the obsolete "slope at trace endpoint < 8 => invalid"
contract with handcrafted post-exit tests:

- **Case A** — a line crossing the band with slope 2 stays slope 2 forever:
  invalid, because it never becomes steep outside the band.
- **Case B** — a Hermite quintic that follows a boundary, crosses the band
  with endpoint slopes ~3.5 (< 8), but reaches the ±5 margin within ~0.75
  x-units with |P'| ~ 10.8: valid, and passes V1-V4.
- **Case C** — a curve that rises out the top, peaks, and falls back through
  the band (multiple trace components): invalid.
- **Case D** — a curve reaching the ±5 margin only after more than 5
  horizontal units: invalid.

## Seed construction: endpoint-anchored degree-5 bent seeds

Seeds are built from the local-contour `p1, p2` pair (segment-scored by
`(mean segment distance, max segment distance, -point distance)`, `p2` must be
at least `MIN_X_SEPARATION = 0.1` in x away from `p1`). Each restart gets
exactly five initial candidates in `u = (x-50)/50` coordinates:

1. the ordinary line `L(u)` through `p1, p2`;
2. four **degree-5 bent seeds** `P(u) = L(u) + aQ(u) + bR(u)` with
   `Q(u) = (u-ul)² (u-ur)²` and `R(u) = Q(u)(u-m)`, `m = (ul+ur)/2`, anchored
   at the line's **provisional trace endpoints** `[l0, r0]` — one for each
   tail orientation (up/up, down/down, up/down, down/up).

The provisional endpoints are the line's natural band exits (analytic
intersections with `ymin`/`ymax`); for an unbounded horizontal line a working
window of half-width `UNBOUNDED_SEED_HALF_WIDTH = 15` around the seed-pair
midpoint is used instead, clamped to the padded glyph extents. Because both
bases vanish with zero derivative at `ul`/`ur`, every bent seed follows the
provisional straight surface route all the way to its natural band exits and
bends only outside them. The pair `(a, b)` is solved exactly so each seed hits
its requested tail levels at `xL = l0 - SEED_TAIL_X_RUN` /
`xR = r0 + SEED_TAIL_X_RUN` with `SEED_TAIL_X_RUN = MAX_TAIL_X_RUN`; a
singular/ill-conditioned system skips that seed. Tail target positions are
therefore relative to the provisional exits — never global glyph x-extents.

Exact-same-x pairs are rejected at selection time (`_seed_pair`), and
`_line_seed_u` never fabricates an x-separation; it computes the actual line
through the two accepted points.

Every generated seed is independently refined: `REFINE_STEPS` is a
**per-restart budget** split deterministically across the five hills, so
exploring all basins costs no more work than refining one. Bent-seed hills use
structured mutations (50 % coefficient / 25 % Q-direction / 25 % R-direction)
with degree mutation suppressed during their first half.

## Exploration merit

Search uses a single scalar merit per candidate:

```
merit = coverage_fraction
        - 4.0 * bad_surface_fraction
        - 0.5 * mean_surface_excess
        - 2.0 * trace_penalty
        - 1.0 * tail_penalty
        - 0.005 * degree
```

The inputs are continuous so greedy hill climbing has useful gradients:

- **Surface metrics score ALL finite trace components**, and unbounded
  in-band components are sampled over a finite viewport so a horizontal
  stroke earns its real surface/coverage while keeping the +2.0 structural
  penalty. A candidate whose second component is a huge spurious stroke
  through empty space can no longer hide it by reporting only the widest
  component's adherence.
- **trace_penalty is continuous**: `2.0 * extra_component_fraction` where
  `extra_component_fraction = extra_arc / total_arc` over all sampled
  components (+2.0 for an unbounded component, 2.0 for an empty trace).
  Shrinking a spurious component improves merit continuously before it
  vanishes.
- **tail_penalty is continuous per side**: a missing ±5 margin root probes at
  `end ± MAX_TAIL_X_RUN` and charges `1.0 + remaining_vertical_fraction`; a
  wrong direction adds a fixed larger penalty; when a margin root exists the
  x-run and slope deficits are proportional; derivative roots beyond the
  trace endpoint count as turns **once each** (the side penalties are the
  only place turns are charged — `deriv_outside` remains a separate field
  for hard feasibility and diagnostics). A near-valid tail scores much better
  than a bad one, and a one-component/nearly-valid candidate naturally
  out-ranks a two-component candidate.

The coverage/surface/trace/degree weights are intentionally untouched.

Feasible candidates are judged strictly (single finite trace, >= 95 % trace
adherence, no derivative roots outside, both tails reach the ±5 margin within
`MAX_TAIL_X_RUN` with `|P'| >= MIN_TAIL_SLOPE` there, newly >= MIN_NEW_POINTS)
and compared by `(newly_covered, -degree, -mean_surface_distance)`. Every hill
climb tracks current, best exploratory (highest-merit state seen anywhere, via
`HillResult`), and best feasible states separately, and returns both; the
search retains the best exploratory state across restarts for diagnostics.

Candidate analysis is single-pass: `analyze_candidate` converts `u -> x`
once, computes the roots of `P-ymin`, `P-ymax`, `P'`, `P-(ymin-5)` and
`P-(ymax+5)` once, derives trace components and post-exit tail geometry once,
samples every finite component once, and derives every metric from those
cached values. Search geometry is cheap: a deterministic `SEARCH_BOUNDARY_MAX`
subset of the boundary and at most `SEARCH_GRAPH_MAX` trace samples per
hill-climb step; dense full-boundary evaluation is reserved for accepted
curves and final validation.

## Manual behavior for real letters (current)

The default pytest suite intentionally does not run real-letter fitting.
Manual acceptance is a separate development step, run in the order
`V`, `A`, `C`, `O` (`V` first: its useful outer geometry is simple and
x-monotone). Current results (default seed, this environment/lockfile):
`fit_curves()` finds **no feasible first curve** for all four letters and
`validate()` reports V1 = 0. The zero-curve diagnostics (stderr) describe the
actual best explored state across all restarts **and all five seed hills**:

- `V` — winner seed=`line`, degree 1, merit -1.71, surface 0.85, new=64,
  single trace `[53.42, 91.60]`, both tails direction ok, margin reached at
  x_run 1.91 with slope **2.61 < 8**, turns 0.
- `A` — winner seed=`line`, degree 1, merit -0.97, surface 1.00, new=81,
  single trace `[2.31, 37.02]`, margin slopes **2.87 < 8**.
- `C` — winner seed=`line`, degree 1, merit -1.96, surface 0.53, new=60,
  single trace `[10.33, 20.04]`, margin slopes 10.27 (tails fine).
- `O` — winner seed=`down/down`, degree 4, merit -2.51, surface 0.41, new=2,
  single trace `[65.34, 66.62]`, margin slopes 17.25/16.46 (steep but the
  covered sliver is tiny).

## Measured evidence from this iteration

The iteration's goal was to give the degree-5 escape-capable seed family a
fair chance. What was measured:

1. **All-five-seed refinement works, but the line still wins V/A/C.**
   Instrumenting one `V` restart shows each basin's best exploratory merit:
   line -2.68 -> -2.60, up/up -7.81 -> -7.17, down/down -6.96 -> -5.40,
   up/down -8.08 -> -7.21, down/up -6.71 -> -4.95. The structured Q/R
   mutations and continuous penalties genuinely improve every bent basin
   (down/up gains +1.76 merit in its 24 steps), but none closes the ~2.5-point
   gap to the line within the fixed per-restart budget.
2. **Why the bent seeds start so far behind: the exact-solve fights its own
   feasibility rule.** Pinning P(xL) = target at exactly xL = l0 -
   MAX_TAIL_X_RUN forces an average post-exit slope of margin/run = 5/5 = 1 —
   *shallower* than any stroke steeper than 1. On a slope-2.5 diagonal the
   natural-orientation seed arrives at the ±5 margin with |P'| ≈ 1.16 (< the
   line's own 2.5), overshoots to about -5.6, turns, and re-enters the band
   further out. Every bent seed therefore starts as a multi-component trace
   (ncomp 2–4, trace_penalty ≈ 1.3–1.6, i.e. a built-in -2.6..-3.2 merit
   handicap) even though its interior follows the stroke.
3. **Unbounded horizontal lines finally score usefully.** A constant y=50
   line over a crossbar glyph analyzes (search mode) at surface_fraction
   1.00 with real coverage while remaining structurally unbounded
   (trace_penalty 2.0, infeasible). The optimizer now has a meaningful
   gradient toward bending it into a finite trace instead of seeing zero.
4. `O`'s best explored state switched from the line to a degree-4
   `down/down` bend — evidence that the new basins can win where the local
   geometry suits them — but it covers only 2 boundary points (surface 0.41).

The precise blocker for `V`/`A` is therefore no longer generic convergence:
the seed family that could fix their too-shallow margins starts handicapped
by construction-induced extra components, and the fixed budget cannot erase
that handicap before the line's clean single-component state wins the
comparison. Candidate next steps (not taken in this iteration): solve the
bent-seed targets with the run as an inequality (reach ±margin *within*
`SEED_TAIL_X_RUN` rather than exactly at it), or drop position constraints in
favor of slope constraints at the exits.

A synthetic steep vertical bar is solved end-to-end (two feasible degree-1
curves, `validate()` passes all of V1-V4), so the pipeline mechanics (seeds,
merit, feasibility, reduction, validation) remain correct on geometries where
feasible curves exist.

## The global-trace limitation (explicit)

Every finite trace starts and ends where the polynomial reaches the glyph's
**global** `ymin` / `ymax`. Therefore a separate boundary component whose own
y-range lies strictly inside `[ymin, ymax]` may be impossible to cover.
Examples include many inner-hole/counter contours (the inner loop of `O`,
`A`'s triangular counter, `R`/`B`/`P` bowls, etc.). This is a real limitation
of the current model, not a search tuning issue. It is documented rather than
solved: V1 is not weakened, holes are not silently excluded, and development
focuses on connected outer geometry.

## The x-monotonicity limitation (explicit)

A `y = f(x)` trace is intrinsically an **x-monotone geometric path**: its
useful visible part is a graph over `x`. Some connected boundary routes are
therefore impossible to follow even when the outline is connected and has no
inner counter, because following the outline would require `x` to reverse
direction. `C`'s outline is a candidate for this: covering a long useful
route may require the curve to swing left again, which a single-valued graph
cannot do. This is separate from the inner/disconnected contour limitation
above and is also not solved in this task.

Per the task instructions, iteration budgets were not increased to chase
convergence; the failures above are reported as measured.

## Default pytest runtime

`uv run pytest` completes the whole suite in roughly 0.6 s (the one-second
deadline in `tests/conftest.py` fires otherwise). It contains no stochastic
real-letter fitting and no alphabet sweep; alphabet evaluation is a manual
development benchmark run outside pytest.