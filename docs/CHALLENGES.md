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

## Seed construction: two-parameter degree-5 bent seeds

Seeds are built from the local-contour `p1, p2` pair (segment-scored by
`(mean segment distance, max segment distance, -point distance)`, `p2` must be
at least `MIN_X_SEPARATION = 0.1` in x away from `p1`). Each restart gets
exactly five initial candidates in `u = (x-50)/50` coordinates:

1. the ordinary line `L(u)` through `p1, p2`;
2. four **degree-5 bent seeds** `P(u) = L(u) + aQ(u) + bR(u)` with
   `Q(u) = (u-u1)² (u-u2)²` and `R(u) = Q(u)(u-m)`, `m = (u1+u2)/2` — one for
   each tail orientation (up/up, down/down, up/down, down/up).

Both `Q` and `R` vanish with zero derivative at `u1` and `u2`, so every bent
seed preserves the seed values and the local stroke slope
(`P(u_i) = p_i.y`, `P'(u_i) = L'(u_i)`). The pair `(a, b)` is solved exactly
from the two tail targets at the global padded glyph x-extents
(`xL = xmin - 5`, `xR = xmax + 5`), so each seed hits both requested tail
levels exactly; a singular/ill-conditioned system skips that seed. The earlier
single-`k` degree-4/degree-5 family was removed as unnecessarily restrictive.

Exact-same-x pairs are rejected at selection time (`_seed_pair`), and
`_line_seed_u` never fabricates an x-separation; it computes the actual line
through the two accepted points.

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

- **Surface metrics score ALL finite trace components.** A candidate whose
  second component is a huge spurious stroke through empty space can no longer
  hide it by reporting only the widest component's adherence.
- **trace_penalty is continuous**: `2.0 * extra_component_fraction` where
  `extra_component_fraction = extra_arc / total_arc` over all sampled
  components (+2.0 for an unbounded component, 2.0 for an empty trace).
  Shrinking a spurious component improves merit continuously before it
  vanishes.
- **tail_penalty is continuous per side**: a missing ±5 margin root probes at
  `end ± MAX_TAIL_X_RUN` and charges `1.0 + remaining_vertical_fraction`; a
  wrong direction adds a fixed larger penalty; when a margin root exists the
  x-run and slope deficits are proportional; derivative roots beyond the trace
  endpoint count as turns. A near-valid tail scores much better than a bad
  one, and a one-component/nearly-valid candidate naturally out-ranks a
  two-component candidate.

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
Manual acceptance is a separate development step:

```
uv run denysko V
```

Current results (default seed, on this environment/lockfile). `fit_curves()`
finds **no feasible first curve** for all real letters tried; `validate()`
then reports V1 = 0. The zero-curve diagnostics (stderr) describe the actual
**best explored state** across all restarts, with its real `new=` count and
per-tail detail:

- `V` — best explored: degree-1 line, merit -1.71, surface 0.85, new=64,
  single trace, both tails reach the margin within x_run 1.92 but with slope
  **2.61 < 8**. Blocker: margin slope far below `MIN_TAIL_SLOPE`.
- `A` — best explored: degree-1 line, merit -0.97, surface 1.00, new=82,
  single trace, margin slope **2.85 < 8**. Blocker: same shallow-leg slope.
- `C` — best explored: degree-4, merit -1.86, surface 0.65, new=75, single
  trace, tails fine (9.39 / 9.82). Blocker: **surface 0.65 < 0.95**.
- `O` — best explored: degree-4, merit -1.37, surface 0.70, new=86, single
  trace, left slope 8.44 (ok), right slope **7.54 < 8**. Blocker: one tail
  just short of `MIN_TAIL_SLOPE` plus surface below 0.95.

So the remaining per-letter blockers are now visible and concrete:

1. `V`/`A`: the best single-trace candidate is a straight leg line whose
   post-exit margin slope (~2.6-2.9) is far below `MIN_TAIL_SLOPE = 8`. The
   search must find a curve that hugs the leg and bends to vertical shortly
   after leaving the band.
2. `C`: the best explored curve follows part of the outer geometry but its
   surface adherence is 0.65 — it does not hug enough of the boundary within
   `τ`.
3. `O`: close — one tail slope at 7.54 (just under 8) and surface 0.70.

A synthetic steep vertical bar is solved end-to-end (two feasible degree-1
curves, `validate()` passes all of V1-V4), so the pipeline mechanics (seeds,
merit, feasibility, reduction, validation) are correct on geometries where
feasible curves exist; the real-letter failures are search convergence
failures on the seed geometries, not validation failures.

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