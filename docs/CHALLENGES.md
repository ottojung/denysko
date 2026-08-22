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
surface penalty. Tails must leave the band permanently: no real root of `P'`
below `l` or above `r`, and `|P'(l)|, |P'(r)| >= 8`.

## The removed impossible horizontal-bar test

An earlier version of the test suite contained
`test_horizontal_segment_search_with_escaping_tails`, which demanded that an
isolated horizontal segment be representable. Under the actual semantics that
is impossible: a degree <= 5 polynomial that stays inside a roughly 0.8-unit
band over ~60 horizontal units and already has slope >= 8 at its trace
endpoints would need to turn from horizontal to near-vertical within the
height of the band, which is not the intended geometry and does not exist.
That test, and any synthetic fixture demanding a polynomial abruptly stop
following a surface and escape through empty in-band space, was deleted rather
than weakening the geometry contract.

In its place the suite carries a handcrafted connected-shape validator test: a
steep curve whose boundary crosses `ymin` and `ymax` is surface-valid and
passes V1-V4, while a shallow exit fails the tail-steepness rule. It uses
handcrafted polynomials, not the stochastic fitter.

## Seed construction: bent lines

Seeds are built from the local-contour `p1, p2` pair (kept from the previous
work; segment-scored, tie-break toward the more distant partner). Each restart
gets five initial candidates in `u = (x-50)/50` coordinates:

1. the ordinary line `L(u)` through `p1, p2`;
2. `L(u) + kQ(u)` with `Q(u) = (u-u1)(u-u2)` — both tails up, both tails down;
3. `L(u) + kR(u)` with `R(u) = (u-u1)(u-u2)(u-m)`, `m = (u1+u2)/2` — left
   up/right down, left down/right up.

The scalar `k` is fit by least squares at the global padded glyph x-extents
(`xL = xmin - 5`, `xR = xmax + 5`) toward `ymax+5` / `ymin-5`, so a crossbar
seed can curve onto a leg and keep following the surface instead of escaping
immediately. These seeds replace the earlier four-point cubic interpolation,
which encouraged local immediate escape.

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

with normalized metrics (`coverage_fraction`, `bad_surface_fraction`,
`mean_surface_excess`) and the specified structural penalties (trace_penalty:
0 for exactly one finite non-empty trace, else 2.0 empty, 2.0 unbounded, +1.0
per extra component; tail_penalty: derivative roots outside the trace plus
left/right slope deficits normalized by 8). This replaced a lexicographic
tuple that started with `newly_covered`, where one extra covered point could
beat an arbitrarily disastrous surface/tail violation.

Feasible candidates are judged separately (single finite trace, >= 95 % trace
adherence, no derivative roots outside, exit slopes >= 8, newly >=
MIN_NEW_POINTS) and compared by `(newly_covered, -degree,
-mean_surface_distance)`. Every hill climb keeps the best feasible candidate
encountered and returns it regardless of the final exploratory state.

Candidate analysis is single-pass: `analyze_candidate` converts `u -> x`
once, computes the roots of `P-ymin`, `P-ymax` and `P'` once, derives the
trace components once, samples once, and derives every metric from those
cached values. Search geometry is cheap: a deterministic `SEARCH_BOUNDARY_MAX`
subset of the boundary and at most `SEARCH_GRAPH_MAX` trace samples per
hill-climb step; dense full-boundary evaluation is reserved for accepted
curves and final validation.

## Manual behavior for real letters (current)

The default pytest suite intentionally does not run real-letter fitting.
Manual acceptance is a separate development step:

```
uv run denysko A
```

Current results (default seed, on this environment/lockfile):

- `A` — fails: V1 coverage 0.0000 (no feasible curve found).
- `O` — fails: V1 coverage 0.0000.
- `V` — fails: V1 coverage 0.0000.
- `H` — fails: V1 coverage 0.0000.
- `S` — fails: V1 coverage 0.0000.

A synthetic steep vertical bar is solved end-to-end: two feasible degree-1
curves covering it, and `validate()` passes all of V1-V4. So the pipeline
(seeds, merit, feasibility, reduction, validation) is mechanically correct on
geometries where feasible curves exist; the failures below are search
convergence failures on real letter geometry, not validation failures.

## Concrete remaining convergence failure

The dominant blocker for straight-stroke letters (`A`, `V`, and most others)
is that **a single `y = f(x)` polynomial of degree <= 5 cannot trace a long
straight diagonal stroke and leave the band steeply at both ends**:

- A line hugging the leg has slope ~2.6-2.8, below `MIN_TAIL_SLOPE = 8`, so
  it is infeasible on tails alone.
- Bending the tails up (the both-up quadratic seed, `L + kQ`) with even a
  small `k` makes the polynomial re-enter the band a second time far outside
  the glyph (a parabola that rises steeply must come back down), so the
  single-finite-trace rule is violated and `trace_penalty` jumps to 1.0,
  pushing merit well below the shallow line's local optimum.
- The hill climb therefore sits on the line (merit ≈ -1.0, slope deficit
  ~1.05 of the tail penalty) and never crosses into the geometrically invalid
  basin that would be needed to reach a steep-exit feasible curve.

Concretely, for `A`'s right leg the best line seed covers ~77 boundary points
with surface 1.0 but tail slope 2.6; the four bent seeds all produce two trace
components (they cross the band at x ≈ -70..-3 and again across the leg).
Hand-constructing a quintic that hugs the leg and exits at |P'| = 8 gives
surface fraction only ~0.25-0.4 because the hook regions deviate off-surface.

Vertical strokes (thin bars) are handled well by steep line seeds; the
remaining frontier is **straight diagonal and horizontal strokes**, which
need the curve to transition to a perpendicular stroke or leave through the
glyph's own boundary geometry rather than through empty in-band space.

Per the task instructions, iteration budgets were not increased to chase
convergence. The documented next step is to give the search a way to build a
curve that follows a diagonal/horizontal stroke onto the connecting boundary
before escaping, rather than tuning restarts or steps.

## Default pytest runtime

`uv run pytest` completes the whole suite in roughly 0.6 s (the one-second
deadline in `tests/conftest.py` fires otherwise). It contains no stochastic
real-letter fitting and no alphabet sweep; alphabet evaluation is a manual
development benchmark run outside pytest.