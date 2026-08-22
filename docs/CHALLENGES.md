# Denysko implementation notes: challenges observed

Status: the pipeline is implemented per `docs/SPEC.md` around **unbounded
polynomials** (`y=<poly>`, no Desmos domain restriction). The default pytest
suite is fast (target < 1 s, enforced by a wall-clock deadline in
`tests/conftest.py`; currently ~0.7 s) and contains no stochastic real-letter
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

## Seed construction: bent lines preserving slope

Seeds are built from the local-contour `p1, p2` pair (segment-scored,
tie-break toward the more distant partner, `p2` must be at least
`MIN_X_SEPARATION = 0.1` in x away from `p1`). Each restart gets exactly five
initial candidates in `u = (x-50)/50` coordinates:

1. the ordinary line `L(u)` through `p1, p2`;
2. two **degree-4 same-tail seeds** `L(u) + kQ(u)` with
   `Q(u) = (u-u1)² (u-u2)²` — both tails up, both tails down;
3. two **degree-5 opposite-tail seeds** `L(u) + kR(u)` with
   `R(u) = (u-u1)² (u-u2)² (u-m)`, `m = (u1+u2)/2` — left up/right down,
   left down/right up.

The squared bases vanish with zero derivative at `u1` and `u2`, so every bent
seed preserves both the seed values and the local stroke slope
(`P(u_i) = p_i.y`, `P'(u_i) = L'(u_i)`). `k` is fit by least squares at the
global padded glyph x-extents toward `ymax+5` / `ymin-5`. This replaces the
earlier `(u-u1)(u-u2)` / `(u-u1)(u-u2)(u-m)` bases, which preserved the seed
y-values but changed their local slopes.

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

The tail_penalty component follows the post-exit rules:
derivative-root penalty plus per-side missing-margin penalty (finite, for
tails that never reach the ±5 margin), x-run penalty beyond `MAX_TAIL_X_RUN`,
and slope deficit at the margin point. The coverage/surface/trace/degree
weights are intentionally untouched.

Feasible candidates are judged separately (single finite trace, >= 95 % trace
adherence, no derivative roots outside, both tails reach the ±5 margin within
`MAX_TAIL_X_RUN` with `|P'| >= MIN_TAIL_SLOPE` there, newly >= MIN_NEW_POINTS)
and compared by `(newly_covered, -degree, -mean_surface_distance)`. Every hill
climb keeps the best feasible candidate encountered and returns it regardless
of the final exploratory state.

Candidate analysis is single-pass: `analyze_candidate` converts `u -> x`
once, computes the roots of `P-ymin`, `P-ymax`, `P'`, `P-(ymin-5)` and
`P-(ymax+5)` once, derives trace components and post-exit tail geometry once,
samples once, and derives every metric from those cached values. Search
geometry is cheap: a deterministic `SEARCH_BOUNDARY_MAX` subset of the
boundary and at most `SEARCH_GRAPH_MAX` trace samples per hill-climb step;
dense full-boundary evaluation is reserved for accepted curves and final
validation.

## Manual behavior for real letters (current)

The default pytest suite intentionally does not run real-letter fitting.
Manual acceptance is a separate development step:

```
uv run denysko C
```

Current results (default seed, on this environment/lockfile). `fit_curves()`
finds **no feasible first curve** for all real letters tried; `validate()`
then reports V1 = 0. The zero-curve diagnostics (printed to stderr) give the
best failed restart's breakdown:

- `C` — no feasible first curve. Best restart: surface=0.40,
  trace_components=2, both tails report "no +-5 margin root", new=0.
- `V` — no feasible first curve. Best restart: surface=0.81,
  trace_components=2, both tails "no +-5 margin root", new=0.
- `A` — no feasible first curve. Best restart: surface=1.00,
  trace_components=2, both tails "no +-5 margin root", new=0.
- `O` — no feasible first curve. Best restart: surface=0.31,
  trace_components=1, tails ok, new=0.

Two distinct failure modes remain:

1. **Bent seeds still produce multi-component traces** on connected outer
   strokes (`A`, `V`, `C`): the degree-4/5 bends, fitted at the global padded
   x-extents, cross the band a second time, so `trace_penalty >= 1.0` and the
   seed is structurally unusable. The hill climb then cannot reach a
   single-trace feasible curve from these starts.
2. **The line seed is a merit local optimum**: for `A`/`V`, a line hugging a
   straight diagonal stroke has surface ~1.0 but never reaches the ±5 margin
   steeply enough on the shallow exit side, and the bent alternatives are
   multi-component. For `O`, the best failed restart is single-trace with
   valid tails but surface 0.31 — the seed does not follow the rounded loop.

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
focuses on connected outer geometry (`C` first).

Per the task instructions, iteration budgets were not increased to chase
convergence; the failures above are reported as measured.

## Default pytest runtime

`uv run pytest` completes the whole suite in roughly 0.7 s (the one-second
deadline in `tests/conftest.py` fires otherwise). It contains no stochastic
real-letter fitting and no alphabet sweep; alphabet evaluation is a manual
development benchmark run outside pytest.