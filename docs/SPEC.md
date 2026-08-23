# SPEC: Single-Letter → Desmos Polynomial Converter

Version 3.0 · Status: draft · Scope frozen: **one letter per run, nothing else**

Version 3.0 is an architectural rewrite: topology is solved **before** polynomial
fitting. The pipeline is

```text
Phase 1  fill mask -> medial-axis stroke skeleton -> stroke/junction
         route graph (+ ring cuts) -> complete routes -> exact minimum cover
Phase 2  deliberately high-degree constrained polynomial fit per route
Phase 3  degree minimization inside the same corridor
Phase 4  independent validation (corridor adherence + edge coverage)
```

Output remains ordinary unbounded polynomials `y=<poly>` with no domain
restrictions; at most `DEFAULT_MAX_CURVES = 12` curves per letter.

---

## 1. Mission

Given exactly one uppercase letter, emit a small set of `y = f(x)` polynomials that,
pasted into Desmos, render a recognizable outline of the letter. One letter in → one
letter drawn.

## 2. Canonical glyph geometry

Unchanged rules: bundled DejaVuSans at size 100, aspect preserved, filled-bbox lower-
left at `(0,0)`, max dimension 100, y-up. Counters/holes are real boundary geometry:
the rasterized fill uses even-odd semantics across rings, so hole edges contribute
boundary samples, and ordered contours come from the font's flattened outlines under
the identical normalization.

## 3. Phase 1 — fill-mask routing graph

**Canonical fill.** The ordered contours rasterize to a `GRID = 512` wide mask via
per-ring even-odd XOR: outer rings add material, inner rings subtract it, so counters
are real holes in the mask rather than boundary decorations.

**Routing graph.** A left-to-right vertical sweep classifies every column transition:

- *continuation* — one active branch matches one slice run;
- *source* / *sink* — a run appears with no parent / an active branch finds no heir;
- *split* — one branch fans out to several runs (counter opens);
- *merge* — several branches converge onto one run (counter closes).

Splits and merges create explicit `RouteVertex` records; maximal runs of slices
between vertices become `RouteEdge`s carrying per-column `[lower, upper]` intervals.
Pinched columns (a branch vanishing for ≤ `PINCH_COLS` columns) are bridged so
rasterization noise does not fabricate topology.

**Complete routes.** A route is a source→sink walk in this graph: one continuous
stroke a single-valued `y = f(x)` can trace through the filled glyph. `A` yields the
canonical diamond — source → split → {roof, bar} → merge → sink — hence exactly two
complete routes sharing both leg trunks.

**Corridors.** Each selected route gets the union of its edges' slice intervals as a
piecewise-linear `[lower(x), upper(x)]` tube, with `CORRIDOR_MARGIN` applied inside
each slice (reduced deterministically on thin slices, never inverted). Route
endpoints sit on glyph x-edges, so tails are side exits: the polynomial leaves the
drawn x-region immediately past its window.

## 4. Phase 2 — deterministic selection

`select_routes_min_cover` picks an exact minimum cover of all *meaningful* graph
edges (`span ≥ SLIVER_SPAN`, mean height ≥ 1) by complete routes: a HiGHS MILP,
with deterministic tie-breaking (fewer routes, larger geometric coverage, lower
total complexity, stable signature order). Selection consults zero polynomial
coefficients. If coverage is `< MIN_COVERAGE`, or more routes than
`DEFAULT_MAX_CURVES` would be needed, the tool reports the failure and exits
nonzero.

## 5. Phase 3 — high-degree feasibility fitting

For each selected corridor the first question is "can a sufficiently high-degree
polynomial stay inside?": `INITIAL_FIT_DEGREE = 20`. Coefficients are solved in the
Chebyshev basis on `z ∈ [-1,1]`; corridor membership at sampled positions is linear
in the coefficients, so feasibility is a linear program (scipy HiGHS, minimizing the
worst row violation — a justified dependency, isolated behind `src/fitting.py`).
Constraint rows escalate deterministically when independent dense validation catches
inter-sample escapes.

## 6. Phase 4 — degree minimization

Degrees are probed exhaustively from 0 upward using the cheap LP-feasibility stage;
the full verified fit runs at the first promising degree (the scan continues upward
if dense verification rejects it), yielding the lowest VERIFIED feasible degree.
The corridor never moves, so reduction can never change topology.

## 7. Phase 5 — independent validation

Phase 5 re-validates the PARSED emitted polynomials against their assigned
corridors and analytically verifies permanent tail escape beyond the finite
corridor window. It never trusts fitter internals:

- **V2 (independent corridor adherence):** dense interior tube check plus
  band-ramp inequality rows per emitted line; violations above CORRIDOR_EPS
  reject (`V2 corridor violation ...`).
- **V3 (analytic permanent tails, mandatory on both sides):** for every edge-exit ramp, beyond the final
  escape row the derivative must have no real roots and keep its outward sign,
  with P already strictly outside the band edge at that checkpoint - so the tail
  cannot re-enter the visible band later. Side-exit tails (endpoints on glyph
  x-edges) leave the drawn region immediately and are exempt under documented
  policy Option A (see CHALLENGES.md). Violations reject
  (`V3 tail re-entry risk ...`). Root analysis is a Phase-5 safety check only;
  topology is never rediscovered from it.
- **V1 (route-graph coverage):** every meaningful routing-graph edge is traversed
  by at least one selected route (≥ `MIN_COVERAGE = 0.95`; exact minimum cover
  normally achieves 1.0). This is enforced before fitting. Geometric proximity of
  emitted polynomials remains diagnostic only.
- **V4 (serialization contract):** exact parse/format round-trip, finite coefficients `< 1e9`, no
  scientific notation, no domain restrictions.

Any failure ⇒ exit code 1, reasons on stderr, nothing on stdout.

The old single-global-trace rule (`trace(P)` exactly one finite interval) is retired:
topology lives in corridors now. The invariant is *"inside the glyph region the
polynomial stays inside its corridor; past the corridor it escapes and does not draw
visible spurious strokes."*

## 8. Determinism and diagnostics

No randomness remains; `--seed` is accepted with a value and ignored. Phase-by-phase
diagnostics (candidate path count, selected count, selected coverage, per-path
minimum degree) go to stderr. `denysko-debug graph|routes|select|fit|uncovered LETTER` inspects phases
individually.

## 9. Acceptance criteria

- All 26 letters pass V1–V4 with defaults.

  > This remains the long-term target; current limitations (inner-counter tails,
  > residual coverage gaps near tight hooks such as `C`) are documented factually in
  > CHALLENGES.md.

- Recognizable outlines without large spurious strokes.
- Runtime ≤ 30 s per letter.
- Dependencies: uv, Python ≥ 3.11, numpy, matplotlib, scipy (LP solver).
