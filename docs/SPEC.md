# SPEC: Single-Letter → Desmos Polynomial Converter

Version 3.0 · Status: draft · Scope frozen: **one letter per run, nothing else**

Version 3.0 is an architectural rewrite: topology is solved **before** polynomial
fitting. The pipeline is

```text
Phase 1  boundary contours -> x-monotone paths -> corridors -> selection
Phase 2  deliberately high-degree constrained polynomial fit per corridor
Phase 3  degree minimization inside the same corridor
Phase 4  independent validation (corridor adherence + global coverage)
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

## 3. Phase 1 — explicit boundary topology

**Paths.** Each ordered contour is split into maximal x-monotone chains by grouping
edge travel directions cyclically (every outline edge lands in exactly one chain,
including across the rotation seam). Decreasing-x chains are reversed. A path is
therefore a route a graph `y = f(x)` can follow: x never reverses. Near-vertical
geometry is kept as a narrow-span steep path rather than rejected; slivers narrow in
both x and y are dropped. Chains are arc-length resampled to bounded node counts.
Holes/counters produce their own paths (`O` yields outer 2 + inner 2).

**Coverage.** Each path records which rasterized boundary samples lie within
`TAU = 2` of it. Paths whose covered sets nearly duplicate an earlier one (Jaccard
> 0.85) are deduplicated to a canonical representative.

**Corridors.** Every path gets an allowed region: piecewise-linear `[lower(x),
upper(x)]` around its nodes. Width hugs surface tolerance but shrinks toward half the
distance to the nearest boundary sample NOT covered by this path (floored so it never
degenerates) — a corridor never merges two distinct nearby strokes.

**Escapes are corridors, not tail search.** Each end is classified:

- endpoint on a glyph x-edge → *far-field* rows only: the tail leaves the drawn
  region immediately; a few rows far outside forbid swinging back into the band;
- interior endpoint → *band-exit* rows: anchored at the endpoint, moving outward at
  `ESCAPE_RATE = 2.5` per unit x with a run proportional to the distance to the
  nearer band edge (linear ramps, no kink, no cliff). Direction is toward the nearer
  band edge.

All escape constraints are inequalities — never exact tail targets.

## 4. Phase 2 — deterministic selection

Greedy set cover picks at most 12 corridors maximizing newly covered boundary
samples; ties break by longer path then stable index. Selection consults zero
polynomial coefficients. If the union covers `< 95 %`, or more paths would be
needed than allowed, the tool reports the failure and exits nonzero.

## 5. Phase 3 — high-degree feasibility fitting

For each selected corridor the first question is "can a sufficiently high-degree
polynomial stay inside?": `INITIAL_FIT_DEGREE = 20`. Coefficients are solved in the
Chebyshev basis on `z ∈ [-1,1]`; corridor membership at sampled positions is linear
in the coefficients, so feasibility is a linear program (scipy HiGHS, minimizing the
worst row violation — a justified dependency, isolated behind `src/fitting.py`).
Constraint rows escalate deterministically when independent dense validation catches
inter-sample escapes.

## 6. Phase 4 — degree minimization

Binary search finds the lowest degree that stays inside the **same** corridor; the
neighbor below is verified infeasible. The corridor never moves, so reduction can
never change topology.

## 7. Phase 5 — validation

- Per curve: dense re-check of corridor bounds (surface adherence is largely
  automatic by construction).
- Global V1: ≥ 95 % of actual glyph boundary samples within `TAU` of at least one
  emitted polynomial's visible trace — computed from emitted polynomials, not path
  bookkeeping.
- V4: exact parse/format round-trip, finite coefficients `< 1e9`, no scientific
  notation, no domain restrictions.

Any failure ⇒ exit code 1, reasons on stderr, nothing on stdout.

The old single-global-trace rule (`trace(P)` exactly one finite interval) is retired:
topology lives in corridors now. The invariant is *"inside the glyph region the
polynomial stays inside its corridor; past the corridor it escapes and does not draw
visible spurious strokes."*

## 8. Determinism and diagnostics

No randomness remains; `--seed` is accepted with a value and ignored. Phase-by-phase
diagnostics (candidate path count, selected count, selected coverage, per-path
minimum degree) go to stderr. `denysko-debug paths|select|fit LETTER` inspects
phases individually.

## 9. Acceptance criteria

- All 26 letters pass V1–V4 with defaults.

  > This remains the long-term target; current limitations (inner-counter tails,
  > residual coverage gaps near tight hooks such as `C`) are documented factually in
  > CHALLENGES.md.

- Recognizable outlines without large spurious strokes.
- Runtime ≤ 30 s per letter.
- Dependencies: uv, Python ≥ 3.11, numpy, matplotlib, scipy (LP solver).
