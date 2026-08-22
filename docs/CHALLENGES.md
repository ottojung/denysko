# Denysko implementation notes: challenges observed

Status: the pipeline prescribed by `docs/SPEC.md` plus the refinement contract
is implemented (`src/denysko.py`) and mechanically validated by a committed
suite. After a round of correctness fixes (below), **7/26 letters pass V1-V4
at the default seed** (A B D J O Q V); the rest fail cleanly on V1 coverage.
No thresholds were weakened and no algorithm step was replaced.

## Correctness fixes applied first (as instructed)

These bugs predate any tuning questions:

- **Serialization was wrong in x.** Internal u-basis coefficients were printed
  directly as powers of x, and validation evaluated parsed expressions through
  the same u-substitution - so emitted polynomials drew the wrong shape while
  self-consistently passing their own checks. Serialization now substitutes
  `u = 0.02x - 1` via numpy `Polynomial`, parsing/validation operate on
  ordinary x-polynomials, and regression tests pin semantics
  (`100 + 100u` -> `y=2x`).
- **Fixed-point precision had to rise to 12 decimals.** At 6 decimals, the
  rounding of tiny high-order x-coefficients shifted y-values by thousands of
  units at x = 105 once expansion became semantic. At 12 decimals the error is
  bounded near 0.01 units.
- **Normalization** now maps the filled bbox lower-left corner to `(0, 0)`
  (it previously centered x at 50).
- **`sample_graph` cap truncation** dropped the right-hand side of steep
  graphs; it now evenly downsamples the full domain and keeps both endpoints.
- **V2 direction bug:** the per-curve escape check unpacked the wrong distance
  vector from `_min_dists` and was re-measuring boundary *coverage* instead of
  graph *escape*. Fixed; V2 is now exactly ">= 95% of each curve's sampled
  graph within TAU of the boundary", with V3 kept separate for bbox
  confinement (exact endpoint/extremum checks).

## Search reward change

The lexicographic score led with `-escaped_sample_count`, so shrinking into a
tiny zero-escape pocket always dominated any larger useful curve. Hill climbing
now scores `(newly_covered, -escape_penalty, -degree, -mean_distance)`, while
acceptance keeps the hard gate "at most 5% of the graph beyond TAU" plus the
8-new-points floor. An earlier trace showing candidates freezing at domain
width 0.23 for ~75 steps described the old score and no longer occurs; the
remaining failure mode is different (below).

## Where it stands now (default seed 0)

- Passing: A B D J O Q V. `uv run denysko A` emits semantically sensible
  curves: legs with slopes ~±2.8 spanning x≈[1.5,29] and [60,90], crossbar
  `y=25.68`, apex bar `y=99.90`.
- Failing letters stall on V1 coverage, e.g. C 0.83, E 0.42, H 0.53,
  I 0.30, M 0.42, S 0.51, X 0.92, Z 0.69.
- The near-vertical synthetic test passes end-to-end: three steep curves cover
  a thin vertical bar fully within V1-V4, using ordinary slopes of ±7-15 in
  x-space.

## Challenge 1: vertical strokes remain structurally hostile to y = f(x)

A single-valued y(x) cannot zigzag between the two edge-columns of a stem, so
each column needs its own stack of steep segments or one full-height hugger
(slope_dx ≈ 25-35 over a ~3-unit-wide domain). Such solutions exist and qualify
when seeded well (hand-forced full-column seeds give large coverage with clean
escape fractions). The solution space is not the problem.

## Challenge 2: uniform two-point seeding rarely produces those solutions

`p2` is chosen uniformly among all boundary points within distance 15 of
`p1`. For a stem point, same-column partners are roughly 5% of valid choices;
the rest pair across the stroke or around corners and yield misaligned slopes.
With only 8 restarts (+16 rescue), most rounds produce short or misaligned
segments worth 20-170 points instead of the ~200-500-point column huggers.
This remains the prime suspect for the failing letters, but per instruction it
has not been touched yet; the numbers above are the post-fix baseline to beat.

## Challenge 3: the curve budget arithmetic

V1 needs ~95% of |P| covered within at most 12 curves. For H (|P| ≈ 2704) that
is ~215 newly-covered points per accepted curve on average. Horizontal strokes
deliver that readily; vertical-heavy rounds usually do not, and rounds that
qualify with barely 8 new points burn scarce curve slots.

## Thoughts on possible remedies (not yet applied)

1. Bias `p2` toward locally-collinear pairs. Directly attacks challenge 2;
   smallest surgical change consistent with everything else.
2. Make coefficient sigma relative to coefficient magnitude. Attacks slope
   rotation, though coverage-first scoring already reduced the need.
3. Budget-only changes (more restarts/steps/curves) cannot fix a ~5%
   alignment rate within 12 curves and stay out of scope until seeding is
   addressed.
