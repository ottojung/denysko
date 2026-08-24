# Denysko implementation notes — corridor architecture, hardening pass

Status: the Phase 1–4 corridor architecture is intact and hardened.
Default suite: 23 tests in ~0.85 s (1-second session deadline retained),
no stochastic optimizer, no real-letter fitting.

## Manual results after this pass (default settings)

| letter | candidates | selected | phase-1 cov | min degrees | result |
|--------|-----------|----------|-------------|-------------|--------|
| V | 2 | 2 | 1.0000 | 6, 12 | **pass** |
| A | 4 | 4 | 1.0000 | 6, 12, 8, 0 | **pass** |
| O | 10 | 9 | 0.9753 | 4,4,2,2,1×5 | **pass** |
| C | 9 | 8 | 0.9775 | 3,3,2,2,1×4 | fails V1 at 0.9050 |
| H | 12 | 10 | 0.9641 | 1,1,0,0,1,1,1,1,3,3 | **pass** |
| I | 4 | 3 | 0.9585 | 1, 1, 0 | **pass** |
| E | 12 | 9 | 0.9574 | 1,0×5,1,1 | **pass** |
| F | 10 | 8 | 0.9622 | 1,0,1,0,0,0,1,3 | **pass** |
| L | 6 | 5 | 0.9771 | 1,1,0,0,3 | **pass** |
| T | 8 | 6 | 0.9597 | 1,1,0,0,0,4 | **pass** |

All passing letters also clear V2 (corridor adherence of parsed
polynomials) and V3 (analytic tail re-entry) independently.

## Vertical-edge bug and fix

`_max_x_monotone_chains` skipped edges with `dx == 0`, silently
dropping exact vertical contour segments (rectangle sides, I/H/E/F/L/T
stems). Fixed: maximal runs of exactly-vertical edges become one narrow
x-monotone path each, sweeping VERTICAL_PATH_X_SPAN = 0.5 across the
run's full y extent; near-vertical runs (local slope steeper than
STEEP_RUN_TAN = 3) get the same treatment, which also fixed O/A/C arc
tips whose vertical tangents previously forced degree-20 polynomials
into ±2+ violations. Paths now carry source_edge_ids provenance, making
the invariant "every non-degenerate contour edge belongs to exactly one
path" directly testable (`contour_edge_count` helper). The vertical
fix is what brought C's hooks, T's stem, and the vertical-heavy
letters above into fitting range at all.

## Production-LP smoke coverage

Previously every feasibility test disabled USE_LP, so production-path
regressions slipped through green tests. There is now an unmocked
smoke test (`test_production_lp_smoke`) running constraint
construction → scipy HiGHS LP → polish → dense validation on a tiny
synthetic corridor with real grid sizes.

## Solver tolerance and margin split

CORRIDOR_EPS = 0.35 replaces the old 0.25·TAU acceptance gate. It is a
solver-numerics tolerance, deliberately not a fraction of TAU:
measured degree-24..32 Gibbs floors at serif corners are ~0.13–0.28 on
glyph-wide windows, so 0.35 is the smallest clearly justified value.
CORRIDOR_MARGIN = 0.4 reserves room inside corridors so that worst-case
emitted deviation stays within actual TAU (tube half-width ≤ 1.6, plus
0.35 ≤ 1.95 < 2).

## Degree minimization

min_degree verifies hi, binary-searches an approximate bound, then
sweeps deterministically downward to the first failure - returning the
lowest verified feasible degree even under non-monotone numerics. The
old minimum-degree test asserted nothing meaningful
(`below.degree > best.degree`); it now checks that every degree below
the returned minimum is infeasible.

## Inner contours: representation vs escape

- inner-contour path representation: **solved** (O inner ring and A
  counter produce candidate corridors that are selected and fitted);
- clean tail escape without crossing unrelated glyph geometry: **not
  generally possible** for enclosed counters under unbounded y=f(x)
  output. This is a topological consequence of continuous graphs, not
  a search failure.

Validator policy (documented Option A): an inner-contour polynomial
may cross unrelated glyph geometry during its escape provided the
crossing lies outside its intended path domain and the tail leaves
permanently (verified analytically for edge-exit ramps). Side-exit
tails leave the drawn x-region immediately and carry no band rows this
iteration.

## Known failures / measured state (local vertical unfolding)

Landed: LOCAL vertical unfolding. A vertical atom spreads only across
its OWN stroke width; any overlap with following real geometry decays
as microscopic crawl steps above the synthetic frontier, and EXACT raw
x resumes once raw catches up. No persistent downstream translation.
Per-sample provenance (raw_x/raw_y/realized_x/deform) stored on every
corridor; R1 gate asserts nonvertical atoms resume exact skeleton x
(tolerance 0.5 = ~2.5 raster steps); V5 uses contiguous fill runs so
counters are never "filled"; V6 strict on realized embeddings.

Measured matrix (all pass end-to-end):

| letter | curves | degrees | notes |
|--------|--------|---------|-------|
| A | 2 | 4, 6 | |
| B | 4 | 7,6,6,5 | proven minimum 4 |
| C | 2 | 9, 9 | x-min source -> two arms |
| H | 2 | 9, 9 | stems unfold ~6 units each within own width; crossbar real-x preserved; was deg-16 garbage under persistent shift |
| O | 2 | 7, 8 | |
| T | 2 | 5, 12 | |
| I | 1 | 1 | |
| L | 1 | 5 | |
| F | 2 | 6, 5 | |
| E | 3 | 5, 6, 5 | previously infeasible — local unfolding fixed it |

H semantic geometry is regression-tested: each route realizes a stem
traversal (dy > half glyph, unfolded dx > 1), a long crossbar run, and
the opposite stem. `denysko-debug realize H` prints per-atom raw vs
realized geometry.
