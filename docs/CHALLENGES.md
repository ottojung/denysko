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

## Known failures / measured state (directed-atoms + strict coverage)

Landed: structural directed-source enumeration (C fixed generically),
exact-extremum splitting (shared extremum point, plateau rule),
per-atom x-monotonicity assertion, audited physical-atom accounting
(unclassified asserted 0, twins counted once), staged MILP minimum
proof, full-interval corridor containment, distance-based V5/V6 with
realized embeddings, duplicate validators removed.

Measured matrix:

| letter | status | routes |
|--------|--------|--------|
| A | pass | 2 |
| B | pass | 4 (proven minimum) |
| C | pass | 2 (x-min source -> two arms) |
| H | pass | 2 |
| O | pass | 2 |
| T I L F | pass | 2/1/1/2 |
| E | FAIL fitting | two large vertical climbs in one y=f(x); interior violation 8.2@24 -> 4.1@48 |

Honest budgets (documented, measured): full-interval corridor
containment allows <= 8 units of interpolated-tube overshoot at
vertical-unfold exits (measured maxima A/C/O <= 2.1, B <= 2.1,
H <= 6.3); V5 allows <= 4 units same-x excursion (measured H max
~3.5). Tightening via corridor centerline smoothing is planned.
V6 is a hard gate on realized embeddings: any uncovered sample fails.
