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

## Known failures / measured state

NORMALIZED_SIZE=1.0 landed with full dimensional audit. All geometric
length constants scaled by /100; slope/dimensionless constants preserved.
CORRIDOR_EPS scaled 0.35->0.0035. ESCAPE_RATE confirmed slope-invariant.

Production: 50/52 letters pass end-to-end. T/m fail at stage-1 LP
FEASIBILITY, not tail proof: per-(route, degree, orientation)
instrumentation (normalized-z V3 landed first) shows T route 1 has a
stage-1 violation plateau ~0.046 for EVERY degree <= 24 and all four
orientations. Its corridor (bar + unfolded stem descent) demands a
y-dive of ~0.85 units within dx~0.055 - no polynomial of degree <= 24
threads that regardless of tolerance representation. The former
hypothesis "raw-x V3 causes T/m" is disproven; raw-x V3 removal was
still required for scale-equivariance. Real fix direction: corridor
centerline smoothing of unfolded vertical descents (documented future
work), not CORRIDOR_EPS loosening.

Test suite: 44/55 pass. The 10 failing tests use synthetic corridors
with old-scale coordinates (x in [10,60], ylo=0, yhi=100 etc.) that need
mechanical conversion to normalized [0,1] coordinates. This is test-data
work, not a design issue.

## Stacked-landmark walls: W fixed; Z/z still open (measured)

Root cause (proven by instrumentation, affects old b79ddd4 identically):
the vertical-unfold crawl parks ~100 consecutive landmarks at a single
frontier column, producing multi-valued corridor bounds at one x - a
"vertical wall" no single-valued polynomial can thread. The known-good
reference fails the SAME letters internally (its `denysko W` exits 0
but emits zero equations); "52/52" was never validated end-to-end.

Fix landed for W (and verified harmless for all other passing letters):
build_route_corridor now merges same-column landmark stacks into ONE
single-valued band per column - chain adjacent bands, clip to the
column's actual fill run, then re-sample chords whose interpolation
crosses unfilled space by inserting fill-clipped nodes. T/m/H/A xa/xb
shift <=0.0013 (T xb .6556->.6543); reference-facts test re-frozen.

Z/z were traced to a second, distinct mechanism (their routes traverse
long diagonals whose landmarks end up almost entirely parked at the
crawl frontier, so even after merging the surviving chord cuts across
empty space - containment miss ~0.4 at the bar-diagonal junction).
That mechanism is analyzed and fixed in the next section.

## Z/z fixed: unfold run-width gate (measured, issue #7)

Second, distinct mechanism (completing the W fix described above).
Instrumentation over the selected Z/z routes (`build_route_corridor`
landmark loop) shows the exact capture mechanism. The per-landmark
verticality test `|dx| < 0.25*dy` fires on raster staircase noise:
single-raster-step pairs with dx ~= 0.002 (=1/512) and dy ~= 0.009.
The unfold window is then derived from the CONTAINING ROW RUN of the
group; at bar/diagonal junctions that row run spans nearly the whole
glyph: measured chosen-run widths 0.78-0.82 on Z and z versus a local
stroke diameter of only ~0.11-0.18 (2 * distance-transform radius).
Landmarks are spread across that oversized window, the synthetic crawl
frontier lands up to ~0.5 to the right of the diagonal, and the
unfold-exit-overlap branch then pins ~86-99 consecutive landmarks at
the frontier (max |realized_x - raw_x| measured 0.513-0.525). The
resulting single-column chord crosses unfilled space and Phase-1
containment misses by ~0.40 (Z) / ~0.36 (z).

The generic discriminator is geometric, not glyph-specific: a GENUINE
vertical regime cuts its containing row run at approximately the local
stroke width, while a staircase diagonal embedded in a wider horizontal
structure cuts it at a multiple of that width. Measured ratios
(run width / local stroke diameter): W's legitimately unfolded
near-vertical strokes cut at 1.00-1.10x; Z/z staircase artifacts at
1.39-7.4x. build_route_corridor now gates every candidate unfold
group: if the narrowest containing row run exceeds
UNFOLD_RUN_WIDTH_GAIN = 1.25 times twice the maximum local stroke
radius over the group, the group is NOT unfolded - raw arc x is kept,
no frontier is set, and the diagonal stays a monotone atom.

Measured result: all Z/z route corridor violations drop from
0.403/0.357 to 0.0000; max realized-x shift on Z/z drops from ~0.52
to <= 0.002 (one raster step); `denysko Z`, `denysko z`, `denysko W`
all emit valid equations and exit 0. Full A-Z/a-z CLI sweep: 52/52
pass. W keeps its legitimate unfolds (69 vertical-unfold landmarks,
worst shift 0.046 < stroke width) and its merged-wall fix untouched.

Review hardening (PR #8): a width-gate-rejected apparent-vertical group
no longer bypasses placement entirely. All its raw points now flow
through the same frontier/catch-up path as ordinary nonvertical
landmarks (shared `_place_raw_point` helper), so after a prior
legitimate unfold an active synthetic frontier still yields strict
monotone overlap-exit crawl until raw x catches up, then exact raw x
resumes. Covered by a focused synthetic regression (stem unfold ->
wide-band rejected staircase -> crawl -> exact resume).
