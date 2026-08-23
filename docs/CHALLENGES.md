# Denysko implementation notes — corridor architecture

Status: the pipeline is the Phase 1–4 architecture described in
`docs/SPEC.md` (topology first, then corridor-constrained Chebyshev
fitting, then degree minimization, then independent validation). The
stochastic polynomial-first search is deleted. The default pytest suite
runs in well under one second and contains no stochastic optimizer and
no real-letter fitting.

## Manual results (default settings)

| letter | phase 1 | fitting | result |
|--------|---------|---------|--------|
| `V` | 2 candidate paths, both selected, path-coverage 1.0000 | degrees 6 and 14 | **pass**, V1 satisfied |
| `A` | 4 candidates (incl. counter), all selected, coverage 1.0000 | degrees 6, 14, 6, 6 | **pass** |
| `O` | 4 candidates (outer 2 + inner hole 2), all selected, coverage 1.0000 | degrees 8, 12, 16, 8 | **pass** |
| `C` | 4 candidates, all selected, path-coverage 0.9550 | all four fit (degrees 12, 7, 7, 16) | fails V1 at 0.8918 |

Holes and counters are now first-class: `O`'s inner ring and `A`'s
counter are ordinary selected corridors, refuting the old
"inner contours are inherently unrepresentable" conclusion.

## Remaining limitations (measured, not generic)

1. **`C` emitted-trace coverage 0.8918 < 0.95.** All four corridors fit,
   but the emitted traces cover only ~89 % of the dense boundary cloud.
   The gap sits near the hook tips: the path nodes are arc-length
   resampled from coarse font polygons, so within-`TAU` path coverage
   (95.5 %) already starts below target and the fit's visible trace
   loses a little more at the mouth. Candidate fixes: densify path
   endpoints/append exact tip nodes before resampling; or add a
   per-path endpoint-inclusion rule to coverage assignment.
2. **Inner-counter tails cross the letter once.** An enclosed hole has
   no tail exit that avoids the glyph entirely; each inner-path tail
   crosses a stroke at a shallow angle exactly once before escaping.
   The corridor confines that crossing to leave through the band edge;
   visually this reads as part of the outline, but it is a real
   property worth knowing when inspecting `O`/`A` output.
3. **One path = one polynomial.** No sharing/opportunistic multi-
   corridor coverage yet; letters needing more granularity than 12
   pieces would require splitting paths further.

## Removed with the old architecture

Random restarts, hill-climbing refinement, coefficient mutation,
bent-seed families (`line + aQ + bR`), merit landscapes trading
coverage against surface/tail penalties, trace-component exploration
penalties, and stochastic tail search are gone. Escape behaviour lives
in corridor inequalities; topology lives in explicit paths. The scalar
exploratory merit and its weight table no longer exist.

The earlier conclusions about global-trace semantics ("inner contours
are inherently unrepresentable"; "x-monotonicity limits connected
routes") described the retired single-global-trace model and no longer
apply: x-monotone decomposition splits reversing routes into separate
paths, and holes have their own corridors.

## Default test suite

23 tests, ~0.6 s wall, hard 1-second session deadline retained
(`tests/conftest.py`). Covers: monotone decomposition (rectangle /
C-split / holes), corridor containment + parallel exclusion +
escape-direction rows, synthetic greedy set cover, feasibility on a
`0.9x..1.1x` corridor, impossible low-degree corridor rejection,
binary-searched degree minimization inside a fixed corridor,
out-of-corridor rejection, upward escape divergence of the fitted
polynomial, independent V1 flagging, serialization round-trips
(including tiny-coefficient positional formatting without scientific
notation), and the CLI contract (`--seed` accepted-and-ignored).
