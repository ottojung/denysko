# Denysko implementation notes: challenges observed

Status: the pipeline prescribed by `docs/SPEC.md` plus the refinement contract is
fully implemented (`src/denysko.py`) and mechanically sound, but **only ~6–9 of
the 26 letters pass V1–V4 at defaults**, depending on seed. This document records
what happens, why, and what I think about it. No thresholds were weakened and no
algorithm step was replaced.

## Where it stands

- With default seed 0 (and an interim linear reading of the sigma schedule),
  9/26 letters passed: A B O P Q T V W Y.
- After switching to the geometrically decreasing sigma schedule, a sweep of
  default-seed candidates 0..23 topped out at 9/26. The failing set is stable:
  C D E F G H I J K L M N R S U X Z.
- Failures are always clean: V1 coverage below 0.95, stderr only, exit 1.
  Runtime is fine (< 1.5 s/letter, budget 30 s).

## Challenge 1: vertical strokes are structurally hostile to y = f(x)

A curve can never zigzag between the two edge-columns of a stem (x(y) would have
to be non-monotone). Each side of each stem therefore needs its own coverage.
The efficient shape is one near-vertical hugger per column: slope_dx ≈ 25–35,
domain width ≈ 3 units, covering an entire ~100-unit column within tau = 2.
Those solutions exist and score beautifully — forcing such a seed by hand gives
`escaped == 0` and up to ~500 newly covered points in one shot. The solution
space is not the problem.

## Challenge 2: the prescribed seeding almost never produces those solutions

Seeding picks `p2` uniformly among all boundary points within distance 15 of
`p1`. For a stem point, same-column pairs are roughly 5% of valid partners; the
other 95% pair `p1` with points across the stroke or around corners, producing
misaligned slopes. Empirically only ~10% of seeded steep candidates end up
column-aligned, so "jackpot" rounds (full-column curves worth 300–500 points)
are rare; typical rounds yield 20–170 points.

## Challenge 3: refinement cannot rotate a line onto a column

Coefficients live in u-space where a full-height hugger needs slope_u ≈ 1250.
The coefficient mutation sigma is absolute and tops out at 10, i.e. ≤1% of the
needed slope. Rotating a misaligned seed onto its column would need hundreds of
consecutive maximal accepted steps in one direction; single-coordinate greedy
hill climbing cannot do it. Only domain widening/translating remain useful, and
those require an already-aligned seed (challenge 2).

Traced concretely: a steep seed with escaped=83 first *shrinks* its domain
0.78 -> 0.23 wide to reach `escaped == 0`, then freezes for the remaining ~75
steps — any widening re-introduces escapes, and the lexicographic rule rejects
everything that does not strictly improve.

## Challenge 4: the curve budget arithmetic does not close

H has |P| = 2704; V1 needs 2569 covered within 12 curves, i.e. ~214 newly
covered points per accepted curve on average. Horizontal strokes deliver
100–370 per curve, but vertical-heavy rounds often qualify with 20–30 (the
acceptance floor is only 8), burning scarce curve slots. Best observed totals
for H: 49–74% coverage. M, N, U behave similarly; I sometimes stalls at 2
curves / 19%.

## Smaller observations

- Sigma schedule wording matters: the interim linear interpolation
  (10 -> 0.2 / 5 -> 0.1) outperformed the faithful geometric one (large mid-run
  noise occasionally kicks candidates out of pockets). T W X U Y lost passes
  when switching. Kept geometric for contract fidelity.
- Everything else behaves exactly as specified: deterministic outputs for fixed
  seeds, holes fall out of the rasterizer naturally (O's inner ring verified),
  serialization round-trips, and the validation gate refuses to emit anything
  imperfect.

## Thoughts on possible remedies (none implemented; each breaks a frozen rule)

1. Bias `p2` toward locally-collinear pairs (e.g. prefer pairs whose segment
   direction matches p1's neighbourhood). Directly attacks challenge 2.
   Violates "choose one uniformly at random".
2. Make coefficient sigma relative to coefficient magnitude (or mutate in a
   log-scaled space). Attacks challenge 3. Violates the fixed 10.0 -> 0.2
   schedule.
3. More restarts / steps / a higher default --max-curves. Pure budget; helps
   but cannot fix a ~5% jackpot rate within 12 curves. Violates frozen counts.
4. Trigger the rescue round whenever the best qualifier is weak (not just when
   none exists), or keep searching instead of stopping. Violates section 7's
   stop condition.

My assessment: the algorithm as frozen relies on lucky alignment for exactly
the letter shapes that dominate the alphabet's difficulty. If the goal is all
26 passing, remedy 1 or 2 is the smallest, most surgical change; everything
else in the pipeline is already in place and validated.
