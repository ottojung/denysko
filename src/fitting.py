"""Phase 2/3: corridor-constrained polynomial fitting.

For a fixed corridor the fitting problem is linear in the polynomial
coefficients: at sampled positions x_i the corridor demands
lower(x_i) <= P(x_i) <= upper(x_i). Because escape ramps are part of
the corridor geometry (extended path nodes outside the glyph band),
every constraint - surface adherence and outward escape alike - is an
ordinary two-sided interval row.

Coefficients are solved in the Chebyshev basis on z in [-1,1]
(numerically stable) via least squares followed by deterministic cyclic
projections (POCS) onto the row intervals - no stochastic search. The
first fit is deliberately high-degree (INITIAL_FIT_DEGREE): feasibility
first, elegance later. Degree minimization re-solves the SAME corridor
with fewer coefficients; topology never changes because the corridor is
fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from numpy.polynomial import chebyshev as cheb

from src.topology import (
    Corridor, TAU, ESC_OFFSETS, ESCAPE_RATE, CORRIDOR_MARGIN, CORRIDOR_EPS,
    escape_bound_at,
)

INITIAL_FIT_DEGREE = 24   # measured: deg-20 rings ~0.34 inside
                          # tight tubes on wide windows; 24 halves it
FIT_GRID = 128           # constraint samples across the whole window
DENSE_GRID = 900         # validation samples (denser than fitting)
POCS_SWEEPS = 240
FEAS_TOL = 1e-6
USE_LP = True   # unit tests may disable for speed (pure POCS)


@dataclass
class PathFit:
    """A polynomial fitted to one corridor (topology fixed)."""
    corridor: Corridor
    degree: int
    coef_cheb: np.ndarray            # Chebyshev coefficients in z
    poly: np.polynomial.Polynomial   # ordinary powers of x
    dense_max_violation: float


def _zmap(x, xa: float, xb: float):
    return (2.0 * np.asarray(x, dtype=float) - xa - xb) / (xb - xa)


def _constraint_set(corridor: Corridor, degree: int,
                    n_int: int = FIT_GRID, n_esc: int = 40):
    """Build the full deterministic constraint system (A, lo, hi).

    Interior rows are two-sided interval constraints across the path
    domain. Band escapes contribute one-sided rows sampled along their
    whole continuous ramp; far-field rows are included exactly.
    """
    xs_int = np.linspace(corridor.xs[0], corridor.xs[-1], n_int)
    lo_i = corridor.lower_at(xs_int)
    hi_i = corridor.upper_at(xs_int)

    esc_xs, esc_lo, esc_hi = [], [], []
    for spec in corridor.escapes:
        if spec.kind != "band" or not spec.rows:
            continue   # side-exit: no fitting rows (Phase 5 checks pads)
        # Sample the continuous ramp bound densely so the polynomial
        # cannot swing between discrete rows.
        sign = -1.0 if spec.side == "L" else 1.0
        run = max(abs(r[0] - spec.x_end) for r in spec.rows)
        xs_e = spec.x_end + sign * np.linspace(
            min(ESC_OFFSETS[0], 0.5), run, n_esc
        )
        bnd = escape_bound_at(spec, xs_e)
        esc_xs.extend(xs_e)
        esc_lo.extend(bnd if spec.sigma == 1 else
                      np.full(len(xs_e), -np.inf))
        esc_hi.extend(bnd if spec.sigma == -1 else
                      np.full(len(xs_e), np.inf))

    all_x = np.concatenate([xs_int, np.asarray(esc_xs)])
    A = cheb.chebvander(_zmap(all_x, corridor.xa, corridor.xb), degree)
    lo = np.concatenate([lo_i, np.asarray(esc_lo)])
    hi = np.concatenate([hi_i, np.asarray(esc_hi)])
    return A, lo, hi


def _project_sequential(A, lo, hi, c0, sweeps):
    """Plain cyclic projection (used directly when USE_LP is disabled)."""
    c = c0.copy()
    norm2 = np.einsum("ij,ij->i", A, A)
    norm2[norm2 == 0] = 1.0
    for _ in range(sweeps):
        v = A @ c
        if not np.all(np.isfinite(v)):
            return c, float("inf")
        worst = max(0.0, float((lo - v).max()), float((v - hi).max()))
        if worst <= FEAS_TOL:
            return c, 0.0
        if worst > 1e8 or not np.isfinite(worst):
            return c, float("inf")
        for i in range(len(v)):
            if v[i] < lo[i]:
                c += (lo[i] - v[i]) * A[i] / norm2[i]
                v[i] = lo[i]
            elif v[i] > hi[i]:
                c -= (v[i] - hi[i]) * A[i] / norm2[i]
                v[i] = hi[i]
    v = A @ c
    if not np.all(np.isfinite(v)):
        return c, float("inf")
    return c, max(0.0, float((lo - v).max()), float((v - hi).max()))


def _project_feasible(A, lo, hi, c0, sweeps=None):
    """Deterministic feasibility solve.

    Production uses scipy's HiGHS LP (minimize max row violation).
    With USE_LP disabled (fast unit tests) plain cyclic projection runs
    instead.
    """
    if not USE_LP:
        return _project_sequential(A, lo, hi, c0, sweeps or POCS_SWEEPS)

    m, n = A.shape
    # variables: [c (n), s (1)]
    # A c - s <= hi ; -A c - s <= -lo ; s >= 0
    A_ub = np.zeros((2 * m, n + 1))
    A_ub[:m, :n] = A
    A_ub[:m, n] = -1.0
    A_ub[m:, :n] = -A
    A_ub[m:, n] = -1.0
    b_ub = np.concatenate([hi, -lo]).astype(float)
    b_ub[~np.isfinite(b_ub)] = np.inf
    # drop infinite rows
    keep = np.isfinite(b_ub)
    cost = np.zeros(n + 1)
    cost[n] = 1.0
    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(cost, A_ub=A_ub[keep], b_ub=b_ub[keep], bounds=bounds,
                  method="highs")
    if not res.success or res.x is None:
        return np.zeros(A.shape[1]), float("inf")
    s = float(res.x[n])
    return np.asarray(res.x[:n]), s


def _dense_violation(corridor: Corridor, coef: np.ndarray,
                     grid: int = DENSE_GRID) -> float:
    """Dense validation against the corridor.

    Interior bounds are checked on a dense grid; band-escape regions are
    checked against their continuous ramp bound via the row rule
    violation = max(0, sigma * (bound - P)). Returns the worst
    violation.
    """
    viol = 0.0
    # Interior/path-domain adherence only. Pad strips beyond the domain
    # are governed by the Phase-5 tail policy (analytic re-entry check
    # for edge-exit ramps; visibility check for side exits).
    xs = np.linspace(corridor.xs[0], corridor.xs[-1], grid)
    vals = cheb.chebval(_zmap(xs, corridor.xa, corridor.xb), coef)
    lo = corridor.lower_at(xs)
    hi = corridor.upper_at(xs)
    d = np.maximum(lo - vals, vals - hi)
    viol = max(viol, float(d.max()))
    for spec in corridor.escapes:
        if spec.kind != "band" or not spec.rows:
            continue
        sign = -1.0 if spec.side == "L" else 1.0
        run = abs(spec.rows[-1][0] - spec.x_end)
        xs_e = spec.x_end + sign * np.linspace(
            min(ESC_OFFSETS[0], 0.5), run, max(60, grid // 4)
        )
        bnd = escape_bound_at(spec, xs_e)
        d_e = np.maximum(
            0.0, spec.sigma * (bnd - cheb.chebval(_zmap(xs_e, corridor.xa,
                                                    corridor.xb), coef))
        )
        viol = max(viol, float(d_e.max()))
    return viol


def _weighted_init(corridor: Corridor, degree: int):
    """Least-squares start tracking the path centerline, so the initial
    polynomial already has roughly the right shape across the corridor
    and the LP only needs small corrections."""
    xs_p = corridor.path.points[:, 0]
    ys_p = corridor.path.points[:, 1]
    z = _zmap(xs_p, corridor.xa, corridor.xb)
    A = cheb.chebvander(z, degree)
    coef, *_ = np.linalg.lstsq(A, ys_p, rcond=None)
    return coef


def fit_degree(corridor: Corridor, degree: int) -> PathFit | None:
    """Fit one polynomial of exactly this degree inside the corridor.

    Deterministic pipeline: LP feasibility on a moderate grid, POCS
    polish on a denser constraint set (so nothing slips between samples),
    then an independent dense validation. Returns None when infeasible.
    """
    if degree < 0 or corridor.xb <= corridor.xa:
        return None
    c0 = _weighted_init(corridor, degree)

    A, lo, hi = _constraint_set(corridor, degree)
    coef, viol = _project_feasible(A, lo, hi, c0)
    if not np.isfinite(viol) or viol > 1e5:
        return None

    A_d, lo_d, hi_d = _constraint_set(corridor, degree,
                                      n_int=DENSE_GRID // 3, n_esc=200)
    coef, dviol = _project_feasible(A_d, lo_d, hi_d, coef)

    dv = _dense_violation(corridor, coef)
    if dv > CORRIDOR_EPS:
        return None
    power_z = cheb.cheb2poly(coef)
    zpoly = np.polynomial.Polynomial(power_z)
    affine = np.polynomial.Polynomial(
        [
            -(corridor.xa + corridor.xb) / (corridor.xb - corridor.xa),
            2.0 / (corridor.xb - corridor.xa),
        ]
    )
    poly = zpoly(affine)
    if not np.all(np.isfinite(poly.coef)):
        return None
    return PathFit(
        corridor=corridor,
        degree=degree,
        coef_cheb=coef,
        poly=poly,
        dense_max_violation=dv,
    )


def min_degree(corridor: Corridor, hi: int = INITIAL_FIT_DEGREE) -> PathFit | None:
    """Lowest VERIFIED feasible degree inside the unchanged corridor.

    fit_degree is a numerical oracle and its success can be non-monotone
    in degree, so: verify hi; binary-search an approximate bound; then
    deterministically sweep downward from the best known degree until
    the first failure, returning the lowest verified feasible fit.
    """
    top = fit_degree(corridor, hi)
    if top is None:
        return None
    lo, hi_, best = 0, hi, top
    while lo < hi_:
        mid = (lo + hi_) // 2
        trial = fit_degree(corridor, mid)
        if trial is None:
            lo = mid + 1
        else:
            hi_, best = mid, trial
    d = best.degree
    while d > 0:
        lower = fit_degree(corridor, d - 1)
        if lower is None:
            break
        best, d = lower, d - 1
    assert best.degree == d
    return best
