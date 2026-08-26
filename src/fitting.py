"""Phase 2/3: corridor-constrained polynomial fitting with mandatory
tail escape.

For a fixed corridor the fitting problem is linear in the polynomial
coefficients: at sampled positions x_i the corridor demands
lower(x_i) <= P(x_i) <= upper(x_i), and the tail ramps demand one-sided
rows beyond both route endpoints. Every constraint is therefore an
ordinary (possibly one-sided) linear row.

Coefficients are solved in the Chebyshev basis on z in [-1,1]
(numerically stable) via scipy's HiGHS LP (minimizing worst row
violation). No stochastic search.

Tail escape is MANDATORY on both sides for every emitted polynomial:
after traversing its route the polynomial must leave the glyph's
vertical band and never re-enter. The escape direction per side is a
deterministic finite choice (up/down, four combinations) solved during
fitting; among feasible orientations at a degree the simplest fit wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as cheb

from src.topology import (
    Corridor,
    TAU,
    ESC_OFFSETS,
    ESCAPE_RATE,
    ESC_SLOPE_MIN,
    CORRIDOR_EPS,
)

INITIAL_FIT_DEGREE = 24   # measured: deg-20 rings ~0.34 inside tight
                          # tubes on wide windows; 24 halves it
FIT_GRID = 128            # constraint samples across the whole window
DENSE_GRID = 900          # validation samples (denser than fitting)
POCS_SWEEPS = 240
FEAS_TOL = 1e-6
FAMILY_HALF_WIDTH_FLOOR = 0.005   # ~2.5 raster steps at normalized scale
HORNER_MIN_DEGREE = 10             # degree >= this uses Horner serialization
CERT_TOL = 2.0 * (1.0 / 512)   # certificate violation tolerance (~2 raster steps)
USE_LP = True   # unit tests may disable for speed (pure POCS)

ORIENTATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))  # stable order
# Stage-1 skip rule (sound): the probe solves a strict SUBSET of the
# full constraint rows, so probe-infeasible => truly infeasible; only
# then is the degree/orientation skipped without a false negative.
STAGE1_SKIP_VIOL = 0.005


@dataclass
class PathFit:
    """A polynomial fitted to one corridor (topology fixed)."""
    corridor: Corridor
    degree: int
    coef_cheb: np.ndarray            # Chebyshev coefficients in z
    poly: np.polynomial.Polynomial   # ordinary powers of x
    dense_max_violation: float
    orientation: tuple               # (sigma_left, sigma_right), each +-1


def _zmap(x, xa: float, xb: float):
    return (2.0 * np.asarray(x, dtype=float) - xa - xb) / (xb - xa)


def _escape_bound(sigma: int, offset: float, corridor: Corridor):
    """Outward ramp bound at `offset` units beyond the route endpoint.

    At least TAU + ESCAPE_RATE*offset beyond the chosen band edge -
    'at least this far outside by here', never an exact target.
    """
    edge = corridor.yhi if sigma == 1 else corridor.ylo
    return edge + sigma * (TAU + ESCAPE_RATE * offset)


CORRIDOR_PAD = 0.30   # corridor window padding: ramp rows reach out here


def _side_rows(corridor: Corridor, sigma: int, side: str,
               n_esc: int, max_off: float | None = None):
    """Value rows (one-sided) along the continuous escape ramp."""
    off_max = max_off or ESC_OFFSETS[-1]
    offs = np.linspace(ESC_OFFSETS[0], off_max, n_esc)
    sgn = -1.0 if side == "L" else 1.0
    x_end = corridor.xs[0] if side == "L" else corridor.xs[-1]
    xs_e = x_end + sgn * offs
    bnd = _escape_bound(sigma, offs, corridor)
    lo = bnd if sigma == 1 else np.full(len(xs_e), -np.inf)
    hi = np.full(len(xs_e), np.inf) if sigma == 1 else bnd
    return xs_e, lo, hi


def _side_slope_rows(corridor: Corridor, degree: int,
                     sigma: int, side: str, n_pts: int = 24):
    """Linear rows enforcing outward motion along the whole escape ramp:
    sign(sigma * sgn_x * dP/dx) >= ESC_SLOPE_MIN at sampled offsets.
    Returns (A_rows, lo, hi) in the SAME Chebyshev-z variable order."""
    x_end = corridor.xs[0] if side == "L" else corridor.xs[-1]
    sgn = -1.0 if side == "L" else 1.0
    offs = np.linspace(ESC_OFFSETS[0] * 0.5, ESC_OFFSETS[-1] * 1.02, n_pts)
    xs_d = x_end + sgn * offs
    z = _zmap(xs_d, corridor.xa, corridor.xb)
    dzdx = 2.0 / (corridor.xb - corridor.xa)
    A = np.zeros((len(z), degree + 1))
    for k in range(1, degree + 1):
        dcoef = cheb.chebder(np.eye(degree + 1)[k])
        A[:, k] = dzdx * cheb.chebval(z, dcoef)
    req = float(sigma * sgn) * ESC_SLOPE_MIN   # required dP/dx value
    if req >= 0:
        lo, hi = np.full(len(z), req), np.full(len(z), np.inf)
    else:
        lo, hi = np.full(len(z), -np.inf), np.full(len(z), req)
    return A, lo, hi






def certify_halfline_min(coef_cheb: np.ndarray, corridor: Corridor, *,
                         side: str, sigma: int):
    """Exact minimum of

        Q(t) = sigma*sgn_x*P'(x_checkpoint + sgn_x*t) - ESC_SLOPE_MIN

    over t >= 0 via roots of Q'(t). Returns (min_value, argmin_t).
    """
    from numpy.polynomial import Polynomial as Poly

    xa, xb = corridor.xa, corridor.xb
    scale = (xb - xa) / 2.0
    mid = (xa + xb) / 2.0
    x_end = float(corridor.xs[0] if side == "L" else corridor.xs[-1])
    sgn_x = -1.0 if side == "L" else 1.0
    z_of_t = Poly([(x_end - mid) / scale, sgn_x / scale])

    px = Poly(np.asarray(cheb.cheb2poly(np.asarray(
        coef_cheb, dtype=float))))
    px_t = px(z_of_t)
    q = (sigma * sgn_x) * px_t.deriv() - Poly([ESC_SLOPE_MIN])

    # find minimum on t >= 0: t=0 + real nonneg roots of Q'(t)
    cand = [0.0]
    for r in q.deriv().roots():
        if abs(r.imag) < 1e-9 and r.real >= -1e-12:
            cand.append(float(r.real))
    vals = [float(q(max(t, 0.0))) for t in cand]

    # asymptotic check: leading coefficient must be positive or zero
    ctr = np.asarray(q.coef)
    nz = np.nonzero(ctr)[0]
    if len(nz) >= 1 and nz[-1] >= 1 and ctr[nz[-1]] < 0:
        return -np.inf, np.inf   # Q -> -inf as t -> inf
    k = int(np.argmin(vals))
    return vals[k], cand[k]




def _constraint_set(corridor: Corridor, degree: int,
                    sig_l: int, sig_r: int,
                    n_int: int = FIT_GRID, n_esc: int = 40,
                    slope_rows: bool = True,
                    tail_cert: bool = False,
                    cert_u_points=None):
    """Build the deterministic constraint system (A, lo, hi).

    Interior rows are two-sided interval constraints across the path
    domain; both tails contribute one-sided ramp rows sampled densely
    along their continuous bounds so nothing swings between checkpoints,
    plus outward-slope rows on P'. With slope_rows=False the system is
    a strict subset (used by the sound stage-1 feasibility probe).
    """
    xs_int = np.linspace(corridor.xs[0], corridor.xs[-1], n_int)
    lo_i = corridor.lower_at(xs_int)
    hi_i = corridor.upper_at(xs_int)

    blocks = [(cheb.chebvander(_zmap(xs_int, corridor.xa, corridor.xb),
                               degree), lo_i, hi_i)]
    if n_esc > 0:
        for sigma, side in ((sig_l, "L"), (sig_r, "R")):
            xs_e, lo_e, hi_e = _side_rows(corridor, sigma, side, n_esc)
            blocks.append((
                cheb.chebvander(_zmap(xs_e, corridor.xa, corridor.xb),
                                degree), lo_e, hi_e))
            if slope_rows:
                blocks.append(
                    _side_slope_rows(corridor, degree, sigma, side))
            if tail_cert:
                A_c, lo_c = _tail_certificate_rows(
                    corridor, degree, sigma, side)
                hi_c = np.full(A_c.shape[0], np.inf)
                blocks.append((A_c, lo_c, hi_c))

    A = np.vstack([b[0] for b in blocks])
    lo = np.concatenate([b[1] for b in blocks])
    hi = np.concatenate([b[2] for b in blocks])
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

    from scipy.optimize import linprog  # lazy: keeps import cost out of
                                       # modules that never solve

    m, n = A.shape
    # variables: [c (n), s (1)]
    # A c - s <= hi ; -A c - s <= -lo ; s >= 0
    A_ub = np.zeros((2 * m, n + 1))
    A_ub[:m, :n] = A
    A_ub[:m, n] = -1.0
    A_ub[m:, :n] = -A
    A_ub[m:, n] = -1.0
    b_ub = np.concatenate([hi, -lo]).astype(float)
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
                     sig_l: int, sig_r: int,
                     grid: int = DENSE_GRID,
                     power_coef: np.ndarray = None,
                     allow: float = 0.0) -> float:
    """Dense validation against the corridor and both tail ramps.

    When power_coef is given, the EMITTED power-basis polynomial is
    validated instead of the Chebyshev solution: cheb->power conversion
    can lose fractions of a unit on steep corridors, and only the
    emitted polynomial is what users paste.
    """
    viol = 0.0
    # include the exact constraint NODES: steep unfolded sections can
    # spike between uniform samples
    xs = np.union1d(np.linspace(corridor.xs[0], corridor.xs[-1], grid),
                    corridor.xs)

    def _eval(xq):
        if power_coef is not None:
            return np.polynomial.polynomial.polyval(xq, power_coef)
        return cheb.chebval(_zmap(xq, corridor.xa, corridor.xb), coef)

    vals = _eval(xs)
    lo = corridor.lower_at(xs) - allow
    hi = corridor.upper_at(xs) + allow
    viol = max(viol, float(np.maximum(lo - vals, vals - hi).max()))
    for sigma, side in ((sig_l, "L"), (sig_r, "R")):
        xs_e, lo_e, hi_e = _side_rows(corridor, sigma, side,
                                      max(60, grid // 4))
        v_e = _eval(xs_e)
        viol = max(
            viol,
            float(max(np.max(lo_e - v_e), np.max(v_e - hi_e), 0.0)),
        )
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


def fit_degree(corridor: Corridor, degree: int,
               sig_l: int = 1, sig_r: int = 1) -> PathFit | None:
    """Fit one polynomial of exactly this degree inside the corridor
    with the given tail orientations.

    Deterministic pipeline: LP feasibility on a moderate grid, LP
    polish on a denser constraint set (so nothing slips between
    samples), then independent dense validation of interior AND ramp
    rows. Returns None when infeasible.
    """
    if degree < 0 or corridor.xb <= corridor.xa:
        return None
    c0 = _weighted_init(corridor, degree)

    A, lo, hi = _constraint_set(corridor, degree, sig_l, sig_r)
    coef, viol = _project_feasible(A, lo, hi, c0)
    if not np.isfinite(viol) or viol > 1e5:
        return None

    # Polish + verify on progressively denser ramp sampling until the
    # independent dense check passes (steep escape cliffs ring between
    # sparse rows).
    dv = None
    for n_esc_d, n_int_d in ((200, DENSE_GRID), (600, 2 * DENSE_GRID),
                              (900, 3 * DENSE_GRID)):
        A_d, lo_d, hi_d = _constraint_set(corridor, degree, sig_l, sig_r,
                                          n_int=n_int_d, n_esc=n_esc_d)
        coef, dviol = _project_feasible(A_d, lo_d, hi_d, coef)
        if not np.isfinite(dviol) or dviol > 1e5:
            return None
        dv = _dense_violation(corridor, coef, sig_l, sig_r)
        if dv <= CORRIDOR_EPS:
            break
    if dv is None or dv > CORRIDOR_EPS:
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
    # verify the EMITTED polynomial representation:
    # low degree -> raw x-power output; validate power-basis conversion
    # high degree -> Horner/normalized-z output; Chebyshev validation
    #   (step 1 above) is authoritative because raw-x expansion is
    #   numerically unstable and NOT what users receive
    if degree < HORNER_MIN_DEGREE:
        dv_p = _dense_violation(corridor, coef, sig_l, sig_r,
                                power_coef=np.asarray(poly.coef))
        if dv_p > CORRIDOR_EPS:
            return None
        dv = max(dv, dv_p)
    return PathFit(
        corridor=corridor,
        degree=degree,
        coef_cheb=coef,
        poly=poly,
        dense_max_violation=dv,
        orientation=(int(sig_l), int(sig_r)),
    )


def fit_route(corridor: Corridor, hi: int = INITIAL_FIT_DEGREE) -> PathFit | None:
    """Lowest VERIFIED feasible degree with valid escaping tails.

    Degrees are probed exhaustively from 0 upward (fit_degree is a
    numerical oracle whose success can be non-monotone in degree). A
    cheap stage-1 LP pre-filter skips only clearly infeasible
    (degree, orientation) pairs; anything marginal runs the FULL
    verified pipeline. All four tail orientations are deterministic
    finite choices; among feasible fits at a degree the simplest
    (smallest max |Chebyshev coefficient|) wins, ties broken by stable
    ORIENTATIONS order. The first degree with any feasible fit returns
    immediately.
    """
    for d in range(0, hi + 1):
        best = None
        best_key = None
        for ori in ORIENTATIONS:
            c0 = _weighted_init(corridor, d)
            A, lo, hi_b = _constraint_set(corridor, d, *ori,
                                          slope_rows=False)
            _, viol = _project_feasible(A, lo, hi_b, c0)
            if not np.isfinite(viol) or viol > STAGE1_SKIP_VIOL:
                continue   # sound: subset infeasible => full infeasible
            fit = fit_degree(corridor, d, ori[0], ori[1])
            if fit is None:
                continue
            if tail_reentry_violation(
                    np.asarray(fit.poly.coef, dtype=float),
                    corridor, ori) != 0.0:
                continue   # ramps satisfied but not permanently outward
            key = (float(np.max(np.abs(fit.coef_cheb))),
                   ORIENTATIONS.index(ori))
            if best_key is None or key < best_key:
                best_key, best = key, fit
        if best is not None:
            return best
    return None


def _real_roots(coefs, tol=1e-9):
    """Real roots of a polynomial given ascending power coefficients."""
    c = np.asarray(coefs, dtype=float)
    nz = np.nonzero(np.abs(c) > tol * max(1.0, np.max(np.abs(c))))[0]
    if len(nz) == 0:
        return None            # identically zero
    c = c[: nz[-1] + 1]        # trim trailing zeros (true degree)
    if len(c) < 2:
        return np.array([])    # nonzero constant: no roots
    r = np.roots(c[::-1])
    return np.sort(r[np.abs(r.imag) < 1e-7].real)


def tail_reentry_violation(poly_coef, corridor, orientation):
    """V3 analytic permanent-tail check for one emitted polynomial.

    For each side with chosen orientation sigma (+1 up / -1 down):
    beyond the final ramp checkpoint the polynomial must

      1. already be strictly outside the glyph vertical band on the
         outward side;
      2. have no critical point on the outward half-line whose value
         dips back to or inside the band edge;
      3. diverge permanently outward (correct leading behaviour).

    The minimum of P on the half-line occurs at the checkpoint, at a
    root of P', or in the asymptotic limit, so these three conditions
    are exactly permanent escape - verified by root analysis, not
    sampling.
    """
    viol = 0.0
    poly = np.polynomial.Polynomial(np.asarray(poly_coef, dtype=float))
    # exact emitted degree: strip ONLY exact zeros - a 1e-16 leading
    # coefficient still dominates at infinity
    c_arr = np.asarray(poly_coef, dtype=float)
    nz = np.nonzero(c_arr != 0.0)[0]
    ptrim = (poly if len(nz) == 0
             else np.polynomial.Polynomial(c_arr[:nz[-1] + 1]))
    droots_all = _real_roots(ptrim.deriv().coef)

    sig_l, sig_r = int(orientation[0]), int(orientation[1])
    for sigma, side in ((sig_l, "L"), (sig_r, "R")):
        sgn = -1.0 if side == "L" else 1.0
        # final ramp checkpoint: escape rows guarantee outwardness here
        x_c = float(corridor.xs[0] if side == "L" else corridor.xs[-1])
        x_c += sgn * ESC_OFFSETS[-1]
        edge = corridor.yhi if sigma == 1 else corridor.ylo
        right = side == "R"

        # 1) outside at the final checkpoint
        p_c = float(poly(x_c))
        if sigma * (p_c - edge) <= 0:
            viol = max(viol, 3.0)
            continue

        def outward(xs):
            xs = np.atleast_1d(np.asarray(xs, dtype=float))
            return xs[xs > x_c] if right else xs[xs < x_c]

        # 2) no critical point dips back into the band
        if droots_all is not None:
            crit = outward(droots_all)
            for r in crit:
                if sigma * (float(poly(r)) - edge) <= 0:
                    viol = max(viol, 2.0 + min(abs(float(r) - x_c), 1.0))
                    break

        # 3) asymptotic direction must be outward. ptrim already
        # stripped EXACT zeros only: a 1e-16 leading coefficient still
        # dominates at infinity and must be honoured.
        c = ptrim.coef
        if len(c) >= 2:
            lead = c[-1] > 0
            even = (len(c) - 1) % 2 == 0
            # P -> +infinity on the right iff lead > 0; on the left iff
            # lead > 0 for even degree, lead < 0 for odd degree.
            up_right = lead
            up_left = lead if even else not lead
            limit_outward = (
                (up_right if right else up_left) == (sigma == 1)
            )
            if not limit_outward:
                viol = max(viol, 4.0)
    return viol



def fit_variant(corridor: Corridor, degree: int, target: np.ndarray,
                sig_l: int, sig_r: int):
    """Seeded realization of a corridor at a FIXED degree.

    All corridor/tail/slope constraints stay HARD; the objective
    minimizes total absolute deviation from the seeded guide trajectory
    sampled at the corridor nodes. Returns None when no polynomial of
    this degree satisfies the hard constraints.
    """
    A, lo, hi = _constraint_set(corridor, degree, sig_l, sig_r)
    # match V2 semantics: the solver-level allowance equals the
    # validation EPS, so anything the baseline fit accepts stays
    # feasible here
    fin = np.isfinite(lo)
    lo = np.where(fin, lo - CORRIDOR_EPS, lo)
    fin = np.isfinite(hi)
    hi = np.where(fin, hi + CORRIDOR_EPS, hi)

    xs_t = np.asarray(corridor.xs, dtype=float)
    A_t = cheb.chebvander(_zmap(xs_t, corridor.xa, corridor.xb), degree)
    t = np.asarray(target, dtype=float)

    n = A.shape[1]
    m = len(xs_t)
    nv = n + m

    cost = np.zeros(nv)
    cost[n:] = 1.0

    rows = []
    rhs = []
    fin_hi = np.isfinite(hi)
    if fin_hi.any():
        rows.append(np.hstack([A[fin_hi], np.zeros((int(fin_hi.sum()), m))]))
        rhs.append(hi[fin_hi])
    fin_lo = np.isfinite(lo)
    if fin_lo.any():
        rows.append(np.hstack([-A[fin_lo], np.zeros((int(fin_lo.sum()), m))]))
        rhs.append(-lo[fin_lo])
    # e >= |A_t c - t|  <=>  A_t c - e <= t ; -A_t c - e <= -t
    rows.append(np.hstack([A_t, -np.eye(m)]))
    rhs.append(t)
    rows.append(np.hstack([-A_t, -np.eye(m)]))
    rhs.append(-t)

    from scipy.optimize import linprog  # lazy

    A_ub = np.vstack(rows)
    b_ub = np.concatenate(rhs)
    bounds = [(None, None)] * n + [(0.0, None)] * m

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method="highs")
    if not res.success or res.x is None:
        return None
    coef = np.asarray(res.x[:n])

    dv_p = _dense_violation(corridor, coef, sig_l, sig_r,
                            grid=DENSE_GRID, allow=CORRIDOR_EPS)
    if dv_p > CORRIDOR_EPS:
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
        dense_max_violation=dv_p,
        orientation=(int(sig_l), int(sig_r)),
    )


def solve_anchor(corridor: Corridor, degree: int, sig_l: int, sig_r: int,
                 weights: np.ndarray, maximize: bool):
    """Solve min/max of the normalized guide functional subject to hard
    corridor/ramp constraints. Returns Chebyshev coefficients or None.
    """
    from scipy.optimize import linprog
    from src.fitting import _constraint_set

    samp_x = corridor.xs
    half = np.maximum((corridor.upper - corridor.lower) / 2.0, FAMILY_HALF_WIDTH_FLOOR)
    A_f = cheb.chebvander(_zmap(samp_x, corridor.xa, corridor.xb), degree)
    w_scaled = weights / half
    cost = A_f.T @ ((-w_scaled) if maximize else w_scaled)

    A_base, lo_base, hi_base = _constraint_set(
        corridor, degree, sig_l, sig_r)

    fin_hi = np.isfinite(hi_base)
    fin_lo = np.isfinite(lo_base)
    A_ub = np.vstack([A_base[fin_hi], -A_base[fin_lo]])
    b_ub = np.concatenate([hi_base[fin_hi], -lo_base[fin_lo]])

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None,None)]*(degree+1), method="highs")
    if not res.success or res.x is None:
        return None
    return np.asarray(res.x)




def certify_anchor(corridor: Corridor, coef_cheb: np.ndarray,
                   sig_l: int, sig_r: int) -> float:
    """Dense POWER-domain worst violation (validation-domain check)."""
    power_z = cheb.cheb2poly(coef_cheb)
    affine = np.polynomial.Polynomial(
        [-(corridor.xa + corridor.xb) / (corridor.xb - corridor.xa),
         2.0 / (corridor.xb - corridor.xa)])
    poly = np.polynomial.Polynomial(power_z)(affine)
    return _dense_violation(corridor, coef_cheb, sig_l, sig_r,
                            power_coef=np.asarray(poly.coef),
                            allow=CORRIDOR_EPS)
