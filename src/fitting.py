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
vertical band and never re-enter. The escape direction per side is chosen deterministically from endpoint/glyph
geometry before fitting; polynomial optimization may not change that topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as cheb

from src.topology import (
    Corridor,
    NORMALIZED_SIZE,
    TAU,
    ESC_OFFSETS,
    ESCAPE_RATE,
    ESC_SLOPE_MIN,
    CORRIDOR_EPS,
)

# Canonical fitting budget. Cormorant (calligraphic, taller/thinner vertical
# strokes) needs higher-degree polynomials than the previous sans-serif face
# for some capitals; the minimum feasible degree is re-measured per corridor and
# is a measurement, not a quality gate, so a larger cap only lets harder glyphs
# fit without changing any validated output for letters that already fit low.
# B's heaviest corridor requires ~degree 120, so the cap is set safely above it.
INITIAL_FIT_DEGREE = 140
FIT_GRID = 128            # constraint samples across the whole window
DENSE_GRID = 900          # validation samples (denser than fitting)
# Stage-1 feasibility probe uses the canonical FIT_GRID resolution so a
# probe-feasible degree is genuinely feasible (fit_degree is then attempted
# only for the true minimal feasible degree). A subset that is infeasible
# still proves the full set infeasible, so no feasible corridor is skipped.
POCS_SWEEPS = 240
FEAS_TOL = 1e-6
FAMILY_HALF_WIDTH_FLOOR = 0.005   # ~2.5 raster steps at normalized scale
# Strict coordinate-equivalent of the old 1e-6 y-length degeneracy
# threshold (0..100 world): two anchors closer than this are the same
# polynomial geometrically.
FAMILY_MIN_SPAN = 1e-6 * NORMALIZED_SIZE / 100.0
HORNER_MIN_DEGREE = 10             # degree >= this uses Horner serialization
CERT_TOL = 2.0 * (1.0 / 512)   # certificate violation tolerance (~2 raster steps)
USE_LP = True   # unit tests may disable for speed (pure POCS)

# Stage-1 skip rule (sound): the probe solves a strict SUBSET of the
# full constraint rows, so probe-infeasible => truly infeasible; only
# then is the degree/orientation skipped without a false negative.
STAGE1_SKIP_VIOL = 0.005
ORIENTATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))  # compatibility/reference


@dataclass
class PathFit:
    """A polynomial fitted to one corridor (topology fixed)."""
    corridor: Corridor
    degree: int
    coef_cheb: np.ndarray            # Chebyshev coefficients in z
    poly: np.polynomial.Polynomial   # ordinary powers of x
    dense_max_violation: float
    orientation: tuple               # (sigma_left, sigma_right), each +-1


def preferred_tail_orientation(corridor: Corridor) -> tuple[int, int]:
    """Choose tail directions from endpoint/glyph geometry before fitting.

    Issue #4: a component-level preference (set on ``corridor`` during
    Phase 1 for disconnected glyph components) is applied FIRST and
    overrides the per-endpoint rule - the fitter must never flip it merely
    because another orientation is easier to fit.

    Otherwise each endpoint escapes toward the nearer vertical exterior
    boundary. Distances are measured from the endpoint corridor interval
    itself, not from a fitted value. Exact ties deterministically prefer
    upward escape.
    """
    pref = getattr(corridor, "preferred_orientation", None)
    if pref is not None:
        return tuple(int(s) for s in pref)
    orientation = []
    for i in (0, -1):
        down_distance = float(corridor.lower[i] - corridor.ylo)
        up_distance = float(corridor.yhi - corridor.upper[i])
        orientation.append(-1 if down_distance < up_distance else 1)
    return tuple(orientation)


def _zmap(x, xa: float, xb: float):
    return (2.0 * np.asarray(x, dtype=float) - xa - xb) / (xb - xa)


# Numerical Chebyshev basis domain. Corridor.xa/xb remain the semantic
# local stroke interval (issue #28); the basis alone spans every x used
# by interior, value-tail, and slope-tail constraints. This keeps all
# constrained evaluations inside [-1, 1], avoiding exponential T_n(z)
# growth outside that interval and the resulting HiGHS conditioning
# failures. An affine basis change does not change the degree-d
# polynomial function space or any geometric acceptance criterion.
_SLOPE_RAMP_FACTOR = 1.02


def chebyshev_domain(corridor: Corridor) -> tuple[float, float]:
    pad = ESC_OFFSETS[-1] * _SLOPE_RAMP_FACTOR
    return (float(corridor.xs[0] - pad),
            float(corridor.xs[-1] + pad))


def _corridor_zmap(x, corridor: Corridor):
    xa, xb = chebyshev_domain(corridor)
    return _zmap(x, xa, xb)


def _basis_affine(corridor: Corridor):
    xa, xb = chebyshev_domain(corridor)
    return np.polynomial.Polynomial(
        [-(xa + xb) / (xb - xa), 2.0 / (xb - xa)])


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
    offs = np.linspace(ESC_OFFSETS[0] * 0.5, ESC_OFFSETS[-1] * _SLOPE_RAMP_FACTOR, n_pts)
    xs_d = x_end + sgn * offs
    z = _corridor_zmap(xs_d, corridor)
    basis_xa, basis_xb = chebyshev_domain(corridor)
    dzdx = 2.0 / (basis_xb - basis_xa)
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






def _constraint_set(corridor: Corridor, degree: int,
                    sig_l: int, sig_r: int,
                    n_int: int = FIT_GRID, n_esc: int = 40,
                    slope_rows: bool = True):
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

    blocks = [(cheb.chebvander(_corridor_zmap(xs_int, corridor),
                               degree), lo_i, hi_i)]
    if n_esc > 0:
        for sigma, side in ((sig_l, "L"), (sig_r, "R")):
            xs_e, lo_e, hi_e = _side_rows(corridor, sigma, side, n_esc)
            blocks.append((
                cheb.chebvander(_corridor_zmap(xs_e, corridor),
                                degree), lo_e, hi_e))
            if slope_rows:
                blocks.append(
                    _side_slope_rows(corridor, degree, sigma, side))

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
        return cheb.chebval(_corridor_zmap(xq, corridor), coef)

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
    z = _corridor_zmap(xs_p, corridor)
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
    affine = _basis_affine(corridor)
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
    """Lowest VERIFIED feasible degree for the required tail geometry.

    Tail orientation is fixed from the route endpoint corridors before any
    polynomial optimization. Degrees are then probed exhaustively from 0
    upward (fit_degree is a numerical oracle whose success can be
    non-monotone in degree). A cheap stage-1 LP pre-filter skips only clearly
    infeasible degrees; marginal cases run the full verified pipeline. If the
    geometrically required orientation cannot be fitted within ``hi``, fail
    rather than silently flipping topology.
    """
    ori = preferred_tail_orientation(corridor)
    for d in range(0, hi + 1):
        c0 = _weighted_init(corridor, d)
        A, lo, hi_b = _constraint_set(corridor, d, *ori, slope_rows=False)
        _, viol = _project_feasible(A, lo, hi_b, c0)
        if not np.isfinite(viol) or viol > STAGE1_SKIP_VIOL:
            continue   # sound: subset infeasible => full infeasible
        fit = fit_degree(corridor, d, ori[0], ori[1])
        if fit is None:
            continue
        if tail_reentry_violation_cheb(fit.coef_cheb, corridor, ori) != 0.0:
            continue   # ramps satisfied but not permanently outward
        return fit
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


def _real_roots_z(coefs):
    """Real roots in z-space with exact-degree semantics.

    Trailing coefficients are stripped ONLY when exactly zero - a
    1e-16 leading coefficient still dominates at infinity and must be
    honoured. Coefficients are then rescaled by their max magnitude
    purely for np.roots conditioning; this changes no zero of the
    polynomial.
    """
    c = np.asarray(coefs, dtype=float)
    nz = np.nonzero(c != 0.0)[0]
    if len(nz) == 0:
        return None            # identically zero
    c = c[: nz[-1] + 1]        # true degree: exact zeros only
    if len(c) < 2:
        return np.array([])    # nonzero constant: no roots
    c = c / float(np.max(np.abs(c)))
    r = np.roots(c[::-1])
    return np.sort(r[np.abs(r.imag) < 1e-7].real)


def tail_reentry_violation_cheb(coef_cheb, corridor, orientation,
                                esc_offset: float = None):
    """Canonical V3 tail proof on Chebyshev coefficients in z.

    Same three conditions as `tail_reentry_violation`, but every
    evaluation happens through `chebval` on the normalized z map and
    root finding happens on power coefficients in z (converted via
    cheb2poly), never on expanded raw powers of x. This is
    scale-equivariant under x -> x/100 because the z map is canonical.
    """
    viol = 0.0
    if esc_offset is None:
        esc_offset = ESC_OFFSETS[-1]
    cc = np.asarray(coef_cheb, dtype=float)
    xa, xb = chebyshev_domain(corridor)

    def peval(xq):
        return cheb.chebval(_zmap(xq, xa, xb), cc)

    # derivative roots in z (exact-degree preserved)
    droots_all = _real_roots_z(cheb.cheb2poly(cheb.chebder(cc)))
    # asymptotic direction from power form in z
    pcoef_pow = cheb.cheb2poly(cc)

    sig_l, sig_r = int(orientation[0]), int(orientation[1])
    for sigma, side in ((sig_l, "L"), (sig_r, "R")):
        sgn = -1.0 if side == "L" else 1.0
        x_c = float(corridor.xs[0] if side == "L" else corridor.xs[-1])
        x_c += sgn * esc_offset
        edge = corridor.yhi if sigma == 1 else corridor.ylo
        right = side == "R"
        z_c = float(_zmap(x_c, xa, xb))

        p_c = float(peval(x_c))
        if sigma * (p_c - edge) <= 0:
            viol = max(viol, 3.0)
            continue

        def outward(zs):
            zs = np.atleast_1d(np.asarray(zs, dtype=float))
            return zs[zs > z_c] if right else zs[zs < z_c]

        if droots_all is not None:
            for r in outward(droots_all):
                if sigma * (float(cheb.chebval(float(r), cc)) - edge) <= 0:
                    viol = max(
                        viol, 2.0 + min(abs(float(r) - z_c), 1.0))
                    break

        c = pcoef_pow
        nz = np.nonzero(c != 0.0)[0]
        if len(nz) > 0:
            c = c[: nz[-1] + 1]
        if len(c) >= 2:
            lead = c[-1] > 0
            even = (len(c) - 1) % 2 == 0
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
    A_t = cheb.chebvander(_corridor_zmap(xs_t, corridor), degree)
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
    affine = _basis_affine(corridor)
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
    A_f = cheb.chebvander(_corridor_zmap(samp_x, corridor), degree)
    w_scaled = weights / half
    cost = A_f.T @ ((-w_scaled) if maximize else w_scaled)

    # Solve directly on the DENSE constraint grid: the coarse fitting
    # grid rings between samples on wide corridors, and re-projecting
    # afterwards collapses distinct objective optima onto the same
    # point (killing family span). The dense-LP optimum is distinct
    # per guide direction AND dense-feasible by construction.
    A_base, lo_base, hi_base = _constraint_set(
        corridor, degree, sig_l, sig_r,
        n_int=DENSE_GRID, n_esc=600)

    fin_hi = np.isfinite(hi_base)
    fin_lo = np.isfinite(lo_base)
    A_ub = np.vstack([A_base[fin_hi], -A_base[fin_lo]])
    b_ub = np.concatenate([hi_base[fin_hi], -lo_base[fin_lo]])

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None,None)]*(degree+1), method="highs")
    if not res.success or res.x is None:
        return None
    coef = np.asarray(res.x)
    dv = _dense_violation(corridor, coef, sig_l, sig_r)
    if not np.isfinite(dv) or dv > CORRIDOR_EPS:
        return None
    return coef




def certify_anchor(corridor: Corridor, coef_cheb: np.ndarray,
                   sig_l: int, sig_r: int) -> float:
    """Dense corridor/ramp certification in CANONICAL coordinates.

    Evaluates purely through chebval on the corridor z map - no raw-x
    expansion is constructed for any degree. Acceptance contract:
    returned violation must be <= CORRIDOR_EPS (the allowance lives in
    the caller's comparison only, never twice).
    """
    return _dense_violation(corridor, coef_cheb, sig_l, sig_r,
                            allow=0.0)
