from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

MAX_DEGREE = 5
TAU = 2.0
RESTARTS_PER_CURVE = 8
REFINE_STEPS = 120
RESCUE_RESTARTS = 16
REDUCE_STEPS = 60

GRID = 512
SIZE = 100.0
MIN_COVERAGE = 0.95
DEFAULT_SEED = 0
DEFAULT_MAX_CURVES = 12
PRECISION = 12

SEARCH_STEP = 1.0
VALIDATE_STEP = 0.1
SEARCH_GRAPH_MAX = 96
SEARCH_BOUNDARY_MAX = 256
VALIDATE_SAMPLE_CAP = 40000

MIN_NEW_POINTS = 8
MIN_DU = 0.01
SEED_MIN_DIST = 3.0
SEED_MAX_DIST = 15.0
SEED_EXPANDED_DIST = 25.0
SEED_P2_CHOICES = 8
SEED_SEGMENT_PTS = 33
MIN_TAIL_SLOPE = 8.0
COEF_MUTATION_PROB = 0.80
COEF_SIGMA_START, COEF_SIGMA_END = 10.0, 0.2

TRACE_EPS = 1e-9


@dataclass
class Candidate:
    degree: int
    coef: np.ndarray


@dataclass
class XCurve:
    poly: np.polynomial.Polynomial


@dataclass
class Glyph:
    points: np.ndarray
    search_points: np.ndarray
    search_idx: np.ndarray
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class Analysis:
    samples: np.ndarray
    sample_d: np.ndarray
    point_d: np.ndarray
    newly_covered: int
    coverage_fraction: float
    surface_fraction: float
    bad_surface_fraction: float
    mean_surface_excess: float
    surface_penalty: float
    mean_surface_distance: float
    trace_penalty: float
    tail_penalty: float
    bounds: tuple[float, float] | None
    left_slope: float
    right_slope: float
    deriv_outside: int
    feasible: bool
    merit: float


U_OF_X = np.polynomial.Polynomial([-1.0, 0.02])
X_OF_U = np.polynomial.Polynomial([50.0, 50.0])


def x_curve_of_candidate(cand: Candidate) -> XCurve:
    return XCurve(np.polynomial.Polynomial(cand.coef)(U_OF_X))


def _poly_u(coef: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.polyval(coef[::-1], (np.asarray(xs, dtype=float) - 50.0) / 50.0)


def _even_subset(points: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    if n <= cap:
        return points, np.arange(n)
    idx = np.unique(np.round(np.linspace(0, n - 1, cap)).astype(np.int64))
    return points[idx], idx


def make_glyph(points: np.ndarray) -> Glyph:
    search_points, search_idx = _even_subset(points, SEARCH_BOUNDARY_MAX)
    return Glyph(
        points,
        search_points,
        search_idx,
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


# ---------------------------------------------------------------------------
# Roots, trace components, sampling, distances
# ---------------------------------------------------------------------------


def _real_roots(poly: np.polynomial.Polynomial) -> np.ndarray:
    roots = poly.roots()
    real = roots[np.abs(roots.imag) < 1e-8].real
    return np.sort(real)


def _components_from_breaks(
    breaks: np.ndarray, poly: np.polynomial.Polynomial, ymin: float, ymax: float
) -> list[tuple[float, float]]:
    edges = np.concatenate([[-np.inf], breaks, [np.inf]])
    comps: list[tuple[float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isfinite(lo) and np.isfinite(hi):
            mid = 0.5 * (lo + hi)
        elif np.isfinite(hi):
            mid = hi - 1.0
        elif np.isfinite(lo):
            mid = lo + 1.0
        else:
            mid = 0.0
        y = float(poly(mid))
        if not (ymin <= y <= ymax):
            continue
        if comps and lo - comps[-1][1] <= TRACE_EPS:
            comps[-1] = (comps[-1][0], hi)
        else:
            comps.append((float(lo), float(hi)))
    return comps


def _adaptive_sample(eval_fn, a: float, b: float, max_step: float, cap: int):
    n0 = min(129, max(2, cap // 2))
    xs = np.linspace(a, b, n0)
    for _ in range(64):
        if xs.size >= cap:
            break
        ys = eval_fn(xs)
        gaps = np.hypot(np.diff(xs), np.diff(ys))
        bad = gaps > max_step
        if not bad.any():
            break
        mids = (xs[:-1][bad] + xs[1:][bad]) / 2.0
        xs = np.sort(np.concatenate([xs, mids]))
    if xs.size > cap:
        idx = np.round(np.linspace(0.0, xs.size - 1.0, cap)).astype(np.int64)
        idx[0] = 0
        idx[-1] = xs.size - 1
        idx = np.unique(idx)
        xs = xs[idx]
    return np.column_stack([xs, eval_fn(xs)])


def sample_curve(curve: XCurve, lo: float, hi: float, max_step: float, cap: int):
    return _adaptive_sample(curve.poly, lo, hi, max_step, cap)


def _min_dists(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    da = np.full(len(a), np.inf)
    db = np.full(len(b), np.inf)
    b2 = (b * b).sum(axis=1)
    block = max(1, int(2_000_000 / max(1, len(b))))
    for i in range(0, len(a), block):
        ai = a[i : i + block]
        d2 = (ai * ai).sum(axis=1)[:, None] - 2.0 * (ai @ b.T) + b2[None, :]
        np.maximum(d2, 0.0, out=d2)
        np.minimum(da[i : i + block], np.sqrt(d2.min(axis=1)), out=da[i : i + block])
        np.minimum(db, np.sqrt(d2.min(axis=0)), out=db)
    return da, db


# ---------------------------------------------------------------------------
# Single-pass candidate analysis
# ---------------------------------------------------------------------------


def _working_interval(
    comps: list[tuple[float, float]], glyph: Glyph
) -> tuple[float, float]:
    finite = [(a, b) for a, b in comps if np.isfinite(a) and np.isfinite(b)]
    if finite:
        return max(finite, key=lambda t: t[1] - t[0])
    return glyph.xmin - 5.0, glyph.xmax + 5.0


def _analyze(
    xc: XCurve,
    degree: int,
    glyph: Glyph,
    uncovered: np.ndarray,
    *,
    dense: bool,
) -> Analysis:
    """Single-pass measurement of an unbounded polynomial.

    Converts u -> x once, computes the roots of P-ymin, P-ymax and P'
    once, derives the trace components once, samples once, and derives
    every surface/coverage/structural metric from those cached values.
    `uncovered` must be aligned with the boundary actually used
    (search_points unless dense, else points).
    """
    poly = xc.poly
    ymin, ymax = glyph.ymin, glyph.ymax
    roots_ymin = _real_roots(poly - ymin)
    roots_ymax = _real_roots(poly - ymax)
    deriv = poly.deriv()
    droots = _real_roots(deriv)
    breaks = np.unique(np.concatenate([roots_ymin, roots_ymax]))
    comps = _components_from_breaks(breaks, poly, ymin, ymax)

    bounds: tuple[float, float] | None = None
    if len(comps) == 1:
        l, r = comps[0]
        if np.isfinite(l) and np.isfinite(r) and r > l:
            bounds = (l, r)

    lo, hi = bounds if bounds is not None else _working_interval(comps, glyph)
    if dense:
        raw = sample_curve(xc, lo, hi, VALIDATE_STEP, VALIDATE_SAMPLE_CAP)
    else:
        raw = sample_curve(xc, lo, hi, SEARCH_STEP, SEARCH_GRAPH_MAX)
    in_band = (raw[:, 1] >= ymin) & (raw[:, 1] <= ymax)
    samples = raw[in_band]

    boundary = glyph.points if dense else glyph.search_points
    if len(samples) == 0:
        sample_d = np.array([])
        point_d = np.full(len(boundary), np.inf)
    else:
        sample_d, point_d = _min_dists(samples, boundary)

    newly_covered = int(((point_d <= TAU) & uncovered).sum())
    n_uncovered = int(uncovered.sum())
    coverage_fraction = newly_covered / max(1, n_uncovered)

    if len(samples) == 0:
        surface_fraction = 0.0
        bad_surface_fraction = 0.0
        mean_surface_excess = 0.0
        surface_penalty = 0.0
        mean_surface_distance = 0.0
    else:
        surface_fraction = float((sample_d <= TAU).mean())
        bad_surface_fraction = float((sample_d > TAU).mean())
        excess = np.clip(sample_d - TAU, 0.0, None)
        mean_surface_excess = float(excess.mean()) / TAU
        surface_penalty = float((excess**2).sum())
        mean_surface_distance = float(sample_d.mean())

    if not comps:
        trace_penalty = 2.0
    else:
        unbounded = any(
            not (np.isfinite(a) and np.isfinite(b)) for a, b in comps
        )
        trace_penalty = (2.0 if unbounded else 0.0) + max(0, len(comps) - 1)

    if bounds is not None:
        l, r = bounds
        left_slope = float(deriv(l))
        right_slope = float(deriv(r))
        deriv_outside = int(((droots < l) | (droots > r)).sum())
        tail_penalty = (
            deriv_outside
            + max(0.0, MIN_TAIL_SLOPE - abs(left_slope)) / MIN_TAIL_SLOPE
            + max(0.0, MIN_TAIL_SLOPE - abs(right_slope)) / MIN_TAIL_SLOPE
        )
    else:
        left_slope = right_slope = 0.0
        deriv_outside = 0
        tail_penalty = 0.0

    feasible = (
        bounds is not None
        and surface_fraction >= MIN_COVERAGE
        and deriv_outside == 0
        and abs(left_slope) >= MIN_TAIL_SLOPE
        and abs(right_slope) >= MIN_TAIL_SLOPE
        and newly_covered >= MIN_NEW_POINTS
    )
    merit = (
        coverage_fraction
        - 4.0 * bad_surface_fraction
        - 0.5 * mean_surface_excess
        - 2.0 * trace_penalty
        - 1.0 * tail_penalty
        - 0.005 * degree
    )
    return Analysis(
        samples,
        sample_d,
        point_d,
        newly_covered,
        coverage_fraction,
        surface_fraction,
        bad_surface_fraction,
        mean_surface_excess,
        surface_penalty,
        mean_surface_distance,
        trace_penalty,
        tail_penalty,
        bounds,
        left_slope,
        right_slope,
        deriv_outside,
        feasible,
        merit,
    )


def analyze_candidate(
    cand: Candidate, glyph: Glyph, uncovered: np.ndarray, *, dense: bool = False
) -> Analysis:
    return _analyze(x_curve_of_candidate(cand), cand.degree, glyph, uncovered, dense=dense)


def feasible_score(an: Analysis, degree: int):
    return (an.newly_covered, -degree, -an.mean_surface_distance)


def structurally_feasible(an: Analysis) -> bool:
    """Trace, surface, and tail validity, independent of coverage.

    Used where newly-covered counts are meaningless (degree reduction,
    which keeps a fixed set of assigned points).
    """
    return (
        an.bounds is not None
        and an.surface_fraction >= MIN_COVERAGE
        and an.deriv_outside == 0
        and abs(an.left_slope) >= MIN_TAIL_SLOPE
        and abs(an.right_slope) >= MIN_TAIL_SLOPE
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _mutate(cand: Candidate, rng, sigma_c: float) -> Candidate | None:
    if rng.random() < COEF_MUTATION_PROB:
        coef = cand.coef.copy()
        k = int(rng.integers(0, cand.degree + 1))
        coef[k] += rng.normal(0.0, sigma_c)
        return Candidate(cand.degree, coef)
    can_inc = cand.degree < MAX_DEGREE
    can_dec = cand.degree > 0
    if can_inc and (not can_dec or rng.random() < 0.5):
        return Candidate(
            cand.degree + 1, np.append(cand.coef, rng.normal(0.0, sigma_c))
        )
    if can_dec:
        return Candidate(cand.degree - 1, cand.coef[:-1].copy())
    return None


def _mutate_coef_only(cand: Candidate, rng, sigma_c: float) -> Candidate:
    coef = cand.coef.copy()
    k = int(rng.integers(0, cand.degree + 1))
    coef[k] += rng.normal(0.0, sigma_c)
    return Candidate(cand.degree, coef)


def _coef_sigma(step: int, steps: int) -> float:
    frac = step / max(steps - 1, 1)
    return COEF_SIGMA_START * (COEF_SIGMA_END / COEF_SIGMA_START) ** frac


def _choose_p2(p: np.ndarray, p1: np.ndarray, ids: np.ndarray, rng):
    if len(ids) > SEED_P2_CHOICES:
        ids = rng.choice(ids, size=SEED_P2_CHOICES, replace=False)
    best_key, best_p2 = None, None
    for pid in ids:
        p2 = p[pid]
        ts = np.linspace(0.0, 1.0, SEED_SEGMENT_PTS)[:, None]
        seg = p1[None, :] * (1.0 - ts) + p2[None, :] * ts
        d, _ = _min_dists(seg, p)
        key = (float(d.mean()), -float(np.hypot(*(p2 - p1))))
        if best_key is None or key < best_key:
            best_key, best_p2 = key, p2
    return best_p2


def _seed_pair(p: np.ndarray, uncovered_idx: np.ndarray, rng):
    p1 = p[uncovered_idx[int(rng.integers(0, len(uncovered_idx)))]]
    dists = np.hypot(p[:, 0] - p1[0], p[:, 1] - p1[1])
    for rmax in (SEED_MAX_DIST, SEED_EXPANDED_DIST):
        mask = (dists >= SEED_MIN_DIST) & (dists <= rmax)
        ids = np.flatnonzero(mask)
        if len(ids):
            p2 = _choose_p2(p, p1, ids, rng)
            if p2 is not None:
                return p1, p2
    return None


def _line_seed_u(p1: np.ndarray, p2: np.ndarray) -> Candidate:
    u1, u2 = (p1[0] - 50.0) / 50.0, (p2[0] - 50.0) / 50.0
    du = u2 - u1
    if abs(du) < MIN_DU:
        du = MIN_DU if du >= 0.0 else -MIN_DU
    slope = (p2[1] - p1[1]) / du
    intercept = p1[1] - slope * u1
    return Candidate(1, np.array([intercept, slope]))


def _bent_seeds_u(p1: np.ndarray, p2: np.ndarray, glyph: Glyph):
    """Line plus bent tails: L(u) + k*Q(u) and L(u) + k*R(u).

    Q(u) = (u-u1)(u-u2) bends both tails the same way (two quadratic
    seeds: both up / both down). R(u) = (u-u1)(u-u2)(u-m) with
    m = (u1+u2)/2 sends the two tails in opposite directions (two cubic
    seeds). k is fit by least squares at the global padded glyph
    x-extents xmin-5 / xmax+5, so a crossbar seed can curve onto a leg
    and keep following the surface instead of escaping immediately.
    """
    u1 = (p1[0] - 50.0) / 50.0
    u2 = (p2[0] - 50.0) / 50.0
    du = u2 - u1
    if abs(du) < MIN_DU:
        du = MIN_DU if du >= 0.0 else -MIN_DU
    slope = (p2[1] - p1[1]) / du
    intercept = p1[1] - slope * u1
    L = np.array([intercept, slope])

    xL = glyph.xmin - 5.0
    xR = glyph.xmax + 5.0
    uL = (xL - 50.0) / 50.0
    uR = (xR - 50.0) / 50.0

    Q = np.array([u1 * u2, -(u1 + u2), 1.0])
    m = (u1 + u2) / 2.0
    R = np.array(
        [-m * u1 * u2, u1 * u2 + m * (u1 + u2), -(u1 + u2 + m), 1.0]
    )

    def fitted(basis, target_lo, target_hi):
        q = np.array([np.polyval(basis[::-1], uL), np.polyval(basis[::-1], uR)])
        Lv = np.array([np.polyval(L[::-1], uL), np.polyval(L[::-1], uR)])
        t = np.array([target_lo, target_hi])
        denom = float((q * q).sum())
        if denom < 1e-12:
            return None
        k = float((q * (t - Lv)).sum()) / denom
        if not np.isfinite(k):
            return None
        coef = np.zeros(len(basis))
        coef[: len(L)] = L
        coef += k * basis
        return Candidate(len(basis) - 1, coef)

    up = glyph.ymax + 5.0
    dn = glyph.ymin - 5.0
    seeds = [
        fitted(Q, up, up),
        fitted(Q, dn, dn),
        fitted(R, up, dn),
        fitted(R, dn, up),
    ]
    return [s for s in seeds if s is not None]


def _initial_seeds(p1: np.ndarray, p2: np.ndarray, glyph: Glyph):
    return [_line_seed_u(p1, p2)] + _bent_seeds_u(p1, p2, glyph)


def _hill_climb(cand, glyph, uncovered, steps, rng):
    cur = cand
    cur_an = analyze_candidate(cur, glyph, uncovered)
    best = (cur, cur_an) if cur_an.feasible else None
    for t in range(steps):
        mutant = _mutate(cur, rng, _coef_sigma(t, steps))
        if mutant is None:
            continue
        man = analyze_candidate(mutant, glyph, uncovered)
        if man.merit > cur_an.merit:
            cur, cur_an = mutant, man
        if man.feasible and (
            best is None
            or feasible_score(man, mutant.degree) > feasible_score(best[1], best[0].degree)
        ):
            best = (mutant, man)
    return best


def find_curve(glyph: Glyph, uncovered: np.ndarray, rng, restarts):
    uncovered_idx = np.flatnonzero(uncovered)
    best_q = None
    for _ in range(restarts):
        pair = _seed_pair(glyph.search_points, uncovered_idx, rng)
        if pair is None:
            continue
        p1, p2 = pair
        seeds = _initial_seeds(p1, p2, glyph)
        scored = [(s, analyze_candidate(s, glyph, uncovered)) for s in seeds]
        feasible = [sm for sm in scored if sm[1].feasible]
        if feasible:
            start = max(feasible, key=lambda sm: feasible_score(sm[1], sm[0].degree))[0]
        else:
            start = max(scored, key=lambda sm: sm[1].merit)[0]
        result = _hill_climb(start, glyph, uncovered, REFINE_STEPS, rng)
        if result is None:
            continue
        cand, an = result
        if best_q is None or feasible_score(an, cand.degree) > feasible_score(
            best_q[1], best_q[0].degree
        ):
            best_q = (cand, an)
    return None if best_q is None else best_q[0]


def _refine_coef_only(cand: Candidate, glyph: Glyph, steps: int, rng):
    uncovered = np.zeros(len(glyph.search_points), dtype=bool)
    best = cand
    best_an = analyze_candidate(best, glyph, uncovered)
    best_merit = best_an.merit
    trial, trial_an = best, best_an
    for t in range(steps):
        mutant = _mutate_coef_only(trial, rng, _coef_sigma(t, steps))
        man = analyze_candidate(mutant, glyph, uncovered)
        if man.merit > trial_an.merit:
            trial, trial_an = mutant, man
        if man.merit > best_merit:
            best, best_an, best_merit = mutant, man, man.merit
    return best, best_an


def _reduce_degree(cand: Candidate, assigned_idx: np.ndarray, glyph: Glyph, rng):
    cur = cand
    while cur.degree > 0:
        truncated = Candidate(cur.degree - 1, cur.coef[:-1].copy())
        trial, _ = _refine_coef_only(truncated, glyph, REDUCE_STEPS, rng)
        an = analyze_candidate(
            trial, glyph, np.zeros(len(glyph.points), dtype=bool), dense=True
        )
        covers_all = (
            len(assigned_idx) > 0
            and bool((an.point_d[assigned_idx] <= TAU).all())
        )
        if structurally_feasible(an) and covers_all:
            cur = trial
        else:
            break
    return cur


def _assign(glyph: Glyph, covered: np.ndarray, cand: Candidate):
    an = analyze_candidate(
        cand, glyph, np.zeros(len(glyph.points), dtype=bool), dense=True
    )
    hit = an.point_d <= TAU
    new = np.flatnonzero(hit & ~covered)
    covered |= hit
    return new


def fit_curves(glyph: Glyph, rng, max_curves):
    n = len(glyph.points)
    covered = np.zeros(n, dtype=bool)
    curves = []
    while len(glyph.points) == 0 or covered.sum() < MIN_COVERAGE * n:
        if len(curves) >= max_curves:
            break
        uncovered_search = ~covered[glyph.search_idx]
        cand = find_curve(glyph, uncovered_search, rng, RESTARTS_PER_CURVE)
        if cand is None:
            cand = find_curve(glyph, uncovered_search, rng, RESCUE_RESTARTS)
        if cand is None:
            break
        assigned = _assign(glyph, covered, cand)
        curves.append(_reduce_degree(cand, assigned, glyph, rng))
    return curves


@lru_cache(maxsize=None)
def glyph_boundary(letter: str) -> np.ndarray:
    font = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    tp = TextPath((0, 0), letter, size=100, prop=FontProperties(fname=font))
    polys = tp.to_polygons()
    pts = np.vstack(polys)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    w, h = mx[0] - mn[0], mx[1] - mn[1]
    scale = SIZE / max(w, h)

    verts = []
    codes = []
    for poly in polys:
        t = np.empty_like(poly)
        t[:, 0] = (poly[:, 0] - mn[0]) * scale
        t[:, 1] = (poly[:, 1] - mn[1]) * scale
        verts.append(t)
        codes.append([Path.MOVETO] + [Path.LINETO] * (len(t) - 1))
    path = Path(np.vstack(verts), np.concatenate(codes))

    step = SIZE / GRID
    axis = (np.arange(GRID) + 0.5) * step
    gx, gy = np.meshgrid(axis, axis)
    filled = path.contains_points(
        np.column_stack([gx.ravel(), gy.ravel()])
    ).reshape(GRID, GRID)

    f = np.pad(filled, 1, constant_values=False)
    interior_filled = (
        f[1:-1, 1:-1]
        & f[:-2, 1:-1]
        & f[2:, 1:-1]
        & f[1:-1, :-2]
        & f[1:-1, 2:]
    )
    boundary = filled & ~interior_filled
    iy, ix = np.nonzero(boundary)
    return np.column_stack([(ix + 0.5) * step, (iy + 0.5) * step])


# ---------------------------------------------------------------------------
# Serialization and parsing
# ---------------------------------------------------------------------------


def fmt_num(v: float) -> str:
    s = f"{v:.{PRECISION}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def poly_str(coef: np.ndarray) -> str:
    rendered = [fmt_num(float(c)) for c in coef]
    parts = []
    first = True
    for k in range(len(coef) - 1, 0, -1):
        s = rendered[k]
        if s == "0":
            continue
        neg = s.startswith("-")
        mag = s[1:] if neg else s
        body = "x" if k == 1 else f"x^{k}"
        prefix = "" if mag == "1" else mag
        sign = "-" if neg else ("" if first else "+")
        parts.append(sign + prefix + body)
        first = False
    if rendered[0] != "0" or not parts:
        s = rendered[0]
        neg = s.startswith("-")
        mag = s[1:] if neg else s
        sign = "-" if neg else ("" if first else "+")
        parts.append(sign + mag)
    out = "".join(parts)
    return out if out else "0"


def format_expression(curve: XCurve) -> str:
    return f"y={poly_str(curve.poly.coef)}"


def serialize(cand: Candidate) -> str:
    return format_expression(x_curve_of_candidate(cand))


_EXPR_RE = re.compile(r"^y=(.+)$")
_TERM_RE = re.compile(
    r"([+-]?)((?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)?)(?:x(?:\^([0-9]+))?)?"
)


def parse_poly(s: str):
    terms = re.findall(r"[+-]?[^+-]+", s)
    acc = {}
    for term in terms:
        m = _TERM_RE.fullmatch(term)
        if m is None:
            return None
        num, exp_s = m.group(2), m.group(3)
        has_x = "x" in term
        if num == ".":
            return None
        if num == "":
            if not has_x:
                return None
            val = 1.0
        else:
            try:
                val = float(num)
            except ValueError:
                return None
        exp = int(exp_s) if exp_s is not None else (1 if has_x else 0)
        sign = -1.0 if m.group(1) == "-" else 1.0
        acc[exp] = acc.get(exp, 0.0) + sign * val
    if not acc:
        return None
    deg = max(acc)
    coef = np.zeros(deg + 1)
    for k, v in acc.items():
        coef[k] = v
    return coef


def parse_line(line: str):
    m = _EXPR_RE.match(line)
    if m is None:
        return None
    coef = parse_poly(m.group(1))
    if coef is None:
        return None
    return XCurve(np.polynomial.Polynomial(coef))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(lines, glyph: Glyph):
    problems = []
    parsed = [parse_line(l) for l in lines]
    good = [c for c in parsed if c is not None]

    bad4 = [l for l, c in zip(lines, parsed) if c is None or format_expression(c) != l]
    if bad4:
        problems.append(f"V4 round-trip failed for {len(bad4)} expression(s)")
    for c in good:
        coef = c.poly.coef
        if not np.all(np.isfinite(coef)):
            problems.append("V4 non-finite coefficient")
        elif np.abs(coef).max() >= 1e9:
            problems.append("V4 coefficient magnitude >= 1e9")
    if any("\\left" in l for l in lines):
        problems.append("V4 domain restriction emitted")

    dense = []
    for c in good:
        an = _analyze(
            c, len(c.poly.coef) - 1, glyph,
            np.zeros(len(glyph.points), dtype=bool), dense=True,
        )
        if an.bounds is None:
            problems.append("V3 expression has no single finite trace interval")
            dense.append(None)
            continue
        if an.deriv_outside > 0:
            problems.append("V3 derivative roots beyond trace interval")
            dense.append(None)
            continue
        if (
            abs(an.left_slope) < MIN_TAIL_SLOPE
            or abs(an.right_slope) < MIN_TAIL_SLOPE
        ):
            problems.append("V3 exit slope below MIN_TAIL_SLOPE")
            dense.append(None)
            continue
        dense.append(an.samples)

    valid = [s for s in dense if s is not None]
    all_samples = np.vstack(valid) if valid else np.zeros((0, 2))

    p = glyph.points
    if len(p):
        if len(all_samples) == 0:
            problems.append(f"V1 coverage 0.0000 below {MIN_COVERAGE}")
        else:
            pd, _ = _min_dists(p, all_samples)
            coverage = float((pd <= TAU).mean())
            if coverage < MIN_COVERAGE:
                problems.append(f"V1 coverage {coverage:.4f} below {MIN_COVERAGE}")
            for i, samples in enumerate(dense):
                if samples is None:
                    continue
                sd, _ = _min_dists(samples, p)
                near_frac = float((sd <= TAU).mean())
                if near_frac < MIN_COVERAGE:
                    problems.append(
                        f"V2 expression {i}: only {near_frac:.4f} of its "
                        f"trace lies within {TAU} of the glyph boundary"
                    )
    return problems


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(argv) -> int:
    letter = None
    seed = DEFAULT_SEED
    max_curves = DEFAULT_MAX_CURVES
    i = 0
    positionals = []
    while i < len(argv):
        arg = argv[i]
        if arg in ("--seed", "--max-curves"):
            if i + 1 >= len(argv):
                return 2
            value = argv[i + 1]
            try:
                number = int(value)
            except ValueError:
                return 2
            if str(number) != value and not (
                value.startswith("+") and str(number) == value[1:]
            ):
                return 2
            if arg == "--seed":
                seed = number
            else:
                max_curves = number
            i += 2
        elif arg.startswith("--"):
            return 2
        else:
            positionals.append(arg)
            i += 1
    if len(positionals) != 1:
        return 2
    letter = positionals[0]
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return 2

    glyph = make_glyph(glyph_boundary(letter))
    curves = fit_curves(glyph, np.random.default_rng(seed), max_curves)
    lines = [serialize(c) for c in curves]
    problems = validate(lines, glyph)
    if problems:
        for msg in problems:
            print(msg, file=sys.stderr)
        return 1
    sys.stdout.write("".join(l + "\n" for l in lines))
    return 0
