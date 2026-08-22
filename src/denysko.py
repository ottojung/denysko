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
MIN_X_SEPARATION = 0.1
SEED_MIN_DIST = 3.0
SEED_MAX_DIST = 15.0
SEED_EXPANDED_DIST = 25.0
SEED_P2_CHOICES = 8
SEED_SEGMENT_PTS = 33
MIN_TAIL_SLOPE = 8.0
TAIL_VERTICAL_MARGIN = 5.0
MAX_TAIL_X_RUN = 5.0
SEED_TAIL_X_RUN = MAX_TAIL_X_RUN
UNBOUNDED_SEED_HALF_WIDTH = 15.0
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
class HillResult:
    best_feasible_candidate: Candidate | None
    best_feasible_analysis: Analysis | None
    best_exploratory_candidate: Candidate
    best_exploratory_analysis: Analysis


@dataclass
class SearchBasis:
    """Structured deformation basis for endpoint-anchored seed hills.

    Q bends both tails together; R alters left-vs-right asymmetry. Both
    vanish with zero derivative at the provisional trace endpoints.
    """
    q: np.ndarray
    r: np.ndarray


@dataclass
class SeedEntry:
    """One initial seed with its family name and optional search basis."""
    name: str
    candidate: Candidate
    basis: SearchBasis | None


@dataclass
class RestartResult:
    best_feasible_candidate: Candidate | None
    best_feasible_analysis: Analysis | None
    best_exploratory_candidate: Candidate | None
    best_exploratory_analysis: Analysis | None
    best_seed_name: str | None


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
class TailInfo:
    """Per-side post-exit tail measurements (for diagnostics and penalties)."""
    margin_root_exists: bool
    x_run: float
    slope: float
    direction_ok: bool
    turns: int
    valid: bool


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
    n_components: int
    extra_component_fraction: float
    deriv_outside: int
    left_tail: TailInfo
    right_tail: TailInfo
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


def _trace_penalty_continuous(
    comps: list[tuple[float, float]], arcs: list[float]
) -> float:
    """Continuous structural trace penalty.

    For multiple finite components: 2.0 * extra_component_fraction where
    extra_component_fraction = extra_arc / max(total_arc, 1e-9), so
    shrinking a spurious component improves merit continuously before it
    vanishes. +2.0 for any unbounded in-band component; 2.0 for empty.
    """
    if not comps:
        return 2.0
    finite_arcs = [
        arc for (a, b), arc in zip(comps, arcs)
        if np.isfinite(a) and np.isfinite(b)
    ]
    unbounded = any(
        not (np.isfinite(a) and np.isfinite(b)) for a, b in comps
    )
    pen = 2.0 if unbounded else 0.0
    if len(finite_arcs) > 1:
        total = sum(finite_arcs)
        main = max(finite_arcs)
        extra = total - main
        fraction = extra / max(total, 1e-9)
        pen += 2.0 * fraction
    return pen


def _analyze(
    xc: XCurve,
    degree: int,
    glyph: Glyph,
    uncovered: np.ndarray,
    *,
    dense: bool,
) -> Analysis:
    """Single-pass measurement of an unbounded polynomial.

    Converts u -> x once, computes the roots of P-ymin, P-ymax, P' and the
    two margin levels once, derives the trace components once, samples
    EVERY finite trace component once, and derives all surface/coverage/
    structural metrics from those cached values. `uncovered` must be
    aligned with the boundary actually used (search_points unless dense,
    else points).
    """
    poly = xc.poly
    ymin, ymax = glyph.ymin, glyph.ymax
    roots_ymin = _real_roots(poly - ymin)
    roots_ymax = _real_roots(poly - ymax)
    deriv = poly.deriv()
    droots = _real_roots(deriv)
    roots_abv = _real_roots(poly - (ymax + TAIL_VERTICAL_MARGIN))
    roots_blw = _real_roots(poly - (ymin - TAIL_VERTICAL_MARGIN))
    breaks = np.unique(np.concatenate([roots_ymin, roots_ymax]))
    comps = _components_from_breaks(breaks, poly, ymin, ymax)

    bounds: tuple[float, float] | None = None
    if len(comps) == 1:
        l, r = comps[0]
        if np.isfinite(l) and np.isfinite(r) and r > l:
            bounds = (l, r)

    step, cap = (
        (VALIDATE_STEP, VALIDATE_SAMPLE_CAP) if dense else (SEARCH_STEP, SEARCH_GRAPH_MAX)
    )

    finite = [(a, b) for a, b in comps if np.isfinite(a) and np.isfinite(b)]
    all_sampled = []
    arcs = []
    view_lo, view_hi = glyph.xmin, glyph.xmax
    for a, b in comps:
        if np.isfinite(a) and np.isfinite(b):
            raw = sample_curve(xc, a, b, step, cap)
        else:
            # Unbounded in-band component: sample it over a finite
            # exploratory viewport so a horizontal stroke scores its real
            # surface/coverage instead of zero, while remaining
            # structurally penalized (arc stays infinite).
            raw = sample_curve(xc, view_lo, view_hi, step, cap)
        in_band = (raw[:, 1] >= ymin) & (raw[:, 1] <= ymax)
        s = raw[in_band]
        all_sampled.append(s)
        if not (np.isfinite(a) and np.isfinite(b)):
            arcs.append(float("inf"))
        elif len(s) < 2:
            arcs.append(0.0)
        else:
            arcs.append(
                float(np.hypot(np.diff(s[:, 0]), np.diff(s[:, 1])).sum())
            )
    samples = (
        np.vstack(all_sampled) if all_sampled else np.zeros((0, 2))
    )

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

    trace_penalty = _trace_penalty_continuous(comps, arcs)

    if bounds is not None:
        l, r = bounds
        deriv_outside = int(((droots < l) | (droots > r)).sum())
        ltail = _analyze_tail(poly, deriv, droots, ymin, ymax, l, "L", roots_abv, roots_blw)
        rtail = _analyze_tail(poly, deriv, droots, ymin, ymax, r, "R", roots_abv, roots_blw)
        # Each side penalty already includes its own derivative-turn count
        # (left.turns + right.turns == deriv_outside), so adding
        # deriv_outside here would double-count every turn. It remains a
        # separate field for hard feasibility and diagnostics only.
        tail_penalty = (
            _tail_side_penalty(ltail, ymin, ymax, poly, deriv, l, "L")
            + _tail_side_penalty(rtail, ymin, ymax, poly, deriv, r, "R")
        )
    else:
        deriv_outside = 0
        ltail = TailInfo(False, float("inf"), 0.0, False, 0, False)
        rtail = TailInfo(False, float("inf"), 0.0, False, 0, False)
        tail_penalty = 0.0

    feasible = (
        bounds is not None
        and surface_fraction >= MIN_COVERAGE
        and deriv_outside == 0
        and ltail.valid
        and rtail.valid
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
    finite_arcs = [
        arc for (a, b), arc in zip(comps, arcs)
        if np.isfinite(a) and np.isfinite(b)
    ]
    if len(finite_arcs) > 1:
        total = sum(finite_arcs)
        extra = total - max(finite_arcs)
        extra_fraction = extra / max(total, 1e-9)
    else:
        extra_fraction = 0.0
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
        len(comps),
        extra_fraction,
        deriv_outside,
        ltail,
        rtail,
        feasible,
        merit,
    )


def _analyze_tail(
    poly,
    deriv,
    droots,
    ymin: float,
    ymax: float,
    end: float,
    side: str,
    roots_abv,
    roots_blw,
) -> TailInfo:
    """Post-exit tail analysis for one side.

    `end` is a trace endpoint (`l` for "L", `r` for "R"). Determines the
    exit direction, finds the nearest ±5 vertical-margin point outward via
    polynomial roots, measures x-run and slope there, and validates
    permanent escape.
    """
    eps = 1e-5
    if side == "R":
        probe = float(poly(end + eps))
        above = roots_abv[roots_abv > end]
        below = roots_blw[roots_blw > end]
        want = 1.0 if probe > ymax else -1.0 if probe < ymin else 0.0
        turns = int((droots > end).sum())
        run_of = lambda t: t - end
    else:
        probe = float(poly(end - eps))
        above = roots_abv[roots_abv < end]
        below = roots_blw[roots_blw < end]
        want = -1.0 if probe > ymax else 1.0 if probe < ymin else 0.0
        turns = int((droots < end).sum())
        run_of = lambda t: end - t

    if want == 0.0:
        return TailInfo(False, float("inf"), 0.0, False, turns, False)

    cands = above if probe > ymax else below
    if len(cands) == 0:
        return TailInfo(False, float("inf"), 0.0, False, turns, False)

    t = float(cands[0] if side == "R" else cands[-1])
    x_run = run_of(t)
    slope = abs(float(deriv(t)))
    away = want * np.sign(float(deriv(end + eps if side == "R" else end - eps))) > 0
    valid = (
        turns == 0
        and away
        and x_run <= MAX_TAIL_X_RUN
        and slope >= MIN_TAIL_SLOPE
    )
    return TailInfo(True, x_run, slope, away, turns, valid)


def _tail_side_penalty(info: TailInfo, ymin, ymax, poly, deriv, end, side) -> float:
    """Continuous exploration penalty for one tail.

    Missing margin: estimate how close the tail gets by probing at
    end +/- MAX_TAIL_X_RUN and computing the remaining vertical
    fraction toward the target; penalty 1.0 + remaining_fraction. A wrong
    direction adds a larger fixed penalty. When a margin root exists, the
    x-run and slope deficits are proportional.
    """
    pen = float(info.turns)
    if not info.direction_ok:
        pen += 2.0
    if not info.margin_root_exists:
        probe_x = end + MAX_TAIL_X_RUN if side == "R" else end - MAX_TAIL_X_RUN
        probe = float(poly(probe_x))
        if probe > ymax:
            target = ymax + TAIL_VERTICAL_MARGIN
            remaining = max(0.0, target - probe) / TAIL_VERTICAL_MARGIN
        elif probe < ymin:
            target = ymin - TAIL_VERTICAL_MARGIN
            remaining = max(0.0, probe - target) / TAIL_VERTICAL_MARGIN
        else:
            remaining = 1.0
        pen += 1.0 + remaining
    else:
        pen += max(0.0, info.x_run - MAX_TAIL_X_RUN) / MAX_TAIL_X_RUN
        pen += max(0.0, MIN_TAIL_SLOPE - info.slope) / MIN_TAIL_SLOPE
    return pen


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
        and an.left_tail.valid
        and an.right_tail.valid
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
    return _mutate_degree(cand, rng, sigma_c)


def _mutate_degree(cand: Candidate, rng, sigma_c: float) -> Candidate | None:
    can_inc = cand.degree < MAX_DEGREE
    can_dec = cand.degree > 0
    if can_inc and (not can_dec or rng.random() < 0.5):
        return Candidate(
            cand.degree + 1, np.append(cand.coef, rng.normal(0.0, sigma_c))
        )
    if can_dec:
        return Candidate(cand.degree - 1, cand.coef[:-1].copy())
    return None


def _mutate_search(
    cand: Candidate,
    basis: SearchBasis | None,
    rng,
    sigma_c: float,
    allow_degree: bool,
) -> Candidate | None:
    """One search mutation.

    Plain-line hills keep the ordinary 80 % coefficient / 20 % degree
    behaviour. Structured endpoint-anchored hills use 50 % coefficient /
    25 % Q-direction / 25 % R-direction moves (Q bends both tails
    together, R alters left-vs-right asymmetry), with degree mutation
    only enabled in the second half of refinement so early steps refine
    the constructed quintic basin instead of destroying it.
    """
    if basis is None:
        if not allow_degree:
            coef = cand.coef.copy()
            k = int(rng.integers(0, cand.degree + 1))
            coef[k] += rng.normal(0.0, sigma_c)
            return Candidate(cand.degree, coef)
        return _mutate(cand, rng, sigma_c)

    if allow_degree and rng.random() < 0.20:
        return _mutate_degree(cand, rng, sigma_c)

    r = rng.random()
    coef = cand.coef.copy()
    if r < 0.50:
        k = int(rng.integers(0, cand.degree + 1))
        coef[k] += rng.normal(0.0, sigma_c)
    elif r < 0.75:
        coef[: len(basis.q)] += rng.normal(0.0, sigma_c) * basis.q
    else:
        coef[: len(basis.r)] += rng.normal(0.0, sigma_c) * basis.r
    return Candidate(cand.degree, coef)


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
        # Lower mean then lower max is better; farther p2 wins final ties.
        key = (
            float(d.mean()),
            float(d.max()),
            -float(np.hypot(*(p2 - p1))),
        )
        if best_key is None or key < best_key:
            best_key, best_p2 = key, p2
    return best_p2


def _seed_pair(p: np.ndarray, uncovered_idx: np.ndarray, rng):
    p1 = p[uncovered_idx[int(rng.integers(0, len(uncovered_idx)))]]
    dists = np.hypot(p[:, 0] - p1[0], p[:, 1] - p1[1])
    dx = np.abs(p[:, 0] - p1[0])
    for rmax in (SEED_MAX_DIST, SEED_EXPANDED_DIST):
        mask = (
            (dists >= SEED_MIN_DIST)
            & (dists <= rmax)
            & (dx >= MIN_X_SEPARATION)
        )
        ids = np.flatnonzero(mask)
        if len(ids):
            p2 = _choose_p2(p, p1, ids, rng)
            if p2 is not None:
                return p1, p2
    return None


def _line_seed_u(p1: np.ndarray, p2: np.ndarray) -> Candidate:
    u1, u2 = (p1[0] - 50.0) / 50.0, (p2[0] - 50.0) / 50.0
    slope = (p2[1] - p1[1]) / (u2 - u1)
    intercept = p1[1] - slope * u1
    return Candidate(1, np.array([intercept, slope]))


def _provisional_trace_window(
    p1: np.ndarray, p2: np.ndarray, line: Candidate, glyph: Glyph
) -> tuple[float, float]:
    """Provisional in-band trace endpoints [l0, r0] for a seed line.

    For a line with one finite trace interval these are its natural band
    exits, derived analytically from intersections with ymin/ymax. For an
    unbounded (horizontal/nearly horizontal) line a finite working window
    is defined around the seed-pair midpoint with a fixed half-width,
    clamped to the padded glyph extents. Used only for seed construction.
    """
    xc = x_curve_of_candidate(line)
    breaks = np.unique(
        np.concatenate(
            [
                _real_roots(xc.poly - glyph.ymin),
                _real_roots(xc.poly - glyph.ymax),
            ]
        )
    )
    comps = _components_from_breaks(breaks, xc.poly, glyph.ymin, glyph.ymax)
    finite = [
        (a, b) for a, b in comps if np.isfinite(a) and np.isfinite(b)
    ]
    unbounded = any(
        not (np.isfinite(a) and np.isfinite(b)) for a, b in comps
    )
    if len(finite) == 1 and not unbounded:
        l0, r0 = finite[0]
    else:
        center = 0.5 * (p1[0] + p2[0])
        l0 = center - UNBOUNDED_SEED_HALF_WIDTH
        r0 = center + UNBOUNDED_SEED_HALF_WIDTH
    lo, hi = glyph.xmin - 5.0, glyph.xmax + 5.0
    l0 = min(max(l0, lo), hi)
    r0 = min(max(r0, lo), hi)
    if r0 <= l0:
        l0, r0 = lo, hi
    return float(l0), float(r0)


def _seed_family(p1: np.ndarray, p2: np.ndarray, glyph: Glyph):
    """Five seeds anchored at the line's provisional trace exits.

    The line `L` is kept for naturally steep strokes and lower-degree
    solutions. Four degree-5 bent seeds P(u) = L(u) + aQ(u) + bR(u) use
    Q(u) = (u-ul)^2 (u-ur)^2 and R(u) = Q(u)(u-m), m = (ul+ur)/2, where
    ul/ur correspond to the provisional trace endpoints l0/r0. Both bases
    vanish with zero derivative there, so each bent seed follows the
    provisional straight surface route all the way to its natural band
    exits, then bends outside the band. The pair (a,b) is solved exactly
    so that P reaches the requested tail levels at xL = l0 -
    SEED_TAIL_X_RUN and xR = r0 + SEED_TAIL_X_RUN.
    """
    line = _line_seed_u(p1, p2)
    l0, r0 = _provisional_trace_window(p1, p2, line, glyph)
    ul = (l0 - 50.0) / 50.0
    ur = (r0 - 50.0) / 50.0

    w = np.array([ul * ur, -(ul + ur), 1.0])
    Q = np.polynomial.polynomial.polymul(w, w)
    m = 0.5 * (ul + ur)
    R = np.polynomial.polynomial.polymul(Q, np.array([-m, 1.0]))

    Lcoef = line.coef
    xL = l0 - SEED_TAIL_X_RUN
    xR = r0 + SEED_TAIL_X_RUN
    uL = (xL - 50.0) / 50.0
    uR = (xR - 50.0) / 50.0
    LvL = float(np.polyval(Lcoef[::-1], uL))
    LvR = float(np.polyval(Lcoef[::-1], uR))

    def solve_seed(target_lo, target_hi):
        qL, qR = float(np.polyval(Q[::-1], uL)), float(np.polyval(Q[::-1], uR))
        rL, rR = float(np.polyval(R[::-1], uL)), float(np.polyval(R[::-1], uR))
        M = np.array([[qL, rL], [qR, rR]])
        rhs = np.array([target_lo - LvL, target_hi - LvR])
        try:
            cond = np.linalg.cond(M)
            if cond > 1e12 or not np.isfinite(cond):
                return None
            a, b = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None
        if not (np.isfinite(a) and np.isfinite(b)):
            return None
        coef = np.zeros(len(R))
        coef[: len(Lcoef)] = Lcoef
        q_pad = np.zeros(len(R))
        q_pad[: len(Q)] = Q
        coef += a * q_pad + b * R
        return Candidate(len(R) - 1, coef)

    up = glyph.ymax + TAIL_VERTICAL_MARGIN
    dn = glyph.ymin - TAIL_VERTICAL_MARGIN
    entries = [SeedEntry("line", line, None)]
    for name, tl, tr in (
        ("up/up", up, up),
        ("down/down", dn, dn),
        ("up/down", up, dn),
        ("down/up", dn, up),
    ):
        cand = solve_seed(tl, tr)
        if cand is not None:
            entries.append(SeedEntry(name, cand, SearchBasis(Q, R)))
    return entries


def _hill_climb(cand, glyph, uncovered, steps, rng, basis=None) -> HillResult:
    """Greedy hill climb tracking current, best exploratory, and best
    feasible states separately. The best exploratory state is the
    highest-merit candidate seen anywhere (including the start).

    When a structured SearchBasis is supplied, mutations use the
    Q/R-direction moves and degree mutation is suppressed during the
    first half of refinement so the constructed quintic basin is refined
    before structural exploration resumes.
    """
    cur = cand
    cur_an = analyze_candidate(cur, glyph, uncovered)
    best_explore = (cur, cur_an)
    best_feasible = (cur, cur_an) if cur_an.feasible else None
    half = steps // 2
    for t in range(steps):
        mutant = _mutate_search(
            cur, basis, rng, _coef_sigma(t, steps), allow_degree=(t >= half)
        )
        if mutant is None:
            continue
        man = analyze_candidate(mutant, glyph, uncovered)
        if man.merit > cur_an.merit:
            cur, cur_an = mutant, man
        if man.merit > best_explore[1].merit:
            best_explore = (mutant, man)
        if man.feasible and (
            best_feasible is None
            or feasible_score(man, mutant.degree)
            > feasible_score(best_feasible[1], best_feasible[0].degree)
        ):
            best_feasible = (mutant, man)
    return HillResult(
        None if best_feasible is None else best_feasible[0],
        None if best_feasible is None else best_feasible[1],
        best_explore[0],
        best_explore[1],
    )


def _split_steps(total: int, n: int):
    """Deterministic split of a per-restart refinement budget across n
    seeds; the remainder goes to the first seeds."""
    base = max(0, total) // max(1, n)
    rem = max(0, total) - base * max(1, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _run_restart(
    entries, glyph: Glyph, uncovered: np.ndarray, rng, total_steps: int
) -> RestartResult:
    """Independently refine every seed family member, then compare.

    The per-restart refinement budget is split deterministically across
    the seeds rather than multiplied, so exploring all five basins costs
    the same total work as refining one.
    """
    steps_list = _split_steps(total_steps, len(entries))
    best_feasible = None
    best_explore = None
    best_name = None
    for entry, steps in zip(entries, steps_list):
        result = _hill_climb(
            entry.candidate, glyph, uncovered, steps, rng, basis=entry.basis
        )
        if result.best_feasible_candidate is not None:
            cand, an = (
                result.best_feasible_candidate,
                result.best_feasible_analysis,
            )
            if best_feasible is None or feasible_score(
                an, cand.degree
            ) > feasible_score(best_feasible[1], best_feasible[0].degree):
                best_feasible = (cand, an)
        if (
            best_explore is None
            or result.best_exploratory_analysis.merit > best_explore[1].merit
        ):
            best_explore = (
                result.best_exploratory_candidate,
                result.best_exploratory_analysis,
            )
            best_name = entry.name
    return RestartResult(
        None if best_feasible is None else best_feasible[0],
        None if best_feasible is None else best_feasible[1],
        None if best_explore is None else best_explore[0],
        None if best_explore is None else best_explore[1],
        best_name,
    )


def find_curve(glyph: Glyph, uncovered: np.ndarray, rng, restarts):
    """Return (feasible candidate or None, best explored candidate or None,
    best explored analysis or None, winning seed family name or None).

    The best exploratory state is the highest-merit candidate seen across
    all restarts and all five seed hills, used for diagnostics when no
    feasible curve is found.
    """
    uncovered_idx = np.flatnonzero(uncovered)
    best_q = None
    best_explore = None
    best_name = None
    for _ in range(restarts):
        pair = _seed_pair(glyph.search_points, uncovered_idx, rng)
        if pair is None:
            continue
        p1, p2 = pair
        entries = _seed_family(p1, p2, glyph)
        result = _run_restart(entries, glyph, uncovered, rng, REFINE_STEPS)
        if result.best_feasible_candidate is not None:
            cand, an = (
                result.best_feasible_candidate,
                result.best_feasible_analysis,
            )
            if best_q is None or feasible_score(an, cand.degree) > feasible_score(
                best_q[1], best_q[0].degree
            ):
                best_q = (cand, an)
        if (
            result.best_exploratory_analysis is not None
            and (
                best_explore is None
                or result.best_exploratory_analysis.merit > best_explore[1].merit
            )
        ):
            best_explore = (
                result.best_exploratory_candidate,
                result.best_exploratory_analysis,
            )
            best_name = result.best_seed_name
    return (
        None if best_q is None else best_q[0],
        None if best_explore is None else best_explore[0],
        None if best_explore is None else best_explore[1],
        best_name,
    )


def _tail_detail(info: TailInfo) -> str:
    direction = "ok" if info.direction_ok else "bad"
    margin = "yes" if info.margin_root_exists else "no"
    return (
        f"direction={direction}\n  margin={margin}\n  x_run={info.x_run:.2f}\n"
        f"  slope={info.slope:.2f}\n  turns={info.turns}"
    )


def _report_no_first_curve(
    glyph: Glyph,
    cand: Candidate | None,
    an: Analysis | None,
    uncovered: np.ndarray,
    seed_name: str | None,
):
    """Print a concise stderr diagnostic when fit_curves finds nothing.

    Describes the actual best explored state (the highest-merit candidate
    across all restarts and all five seed hills), using the real
    uncovered mask so `new=` is the genuine newly-covered count. Never on
    stdout.
    """
    if cand is None or an is None:
        print("search: no feasible first curve (no restart produced a candidate)",
              file=sys.stderr)
        return
    print("search: no feasible first curve; best explored state:", file=sys.stderr)
    print(f"seed={seed_name if seed_name is not None else 'unknown'}", file=sys.stderr)
    print(f"degree={cand.degree}", file=sys.stderr)
    print(f"merit={an.merit:.2f}", file=sys.stderr)
    print(f"surface={an.surface_fraction:.2f}", file=sys.stderr)
    print(f"new={an.newly_covered}", file=sys.stderr)
    print(f"trace_components={an.n_components}", file=sys.stderr)
    print(f"extra_trace_fraction={an.extra_component_fraction:.2f}", file=sys.stderr)
    if an.bounds is not None:
        print(f"trace_bounds=[{an.bounds[0]:.2f}, {an.bounds[1]:.2f}]", file=sys.stderr)
        print("left_tail:", file=sys.stderr)
        print("  " + _tail_detail(an.left_tail), file=sys.stderr)
        print("right_tail:", file=sys.stderr)
        print("  " + _tail_detail(an.right_tail), file=sys.stderr)
    else:
        print("tails=not analyzed: trace is not single-component", file=sys.stderr)


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
        cand, expl, expl_an, expl_name = find_curve(
            glyph, uncovered_search, rng, RESTARTS_PER_CURVE
        )
        if cand is None:
            cand2, expl2, expl_an2, name2 = find_curve(
                glyph, uncovered_search, rng, RESCUE_RESTARTS
            )
            cand = cand2
            if expl is None or (expl2 is not None and expl_an2.merit > expl_an.merit):
                expl, expl_an, expl_name = expl2, expl_an2, name2
        if cand is None:
            if len(curves) == 0:
                _report_no_first_curve(
                    glyph, expl, expl_an, uncovered_search, expl_name
                )
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
        if not an.left_tail.valid or not an.right_tail.valid:
            problems.append("V3 tail does not leave the band steeply and permanently")
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
