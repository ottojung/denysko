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
MARK_STEP = 0.25
SEARCH_SAMPLE_CAP = 1200
VALIDATE_SAMPLE_CAP = 40000

MIN_NEW_POINTS = 8
MIN_DU = 0.01
SEED_MIN_DIST = 3.0
SEED_MAX_DIST = 15.0
SEED_EXPANDED_DIST = 25.0
SEED_P2_CHOICES = 8
SEED_SEGMENT_PTS = 33
TAIL_PAD = 5.0
MIN_TAIL_SLOPE = 8.0
COEF_MUTATION_PROB = 0.80
COEF_SIGMA_START, COEF_SIGMA_END = 10.0, 0.2

BIG_PENALTY = 1e6
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
    xmin: float
    xmax: float
    ymin: float
    ymax: float


U_OF_X = np.polynomial.Polynomial([-1.0, 0.02])


def x_curve_of_candidate(cand: Candidate) -> XCurve:
    return XCurve(np.polynomial.Polynomial(cand.coef)(U_OF_X))


def _poly_u(coef: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.polyval(coef[::-1], (np.asarray(xs, dtype=float) - 50.0) / 50.0)


def make_glyph(points: np.ndarray) -> Glyph:
    return Glyph(
        points,
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


# ---------------------------------------------------------------------------
# Trace and tail geometry of unbounded polynomials
# ---------------------------------------------------------------------------


def _real_roots(poly: np.polynomial.Polynomial) -> np.ndarray:
    roots = poly.roots()
    real = roots[np.abs(roots.imag) < 1e-8].real
    return np.sort(real)


def trace_intervals(curve: XCurve, ymin: float, ymax: float) -> list[tuple[float, float]]:
    """Connected x-intervals where ymin <= curve(x) <= ymax.

    Endpoints are breakpoints where the graph crosses the band edges; an
    interval may extend to -inf or +inf when a tail never leaves the band.
    """
    breaks = np.concatenate(
        [
            _real_roots(curve.poly - ymin),
            _real_roots(curve.poly - ymax),
        ]
    )
    breaks = np.unique(breaks)
    edges = np.concatenate([[-np.inf], breaks, [np.inf]])
    intervals: list[tuple[float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isfinite(lo) and np.isfinite(hi):
            mid = 0.5 * (lo + hi)
        elif np.isfinite(hi):
            mid = hi - 1.0
        elif np.isfinite(lo):
            mid = lo + 1.0
        else:
            mid = 0.0
        y = float(curve.poly(mid))
        if not (ymin <= y <= ymax):
            continue
        if intervals and lo - intervals[-1][1] <= TRACE_EPS:
            intervals[-1] = (intervals[-1][0], hi)
        else:
            intervals.append((float(lo), float(hi)))
    return intervals


def trace_bounds(
    curve: XCurve, ymin: float, ymax: float
) -> tuple[float, float] | None:
    """The single finite non-empty trace interval [l, r], or None."""
    ints = trace_intervals(curve, ymin, ymax)
    if len(ints) != 1:
        return None
    l, r = ints[0]
    if not (np.isfinite(l) and np.isfinite(r)) or r <= l:
        return None
    return (l, r)


def tail_penalty(curve: XCurve, ymin: float, ymax: float) -> float:
    """Deterministic scalar penalizing unusable tails.

    Reflects wrong trace component count, unbounded/empty trace, derivative
    roots beyond the trace interval, and exit-slope deficit below
    MIN_TAIL_SLOPE.
    """
    ints = trace_intervals(curve, ymin, ymax)
    pen = BIG_PENALTY * max(0, len(ints) - 1)
    if len(ints) == 0:
        return pen + BIG_PENALTY
    finite = [(lo, hi) for lo, hi in ints if np.isfinite(lo) and np.isfinite(hi)]
    if not finite:
        return pen + BIG_PENALTY
    if len(finite) != len(ints):
        pen += BIG_PENALTY * (len(ints) - len(finite))
    l, r = max(finite, key=lambda t: t[1] - t[0])
    deriv = curve.poly.deriv()
    droots = _real_roots(deriv)
    pen += float(((droots < l) | (droots > r)).sum()) ** 2
    for end in (l, r):
        slope = abs(float(deriv(end)))
        pen += max(0.0, MIN_TAIL_SLOPE - slope) ** 2
    return pen


def tail_ok(curve: XCurve, l: float, r: float) -> bool:
    """Monotone-away tails plus steep exits at both trace endpoints."""
    deriv = curve.poly.deriv()
    droots = _real_roots(deriv)
    if ((droots < l) | (droots > r)).any():
        return False
    return (
        abs(float(deriv(l))) >= MIN_TAIL_SLOPE
        and abs(float(deriv(r))) >= MIN_TAIL_SLOPE
    )


# ---------------------------------------------------------------------------
# Sampling and distances
# ---------------------------------------------------------------------------


def _adaptive_sample(eval_fn, a: float, b: float, max_step: float, cap: int):
    xs = np.linspace(a, b, 129)
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
        d2 = (
            (ai * ai).sum(axis=1)[:, None]
            - 2.0 * (ai @ b.T)
            + b2[None, :]
        )
        np.maximum(d2, 0.0, out=d2)
        np.minimum(da[i : i + block], np.sqrt(d2.min(axis=1)), out=da[i : i + block])
        np.minimum(db, np.sqrt(d2.min(axis=0)), out=db)
    return da, db


@dataclass
class Measurement:
    samples: np.ndarray
    point_d: np.ndarray | None
    newly_covered: int
    surface_penalty: float
    mean_surface_distance: float
    tail_pen: float
    trace_single: bool
    tails_monotone: bool
    surface_valid: bool

    @property
    def structurally_feasible(self) -> bool:
        return self.trace_single and self.tails_monotone and self.surface_valid


def _working_interval(curve: XCurve, glyph: Glyph) -> tuple[float, float]:
    """Interval whose in-band samples drive scoring when the trace itself
    is not usable: the widest finite component, else the grown glyph range."""
    ints = [
        (lo, hi)
        for lo, hi in trace_intervals(curve, glyph.ymin, glyph.ymax)
        if np.isfinite(lo) and np.isfinite(hi)
    ]
    if ints:
        return max(ints, key=lambda t: t[1] - t[0])
    return glyph.xmin - TAIL_PAD, glyph.xmax + TAIL_PAD


def measure(cand: Candidate, glyph: Glyph, uncovered: np.ndarray) -> Measurement:
    curve = x_curve_of_candidate(cand)
    bounds = trace_bounds(curve, glyph.ymin, glyph.ymax)
    lo, hi = bounds if bounds is not None else _working_interval(curve, glyph)
    raw = sample_curve(curve, lo, hi, SEARCH_STEP, SEARCH_SAMPLE_CAP)
    in_band = (raw[:, 1] >= glyph.ymin) & (raw[:, 1] <= glyph.ymax)
    samples = raw[in_band]
    tail_pen = tail_penalty(curve, glyph.ymin, glyph.ymax)
    tails_monotone = bool(
        bounds is not None and tail_ok(curve, bounds[0], bounds[1])
    )
    if len(samples) == 0:
        return Measurement(
            samples, np.full(len(glyph.points), np.inf), 0,
            0.0, 0.0, tail_pen,
            bounds is not None, tails_monotone, False,
        )
    sample_d, point_d = _min_dists(samples, glyph.points)
    excess = np.clip(sample_d - TAU, 0.0, None)
    surface_valid = bool((sample_d <= TAU).mean() >= MIN_COVERAGE)
    return Measurement(
        samples,
        point_d,
        int(((point_d <= TAU) & uncovered).sum()),
        float((excess**2).sum()),
        float(sample_d.mean()),
        tail_pen,
        bounds is not None,
        tails_monotone,
        surface_valid,
    )


def exploration_score(m: Measurement, degree: int):
    return (
        m.newly_covered,
        -m.surface_penalty,
        -m.tail_pen,
        -degree,
        -m.mean_surface_distance,
    )


def feasible_score(m: Measurement, degree: int):
    return (m.newly_covered, -degree, -m.mean_surface_distance)


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


def _choose_p2(p: np.ndarray, p1: np.ndarray, ids: np.ndarray, rng) -> np.ndarray | None:
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


def _cubic_seeds_u(p1: np.ndarray, p2: np.ndarray, glyph: Glyph):
    xl = min(p1[0], p2[0]) - TAIL_PAD
    xr = max(p1[0], p2[0]) + TAIL_PAD
    targets = [
        (glyph.ymax + TAIL_PAD, glyph.ymax + TAIL_PAD),
        (glyph.ymin - TAIL_PAD, glyph.ymin - TAIL_PAD),
        (glyph.ymax + TAIL_PAD, glyph.ymin - TAIL_PAD),
        (glyph.ymin - TAIL_PAD, glyph.ymax + TAIL_PAD),
    ]
    us = np.array(
        [
            (p1[0] - 50.0) / 50.0,
            (p2[0] - 50.0) / 50.0,
            (xl - 50.0) / 50.0,
            (xr - 50.0) / 50.0,
        ]
    )
    if abs(us[1] - us[0]) < MIN_DU:
        us[1] = us[0] + (MIN_DU if us[1] >= us[0] else -MIN_DU)
    vander = np.vander(us, 4, increasing=True)
    seeds = []
    for tl, tr in targets:
        ys = np.array([p1[1], p2[1], tl, tr])
        try:
            coef = np.linalg.solve(vander, ys)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(coef)):
            seeds.append(Candidate(3, coef))
    return seeds


def find_curve(p, glyph: Glyph, uncovered: np.ndarray, rng, restarts):
    uncovered_idx = np.flatnonzero(uncovered)
    best_q = None
    for _ in range(restarts):
        pair = _seed_pair(p, uncovered_idx, rng)
        if pair is None:
            continue
        p1, p2 = pair
        seeds = [_line_seed_u(p1, p2)] + _cubic_seeds_u(p1, p2, glyph)
        scored = [(seed, measure(seed, glyph, uncovered)) for seed in seeds]
        feasible = [sm for sm in scored if sm[1].structurally_feasible]
        if feasible:
            key = lambda sm: feasible_score(sm[1], sm[0].degree)
            start = max(feasible, key=key)[0]
        else:
            key = lambda sm: exploration_score(sm[1], sm[0].degree)
            start = max(scored, key=key)[0]
        best_cand, best_m = _hill_climb(start, glyph, uncovered, REFINE_STEPS, rng)
        qualifies = (
            best_cand is not None
            and best_m.structurally_feasible
            and best_m.newly_covered >= MIN_NEW_POINTS
        )
        if not qualifies:
            continue
        if best_q is None or feasible_score(
            best_m, best_cand.degree
        ) > feasible_score(best_q[1], best_q[0].degree):
            best_q = (best_cand, best_m)
    return None if best_q is None else best_q[0]


def _hill_climb(cand: Candidate, glyph: Glyph, uncovered: np.ndarray, steps: int, rng):
    cur = cand
    cur_m = measure(cur, glyph, uncovered)
    best: tuple[Candidate, Measurement] | None = (
        (cur, cur_m) if cur_m.structurally_feasible else None
    )
    for t in range(steps):
        mutant = _mutate(cur, rng, _coef_sigma(t, steps))
        if mutant is None:
            continue
        mm = measure(mutant, glyph, uncovered)
        if exploration_score(mm, mutant.degree) > exploration_score(cur_m, cur.degree):
            cur, cur_m = mutant, mm
        if mm.structurally_feasible and (
            best is None
            or feasible_score(mm, mutant.degree) > feasible_score(best[1], best[0].degree)
        ):
            best = (mutant, mm)
    return (None, cur_m) if best is None else best


def assign_points(glyph: Glyph, covered: np.ndarray, cand: Candidate) -> np.ndarray:
    m = measure(cand, glyph, np.zeros(len(glyph.points), dtype=bool))
    hit = (m.point_d <= TAU) if m.point_d is not None else np.zeros(
        len(glyph.points), dtype=bool
    )
    new = np.flatnonzero(hit & ~covered)
    covered |= hit
    return new


def _refine_coef_only(cand: Candidate, glyph: Glyph, steps: int, rng):
    best = cand
    best_m = measure(best, glyph, np.zeros(len(glyph.points), dtype=bool))
    best_score = exploration_score(best_m, best.degree)
    trial, trial_score = best, best_score
    for t in range(steps):
        mutant = _mutate_coef_only(trial, rng, _coef_sigma(t, steps))
        mm = measure(mutant, glyph, np.zeros(len(glyph.points), dtype=bool))
        ms = exploration_score(mm, mutant.degree)
        if ms > trial_score:
            trial, trial_score = mutant, ms
        if ms > best_score:
            best, best_m, best_score = mutant, mm, ms
    return best, best_m


def _reduce_degree(cand: Candidate, assigned_idx: np.ndarray, glyph: Glyph, rng):
    cur = cand
    while cur.degree > 0:
        truncated = Candidate(cur.degree - 1, cur.coef[:-1].copy())
        trial, m = _refine_coef_only(truncated, glyph, REDUCE_STEPS, rng)
        covers_all = (
            len(assigned_idx) > 0
            and bool((m.point_d[assigned_idx] <= TAU).all())
        )
        if m.structurally_feasible and covers_all:
            cur = trial
        else:
            break
    return cur


def fit_curves(glyph: Glyph, rng, max_curves):
    p = glyph.points
    covered = np.zeros(len(p), dtype=bool)
    curves = []
    while len(p) == 0 or covered.sum() < MIN_COVERAGE * len(p):
        if len(curves) >= max_curves:
            break
        cand = find_curve(p, glyph, ~covered, rng, RESTARTS_PER_CURVE)
        if cand is None:
            cand = find_curve(p, glyph, ~covered, rng, RESCUE_RESTARTS)
        if cand is None:
            break
        assigned = assign_points(glyph, covered, cand)
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
            continue
        if np.abs(coef).max() >= 1e9:
            problems.append("V4 coefficient magnitude >= 1e9")
    if any("\\left" in l for l in lines):
        problems.append("V4 domain restriction emitted")

    traces = []
    dense_samples = []
    for c in good:
        bounds = trace_bounds(c, glyph.ymin, glyph.ymax)
        if bounds is None:
            problems.append(
                "V3 expression does not have exactly one finite trace interval"
            )
            traces.append(None)
            dense_samples.append(None)
            continue
        l, r = bounds
        if not tail_ok(c, l, r):
            problems.append(
                f"V3 tail behaviour violated in expression over [{fmt_num(l)}, {fmt_num(r)}]"
            )
            traces.append(None)
            dense_samples.append(None)
            continue
        traces.append(bounds)
        dense_samples.append(sample_curve(c, l, r, VALIDATE_STEP, VALIDATE_SAMPLE_CAP))

    valid = [s for s in dense_samples if s is not None]
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
            for i, samples in enumerate(dense_samples):
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
