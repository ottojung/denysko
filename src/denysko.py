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
DOMAIN_LO, DOMAIN_HI = -5.0, 105.0
MARGIN = 5.0
MIN_COVERAGE = 0.95
DEFAULT_SEED = 0
DEFAULT_MAX_CURVES = 12
PRECISION = 6

SEARCH_STEP = 1.0
VALIDATE_STEP = 0.1
MARK_STEP = 0.25
SEARCH_SAMPLE_CAP = 1200
VALIDATE_SAMPLE_CAP = 40000

MIN_NEW_POINTS = 8
SEED_MIN_DIST = 3.0
SEED_MAX_DIST = 15.0
SEED_EXPANDED_DIST = 25.0
MIN_DOMAIN_WIDTH = 0.1
MIN_X_SEPARATION = 0.1
COEF_SIGMA_START, COEF_SIGMA_END = 10.0, 0.2
DOMAIN_SIGMA_START, DOMAIN_SIGMA_END = 5.0, 0.1


@dataclass
class Candidate:
    degree: int
    coef: np.ndarray
    a: float
    b: float


def _poly_u(coef: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.polyval(coef[::-1], (np.asarray(xs, dtype=float) - 50.0) / 50.0)


def sample_graph(cand: Candidate, max_step: float, cap: int) -> np.ndarray:
    xs = np.linspace(cand.a, cand.b, 129)
    for _ in range(64):
        if xs.size >= cap:
            break
        ys = _poly_u(cand.coef, xs)
        gaps = np.hypot(np.diff(xs), np.diff(ys))
        bad = gaps > max_step
        if not bad.any():
            break
        mids = (xs[:-1][bad] + xs[1:][bad]) / 2.0
        xs = np.sort(np.concatenate([xs, mids]))
    return np.column_stack([xs, _poly_u(cand.coef, xs)])[:cap]


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


def _box_filter(p: np.ndarray, samples: np.ndarray) -> np.ndarray:
    lo = samples.min(axis=0) - TAU
    hi = samples.max(axis=0) + TAU
    return np.flatnonzero(
        (p[:, 0] >= lo[0])
        & (p[:, 0] <= hi[0])
        & (p[:, 1] >= lo[1])
        & (p[:, 1] <= hi[1])
    )


def _search_score(cand: Candidate, p: np.ndarray, uncovered: np.ndarray):
    samples = sample_graph(cand, SEARCH_STEP, SEARCH_SAMPLE_CAP)
    sel = _box_filter(p, samples)
    if len(samples) == 0 or len(sel) == 0:
        return (-10**9, -(10.0**18), 0, -cand.degree, 10.0**18), 10**9, 0
    curve_d, point_d = _min_dists(samples, p[sel])
    escaped = int((curve_d > TAU).sum())
    penalty = float((np.clip(curve_d - TAU, 0.0, None) ** 2).sum())
    newly = int(((point_d <= TAU) & uncovered[sel]).sum())
    score = (
        -escaped,
        -penalty,
        newly,
        -cand.degree,
        -float(curve_d.mean()),
    )
    return score, escaped, newly


def _mutate(cand: Candidate, rng, sigma_c: float, sigma_d: float):
    r = rng.random()
    coef = cand.coef.copy()
    degree, a, b = cand.degree, cand.a, cand.b
    if r < 0.65:
        k = int(rng.integers(0, degree + 1))
        coef[k] += rng.normal(0.0, sigma_c)
    elif r < 0.90:
        u = rng.random()
        if u < 0.45:
            a += rng.normal(0.0, sigma_d)
        elif u < 0.90:
            b += rng.normal(0.0, sigma_d)
        else:
            shift = rng.normal(0.0, sigma_d)
            a += shift
            b += shift
        a = min(max(a, DOMAIN_LO), DOMAIN_HI)
        b = min(max(b, DOMAIN_LO), DOMAIN_HI)
        if b < a:
            a, b = b, a
        if b - a < MIN_DOMAIN_WIDTH:
            return None
    else:
        can_inc = degree < MAX_DEGREE
        can_dec = degree > 0
        if can_inc and (not can_dec or rng.random() < 0.5):
            degree += 1
            coef = np.append(coef, rng.normal(0.0, sigma_c))
        elif can_dec:
            degree -= 1
            coef = coef[:-1].copy()
        else:
            return None
    return Candidate(degree, coef, a, b)


def _sigmas(step: int, steps: int) -> tuple[float, float]:
    frac = step / max(steps - 1, 1)
    sc = COEF_SIGMA_START * (COEF_SIGMA_END / COEF_SIGMA_START) ** frac
    sd = DOMAIN_SIGMA_START * (DOMAIN_SIGMA_END / DOMAIN_SIGMA_START) ** frac
    return sc, sd


def _hill_climb(cand, p, uncovered, steps, rng):
    best_cand = cand
    score, esc, new = _search_score(cand, p, uncovered)
    for t in range(steps):
        sigma_c, sigma_d = _sigmas(t, steps)
        mutant = _mutate(best_cand, rng, sigma_c, sigma_d)
        if mutant is None:
            continue
        mscore, mesc, mnew = _search_score(mutant, p, uncovered)
        if mscore > score:
            best_cand, score, esc, new = mutant, mscore, mesc, mnew
    return best_cand, score, esc, new


def _seed_candidate(rng, p: np.ndarray, uncovered_idx: np.ndarray):
    p1 = p[uncovered_idx[int(rng.integers(0, len(uncovered_idx)))]]
    dists = np.hypot(p[:, 0] - p1[0], p[:, 1] - p1[1])
    dx = np.abs(p[:, 0] - p1[0])
    for rmax in (SEED_MAX_DIST, SEED_EXPANDED_DIST):
        mask = (dists >= SEED_MIN_DIST) & (dists <= rmax) & (dx >= MIN_X_SEPARATION)
        ids = np.flatnonzero(mask)
        if len(ids):
            p2 = p[ids[int(rng.integers(0, len(ids)))]]
            u1 = (p1[0] - 50.0) / 50.0
            u2 = (p2[0] - 50.0) / 50.0
            slope = (p2[1] - p1[1]) / (u2 - u1)
            intercept = p1[1] - slope * u1
            return Candidate(
                1,
                np.array([intercept, slope]),
                float(min(p1[0], p2[0])),
                float(max(p1[0], p2[0])),
            )
    return None


def find_curve(p, uncovered, rng, restarts):
    uncovered_idx = np.flatnonzero(uncovered)
    best_q = None
    for _ in range(restarts):
        seed = _seed_candidate(rng, p, uncovered_idx)
        if seed is None:
            continue
        cand, score, esc, new = _hill_climb(seed, p, uncovered, REFINE_STEPS, rng)
        if esc == 0 and new >= MIN_NEW_POINTS and (best_q is None or score > best_q[1]):
            best_q = (cand, score)
    return best_q[0] if best_q is not None else None


def assign_points(p, covered, cand):
    samples = sample_graph(cand, MARK_STEP, VALIDATE_SAMPLE_CAP)
    point_d, _ = _min_dists(p, samples)
    hit = point_d <= TAU
    new = np.flatnonzero(hit & ~covered)
    covered |= hit
    return new


def _reduction_score(cand, assigned):
    samples = sample_graph(cand, SEARCH_STEP, SEARCH_SAMPLE_CAP)
    sel = _box_filter(assigned, samples)
    if len(samples) == 0 or len(sel) == 0:
        return (False, -1, -cand.degree, 10.0**18), False
    curve_d, point_d = _min_dists(samples, assigned[sel])
    escaped = bool((curve_d > TAU).any())
    covered_n = int((point_d <= TAU).sum())
    score = (
        not escaped,
        covered_n,
        -cand.degree,
        -float(curve_d.mean()),
    )
    return score, escaped


def _reduce_degree(cand, assigned_pts, rng):
    target = cand.degree
    cur = cand
    while cur.degree > 0:
        trial = Candidate(cur.degree - 1, cur.coef[:-1].copy(), cur.a, cur.b)
        score, escaped = _reduction_score(trial, assigned_pts)
        for t in range(REDUCE_STEPS):
            sigma_c, sigma_d = _sigmas(t, REDUCE_STEPS)
            mutant = _mutate(trial, rng, sigma_c, sigma_d)
            if mutant is None:
                continue
            mscore, _ = _reduction_score(mutant, assigned_pts)
            if mscore > score:
                trial, score = mutant, mscore
        final_score, final_escaped = _reduction_score(trial, assigned_pts)
        if not final_escaped and final_score[1] == len(assigned_pts):
            cur = trial
        else:
            break
    return cur


def fit_curves(p, rng, max_curves):
    covered = np.zeros(len(p), dtype=bool)
    curves = []
    while len(p) == 0 or covered.sum() < MIN_COVERAGE * len(p):
        if len(curves) >= max_curves:
            break
        cand = find_curve(p, ~covered, rng, RESTARTS_PER_CURVE)
        if cand is None:
            cand = find_curve(p, ~covered, rng, RESCUE_RESTARTS)
        if cand is None:
            break
        assigned = assign_points(p, covered, cand)
        curves.append(_reduce_degree(cand, p[assigned], rng))
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
    cx = (mn[0] + mx[0]) / 2.0

    verts = []
    codes = []
    for poly in polys:
        t = np.empty_like(poly)
        t[:, 0] = (poly[:, 0] - cx) * scale + 50.0
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


def fmt_num(v: float) -> str:
    s = f"{v:.{PRECISION}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def poly_str(coef: np.ndarray) -> str:
    parts = []
    first = True
    for k in range(len(coef) - 1, 0, -1):
        c = coef[k]
        cs = fmt_num(abs(c))
        if cs == "0":
            continue
        body = "x" if k == 1 else f"x^{k}"
        prefix = "" if cs == "1" else cs
        sign = "-" if c < 0 else ("" if first else "+")
        parts.append(sign + prefix + body)
        first = False
    if coef[0] != 0 or not parts:
        sign = "-" if coef[0] < 0 else ("" if first else "+")
        parts.append(sign + fmt_num(abs(coef[0])))
    out = "".join(parts)
    return out if out else "0"


def serialize(cand: Candidate) -> str:
    return (
        f"y={poly_str(cand.coef)}"
        f"\\ \\left\\{{{fmt_num(cand.a)}\\le x\\le {fmt_num(cand.b)}\\right\\}}"
    )


_EXPR_RE = re.compile(
    r"^y=(.+?)\\ \\left\\\{(-?[0-9.]+)\\le x\\le (-?[0-9.]+)\\right\\\}$"
)
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
    try:
        a, b = float(m.group(2)), float(m.group(3))
    except ValueError:
        return None
    return Candidate(len(coef) - 1, coef, a, b)


def _extreme_values(cand: Candidate) -> np.ndarray:
    xs = [cand.a, cand.b]
    deriv_roots = np.polynomial.Polynomial(cand.coef).deriv().roots()
    real = deriv_roots[np.abs(deriv_roots.imag) < 1e-8].real
    xu = 50.0 * real + 50.0
    inner = xu[(xu >= cand.a) & (xu <= cand.b)]
    return _poly_u(cand.coef, np.array(xs + list(inner)))


def validate(lines, p):
    problems = []
    parsed = [parse_line(l) for l in lines]
    bad4 = [
        l
        for l, c in zip(lines, parsed)
        if c is None or serialize(c) != l
    ]
    if bad4:
        problems.append(f"V4 round-trip failed for {len(bad4)} expression(s)")
    good = [c for c in parsed if c is not None]

    dense = [
        sample_graph(c, VALIDATE_STEP, VALIDATE_SAMPLE_CAP) for c in good
    ]
    all_samples = np.vstack(dense) if dense else np.zeros((0, 2))

    if len(good) != len(lines):
        problems.append("V4 unparsable expression(s)")
    elif len(all_samples) == 0:
        problems.append(f"V1 coverage 0.0000 below {MIN_COVERAGE}")
    elif len(p):
        pd, sd = _min_dists(p, all_samples)
        coverage = float((pd <= TAU).mean())
        if coverage < MIN_COVERAGE:
            problems.append(
                f"V1 coverage {coverage:.4f} below {MIN_COVERAGE}"
            )
        max_escape = float(sd.max())
        if max_escape > MARGIN:
            problems.append(
                f"V2 confinement exceeded: {max_escape:.3f} > {MARGIN}"
            )

    for i, c in enumerate(good):
        if not np.all(np.isfinite(c.coef)):
            problems.append(f"V3 non-finite coefficient in expression {i}")
            continue
        if np.abs(c.coef).max() >= 1e9:
            problems.append(f"V3 coefficient magnitude >= 1e9 in expression {i}")
            continue
        if not (DOMAIN_LO <= c.a <= DOMAIN_HI and DOMAIN_LO <= c.b <= DOMAIN_HI):
            problems.append(f"V3 domain endpoint outside [-5,105] in expression {i}")
            continue
        vals = _extreme_values(c)
        if vals.min() < DOMAIN_LO or vals.max() > DOMAIN_HI:
            problems.append(
                f"V3 confinement exceeded in expression {i}: "
                f"[{vals.min():.3f}, {vals.max():.3f}]"
            )
    return problems


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

    p = glyph_boundary(letter)
    curves = fit_curves(p, np.random.default_rng(seed), max_curves)
    lines = [serialize(c) for c in curves]
    problems = validate(lines, p)
    if problems:
        for msg in problems:
            print(msg, file=sys.stderr)
        return 1
    sys.stdout.write("".join(l + "\n" for l in lines))
    return 0
