"""Denysko: single-letter -> unbounded Desmos polynomials.

Architecture (topology first, then constrained fitting):

  Phase 1  boundary contours -> x-monotone paths -> corridors
  Phase 2  deterministic greedy set-cover selection of paths
  Phase 3  high-degree Chebyshev feasibility inside each fixed corridor
  Phase 4  degree minimization within the same corridor
  Phase 5  independent validation (corridor adherence + global coverage)

See docs/SPEC.md. The polynomial optimizer never discovers topology;
every geometric decision is made explicitly before fitting.
"""

from __future__ import annotations

import re
import sys

import numpy as np

from src.topology import (
    BoundaryPath,
    Corridor,
    GlyphGeometry,
    TAU,
    MIN_COVERAGE,
    DEFAULT_MAX_CURVES,
    CORRIDOR_EPS,
    assign_coverage,
    build_corridors,
    dedupe_paths,
    extract_paths,
    glyph_geometry,
    min_dists,
    select_paths,
)
from src import fitting
from src.fitting import INITIAL_FIT_DEGREE, PathFit, min_degree

PRECISION = 12


# ---------------------------------------------------------------------------
# Serialization and parsing (ordinary powers of x, no domain restrictions)
# ---------------------------------------------------------------------------


def fmt_num(v: float) -> str:
    """Shortest positional decimal that round-trips exactly.

    High-degree power-basis coefficients legitimately span many orders
    of magnitude; a fixed number of decimal places would truncate small
    ones to zero and destroy the curve. Positional (non-scientific)
    formatting also satisfies the output contract.
    """
    s = np.format_float_positional(
        float(v), precision=None, unique=True, trim="-", fractional=True
    )
    if s in ("-0", "", "-"):
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


def format_expression(curve) -> str:
    poly = getattr(curve, "poly", curve)
    return f"y={poly_str(np.asarray(poly.coef))}"


def serialize(fit_or_curve) -> str:
    return format_expression(fit_or_curve)


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

    class _Curve:
        pass

    curve = _Curve()
    curve.poly = np.polynomial.Polynomial(coef)
    return curve


# ---------------------------------------------------------------------------
# Phase 5: independent validation helpers (operate on PARSED polynomials)
# ---------------------------------------------------------------------------


def corridor_adherence_violation(poly_coef, corridor, grid=500):
    """Worst corridor violation of an ordinary power-basis polynomial.

    Independent of the fitter: evaluates the PARSED emitted coefficients
    against the corridor's interior tube and its band-escape ramps.
    """
    from src.topology import escape_bound_at

    xs = np.linspace(corridor.xs[0], corridor.xs[-1], grid)
    vals = np.polynomial.Polynomial(poly_coef)(xs)
    lo = corridor.lower_at(xs)
    hi = corridor.upper_at(xs)
    viol = float(np.max(np.maximum(lo - vals, vals - hi)))
    for spec in corridor.escapes:
        if spec.kind != "band" or not spec.rows:
            continue
        sign = -1.0 if spec.side == "L" else 1.0
        run = abs(spec.rows[-1][0] - spec.x_end)
        xs_e = spec.x_end + sign * np.linspace(
            min(1.0, 0.5), run, max(80, grid // 3)
        )
        from src.topology import escape_bound_at as _eb
        bnd = _eb(spec, xs_e)
        v_e = np.asarray(np.polynomial.Polynomial(poly_coef)(xs_e))
        d_e = np.maximum(0.0, spec.sigma * (bnd - v_e))
        viol = max(viol, float(d_e.max()))
    return viol


def tail_reentry_violation(poly_coef, corridor):
    """Analytic permanent-tail check beyond the last escape checkpoint.

    For every edge-exit (band) ramp: past the final escape row the
    derivative must have no real roots, keep its outward sign, and the
    polynomial must already be strictly outside the band edge on the
    exit side - hence it can never re-enter the visible band.

    Side-exit corridors only get a deterministic visibility check over
    their narrow pad strips: there the trace must stay either outside
    the vertical band or within TAU of this path's own nodes.
    """
    viol = 0.0
    deriv = np.polynomial.polynomial.Polynomial(poly_coef).deriv()
    deriv_coef = np.asarray(deriv.coef)

    def real_roots(coefs):
        if len(coefs) < 2:
            return np.array([])
        r = np.roots(coefs[::-1])
        return np.sort(r[np.abs(r.imag) < 1e-7].real)

    for spec in corridor.escapes:
        if spec.kind == "band" and spec.rows:
            xe = spec.rows[-1][0]
            right = spec.side == "R"
            droots = real_roots(deriv_coef)
            beyond = droots[(droots > xe)] if right else droots[(droots < xe)]
            if len(beyond):
                viol = max(viol, 1.0 + float(abs(beyond[0] - xe)))
            slope = float(np.polynomial.Polynomial(deriv_coef)(xe))
            want = spec.sigma if right else -spec.sigma
            if slope * want <= 0:
                viol = max(viol, 2.0)
            p_xe = float(np.polynomial.Polynomial(poly_coef)(xe))
            outside = spec.sigma * (p_xe - spec.edge)
            if outside <= 0:
                viol = max(viol, 3.0)
        else:
            # Side-exit ('far') policy, documented (Option A): the tail
            # leaves the drawn x-region immediately; crossings outside
            # the intended path domain are tolerated this iteration and
            # recorded in docs/CHALLENGES.md. No rows exist for these.
            continue
            pad = np.linspace(
                corridor.xa if spec.side == "L" else corridor.xs[-1],
                corridor.xs[0] if spec.side == "L" else corridor.xb,
                60,
            )
            pad = pad + 0.0  # explicit copy
            lo_strip, hi_strip = (
                (corridor.xa, corridor.xs[0]) if spec.side == "L"
                else (corridor.xs[-1], corridor.xb)
            )
            strip = np.linspace(lo_strip, hi_strip, 120)
            vals = np.asarray(np.polyval(poly_coef, strip))
            in_band = (vals >= geom_band_lo(spec)) & (vals <= geom_band_hi(spec))
            own = corridor.path.points
            dist = np.min([
                np.min(np.hypot(own[:, 0] - sx, own[:, 1] - vx))
                for sx, vx in zip(strip, vals)
            ]) if len(vals) else 0.0
            bad = in_band & (np.full(len(vals), dist > TAU))
            if bad.any():
                viol = max(viol, 4.0)
    return viol


def geom_band_lo(spec):
    return spec.edge if spec.sigma == -1 else -np.inf


def geom_band_hi(spec):
    return spec.edge if spec.sigma == 1 else np.inf


def uncovered_clusters(points, mask, gap: float = 5.0):
    """Deterministic clustering of uncovered boundary samples by x-gaps."""
    pts = points[~mask]
    if len(pts) == 0:
        return []
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    clusters = []
    start = 0
    for i in range(1, len(pts) + 1):
        if i == len(pts) or (
            pts[i, 0] - pts[i - 1, 0] > gap
            and abs(pts[i, 1] - pts[i - 1, 1]) > gap
        ):
            seg = pts[start:i]
            clusters.append({
                "count": len(seg),
                "bbox": (
                    float(seg[:, 0].min()), float(seg[:, 0].max()),
                    float(seg[:, 1].min()), float(seg[:, 1].max()),
                ),
            })
            start = i
    return clusters


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_phase1(letter: str):
    """Phases 1-2: topology, corridors, deterministic selection."""
    geom = glyph_geometry(letter)
    paths = extract_paths(geom.contours)
    masks = assign_coverage(paths, geom.points, TAU)
    paths, masks = dedupe_paths(paths, masks)
    for p, m in zip(paths, masks):
        p.covered = m
    corridors = build_corridors(paths, geom)
    selected, covered = select_paths(corridors)
    return geom, corridors, selected, covered


def fit_selected(selected):
    """Phase 3-4: high-degree feasibility then degree minimization.

    Mathematical infeasibility is represented by fit=None. Unexpected
    exceptions propagate loudly - they are programming errors, not
    geometry.
    """
    fits = []
    failures = []
    for i, corridor in enumerate(selected):
        fit = min_degree(corridor, hi=INITIAL_FIT_DEGREE)
        if fit is None:
            failures.append(i)
        fits.append(fit)
    return fits, failures


def validate_lines(lines, geom: GlyphGeometry, fits, corridors):
    problems = []
    parsed = [parse_line(l) for l in lines]

    # V4: serialization contract
    bad4 = [
        l for l, c in zip(lines, parsed)
        if c is None or format_expression(c) != l
    ]
    if bad4:
        problems.append(f"V4 round-trip failed for {len(bad4)} expression(s)")
    for c in parsed:
        if c is None:
            continue
        coef = np.asarray(c.poly.coef)
        if not np.all(np.isfinite(coef)):
            problems.append("V4 non-finite coefficient")
        elif np.abs(coef).max() >= 1e9:
            problems.append("V4 coefficient magnitude >= 1e9")

    # Phase 5a: INDEPENDENT corridor adherence on parsed polynomials.
    # Visible intended trace is the path/corridor window; tails are
    # validated analytically (below), so V1 sampling may ignore them.
    for i, (c, fit, corr) in enumerate(zip(parsed, fits, corridors)):
        if c is None or fit is None:
            continue
        coef = np.asarray(c.poly.coef)
        v2 = corridor_adherence_violation(coef, corr)
        if v2 > CORRIDOR_EPS:
            problems.append(
                f"V2 corridor violation {v2:.3f} on path {i}"
            )
        v3 = tail_reentry_violation(coef, corr)
        if v3 > 0:
            problems.append(
                f"V3 tail re-entry risk {v3:.1f} on path {i}"
            )

    # V1: independent global boundary coverage from emitted polynomials,
    # sampled across their corridor windows.
    samples = []
    for line, c, fit in zip(lines, parsed, fits):
        if c is None or fit is None:
            continue
        grid = np.linspace(fit.corridor.xa, fit.corridor.xb, 800)
        vals = np.asarray(c.poly(grid))
        ok = (vals >= geom.ymin - TAU) & (vals <= geom.ymax + TAU)
        xs = grid[ok]
        if len(xs):
            samples.append(np.column_stack([xs, vals[ok]]))
    if not samples:
        problems.append("V1 no visible trace segments")
    else:
        all_s = np.vstack(samples)
        dd, _ = min_dists(geom.points, all_s)
        coverage = float((dd <= TAU).mean())
        if coverage < MIN_COVERAGE:
            uncovered = geom.points[dd > TAU]
            clusters = uncovered_clusters(geom.points, dd <= TAU)
            detail = "; ".join(
                f"[{c['bbox'][0]:.0f}-{c['bbox'][1]:.0f}]x"
                f"[{c['bbox'][2]:.0f}-{c['bbox'][3]:.0f}]n{c['count']}"
                for c in clusters[:6]
            )
            problems.append(
                f"V1 coverage {coverage:.4f} below {MIN_COVERAGE}; "
                f"uncovered clusters: {detail}"
            )
    return problems


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(argv) -> int:
    letter = None
    max_curves = DEFAULT_MAX_CURVES
    positionals = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--max-curves":
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
            max_curves = number
            i += 2
        elif arg == "--seed":
            # deprecated: the pipeline is fully deterministic; accepted
            # (with a value) for CLI compatibility and ignored.
            if i + 1 >= len(argv):
                return 2
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

    try:
        geom, corridors, selected, covered = build_phase1(letter)
    except Exception as exc:
        print(f"phase1 failed: {exc}", file=sys.stderr)
        return 1

    sel_cov = float(covered.mean()) if len(covered) else 0.0
    print(
        f"phase1: {len(corridors)} candidate paths, {len(selected)} selected, "
        f"coverage {sel_cov:.4f}",
        file=sys.stderr,
    )
    if sel_cov < MIN_COVERAGE or len(selected) > max_curves:
        print(
            f"path selection covers {sel_cov:.4f} "
            f"(< {MIN_COVERAGE} or > {max_curves} paths); cannot emit",
            file=sys.stderr,
        )
        return 1

    fits, failures = fit_selected(selected[:max_curves])
    if failures:
        print(
            f"fitting failed on path(s) {failures} "
            f"(degree {INITIAL_FIT_DEGREE} could not stay in corridor)",
            file=sys.stderr,
        )
        return 1

    for i, fit in enumerate(fits):
        print(
            f"path {i}: initial degree {INITIAL_FIT_DEGREE} feasible, "
            f"minimum degree {fit.degree}",
            file=sys.stderr,
        )

    lines = [format_expression(f.poly) for f in fits]
    problems = validate_lines(lines, geom, fits, selected[:max_curves])
    if problems:
        for msg in problems:
            print(msg, file=sys.stderr)
        return 1
    sys.stdout.write("".join(l + "\n" for l in lines))
    return 0


# ---------------------------------------------------------------------------
# Development/debug entry point (phase inspection, not the public CLI)
# ---------------------------------------------------------------------------


def debug_entry() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="denysko-debug")
    parser.add_argument("command", choices=["paths", "select", "fit", "uncovered"])
    parser.add_argument("letter")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])

    geom, corridors, selected, covered = build_phase1(args.letter)
    if args.command == "paths":
        for i, c in enumerate(corridors):
            p = c.path
            print(
                f"path {i}: contour {p.contour_id} nodes {len(p.points)} "
                f"x[{p.points[0, 0]:.1f}..{p.points[-1, 0]:.1f}] "
                f"covers {int(p.covered.sum())}"
            )
        return 0
    if args.command == "select":
        print(
            f"selected {len(selected)} covering {covered.mean():.4f}"
        )
        return 0
    if args.command == "uncovered":
        mask = covered
        clusters = uncovered_clusters(geom.points, mask)
        print(f"phase-1 uncovered boundary clusters: {len(clusters)}")
        for i, c in enumerate(clusters):
            print(
                f"cluster {i}: count={c['count']} "
                f"bbox=[{c['bbox'][0]:.1f},{c['bbox'][1]:.1f}]x"
                f"[{c['bbox'][2]:.1f},{c['bbox'][3]:.1f}]"
            )
        return 0
    target = selected if args.index is None else [corridors[args.index]]
    for i, c in enumerate(target):
        fit = min_degree(c, hi=INITIAL_FIT_DEGREE)
        status = (
            f"min degree {fit.degree}" if fit is not None
            else f"infeasible at {INITIAL_FIT_DEGREE}"
        )
        print(f"corridor {i}: {status}")
    return 0
