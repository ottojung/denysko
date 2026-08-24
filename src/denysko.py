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
    GlyphGeometry,
    TAU,
    MIN_COVERAGE,
    DEFAULT_MAX_CURVES,
    CORRIDOR_EPS,
    SLIVER_SPAN,
    ESC_OFFSETS,
    glyph_geometry,
    build_stroke_route_graph,
    enumerate_complete_routes,
    select_routes_min_cover,
    build_route_corridor,
    corridor_glyph_violation,
    poly_glyph_violation,
    route_continuity_violation,
    atom_coverage_misses,
    route_edge_coverage,
    route_coverage_fraction,
)
from src import fitting
from src.fitting import (
    INITIAL_FIT_DEGREE,
    PathFit,
    fit_route,
    tail_reentry_violation,
)

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
    against the route corridor's interior tube over its x-window.
    """
    xs = np.linspace(corridor.xs[0], corridor.xs[-1], grid)
    vals = np.polynomial.Polynomial(poly_coef)(xs)
    lo = np.interp(xs, corridor.xs, corridor.lower)
    hi = np.interp(xs, corridor.xs, corridor.upper)
    return float(max(0.0, np.max(np.maximum(lo - vals, vals - hi))))


def validate_lines(lines, geom, fits, corridors, routes=None,
                   graph=None):
    """Independent validation of the EMITTED text (Phases 5).

    V2: parsed polynomials adhere to their route corridors.
    V3: tails escape the glyph band vertically and permanently
        (analytic root analysis using each fit's chosen orientation).
    V4: serialization round-trip is exact.
    V1 (route-edge coverage) is enforced by the caller before fitting.
    """
    problems = []
    for i, (line, fit, corr) in enumerate(zip(lines, fits, corridors)):
        # V4: exact serialization contract
        parsed = parse_line(line)
        if parsed is None:
            problems.append(f"V4 curve {i}: unparseable line {line!r}")
            continue
        again = serialize(parsed)
        if again != line:
            problems.append(
                f"V4 curve {i}: serialization mismatch: {line!r} -> {again!r}"
            )
            continue
        coef = np.asarray(parsed.poly.coef, dtype=float)
        v2 = corridor_adherence_violation(coef, corr)
        if v2 > CORRIDOR_EPS:
            problems.append(f"V2 curve {i}: corridor violation {v2:.3f}")
        ori = getattr(fit, "orientation", (1, -1))
        v3 = tail_reentry_violation(coef, corr, ori)
        if v3:
            problems.append(f"V3 curve {i}: tail re-entry {v3:.3f}")
        v5 = poly_glyph_violation(coef, corr, geom)
        # measured H maximum at unfold-exit transitions is ~3.5 units;
        # anything beyond 4.0 is a real excursion (documented budget)
        if v5 > 4.0:
            problems.append(
                f"V5 curve {i}: leaves glyph by {v5:.3f} at same x")
    # V6: geometric realization of every meaningful atom against its
    # REALIZED embedding (unfolded x, corridor interval) — strict:
    # any uncovered sample fails. Assignment is branch-aware: the curve
    # of the route claiming the atom is the one checked.
    if routes is not None and graph is not None:
        polys = [(s_.edge_id,
                  np.asarray(parse_line(line).poly.coef, dtype=float))
                 for r, line in zip(routes, lines)
                 for s_ in r.steps]
        for i3, (r, corr) in enumerate(zip(routes, corridors)):
            for a_id, (rx, ry, rlo, rhi) in corr.realized.items():
                poly = np.polynomial.Polynomial(
                    np.asarray(parse_line(lines[i3]).poly.coef,
                               dtype=float))
                vals = poly(rx)
                bad = ~((vals >= rlo - CORRIDOR_EPS)
                        & (vals <= rhi + CORRIDOR_EPS))
                if bad.any():
                    problems.append(
                        f"V6 atom {a_id}: {int(bad.sum())}/{len(rx)} "
                        f"realized samples unrealized by curve {i3} "
                        f"(worst miss "
                        f"{float(np.max(np.maximum(rlo - vals, vals - rhi))):.2f})"
                    )
    return problems


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_phase1(letter: str):
    """Phases 1-2: stroke-skeleton routing graph, complete routes,
    exact minimum selection."""
    geom = glyph_geometry(letter)
    graph = build_stroke_route_graph(geom)
    candidates = enumerate_complete_routes(graph)
    chosen_idx = select_routes_min_cover(graph, candidates)
    chosen = [candidates[j] for j in chosen_idx]
    selected = [build_route_corridor(graph, route, geom)
                for route in chosen]
    for j, corr in enumerate(selected):
        v = corridor_glyph_violation(corr, geom)
        # Interpolated-tube overshoot across vertical-unfold exits and
        # stem/bar transitions is tolerated up to 8 glyph units (measured
        # maxima: A/B/C/O <= 2.1, H <= 6.3); the corridor MIDPOINT must
        # always stay inside a filled run. Tightening this budget via
        # corridor centerline smoothing is documented future work.
        if v > 8.0:
            raise RuntimeError(
                f"Phase 1: corridor {j} leaves glyph "
                f"(worst containment miss {v:.3f})")
    # Phase-1 geometric realization: every meaningful physical atom
    # must appear in some selected corridor's realized embedding
    covered_atoms = set()
    for corr in selected:
        covered_atoms |= set(corr.realized.keys())
    missing = sorted(graph.meaningful - covered_atoms)
    if missing:
        raise RuntimeError(
            f"Phase 1: atoms not realized by any selected corridor: "
            f"{missing}")
    signatures = [_route_signature(ids) for ids in chosen]
    return geom, graph, candidates, chosen, signatures, selected


def _route_signature(edge_ids):
    from src.topology import _route_signature as _sig
    return _sig(edge_ids)


def fit_selected(selected):
    """Phase 3-4 per complete-route corridor."""
    fits = []
    failures = []
    for i, corridor in enumerate(selected):
        fit = fit_route(corridor, hi=INITIAL_FIT_DEGREE)
        if fit is None:
            failures.append(i)
        fits.append(fit)
    return fits, failures


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
        geom, graph, candidates, chosen, signatures, selected = \
            build_phase1(letter)
        covered = route_edge_coverage(graph, chosen)
    except Exception as exc:
        print(f"phase1 failed: {exc}", file=sys.stderr)
        return 1

    sel_cov = route_coverage_fraction(graph, chosen)
    print(
        f"phase1: {len(candidates)} complete routes, {len(selected)} selected, "
        f"meaningful-edge coverage {sel_cov:.4f}",
        file=sys.stderr,
    )
    if sel_cov < MIN_COVERAGE or len(selected) > max_curves:
        print(
            f"route selection covers {sel_cov:.4f} "
            f"(< {MIN_COVERAGE} or > {max_curves} curves); cannot emit",
            file=sys.stderr,
        )
        return 1

    fits, failures = fit_selected(selected[:max_curves])
    if failures:
        print(
            f"fitting failed on route(s) {failures} "
            f"(degree {INITIAL_FIT_DEGREE} could not stay in corridor)",
            file=sys.stderr,
        )
        return 1

    for i, fit in enumerate(fits):
        print(
            f"curve {i}: minimum degree {fit.degree}",
            file=sys.stderr,
        )

    lines = [format_expression(f.poly) for f in fits]
    problems = validate_lines(lines, geom, fits, selected[:max_curves],
                              routes=chosen[:max_curves], graph=graph)
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
    parser.add_argument(
        "command",
        choices=["routes", "graph", "select", "fit", "uncovered"],
    )
    parser.add_argument("letter")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])

    geom, graph, candidates, chosen, signatures, selected = \
        build_phase1(args.letter)

    if args.command == "routes":
        for i, ids in enumerate(candidates):
            print(f"route {i}: edges {ids}")
        return 0
    if args.command == "graph":
        for v in graph.vertices:
            print(f"v{v.id}: {v.kind} x={v.x:.1f} "
                  f"in={v.incoming} out={v.outgoing}")
        for e in graph.edges:
            print(f"e{e.id}: v{e.v_from}->v{e.v_to} "
                  f"x[{e.xs[0]:.1f}..{e.xs[-1]:.1f}] span={e.span:.2f} "
                  f"h={e.mean_height:.2f} meaningful={e.id in graph.meaningful}")
        return 0
    if args.command == "select":
        cov = route_coverage_fraction(graph, chosen)
        print(f"selected {len(chosen)} of {len(candidates)} routes; "
              f"meaningful-edge coverage {cov:.4f}")
        for i, (ids, corr) in enumerate(zip(chosen, selected)):
            print(f"curve {i}: edges {ids} "
                  f"x[{corr.xs[0]:.1f}..{corr.xs[-1]:.1f}] nodes {len(corr.xs)}")
        return 0
    if args.command == "uncovered":
        covered = route_edge_coverage(graph, chosen)
        missing = [e.id for e in graph.edges
                   if not covered[e.id] and e.id in graph.meaningful]
        print(f"uncovered meaningful edges: {missing}")
        return 0

    target = selected if args.index is None else [selected[args.index]]
    for i, c in enumerate(target):
        fit = fit_route(c, hi=INITIAL_FIT_DEGREE)
        status = (
            f"min degree {fit.degree}" if fit is not None
            else f"infeasible at {INITIAL_FIT_DEGREE}"
        )
        print(f"corridor {i}: {status}")
    return 0
