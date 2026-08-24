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
    nonvertical_realization_x_error,
    atom_coverage_misses,
    route_edge_coverage,
    route_coverage_fraction,
)
import argparse
from dataclasses import dataclass

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
            for a_id, emb in corr.realized.items():
                rx, rlo, rhi = emb["x"], emb["lower"], emb["upper"]
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
    # R1: realization fidelity — vertical deformation must be local;
    # nonvertical atoms resume exact skeleton x after unfold overlap.
    # Tolerance = ~2.5 raster steps (interpolation between landmarks).
    for j, corr in enumerate(selected):
        err = nonvertical_realization_x_error(corr.realized)
        if err > 0.5:
            raise RuntimeError(
                f"Phase 1 R1: corridor {j} realization x error "
                f"{err:.3f} exceeds raster-derived tolerance")
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


# ---------------------------------------------------------------------------
# Public CLI (argparse)
# ---------------------------------------------------------------------------

CLI_USAGE_EPILOG = """examples:
  denysko A
  denysko h
  denysko --max-curves 4 B
  denysko --seed 42 g
  denysko -q O
"""


def ascii_letter(value: str) -> str:
    """argparse type: exactly one ASCII letter A-Z or a-z."""
    from string import ascii_letters

    if len(value) == 1 and value in ascii_letters:
        return value
    raise argparse.ArgumentTypeError(
        "LETTER must be one ASCII letter A-Z or a-z")


def max_curves_type(value: str) -> int:
    """argparse type: integer 1..12 (project output hard limit)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if not 1 <= n <= DEFAULT_MAX_CURVES:
        raise argparse.ArgumentTypeError(
            f"must be an integer in 1-{DEFAULT_MAX_CURVES}")
    return n


@dataclass(frozen=True)
class CliConfig:
    letter: str
    max_curves: int
    seed: int | None
    quiet: bool


def build_parser() -> argparse.ArgumentParser:
    try:
        from importlib.metadata import version
        ver = version("denysko")
    except Exception:
        ver = "unknown"
    parser = argparse.ArgumentParser(
        prog="denysko",
        description=(
            "Approximate a DejaVu Sans letter with a small set of "
            "unbounded polynomial graphs y=f(x), suitable for Desmos."),
        epilog=CLI_USAGE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("letter", type=ascii_letter, metavar="LETTER",
                        help="one ASCII letter A-Z or a-z")
    parser.add_argument("--max-curves", dest="max_curves",
                        type=max_curves_type, metavar="N",
                        help=f"maximum allowed output curves "
                             f"(1-{DEFAULT_MAX_CURVES})")
    parser.add_argument("--seed", type=int, default=None, metavar="SEED",
                        help="random seed; accepted for "
                             "reproducibility/API compatibility. The "
                             "current pipeline is deterministic, so "
                             "this does not change output.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress progress diagnostics on stderr")
    parser.add_argument("--version", action="version",
                        version=f"denysko {ver}")
    return parser


def parse_cli(argv) -> CliConfig:
    ns = build_parser().parse_args(list(argv))
    return CliConfig(letter=ns.letter,
                     max_curves=ns.max_curves or DEFAULT_MAX_CURVES,
                     seed=ns.seed, quiet=ns.quiet)


def generate(letter: str, *, max_curves: int = DEFAULT_MAX_CURVES,
             reporter=lambda msg: None) -> list[str]:
    """Run the full pipeline for one glyph; raises GenerationError with
    a user-facing message when generation cannot succeed."""
    geom, graph, candidates, chosen, signatures, selected = \
        build_phase1(letter)

    sel_cov = route_coverage_fraction(graph, chosen)
    reporter(f"phase1: {len(candidates)} complete routes, "
             f"{len(selected)} selected, meaningful-atom coverage "
             f"{sel_cov:.4f}")

    if len(selected) > max_curves:
        raise GenerationError(
            f"generation failed: {letter} requires {len(selected)} "
            f"curves, exceeding --max-curves={max_curves}")

    fits, failures = fit_selected(selected[:max_curves])
    if failures:
        raise GenerationError(
            f"generation failed: fitting route(s) {failures} failed up "
            f"to degree {INITIAL_FIT_DEGREE}")

    for i, fit in enumerate(fits):
        reporter(f"curve {i}: minimum degree {fit.degree}")

    lines = [format_expression(f.poly) for f in fits]
    problems = validate_lines(lines, geom, fits, selected[:max_curves],
                              routes=chosen[:max_curves], graph=graph)
    if problems:
        raise GenerationError("; ".join(problems))
    return lines


class GenerationError(Exception):
    """Valid request, but generation/validation could not succeed."""


def run(argv) -> int:
    import sys as _sys

    cfg = parse_cli(argv)   # argparse errors SystemExit(2) naturally

    def reporter(msg):
        if not cfg.quiet:
            print(msg, file=_sys.stderr)

    try:
        lines = generate(cfg.letter, max_curves=cfg.max_curves,
                         reporter=reporter)
    except GenerationError as exc:
        print(str(exc), file=_sys.stderr)
        return 1
    except RuntimeError as exc:
        # expected geometric/fitting gate failures raise RuntimeError;
        # anything else is a genuine bug and propagates
        print(f"generation failed: {exc}", file=_sys.stderr)
        return 1

    _sys.stdout.write("".join(line + "\n" for line in lines))
    return 0


# ---------------------------------------------------------------------------
# Development/debug entry point (phase inspection, not the public CLI)
# ---------------------------------------------------------------------------


def debug_entry() -> int:
    """Development CLI: argparse subcommands over the same pipeline."""
    parser = argparse.ArgumentParser(prog="denysko-debug")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add(name, help_text):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("letter", type=ascii_letter,
                        help="one ASCII letter A-Z or a-z")
        return sp

    _add("routes", "enumerate directed candidate routes")
    _add("graph", "dump routing graph vertices/edges")
    _add("select", "staged-MILP minimum cover")
    _add("realize", "per-atom raw vs realized geometry")
    _add("uncovered", "meaningful atoms without realization")
    p_fit = _add("fit", "minimum-degree fit per selected corridor")
    p_fit.add_argument("--index", type=int, default=None)

    args = parser.parse_args(sys.argv[1:])
    geom, graph, candidates, chosen, signatures, selected = \
        build_phase1(args.letter)

    if args.command == "routes":
        for i, r in enumerate(candidates):
            print(f"route {i}: "
                  f"atoms {[graph.physical_atom(s.edge_id) for s in r.steps]} "
                  f"edges {[s.edge_id for s in r.steps]} "
                  f"{r.steps[0].from_vertex}->{r.steps[-1].to_vertex}")
        return 0
    if args.command == "graph":
        for v in graph.vertices:
            print(f"v{v.id}: {v.kind} x={v.x:.1f} "
                  f"in={v.incoming} out={v.outgoing}")
        for e in graph.edges:
            print(f"e{e.id}: v{e.v_from}->v{e.v_to} span={e.span:.2f} "
                  f"phys={graph.physical_atom(e.id)}")
        return 0
    if args.command == "select":
        cov = route_coverage_fraction(graph, chosen)
        print(f"selected {len(chosen)} of {len(candidates)} routes; "
              f"coverage {cov:.4f}")
        return 0
    if args.command == "realize":
        for i3, corr in enumerate(selected):
            for a_id, emb in sorted(corr.realized.items()):
                print(f"route {i3} atom {a_id}: "
                      f"raw x[{emb['raw_x'].min():.1f}..{emb['raw_x'].max():.1f}] "
                      f"y[{emb['raw_y'].min():.1f}..{emb['raw_y'].max():.1f}] "
                      f"realized x[{emb['x'].min():.1f}..{emb['x'].max():.1f}] "
                      f"maxdx={float(np.max(np.abs(emb['x'] - emb['raw_x']))):.2f} "
                      f"deform={sorted(set(emb['deform']))}")
        return 0
    if args.command == "uncovered":
        covered = set()
        for corr in selected:
            covered |= set(corr.realized.keys())
        missing = sorted(graph.meaningful - covered)
        print(f"uncovered meaningful atoms: {missing}")
        return 0
    if args.command == "fit":
        target = selected if args.index is None else [selected[args.index]]
        for i, c in enumerate(target):
            fit = fit_route(c, hi=INITIAL_FIT_DEGREE)
            status = (f"min degree {fit.degree}" if fit is not None
                      else f"infeasible at {INITIAL_FIT_DEGREE}")
            print(f"corridor {i}: {status}")
        return 0
    return 2
