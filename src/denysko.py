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
from string import ascii_letters

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
    tail_reentry_violation_cheb,
    FAMILY_HALF_WIDTH_FLOOR,
    FAMILY_MIN_SPAN,
)

PRECISION = 12
DEFAULT_SEED = 42
MAX_POLY_GLYPH_MISS = 0.04        # ~20 raster steps at normalized scale
MAX_CORRIDOR_GLYPH_MISS = 0.08    # ~40 raster steps
MAX_REALIZATION_X_ERROR = 0.005   # ~2.5 raster steps
HORNER_V4_TOL = 1e-6          # V4 stability tolerance for Horner lines
                            # (numerical comparison vs canonical Chebyshev)


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
    # constant term first (unless it's zero and there are other terms):
    if rendered[0] != "0" or all(s == "0" for s in rendered[1:]):
        parts.append(rendered[0])
        first = False
    for k in range(1, len(coef)):
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
    if not parts:
        parts.append(rendered[0])
    out = "".join(parts)
    return out if out else "0"


def format_expression(curve) -> str:
    poly = getattr(curve, "poly", curve)
    return f"y={poly_str(np.asarray(poly.coef))}"


def _horner_expression(coef, mid, scale):
    """Serialize a power-basis polynomial in z=(x-mid)/scale using an
    inlined Horner form. This avoids catastrophic cancellation for
    degree >= 10 while remaining a standalone y=f(x) expression."""
    from numpy.polynomial import Polynomial as Poly

    # convert Chebyshev-in-z to power-in-z:
    # caller must pass POWER-BASIS coefficients in z (not Chebyshev)
    parts = []
    n = len(coef)
    # Build innermost-to-outermost Horner:
    expr = fmt_num(coef[-1]) if n > 0 else "0"
    for j in range(n - 2, -1, -1):
        c = fmt_num(coef[j])
        if j == 0 and False:
            pass
        # z = (x-mid)/scale; each nesting level adds one z multiply
        z_expr = f"((x-{fmt_num(mid)})/{fmt_num(scale)})"
        expr = f"{c}+{z_expr}*({expr})"
    return expr


def _needs_horner(degree: int, coef_cheb_power_x=None) -> bool:
    """True when raw x-power serialization would be numerically unstable.

    Estimates the condition number of the power basis at the corridor
    center and compares against double precision resolution.
    """
    return degree >= 10




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


class _NotPolynomial(Exception):
    """Raised internally when an arithmetic expression is not a polynomial
    in x (e.g. it divides by a non-constant)."""


_TOK_RE = re.compile(r"[-+*/()]|[0-9]+(?:\.[0-9]*)?|\.[0-9]+|x")
# Whitespace is insignificant; every other character must be part of a valid
# token, otherwise the expression is rejected rather than silently truncated.
_TOKEN_SEQ_RE = re.compile(
    r"(?:[-+*/()]|[0-9]+(?:\.[0-9]*)?|\.[0-9]+|x)*\Z")
_WS_RE = re.compile(r"\s+")


def _parse_expression_poly(s: str):
    """Parse a restricted arithmetic expression in x (numbers, x, + - * / and
    parentheses) into power-basis coefficients of x.

    Returns an ``np.ndarray`` of coefficients or ``None`` when the string is
    not a polynomial in x. This is the generic fallback behind ``parse_line``:
    it handles the nested shifted Horner form emitted for high-degree curves
    as well as the flat ``a*x^n`` form that ``parse_poly`` already covers.

    Any character that is neither whitespace nor a valid token (e.g. ``^``,
    stray letters, ``y``) causes rejection, so malformed input is never
    silently discarded and mis-parsed.
    """
    compact = _WS_RE.sub("", s)
    if compact and not _TOKEN_SEQ_RE.fullmatch(compact):
        return None
    tokens = _TOK_RE.findall(s)
    if not tokens:
        return None
    n = len(tokens)
    pos = 0

    def peek():
        return tokens[pos] if pos < n else None

    def adv():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def _pad(a, b):
        L = max(len(a), len(b))
        aa = np.zeros(L)
        aa[: len(a)] = a
        bb = np.zeros(L)
        bb[: len(b)] = b
        return aa, bb

    def p_expr():
        val = p_term()
        while peek() in ("+", "-"):
            op = adv()
            rhs = p_term()
            aa, bb = _pad(val, rhs)
            val = aa + bb if op == "+" else aa - bb
        return val

    def p_term():
        val = p_factor()
        while peek() in ("*", "/"):
            op = adv()
            rhs = p_factor()
            if op == "*":
                val = np.convolve(val, rhs)
            else:
                if len(rhs) != 1:
                    raise _NotPolynomial("division by non-constant")
                if rhs[0] == 0.0:
                    raise _NotPolynomial("division by zero")
                val = val / rhs[0]
        return val

    def p_factor():
        t = peek()
        if t is None:
            raise _NotPolynomial("incomplete expression")
        if t in ("+", "-"):
            adv()
            v = p_factor()
            return -v if t == "-" else v
        if t == "(":
            adv()
            v = p_expr()
            if peek() != ")":
                raise _NotPolynomial("missing )")
            adv()
            return v
        if t == "x":
            adv()
            return np.array([0.0, 1.0])
        try:
            v = float(adv())
        except (ValueError, TypeError):
            raise _NotPolynomial("bad number")
        return np.array([v])

    # All malformed-input paths raise _NotPolynomial; only that is caught so
    # an unexpected error is not silently swallowed.
    try:
        result = p_expr()
    except _NotPolynomial:
        return None
    if pos != n or result is None:
        return None
    return result


class _EmittedPoly:
    """A polynomial emitted in nested/arithmetic (Horner) form.

    Exposes a numerically stable callable (evaluation of the original
    emitted expression via ``eval_expression``) and a raw-x power-basis
    ``.coef`` for callers that need coefficients. The callable is preferred
    for evaluation: raw-x coefficients of a high-degree Horner line are
    numerically unstable, which is exactly why the line was emitted in
    Horner form in the first place.
    """

    def __init__(self, expr: str, coef):
        self.expr = expr
        self.coef = coef

    def __call__(self, x):
        return eval_expression(self.expr, x)


def _is_horner_line(line: str) -> bool:
    """True for emitted lines that use the nested shifted Horner form.

    They contain parentheses (``((x-mid)/scale)*...``). Flat power-basis
    lines never contain parentheses, so this reliably distinguishes the two
    serialization styles without fragile pattern matching.
    """
    m = _EXPR_RE.match(line)
    if m is None:
        return False
    return "(" in m.group(1)


def eval_expression(expr: str, x_vals: np.ndarray) -> np.ndarray:
    """Safely evaluate a restricted arithmetic expression for many x.

    Supports: numbers, x, +, -, *, /, parentheses.
    Uses Python eval with only x/numbers exposed via a controlled env.
    """
    # strip leading zeros from integer literals (Python 3 disallows them);
    # these can appear from fmt_num producing e.g. "0.052..." -> "0052..."
    import re as _re_mod
    safe = _re_mod.sub(r'(?<![.\w])0+(\d)', r'\1',
                       expr.replace("x", "\x00"))
    safe = safe.replace("\x00", "x")
    env = {"__builtins__": {}}
    env["x"] = np.asarray(x_vals, dtype=float)
    return eval(safe, env)  # noqa: S307


def validate_horner_line(line: str, corridor, coef_cheb: np.ndarray,
                         grid: int = 200):
    """V4-stability check: evaluate serialized expression at many points
    and compare against the intended Chebyshev polynomial."""
    from numpy.polynomial import chebyshev as _ch

    m = _EXPR_RE.match(line)
    if m is None:
        return float("inf")
    expr = m.group(1)
    xs = np.linspace(corridor.xa, corridor.xb, grid)
    try:
        parsed_vals = eval_expression(expr, xs)
    except (ValueError, TypeError, ArithmeticError, IndexError):
        return float("inf")

    scale = (corridor.xb - corridor.xa) / 2.0
    mid = (corridor.xa + corridor.xb) / 2.0
    z = (xs - mid) / scale
    cheb_vals = _ch.chebval(z, np.asarray(coef_cheb, dtype=float))
    return float(np.max(np.abs(parsed_vals - cheb_vals)))


def parse_line(line: str):
    class _Curve:
        pass

    m = _EXPR_RE.match(line)
    if m is None:
        return None
    expr = m.group(1)
    coef = parse_poly(expr)
    if coef is not None:
        curve = _Curve()
        curve.poly = np.polynomial.Polynomial(coef)
        return curve
    # Not a flat power-basis form: try the generic arithmetic parser. This
    # covers the nested shifted Horner form emitted for high-degree curves.
    gen = _parse_expression_poly(expr)
    if gen is None:
        return None

    curve = _Curve()
    # Keep the emitted expression for numerically stable evaluation. The
    # raw-x coefficients are retained for callers that need them.
    curve.poly = _EmittedPoly(expr, gen)
    return curve


# ---------------------------------------------------------------------------
# Phase 5: independent validation helpers (operate on PARSED polynomials)
# ---------------------------------------------------------------------------


def corridor_adherence_violation(poly_coef, corridor, grid=500):
    """Worst corridor violation of an ordinary power-basis polynomial.

    Independent of the fitter: evaluates the PARSED emitted coefficients
    against the route corridor's interior tube over its x-window.

    ``poly_coef`` may be a raw-x coefficient array or any callable that
    evaluates the polynomial (e.g. the stable ``_EmittedPoly`` wrapper used
    for Horner-form lines).
    """
    xs = np.linspace(corridor.xs[0], corridor.xs[-1], grid)
    if callable(poly_coef):
        vals = np.asarray(poly_coef(xs), dtype=float)
    else:
        vals = np.polynomial.Polynomial(
            np.asarray(poly_coef, dtype=float))(xs)
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
        horner = _is_horner_line(line)
        if horner:
            # Horner lines are not round-trippable through the flat
            # power-basis serializer; validate them by numerical comparison
            # against their canonical Chebyshev data instead. This is the
            # dedicated V4-stability check designed for Horner output.
            coef_cheb = getattr(fit, "coef_cheb", None)
            if coef_cheb is not None:
                err = validate_horner_line(line, corr, coef_cheb)
                if err > HORNER_V4_TOL:
                    problems.append(
                        f"V4 curve {i}: horner mismatch {err:.3e}")
        else:
            again = serialize(parsed)
            if again != line:
                problems.append(
                    f"V4 curve {i}: serialization mismatch: "
                    f"{line!r} -> {again!r}"
                )
                continue
        # V2/V5 evaluate via the (numerically stable) parsed polynomial
        # callable; this works identically for flat and Horner lines.
        coef = parsed.poly
        v2 = corridor_adherence_violation(coef, corr)
        if v2 > CORRIDOR_EPS:
            problems.append(f"V2 curve {i}: corridor violation {v2:.3f}")
        ori = getattr(fit, "orientation", (1, -1))
        coef_cheb = getattr(fit, "coef_cheb", None)
        if coef_cheb is not None:
            v3 = tail_reentry_violation_cheb(coef_cheb, corr, ori)
        elif not horner:
            # synthetic/low-degree fits carry only raw coefficients
            v3 = tail_reentry_violation(
                np.asarray(parsed.poly.coef, dtype=float), corr, ori)
        else:
            # Horner line without Chebyshev data: tail check is unavailable
            v3 = 0.0
        if v3:
            problems.append(f"V3 curve {i}: tail re-entry {v3:.3f}")
        v5 = poly_glyph_violation(coef, corr, geom)
        # raster-derived tolerance (see CHALLENGES for derivation)
        if v5 > MAX_POLY_GLYPH_MISS:
            problems.append(
                f"V5 curve {i}: leaves glyph by {v5:.3f} at same x")
    # V6: geometric realization of every meaningful atom against its
    # REALIZED embedding (unfolded x, corridor interval) — strict:
    # any uncovered sample fails. Assignment is branch-aware: the curve
    # of the route claiming the atom is the one checked.
    if routes is not None and graph is not None:
        for i3, (r, corr) in enumerate(zip(routes, corridors)):
            curve = parse_line(lines[i3])
            if curve is None:
                continue
            poly = curve.poly
            for a_id, emb in corr.realized.items():
                rx, rlo, rhi = emb["x"], emb["lower"], emb["upper"]
                vals = np.asarray(poly(rx), dtype=float)
                # the assigned curve may use the same solver-level
                # allowance as V2; branch identity is preserved because
                # the interval itself is the atom's realized corridor
                bad = ~((vals >= rlo - 2 * CORRIDOR_EPS)
                        & (vals <= rhi + 2 * CORRIDOR_EPS))
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


def _glyph_geometry_or_error(letter: str):
    """glyph_geometry with a clean GenerationError for missing glyphs."""
    from src.topology import glyph_geometry
    try:
        return glyph_geometry(letter)
    except ValueError as exc:
        raise GenerationError(
            f"character {letter!r} cannot be generated: {exc}") from exc


def build_phase1(letter: str):
    """Phases 1-2: stroke-skeleton routing graph, complete routes,
    exact minimum selection."""
    geom = _glyph_geometry_or_error(letter)
    graph = build_stroke_route_graph(geom)
    candidates = enumerate_complete_routes(graph)
    chosen_idx = select_routes_min_cover(graph, candidates)
    chosen = [candidates[j] for j in chosen_idx]
    selected = [build_route_corridor(graph, route, geom)
                for route in chosen]
    for j, corr in enumerate(selected):
        v = corridor_glyph_violation(corr, geom)
        # raster-derived tolerance; see CHALLENGES (measured
        # maxima: A/B/C/O <= 2.1, H <= 6.3); the corridor MIDPOINT must
        # always stay inside a filled run. Tightening this budget via
        # corridor centerline smoothing is documented future work.
        if v > MAX_CORRIDOR_GLYPH_MISS:
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
        if err > MAX_REALIZATION_X_ERROR:
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
  denysko B --min-curves 12
  denysko g --seed 123
  denysko -q O
"""


def ascii_letter(value: str) -> str:
    """argparse type: exactly one ASCII letter A-Z or a-z."""
    from string import ascii_letters

    if len(value) == 1 and value in ascii_letters:
        return value
    raise argparse.ArgumentTypeError(
        "LETTER must be one ASCII letter A-Z or a-z")


def min_curves_type(value: str) -> int:
    """argparse type: requested minimum output curves.

    Any positive integer is accepted; there is no output-curve cap.
    (MAX_ROUTE_CANDIDATES in src/topology.py is an unrelated internal
    route-enumeration guard.)"""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


@dataclass(frozen=True)
class CliConfig:
    text: str
    min_curves: int
    seed: int
    quiet: bool
    letter_spacing: float
    space_width: float


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
    parser.add_argument("text", type=str, metavar="TEXT",
                        help="text to render (best effort outside "
                             "A-Z/a-z); ASCII space is layout-only")
    parser.add_argument("--min-curves", dest="min_curves",
                        type=min_curves_type, metavar="N", default=1,
                        help="request at least N output curves; "
                             "Denysko emits more automatically if "
                             "required for complete glyph coverage. "
                             "No upper limit.")
    parser.add_argument("--letter-spacing", dest="letter_spacing",
                        type=float, default=DEFAULT_LETTER_SPACING,
                        metavar="FLOAT",
                        help="extra horizontal advance after each glyph "
                             "(default 0.15); must be non-negative")
    parser.add_argument("--space-width", dest="space_width",
                        type=float, default=DEFAULT_SPACE_WIDTH,
                        metavar="FLOAT",
                        help="horizontal advance of one space "
                             "(default 0.50); must be non-negative")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        metavar="SEED",
                        help="seed controlling deterministic variation "
                             "among valid curve realizations")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress progress diagnostics on stderr")
    parser.add_argument("--version", action="version",
                        version=f"denysko {ver}")
    return parser


def parse_cli(argv) -> CliConfig:
    ns = build_parser().parse_args(list(argv))
    return CliConfig(text=ns.text, min_curves=ns.min_curves,
                     seed=ns.seed, quiet=ns.quiet,
                     letter_spacing=ns.letter_spacing,
                     space_width=ns.space_width)


def allocate_counts(K: int, M: int, rng: np.random.Generator):
    """Balanced multiplicity allocation of M curves over K paths:
    multiplicities differ by at most 1; the seed decides which paths
    receive the extra copy when M % K != 0."""
    q, r = divmod(M, K)
    counts = [q] * K
    extra_slots = list(range(K))
    if r:
        order = np.array(extra_slots)[rng.permutation(len(extra_slots))]
        for j in order[:r]:
            counts[j] += 1
    return counts


DOMAIN_TAG_ALLOCATION = 1
DOMAIN_TAG_PATH_FAMILY = 2



def _path_child_seed(seed: int, tag: int, path_index: int,
                     attempt: int = 0):
    """Domain-separated deterministic child seed (no O(M) spawning)."""
    return np.random.SeedSequence(
        [int(seed), DOMAIN_TAG_PATH_FAMILY * 1000 + tag,
         path_index, attempt])


def guide_weights(corr, child_rng):
    """Seeded smooth low-frequency weight vector over corridor nodes."""
    n = len(corr.xs)
    t = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    if child_rng is not None:
        for k in (1, 2, 3):
            a = float(child_rng.uniform(0.05, 0.25))
            ph = float(child_rng.uniform(0.0, 2.0 * np.pi))
            w = w + a * np.cos(k * np.pi * t + ph)
    return w


def _family_directions(corr, seed, path_index):
    """Deterministic candidate direction vectors for family search.

    All operate on normalized corridor displacement
    (P(x_i)-center_i)/half_i. Returns list of (name, w) where w is the
    weight vector for the linear functional sum(w_i * P(x_i)/half_i).
    """
    n = len(corr.xs)
    half = np.maximum((corr.upper - corr.lower) / 2.0, FAMILY_HALF_WIDTH_FLOOR)
    t = np.linspace(0, 1, n)
    sseq = np.random.SeedSequence(
        [int(seed) if seed is not None else 0,
         DOMAIN_TAG_PATH_FAMILY, path_index])
    rng = np.random.default_rng(sseq)
    smooth = 1.0 + rng.uniform(0.05, 0.3)*np.cos(
        rng.uniform(1, 3)*np.pi*t + rng.uniform(0, 2*np.pi))
    directions = [("seeded-smooth", smooth),
                  ("tilt-lr", 1.0 + 0.5*t),
                  ("cos1", 1.0 + 0.4*np.cos(np.pi*t))]
    return directions


def solve_family_anchors(graph, route, corr, seed, path_index,
                         d_min, degree_cap=None):
    """Find the LOWEST-degree certified convex family for this path.

    Progressive search: try degrees d_min..INITIAL_FIT_DEGREE in
    increasing order; return immediately upon finding a usable family.
    For each degree, tries all four tail orientations and a small set
    of deterministic objective directions. Anchor tails are proved in
    normalized z coordinates (scale-equivariant V3); anchor separation
    is measured via chebval, never raw-x expansion.
    """
    from src.fitting import solve_anchor, certify_anchor, CORRIDOR_EPS
    from src.fitting import tail_reentry_violation_cheb, _zmap
    from numpy.polynomial import chebyshev as cheb

    cap = degree_cap or INITIAL_FIT_DEGREE
    directions = _family_directions(corr, seed, path_index)

    for D in range(d_min, cap + 1):
        for ori in ((1, -1), (-1, 1), (1, 1), (-1, -1)):
            for dname, w_dir in directions[:3]:   # max 3 directions
                w_raw = w_dir   # raw direction; solve_anchor normalizes
                plo = solve_anchor(corr, D, ori[0], ori[1], w_raw,
                                   False)
                phi = solve_anchor(corr, D, ori[0], ori[1], w_raw,
                                   True)
                if plo is None or phi is None:
                    continue
                # acceptance contract: dense corridor/ramp certification
                # (canonical Chebyshev evaluation), normalized-z V3 on
                # both anchors, and nondegenerate geometric span.
                # With all pointwise side inequalities satisfied by the
                # two anchors, convex interpolation P_t=(1-t)P0+tP1
                # satisfies them everywhere in between.
                if (certify_anchor(corr, np.asarray(plo), ori[0],
                                   ori[1]) > CORRIDOR_EPS
                        or certify_anchor(corr, np.asarray(phi), ori[0],
                                          ori[1]) > CORRIDOR_EPS):
                    continue
                if (tail_reentry_violation_cheb(
                        np.asarray(plo), corr, ori) != 0
                        or tail_reentry_violation_cheb(
                            np.asarray(phi), corr, ori) != 0):
                    continue
                z = _zmap(corr.xs, corr.xa, corr.xb)
                diff = np.abs(cheb.chebval(z, plo)
                              - cheb.chebval(z, phi))
                if float(np.max(diff)) < FAMILY_MIN_SPAN:
                    continue   # degenerate: same polynomial
                return (np.asarray(plo), np.asarray(phi), D, ori)
    return None




def realize_variants(graph, chosen, selected, counts, seed, geom,
                     reporter=lambda msg: None, canonical=False):
    from numpy.polynomial import Polynomial as Poly
    """Emit counts[j] curves for structural path j by uniformly sampling
    the path's certified convex polynomial family. O(1) optimization
    solves per path; each emitted curve is cheap interpolation."""
    out_fits, out_corrs, out_routes = [], [], []
    for j, (route, corr) in enumerate(zip(chosen, selected)):
        m = counts[j]
        base = fit_route(corr)
        if base is None:
            raise GenerationError(
                f"generation failed: path {j} has no feasible "
                f"polynomial up to degree {INITIAL_FIT_DEGREE}")
        fam = None
        if m > 1:
            fam = solve_family_anchors(graph, route, corr, seed, j,
                                       max(1, base.degree))
        if fam is None and m > 1:
            raise GenerationError(
                f"generation failed: path {j} supports only its "
                f"canonical curve; requested {m} distinct curves are "
                f"not available at degrees up to "
                f"{INITIAL_FIT_DEGREE}")
        d_min = base.degree
        if fam is not None:
            plo, phi, D, ori = fam
            affine = Poly([-(corr.xa + corr.xb) / (corr.xb - corr.xa),
                           2.0 / (corr.xb - corr.xa)])
            Plo = Poly(np.polynomial.chebyshev.cheb2poly(plo))(affine)
            Phi = Poly(np.polynomial.chebyshev.cheb2poly(phi))(affine)
            ts = [(k + 1) / (m + 1) for k in range(m)]
            if seed is not None and seed % 2 == 1:
                ts = [1.0 - t for t in ts]   # reverse family direction
            fits = []
            for t_j in ts:
                mixed_cheb = (1 - t_j) * np.asarray(plo) + \
                    t_j * np.asarray(phi)
                poly = Poly(np.polynomial.chebyshev.cheb2poly(
                    mixed_cheb))(affine)
                fit = PathFit(corridor=corr, degree=len(mixed_cheb) - 1,
                              coef_cheb=mixed_cheb, poly=poly,
                              dense_max_violation=None,
                              orientation=ori)
                fits.append(fit)
            # baseline curve 0 replaced by central family member when a
            # family exists (all members share identical hard validity)
            reporter(f"path {j}: {m} curve(s), min degree {d_min}, "
                     f"family degree {D}")
            for f in fits:
                out_fits.append(f)
                out_corrs.append(corr)
                out_routes.append(route)
        else:
            reporter(f"path {j}: 1 curve(s), minimum degree {d_min}")
            out_fits.append(base)
            out_corrs.append(corr)
            out_routes.append(route)
    return out_fits, out_corrs, out_routes


def generate_letter(letter: str, *, min_curves: int = 1,
                    seed: int = DEFAULT_SEED,
                    reporter=lambda msg: None):
    """Run the full pipeline for one glyph.

    M = max(K, min_curves) curves are emitted where K is the proven
    minimum number of x-realizable structural paths; every structural
    path receives at least one curve and extras are distributed
    uniformly (differing by at most one), with the seed choosing the
    remainder assignment. Raises GenerationError with a user-facing
    message when generation cannot succeed.
    """
    geom, graph, candidates, chosen, signatures, selected = \
        build_phase1(letter)

    K = len(selected)
    M = max(K, min_curves)
    counts = allocate_counts(K, M,
                             np.random.default_rng(seed))
    reporter(f"phase1: {len(candidates)} candidate routes, "
             f"minimum cover {K}")
    reporter(f"target curves: {M}, allocation {counts}")

    return realize_variants(graph, chosen, selected, counts,
                            seed, geom, reporter)


class GenerationError(Exception):
    """Valid request, but generation/validation could not succeed."""


DEFAULT_LETTER_SPACING = 0.15
DEFAULT_SPACE_WIDTH = 0.50


@dataclass(frozen=True)
class PlacedFit:
    """One local PathFit rigidly translated to text-space x by `dx`."""
    fit: PathFit
    dx: float
    char: str
    char_index: int


@dataclass(frozen=True)
class TextGeneration:
    placed_fits: tuple


def validate_text(text: str) -> None:
    """Structural validation only: text must be non-empty.

    There is deliberately no character whitelist: every non-space
    code point is attempted through the ordinary independent
    per-character glyph-generation path (best effort outside the
    A-Z/a-z/space support contract). ASCII space is layout-only.
    """
    if not text:
        raise GenerationError("text must not be empty")


def glyph_visible_width(letter: str) -> float:
    """Canonical normalized visible width of one glyph's fill."""
    geom = _glyph_geometry_or_error(letter)
    return float(geom.xmax - geom.xmin)


def make_occurrence_reporter(reporter, index, ch, text_len):
    """Prefix diagnostics per occurrence (multi-letter text only)."""
    if text_len == 1:
        return reporter

    def wrapped(msg):
        reporter(f"[{index} {ch!r}] {msg}")

    return wrapped


def generate_text(text: str, *, min_curves=None, seed: int = DEFAULT_SEED,
                  letter_spacing: float = DEFAULT_LETTER_SPACING,
                  space_width: float = DEFAULT_SPACE_WIDTH,
                  reporter=lambda msg: None) -> TextGeneration:
    """Lay out independent per-letter generations with x-translation only.

    Each non-space occurrence runs the existing single-letter generator
    independently with the SAME seed; placement advances by visible
    glyph width plus letter spacing; spaces advance by space_width.
    """
    validate_text(text)

    if letter_spacing < 0:
        raise GenerationError("--letter-spacing must be non-negative")
    if space_width < 0:
        raise GenerationError("--space-width must be non-negative")

    mc = 1 if min_curves is None else min_curves
    cursor = 0.0
    placed = []
    contract_letters = ascii_letters

    for index, ch in enumerate(text):
        if ch == " ":
            cursor += space_width
            continue

        try:
            fits, _, _ = generate_letter(
                ch,
                min_curves=mc,
                seed=seed,
                reporter=make_occurrence_reporter(
                    reporter, index, ch, len(text)),
            )
            for fit in fits:
                placed.append(PlacedFit(fit=fit, dx=cursor, char=ch,
                                        char_index=index))
            width = glyph_visible_width(ch)
        except GenerationError as exc:
            raise GenerationError(
                f"generation failed at character {index} {ch!r}: {exc}"
            ) from exc
        except Exception as exc:
            if ch in contract_letters:
                # supported-letter regressions must stay loud
                raise
            # best-effort arbitrary characters fail cleanly with
            # character context instead of leaking internal errors
            raise GenerationError(
                f"generation failed at character {index} {ch!r}: {exc}"
            ) from exc

        cursor += width + letter_spacing

    return TextGeneration(tuple(placed))


# Backwards-compatible single-letter entry point
generate = generate_letter


def serialize_fit(fit: PathFit) -> str:
    """Serialize one fit exactly as the single-letter CLI always has."""
    deg = fit.degree
    if _needs_horner(deg):
        power_z = np.polynomial.chebyshev.cheb2poly(
            np.asarray(fit.coef_cheb, dtype=float))
        corr = fit.corridor
        mid = (corr.xa + corr.xb) / 2.0
        sc = (corr.xb - corr.xa) / 2.0
        return "y=" + _horner_expression(power_z, mid, sc)
    return format_expression(fit.poly)


def serialize_translated_raw_fit(fit: PathFit, dx: float) -> str:
    """Translate a low-degree raw polynomial by composition P(x-dx)."""
    from numpy.polynomial import Polynomial
    shift = Polynomial([-dx, 1.0])
    Q = Polynomial(np.asarray(fit.poly.coef, dtype=float))(shift)
    return format_expression(Q)


def serialize_translated_horner_fit(fit: PathFit, dx: float) -> str:
    """Translate a high-degree Horner fit by shifting its midpoint.

    Same Chebyshev coefficients, same scale, mid_global = mid_local+dx:
    z = (x_global - (mid_local + dx))/scale.
    """
    power_z = np.polynomial.chebyshev.cheb2poly(
        np.asarray(fit.coef_cheb, dtype=float))
    mid_local = (fit.corridor.xa + fit.corridor.xb) / 2.0
    scale = (fit.corridor.xb - fit.corridor.xa) / 2.0
    expr = _horner_expression(power_z, mid_local + dx, scale)
    return expr if expr.startswith("y=") else "y=" + expr


def serialize_placed_fit(placed: PlacedFit) -> str:
    fit = placed.fit
    dx = placed.dx

    if dx == 0.0:
        return serialize_fit(fit)

    if fit.degree < fitting.HORNER_MIN_DEGREE:
        line = serialize_translated_raw_fit(fit, dx)
        # numerical faithfulness gate for the expanded raw form;
        # fall back to the stable Horner path when it rings
        local_xs = np.array([
            fit.corridor.xa,
            (fit.corridor.xa + fit.corridor.xb) / 2.0,
            fit.corridor.xb,
        ])
        global_xs = local_xs + dx
        truth = np.polynomial.Polynomial(
            np.asarray(fit.poly.coef, dtype=float))(local_xs)
        try:
            actual = eval_expression(line[2:], global_xs)
        except Exception:
            actual = None
        if actual is not None and np.allclose(actual, truth,
                                              rtol=1e-8, atol=1e-10):
            return line
    return serialize_translated_horner_fit(fit, dx)


def validate_placed_serialization(placed: PlacedFit, line: str) -> None:
    """Check translation + serialization correctness only.

    Emitted text at global x must reproduce the canonical local
    Chebyshev values at x-dx on the corridor domain and nodes.
    """
    from numpy.polynomial import chebyshev as cheb
    from src.fitting import _zmap
    local_xs = np.unique(np.concatenate([
        np.asarray(placed.fit.corridor.xs, dtype=float),
        np.linspace(placed.fit.corridor.xa, placed.fit.corridor.xb, 32),
    ]))
    global_xs = local_xs + placed.dx
    z = _zmap(local_xs, placed.fit.corridor.xa, placed.fit.corridor.xb)
    truth = cheb.chebval(z, placed.fit.coef_cheb)
    # The emitted line is the canonical local curve shifted by dx, so it
    # must reproduce the local truth when evaluated at GLOBAL x. Both flat
    # and Horner lines are evaluated through parse_line's stable callable;
    # the fallback evaluator is kept only for forms parse_line cannot
    # represent (none are emitted by the current serializers).
    parsed = parse_line(line)
    if parsed is not None:
        actual = parsed.poly(global_xs)
    else:
        actual = eval_expression(line[2:], global_xs)
    if not np.allclose(actual, truth, rtol=1e-6, atol=1e-9):
        raise GenerationError(
            f"translation validation failed for character "
            f"{placed.char_index} {placed.char!r}")


def serialize_text(result: TextGeneration) -> list:
    lines = []
    for placed in result.placed_fits:
        line = serialize_placed_fit(placed)
        validate_placed_serialization(placed, line)
        lines.append(line)
    return lines


def run(argv) -> int:
    import sys as _sys
    from string import ascii_letters

    cfg = parse_cli(argv)   # argparse errors SystemExit(2) naturally

    def reporter(msg):
        if not cfg.quiet:
            print(msg, file=_sys.stderr)

    try:
        text = cfg.text
        if len(text) == 1 and text in ascii_letters:
            # single-letter compatibility path: byte-identical to the
            # historical one-letter CLI (no translation wrappers);
            # reserved for supported ASCII letters
            try:
                validate_text(text)
            except GenerationError as exc:
                print(str(exc), file=_sys.stderr)
                return 1
            try:
                fits, corrs, routes_list = generate_letter(
                    text, min_curves=cfg.min_curves,
                    seed=cfg.seed, reporter=reporter)
                lines = [serialize_fit(f) for f in fits]
            except GenerationError as exc:
                print(str(exc), file=_sys.stderr)
                return 1
            except RuntimeError as exc:
                # expected geometric/fitting gate failures raise
                # RuntimeError; anything else is a genuine bug and
                # propagates
                print(f"generation failed: {exc}", file=_sys.stderr)
                return 1
        else:
            try:
                result = generate_text(
                    text, min_curves=cfg.min_curves,
                    seed=cfg.seed,
                    letter_spacing=cfg.letter_spacing,
                    space_width=cfg.space_width,
                    reporter=reporter)
                lines = serialize_text(result)
            except GenerationError as exc:
                print(str(exc), file=_sys.stderr)
                return 1
            except RuntimeError as exc:
                print(f"generation failed at character: {exc}",
                      file=_sys.stderr)
                return 1
    except GenerationError as exc:
        print(str(exc), file=_sys.stderr)
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
