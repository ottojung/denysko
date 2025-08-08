#!/usr/bin/env python3
"""
Integration-like test for the Text -> Points -> Polynomials -> Validation pipeline.

Steps:
- take a letter ("B")
- transform it into points (centerline contours)
- fit the curves and generate function descriptions (y = f(x))
- interpret those functions symbolically to generate polynomials
- check how close the interpreted functions are to the target points

This test uses only the project code and the Python standard library for parsing
and error metrics. It is written to work with pytest, but can also be run as a
script.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

# Make the repository root importable so `src.*` can be imported when running tests
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.text_extractor import TextExtractor  # type: ignore  # noqa: E402
from src.text_to_desmos import TextToDesmos  # type: ignore  # noqa: E402


Point = Tuple[float, float]


def _parse_polynomial_from_function_string(func: str) -> Dict[int, float]:
    """Parse a function string like "y = -0.5x^3+2x^2-3.4x+5" into a power->coeff dict.

    Supports:
    - optional leading sign for each term
    - omitted coefficient for x terms (e.g., "-x^3" => -1)
    - explicit multiplication (e.g., "-0.5*x^3")
    - integer powers (x, x^2, x^3, ...)
    - scientific notation coefficients (e.g., 1.2e-3*x^2)
    - optional surrounding parentheses around the polynomial
    - optional domain constraints appended with "{a <= x <= b}"
    - whitespace agnostic

    Returns a dict mapping power -> coefficient. E.g., {3: -0.5, 2: 2.0, 1: -3.4, 0: 5.0}
    """
    s = func.strip()
    if not s.lower().startswith("y ="):
        raise ValueError(f"Not a y = f(x) function: {func}")
    expr = s.split("=", 1)[1].strip().replace(" ", "")

    # Strip any domain constraints and surrounding parentheses
    if "{" in expr:
        expr = expr.split("{", 1)[0]
    # Remove optional wrapping parentheses around the entire expression
    if expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1]
    # Also tolerate stray parentheses
    expr = expr.replace("(", "").replace(")", "")

    # Split into terms while respecting scientific notation exponents (e.g., e-5)
    terms: List[str] = []
    i = 0
    start = 0
    while i < len(expr):
        ch = expr[i]
        if ch in "+-" and i != start:
            # Check if this sign is part of an exponent (e.g., 'e-3')
            prev = expr[i - 1]
            if prev not in ("e", "E"):
                terms.append(expr[start:i])
                start = i
        i += 1
    # last term
    if start < len(expr):
        terms.append(expr[start:])

    power_to_coeff: Dict[int, float] = {}

    for term in terms:
        if not term:
            continue
        if "x" in term:
            # Split into coefficient part and power part
            coeff_part, _, power_part = term.partition("x")
            # Allow explicit multiplication (e.g., '1.2e-3*x^2')
            if coeff_part.endswith("*"):
                coeff_part = coeff_part[:-1]
            # Determine coefficient
            if coeff_part in ("", "+"):
                coeff = 1.0
            elif coeff_part == "-":
                coeff = -1.0
            else:
                coeff = float(coeff_part)
            # Determine power
            if power_part.startswith("^"):
                pow_str = power_part[1:]
                # Trim any stray characters
                j = 0
                while j < len(pow_str) and (pow_str[j].isdigit() or pow_str[j] in "+-"):
                    j += 1
                pow_str = pow_str[:j] if j else pow_str
                power = int(pow_str)
            else:
                power = 1
        else:
            # Constant term
            coeff = float(term)
            power = 0

        # Accumulate (in case of duplicated powers)
        power_to_coeff[power] = power_to_coeff.get(power, 0.0) + coeff

    return power_to_coeff


def _parse_domain(func: str) -> Tuple[float, float] | None:
    """Parse an optional domain constraint of the form '{a <= x <= b}' at the end."""
    s = func.strip()
    if '{' not in s or '}' not in s:
        return None
    try:
        brace = s[s.index('{') + 1 : s.rindex('}')]
        # Expected like 'a <= x <= b'
        parts = brace.replace(' ', '').split('<=x<=')
        if len(parts) == 2:
            a_str, b_str = parts
            return float(a_str), float(b_str)
    except Exception:
        return None
    return None


def _eval_poly(power_to_coeff: Dict[int, float], x: float) -> float:
    return sum((c * (x ** p) for p, c in power_to_coeff.items()))


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": math.nan, "median": math.nan, "p90": math.nan, "p99": math.nan, "max": math.nan}
    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    def pct(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        idx = min(max(int(round(p * (n - 1))), 0), n - 1)
        return sorted_vals[idx]
    return {
        "count": float(n),
        "mean": mean,
        "median": median,
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": sorted_vals[-1],
    }


def test_integration_letter_B(capfd=None):  # capfd is a pytest fixture, optional here
    # 1) Extract centerline points for letter 'B'
    extractor = TextExtractor()
    paths = extractor.text_to_paths("B", font_size=100)
    assert paths, "No paths generated for letter 'B'"

    all_points: List[Point] = []
    contours_per_char: List[List[Point]] = []

    for path in paths:
        contours = extractor.extract_contour_points(path, num_points=500)
        for contour in contours:
            if len(contour) >= 5:
                contours_per_char.append(contour)
                all_points.extend(contour)

    assert len(all_points) >= 50, f"Insufficient points extracted: {len(all_points)}"

    # 2) Generate function descriptions using the full pipeline
    converter = TextToDesmos(origin=(0, 0), scale=1.0, max_degree=6)
    functions = converter.text_to_desmos_functions("B", font_size=100, points_per_char=500)

    y_functions = [f for f in functions if f.strip().lower().startswith("y =")]
    assert y_functions, "No y = f(x) functions generated"

    # 3) Interpret function strings into polynomials
    polynomials: List[Dict[int, float]] = []
    domains: List[Tuple[float, float] | None] = []
    for fstr in y_functions:
        polynomials.append(_parse_polynomial_from_function_string(fstr))
        domains.append(_parse_domain(fstr))

    # 4) Assign points to the best-matching polynomial within domain and compute errors
    errors: List[float] = []
    assignments: List[int] = [0] * len(polynomials)

    if not all_points:
        raise AssertionError("No points to validate against")

    ys = [p[1] for p in all_points]
    y_min, y_max = min(ys), max(ys)
    y_range = max(y_max - y_min, 1e-6)

    for (x, y_true) in all_points:
        best_err = float("inf")
        best_idx = -1
        for idx, poly in enumerate(polynomials):
            dom = domains[idx]
            if dom is not None:
                a, b = dom
                if not (a - 1e-9 <= x <= b + 1e-9):
                    continue  # outside domain, skip
            y_pred = _eval_poly(poly, x)
            err = abs(y_pred - y_true)
            if err < best_err:
                best_err = err
                best_idx = idx
        # Only count if some function covered this x
        if best_idx >= 0 and best_err < float("inf"):
            errors.append(best_err)
            assignments[best_idx] += 1

    stats = _summary_stats(errors)

    # Print a small report (visible in pytest -s or when run as a script)
    report = [
        "Integration validation for letter 'B'",
        f"Total points: {int(stats['count'])}",
        f"Functions generated: {len(y_functions)}",
        f"Mean abs error: {stats['mean']:.6g}",
        f"Median abs error: {stats['median']:.6g}",
        f"p90 abs error: {stats['p90']:.6g}",
        f"p99 abs error: {stats['p99']:.6g}",
        f"Max abs error: {stats['max']:.6g}",
        f"Relative (median / y-range): {(stats['median']/y_range):.4f}",
        f"Relative (p90 / y-range): {(stats['p90']/y_range):.4f}",
    ]
    print("\n".join(report))

    # 5) Basic assertions (generous thresholds, adjust as needed)
    # These are relative to the vertical scale of the glyph.
    assert stats["median"] <= 0.10 * y_range, "Median error too large (relative to y-range)"
    assert stats["p90"] <= 0.25 * y_range, "90th percentile error too large (relative to y-range)"
    # Max can be high due to stroke separation; keep it lenient
    assert stats["max"] <= 0.75 * y_range, "Max error too large (relative to y-range)"


if __name__ == "__main__":  # Allow running as a script for quick checks
    try:
        test_integration_letter_B()
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)
    print("OK")
