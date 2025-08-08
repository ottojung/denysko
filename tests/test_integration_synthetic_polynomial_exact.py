#!/usr/bin/env python3
"""
Integration-like sanity tests on simple, known polynomials.

Goal: Use shapes where the correct plot is unambiguous and exact fitting
should succeed, so we can reliably validate the whole pipeline.

We bypass text extraction and feed synthetic (x, y) contours directly into the
PolynomialFitter. We then:
- fit the curves
- parse generated function strings
- verify coefficients match the ground truth (within tight tolerance)
- verify evaluation errors are near machine precision on a test grid

These tests also ensure NO domain ranges are added (functions valid for all x).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List

import numpy as np

# Make repository root importable when running as a script
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.polynomial_fitter import PolynomialFitter  # type: ignore  # noqa: E402


def _eval_poly_coeffs(coeffs_hi_to_lo: List[float], x: float) -> float:
    y = 0.0
    deg = len(coeffs_hi_to_lo) - 1
    for i, c in enumerate(coeffs_hi_to_lo):
        p = deg - i
        y += c * (x ** p)
    return y


def _parse_poly(func: str) -> Dict[int, float]:
    """Parse 'y = a*x^n + ... + b*x + c' into {power: coeff}.
    Supports optional '*' and scientific notation.
    """
    s = func.strip()
    assert s.lower().startswith("y ="), f"Not y=f(x): {func}"
    expr = s.split("=", 1)[1].strip().replace(" ", "")
    assert '{' not in expr and '}' not in expr, "Function must not contain domain ranges"
    # remove any stray parentheses
    expr = expr.replace("(", "").replace(")", "")

    terms: List[str] = []
    i = 0
    start = 0
    while i < len(expr):
        ch = expr[i]
        if ch in '+-' and i != start:
            prev = expr[i-1]
            if prev not in ('e', 'E'):
                terms.append(expr[start:i])
                start = i
        i += 1
    if start < len(expr):
        terms.append(expr[start:])

    power_to_coeff: Dict[int, float] = {}
    for term in terms:
        if not term:
            continue
        if 'x' in term:
            coeff_part, _, pow_part = term.partition('x')
            if coeff_part.endswith('*'):
                coeff_part = coeff_part[:-1]
            if coeff_part in ('', '+'):
                coeff = 1.0
            elif coeff_part == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_part)
            if pow_part.startswith('^'):
                power = int(pow_part[1:])
            else:
                power = 1
        else:
            coeff = float(term)
            power = 0
        power_to_coeff[power] = power_to_coeff.get(power, 0.0) + coeff
    return power_to_coeff


def _coeff_dict_to_list(power_to_coeff: Dict[int, float]) -> List[float]:
    max_p = max(power_to_coeff.keys()) if power_to_coeff else 0
    return [power_to_coeff.get(p, 0.0) for p in range(max_p, -1, -1)]


def test_exact_fit_known_degree5():
    # Ground truth polynomial: y = 2x^5 + 0x^4 - 3x^3 + 0x^2 + 1x - 4
    true_coeffs = [2.0, 0.0, -3.0, 0.0, 1.0, -4.0]  # high->low
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])  # 6 points => degree 5 exact
    ys = np.array([_eval_poly_coeffs(true_coeffs, x) for x in xs])

    # Build contour (n,2)
    contour = np.column_stack([xs, ys])

    fitter = PolynomialFitter(max_degree=None)
    # Fit directly; expect a single function, degree 5; no domains
    funcs = fitter.fit_contour_polynomials(contour)

    assert funcs, "No function generated"
    # We expect a small number of segments; here data is tiny, should be 1
    assert len(funcs) == 1, f"Expected 1 function, got {len(funcs)}"
    f = funcs[0]
    assert '{' not in f and '}' not in f, "Function must not contain domain ranges"

    # Parse and compare coefficients
    parsed = _parse_poly(f)
    got_coeffs = _coeff_dict_to_list(parsed)
    # Pad to same length
    n = max(len(got_coeffs), len(true_coeffs))
    got_coeffs = [0.0]*(n - len(got_coeffs)) + got_coeffs
    true_coeffs_padded = [0.0]*(n - len(true_coeffs)) + true_coeffs

    # Tight numerical tolerance
    for i, (g, t) in enumerate(zip(got_coeffs, true_coeffs_padded)):
        assert math.isclose(g, t, rel_tol=1e-9, abs_tol=1e-9), (
            f"Coeff mismatch at power {n-1-i}: got {g}, expected {t}, func={f}")

    # Evaluate on a small grid
    grid = np.linspace(-3.0, 3.0, 25)
    err = []
    for x in grid:
        y_true = _eval_poly_coeffs(true_coeffs, x)
        # evaluate parsed polynomial
        y_pred = sum(c * (x ** p) for p, c in parsed.items())
        err.append(abs(y_pred - y_true))
    assert max(err) < 1e-7, f"Evaluation error too high: max={max(err)}"


def test_exact_fit_parabola():
    # y = x^2 - 4x + 3 (degree 2)
    true_coeffs = [1.0, -4.0, 3.0]
    xs = np.array([-1.0, 0.0, 2.0])  # 3 points => degree 2 exact
    ys = np.array([_eval_poly_coeffs(true_coeffs, x) for x in xs])
    contour = np.column_stack([xs, ys])

    fitter = PolynomialFitter(max_degree=None)
    funcs = fitter.fit_contour_polynomials(contour)

    assert funcs, "No function generated"
    assert len(funcs) == 1, f"Expected 1 function, got {len(funcs)}"
    f = funcs[0]
    assert '{' not in f and '}' not in f, "Function must not contain domain ranges"

    parsed = _parse_poly(f)
    got_coeffs = _coeff_dict_to_list(parsed)
    n = max(len(got_coeffs), len(true_coeffs))
    got_coeffs = [0.0]*(n - len(got_coeffs)) + got_coeffs
    true_coeffs_padded = [0.0]*(n - len(true_coeffs)) + true_coeffs

    for i, (g, t) in enumerate(zip(got_coeffs, true_coeffs_padded)):
        assert math.isclose(g, t, rel_tol=1e-9, abs_tol=1e-9), (
            f"Coeff mismatch at power {n-1-i}: got {g}, expected {t}, func={f}")

    grid = np.linspace(-2.0, 3.0, 21)
    err = []
    for x in grid:
        y_true = _eval_poly_coeffs(true_coeffs, x)
        y_pred = sum(c * (x ** p) for p, c in parsed.items())
        err.append(abs(y_pred - y_true))
    assert max(err) < 1e-9, f"Evaluation error too high: max={max(err)}"
