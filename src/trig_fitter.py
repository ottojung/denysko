#!/usr/bin/env python3
"""
Trigonometric fitting module (exact interpolation using sin/cos basis).

Principles:
1. Exact fitting: solve a linear system in a real Fourier-like basis so the curve
   passes exactly through all provided points in a stroke.
2. Minimal number of curves: only split vertically when multiple strokes overlap
   the same x-domain (upper/lower separation). No x-domain segmentation.

Implementation notes:
- Normalize x to t in [0, 2π] (affine map) for numerical stability.
- For n points, construct a basis of size n and solve exactly A c = y:
  * If n is odd: columns = [1, cos(1t), sin(1t), ..., cos(Kt), sin(Kt)] with K = (n-1)//2.
  * If n is even: columns = [1, cos(1t), sin(1t), ..., cos((K-1)t), sin((K-1)t), cos(Kt)] with K = n//2.
  This yields exactly n parameters for n equations.
- Output functions are global (no domain constraints), formatted for Desmos (no scientific notation).
"""

from __future__ import annotations

import numpy as np


class TrigFitter:
    """Fits exact trigonometric (sin/cos) functions to letter coordinate points."""

    def __init__(self):
        self.min_points_per_stroke = 5

    def fit_contour_functions(self, contour):
        """Fit one exact sin/cos function per stroke within a contour.
        Returns a list of Desmos-ready strings like: y = c0 + a1*cos(w*x) + b1*sin(w*x) + ...
        """
        if len(contour) < 10:
            print(f"Warning: Only {len(contour)} points, need at least 10 for quality fitting")
            return []

        print(f"Fitting (trig) {len(contour)} letter points...")
        curves = self._detect_overlapping_strokes(contour)
        print(f"Found {len(curves)} separate curves (trig)")

        functions = []
        for i, curve in enumerate(curves):
            print(f"Curve {i+1}: {len(curve)} points")
            func = self._fit_exact_trig_single(curve)
            if func:
                functions.append(func)
        print(f"Generated {len(functions)} exact trig functions")
        return functions

    def _detect_overlapping_strokes(self, points):
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        x_span = x_max - x_min
        if x_span < 1e-6:
            return [points]
        num_segments = 8
        overlap_found = False
        for i in range(num_segments):
            seg_start = x_min + i * x_span / num_segments
            seg_end = x_min + (i + 1) * x_span / num_segments
            in_segment = (x_coords >= seg_start) & (x_coords <= seg_end)
            if np.sum(in_segment) < 2:
                continue
            seg_y = y_coords[in_segment]
            y_variation = np.max(seg_y) - np.min(seg_y)
            if y_variation > x_span * 0.2:  # heuristic
                overlap_found = True
                break
        if not overlap_found:
            return [points]
        y_median = np.median(y_coords)
        upper = points[y_coords >= y_median]
        lower = points[y_coords < y_median]
        strokes = []
        if len(upper) >= self.min_points_per_stroke:
            strokes.append(upper)
        if len(lower) >= self.min_points_per_stroke:
            strokes.append(lower)
        return strokes if strokes else [points]

    def _fit_exact_trig_single(self, points):
        # Sort by x and deduplicate x by averaging y
        x_data, y_data = points[:, 0], points[:, 1]
        idx = np.argsort(x_data)
        x_sorted, y_sorted = x_data[idx], y_data[idx]
        unique_x, inv = np.unique(x_sorted, return_inverse=True)
        unique_y = np.array([np.mean(y_sorted[inv == i]) for i in range(len(unique_x))])
        n = len(unique_x)
        if n < self.min_points_per_stroke:
            print(f"  Skipping: only {n} unique x-coordinates, need ≥{self.min_points_per_stroke}")
            return None

        # Normalize x to t in [0, 2π]
        xmin = float(np.min(unique_x))
        xmax = float(np.max(unique_x))
        span = xmax - xmin if xmax > xmin else 1.0
        t = 2.0 * np.pi * (unique_x - xmin) / span
        omega = 2.0 * np.pi / span  # fundamental frequency in x-space

        # Build exact basis (n columns)
        cols = [np.ones_like(t)]
        if n % 2 == 1:
            K = (n - 1) // 2
            for k in range(1, K + 1):
                cols.append(np.cos(k * t))
                cols.append(np.sin(k * t))
        else:
            K = n // 2
            for k in range(1, K):
                cols.append(np.cos(k * t))
                cols.append(np.sin(k * t))
            cols.append(np.cos(K * t))  # no sine at Nyquist-like term
        A = np.column_stack(cols)

        try:
            coeffs = np.linalg.solve(A, unique_y)
        except np.linalg.LinAlgError as e:
            print(f"  Linear solve failed (trig): {e}")
            return None

        # Verify exactness
        y_fit = A @ coeffs
        max_err = float(np.max(np.abs(y_fit - unique_y)))
        print(f"  Max error (trig) = {max_err:.8e}")
        if not np.isfinite(max_err) or max_err > 1e-8:
            print("  WARNING: Residual detected; expected exact trig fit")

        # Build function string
        func = self._coeffs_to_string_trig(coeffs, omega)
        if func is None:
            return None
        print("  Generated trig function (no domain)")
        return func

    def _format_number(self, value, precision: int = 15) -> str:
        try:
            s = np.format_float_positional(float(value), precision=precision, unique=False, trim='-')
            if s.startswith('-0') and float(value) == 0.0:
                return '0'
            return s
        except Exception:
            return str(value)

    def _coeffs_to_string_trig(self, coeffs: np.ndarray, omega: float) -> str | None:
        """Format y = c0 + a1*cos(omega*x) + b1*sin(omega*x) + ...
        Coeff order matches basis construction above.
        """
        n = len(coeffs)
        if n < 1:
            return None
        terms = []
        c0 = coeffs[0]
        if abs(c0) >= 1e-15:
            terms.append(self._format_number(c0, precision=15))

        idx = 1
        k = 1
        if n % 2 == 1:
            # odd: pairs for k=1..K
            while idx + 1 < n:
                a_k = coeffs[idx]
                b_k = coeffs[idx + 1]
                idx += 2
                if abs(a_k) >= 1e-15:
                    a_str = self._format_number(a_k, precision=15)
                    terms.append(f"{a_str}*cos({self._format_number(omega*k, precision=15)}*x)")
                if abs(b_k) >= 1e-15:
                    b_str = self._format_number(b_k, precision=15)
                    terms.append(f"{b_str}*sin({self._format_number(omega*k, precision=15)}*x)")
                k += 1
        else:
            # even: pairs up to k=K-1, then cos(Kt)
            while k < (n // 2):
                a_k = coeffs[idx]
                b_k = coeffs[idx + 1]
                idx += 2
                if abs(a_k) >= 1e-15:
                    a_str = self._format_number(a_k, precision=15)
                    terms.append(f"{a_str}*cos({self._format_number(omega*k, precision=15)}*x)")
                if abs(b_k) >= 1e-15:
                    b_str = self._format_number(b_k, precision=15)
                    terms.append(f"{b_str}*sin({self._format_number(omega*k, precision=15)}*x)")
                k += 1
            # final cos(Kt)
            aK = coeffs[idx]
            if abs(aK) >= 1e-15:
                aK_str = self._format_number(aK, precision=15)
                terms.append(f"{aK_str}*cos({self._format_number(omega*k, precision=15)}*x)")

        if not terms:
            return "y = 0"

        result = "y = " + terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += " - " + term[1:]
            else:
                result += " + " + term
        result = result.replace(" + -", " - ")
        print(f"  Generated trig function: {result}")
        return result
