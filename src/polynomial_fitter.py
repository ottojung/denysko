#!/usr/bin/env python3
"""
Polynomial fitting module - REIMPLEMENTED FROM SCRATCH

Two guiding principles:
1. Exact fitting to as many points as possible
2. Separation into multiple curves when multiple strokes cover same horizontal domain

Algorithm:
- Detect horizontal overlap (multiple y values at similar x coordinates)
- Split overlapping regions into separate curves  
- Fit exact polynomial through ALL points in each curve
- Use degree = n-1 for n points (exact interpolation) on each piece
- IMPORTANT: No domain restrictions in the output; functions are for all real x
"""

import numpy as np
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polynomial as P  # for polyadd, polymul, polypow


class PolynomialFitter:
    """Fits exact polynomials to letter coordinate points."""
    
    def __init__(self, max_degree=None):
        """Initialize (max_degree ignored to allow exact fitting)."""
        self.max_degree = max_degree  # kept for compatibility; not used as a cap
        # Piecewise control to maintain numerical stability while keeping exactness
        self.min_points_per_piece = 5
        self.max_points_per_piece = 12  # not a degree cap; we segment instead
        self.max_x_span_ratio = 0.15  # each piece spans at most 15% of total x-range
    
    def fit_contour_polynomials(self, contour, max_degree=None):
        """
        Main fitting method - implements the two core principles.
        
        Args:
            contour: Array of (x, y) letter centerline points
            max_degree: (ignored) kept for API compatibility
            
        Returns:
            list: Polynomial function strings that pass exactly through points
        """
        if len(contour) < 10:  # Require many points for quality fitting
            print(f"Warning: Only {len(contour)} points, need at least 10 for quality fitting")
            return []
        
        print(f"Fitting {len(contour)} letter points...")
        
        # PRINCIPLE 2: Detect and separate overlapping horizontal strokes
        curves = self._detect_overlapping_strokes(contour)
        
        print(f"Found {len(curves)} separate curves")
        
        # PRINCIPLE 1: Fit exact polynomials to each curve (piecewise if large)
        functions = []
        for i, curve in enumerate(curves):
            print(f"Curve {i+1}: {len(curve)} points")
            funcs = self._fit_exact_polynomial_piecewise(curve)
            functions.extend(funcs)
        
        print(f"Generated {len(functions)} exact polynomials")
        return functions
    
    def _detect_overlapping_strokes(self, points):
        """
        Detect if multiple strokes occupy the same horizontal space.
        Key for separating letter "A" diagonals from crossbar.
        
        Args:
            points: Array of (x, y) coordinates
            
        Returns:
            list: Separated stroke arrays
        """
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Check for significant y-variation within x-ranges (indicates overlap)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        x_span = x_max - x_min
        
        if x_span < 1e-6:
            return [points]  # All same x
        
        # Divide into x-segments and check y-variation in each
        num_segments = 8
        overlap_found = False
        
        for i in range(num_segments):
            seg_start = x_min + i * x_span / num_segments
            seg_end = x_min + (i + 1) * x_span / num_segments
            
            # Points in this x-segment
            in_segment = (x_coords >= seg_start) & (x_coords <= seg_end)
            if np.sum(in_segment) < 2:
                continue
                
            seg_y = y_coords[in_segment]
            y_variation = np.max(seg_y) - np.min(seg_y)
            
            # If y varies significantly, we have overlapping strokes
            if y_variation > x_span * 0.2:  # 20% threshold
                overlap_found = True
                break
        
        if not overlap_found:
            return [points]  # Single stroke
        
        # Separate into upper and lower strokes
        y_median = np.median(y_coords)
        
        upper = points[y_coords >= y_median]
        lower = points[y_coords < y_median]
        
        strokes = []
        if len(upper) >= self.min_points_per_piece:
            strokes.append(upper)
        if len(lower) >= self.min_points_per_piece:
            strokes.append(lower)
        
        return strokes if strokes else [points]

    def _fit_exact_polynomial_piecewise(self, points):
        """Exact interpolation on one or more pieces to avoid numerical blowup.
        Returns a list of function strings, each valid for all real x (no domain suffix).
        """
        # Sort by x and handle duplicates
        x_data, y_data = points[:, 0], points[:, 1]
        sort_idx = np.argsort(x_data)
        x_sorted, y_sorted = x_data[sort_idx], y_data[sort_idx]
        
        # Average duplicate x-values to ensure unique x's for interpolation
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        unique_y = np.array([np.mean(y_sorted[inverse == i]) for i in range(len(unique_x))])
        
        n = len(unique_x)
        if n < self.min_points_per_piece:
            print(f"  Skipping: only {n} unique x-coordinates, need ≥{self.min_points_per_piece}")
            return []
        
        funcs = []
        total_span = float(np.max(unique_x) - np.min(unique_x)) if n > 1 else 0.0
        max_span = self.max_x_span_ratio * total_span if total_span > 0 else float('inf')
        
        if n <= self.max_points_per_piece and (total_span <= max_span or not np.isfinite(max_span)):
            func = self._fit_exact_single(unique_x, unique_y)
            if func:
                funcs.append(func)
            return funcs
        
        # Segment into contiguous chunks by x with both point-count and x-span constraints
        start = 0
        while start < n:
            # initial tentative end by count
            tentative_end = min(start + self.max_points_per_piece, n)
            end = tentative_end
            # enforce x-span constraint by shrinking end if needed
            while end - start >= self.min_points_per_piece and (unique_x[end - 1] - unique_x[start]) > max_span:
                end -= 1
            # if still violates (e.g., enormous gaps), force minimal viable chunk
            if end - start < self.min_points_per_piece:
                end = min(start + self.min_points_per_piece, n)
            x_seg = unique_x[start:end]
            y_seg = unique_y[start:end]
            if len(x_seg) >= self.min_points_per_piece:
                func = self._fit_exact_single(x_seg, y_seg)
                if func:
                    funcs.append(func)
            start = end
        return funcs

    def _fit_exact_single(self, x_vals, y_vals):
        """Fit a single exact polynomial of degree len(x_vals)-1 to the segment.
        Uses affine scaling to [-1,1] to stabilize the Vandermonde solve, then
        composes back to the x-basis to emit a global polynomial (no domain).
        """
        n_points = len(x_vals)
        degree = max(2, n_points - 1)
        print(f"  Points: {n_points}, Natural degree: {n_points - 1}, Using: {degree}")
        try:
            xmin = float(np.min(x_vals))
            xmax = float(np.max(x_vals))
            span = xmax - xmin if xmax > xmin else 1.0
            # Affine map: x in [xmin,xmax] -> z in [-1,1]: z = a*x + b
            a = 2.0 / span
            b = -(xmax + xmin) / span
            z_vals = a * x_vals + b

            # Build stable Vandermonde in ascending powers of z and solve for exact coefficients
            Vz = np.vander(z_vals, N=degree + 1, increasing=True)  # columns: z^0, z^1, ..., z^degree
            cz = np.linalg.solve(Vz, y_vals)  # cz[k] is coeff for z^k

            # Compose p(z) with z=a*x+b in coefficient space (ascending order)
            # px(x) = sum_k cz[k] * (a*x + b)^k
            z_poly = np.array([b, a], dtype=float)  # coefficients of b + a*x
            px = np.array([0.0], dtype=float)
            for k, ck in enumerate(cz):
                if abs(ck) < 1e-18:
                    continue
                # (a*x + b)^k
                zk = P.polypow(z_poly, k)
                term = zk * ck
                # pad and add
                px = P.polyadd(px, term)

            # Verify exactness back on original x
            y_fit = P.polyval(x_vals, px)  # ascending order
            max_err = float(np.max(np.abs(y_fit - y_vals)))
            print(f"  Max error = {max_err:.8e}")
            if not np.isfinite(max_err) or max_err > 1e-8:
                print("  WARNING: Residual detected; expected exact fit")

            # Convert to string (need high->low order)
            func = self._coeffs_to_string(px[::-1])  # reverse to high->low for string builder
            if func is None:
                return None
            print("  Generated function (no domain)")
            return func
        except np.linalg.LinAlgError as e:
            print(f"  Linear solve failed: {e}")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None

    def _encode_exponent(self, power):
        """Return exponent as a standard integer string for Desmos.
        Note: Desmos accepts multi-digit exponents directly as x^12. Avoid chaining carets.
        """
        return str(int(power))

    def _format_number(self, value, precision: int = 15) -> str:
        """Format a float without scientific notation (no 'e'/'E'), trimming trailing zeros.
        Ensures outputs like 0.0000001 instead of 1e-7 for better Desmos compatibility.
        """
        try:
            s = np.format_float_positional(float(value), precision=precision, unique=False, trim='-')
            # Normalize negative zero to plain zero
            if s.startswith('-0') and float(value) == 0.0:
                return '0'
            return s
        except Exception:
            return str(value)

    def _coeffs_to_string(self, coeffs):
        """Convert coefficients (high->low) to clean function string (no domain constraints)."""
        if len(coeffs) < 3:
            print(f"  Error: Polynomial degree too low - got {len(coeffs)-1}, need ≥ 2")
            return None
        terms = []
        degree = len(coeffs) - 1
        print(f"  Converting degree {degree} polynomial to string")
        for i, c in enumerate(coeffs):
            power = degree - i
            if not np.isfinite(c) or abs(c) < 1e-15:
                continue
            
            if power == 0:
                c_str = self._format_number(c, precision=15)
                terms.append(c_str)
            elif power == 1:
                if abs(c - 1.0) < 1e-15:
                    terms.append("x")
                elif abs(c + 1.0) < 1e-15:
                    terms.append("-x")
                else:
                    c_str = self._format_number(c, precision=15)
                    terms.append(f"{c_str}*x")
            else:
                power_str = self._encode_exponent(power)
                if abs(c - 1.0) < 1e-15:
                    terms.append(f"x^{power_str}")
                elif abs(c + 1.0) < 1e-15:
                    terms.append(f"-x^{power_str}")
                else:
                    c_str = self._format_number(c, precision=15)
                    terms.append(f"{c_str}*x^{power_str}")
        
        if not terms:
            print("  Warning: All coefficients were essentially zero")
            return "y = 0"
        result = "y = " + terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += " - " + term[1:]
            else:
                result += " + " + term
        result = result.replace(" + -", " - ")
        print(f"  Generated function: {result}")
        return result
    
    def fit_contours_to_polynomials(self, contours, max_degree=None):
        """Fit polynomials to multiple contours (exact fit; piecewise)."""
        all_functions = []
        
        for i, contour in enumerate(contours):
            print(f"\nContour {i+1}/{len(contours)}:")
            functions = self.fit_contour_polynomials(contour)
            all_functions.extend(functions)
        
        return all_functions
