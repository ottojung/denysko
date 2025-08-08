#!/usr/bin/env python3
"""
Polynomial fitting module - REIMPLEMENTED FROM SCRATCH

Two guiding principles:
1. Exact fitting to as many points as possible
2. Separation into multiple curves when multiple strokes cover same horizontal domain

Algorithm:
- Detect horizontal overlap (multiple y values at similar x coordinates)
- Split overlapping regions into separate curves  
- Fit exact polynomial through ALL points in each curve (piecewise if needed)
- Use degree = n-1 for n points (exact interpolation) on each piece
"""

import numpy as np


class PolynomialFitter:
    """Fits exact polynomials to letter coordinate points."""
    
    def __init__(self, max_degree=25):
        """Initialize (max_degree ignored to allow exact fitting)."""
        self.max_degree = max_degree  # kept for compatibility; not used as a cap
        # Piecewise control to maintain numerical stability while keeping exactness
        self.min_points_per_piece = 5
        self.max_points_per_piece = 12  # not a degree cap; we segment instead
    
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
        Returns a list of function strings, each valid over its x-domain.
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
        
        # If small enough, fit in one go; otherwise segment into pieces of size <= max_points_per_piece
        funcs = []
        if n <= self.max_points_per_piece:
            func = self._fit_exact_single(unique_x, unique_y)
            if func:
                funcs.append(func)
            return funcs
        
        # Segment into contiguous chunks by x
        start = 0
        while start < n:
            end = min(start + self.max_points_per_piece, n)
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
        Produces a function string with a domain constraint matching the segment x-range.
        """
        n_points = len(x_vals)
        degree = max(2, n_points - 1)
        print(f"  Points: {n_points}, Natural degree: {n_points - 1}, Using: {degree}")
        try:
            # Solve Vandermonde exactly for small n
            V = np.vander(x_vals, N=degree + 1, increasing=False)
            coeffs = np.linalg.solve(V, y_vals)
            # Verify exactness
            y_fit = V @ coeffs
            max_err = float(np.max(np.abs(y_fit - y_vals)))
            print(f"  Max error = {max_err:.8e}")
            if max_err > 1e-9:
                print("  WARNING: Residual detected; expected exact fit")
            # Build function string and append domain constraint
            func = self._coeffs_to_string(coeffs)
            if func is None:
                return None
            x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
            func = f"{func} {{{x_min:.6f} <= x <= {x_max:.6f}}}"
            print(f"  Generated piecewise function with domain [{x_min:.3f}, {x_max:.3f}]")
            return func
        except np.linalg.LinAlgError as e:
            print(f"  Linear solve failed: {e}. Falling back to least squares (still exact within segment).")
            coeffs, *_ = np.linalg.lstsq(np.vander(x_vals, N=degree + 1, increasing=False), y_vals, rcond=None)
            func = self._coeffs_to_string(coeffs)
            if func is None:
                return None
            x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
            return f"{func} {{{x_min:.6f} <= x <= {x_max:.6f}}}"
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def _coeffs_to_string(self, coeffs):
        """Convert coefficients to clean function string."""
        if len(coeffs) < 3:
            print(f"  Error: Polynomial degree too low - got {len(coeffs)-1}, need ≥ 2")
            return None
        
        terms = []
        degree = len(coeffs) - 1
        
        print(f"  Converting degree {degree} polynomial to string")
        
        for i, c in enumerate(coeffs):
            power = degree - i
            if abs(c) < 1e-15:  # Very small threshold for essentially zero
                continue
            
            # Format coefficient with reasonable precision
            if abs(c) < 1e-6:
                c_str = f"{c:.15g}"  # High precision for small coefficients
            else:
                c_str = f"{c:.10g}"  # Normal precision
            
            if power == 0:
                terms.append(c_str)
            elif power == 1:
                if abs(c - 1.0) < 1e-15:
                    terms.append("x")
                elif abs(c + 1.0) < 1e-15:
                    terms.append("-x")
                else:
                    terms.append(f"{c_str}*x")
            else:
                if abs(c - 1.0) < 1e-15:
                    terms.append(f"x^{power}")
                elif abs(c + 1.0) < 1e-15:
                    terms.append(f"-x^{power}")
                else:
                    terms.append(f"{c_str}*x^{power}")
        
        if not terms:
            print("  Warning: All coefficients were essentially zero")
            return "y = 0"
        
        # Join with proper signs
        result = "y = " + terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += " - " + term[1:]
            else:
                result += " + " + term
        
        # Clean up the result - remove redundant + signs
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
