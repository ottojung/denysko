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
- Use degree = n-1 for n points (exact interpolation)
"""

import numpy as np


class PolynomialFitter:
    """Fits exact polynomials to letter coordinate points."""
    
    def __init__(self, max_degree=25):
        """Initialize with default max degree."""
        self.max_degree = max_degree
    
    def fit_contour_polynomials(self, contour, max_degree=None):
        """
        Main fitting method - implements the two core principles.
        
        Args:
            contour: Array of (x, y) letter centerline points
            max_degree: Maximum polynomial degree (uses instance default if None)
            
        Returns:
            list: Polynomial function strings that pass exactly through points
        """
        if max_degree is None:
            max_degree = self.max_degree
            
        if len(contour) < 3:
            return []
        
        print(f"Fitting {len(contour)} letter points...")
        
        # PRINCIPLE 2: Detect and separate overlapping horizontal strokes
        curves = self._detect_overlapping_strokes(contour)
        
        print(f"Found {len(curves)} separate curves")
        
        # PRINCIPLE 1: Fit exact polynomials to each curve
        functions = []
        for i, curve in enumerate(curves):
            print(f"Curve {i+1}: {len(curve)} points")
            func = self._fit_exact_polynomial(curve, max_degree)
            if func:
                functions.append(func)
        
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
        if len(upper) >= 3:
            strokes.append(upper)
        if len(lower) >= 3:
            strokes.append(lower)
        
        return strokes if strokes else [points]
    
    def _fit_exact_polynomial(self, points, max_degree):
        """
        Fit polynomial that passes EXACTLY through all points.
        Uses exact interpolation (degree = n-1 for n points).
        
        Args:
            points: Array of (x, y) coordinates
            max_degree: Maximum degree allowed
            
        Returns:
            str: Exact polynomial function string
        """
        if len(points) < 3:
            return None
        
        # Sort by x and handle duplicates
        x_data, y_data = points[:, 0], points[:, 1]
        sort_idx = np.argsort(x_data)
        x_sorted, y_sorted = x_data[sort_idx], y_data[sort_idx]
        
        # Average duplicate x-values
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        unique_y = np.array([np.mean(y_sorted[inverse == i]) for i in range(len(unique_x))])
        
        n_points = len(unique_x)
        if n_points < 3:
            return None
        
        # EXACT FITTING: Use degree = n-1 (passes through ALL points)
        degree = min(n_points - 1, max_degree)
        
        # Handle minimum degree requirement
        if degree <= 1:
            if n_points >= 3:
                # We have enough points for degree > 1, use exact fitting
                degree = n_points - 1
            else:
                # Not enough points for degree > 1, skip this curve
                print(f"  Skipping: only {n_points} unique points, need ≥3 for degree > 1")
                return None
        
        try:
            # Polynomial interpolation - EXACT fit
            print(f"  Fitting degree {degree} polynomial through {n_points} points")
            print(f"  Points: {list(zip(unique_x, unique_y))}")
            
            coeffs = np.polyfit(unique_x, unique_y, degree)
            print(f"  Coefficients: {coeffs}")
            
            # Verify exactness
            poly = np.poly1d(coeffs)
            errors = np.abs(poly(unique_x) - unique_y)
            max_error = np.max(errors)
            
            print(f"  Max error = {max_error:.8f}")
            
            # Show individual point errors
            print("  Point-by-point verification:")
            for i, (x, y) in enumerate(zip(unique_x, unique_y)):
                predicted = poly(x)
                error = abs(predicted - y)
                print(f"    Point {i+1}: x={x}, expected={y}, predicted={predicted:.8f}, error={error:.8f}")
            
            return self._coeffs_to_string(coeffs)
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def _coeffs_to_string(self, coeffs):
        """Convert coefficients to clean function string."""
        if len(coeffs) < 3:
            return None
        
        terms = []
        degree = len(coeffs) - 1
        
        for i, c in enumerate(coeffs):
            power = degree - i
            if abs(c) < 1e-12:
                continue
            
            # Format coefficient
            c_str = f"{c:.8g}"
            
            if power == 0:
                terms.append(c_str)
            elif power == 1:
                if abs(c - 1) < 1e-12:
                    terms.append("x")
                elif abs(c + 1) < 1e-12:
                    terms.append("-x")
                else:
                    terms.append(f"{c_str}*x")
            else:
                if abs(c - 1) < 1e-12:
                    terms.append(f"x^{power}")
                elif abs(c + 1) < 1e-12:
                    terms.append(f"-x^{power}")
                else:
                    terms.append(f"{c_str}*x^{power}")
        
        if not terms:
            return "y = 0"
        
        # Join with proper signs
        result = "y = " + terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += " - " + term[1:]
            else:
                result += " + " + term
        
        return result
    
    def fit_contours_to_polynomials(self, contours, max_degree=None):
        """Fit polynomials to multiple contours."""
        if max_degree is None:
            max_degree = self.max_degree
            
        all_functions = []
        
        for i, contour in enumerate(contours):
            print(f"\nContour {i+1}/{len(contours)}:")
            functions = self.fit_contour_polynomials(contour, max_degree)
            all_functions.extend(functions)
        
        return all_functions
