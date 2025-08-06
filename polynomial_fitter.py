#!/usr/bin/env python3
"""
Polynomial fitting module - fits y=f(x) polynomials to coordinate points.
"""

import numpy as np


class PolynomialFitter:
    """Fits y=f(x) polynomials to coordinate point data."""
    
    def __init__(self):
        """Initialize the polynomial fitter."""
        pass
    
    def split_contour_into_x_monotonic_segments(self, contour):
        """
        Split a contour into segments where x is monotonic (always increasing or decreasing).
        This ensures each segment can be represented as y=f(x).
        
        Args:
            contour (np.array): Array of (x, y) points along the contour
            
        Returns:
            list: List of segments, each being an array of (x, y) points
        """
        if len(contour) < 3:
            return [contour]
        
        segments = []
        current_segment = [contour[0]]
        
        for i in range(1, len(contour)):
            current_point = contour[i]
            current_segment.append(current_point)
            
            # Check if we should start a new segment
            if len(current_segment) >= 3:
                # Check if x direction has changed significantly
                x_values = np.array([p[0] for p in current_segment])
                
                # Calculate x differences to determine direction changes
                x_diffs = np.diff(x_values)
                
                # Count direction changes (sign changes in differences)
                sign_changes = np.sum(np.diff(np.sign(x_diffs)) != 0)
                
                # If too many direction changes, start a new segment
                if sign_changes > 2:  # Allow some noise but detect major turns
                    # Start new segment with overlap for continuity
                    segments.append(np.array(current_segment[:-2]))
                    current_segment = current_segment[-2:]  # Keep last 2 points
        
        # Add the final segment
        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))
        
        return segments
    
    def sort_segment_by_x(self, segment):
        """
        Sort a segment by x coordinate and handle duplicate x values.
        
        Args:
            segment (np.array): Array of (x, y) points
            
        Returns:
            tuple: (x_sorted, y_sorted) arrays ready for polynomial fitting
        """
        x_data = segment[:, 0]
        y_data = segment[:, 1]
        
        # Sort by x
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        
        # Handle duplicate x values by averaging y values
        x_unique, indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        
        if len(x_unique) < len(x_sorted):
            # We have duplicate x values, average the y values
            y_averaged = np.array([
                np.mean(y_sorted[indices == i]) for i in range(len(x_unique))
            ])
            return x_unique, y_averaged
        
        return x_sorted, y_sorted
    
    def fit_polynomial_to_segment(self, segment, max_degree=8):
        """
        Fit a y=f(x) polynomial to a segment of points.
        
        Args:
            segment (np.array): Array of (x, y) points
            max_degree (int): Maximum polynomial degree
            
        Returns:
            str: Polynomial function string, or None if fitting fails
        """
        if len(segment) < 2:
            return None
        
        try:
            # Sort and clean the data
            x_sorted, y_sorted = self.sort_segment_by_x(segment)
            
            if len(x_sorted) < 2:
                return None
            
            # Choose degree based on number of points
            degree = min(max_degree, len(x_sorted) - 1)
            
            # For very few points, use linear
            if len(x_sorted) <= 2:
                degree = 1
            elif len(x_sorted) <= 4:
                degree = min(2, degree)
            
            # Fit polynomial
            coeffs = np.polyfit(x_sorted, y_sorted, degree)
            
            # Generate function string
            terms = []
            for i, coeff in enumerate(coeffs):
                if abs(coeff) < 1e-12:
                    continue
                    
                power = degree - i
                if power == 0:
                    terms.append(f"{coeff:.8f}")
                elif power == 1:
                    terms.append(f"{coeff:.8f}*x")
                else:
                    terms.append(f"{coeff:.8f}*x^{power}")
            
            if terms:
                func_str = " + ".join(terms).replace("+ -", "- ")
                return f"y = {func_str}"
            
        except Exception as e:
            print(f"Warning: Failed to fit polynomial to segment: {e}")
        
        return None
    
    def fit_contour_polynomials(self, contour, max_degree=8):
        """
        Fit multiple y=f(x) polynomials to a contour.
        
        Args:
            contour (np.array): Array of (x, y) points along the contour
            max_degree (int): Maximum polynomial degree
            
        Returns:
            list: List of polynomial function strings
        """
        if len(contour) < 2:
            return []
        
        # Split contour into x-monotonic segments
        segments = self.split_contour_into_x_monotonic_segments(contour)
        print(f"    Split into {len(segments)} x-monotonic segments")
        
        functions = []
        
        for i, segment in enumerate(segments):
            if len(segment) < 2:
                continue
                
            # Fit polynomial to this segment
            func = self.fit_polynomial_to_segment(segment, max_degree)
            
            if func:
                functions.append(func)
                print(f"      Segment {i+1}: {len(segment)} points -> polynomial degree {self._get_degree_from_function(func)}")
            else:
                print(f"      Segment {i+1}: Failed to fit polynomial")
        
        return functions
    
    def _get_degree_from_function(self, func_str):
        """Extract the degree from a polynomial function string."""
        if 'x^' not in func_str:
            if '*x' in func_str:
                return 1
            return 0
        
        max_degree = 0
        parts = func_str.split('x^')
        for part in parts[1:]:
            try:
                degree = int(part.split()[0].split('*')[0].split('+')[0].split('-')[0])
                max_degree = max(max_degree, degree)
            except:
                pass
        
        return max_degree
    
    def fit_contours_to_polynomials(self, contours, max_degree=8):
        """
        Fit y=f(x) polynomials to multiple contours.
        
        Args:
            contours (list): List of contour arrays
            max_degree (int): Maximum polynomial degree
            
        Returns:
            list: List of polynomial function strings
        """
        all_functions = []
        
        for i, contour in enumerate(contours):
            print(f"  Fitting contour {i+1}/{len(contours)} ({len(contour)} points)...")
            
            functions = self.fit_contour_polynomials(contour, max_degree)
            all_functions.extend(functions)
            
            print(f"    Generated {len(functions)} polynomial functions")
        
        return all_functions
