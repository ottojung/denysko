#!/usr/bin/env python3
"""
Polynomial fitting module - fits y=f(x) polynomials to coordinate points.
"""

import numpy as np


class PolynomialFitter:
    """Fits y=f(x) polynomials to coordinate point data."""
    
    def __init__(self, max_degree=8):
        """Initialize the polynomial fitter.
        
        Args:
            max_degree (int): Maximum polynomial degree to use for fitting
        """
        self.max_degree = max_degree
    
    def split_contour_into_x_monotonic_segments(self, contour):
        """
        Split a contour into segments where x is monotonic (always increasing or decreasing).
        This ensures each segment can be represented as y=f(x).
        Uses a more sophisticated approach to preserve letter shape.
        
        Args:
            contour (np.array): Array of (x, y) points along the contour
            
        Returns:
            list: List of segments, each being an array of (x, y) points
        """
        if len(contour) < 3:
            return [contour]
        
        # Remove duplicate points first
        unique_points = []
        for point in contour:
            if len(unique_points) == 0 or not np.allclose(point, unique_points[-1], atol=1e-6):
                unique_points.append(point)
        
        if len(unique_points) < 3:
            return [np.array(unique_points)]
        
        contour = np.array(unique_points)
        
        # Find natural breakpoints where x direction changes significantly
        segments = []
        current_segment = [contour[0]]
        
        for i in range(1, len(contour)):
            current_segment.append(contour[i])
            
            # Look ahead to see if we need to break the segment
            if len(current_segment) >= 5 and i < len(contour) - 2:
                # Get x coordinates of recent points
                recent_x = np.array([p[0] for p in current_segment[-5:]])
                
                # Check if x is becoming non-monotonic
                x_diffs = np.diff(recent_x)
                
                # Count sign changes in x differences
                signs = np.sign(x_diffs)
                sign_changes = np.sum(np.abs(np.diff(signs)) > 1)
                
                # If we have multiple direction changes, break the segment
                if sign_changes >= 2:
                    # End current segment with some overlap for continuity
                    segments.append(np.array(current_segment[:-1]))
                    current_segment = current_segment[-2:]  # Keep overlap
        
        # Add the final segment
        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))
        
        # Post-process: merge very short segments with neighbors
        filtered_segments = []
        for segment in segments:
            if len(segment) >= 3:
                filtered_segments.append(segment)
            elif len(filtered_segments) > 0:
                # Merge short segment with previous
                filtered_segments[-1] = np.vstack([filtered_segments[-1], segment])
        
        return filtered_segments if filtered_segments else [contour]
    
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
    
    def fit_polynomial_to_segment(self, segment, max_degree=12):
        """
        Fit a y=f(x) polynomial to a segment of points with OVERFITTING for better accuracy.
        
        Args:
            segment (np.array): Array of (x, y) points
            max_degree (int): Maximum polynomial degree (increased for overfitting)
            
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
            
            # OVERFITTING: Use high degree polynomials for better accuracy
            degree = min(max_degree, len(x_sorted) - 1)
            
            # Use higher degrees even for few points to overfit better
            if len(x_sorted) >= 5:
                degree = min(max_degree, len(x_sorted) - 1)
            elif len(x_sorted) >= 3:
                degree = min(max_degree // 2, len(x_sorted) - 1)
            else:
                degree = 1
            
            # Fit polynomial with high degree for overfitting
            coeffs = np.polyfit(x_sorted, y_sorted, degree)
            
            # Generate function string with higher precision for overfitting
            terms = []
            for i, coeff in enumerate(coeffs):
                if abs(coeff) < 1e-15:  # Increased precision threshold
                    continue
                    
                power = degree - i
                if power == 0:
                    terms.append(f"{coeff:.12f}")  # Higher precision
                elif power == 1:
                    terms.append(f"{coeff:.12f}*x")
                else:
                    terms.append(f"{coeff:.12f}*x^{power}")
            
            if terms:
                func_str = " + ".join(terms).replace("+ -", "- ")
                return f"y = {func_str}"
            
        except Exception as e:
            print(f"Warning: Failed to fit polynomial to segment: {e}")
        
        return None
    
    def decide_curves_for_letter(self, contour):
        """
        Decide how many curves to fit for this letter based on its complexity.
        
        Args:
            contour (np.array): Array of (x, y) points along the contour
            
        Returns:
            int: Number of curves to generate for this letter
        """
        # Calculate contour complexity based on curvature changes
        if len(contour) < 10:
            return 2  # Simple shape, use 2 curves
        
        # Calculate approximate curvature by looking at direction changes
        directions = []
        for i in range(len(contour) - 2):
            v1 = contour[i+1] - contour[i]
            v2 = contour[i+2] - contour[i+1]
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                dot_product = np.clip(dot_product, -1, 1)  # Handle numerical errors
                angle = np.arccos(dot_product)
                directions.append(angle)
        
        # Count significant direction changes
        significant_changes = sum(1 for angle in directions if angle > np.pi/6)  # 30 degrees
        
        # Determine number of curves based on complexity (REDUCED for overfitting)
        if significant_changes < 3:
            return 8  # Simple letter - fewer curves, more points each
        elif significant_changes < 8:
            return 12  # Medium complexity like A, P
        elif significant_changes < 15:
            return 18  # Complex letters like B, R
        else:
            return 25  # Very complex letters

    def generate_smart_sample_points(self, contour, num_points=10):
        """
        Generate smart sample points that are close enough for good polynomial fitting.
        
        Args:
            contour (np.array): Array of (x, y) points along the contour
            num_points (int): Number of points to sample
            
        Returns:
            np.array: Array of sampled (x, y) points
        """
        if len(contour) <= num_points:
            return contour
        
        # Calculate cumulative distance along the contour
        distances = np.cumsum(
            np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        )
        distances = np.insert(distances, 0, 0)  # Add starting point
        total_length = distances[-1]
        
        if total_length == 0:
            return contour[:num_points]
        
        # Generate evenly spaced points along the curve
        target_distances = np.linspace(0, total_length, num_points)
        sampled_points = []
        
        for target_dist in target_distances:
            # Find the closest point on the contour
            closest_idx = np.argmin(np.abs(distances - target_dist))
            sampled_points.append(contour[closest_idx])
        
        return np.array(sampled_points)

    def generate_curve_from_region(self, contour, start_ratio, end_ratio, points_per_curve=10):
        """
        Generate a curve from a specific region of the contour.
        
        Args:
            contour (np.array): Full contour points
            start_ratio (float): Starting position as ratio (0.0 to 1.0)
            end_ratio (float): Ending position as ratio (0.0 to 1.0)
            points_per_curve (int): Number of points to sample for this curve
            
        Returns:
            np.array: Points for this curve region
        """
        # Calculate region boundaries
        start_idx = int(start_ratio * len(contour))
        end_idx = int(end_ratio * len(contour))
        
        # Handle wraparound
        if start_idx >= end_idx:
            end_idx = len(contour)
        
        # Extract region
        region = contour[start_idx:end_idx]
        
        if len(region) < 2:
            return contour[start_idx:start_idx+2] if start_idx+2 <= len(contour) else contour[-2:]
        
        # Sample points smartly from this region
        return self.generate_smart_sample_points(region, points_per_curve)

    def fit_contour_polynomials(self, contour, max_degree=12):
        """
        OVERFITTING STRATEGY: Fit multiple high-degree y=f(x) polynomials with many points each.
        
        For each letter:
        1. Generate fewer curves but with MANY MORE POINTS each
        2. Use HIGH DEGREE polynomials to overfit and capture fine details
        3. Each curve uses 30-50 points for precise fitting
        
        Args:
            contour (np.array): Array of (x, y) points along the contour
            max_degree (int): Maximum polynomial degree (increased for overfitting)
            
        Returns:
            list: List of polynomial function strings
        """
        if len(contour) < 2:
            return []
        
        # Step 1: Decide number of curves (FEWER curves, MORE points each)
        num_curves = self.decide_curves_for_letter(contour)
        print(f"    OVERFITTING strategy: generating {num_curves} curves with many points each")
        
        functions = []
        # INCREASED points per curve for overfitting
        points_per_curve = max(30, len(contour) // num_curves)  # At least 30 points per curve
        points_per_curve = min(points_per_curve, 80)  # Cap at 80 points to avoid excessive computation
        
        # Step 2: Generate curves from different regions with HEAVY OVERLAP
        for curve_idx in range(num_curves):
            # INCREASED overlap for better continuity and more points
            start_ratio = (curve_idx / num_curves) * 0.7  # 30% overlap
            end_ratio = ((curve_idx + 1) / num_curves) * 1.3
            end_ratio = min(1.0, end_ratio)
            
            # Step 3: Generate MANY smart sample points for this region
            curve_points = self.generate_curve_from_region(contour, start_ratio, end_ratio, points_per_curve)
            
            if len(curve_points) < 3:
                continue
            
            # Step 4: Fit HIGH-DEGREE polynomial to overfit these many points
            func = self.fit_polynomial_to_segment(curve_points, max_degree)
            
            if func:
                functions.append(func)
                actual_degree = self._get_degree_from_function(func)
                print(f"      Curve {curve_idx+1}: {len(curve_points)} points -> degree {actual_degree} polynomial (OVERFITTING)")
            else:
                print(f"      Curve {curve_idx+1}: Failed to fit polynomial")
        
        print(f"    Total overfitted functions: {len(functions)}")
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
