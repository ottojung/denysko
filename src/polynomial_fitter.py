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
        Fit a y=f(x) polynomial that is EXTREMELY ACCURATE at letter points.
        Uses weighted least-squares to prioritize accuracy at the actual letter shape.
        We don't care about behavior between or outside the letter points.
        
        Args:
            segment (np.array): Array of (x, y) points ON the letter centerline
            max_degree (int): Maximum polynomial degree
            
        Returns:
            str: Polynomial function string optimized for letter shape accuracy
        """
        if len(segment) < 2:
            return None
        
        try:
            # Sort and clean the data
            x_sorted, y_sorted = self.sort_segment_by_x(segment)
            
            if len(x_sorted) < 2:
                return None
            
            print(f"        SHAPE-OPTIMIZED FITTING: {len(x_sorted)} letter points")
            
            # Strategy: Use higher degree polynomials for better shape approximation
            degree = min(max_degree, max(3, len(x_sorted) // 2))  # At least cubic
            
            # Use polynomial fitting that prioritizes accuracy AT the letter points
            coeffs = self.fit_polynomial_for_shape_accuracy(x_sorted, y_sorted, degree)
            
            # Verify accuracy at the letter points (this is what matters!)
            poly_func = np.poly1d(coeffs)
            letter_errors = np.abs(poly_func(x_sorted) - y_sorted)
            max_error = np.max(letter_errors)
            avg_error = np.mean(letter_errors)
            
            print(f"        Letter shape accuracy: max_error={max_error:.6f}, avg_error={avg_error:.6f}")
            
            if max_error > 0.1:  # If error too high, try higher degree
                degree = min(max_degree, len(x_sorted) - 1)
                coeffs = self.fit_polynomial_for_shape_accuracy(x_sorted, y_sorted, degree)
                poly_func = np.poly1d(coeffs)
                max_error = np.max(np.abs(poly_func(x_sorted) - y_sorted))
                print(f"        Improved with degree {degree}: max_error={max_error:.6f}")
            
            # Generate function string WITH DOMAIN CONSTRAINTS
            # This prevents curves from extending beyond their intended region
            terms = []
            for i, coeff in enumerate(coeffs):
                if abs(coeff) < 1e-16:
                    continue
                    
                power = degree - i
                if power == 0:
                    terms.append(f"{coeff:.12f}")
                elif power == 1:
                    terms.append(f"{coeff:.12f}*x")
                else:
                    terms.append(f"{coeff:.12f}*x^{power}")
            
            if terms:
                func_str = " + ".join(terms).replace("+ -", "- ")
                
                # Add domain constraints to prevent curve from extending beyond its region
                x_min = np.min(x_sorted)
                x_max = np.max(x_sorted)
                
                # Use Desmos conditional syntax to restrict domain
                # This ensures the polynomial only appears in its intended x-range
                constrained_func = f"y = ({func_str}) \\{{\\{x_min:.6f} \\leq x \\leq {x_max:.6f}\\}}"
                
                print(f"        Domain constrained: x ∈ [{x_min:.3f}, {x_max:.3f}]")
                return constrained_func
            
        except Exception as e:
            print(f"Warning: Failed to create shape-optimized polynomial: {e}")
        
        return None
    
    def fit_polynomial_for_shape_accuracy(self, x_points, y_points, degree):
        """
        Fit polynomial that is extremely accurate at the letter shape points.
        Uses techniques to minimize error specifically at the given points.
        
        Args:
            x_points: X coordinates of letter centerline points
            y_points: Y coordinates of letter centerline points
            degree: Polynomial degree
            
        Returns:
            np.array: Polynomial coefficients optimized for shape accuracy
        """
        try:
            # Method 1: Try exact interpolation if we have few enough points
            if len(x_points) <= degree + 1:
                return np.polyfit(x_points, y_points, len(x_points) - 1)
            
            # Method 2: Weighted least squares with very high weights at letter points
            # This forces the polynomial to be very accurate at letter locations
            weights = np.ones(len(x_points)) * 1000.0  # Very high weight for letter points
            
            # Use weighted polynomial fitting
            coeffs = np.polyfit(x_points, y_points, degree, w=weights)
            
            # Check if we can improve by adding more constraint points
            poly_func = np.poly1d(coeffs)
            errors = np.abs(poly_func(x_points) - y_points)
            
            if np.max(errors) > 0.01:  # If still not accurate enough
                # Method 3: Add intermediate constraint points for smoother curves
                enhanced_x, enhanced_y = self.add_shape_constraints(x_points, y_points)
                weights_enhanced = np.ones(len(enhanced_x)) * 1000.0
                coeffs = np.polyfit(enhanced_x, enhanced_y, degree, w=weights_enhanced)
            
            return coeffs
            
        except Exception:
            # Fallback to regular polyfit
            return np.polyfit(x_points, y_points, degree)
    
    def add_shape_constraints(self, x_points, y_points):
        """
        Add intermediate points to help the polynomial follow the letter shape better.
        This creates a smoother curve that respects the letter's structure.
        
        Args:
            x_points: Original x coordinates
            y_points: Original y coordinates
            
        Returns:
            tuple: (enhanced_x, enhanced_y) with additional constraint points
        """
        if len(x_points) < 3:
            return x_points, y_points
        
        # Add points between existing points to encourage smooth curves
        enhanced_x = list(x_points)
        enhanced_y = list(y_points)
        
        for i in range(len(x_points) - 1):
            # Add midpoint with interpolated y value
            mid_x = (x_points[i] + x_points[i + 1]) / 2
            mid_y = (y_points[i] + y_points[i + 1]) / 2
            
            enhanced_x.append(mid_x)
            enhanced_y.append(mid_y)
        
        # Sort by x coordinate
        sort_idx = np.argsort(enhanced_x)
        enhanced_x = np.array(enhanced_x)[sort_idx]
        enhanced_y = np.array(enhanced_y)[sort_idx]
        
        return enhanced_x, enhanced_y
    
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
        SHAPE-ACCURACY STRATEGY: Polynomials are extremely accurate at letter points.
        
        The key insight: We only care about accuracy AT the letter centerline points.
        Behavior elsewhere (between points, outside letter) doesn't matter.
        
        Args:
            contour (np.array): Array of (x, y) centerline points (hundreds of them)
            max_degree (int): Maximum polynomial degree for shape fitting
            
        Returns:
            list: List of polynomial function strings optimized for letter shape
        """
        if len(contour) < 2:
            return []
        
        print(f"    SHAPE-ACCURACY STRATEGY: {len(contour)} centerline points to fit")
        
        # With hundreds of centerline points, we need more curves to capture detail
        # But each curve uses fewer points for better polynomial stability
        points_per_curve = min(20, max(8, len(contour) // 25))  # 8-20 points per curve
        num_curves = max(10, len(contour) // points_per_curve)  # More curves for detail
        
        print(f"    Generating {num_curves} curves with ~{points_per_curve} points each")
        print("    Focus: Maximum accuracy at letter centerline points")
        
        functions = []
        
        # Generate NON-OVERLAPPING curves to minimize visual clutter
        # Each curve will be restricted to its own domain
        for curve_idx in range(num_curves):
            # Minimal or no overlap to prevent curves crossing each other
            start_ratio = curve_idx / num_curves
            end_ratio = (curve_idx + 1) / num_curves
            
            # Add tiny overlap only for curve continuity (1% of curve length)
            if curve_idx > 0:
                overlap = 0.01 / num_curves
                start_ratio = max(0.0, start_ratio - overlap)
            
            if curve_idx < num_curves - 1:
                overlap = 0.01 / num_curves  
                end_ratio = min(1.0, end_ratio + overlap)
            
            # Extract points for this curve region
            curve_points = self.generate_curve_from_region(contour, start_ratio, end_ratio, points_per_curve)
            
            if len(curve_points) < 3:
                continue
            
            # Fit polynomial optimized for shape accuracy at these specific points
            func = self.fit_polynomial_to_segment(curve_points, max_degree)
            
            if func:
                functions.append(func)
                print(f"      Curve {curve_idx+1}: Shape-optimized for {len(curve_points)} centerline points")
            else:
                print(f"      Curve {curve_idx+1}: Failed to create shape-optimized polynomial")
        
        print(f"    Total shape-accurate functions: {len(functions)}")
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
            except Exception:
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
