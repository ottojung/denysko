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
        SIMPLIFIED VERSION - just looks for major x-direction reversals.
        
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
        
        # Find points where x direction changes significantly
        segments = []
        current_segment = [contour[0]]
        
        # Track x direction: 1 for increasing, -1 for decreasing, 0 for unchanged
        current_direction = 0
        
        for i in range(1, len(contour)):
            prev_x = current_segment[-1][0]
            curr_x = contour[i][0]
            
            x_diff = curr_x - prev_x
            
            if abs(x_diff) < 1e-6:  # Essentially same x
                current_segment.append(contour[i])
                continue
            
            new_direction = 1 if x_diff > 0 else -1
            
            # If direction changes significantly, start new segment
            if current_direction != 0 and new_direction != current_direction:
                # Save current segment if it has enough points
                if len(current_segment) >= 3:
                    segments.append(np.array(current_segment))
                
                # Start new segment with overlap for continuity
                current_segment = [current_segment[-1], contour[i]]
                current_direction = new_direction
            else:
                current_segment.append(contour[i])
                if current_direction == 0:
                    current_direction = new_direction
        
        # Add the final segment
        if len(current_segment) >= 3:
            segments.append(np.array(current_segment))
        elif len(segments) > 0 and len(current_segment) > 0:
            # Merge short final segment with last segment
            segments[-1] = np.vstack([segments[-1], current_segment[1:]])
        
        # If no segments were created, return the whole contour
        if not segments:
            segments = [contour]
        
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
            
            # Generate function string (defined over all real numbers)
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
                return f"y = {func_str}"
            
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

    def fit_contour_polynomials(self, contour, max_degree=15):
        """
        COMPLETELY REWRITTEN FROM SCRATCH:
        Two guiding principles:
        1. Exact fitting to as many points as possible
        2. Separation into multiple curves when multiple strokes cover same horizontal domain
        
        Args:
            contour (np.array): Array of (x, y) centerline points
            max_degree (int): Maximum polynomial degree for fitting
            
        Returns:
            list: List of polynomial function strings that exactly fit the letter points
        """
        if len(contour) < 3:
            return []
        
        print(f"    REWRITTEN ALGORITHM: Processing {len(contour)} letter points")
        
        # Step 1: Detect horizontal overlap (multiple y values at same x)
        overlapping_regions = self.detect_horizontal_overlap(contour)
        
        if overlapping_regions:
            print(f"    Detected {len(overlapping_regions)} overlapping horizontal regions")
            functions = []
            
            # Fit each overlapping region separately 
            for i, region in enumerate(overlapping_regions):
                print(f"    Region {i+1}: {len(region)} points")
                func = self.fit_exact_polynomial(region, max_degree)
                if func:
                    functions.append(func)
            
            return functions
        else:
            print("    No horizontal overlap detected - fitting single curve")
            # No overlap: fit one curve to all points
            func = self.fit_exact_polynomial(contour, max_degree)
            return [func] if func else []
    
    def detect_horizontal_overlap(self, contour):
        """
        Detect regions where multiple strokes occupy the same horizontal space.
        This is the key to separating letter "A" into multiple curves.
        
        Args:
            contour: Array of (x, y) points
            
        Returns:
            list: List of point arrays, each representing a separate stroke
        """
        if len(contour) < 6:  # Need enough points to detect overlap
            return None
        
        # Sort points by x coordinate
        sorted_indices = np.argsort(contour[:, 0])
        sorted_points = contour[sorted_indices]
        
        # Group points by x-coordinate ranges to find overlaps
        x_coords = sorted_points[:, 0]
        y_coords = sorted_points[:, 1]
        
        # Find x-ranges where there are multiple y values (indicating overlap)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        x_range = x_max - x_min
        
        if x_range < 1e-6:  # All points at same x
            return None
        
        # Divide x-range into bins and check for multiple y values in each bin
        num_bins = min(20, len(contour) // 3)
        bin_edges = np.linspace(x_min, x_max, num_bins + 1)
        
        overlap_detected = False
        for i in range(num_bins):
            bin_mask = (x_coords >= bin_edges[i]) & (x_coords <= bin_edges[i + 1])
            if np.sum(bin_mask) >= 2:  # At least 2 points in this x-range
                y_in_bin = y_coords[bin_mask]
                y_span = np.max(y_in_bin) - np.min(y_in_bin)
                
                # If y-span is significant, we have overlap
                if y_span > x_range * 0.1:  # 10% of x-range
                    overlap_detected = True
                    break
        
        if not overlap_detected:
            return None
        
        # Separate into upper and lower strokes based on y-coordinate
        y_median = np.median(y_coords)
        
        upper_stroke = []
        lower_stroke = []
        
        for point in sorted_points:
            if point[1] >= y_median:
                upper_stroke.append(point)
            else:
                lower_stroke.append(point)
        
        strokes = []
        if len(upper_stroke) >= 3:
            strokes.append(np.array(upper_stroke))
        if len(lower_stroke) >= 3:
            strokes.append(np.array(lower_stroke))
        
        return strokes if len(strokes) >= 2 else None
    
    def fit_exact_polynomial(self, points, max_degree):
        """
        Fit a polynomial that passes exactly through as many points as possible.
        Uses exact interpolation when possible, high-degree fitting otherwise.
        
        Args:
            points: Array of (x, y) points to fit
            max_degree: Maximum polynomial degree
            
        Returns:
            str: Polynomial function string that fits the points exactly
        """
        if len(points) < 3:
            return None
        
        # Sort by x and handle duplicate x values
        x_sorted, y_sorted = self.sort_segment_by_x(points)
        
        if len(x_sorted) < 3:
            return None
        
        print(f"      Fitting {len(x_sorted)} unique points")
        
        # Choose degree for exact fitting
        n_points = len(x_sorted)
        
        # Try exact interpolation first (degree = n-1)
        if n_points <= max_degree + 1:
            degree = n_points - 1
        else:
            # Use maximum degree for best approximation
            degree = max_degree
        
        # Ensure degree > 1
        if degree <= 1:
            degree = 2  # Force at least quadratic
        
        try:
            # Fit polynomial
            coeffs = np.polyfit(x_sorted, y_sorted, degree)
            
            # Verify accuracy
            poly_func = np.poly1d(coeffs)
            errors = np.abs(poly_func(x_sorted) - y_sorted)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            print(f"      Degree {degree}: max_error={max_error:.6f}, mean_error={mean_error:.6f}")
            
            # If error is too high and we can increase degree, try higher degree
            if max_error > 0.01 and degree < min(max_degree, n_points - 1):
                degree = min(max_degree, n_points - 1)
                coeffs = np.polyfit(x_sorted, y_sorted, degree)
                poly_func = np.poly1d(coeffs)
                max_error = np.max(np.abs(poly_func(x_sorted) - y_sorted))
                print(f"      Improved to degree {degree}: max_error={max_error:.6f}")
            
            # Convert to function string
            func_str = self.coefficients_to_function_string(coeffs)
            if func_str:
                print("      ✓ Generated exact-fitting polynomial")
                return func_str
            
        except Exception as e:
            print(f"      ✗ Polynomial fitting failed: {e}")
        
        return None
    
    return None
    
    def _get_degree_from_function(self, func_str):
    
    def fit_high_accuracy_polynomial(self, segment, max_degree):
        """
        Fit polynomial with degree > 1 that matches ALL points as accurately as possible.
        
        Args:
            segment: Points to fit
            max_degree: Maximum degree allowed
            
        Returns:
            str: Polynomial function string with degree > 1
        """
        if len(segment) < 3:
            return None
        
        try:
            # Sort and clean data
            x_sorted, y_sorted = self.sort_segment_by_x(segment)
            
            if len(x_sorted) < 3:
                return None
            
            # Requirement: degree > 1 (minimum quadratic)
            min_degree = 2  # quadratic minimum
            max_feasible_degree = min(max_degree, len(x_sorted) - 1)
            
            if max_feasible_degree < min_degree:
                return None  # Cannot satisfy degree > 1 requirement
            
            # Use degree that can match all points well
            optimal_degree = min(max_feasible_degree, max(min_degree, len(x_sorted) // 2))
            
            print(f"        Using degree {optimal_degree} for {len(x_sorted)} points")
            
            # Fit polynomial to match ALL points as accurately as possible
            coeffs = np.polyfit(x_sorted, y_sorted, optimal_degree)
            
            # Verify it's degree > 1
            if optimal_degree <= 1:
                return None
            
            # Verify accuracy at all points
            poly_func = np.poly1d(coeffs)
            errors = np.abs(poly_func(x_sorted) - y_sorted)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            print(f"        Accuracy: max_error={max_error:.3f}, mean_error={mean_error:.3f}")
            
            # Convert to function string
            func_str = self.coefficients_to_function_string(coeffs)
            return func_str
            
        except Exception as e:
            print(f"        Error fitting high-accuracy polynomial: {e}")
            return None
    
    def coefficients_to_function_string(self, coeffs):
        """
        Convert polynomial coefficients to a function string.
        Ensures degree > 1 by design.
        
        Args:
            coeffs: Polynomial coefficients (highest degree first)
            
        Returns:
            str: Function string in form "y = ..."
        """
        if len(coeffs) < 3:  # Degree must be > 1 
            return None
            
        # Filter out essentially zero coefficients but keep structure
        filtered_coeffs = []
        for coeff in coeffs:
            if abs(coeff) < 1e-12:
                filtered_coeffs.append(0.0)
            else:
                filtered_coeffs.append(coeff)
        
        terms = []
        degree = len(filtered_coeffs) - 1
        
        for i, coeff in enumerate(filtered_coeffs):
            power = degree - i
            
            if abs(coeff) < 1e-12:  # Skip zero coefficients
                continue
            
            # Format the coefficient
            coeff_str = f"{coeff:.8f}"
            
            # Build the term
            if power == 0:
                # Constant term
                if len(terms) > 0:
                    terms.append(" + " + coeff_str if coeff >= 0 else " - " + coeff_str[1:])
                else:
                    terms.append(coeff_str)
            elif power == 1:
                # Linear term
                if abs(coeff - 1.0) < 1e-12:
                    terms.append(" + x" if len(terms) > 0 else "x")
                elif abs(coeff + 1.0) < 1e-12:
                    terms.append(" - x")
                else:
                    if len(terms) > 0:
                        terms.append(" + " + coeff_str + "*x" if coeff >= 0 else " - " + coeff_str[1:] + "*x")
                    else:
                        terms.append(coeff_str + "*x")
            else:
                # Higher degree terms
                if abs(coeff - 1.0) < 1e-12:
                    terms.append(f" + x^{power}" if len(terms) > 0 else f"x^{power}")
                elif abs(coeff + 1.0) < 1e-12:
                    terms.append(f" - x^{power}")
                else:
                    if len(terms) > 0:
                        terms.append(f" + {coeff_str}*x^{power}" if coeff >= 0 else f" - {coeff_str[1:]}*x^{power}")
                    else:
                        terms.append(f"{coeff_str}*x^{power}")
        
        if not terms:
            return "y = 0"
        
        func_str = "y = " + "".join(terms)
        return func_str
    
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
