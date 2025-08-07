#!/usr/bin/env python3
"""
Text extraction module - handles font path extraction and contour point sampling.
"""

import numpy as np
from matplotlib import font_manager
from matplotlib.path import Path


class TextExtractor:
    """Handles extracting character paths and contour points from text."""

    def __init__(self):
        self.font_path = self._get_font_path()

    def _get_font_path(self):
        """Get a suitable font path for text rendering."""
        try:
            # Try to find a common system font
            fonts = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
            for font_name in fonts:
                font_path = font_manager.findfont(
                    font_manager.FontProperties(family=font_name)
                )
                if font_path:
                    return font_path
        except Exception:
            pass

        # Fallback to default font
        return font_manager.findfont(font_manager.FontProperties())

    def text_to_paths(self, text, font_size=100):
        """
        Convert text to Path objects using ZERO stroke width (outline only).
        This eliminates thickness and creates clean single-line letter shapes.

        Args:
            text (str): Input text
            font_size (int): Font size for rendering

        Returns:
            list: List of simplified Path objects for each character (stroke width = 0)
        """
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties

        paths = []
        x_offset = 0

        # Use a font that works well for outlines
        font_props = FontProperties(family="sans-serif", weight="normal")

        for char in text:
            if char == " ":
                x_offset += font_size * 0.3  # Space width
                continue

            try:
                # Create text path with specific font properties for clean outlines
                path = TextPath((x_offset, 0), char, size=font_size, prop=font_props)

                if len(path.vertices) > 0:
                    # Simplify the path to reduce thickness artifacts
                    simplified_path = self.simplify_path_for_zero_stroke(path)
                    if simplified_path is not None:
                        paths.append(simplified_path)

                # Calculate character width for next character positioning
                bbox = path.get_extents()
                char_width = bbox.width if bbox.width > 0 else font_size * 0.5
                x_offset += char_width + font_size * 0.05  # Small spacing between chars

            except Exception as e:
                print(f"Warning: Could not process character '{char}': {e}")
                x_offset += font_size * 0.5  # Default spacing

        return paths

    def simplify_path_for_zero_stroke(self, path):
        """
        Extract the TRUE CENTERLINE (medial axis) from a font path.
        This creates a skeletal representation using only the midpoints of the letter strokes.

        Args:
            path: matplotlib Path object

        Returns:
            Path: Skeletonized path representing only the centerline
        """
        if len(path.vertices) == 0:
            return None

        # Convert path to a rasterized representation for skeletonization
        skeleton_points = self.extract_skeleton_from_path(path)

        if len(skeleton_points) < 3:
            return path  # Fallback to original if skeleton extraction fails

        # Create a new path from the skeleton points
        from matplotlib.path import Path as MPLPath

        # Create path codes for the skeleton (all LINETO except first MOVETO)
        codes = [MPLPath.MOVETO] + [MPLPath.LINETO] * (len(skeleton_points) - 1)

        return MPLPath(skeleton_points, codes)

    def extract_skeleton_from_path(self, path):
        """
        Extract the skeleton (medial axis) from a font path using a midpoint-based approach.
        This finds the true centerline of the letter strokes.

        Args:
            path: matplotlib Path object

        Returns:
            np.array: Array of (x, y) points representing the skeleton/centerline
        """
        vertices = path.vertices

        # Get the bounding box
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)

        # Create a sampling grid to find interior points
        width = max_x - min_x
        height = max_y - min_y

        if width <= 0 or height <= 0:
            return vertices

        # Sample points across the letter shape
        resolution = 50  # Grid resolution
        x_samples = np.linspace(min_x, max_x, resolution)
        y_samples = np.linspace(min_y, max_y, resolution)

        skeleton_points = []

        # For each y-level, find the midpoints between left and right boundaries
        for y in y_samples:
            x_intersections = self.find_x_intersections_at_y(path, y)

            if len(x_intersections) >= 2:
                # Sort intersections and pair them to find interior segments
                x_intersections = sorted(x_intersections)

                # Take pairs of intersections and find their midpoints
                for i in range(0, len(x_intersections) - 1, 2):
                    if i + 1 < len(x_intersections):
                        left_x = x_intersections[i]
                        right_x = x_intersections[i + 1]

                        # Calculate midpoint
                        mid_x = (left_x + right_x) / 2
                        skeleton_points.append([mid_x, y])

        # Similarly, for each x-level, find midpoints between top and bottom boundaries
        for x in x_samples:
            y_intersections = self.find_y_intersections_at_x(path, x)

            if len(y_intersections) >= 2:
                y_intersections = sorted(y_intersections)

                for i in range(0, len(y_intersections) - 1, 2):
                    if i + 1 < len(y_intersections):
                        bottom_y = y_intersections[i]
                        top_y = y_intersections[i + 1]

                        mid_y = (bottom_y + top_y) / 2
                        skeleton_points.append([x, mid_y])

        if not skeleton_points:
            # Fallback: use simplified outline
            return self.simplify_outline_to_centerline(vertices)

        # Remove duplicate points and sort by connectivity
        skeleton_points = np.array(skeleton_points)
        skeleton_points = self.remove_duplicate_points(skeleton_points)
        skeleton_points = self.sort_points_by_connectivity(skeleton_points)

        return skeleton_points

    def find_x_intersections_at_y(self, path, y):
        """
        Find x-coordinates where a horizontal line at y intersects the path.

        Args:
            path: matplotlib Path object
            y: y-coordinate of horizontal line

        Returns:
            list: List of x-coordinates of intersections
        """
        vertices = path.vertices
        intersections = []

        # Check each edge of the path
        for i in range(len(vertices) - 1):
            x1, y1 = vertices[i]
            x2, y2 = vertices[i + 1]

            # Check if the edge crosses the horizontal line y
            if ((y1 <= y <= y2) or (y2 <= y <= y1)) and y1 != y2:
                # Calculate intersection point
                t = (y - y1) / (y2 - y1)
                x_intersect = x1 + t * (x2 - x1)
                intersections.append(x_intersect)

        return intersections

    def find_y_intersections_at_x(self, path, x):
        """
        Find y-coordinates where a vertical line at x intersects the path.

        Args:
            path: matplotlib Path object
            x: x-coordinate of vertical line

        Returns:
            list: List of y-coordinates of intersections
        """
        vertices = path.vertices
        intersections = []

        # Check each edge of the path
        for i in range(len(vertices) - 1):
            x1, y1 = vertices[i]
            x2, y2 = vertices[i + 1]

            # Check if the edge crosses the vertical line x
            if ((x1 <= x <= x2) or (x2 <= x <= x1)) and x1 != x2:
                # Calculate intersection point
                t = (x - x1) / (x2 - x1)
                y_intersect = y1 + t * (y2 - y1)
                intersections.append(y_intersect)

        return intersections

    def simplify_outline_to_centerline(self, vertices):
        """
        Fallback method: simplify outline to approximate centerline.

        Args:
            vertices: Array of path vertices

        Returns:
            np.array: Simplified centerline points
        """
        if len(vertices) < 4:
            return vertices

        # Use every nth point to create a simplified centerline
        step = max(1, len(vertices) // 20)
        simplified = vertices[::step]

        return simplified

    def remove_duplicate_points(self, points, tolerance=1e-6):
        """
        Remove duplicate points from an array.

        Args:
            points: Array of (x, y) points
            tolerance: Distance tolerance for considering points duplicate

        Returns:
            np.array: Array with duplicates removed
        """
        if len(points) <= 1:
            return points

        unique_points = [points[0]]

        for point in points[1:]:
            # Check if this point is too close to any existing point
            is_duplicate = False
            for existing in unique_points:
                if np.linalg.norm(point - existing) < tolerance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_points.append(point)

        return np.array(unique_points)

    def sort_points_by_connectivity(self, points):
        """
        Sort points to create a connected path (simple nearest-neighbor ordering).

        Args:
            points: Array of (x, y) points

        Returns:
            np.array: Points sorted for connectivity
        """
        if len(points) <= 2:
            return points

        # Start with first point
        sorted_points = [points[0]]
        remaining = list(points[1:])

        while remaining:
            current = sorted_points[-1]

            # Find nearest remaining point
            distances = [np.linalg.norm(current - p) for p in remaining]
            nearest_idx = np.argmin(distances)

            sorted_points.append(remaining.pop(nearest_idx))

        return np.array(sorted_points)

    def extract_path_contours(self, path):
        """
        Extract individual contours from a path.

        Args:
            path: matplotlib Path object

        Returns:
            list: List of contour dictionaries with 'vertices' and 'codes'
        """
        vertices = path.vertices
        codes = path.codes

        if codes is None:
            # If no codes, treat as single contour
            return [{"vertices": vertices, "codes": None}]

        contours = []
        current_vertices = []
        current_codes = []

        for i, (vertex, code) in enumerate(zip(vertices, codes)):
            from matplotlib.path import Path as MPLPath

            if code == MPLPath.MOVETO and current_vertices:
                # Start of new contour, save previous one
                contours.append(
                    {
                        "vertices": np.array(current_vertices),
                        "codes": np.array(current_codes),
                    }
                )
                current_vertices = []
                current_codes = []

            current_vertices.append(vertex)
            current_codes.append(code)

        # Add the final contour
        if current_vertices:
            contours.append(
                {
                    "vertices": np.array(current_vertices),
                    "codes": np.array(current_codes),
                }
            )

        return contours

    def find_main_contour(self, contours):
        """
        Find the main (largest area) contour from a list of contours.
        This represents the outer boundary for zero stroke width.

        Args:
            contours (list): List of contour dictionaries

        Returns:
            dict: Main contour dictionary, or None if no suitable contour found
        """
        if not contours:
            return None

        if len(contours) == 1:
            return contours[0]

        # Calculate area for each contour using shoelace formula
        main_contour = None
        max_area = 0

        for contour in contours:
            vertices = contour["vertices"]
            if len(vertices) < 3:
                continue

            # Calculate area using shoelace formula
            area = (
                abs(
                    sum(
                        vertices[i][0] * vertices[(i + 1) % len(vertices)][1]
                        - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]
                        for i in range(len(vertices))
                    )
                )
                / 2.0
            )

            if area > max_area:
                max_area = area
                main_contour = contour

        return main_contour

    def extract_contour_points(self, path, num_points=500):
        """
        Extract HUNDREDS of points along the MIDPOINT/CENTERLINE of the letter path.
        This creates a zero-thickness trace through the center of the letter strokes.

        Args:
            path: matplotlib Path object
            num_points (int): Number of points to extract (default: 500 for high detail)

        Returns:
            list: List of contour arrays, each representing the centerline of the letter
        """
        if not path or len(path.vertices) == 0:
            return []

        vertices = path.vertices
        codes = path.codes

        print(f"    Extracting {num_points} centerline points from letter path...")

        # Extract the centerline by tracing through the middle of the letter strokes
        centerline_points = self.extract_centerline_from_path(
            vertices, codes, num_points
        )

        if len(centerline_points) < 10:
            print(f"    Warning: Only {len(centerline_points)} centerline points found")
            return [centerline_points] if len(centerline_points) > 0 else []

        print(f"    SUCCESS: Generated {len(centerline_points)} centerline points")
        return [centerline_points]

    def extract_centerline_from_path(self, vertices, codes, num_points=500):
        """
        Extract TRUE CENTERLINE points by finding midpoints between opposing boundaries.
        This creates a skeletal representation with zero thickness.

        Args:
            vertices: Path vertices
            codes: Path codes
            num_points (int): Target number of points to generate

        Returns:
            np.array: Array of (x, y) points representing the true letter centerline
        """
        if len(vertices) < 3:
            return np.array(vertices)

        print(
            f"        SKELETON EXTRACTION: Finding true centerline from {len(vertices)} outline points"
        )

        # Create a temporary path object for the skeleton extraction
        from matplotlib.path import Path as MPLPath

        temp_path = MPLPath(vertices, codes)

        # Extract skeleton using the new midpoint-based approach
        skeleton_points = self.extract_skeleton_from_path(temp_path)

        if len(skeleton_points) < 3:
            # Fallback to simplified outline approach
            print(
                f"        FALLBACK: Using simplified outline ({len(skeleton_points)} skeleton points)"
            )
            skeleton_points = self.simplify_outline_to_centerline(vertices)

        # Resample to get exactly the requested number of points
        if len(skeleton_points) >= num_points:
            # Downsample using even spacing
            indices = np.linspace(0, len(skeleton_points) - 1, num_points, dtype=int)
            resampled_points = skeleton_points[indices]
        else:
            # Upsample by interpolating between existing points
            resampled_points = self.upsample_centerline(skeleton_points, num_points)

        print(
            f"        SKELETON SUCCESS: Generated {len(resampled_points)} true centerline points"
        )
        return resampled_points

    def upsample_centerline(self, points, target_count):
        """
        Upsample a centerline by interpolating between existing points.

        Args:
            points: Existing centerline points
            target_count: Desired number of points

        Returns:
            np.array: Upsampled centerline points
        """
        if len(points) < 2:
            return points

        # Calculate cumulative distances along the centerline
        distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)

        total_length = distances[-1]
        if total_length == 0:
            return points

        # Create evenly spaced target distances
        target_distances = np.linspace(0, total_length, target_count)

        # Interpolate points at target distances
        upsampled_points = []
        for target_dist in target_distances:
            # Find the segment containing this distance
            idx = np.searchsorted(distances, target_dist)
            if idx == 0:
                upsampled_points.append(points[0])
            elif idx >= len(points):
                upsampled_points.append(points[-1])
            else:
                # Interpolate between points[idx-1] and points[idx]
                t = (target_dist - distances[idx - 1]) / (
                    distances[idx] - distances[idx - 1]
                )
                interpolated = points[idx - 1] + t * (points[idx] - points[idx - 1])
                upsampled_points.append(interpolated)

        return np.array(upsampled_points)

    def extract_path_segments(self, vertices, codes):
        """
        Extract continuous segments from path vertices and codes.

        Args:
            vertices: Path vertices
            codes: Path codes

        Returns:
            list: List of segments, each being an array of points
        """
        from matplotlib.path import Path as MPLPath

        if codes is None:
            return [vertices]

        segments = []
        current_segment = []

        for i, (vertex, code) in enumerate(zip(vertices, codes)):
            if code == MPLPath.MOVETO:
                if current_segment:
                    segments.append(np.array(current_segment))
                current_segment = [vertex]
            elif code in [MPLPath.LINETO, MPLPath.CURVE3, MPLPath.CURVE4]:
                current_segment.append(vertex)
            elif code == MPLPath.CLOSEPOLY:
                if current_segment and len(current_segment) > 1:
                    current_segment.append(current_segment[0])  # Close the path
                    segments.append(np.array(current_segment))
                current_segment = []

        # Add final segment
        if current_segment and len(current_segment) > 1:
            segments.append(np.array(current_segment))

        return segments

    def calculate_segment_length(self, segment):
        """
        Calculate the total length of a path segment.

        Args:
            segment: Array of (x, y) points

        Returns:
            float: Total length of the segment
        """
        if len(segment) < 2:
            return 0

        distances = np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1))
        return np.sum(distances)

    def interpolate_along_segment(self, segment, ratio):
        """
        Interpolate a point at a given ratio along a segment.

        Args:
            segment: Array of (x, y) points
            ratio: Position along segment (0.0 to 1.0)

        Returns:
            np.array: Interpolated (x, y) point
        """
        if len(segment) < 2:
            return segment[0] if len(segment) > 0 else np.array([0, 0])

        if ratio <= 0:
            return segment[0]
        if ratio >= 1:
            return segment[-1]

        # Calculate cumulative distances
        distances = np.cumsum(np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)

        total_length = distances[-1]
        if total_length == 0:
            return segment[0]

        # Find target distance
        target_distance = ratio * total_length

        # Find the segment index
        idx = np.searchsorted(distances, target_distance)
        if idx == 0:
            return segment[0]
        if idx >= len(segment):
            return segment[-1]

        # Interpolate between segment[idx-1] and segment[idx]
        seg_start = distances[idx - 1]
        seg_end = distances[idx]

        if seg_end == seg_start:
            return segment[idx - 1]

        local_ratio = (target_distance - seg_start) / (seg_end - seg_start)

        return segment[idx - 1] + local_ratio * (segment[idx] - segment[idx - 1])
