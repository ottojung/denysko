#!/usr/bin/env python3
"""
Simplified TextExtractor - main orchestrator class for text to centerline extraction.
"""

from matplotlib.path import Path as MPLPath

from .font_utils import text_to_paths
from .centerline_extraction import extract_skeleton_from_path, upsample_centerline
from .preview_utils import preview_extracted_points, preview_skeleton_extraction_steps


class TextExtractor:
    """Main class for extracting centerlines from text using horizontal-monotonic decomposition."""

    def text_to_paths(self, text, font_size=100):
        """Convert text to matplotlib Path objects."""
        return text_to_paths(text, font_size)

    def simplify_path_for_zero_stroke(self, path):
        """
        Extract the centerline from a font path.
        This creates a skeletal representation using the horizontal-monotonic algorithm.
        """
        if len(path.vertices) == 0:
            return None

        # Convert path to centerline points
        skeleton_points = self.extract_skeleton_from_path(path)

        if len(skeleton_points) < 3:
            return path  # Fallback to original if skeleton extraction fails

        # Create a new path from the skeleton points
        codes = [MPLPath.MOVETO] + [MPLPath.LINETO] * (len(skeleton_points) - 1)
        return MPLPath(skeleton_points, codes)

    def extract_skeleton_from_path(self, path):
        """Extract centerline using horizontal-monotonic component decomposition."""
        return extract_skeleton_from_path(path)

    def extract_contour_points(self, path, num_points=500):
        """Extract centerline points and resample to num_points."""
        if not path or len(path.vertices) == 0:
            return []
        vertices = path.vertices
        codes = path.codes
        pts = self.extract_centerline_from_path(vertices, codes, num_points)
        return [pts] if len(pts) > 0 else []

    def extract_centerline_from_path(self, vertices, codes, num_points=500):
        """Build a Path from vertices/codes, extract skeleton, and resample to num_points."""
        if len(vertices) < 3:
            return vertices

        temp = MPLPath(vertices, codes)
        skel = self.extract_skeleton_from_path(temp)
        if len(skel) < 3:
            # Fallback to simple approximation
            step = max(1, len(vertices) // 10)
            simplified = vertices[::step]
            if len(simplified) < 3:
                simplified = vertices[[0, len(vertices) // 2, -1]]
            skel = simplified

        # Resample to target number of points
        if len(skel) >= num_points:
            idx = range(0, len(skel), max(1, len(skel) // num_points))
            return skel[list(idx)[:num_points]]

        return upsample_centerline(skel, num_points)

    def upsample_centerline(self, points, target_count):
        """Upsample centerline to target_count via arc-length interpolation."""
        return upsample_centerline(points, target_count)

    def preview_extracted_points(
        self, text, font_size=100, num_points=500, save_path=None
    ):
        """Generate preview showing outline and centerline."""
        preview_extracted_points(self, text, font_size, num_points, save_path)

    def preview_skeleton_extraction_steps(self, text, font_size=100, save_path=None):
        """Generate detailed skeleton extraction preview."""
        preview_skeleton_extraction_steps(self, text, font_size, save_path)
