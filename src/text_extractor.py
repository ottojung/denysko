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

    def extract_skeleton_from_path(self, path, num_walks=25, step_distance=3, max_steps=100):
        """Extract skeleton from a path, returning list of separate traces."""
        return extract_skeleton_from_path(path, num_walks, step_distance, max_steps)

    def extract_contour_points(self, path, num_points=500):
        """Extract centerline traces from path."""
        if not path or len(path.vertices) == 0:
            return []
        
        traces = self.extract_skeleton_from_path(path)
        return traces

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
