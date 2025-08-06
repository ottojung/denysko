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
        font_props = FontProperties(family='sans-serif', weight='normal')
        
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
        Simplify a font path to simulate zero stroke width by extracting the main outline.
        This removes inner contours and thickness artifacts.
        
        Args:
            path: matplotlib Path object
            
        Returns:
            Path: Simplified path with zero stroke width effect
        """
        if len(path.vertices) == 0:
            return None
        
        # Group path segments into contours
        contours = self.extract_path_contours(path)
        
        if not contours:
            return path
        
        # For zero stroke width, we want only the main outer contour
        # Find the contour with the largest area (usually the outer boundary)
        main_contour = self.find_main_contour(contours)
        
        if main_contour is None:
            return path
        
        # Create a new simplified path from the main contour
        from matplotlib.path import Path as MPLPath
        
        # Use only the outer contour vertices and codes
        simplified_vertices = main_contour['vertices']
        simplified_codes = main_contour['codes']
        
        return MPLPath(simplified_vertices, simplified_codes)
    
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
            return [{'vertices': vertices, 'codes': None}]
        
        contours = []
        current_vertices = []
        current_codes = []
        
        for i, (vertex, code) in enumerate(zip(vertices, codes)):
            from matplotlib.path import Path as MPLPath
            
            if code == MPLPath.MOVETO and current_vertices:
                # Start of new contour, save previous one
                contours.append({
                    'vertices': np.array(current_vertices),
                    'codes': np.array(current_codes)
                })
                current_vertices = []
                current_codes = []
            
            current_vertices.append(vertex)
            current_codes.append(code)
        
        # Add the final contour
        if current_vertices:
            contours.append({
                'vertices': np.array(current_vertices),
                'codes': np.array(current_codes)
            })
        
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
            vertices = contour['vertices']
            if len(vertices) < 3:
                continue
            
            # Calculate area using shoelace formula
            area = abs(sum(
                vertices[i][0] * vertices[(i + 1) % len(vertices)][1] -
                vertices[(i + 1) % len(vertices)][0] * vertices[i][1]
                for i in range(len(vertices))
            )) / 2.0
            
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
        centerline_points = self.extract_centerline_from_path(vertices, codes, num_points)
        
        if len(centerline_points) < 10:
            print(f"    Warning: Only {len(centerline_points)} centerline points found")
            return [centerline_points] if len(centerline_points) > 0 else []
        
        print(f"    SUCCESS: Generated {len(centerline_points)} centerline points")
        return [centerline_points]
    
    def extract_centerline_from_path(self, vertices, codes, num_points=500):
        """
        Extract centerline points from a letter path by tracing the midpoint of strokes.
        This creates a zero-thickness representation of the letter.
        
        Args:
            vertices: Path vertices
            codes: Path codes
            num_points (int): Target number of points to generate
            
        Returns:
            np.array: Array of (x, y) points representing the letter centerline
        """
        if len(vertices) < 3:
            return np.array(vertices)
        
        # First, get all path segments
        segments = self.extract_path_segments(vertices, codes)
        
        if not segments:
            return np.array(vertices[:num_points] if len(vertices) > num_points else vertices)
        
        # Calculate total path length
        total_length = sum(self.calculate_segment_length(seg) for seg in segments)
        
        if total_length == 0:
            return np.array(vertices[:num_points] if len(vertices) > num_points else vertices)
        
        # Generate evenly spaced points along the centerline
        centerline_points = []
        target_distances = np.linspace(0, total_length, num_points)
        
        current_distance = 0
        segment_idx = 0
        
        for target_dist in target_distances:
            # Find which segment contains this target distance
            while segment_idx < len(segments) and current_distance + self.calculate_segment_length(segments[segment_idx]) < target_dist:
                current_distance += self.calculate_segment_length(segments[segment_idx])
                segment_idx += 1
            
            if segment_idx >= len(segments):
                segment_idx = len(segments) - 1
            
            # Interpolate within the current segment
            segment = segments[segment_idx]
            segment_length = self.calculate_segment_length(segment)
            
            if segment_length == 0:
                centerline_points.append(segment[0])
                continue
            
            # How far into this segment should we be?
            remaining_dist = target_dist - current_distance
            ratio = remaining_dist / segment_length if segment_length > 0 else 0
            ratio = np.clip(ratio, 0, 1)
            
            # Interpolate along this segment
            point = self.interpolate_along_segment(segment, ratio)
            centerline_points.append(point)
        
        return np.array(centerline_points)
    
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
        distances = np.cumsum(
            np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1))
        )
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