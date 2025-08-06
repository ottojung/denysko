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
    
    def extract_contour_points(self, path, num_points=50):
        """
        Extract points along the contour of a path.
        
        Args:
            path: matplotlib Path object
            num_points (int): Number of points to extract per contour
            
        Returns:
            list: List of (x, y) coordinate arrays for each contour
        """
        if len(path.vertices) == 0:
            return []
        
        # Get path vertices and codes
        vertices = path.vertices
        codes = path.codes
        
        contours = []
        current_contour = []
        
        i = 0
        while i < len(vertices):
            if codes is None or codes[i] == Path.MOVETO:
                # Start new contour
                if current_contour:
                    contours.append(np.array(current_contour))
                current_contour = [vertices[i]]
            elif codes[i] == Path.LINETO:
                current_contour.append(vertices[i])
            elif codes[i] == Path.CLOSEPOLY:
                if current_contour:
                    contours.append(np.array(current_contour))
                current_contour = []
            i += 1
        
        # Add the last contour if it exists
        if current_contour:
            contours.append(np.array(current_contour))
        
        # Resample each contour to have approximately num_points
        resampled_contours = []
        for contour in contours:
            if len(contour) < 3:  # Skip very small contours
                continue
            
            # Calculate cumulative distances along the contour
            distances = np.cumsum(
                np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
            )
            distances = np.insert(distances, 0, 0)
            
            # Create evenly spaced parameters
            total_length = distances[-1]
            if total_length == 0:
                continue
            
            even_distances = np.linspace(0, total_length, num_points)
            
            # Interpolate to get evenly spaced points
            x_interp = np.interp(even_distances, distances, contour[:, 0])
            y_interp = np.interp(even_distances, distances, contour[:, 1])
            
            resampled_contour = np.column_stack([x_interp, y_interp])
            resampled_contours.append(resampled_contour)
        
        return resampled_contours