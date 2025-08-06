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
        Convert text to matplotlib Path objects representing character outlines.
        
        Args:
            text (str): Input text to convert
            font_size (int): Font size for rendering
            
        Returns:
            list: List of Path objects for each character
        """
        from matplotlib.textpath import TextPath
        
        paths = []
        x_offset = 0
        
        for char in text:
            if char == " ":
                x_offset += font_size * 0.3  # Space width
                continue
            
            try:
                # Create text path for the character
                path = TextPath((x_offset, 0), char, size=font_size)
                if len(path.vertices) > 0:
                    paths.append(path)
                
                # Calculate character width for next character positioning
                bbox = path.get_extents()
                char_width = bbox.width if bbox.width > 0 else font_size * 0.5
                x_offset += char_width + font_size * 0.05  # Small spacing between chars
                
            except Exception as e:
                print(f"Warning: Could not process character '{char}': {e}")
                x_offset += font_size * 0.5  # Default spacing
        
        return paths
    
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