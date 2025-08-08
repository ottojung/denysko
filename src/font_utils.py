#!/usr/bin/env python3
"""
Font utilities for text rendering.
"""

from matplotlib import font_manager
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


def get_font_path():
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


def text_to_paths(text, font_size=100):
    """
    Convert text to Path objects using ZERO stroke width (outline only).
    This eliminates thickness and creates clean single-line letter shapes.

    Args:
        text (str): Input text
        font_size (int): Font size for rendering

    Returns:
        list: List of simplified Path objects for each character (stroke width = 0)
    """
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
                # Keep original outline path for preview; do not skeletonize here
                paths.append(path)

            # Calculate character width for next character positioning
            bbox = path.get_extents()
            char_width = bbox.width if bbox.width > 0 else font_size * 0.5
            x_offset += char_width + font_size * 0.05  # Small spacing between chars

        except Exception as e:
            print(f"Warning: Could not process character '{char}': {e}")
            x_offset += font_size * 0.5  # Default spacing

    return paths
