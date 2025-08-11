#!/usr/bin/env python3
"""
Debug mask alignment with letter shape.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg
import random

def rasterize_path(path, bounds, resolution=400):
    """Convert a matplotlib Path to a rasterized binary mask using proper rendering."""
    min_x, min_y, max_x, max_y = bounds
    
    # Create figure and axis with exact pixel dimensions
    dpi = 100
    fig_width = resolution / dpi
    fig_height = resolution / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Render the path as a filled patch
    patch = PathPatch(path, facecolor='white', edgecolor='none')
    ax.add_patch(patch)
    ax.set_facecolor('black')  # Background is black (False in mask)
    
    # Render to bitmap
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    
    # Convert to numpy array and extract binary mask
    rgba_array = np.frombuffer(buf, dtype=np.uint8).reshape(resolution, resolution, 4)
    
    # White pixels (R=G=B=255) are foreground (True in mask)
    mask = rgba_array[:, :, 0] > 127  # White pixels
    
    # Flip vertically because matplotlib has origin at bottom-left
    mask = np.flipud(mask)
    
    plt.close(fig)
    
    return mask

def debug_mask_alignment():
    """Debug the alignment between mask and actual letter shape."""
    print("Debugging mask alignment with letter shape...")
    
    # Create path for letter A
    font_prop = font_manager.FontProperties(size=100)
    text_path = TextPath((0, 0), 'A', prop=font_prop)
    
    # Get path bounds
    vertices = text_path.vertices
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    bounds = (min_x, min_y, max_x, max_y)
    
    print(f"Letter bounds: {bounds}")
    print(f"Width: {max_x - min_x:.2f}, Height: {max_y - min_y:.2f}")
    
    # Rasterize the path
    mask = rasterize_path(text_path, bounds, resolution=400)
    print(f"Mask shape: {mask.shape}")
    print(f"Filled pixels: {np.sum(mask)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original letter outline
    ax1 = axes[0]
    vertices_list = []
    for path in text_path.iter_segments():
        if len(path) == 2:
            vertex, code = path
            if len(vertex) == 2:
                vertices_list.append(vertex)
    
    if vertices_list:
        vertices_array = np.array(vertices_list)
        ax1.plot(vertices_array[:, 0], vertices_array[:, 1], 'b-', linewidth=2, label='Letter outline')
        ax1.fill(vertices_array[:, 0], vertices_array[:, 1], alpha=0.3, color='blue')
    
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Letter Shape')
    ax1.legend()
    
    # Plot 2: Rasterized mask
    ax2 = axes[1]
    # Create coordinate arrays for the mask
    height, width = mask.shape
    x_coords = np.linspace(min_x, max_x, width)
    y_coords = np.linspace(min_y, max_y, height)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Show the mask
    ax2.imshow(mask, extent=[min_x, max_x, min_y, max_y], 
              origin='lower', cmap='gray', alpha=0.8)
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Rasterized Mask')
    
    # Plot 3: Overlay comparison
    ax3 = axes[2]
    # Show mask as background
    ax3.imshow(mask, extent=[min_x, max_x, min_y, max_y], 
              origin='lower', cmap='Reds', alpha=0.5, label='Mask')
    
    # Overlay letter outline
    if vertices_list:
        ax3.plot(vertices_array[:, 0], vertices_array[:, 1], 'b-', linewidth=2, label='Letter outline')
    
    ax3.set_xlim(min_x, max_x)
    ax3.set_ylim(min_y, max_y)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Mask vs Letter Overlay')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('debug_mask_alignment_A.png', dpi=150, bbox_inches='tight')
    print("Mask alignment visualization saved as 'debug_mask_alignment_A.png'")
    
    # Test specific points
    print("\nTesting specific points:")
    
    # Test center point
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Convert to mask coordinates
    height, width = mask.shape
    mask_x = int((center_x - min_x) / (max_x - min_x) * width)
    mask_y = int((center_y - min_y) / (max_y - min_y) * height)
    
    print(f"Center point: ({center_x:.2f}, {center_y:.2f})")
    print(f"Mask coordinates: ({mask_x}, {mask_y})")
    print(f"Mask value at center: {mask[mask_y, mask_x]}")
    
    # Test if the center should be inside (for letter A, center should be in the hole = False)
    path_contains = text_path.contains_point((center_x, center_y))
    print(f"Path.contains_point at center: {path_contains}")
    
    # Test corners of the bounding box
    corners = [
        (min_x, min_y), (max_x, min_y), 
        (min_x, max_y), (max_x, max_y)
    ]
    
    for i, (x, y) in enumerate(corners):
        mask_x = int((x - min_x) / (max_x - min_x) * width)
        mask_y = int((y - min_y) / (max_y - min_y) * height)
        mask_x = max(0, min(width - 1, mask_x))
        mask_y = max(0, min(height - 1, mask_y))
        
        mask_val = mask[mask_y, mask_x]
        path_val = text_path.contains_point((x, y))
        
        print(f"Corner {i+1} ({x:.2f}, {y:.2f}): mask={mask_val}, path={path_val}")

if __name__ == "__main__":
    debug_mask_alignment()
