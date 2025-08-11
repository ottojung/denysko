#!/usr/bin/env python3
"""
Debug the rasterization process step by step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg

def debug_rasterization_steps():
    """Debug each step of the rasterization process."""
    print("Debugging rasterization process step by step...")
    
    # Create path for letter A
    font_prop = font_manager.FontProperties(size=100)
    text_path = TextPath((0, 0), 'A', prop=font_prop)
    
    # Get path bounds
    vertices = text_path.vertices
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    bounds = (min_x, min_y, max_x, max_y)
    
    print(f"Letter bounds: {bounds}")
    
    # Test the path.contains_point method for different points
    print("\nTesting path.contains_point:")
    test_points = [
        (0, 0),  # Bottom left corner (should be False)
        (33.79, 36.45),  # Center (should be False for A hole)
        (33.79, 20),  # Lower center (should be True)
        (67.58, 72.91),  # Top right corner (should be False)
    ]
    
    for x, y in test_points:
        contains = text_path.contains_point((x, y))
        print(f"  Point ({x:.2f}, {y:.2f}): {contains}")
    
    # Now test the rasterization process step by step
    print("\nRasterization process:")
    resolution = 400
    
    # Create figure and axis
    dpi = 100
    fig_width = resolution / dpi
    fig_height = resolution / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.axis('off')
    
    print(f"Figure size: {fig_width} x {fig_height} inches at {dpi} DPI")
    print(f"Axes limits: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
    
    # Create the path patch
    patch = PathPatch(text_path, facecolor='white', edgecolor='red', linewidth=2)
    ax.add_patch(patch)
    ax.set_facecolor('black')
    
    # Save the intermediate rendering
    plt.savefig('debug_intermediate_render.png', dpi=dpi, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    print("Intermediate rendering saved as 'debug_intermediate_render.png'")
    
    # Now get the buffer
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    
    # Convert to numpy array
    rgba_array = np.frombuffer(buf, dtype=np.uint8).reshape(resolution, resolution, 4)
    
    print(f"RGBA array shape: {rgba_array.shape}")
    print(f"R channel range: [{rgba_array[:,:,0].min()}, {rgba_array[:,:,0].max()}]")
    print(f"G channel range: [{rgba_array[:,:,1].min()}, {rgba_array[:,:,1].max()}]")
    print(f"B channel range: [{rgba_array[:,:,2].min()}, {rgba_array[:,:,2].max()}]")
    print(f"A channel range: [{rgba_array[:,:,3].min()}, {rgba_array[:,:,3].max()}]")
    
    # Check what colors we actually have
    unique_r = np.unique(rgba_array[:,:,0])
    print(f"Unique R values: {unique_r}")
    
    # Create mask using different thresholds
    masks = {}
    thresholds = [0, 127, 200, 254]
    
    for thresh in thresholds:
        mask = rgba_array[:, :, 0] > thresh
        mask_flipped = np.flipud(mask)
        masks[thresh] = mask_flipped
        filled = np.sum(mask_flipped)
        print(f"Threshold {thresh}: {filled} filled pixels ({filled/160000*100:.1f}%)")
    
    plt.close(fig)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show the RGBA channels
    for i, channel in enumerate(['R', 'G', 'B', 'A']):
        ax = axes[i]
        ax.imshow(rgba_array[:,:,i], cmap='gray', origin='upper')
        ax.set_title(f'{channel} Channel')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Show masks with different thresholds
    ax = axes[4]
    ax.imshow(masks[127], cmap='gray', origin='lower', 
             extent=[min_x, max_x, min_y, max_y])
    ax.set_title('Mask (threshold=127)')
    ax.set_aspect('equal')
    
    ax = axes[5]
    ax.imshow(masks[254], cmap='gray', origin='lower',
             extent=[min_x, max_x, min_y, max_y])
    ax.set_title('Mask (threshold=254)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('debug_rasterization_channels.png', dpi=150, bbox_inches='tight')
    print("Channel analysis saved as 'debug_rasterization_channels.png'")
    
    return masks

if __name__ == "__main__":
    debug_rasterization_steps()
