#!/usr/bin/env python3
"""
Debug script to investigate why letter "A" doesn't look correct.
This script will analyze the letter extraction and polynomial fitting process.
"""

import numpy as np
import matplotlib.pyplot as plt
from main import TextToDesmos

def debug_letter_A():
    """Debug the letter A generation process step by step."""
    print("=== DEBUG: Letter A Analysis ===")
    
    # Create converter
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    text = "A"
    
    print(f"1. Converting text '{text}' to paths...")
    
    # Step 1: Extract paths
    paths = converter.text_to_paths(text, font_size=100)
    print(f"   Generated {len(paths)} character paths")
    
    if not paths:
        print("ERROR: No paths generated!")
        return
    
    path = paths[0]  # Get the "A" path
    print(f"   Path has {len(path.vertices)} vertices")
    print(f"   Path codes: {path.codes[:10] if path.codes is not None else 'None'}...")
    
    # Step 2: Extract contours
    print(f"\n2. Extracting contour points...")
    contours = converter.extract_contour_points(path, num_points=50)
    print(f"   Found {len(contours)} contours")
    
    for i, contour in enumerate(contours):
        print(f"   Contour {i+1}: {len(contour)} points")
        print(f"      X range: {np.min(contour[:, 0]):.2f} to {np.max(contour[:, 0]):.2f}")
        print(f"      Y range: {np.min(contour[:, 1]):.2f} to {np.max(contour[:, 1]):.2f}")
    
    # Step 3: Analyze contour properties
    print(f"\n3. Analyzing contour properties for polynomial fitting...")
    
    for i, contour in enumerate(contours):
        print(f"\n   Contour {i+1} analysis:")
        x_data = contour[:, 0]
        y_data = contour[:, 1]
        
        # Check function properties
        x_unique = np.unique(x_data)
        y_unique = np.unique(y_data)
        
        print(f"      Total points: {len(contour)}")
        print(f"      Unique X values: {len(x_unique)} ({len(x_unique)/len(x_data)*100:.1f}%)")
        print(f"      Unique Y values: {len(y_unique)} ({len(y_unique)/len(y_data)*100:.1f}%)")
        
        # Check if it can be a function
        can_be_y_of_x = converter._has_function_property(x_data, y_data)
        can_be_x_of_y = converter._has_function_property(y_data, x_data)
        
        print(f"      Can be y=f(x): {can_be_y_of_x}")
        print(f"      Can be x=f(y): {can_be_x_of_y}")
        
        # Check contour segments
        segments = converter._split_contour_into_functional_segments(contour)
        print(f"      Split into {len(segments)} segments")
        
        for j, segment in enumerate(segments):
            print(f"         Segment {j+1}: {len(segment)} points")
    
    # Step 4: Generate functions
    print(f"\n4. Generating polynomial functions...")
    functions = converter.text_to_desmos_functions(text, max_degree=12)
    
    print(f"   Generated {len(functions)} functions")
    for i, func in enumerate(functions):
        print(f"   {i+1}. {func}")
    
    # Step 5: Visualize the original contours
    print(f"\n5. Creating visualization...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original path vertices
    plt.subplot(2, 3, 1)
    if len(path.vertices) > 0:
        vertices = path.vertices
        plt.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2, label='Original path')
        plt.scatter(vertices[::5, 0], vertices[::5, 1], c='red', s=20, alpha=0.7)
    plt.title(f'Original Path Vertices\n({len(path.vertices)} points)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Extracted contours
    plt.subplot(2, 3, 2)
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        plt.plot(contour[:, 0], contour[:, 1], 'o-', color=color, 
                label=f'Contour {i+1} ({len(contour)} pts)', markersize=3)
    plt.title(f'Extracted Contours\n({len(contours)} contours)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Contour segments
    plt.subplot(2, 3, 3)
    segment_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    segment_count = 0
    
    for i, contour in enumerate(contours):
        segments = converter._split_contour_into_functional_segments(contour)
        for j, segment in enumerate(segments):
            color = segment_colors[segment_count % len(segment_colors)]
            plt.plot(segment[:, 0], segment[:, 1], 'o-', color=color, 
                    label=f'C{i+1}S{j+1} ({len(segment)})', markersize=2)
            segment_count += 1
    
    plt.title(f'Contour Segments\n({segment_count} segments)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: X vs Y scatter for first contour
    if contours:
        plt.subplot(2, 3, 4)
        contour = contours[0]
        plt.scatter(contour[:, 0], contour[:, 1], c=range(len(contour)), cmap='viridis', s=30)
        plt.colorbar(label='Point order')
        plt.title('First Contour Point Order')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Function property analysis
    if contours:
        plt.subplot(2, 3, 5)
        contour = contours[0]
        x_data = contour[:, 0]
        y_data = contour[:, 1]
        
        # Sort by x and plot
        sort_idx = np.argsort(x_data)
        plt.plot(x_data[sort_idx], y_data[sort_idx], 'b-', label='Sorted by X')
        plt.scatter(x_data[sort_idx], y_data[sort_idx], c='red', s=20, alpha=0.7)
        
        plt.title('Y vs X (sorted by X)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate') 
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Plot 6: Check for multi-valued functions
    if contours:
        plt.subplot(2, 3, 6)
        contour = contours[0]
        x_data = contour[:, 0]
        y_data = contour[:, 1]
        
        # Find x values that appear multiple times
        x_unique, counts = np.unique(x_data, return_counts=True)
        multi_x = x_unique[counts > 1]
        
        plt.hist(x_data, bins=20, alpha=0.7, label=f'X distribution')
        if len(multi_x) > 0:
            plt.axvline(multi_x[0], color='red', linestyle='--', 
                       label=f'{len(multi_x)} repeated X values')
        
        plt.title('X Value Distribution')
        plt.xlabel('X coordinate')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('debug_letter_A_analysis.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved as 'debug_letter_A_analysis.png'")
    
    # Step 6: Detailed function analysis
    print(f"\n6. Detailed function analysis...")
    
    if functions:
        print(f"\n   Function details:")
        for i, func in enumerate(functions):
            print(f"\n   Function {i+1}: {func}")
            
            # Try to extract some info about the polynomial
            if 'x^' in func:
                max_power = 0
                parts = func.split('x^')
                for part in parts[1:]:
                    try:
                        power = int(part.split()[0].split('*')[0].split('+')[0].split('-')[0])
                        max_power = max(max_power, power)
                    except:
                        pass
                print(f"      Maximum degree: {max_power}")
            
            # Count terms
            term_count = func.count('+') + func.count('-') + 1
            print(f"      Number of terms: {term_count}")
    
    else:
        print("   ERROR: No functions were generated!")
        
        # Try to understand why
        print(f"\n   Debugging why no functions were generated...")
        for i, contour in enumerate(contours):
            print(f"\n   Trying contour {i+1} manually...")
            try:
                functions_manual = converter.fit_polynomial_contour_tracing(contour, max_degree=12)
                print(f"      Manual attempt generated {len(functions_manual)} functions")
                for j, func in enumerate(functions_manual):
                    print(f"         {j+1}. {func[:100]}...")
            except Exception as e:
                print(f"      Manual attempt failed: {e}")

if __name__ == "__main__":
    debug_letter_A()
