#!/usr/bin/env python3
"""
Simple test script to analyze letter A generation without user input.
"""

import numpy as np
from main import TextToDesmos

def test_letter_A():
    """Test letter A generation and save results to file."""
    
    # Redirect all output to a file
    output_lines = []
    
    def log(message):
        print(message)
        output_lines.append(str(message))
    
    log("=== LETTER A ANALYSIS ===")
    
    # Create converter
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    text = "A"
    
    log(f"Testing text: '{text}'")
    
    try:
        # Step 1: Extract paths
        paths = converter.text_to_paths(text, font_size=100)
        log(f"Generated {len(paths)} character paths")
        
        if not paths:
            log("ERROR: No paths generated!")
            return
        
        path = paths[0]
        log(f"Path has {len(path.vertices)} vertices")
        
        # Step 2: Extract contours
        contours = converter.extract_contour_points(path, num_points=50)
        log(f"Found {len(contours)} contours")
        
        for i, contour in enumerate(contours):
            log(f"  Contour {i+1}: {len(contour)} points")
            log(f"    X range: {np.min(contour[:, 0]):.2f} to {np.max(contour[:, 0]):.2f}")
            log(f"    Y range: {np.min(contour[:, 1]):.2f} to {np.max(contour[:, 1]):.2f}")
            
            # Check function properties
            x_data = contour[:, 0]
            y_data = contour[:, 1]
            
            x_unique = np.unique(x_data)
            y_unique = np.unique(y_data)
            
            can_be_y_of_x = converter._has_function_property(x_data, y_data)
            can_be_x_of_y = converter._has_function_property(y_data, x_data)
            
            log(f"    Unique X values: {len(x_unique)}/{len(x_data)} ({len(x_unique)/len(x_data)*100:.1f}%)")
            log(f"    Unique Y values: {len(y_unique)}/{len(y_data)} ({len(y_unique)/len(y_data)*100:.1f}%)")
            log(f"    Can be y=f(x): {can_be_y_of_x}")
            log(f"    Can be x=f(y): {can_be_x_of_y}")
            
            # Check segments
            segments = converter._split_contour_into_functional_segments(contour)
            log(f"    Split into {len(segments)} segments")
            
            for j, segment in enumerate(segments):
                x_seg = segment[:, 0]
                y_seg = segment[:, 1]
                x_range = np.max(x_seg) - np.min(x_seg)
                y_range = np.max(y_seg) - np.min(y_seg)
                log(f"      Segment {j+1}: {len(segment)} points, x_range={x_range:.2f}, y_range={y_range:.2f}")
        
        # Step 3: Try to generate functions manually for each contour
        log("\n=== MANUAL FUNCTION GENERATION ===")
        all_functions_manual = []
        
        for i, contour in enumerate(contours):
            log(f"\nTesting contour {i+1} manually...")
            try:
                functions = converter.fit_polynomial_contour_tracing(contour, max_degree=12)
                log(f"  Generated {len(functions)} functions")
                for j, func in enumerate(functions):
                    log(f"    {j+1}. {func}")
                    all_functions_manual.append(func)
            except Exception as e:
                log(f"  ERROR in manual generation: {e}")
        
        # Step 4: Test full pipeline
        log("\n=== FULL PIPELINE TEST ===")
        try:
            functions_full = converter.text_to_desmos_functions(text, max_degree=12)
            log(f"Full pipeline generated {len(functions_full)} functions")
            for i, func in enumerate(functions_full):
                log(f"  {i+1}. {func}")
        except Exception as e:
            log(f"ERROR in full pipeline: {e}")
        
        # Step 5: Analyze specific issues
        log("\n=== ISSUE ANALYSIS ===")
        
        if len(all_functions_manual) == 0:
            log("PROBLEM: No functions generated at all!")
            
            # Check why each contour failed
            for i, contour in enumerate(contours):
                log(f"\nDebugging contour {i+1}:")
                x_data = contour[:, 0]
                y_data = contour[:, 1]
                
                # Check data validity
                log(f"  Data validity: X has {len(x_data)} points, Y has {len(y_data)} points")
                log(f"  X contains NaN: {np.any(np.isnan(x_data))}")
                log(f"  Y contains NaN: {np.any(np.isnan(y_data))}")
                log(f"  X contains Inf: {np.any(np.isinf(x_data))}")
                log(f"  Y contains Inf: {np.any(np.isinf(y_data))}")
                
                # Try simple polynomial fitting
                try:
                    if len(x_data) >= 3:
                        # Try fitting y = ax^2 + bx + c
                        coeffs = np.polyfit(x_data, y_data, min(2, len(x_data)-1))
                        log(f"  Simple polyfit succeeded: coeffs = {coeffs}")
                    else:
                        log(f"  Too few points for polyfit: {len(x_data)}")
                except Exception as e:
                    log(f"  Simple polyfit failed: {e}")
        
        else:
            log(f"Generated {len(all_functions_manual)} functions, but they may not trace the letter correctly.")
            log("POSSIBLE ISSUES:")
            log("1. Contour segmentation may be splitting the letter too much")
            log("2. Polynomial degrees may be too low to capture letter complexity")
            log("3. Function property detection may be too restrictive")
            log("4. Letter 'A' has complex geometry (triangle + crossbar) that's hard to fit")
        
        # Save first few points of each contour for inspection
        log("\n=== CONTOUR POINT SAMPLES ===")
        for i, contour in enumerate(contours):
            log(f"\nContour {i+1} first 10 points:")
            for j in range(min(10, len(contour))):
                x, y = contour[j]
                log(f"  Point {j+1}: ({x:.3f}, {y:.3f})")
    
    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        import traceback
        log("Traceback:")
        log(traceback.format_exc())
    
    # Save all output to file
    with open('letter_A_analysis.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nAnalysis complete. Results saved to 'letter_A_analysis.txt'")
    print(f"Total output lines: {len(output_lines)}")

if __name__ == "__main__":
    test_letter_A()
