#!/usr/bin/env python3
"""
Text to Desmos Polynomial Plotter - Main Entry Point

This program converts text into polynomial approximations that can be plotted on Desmos.
Uses the modular structure with src/ subdirectory.
Only generates y = f(x) functions.
"""

from src.text_to_desmos import TextToDesmos


def main():
    """Main function to demonstrate the text to Desmos conversion."""
    print("=== Text to Desmos Polynomial Converter ===")
    print("Using NEW STRUCTURAL STROKE EXTRACTION")
    print("This program generates ONLY y = f(x) functions")
    print()
    
    # Test with letter A using new structural approach
    print("Testing with letter 'A' using structural extraction...")
    
    try:
        converter = TextToDesmos(origin=(0, 0), scale=1.0, max_degree=6)
        functions = converter.text_to_desmos_functions("A")
        
        print(f"\nGenerated {len(functions)} functions for letter 'A':")
        for i, func in enumerate(functions, 1):
            print(f"{i}. {func}")
        
        # Verify all functions are y = f(x)
        y_functions = [f for f in functions if f.startswith("y =")]
        x_functions = [f for f in functions if f.startswith("x =")]
        
        print(f"\nVerification:")
        print(f"y = f(x) functions: {len(y_functions)}")
        print(f"x = f(y) functions: {len(x_functions)} (should be 0)")
        
        if len(x_functions) > 0:
            print("ERROR: Found x = f(y) functions!")
            for func in x_functions:
                print(f"  {func}")
        else:
            print("SUCCESS: All functions are y = f(x)")
        
        # Test the structural extraction directly
        print("\n=== TESTING STRUCTURAL EXTRACTION ===")
        from src.text_extractor import TextExtractor
        extractor = TextExtractor()
        
        paths = extractor.text_to_paths("A", font_size=100)
        if paths:
            path = paths[0]
            print(f"Letter 'A' outline has {len(path.vertices)} vertices")
            
            # Test key point detection
            key_points = extractor.find_letter_key_points(path.vertices)
            print(f"Found {len(key_points)} key structural points:")
            for i, point in enumerate(key_points):
                print(f"  Point {i+1}: ({point[0]:.1f}, {point[1]:.1f})")
            
            # Test stroke creation
            if len(key_points) >= 3:
                strokes = extractor.create_structural_strokes(key_points)
                print(f"Created {len(strokes)} structural strokes:")
                for i, stroke in enumerate(strokes):
                    start = stroke['start']
                    end = stroke['end']
                    stroke_type = stroke['type']
                    length = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
                    print(f"  Stroke {i+1} ({stroke_type}): ({start[0]:.1f},{start[1]:.1f}) -> ({end[0]:.1f},{end[1]:.1f}) [length: {length:.1f}]")
                    
                print("STRUCTURAL ANALYSIS: This should show 3 clean strokes for letter 'A'!")
            else:
                print("WARNING: Not enough key points found for letter structure")
        
        # Save to file
        converter.save_functions(functions, "letter_A_functions.txt")
        
        return functions
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    main()
