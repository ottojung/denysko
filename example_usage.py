#!/usr/bin/env python3
"""
Example usage of the Text to Desmos Polynomial Plotter

This file demonstrates how to use the TextToDesmos class programmatically
with different parameters and configurations.
"""

from main import TextToDesmos
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

def example_basic_usage():
    """Basic example with default settings."""
    print("=" * 50)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 50)
    
    # Create converter with default settings
    converter = TextToDesmos()
    
    # Convert a simple word
    text = "HI"
    functions = converter.text_to_desmos_functions(
        text=text,
        font_size=60,
        points_per_char=30,
        max_degree=3
    )
    
    # Save to file
    converter.save_functions(functions, "example1_hi.txt")
    
    print(f"Generated {len(functions)} functions for '{text}'")
    print(f"Saved to 'example1_hi.txt'")
    return functions

def example_custom_position_scale():
    """Example with custom origin and scale."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Custom Position and Scale")
    print("=" * 50)
    
    # Create converter with custom origin and scale
    converter = TextToDesmos(origin=(5, -3), scale=2.0)
    
    # Convert text
    text = "OK"
    functions = converter.text_to_desmos_functions(
        text=text,
        font_size=40,
        points_per_char=25,
        max_degree=4
    )
    
    # Save to file
    converter.save_functions(functions, "example2_ok_scaled.txt")
    
    print(f"Generated {len(functions)} functions for '{text}'")
    print(f"Positioned at origin (5, -3) with scale factor 2.0")
    print(f"Saved to 'example2_ok_scaled.txt'")
    return functions

def example_single_letter_detailed():
    """Detailed example with a single letter and high precision."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Single Letter with High Detail")
    print("=" * 50)
    
    # Create converter
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    
    # Convert a single letter with high detail
    text = "A"
    functions = converter.text_to_desmos_functions(
        text=text,
        font_size=100,
        points_per_char=60,
        max_degree=5
    )
    
    # Save to file
    converter.save_functions(functions, "example3_letter_A_detailed.txt")
    
    print(f"Generated {len(functions)} functions for letter '{text}'")
    print(f"Used high detail settings: 60 points per char, degree 5 polynomials")
    print(f"Saved to 'example3_letter_A_detailed.txt'")
    
    # Print first few functions as examples
    print("\nFirst 3 functions:")
    for i, func in enumerate(functions[:3], 1):
        print(f"{i}. {func}")
    
    return functions

def example_simple_word():
    """Example with a simple word optimized for Desmos."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Simple Word (Optimized for Desmos)")
    print("=" * 50)
    
    # Create converter
    converter = TextToDesmos(origin=(-2, 0), scale=1.5)
    
    # Convert a word with optimized settings for Desmos
    text = "MATH"
    functions = converter.text_to_desmos_functions(
        text=text,
        font_size=50,
        points_per_char=35,
        max_degree=3  # Lower degree for simpler functions
    )
    
    # Save to file
    converter.save_functions(functions, "example4_math_optimized.txt")
    
    print(f"Generated {len(functions)} functions for '{text}'")
    print(f"Optimized settings: lower degree polynomials for better Desmos compatibility")
    print(f"Positioned at origin (-2, 0) with scale 1.5")
    print(f"Saved to 'example4_math_optimized.txt'")
    
    return functions

def create_desmos_instructions():
    """Create instructions for using the generated functions in Desmos."""
    instructions = """
# How to Use Generated Functions in Desmos

## Step 1: Copy Functions
- Open one of the generated .txt files (e.g., example1_hi.txt)
- Copy the functions (lines that start with 'y =' or 'x =')

## Step 2: Open Desmos
- Go to https://www.desmos.com/calculator
- Create a new graph

## Step 3: Paste Functions
- Click the + button to add new expressions
- Paste each function into a separate expression box
- Desmos will automatically plot each function

## Step 4: Adjust View
- Use the zoom controls to fit the text in view
- The text should appear as the outline you specified

## Tips for Better Results:
1. Start with simple, short text (1-3 characters)
2. Use lower polynomial degrees (3-4) for smoother curves
3. Adjust the origin and scale to position text where you want
4. If functions don't display properly, try reducing max_degree
5. For complex letters, you might need more polynomial segments

## Function Format Explanation:
- y = (polynomial) * {domain constraint}
- x = (polynomial) * {domain constraint}
- The domain constraints ensure functions only display in the correct regions
- Multiple functions combine to form the complete character outline

## Example Function:
y = (-0.123456*x^3 + 2.345678*x^2 - 1.234567*x + 0.987654) * {x >= 5.000 AND x <= 15.000}

This creates a cubic polynomial that's only visible between x=5 and x=15.
"""
    
    with open("DESMOS_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("Created 'DESMOS_INSTRUCTIONS.md' with detailed usage instructions")

def main():
    """Run all examples."""
    print("Text to Desmos Converter - Example Usage")
    print("Creating multiple examples with different configurations...")
    
    # Run examples
    example_basic_usage()
    example_custom_position_scale()
    example_single_letter_detailed()
    example_simple_word()
    
    # Create instructions
    create_desmos_instructions()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print("Files created:")
    print("- example1_hi.txt")
    print("- example2_ok_scaled.txt") 
    print("- example3_letter_A_detailed.txt")
    print("- example4_math_optimized.txt")
    print("- DESMOS_INSTRUCTIONS.md")
    print("\nTo use with Desmos:")
    print("1. Open any .txt file")
    print("2. Copy the functions (y = ... or x = ... lines)")
    print("3. Paste them into Desmos graphing calculator")
    print("4. Adjust zoom to see the text outline")

if __name__ == "__main__":
    main()
