#!/usr/bin/env python3
"""
Main entry point with user interface.
Simplified and modular approach.
"""

from text_to_desmos import TextToDesmos


def main():
    """Main function with user interface."""
    print("Text to Desmos Polynomial Converter (Simplified)")
    print("=" * 50)
    print("Features:")
    print("- Pure coordinate-based point fitting")
    print("- Only generates y = f(x) functions")
    print("- Modular architecture")
    print("=" * 50)
    
    # Get user input
    text = input("Enter text to convert: ").strip()
    if not text:
        text = "A"
        print(f"Using default text: {text}")
    
    try:
        origin_input = input("Enter origin point (x,y) [default: 0,0]: ").strip()
        if origin_input:
            origin = tuple(map(float, origin_input.split(",")))
        else:
            origin = (0, 0)
    except Exception:
        origin = (0, 0)
        print("Using default origin: (0, 0)")
    
    try:
        scale_input = input("Enter scale factor [default: 1.0]: ").strip()
        scale = float(scale_input) if scale_input else 1.0
    except Exception:
        scale = 1.0
        print("Using default scale: 1.0")
    
    try:
        degree_input = input("Enter max polynomial degree [default: 6]: ").strip()
        max_degree = int(degree_input) if degree_input else 6
    except Exception:
        max_degree = 6
        print("Using default max degree: 6")
    
    # Create converter and generate functions
    converter = TextToDesmos(origin=origin, scale=scale, max_degree=max_degree)
    functions = converter.text_to_desmos_functions(text)
    
    # Display results
    print("\n" + "=" * 50)
    print("DESMOS FUNCTIONS (y = f(x) only)")
    print("=" * 50)
    print("Copy and paste these functions into Desmos:")
    print()
    
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func}")
    
    # Save to file
    converter.save_functions(functions)
    
    print(f"\nGenerated {len(functions)} functions for text: '{text}'")
    print("Functions saved to 'desmos_functions.txt'")


if __name__ == "__main__":
    main()
