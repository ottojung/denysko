#!/usr/bin/env python3
"""
Main entry point for the text to Desmos converter.
Only generates y = f(x) functions.
"""

from .text_to_desmos import TextToDesmos


def main():
    """Main function to demonstrate the text to Desmos conversion."""
    print("=== Text to Desmos Polynomial Converter ===")
    print("This program generates ONLY y = f(x) functions")
    print()

    # Test with letter A
    print("Testing with letter 'A'...")

    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    functions = converter.text_to_desmos_functions("A")

    print(f"\nGenerated {len(functions)} functions for letter 'A':")
    print()
    for i, func in enumerate(functions, 1):
        print(f" {func}")

    print()

    # Verify all functions are y = f(x)
    y_functions = [f for f in functions if f.startswith("y =")]
    x_functions = [f for f in functions if f.startswith("x =")]

    print("\nVerification:")
    print(f"y = f(x) functions: {len(y_functions)}")
    print(f"x = f(y) functions: {len(x_functions)} (should be 0)")

    if len(x_functions) > 0:
        print("ERROR: Found x = f(y) functions!")
        for func in x_functions:
            print(f"{func}")
    else:
        print("SUCCESS: All functions are y = f(x)")

    # Save to file
    converter.save_functions(functions, "letter_A_functions.txt")

    return functions


if __name__ == "__main__":
    main()
