#!/usr/bin/env python3
"""
Main entry point for the text to Desmos converter.
Only generates y = f(x) functions.
"""

from .text_to_desmos import TextToDesmos
import json


def main():
    """Main function to demonstrate the text to Desmos conversion."""
    print("=== Text to Desmos Polynomial Converter ===")
    print("This program generates ONLY y = f(x) functions")
    print()

    text = "C"

    print(f"Testing with letter {json.dumps(text)}...")

    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    functions = converter.text_to_desmos_functions(text)

    print(f"\nGenerated {len(functions)} functions for letter {json.dumps(text)}:")
    print()
    for i, func in enumerate(functions, 1):
        print(f"{func}")

    print()

    # Save to file
    converter.save_functions(functions, f"letter_{text}_functions.txt")

    return functions


if __name__ == "__main__":
    main()
