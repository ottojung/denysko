#!/usr/bin/env python3
"""
Test script for the improved modular algorithm
"""

from text_to_desmos import TextToDesmos


def test_letter_A():
    """Test the improved algorithm with letter A."""
    print("=== TESTING IMPROVED MODULAR ALGORITHM ===")
    print()
    print("Improvements implemented:")
    print("1. Split into multiple source files for modularity")
    print("2. Pure coordinate-based point fitting (no semantic analysis)")
    print("3. Only generates y = f(x) functions")
    print("4. Simplified polynomial fitting approach")
    print()
    
    # Test with letter A
    converter = TextToDesmos(origin=(0, 0), scale=1.0, max_degree=6)
    
    print("Testing with letter 'A'...")
    functions = converter.text_to_desmos_functions("A")
    
    print("\n" + "=" * 50)
    print("RESULTS FOR LETTER 'A'")
    print("=" * 50)
    
    if functions:
        print(f"Generated {len(functions)} y = f(x) functions:")
        print()
        for i, func in enumerate(functions, 1):
            print(f"{i}. {func}")
        
        # Save results
        converter.save_functions(functions, "test_letter_A.txt")
        print(f"\nFunctions saved to 'test_letter_A.txt'")
    else:
        print("ERROR: No functions were generated!")
    
    return functions


if __name__ == "__main__":
    test_letter_A()
