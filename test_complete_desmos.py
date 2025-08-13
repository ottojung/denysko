#!/usr/bin/env python3
"""Test complete letter generation with new Desmos syntax."""

from src.text_to_desmos import TextToDesmos

def test_complete_generation():
    """Test complete letter generation with proper Desmos syntax."""
    print("=== Testing Complete Generation with New Syntax ===")
    
    letter = "O"  # Simple circular letter for quick testing
    print(f"Generating letter '{letter}'...")
    
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    functions = converter.text_to_desmos_functions(letter)
    
    print(f"\nGenerated {len(functions)} functions:")
    for i, func in enumerate(functions, 1):
        print(f"{i}: {func}")
        
        # Verify syntax
        has_desmos_syntax = '\\left\\{' in func and '\\right\\}' in func and '\\le' in func
        print(f"   Correct Desmos syntax: {has_desmos_syntax}")
    
    # Save to file
    filename = f"letter_{letter}_desmos_syntax.txt"
    converter.save_functions(functions, filename)
    print(f"\nSaved to {filename}")

if __name__ == "__main__":
    test_complete_generation()
