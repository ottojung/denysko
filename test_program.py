#!/usr/bin/env python3
"""
Simple test script for the Text to Desmos Polynomial Plotter

This script performs basic functionality tests to ensure the program works correctly.
"""

import sys
import os
from main import TextToDesmos

def test_basic_functionality():
    """Test basic functionality with a simple character."""
    print("Testing basic functionality...")
    
    try:
        # Create converter
        converter = TextToDesmos(origin=(0, 0), scale=1.0)
        
        # Test with a simple character
        functions = converter.text_to_desmos_functions(
            text="A",
            font_size=50,
            points_per_char=20,
            max_degree=3
        )
        
        if len(functions) > 0:
            print(f"✓ Successfully generated {len(functions)} functions for 'A'")
            return True
        else:
            print("✗ No functions generated")
            return False
            
    except Exception as e:
        print(f"✗ Error during basic test: {e}")
        return False

def test_parameter_variations():
    """Test different parameter combinations."""
    print("Testing parameter variations...")
    
    test_cases = [
        {"text": "B", "origin": (1, 1), "scale": 0.5, "max_degree": 2},
        {"text": "C", "origin": (-2, 3), "scale": 2.0, "max_degree": 4},
        {"text": "1", "origin": (0, 0), "scale": 1.0, "max_degree": 3},
    ]
    
    passed = 0
    for i, case in enumerate(test_cases):
        try:
            converter = TextToDesmos(origin=case["origin"], scale=case["scale"])
            functions = converter.text_to_desmos_functions(
                text=case["text"],
                font_size=40,
                points_per_char=15,
                max_degree=case["max_degree"]
            )
            
            if len(functions) > 0:
                print(f"✓ Test case {i+1}: Generated {len(functions)} functions for '{case['text']}'")
                passed += 1
            else:
                print(f"✗ Test case {i+1}: No functions generated for '{case['text']}'")
        
        except Exception as e:
            print(f"✗ Test case {i+1}: Error - {e}")
    
    return passed == len(test_cases)

def test_file_output():
    """Test file output functionality."""
    print("Testing file output...")
    
    try:
        converter = TextToDesmos()
        functions = converter.text_to_desmos_functions(
            text="D",
            font_size=30,
            points_per_char=10,
            max_degree=2
        )
        
        # Test save functionality
        test_filename = "test_output.txt"
        converter.save_functions(functions, test_filename)
        
        # Check if file was created
        if os.path.exists(test_filename):
            # Check file content
            with open(test_filename, 'r') as f:
                content = f.read()
                if "# Desmos Functions" in content and len(content) > 100:
                    print(f"✓ Successfully saved {len(functions)} functions to {test_filename}")
                    # Clean up
                    os.remove(test_filename)
                    return True
                else:
                    print(f"✗ File content appears invalid")
                    return False
        else:
            print(f"✗ File {test_filename} was not created")
            return False
            
    except Exception as e:
        print(f"✗ Error during file output test: {e}")
        return False

def test_function_format():
    """Test that generated functions have correct format."""
    print("Testing function format...")
    
    try:
        converter = TextToDesmos()
        functions = converter.text_to_desmos_functions(
            text="E",
            font_size=40,
            points_per_char=15,
            max_degree=2
        )
        
        if not functions:
            print("✗ No functions to test")
            return False
        
        valid_functions = 0
        for func in functions:
            # Check if function starts with y = or x =
            if func.startswith("y = ") or func.startswith("x = "):
                # Check if it contains domain constraints
                if "{" in func and "}" in func and ("x >=" in func or "y >=" in func):
                    valid_functions += 1
                else:
                    print(f"✗ Function missing domain constraints: {func[:50]}...")
                    return False
            else:
                print(f"✗ Invalid function format: {func[:50]}...")
                return False
        
        print(f"✓ All {valid_functions} functions have correct format")
        return True
        
    except Exception as e:
        print(f"✗ Error during format test: {e}")
        return False

def main():
    """Run all tests."""
    print("Text to Desmos Converter - Test Suite")
    print("=" * 45)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Parameter Variations", test_parameter_variations),
        ("File Output", test_file_output),
        ("Function Format", test_function_format),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  Test failed!")
    
    print("\n" + "=" * 45)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The program is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
