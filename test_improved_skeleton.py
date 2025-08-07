#!/usr/bin/env python3
"""
Test the improved skeleton extraction on letter 'A'
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.text_extractor import TextExtractor

def test_improved_skeleton():
    print("Testing improved stroke centerline extraction...")
    
    # Create text extractor
    extractor = TextExtractor()
    
    # Test with letter 'A'
    test_char = 'A'
    print(f"\nTesting character: '{test_char}'")
    
    # Generate basic preview
    print("Generating basic preview...")
    try:
        extractor.preview_extracted_points(test_char, save_path='improved_basic_A.png')
        print("✓ Basic preview generated successfully")
    except Exception as e:
        print(f"✗ Basic preview failed: {e}")
    
    # Generate detailed skeleton preview
    print("Generating detailed skeleton preview...")
    try:
        extractor.preview_skeleton_extraction_steps(test_char, save_path='improved_skeleton_A.png')
        print("✓ Detailed skeleton preview generated successfully")
    except Exception as e:
        print(f"✗ Detailed skeleton preview failed: {e}")
    
    print("\nTest completed. Check the generated images:")
    print("- improved_basic_A.png: Basic centerline extraction")
    print("- improved_skeleton_A.png: Detailed skeleton extraction process")

if __name__ == "__main__":
    test_improved_skeleton()
