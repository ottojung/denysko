#!/usr/bin/env python3
"""
Test the medial axis skeleton extraction on letter 'A'
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_extractor import TextExtractor

def test_medial_axis():
    print("Testing medial axis skeleton extraction...")
    
    # Create text extractor
    extractor = TextExtractor()
    
    # Test with letter 'A'
    test_char = 'A'
    print(f"\nTesting character: '{test_char}'")
    
    # Generate basic preview with fewer points to see structure better
    print("Generating basic preview (100 points)...")
    try:
        extractor.preview_extracted_points(test_char, num_points=100, save_path='medial_basic_A.png')
        print("✓ Basic preview generated successfully")
    except Exception as e:
        print(f"✗ Basic preview failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate detailed skeleton preview
    print("Generating detailed skeleton preview...")
    try:
        extractor.preview_skeleton_extraction_steps(test_char, save_path='medial_skeleton_A.png')
        print("✓ Detailed skeleton preview generated successfully")
    except Exception as e:
        print(f"✗ Detailed skeleton preview failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed. Check the generated images:")
    print("- medial_basic_A.png: Basic centerline extraction with medial axis")
    print("- medial_skeleton_A.png: Detailed skeleton extraction process")

if __name__ == "__main__":
    test_medial_axis()
