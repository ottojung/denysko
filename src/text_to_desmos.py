#!/usr/bin/env python3
"""
Main TextToDesmos class that coordinates all modules.
Only generates y = f(x) functions - no x = f(y) functions.
"""

from .text_extractor import TextExtractor
from .polynomial_fitter import PolynomialFitter
from .function_transformer import FunctionTransformer


class TextToDesmos:
    """
    Main class for converting text to Desmos polynomial functions.
    Generates ONLY y = f(x) functions.
    """
    
    def __init__(self, origin=(0, 0), scale=1.0, max_degree=8):
        self.extractor = TextExtractor()
        self.fitter = PolynomialFitter(max_degree=max_degree)
        self.transformer = FunctionTransformer(origin=origin, scale=scale)
    
    def text_to_desmos_functions(self, text, font_size=100, points_per_char=500):
        """
        Convert text to Desmos-compatible y = f(x) polynomial functions ONLY.
        Uses HUNDREDS of centerline points for precise letter tracing.
        
        Args:
            text (str): Input text
            font_size (int): Font size for rendering
            points_per_char (int): Number of centerline points to sample per character (default: 500)
            
        Returns:
            list: List of Desmos function strings (y = f(x) ONLY)
        """
        print(f"Converting text '{text}' to y = f(x) polynomial functions...")
        print(f"Using {points_per_char} centerline points per character for high precision")
        print("Note: Only generating y = f(x) functions, no x = f(y)")
        
        # Step 1: Extract character paths
        paths = self.extractor.text_to_paths(text, font_size)
        print(f"Generated {len(paths)} character paths")
        
        if not paths:
            print("Warning: No character paths generated")
            return []
        
        all_functions = []
        
        # Step 2: Process each character with HUNDREDS of points
        for i, path in enumerate(paths):
            print(f"Processing character {i + 1}/{len(paths)}...")
            
            # Extract centerline points (hundreds of points)
            contours = self.extractor.extract_contour_points(path, points_per_char)
            print(f"  Found {len(contours)} contours")
            
            # Fit polynomials to each contour - ONLY y = f(x)
            for j, contour in enumerate(contours):
                print(f"  Processing contour {j + 1} with {len(contour)} centerline points...")
                # Do not pass any degree cap; fitter performs exact interpolation
                functions = self.fitter.fit_contour_polynomials(contour)
                
                # Ensure all functions are y = f(x)
                y_functions = [f for f in functions if f.startswith("y =")]
                
                all_functions.extend(y_functions)
                print(f"    Contour {j + 1}: generated {len(y_functions)} y = f(x) functions")
                
                # Report any non-y functions (should be none)
                non_y = [f for f in functions if not f.startswith("y =")]
                if non_y:
                    print(f"    WARNING: Filtered out {len(non_y)} non-y functions")
        
        # Step 3: Apply coordinate transformations
        transformed_functions = self.transformer.transform_functions(all_functions)
        
        # Step 4: Simplify function strings
        simplified_functions = []
        for func in transformed_functions:
            simplified = self.transformer.simplify_function_string(func)
            simplified_functions.append(simplified)
        
        # Final verification - ensure ALL are y = f(x)
        final_functions = [f for f in simplified_functions if f.startswith("y =")]
        
        if len(final_functions) != len(simplified_functions):
            print(f"WARNING: Filtered out {len(simplified_functions) - len(final_functions)} non-y functions")
        
        print(f"\nGenerated {len(final_functions)} total y = f(x) functions")
        print("All functions are in the form y = f(x)")
        return final_functions
    
    def save_functions(self, functions, filename="desmos_functions.txt"):
        """
        Save generated functions to a text file.
        
        Args:
            functions (list): List of function strings
            filename (str): Output filename
        """
        with open(filename, "w") as f:
            f.write("# Desmos Functions Generated from Text\n")
            f.write("# ALL functions are in the form y = f(x)\n")
            f.write("# NO x = f(y) functions are generated\n")
            f.write("# Copy and paste these functions into Desmos graphing calculator\n\n")
            
            for i, func in enumerate(functions, 1):
                f.write(f"# Function {i}\n")
                f.write(f"{func}\n\n")
        
        print(f"Functions saved to {filename}")
