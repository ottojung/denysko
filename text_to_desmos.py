#!/usr/bin/env python3
"""
Text to Desmos converter - main coordination module.
"""

from text_extractor import TextExtractor
from polynomial_fitter import PolynomialFitter
from function_transformer import FunctionTransformer


class TextToDesmos:
    """Main class that coordinates text to Desmos polynomial conversion."""
    
    def __init__(self, origin=(0, 0), scale=1.0):
        """
        Initialize the text to Desmos converter.
        
        Args:
            origin (tuple): Origin point (x, y) for positioning the text
            scale (float): Scale factor for the text size
        """
        self.text_extractor = TextExtractor()
        self.polynomial_fitter = PolynomialFitter()
        self.function_transformer = FunctionTransformer(origin, scale)
    
    def convert_text_to_functions(self, text, font_size=100, points_per_char=50, max_degree=8):
        """
        Convert text to Desmos-compatible y=f(x) polynomial functions.
        
        Args:
            text (str): Input text
            font_size (int): Font size for rendering
            points_per_char (int): Number of points to extract per character
            max_degree (int): Maximum polynomial degree
            
        Returns:
            list: List of Desmos function strings that approximate the letter shapes
        """
        print(f"Converting text '{text}' to y=f(x) polynomial functions...")
        
        # Step 1: Extract coordinate points from text
        print("Step 1: Extracting coordinate points...")
        contours = self.text_extractor.text_to_contours(text, font_size, points_per_char)
        
        if not contours:
            print("ERROR: No contours extracted from text!")
            return []
        
        print(f"Extracted {len(contours)} contours with coordinate points")
        
        # Step 2: Fit polynomials to coordinate points
        print("Step 2: Fitting y=f(x) polynomials...")
        functions = self.polynomial_fitter.fit_contours_to_polynomials(contours, max_degree)
        
        if not functions:
            print("ERROR: No polynomials fitted!")
            return []
        
        print(f"Generated {len(functions)} polynomial functions")
        
        # Step 3: Apply transformations
        print("Step 3: Applying transformations...")
        transformed_functions = self.function_transformer.transform_functions(functions)
        
        print(f"Final result: {len(transformed_functions)} y=f(x) functions")
        return transformed_functions
    
    def save_functions(self, functions, filename="desmos_functions.txt"):
        """
        Save the generated functions to a text file.
        
        Args:
            functions (list): List of function strings
            filename (str): Output filename
        """
        with open(filename, "w") as f:
            f.write("# Desmos Functions Generated from Text\n")
            f.write("# Copy and paste these functions into Desmos graphing calculator\n\n")
            
            for i, func in enumerate(functions, 1):
                f.write(f"# Function {i}\n")
                f.write(f"{func}\n\n")
        
        print(f"Functions saved to {filename}")


def main():
    """Main function for testing - converts letter 'A' with default settings."""
    print("Text to Desmos Polynomial Converter")
    print("=" * 40)
    
    # Test with letter A
    text = "A"
    print(f"Converting letter '{text}' to polynomial functions...")
    
    # Create converter with default settings
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    
    # Generate functions
    functions = converter.convert_text_to_functions(text, max_degree=6)
    
    # Display results
    print("\n" + "=" * 50)
    print("DESMOS FUNCTIONS")
    print("=" * 50)
    print("Copy and paste these functions into Desmos:")
    print()
    
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func}")
    
    # Save to file
    converter.save_functions(functions)
    
    print(f"\nGenerated {len(functions)} y=f(x) functions for text: '{text}'")
    print("Functions saved to 'desmos_functions.txt'")


if __name__ == "__main__":
    main()
