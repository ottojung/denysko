#!/usr/bin/env python3
"""
Function transformation module - handles coordinate transformations and simplifications.
Only works with y = f(x) functions.
"""

import re


class FunctionTransformer:
    """Transforms and simplifies y = f(x) polynomial functions."""
    
    def __init__(self, origin=(0, 0), scale=1.0):
        """
        Initialize the function transformer.
        
        Args:
            origin (tuple): Origin point (x, y) for positioning
            scale (float): Scale factor for the functions
        """
        self.origin = origin
        self.scale = scale
    
    def transform_function(self, func_str):
        """
        Apply coordinate transformations to a y = f(x) function.
        
        Args:
            func_str (str): Function string in form "y = f(x)" possibly with domain restrictions
            
        Returns:
            str: Transformed function string with preserved domain restrictions
        """
        if not func_str.startswith("y ="):
            return func_str  # Only transform y = f(x) functions
        
        # Split the function and domain restrictions
        if '{' in func_str and '}' in func_str:
            # Extract function and domain parts
            func_part = func_str[:func_str.find('{')].strip()
            domain_part = func_str[func_str.find('{'):].strip()
        else:
            func_part = func_str
            domain_part = ""
        
        # Extract the right side of the equation
        rhs = func_part[4:].strip()  # Remove "y = "
        
        # Apply scaling to x: replace x with (x - origin_x) / scale
        if self.scale != 1.0 or self.origin[0] != 0:
            x_transform = f"(x - {self.origin[0]}) / {self.scale}" if self.origin[0] != 0 else f"x / {self.scale}"
            if self.scale == 1.0:
                x_transform = f"(x - {self.origin[0]})"
            
            # Replace x in the polynomial with the transformation
            # Be careful to only replace standalone x variables
            rhs = re.sub(r'\bx\b', f"({x_transform})", rhs)
        
        # Apply y offset
        if self.origin[1] != 0:
            result = f"y = {rhs} + {self.origin[1]}"
        else:
            result = f"y = {rhs}"
        
        # Add domain restrictions back
        if domain_part:
            result += " " + domain_part
        
        return result
        if self.origin[1] != 0:
            return f"y = {rhs} + {self.origin[1]}"
        
        return f"y = {rhs}"
    
    def transform_functions(self, functions):
        """
        Transform multiple y = f(x) functions.
        
        Args:
            functions (list): List of function strings
            
        Returns:
            list: List of transformed function strings
        """
        transformed = []
        for func in functions:
            if func.startswith("y ="):  # Only transform y = f(x) functions
                transformed.append(self.transform_function(func))
            else:
                # Skip any non-y functions (shouldn't exist but safety check)
                print(f"Warning: Skipping non-y function: {func}")
        
        return transformed
    
    def simplify_function_string(self, func_str):
        """
        Simplify a y = f(x) function string for better readability.
        
        Args:
            func_str (str): Function string to simplify
            
        Returns:
            str: Simplified function string
        """
        if not func_str.startswith("y ="):
            return func_str
        
        # Clean up the function string
        simplified = func_str
        
        # Remove unnecessary spaces around operators
        simplified = re.sub(r'\s*\+\s*', ' + ', simplified)
        simplified = re.sub(r'\s*-\s*', ' - ', simplified)
        simplified = re.sub(r'\s*\*\s*', '*', simplified)
        simplified = re.sub(r'\s*/\s*', '/', simplified)
        
        # Clean up multiple spaces
        simplified = re.sub(r'\s+', ' ', simplified)
        
        # Remove trailing spaces
        simplified = simplified.strip()
        
        # Fix leading minus signs
        simplified = re.sub(r'= -', '= -', simplified)
        simplified = re.sub(r'\+ -', '- ', simplified)
        
        return simplified
