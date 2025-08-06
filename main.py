#!/usr/bin/env python3
"""
Text to Desmos Polynomial Plotter

This program converts text into polynomial approximations that can be plotted on Desmos.
It extracts character outlines from fonts and fits polynomial curves to approximate the shapes.
"""

import numpy as np
from matplotlib import font_manager
from matplotlib.path import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# All required imports are now included above


class TextToDesmos:
    def __init__(self, origin=(0, 0), scale=1.0):
        """
        Initialize the text to Desmos converter.

        Args:
            origin (tuple): Origin point (x, y) for positioning the text
            scale (float): Scale factor for the text size
        """
        self.origin = origin
        self.scale = scale
        self.functions = []

    def get_font_path(self):
        """Get a suitable font path for text rendering."""
        try:
            # Try to find a common system font
            fonts = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
            for font_name in fonts:
                font_path = font_manager.findfont(
                    font_manager.FontProperties(family=font_name)
                )
                if font_path:
                    return font_path
        except Exception:
            pass

        # Fallback to default font
        return font_manager.findfont(font_manager.FontProperties())

    def text_to_paths(self, text, font_size=100):
        """
        Convert text to matplotlib Path objects representing character outlines.

        Args:
            text (str): Input text to convert
            font_size (int): Font size for rendering

        Returns:
            list: List of Path objects for each character
        """
        from matplotlib.textpath import TextPath

        paths = []
        x_offset = 0

        for char in text:
            if char == " ":
                x_offset += font_size * 0.3  # Space width
                continue

            try:
                # Create text path for the character
                path = TextPath((x_offset, 0), char, size=font_size)
                if len(path.vertices) > 0:
                    paths.append(path)

                # Calculate character width for next character positioning
                bbox = path.get_extents()
                char_width = bbox.width if bbox.width > 0 else font_size * 0.5
                x_offset += char_width + font_size * 0.05  # Small spacing between chars

            except Exception as e:
                print(f"Warning: Could not process character '{char}': {e}")
                x_offset += font_size * 0.5  # Default spacing

        return paths

    def extract_contour_points(self, path, num_points=50):
        """
        Extract points along the contour of a path.

        Args:
            path: matplotlib Path object
            num_points (int): Number of points to extract per contour

        Returns:
            list: List of (x, y) coordinate arrays for each contour
        """
        if len(path.vertices) == 0:
            return []

        # Get path vertices and codes
        vertices = path.vertices
        codes = path.codes

        contours = []
        current_contour = []

        i = 0
        while i < len(vertices):
            if codes is None or codes[i] == Path.MOVETO:
                # Start new contour
                if current_contour:
                    contours.append(np.array(current_contour))
                current_contour = [vertices[i]]
            elif codes[i] == Path.LINETO:
                current_contour.append(vertices[i])
            elif codes[i] == Path.CLOSEPOLY:
                if current_contour:
                    contours.append(np.array(current_contour))
                current_contour = []
            i += 1

        # Add the last contour if it exists
        if current_contour:
            contours.append(np.array(current_contour))

        # Resample each contour to have approximately num_points
        resampled_contours = []
        for contour in contours:
            if len(contour) < 3:  # Skip very small contours
                continue

            # Calculate cumulative distances along the contour
            distances = np.cumsum(
                np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
            )
            distances = np.insert(distances, 0, 0)

            # Create evenly spaced parameters
            total_length = distances[-1]
            if total_length == 0:
                continue

            even_distances = np.linspace(0, total_length, num_points)

            # Interpolate to get evenly spaced points
            x_interp = np.interp(even_distances, distances, contour[:, 0])
            y_interp = np.interp(even_distances, distances, contour[:, 1])

            resampled_contour = np.column_stack([x_interp, y_interp])
            resampled_contours.append(resampled_contour)

        return resampled_contours

    def fit_polynomial_contour_tracing(self, contour, max_degree=12):
        """
        Fit polynomials that actually trace the letter contour paths.
        
        This method creates polynomials that follow the actual shape of letters
        by using careful contour analysis and high-degree polynomial fitting.
        
        Args:
            contour (np.array): Array of (x, y) points along the letter contour
            max_degree (int): Maximum polynomial degree (can be high for accuracy)
            
        Returns:
            list: List of polynomial functions that trace the letter shape
        """
        if len(contour) < 4:
            return []
        
        functions = []
        
        # Ensure contour is a closed loop by connecting end to start if needed
        if np.linalg.norm(contour[0] - contour[-1]) > 1e-3:
            contour = np.vstack([contour, contour[0]])
        
        x_data = contour[:, 0]
        y_data = contour[:, 1]
        
        # Method 1: Direct polynomial fitting where possible
        try:
            # Approach 1a: Try to fit y = f(x) where it makes sense
            if self._has_function_property(x_data, y_data):
                # Sort by x and fit y = f(x) directly
                sort_idx = np.argsort(x_data)
                x_sorted = x_data[sort_idx]
                y_sorted = y_data[sort_idx]
                
                # Remove duplicate x values by averaging y values
                x_unique, indices = np.unique(x_sorted, return_inverse=True)
                y_averaged = np.array([np.mean(y_sorted[indices == i]) for i in range(len(x_unique))])
                
                if len(x_unique) >= 3:
                    degree = min(max_degree, len(x_unique) - 1)
                    y_coeffs_direct = np.polyfit(x_unique, y_averaged, degree)
                    
                    # Create function string
                    terms = []
                    for i, coeff in enumerate(y_coeffs_direct):
                        if abs(coeff) < 1e-12:
                            continue
                        power = degree - i
                        if power == 0:
                            terms.append(f"{coeff:.8f}")
                        elif power == 1:
                            terms.append(f"{coeff:.8f}*x")
                        else:
                            terms.append(f"{coeff:.8f}*x^{power}")
                    
                    if terms:
                        func_str = " + ".join(terms).replace("+ -", "- ")
                        functions.append(f"y = {func_str}")
            
            # Approach 1b: Try x = f(y) where appropriate
            if self._has_function_property(y_data, x_data):
                sort_idx = np.argsort(y_data)
                y_sorted = y_data[sort_idx]
                x_sorted = x_data[sort_idx]
                
                # Remove duplicate y values
                y_unique, indices = np.unique(y_sorted, return_inverse=True)
                x_averaged = np.array([np.mean(x_sorted[indices == i]) for i in range(len(y_unique))])
                
                if len(y_unique) >= 3:
                    degree = min(max_degree, len(y_unique) - 1)
                    x_coeffs_direct = np.polyfit(y_unique, x_averaged, degree)
                    
                    # Create function string
                    terms = []
                    for i, coeff in enumerate(x_coeffs_direct):
                        if abs(coeff) < 1e-12:
                            continue
                        power = degree - i
                        if power == 0:
                            terms.append(f"{coeff:.8f}")
                        elif power == 1:
                            terms.append(f"{coeff:.8f}*y")
                        else:
                            terms.append(f"{coeff:.8f}*y^{power}")
                    
                    if terms:
                        func_str = " + ".join(terms).replace("+ -", "- ")
                        functions.append(f"x = {func_str}")
        
        except Exception as e:
            print(f"Warning: Failed to fit direct polynomials: {e}")
        
        # Method 2: Piecewise polynomial approach for complex shapes
        # Split contour into segments that can be represented as functions
        try:
            segments = self._split_contour_into_functional_segments(contour)
            
            for segment in segments:
                if len(segment) < 3:
                    continue
                
                x_seg = segment[:, 0]
                y_seg = segment[:, 1]
                
                # Determine if this segment is better as y=f(x) or x=f(y)
                x_range = np.max(x_seg) - np.min(x_seg)
                y_range = np.max(y_seg) - np.min(y_seg)
                
                if x_range >= y_range and x_range > 1e-6:
                    # Fit y = f(x)
                    sort_idx = np.argsort(x_seg)
                    x_sorted = x_seg[sort_idx]
                    y_sorted = y_seg[sort_idx]
                    
                    degree = min(max_degree, len(x_sorted) - 1)
                    if degree >= 1:
                        coeffs = np.polyfit(x_sorted, y_sorted, degree)
                        
                        terms = []
                        for i, coeff in enumerate(coeffs):
                            if abs(coeff) < 1e-12:
                                continue
                            power = degree - i
                            if power == 0:
                                terms.append(f"{coeff:.8f}")
                            elif power == 1:
                                terms.append(f"{coeff:.8f}*x")
                            else:
                                terms.append(f"{coeff:.8f}*x^{power}")
                        
                        if terms:
                            func_str = " + ".join(terms).replace("+ -", "- ")
                            functions.append(f"y = {func_str}")
                
                elif y_range > 1e-6:
                    # Fit x = f(y)
                    sort_idx = np.argsort(y_seg)
                    y_sorted = y_seg[sort_idx]
                    x_sorted = x_seg[sort_idx]
                    
                    degree = min(max_degree, len(y_sorted) - 1)
                    if degree >= 1:
                        coeffs = np.polyfit(y_sorted, x_sorted, degree)
                        
                        terms = []
                        for i, coeff in enumerate(coeffs):
                            if abs(coeff) < 1e-12:
                                continue
                            power = degree - i
                            if power == 0:
                                terms.append(f"{coeff:.8f}")
                            elif power == 1:
                                terms.append(f"{coeff:.8f}*y")
                            else:
                                terms.append(f"{coeff:.8f}*y^{power}")
                        
                        if terms:
                            func_str = " + ".join(terms).replace("+ -", "- ")
                            functions.append(f"x = {func_str}")
                            
        except Exception as e:
            print(f"Warning: Failed to fit piecewise polynomials: {e}")
        
        return functions
    
    def _has_function_property(self, x_data, y_data):
        """Check if data can be represented as a function (no repeated x values)."""
        x_unique = np.unique(x_data)
        return len(x_unique) > 0.7 * len(x_data)  # Allow some repeated values
    
    def _split_contour_into_functional_segments(self, contour):
        """
        Split a contour into segments that can be represented as functions.
        
        This finds natural break points where the contour changes direction
        significantly or where it would fail the vertical/horizontal line test.
        """
        if len(contour) < 6:
            return [contour]
        
        segments = []
        current_segment = [contour[0]]
        
        for i in range(1, len(contour)):
            current_point = contour[i]
            current_segment.append(current_point)
            
            # Check if we should start a new segment
            if len(current_segment) >= 8:  # Minimum segment size
                # Look for direction changes or function property violations
                segment_array = np.array(current_segment)
                x_data = segment_array[:, 0]
                y_data = segment_array[:, 1]
                
                # Check if segment violates function property badly
                x_range = np.max(x_data) - np.min(x_data)
                y_range = np.max(y_data) - np.min(y_data)
                
                if x_range > y_range:
                    # Check for y=f(x) violations
                    x_unique = np.unique(x_data)
                    if len(x_unique) < 0.5 * len(x_data):  # Too many repeated x values
                        # Start new segment
                        segments.append(np.array(current_segment[:-3]))  # Overlap for continuity
                        current_segment = current_segment[-3:]  # Keep last few points
                else:
                    # Check for x=f(y) violations  
                    y_unique = np.unique(y_data)
                    if len(y_unique) < 0.5 * len(y_data):  # Too many repeated y values
                        segments.append(np.array(current_segment[:-3]))
                        current_segment = current_segment[-3:]
        
        # Add the final segment
        if len(current_segment) >= 3:
            segments.append(np.array(current_segment))
        
        return segments

    def apply_transform(self, functions):
        """
        Apply origin translation and scaling to the functions.

        Args:
            functions (list): List of function strings

        Returns:
            list: Transformed function strings
        """
        transformed = []
        ox, oy = self.origin

        for func in functions:
            if func.startswith("y ="):
                # For y = f(x), transform: y - oy = scale * f((x - ox)/scale)
                rhs = func[4:]  # Remove "y = "
                new_func = f"y = {oy:.3f} + {self.scale:.3f} * ({rhs.replace('x', f'((x - {ox:.3f})/{self.scale:.3f})')})"
                transformed.append(new_func)
            elif func.startswith("x ="):
                # For x = f(y), transform: x - ox = scale * f((y - oy)/scale)
                rhs = func[4:]  # Remove "x = "
                new_func = f"x = {ox:.3f} + {self.scale:.3f} * ({rhs.replace('y', f'((y - {oy:.3f})/{self.scale:.3f})')})"
                transformed.append(new_func)
            else:
                transformed.append(func)

        return transformed

    def text_to_desmos_functions(
        self,
        text,
        font_size=100,
        points_per_char=50,
        max_degree=12,
    ):
        """
        Convert text to Desmos-compatible polynomial functions that trace letter shapes.

        Args:
            text (str): Input text
            font_size (int): Font size for rendering
            points_per_char (int): Number of points to extract per character
            max_degree (int): Maximum polynomial degree

        Returns:
            list: List of Desmos function strings that trace the actual letter shapes
        """
        print(f"Converting text '{text}' to polynomial functions that trace letter shapes...")

        # Convert text to paths
        paths = self.text_to_paths(text, font_size)
        print(f"Generated {len(paths)} character paths")

        all_functions = []

        # Process each character path
        for i, path in enumerate(paths):
            print(f"Processing character {i + 1}/{len(paths)}...")

            # Extract contour points
            contours = self.extract_contour_points(path, points_per_char)
            print(f"  Found {len(contours)} contours")

            # Fit polynomials that actually trace each contour shape
            for j, contour in enumerate(contours):
                functions = self.fit_polynomial_contour_tracing(contour, max_degree)
                all_functions.extend(functions)
                print(f"    Contour {j+1}: {len(functions)} polynomial functions tracing the shape")

        # Apply transformations
        transformed_functions = self.apply_transform(all_functions)

        print(f"\nGenerated {len(transformed_functions)} total functions")
        print("Functions trace the actual letter shapes using high-degree polynomials")
        return transformed_functions

    def visualize_preview(self, text, functions=None):
        """
        Create a preview visualization of the text and fitted functions.

        Args:
            text (str): Original text
            functions (list): List of function strings (optional, will generate if not provided)
        """
        if functions is None:
            functions = self.text_to_desmos_functions(text)

        plt.figure(figsize=(12, 6))

        # Plot original text paths
        plt.subplot(1, 2, 1)
        paths = self.text_to_paths(text)
        for path in paths:
            if len(path.vertices) > 0:
                vertices = path.vertices
                plt.plot(vertices[:, 0], vertices[:, 1], "b-", alpha=0.7, linewidth=1)
        plt.title("Original Text Paths")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)

        # Plot polynomial approximations
        plt.subplot(1, 2, 2)
        # Note: This is a simplified visualization
        # In practice, you'd need to evaluate the polynomial functions
        plt.title("Polynomial Approximations\n(Preview - see Desmos for actual result)")
        plt.text(
            0.5,
            0.5,
            f"Generated {len(functions)} functions\nfor text: '{text}'",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.axis("equal")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_functions(self, functions, filename="desmos_functions.txt"):
        """
        Save the generated functions to a text file.

        Args:
            functions (list): List of function strings
            filename (str): Output filename
        """
        with open(filename, "w") as f:
            f.write("# Desmos Functions Generated from Text\n")
            f.write(
                "# Copy and paste these functions into Desmos graphing calculator\n\n"
            )

            for i, func in enumerate(functions, 1):
                f.write(f"# Function {i}\n")
                f.write(f"{func}\n\n")

        print(f"Functions saved to {filename}")


def main():
    """Main function demonstrating the text to Desmos converter."""
    print("Text to Desmos Polynomial Converter")
    print("=" * 40)

    # Get user input
    text = input("Enter text to convert: ").strip()
    if not text:
        text = "HELLO"
        print(f"Using default text: {text}")

    try:
        origin_input = input("Enter origin point (x,y) [default: 0,0]: ").strip()
        if origin_input:
            origin = tuple(map(float, origin_input.split(",")))
        else:
            origin = (0, 0)
    except Exception:
        origin = (0, 0)
        print("Using default origin: (0, 0)")

    try:
        scale_input = input("Enter scale factor [default: 1.0]: ").strip()
        scale = float(scale_input) if scale_input else 1.0
    except Exception:
        scale = 1.0
        print("Using default scale: 1.0")

    # Create converter
    converter = TextToDesmos(origin=origin, scale=scale)

    # Generate functions
    functions = converter.text_to_desmos_functions(text)

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

    # Show preview
    try:
        show_preview = (
            input("\nShow preview visualization? (y/n) [default: n]: ").strip().lower()
        )
        if show_preview == "y":
            converter.visualize_preview(text, functions)
    except Exception:
        pass

    print(f"\nGenerated {len(functions)} functions for text: '{text}'")
    print("Functions saved to 'desmos_functions.txt'")


if __name__ == "__main__":
    main()
