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

    def fit_polynomial_segments(
        self, contour, max_degree=6, num_functions_per_contour=3
    ):
        """
        Fit multiple polynomial functions to a contour to highlight the letter shape.

        Args:
            contour (np.array): Array of (x, y) points
            max_degree (int): Maximum polynomial degree
            num_functions_per_contour (int): Number of different functions to fit per contour

        Returns:
            list: List of polynomial functions as strings (without domain constraints)
        """
        if len(contour) < 3:
            return []

        functions = []
        n_points = len(contour)

        # Create multiple polynomial fits with different approaches to highlight the shape

        # Approach 1: Fit y = f(x) if x has sufficient variation
        x_data = contour[:, 0]
        y_data = contour[:, 1]
        x_range = np.max(x_data) - np.min(x_data)
        y_range = np.max(y_data) - np.min(y_data)

        if x_range > 1e-3:  # If x has reasonable variation
            # Sort by x
            sort_idx = np.argsort(x_data)
            x_sorted = x_data[sort_idx]
            y_sorted = y_data[sort_idx]

            # Create multiple functions with different degrees
            for degree in range(2, min(max_degree + 1, len(x_sorted))):
                try:
                    coeffs = np.polyfit(x_sorted, y_sorted, degree)

                    # Generate function string without domain constraints
                    terms = []
                    for i, coeff in enumerate(coeffs):
                        if abs(coeff) < 1e-12:
                            continue
                        power = degree - i
                        if power == 0:
                            terms.append(f"{coeff:.6f}")
                        elif power == 1:
                            terms.append(f"{coeff:.6f}*x")
                        else:
                            terms.append(f"{coeff:.6f}*x^{power}")

                    if terms:
                        func_str = " + ".join(terms).replace("+ -", "- ")
                        final_func = f"y = {func_str}"
                        functions.append(final_func)

                        # Only add a few different degrees to avoid too many functions
                        if len(functions) >= num_functions_per_contour // 2:
                            break

                except Exception as e:
                    print(
                        f"Warning: Failed to fit y=f(x) polynomial degree {degree}: {e}"
                    )
                    continue

        # Approach 2: Fit x = f(y) if y has sufficient variation
        if y_range > 1e-3:  # If y has reasonable variation
            # Sort by y
            sort_idx = np.argsort(y_data)
            y_sorted = y_data[sort_idx]
            x_sorted = x_data[sort_idx]

            # Create functions with different degrees
            for degree in range(2, min(max_degree + 1, len(y_sorted))):
                try:
                    coeffs = np.polyfit(y_sorted, x_sorted, degree)

                    # Generate function string without domain constraints
                    terms = []
                    for i, coeff in enumerate(coeffs):
                        if abs(coeff) < 1e-12:
                            continue
                        power = degree - i
                        if power == 0:
                            terms.append(f"{coeff:.6f}")
                        elif power == 1:
                            terms.append(f"{coeff:.6f}*y")
                        else:
                            terms.append(f"{coeff:.6f}*y^{power}")

                    if terms:
                        func_str = " + ".join(terms).replace("+ -", "- ")
                        final_func = f"x = {func_str}"
                        functions.append(final_func)

                        # Limit number of functions
                        if (
                            len([f for f in functions if f.startswith("x =")])
                            >= num_functions_per_contour // 2
                        ):
                            break

                except Exception as e:
                    print(
                        f"Warning: Failed to fit x=f(y) polynomial degree {degree}: {e}"
                    )
                    continue

        # Approach 3: If we have very few functions, try parametric approach
        if len(functions) < 2 and n_points > 5:
            try:
                # Fit x(t) and y(t) separately for parametric representation
                degree = min(4, n_points - 1)

                # Create parametric functions (approximation by eliminating parameter)
                # This is a simplification - in practice, parametric curves are more complex
                # But we'll create a simple approximation
                if degree >= 2:
                    # Simple approach: create a relationship between x and y
                    t_vals = np.linspace(0, 1, len(x_data))

                    # Fit real and imaginary parts
                    real_coeffs = np.polyfit(t_vals, x_data, min(3, len(t_vals) - 1))
                    imag_coeffs = np.polyfit(t_vals, y_data, min(3, len(t_vals) - 1))

                    # Create implicit function (simplified)
                    # This is a rough approximation
                    if len(real_coeffs) >= 2 and len(imag_coeffs) >= 2:
                        # Create a simple implicit relationship
                        a, b, c = (
                            real_coeffs[-3:]
                            if len(real_coeffs) >= 3
                            else [*real_coeffs, 0, 0][:3]
                        )
                        d, e_coeff, f = (
                            imag_coeffs[-3:]
                            if len(imag_coeffs) >= 3
                            else [*imag_coeffs, 0, 0][:3]
                        )

                        # Simple implicit form: ax^2 + by^2 + cxy + dx + ey + f = 0
                        # Convert to explicit form when possible
                        if abs(b) > 1e-6:
                            # Create implicit equation representation
                            implicit_func = f"({a:.6f})*x^2 + ({c:.6f})*x*y + ({d:.6f})*x + ({b:.6f})*y^2 + ({e_coeff:.6f})*y + ({f:.6f}) = 0"
                            functions.append(implicit_func)

            except Exception as e:
                print(f"Warning: Failed to create parametric approximation: {e}")

        return functions

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
        max_degree=6,
        functions_per_contour=3,
    ):
        """
        Convert text to Desmos-compatible polynomial functions without domain constraints.

        Args:
            text (str): Input text
            font_size (int): Font size for rendering
            points_per_char (int): Number of points to extract per character
            max_degree (int): Maximum polynomial degree
            functions_per_contour (int): Number of functions to generate per contour

        Returns:
            list: List of Desmos function strings (without domain constraints)
        """
        print(f"Converting text '{text}' to Desmos functions...")
        print(
            "Note: Functions will have no domain constraints - curves may extend beyond letter boundaries"
        )

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

            # Fit multiple polynomials to each contour to highlight the shape
            for j, contour in enumerate(contours):
                functions = self.fit_polynomial_segments(
                    contour, max_degree, functions_per_contour
                )
                all_functions.extend(functions)
                print(f"    Contour {j + 1}: {len(functions)} polynomial functions")

        # Apply transformations
        transformed_functions = self.apply_transform(all_functions)

        print(f"\nGenerated {len(transformed_functions)} total functions")
        print(
            "Each contour has multiple overlapping polynomial approximations to highlight the letter shape"
        )
        return transformed_functions

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

            # Fit polynomials to each contour
            for j, contour in enumerate(contours):
                functions = self.fit_polynomial_segments(contour, max_degree)
                all_functions.extend(functions)
                print(f"    Contour {j + 1}: {len(functions)} polynomial segments")

        # Apply transformations
        transformed_functions = self.apply_transform(all_functions)

        print(f"\nGenerated {len(transformed_functions)} total functions")
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
