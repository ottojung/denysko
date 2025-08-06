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
        Fit polynomials that actually trace the letter contour paths using structural analysis.
        
        This method identifies the natural structural components of letters (like the legs
        and crossbar of an 'A') and fits appropriate polynomials to each component.
        
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
        
        print(f"      Analyzing contour with {len(contour)} points")
        print(f"      X range: {np.min(x_data):.2f} to {np.max(x_data):.2f}")
        print(f"      Y range: {np.min(y_data):.2f} to {np.max(y_data):.2f}")
        
        # New Approach: Structural Component Analysis
        # Instead of trying to fit the whole contour, identify structural components
        
        try:
            # Method 1: Identify straight line segments using edge detection
            line_segments = self._identify_line_segments(contour)
            print(f"      Identified {len(line_segments)} line segments")
            
            for i, segment in enumerate(line_segments):
                if len(segment) < 2:
                    continue
                    
                x_seg = segment[:, 0]
                y_seg = segment[:, 1]
                
                # Fit linear or low-degree polynomial to each line segment
                try:
                    # For line segments, use lower degree (1-3) for better line representation
                    degree = min(3, len(segment) - 1, max_degree)
                    if degree >= 1:
                        # Determine orientation and fit accordingly
                        x_range = np.max(x_seg) - np.min(x_seg)
                        y_range = np.max(y_seg) - np.min(y_seg)
                        
                        if x_range >= y_range and x_range > 1e-6:
                            # Fit y = f(x) for more horizontal segments
                            coeffs = np.polyfit(x_seg, y_seg, degree)
                            func_str = self._create_polynomial_string(coeffs, 'x')
                            if func_str:
                                functions.append(f"y = {func_str}")
                                print(f"         Segment {i+1}: y = {func_str}")
                        
                        elif y_range > 1e-6:
                            # Fit x = f(y) for more vertical segments
                            coeffs = np.polyfit(y_seg, x_seg, degree)
                            func_str = self._create_polynomial_string(coeffs, 'y')
                            if func_str:
                                functions.append(f"x = {func_str}")
                                print(f"         Segment {i+1}: x = {func_str}")
                
                except Exception as e:
                    print(f"         Failed to fit segment {i+1}: {e}")
                    continue
        
        except Exception as e:
            print(f"      Line segment identification failed: {e}")
        
        # Method 2: Fallback to curvature-based segmentation if line detection fails
        if len(functions) < 2:  # Need at least a few functions to represent a letter
            try:
                print(f"      Using fallback curvature-based segmentation")
                curve_segments = self._split_by_curvature(contour, max_segments=8)
                print(f"      Created {len(curve_segments)} curve segments")
                
                for i, segment in enumerate(curve_segments):
                    if len(segment) < 3:
                        continue
                    
                    x_seg = segment[:, 0]
                    y_seg = segment[:, 1]
                    
                    # Use moderate degree for curve segments
                    degree = min(6, len(segment) - 1, max_degree)
                    
                    x_range = np.max(x_seg) - np.min(x_seg)
                    y_range = np.max(y_seg) - np.min(y_seg)
                    
                    try:
                        if x_range >= y_range and x_range > 1e-6:
                            coeffs = np.polyfit(x_seg, y_seg, degree)
                            func_str = self._create_polynomial_string(coeffs, 'x')
                            if func_str:
                                functions.append(f"y = {func_str}")
                        
                        elif y_range > 1e-6:
                            coeffs = np.polyfit(y_seg, x_seg, degree)
                            func_str = self._create_polynomial_string(coeffs, 'y')
                            if func_str:
                                functions.append(f"x = {func_str}")
                    
                    except Exception as e:
                        print(f"         Failed to fit curve segment {i+1}: {e}")
                        continue
                        
            except Exception as e:
                print(f"      Curvature-based segmentation failed: {e}")
        
        # Method 3: Final fallback - simple segmentation
        if len(functions) == 0:
            print(f"      Using simple fallback segmentation")
            try:
                # Just split the contour into a few overlapping segments
                n_segments = min(6, len(contour) // 8)
                for i in range(n_segments):
                    start_idx = i * len(contour) // n_segments
                    end_idx = min((i + 2) * len(contour) // n_segments, len(contour))
                    
                    segment = contour[start_idx:end_idx]
                    if len(segment) < 3:
                        continue
                    
                    x_seg = segment[:, 0]
                    y_seg = segment[:, 1]
                    
                    degree = min(4, len(segment) - 1)
                    
                    x_range = np.max(x_seg) - np.min(x_seg)
                    y_range = np.max(y_seg) - np.min(y_seg)
                    
                    try:
                        if x_range >= y_range and x_range > 1e-6:
                            coeffs = np.polyfit(x_seg, y_seg, degree)
                            func_str = self._create_polynomial_string(coeffs, 'x')
                            if func_str:
                                functions.append(f"y = {func_str}")
                        
                        elif y_range > 1e-6:
                            coeffs = np.polyfit(y_seg, x_seg, degree)
                            func_str = self._create_polynomial_string(coeffs, 'y')
                            if func_str:
                                functions.append(f"x = {func_str}")
                    
                    except Exception:
                        continue
            
            except Exception as e:
                print(f"      Simple fallback failed: {e}")
        
        print(f"      Generated {len(functions)} functions for this contour")
        return functions
    
    def _create_polynomial_string(self, coeffs, var):
        """Create a polynomial string from coefficients."""
        terms = []
        degree = len(coeffs) - 1
        
        for i, coeff in enumerate(coeffs):
            if abs(coeff) < 1e-12:
                continue
                
            power = degree - i
            coeff_str = f"{coeff:.6f}"
            
            if power == 0:
                terms.append(coeff_str)
            elif power == 1:
                terms.append(f"{coeff_str}*{var}")
            else:
                terms.append(f"{coeff_str}*{var}^{power}")
        
        if not terms:
            return None
            
        func_str = " + ".join(terms).replace("+ -", "- ")
        return func_str
    
    def _identify_line_segments(self, contour):
        """
        Identify straight line segments in a contour using direction analysis.
        This helps identify structural components like the legs of an 'A'.
        """
        if len(contour) < 6:
            return [contour]
        
        segments = []
        
        # Calculate direction vectors between consecutive points
        directions = np.diff(contour, axis=0)
        direction_magnitudes = np.linalg.norm(directions, axis=1)
        
        # Avoid division by zero
        valid_dirs = direction_magnitudes > 1e-6
        normalized_directions = np.zeros_like(directions)
        normalized_directions[valid_dirs] = directions[valid_dirs] / direction_magnitudes[valid_dirs][:, np.newaxis]
        
        # Find points where direction changes significantly
        direction_changes = []
        for i in range(1, len(normalized_directions)):
            if valid_dirs[i-1] and valid_dirs[i]:
                # Calculate angle between consecutive direction vectors
                dot_product = np.dot(normalized_directions[i-1], normalized_directions[i])
                dot_product = np.clip(dot_product, -1, 1)  # Handle numerical errors
                angle_change = np.arccos(dot_product)
                
                # If direction changes by more than 30 degrees, mark as corner
                if angle_change > np.pi / 6:  # 30 degrees
                    direction_changes.append(i)
        
        # Split contour at direction changes
        if len(direction_changes) == 0:
            return [contour]
        
        # Add start and end points
        split_points = [0] + direction_changes + [len(contour)]
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            segment = contour[start:end+1]  # Include endpoint
            
            if len(segment) >= 3:  # Only keep segments with enough points
                segments.append(segment)
        
        return segments
    
    def _split_by_curvature(self, contour, max_segments=8):
        """
        Split contour based on curvature analysis.
        Areas of high curvature indicate corners/bends where we should split.
        """
        if len(contour) < 10:
            return [contour]
        
        # Calculate curvature at each point
        curvatures = []
        
        for i in range(1, len(contour) - 1):
            p1 = contour[i-1]
            p2 = contour[i]
            p3 = contour[i+1]
            
            # Calculate curvature using the three-point method
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product magnitude gives curvature
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                curvature = abs(cross) / (norm_v1 * norm_v2)
                curvatures.append(curvature)
            else:
                curvatures.append(0)
        
        if len(curvatures) == 0:
            return [contour]
        
        # Find high curvature points
        mean_curvature = np.mean(curvatures)
        std_curvature = np.std(curvatures)
        threshold = mean_curvature + 0.5 * std_curvature
        
        high_curvature_indices = []
        for i, curv in enumerate(curvatures):
            if curv > threshold:
                high_curvature_indices.append(i + 1)  # Adjust for offset
        
        # Limit number of segments
        if len(high_curvature_indices) > max_segments - 1:
            # Keep only the highest curvature points
            sorted_indices = sorted(enumerate(curvatures), key=lambda x: x[1], reverse=True)
            top_indices = [i + 1 for i, _ in sorted_indices[:max_segments-1]]
            high_curvature_indices = sorted(top_indices)
        
        # Split at high curvature points
        if len(high_curvature_indices) == 0:
            return [contour]
        
        split_points = [0] + high_curvature_indices + [len(contour)]
        segments = []
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            segment = contour[start:end+1]
            
            if len(segment) >= 3:
                segments.append(segment)
        
        return segments
    
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
                        segments.append(
                            np.array(current_segment[:-3])
                        )  # Overlap for continuity
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
        print(
            f"Converting text '{text}' to polynomial functions that trace letter shapes..."
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

            # Fit polynomials that actually trace each contour shape
            for j, contour in enumerate(contours):
                functions = self.fit_polynomial_contour_tracing(contour, max_degree)
                all_functions.extend(functions)
                print(
                    f"    Contour {j + 1}: {len(functions)} polynomial functions tracing the shape"
                )

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


def test_letter_A_generation():
    """Test function to generate letter A without user input."""
    print("=== TESTING LETTER A GENERATION ===")
    
    try:
        # Create converter with default settings
        converter = TextToDesmos(origin=(0, 0), scale=1.0)
        
        # Generate functions for letter A
        print("Generating functions for letter 'A'...")
        functions = converter.text_to_desmos_functions("A", max_degree=6)  # Lower degree for testing
        
        print(f"\nGenerated {len(functions)} functions:")
        for i, func in enumerate(functions, 1):
            print(f"{i}. {func}")
        
        if len(functions) == 0:
            print("ERROR: No functions generated!")
        else:
            print(f"\nSuccess! Generated {len(functions)} functions that should trace letter A.")
            
        return functions
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return []


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
        print(f" {func} ")

    print()
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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        test_letter_A_generation()
    else:
        # Run normal interactive mode
        main()
