#!/usr/bin/env python3
"""
Preview script for visualizing extracted centerline points.
Run this to see how the zero-width skeleton extraction works.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    from .text_extractor import TextExtractor
    from .preview_utils import plot_path_outline
    import matplotlib.pyplot as plt
    import matplotlib

    # Use non-interactive backend if no display available
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the src/ directory")
    sys.exit(1)


def preview_text(text, font_size=100, num_points=500):
    """
    Preview centerline extraction for given text.

    Args:
        text (str): Text to preview
        font_size (int): Font size for rendering
        num_points (int): Number of centerline points to extract
    """
    print(f"=== Centerline Preview for '{text}' ===")
    print(f"Font size: {font_size}")
    print(f"Points per character: {num_points}")
    print()

    extractor = TextExtractor()

    # Basic preview
    print("Generating basic preview...")
    extractor.preview_extracted_points(
        text,
        font_size=font_size,
        num_points=num_points,
        save_path=f"preview_basic_{text}.png",
    )

    # Detailed skeleton preview
    print("Generating detailed skeleton extraction preview...")
    extractor.preview_skeleton_extraction_steps(
        text, font_size=font_size, save_path=f"preview_skeleton_{text}.png"
    )

    print("Preview complete!")


def compare_different_approaches(text):
    """
    Compare different numbers of points to see the effect.
    """
    print(f"=== Comparing Different Point Counts for '{text}' ===")

    extractor = TextExtractor()
    point_counts = [50, 200, 500]

    fig, axes = plt.subplots(1, len(point_counts), figsize=(15, 5))

    for i, num_points in enumerate(point_counts):
        ax = axes[i]

        # Extract paths
        paths = extractor.text_to_paths(text, font_size=100)

        if paths:
            path = paths[0]  # Use first character

            # Plot original outline faintly
            plot_path_outline(ax, path, color="lightgray", alpha=0.5)

            # Extract and plot centerline
            contours = extractor.extract_contour_points(path, num_points)

            if contours and len(contours[0]) > 0:
                contour = contours[0]
                ax.plot(
                    contour[:, 0],
                    contour[:, 1],
                    "ro-",
                    markersize=2,
                    linewidth=1,
                    alpha=0.8,
                )

                # Mark start and end
                ax.plot(contour[0, 0], contour[0, 1], "go", markersize=8, label="Start")
                ax.plot(contour[-1, 0], contour[-1, 1], "bo", markersize=8, label="End")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{num_points} Points")
        ax.invert_yaxis()

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.suptitle(f"Point Count Comparison: '{text}'", fontsize=14, y=1.05)
    plt.savefig(f"comparison_{text}.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Comparison saved as 'comparison_{text}.png'")


if __name__ == "__main__":
    # Parse command line arguments
    text_to_preview = sys.argv[1] if len(sys.argv) > 1 else "C"
    font_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    num_points = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    # Generate previews
    preview_text(text_to_preview, font_size, num_points)

    # Also generate comparison
    compare_different_approaches(text_to_preview)
