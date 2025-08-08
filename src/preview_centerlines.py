#!/usr/bin/env python3
"""
Preview script for visualizing extracted centerline points.
Run with: python -m src.preview_centerlines "A"
"""

import sys
import os

import matplotlib

# Use non-interactive backend if no display available
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .text_extractor import TextExtractor
from .preview_utils import _plot_outline_with_bounds


def preview_text(text, font_size=100, num_points=500):
    """
    Preview centerline extraction for given text.

    Args:
        text (str): Text to preview
        font_size (int): Font size for rendering
        num_points (int): Unused placeholder kept for CLI compatibility
    """
    print(f"=== Centerline Preview for '{text}' ===")
    print(f"Font size: {font_size}")
    print(f"Points per character (ignored): {num_points}")
    print()

    extractor = TextExtractor()

    # Basic preview (per-character outline + all traces)
    print("Generating basic preview...")
    extractor.preview_extracted_points(
        text,
        font_size=font_size,
        num_points=num_points,
        save_path=f"preview_basic_{text}.png",
    )

    # Detailed skeleton preview (same drawing with different styling)
    print("Generating detailed skeleton extraction preview...")
    extractor.preview_skeleton_extraction_steps(
        text, font_size=font_size, save_path=f"preview_skeleton_{text}.png"
    )

    print("Preview complete!")


def compare_different_approaches(text):
    """
    Simple comparison panel drawing the current extractor output multiple times.
    Kept for parity with previous CLI; the numeric labels do not change behavior.
    """
    print(f"=== Comparison Panel for '{text}' ===")

    extractor = TextExtractor()
    point_counts = [50, 200, 500]

    n = len(point_counts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    paths = extractor.text_to_paths(text, font_size=100)

    for i, num_points in enumerate(point_counts):
        ax = axes[i]
        ax.set_title(f"Panel {i + 1} ({num_points} pts label)")

        if paths:
            path = paths[0]  # Use first character only

            # Plot original outline faintly and set bounds
            _plot_outline_with_bounds(
                ax, path, color="lightgray", alpha=0.5, label="Outline"
            )

            # Extract and plot all traces for this character
            traces = extractor.extract_skeleton_from_path(path)

            segments = [t.astype(float) for t in traces if len(t) >= 2]
            if segments:
                cmap = plt.cm.get_cmap("tab20")
                colors = [cmap(j % 20) for j in range(len(segments))]
                lc = LineCollection(
                    segments, colors=colors, linewidths=1.8, alpha=0.9, zorder=5
                )
                ax.add_collection(lc)
            else:
                print(
                    "Warning: No trace segments with >=2 points to draw in comparison panel."
                )

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.suptitle(f"Extractor comparison: '{text}'", fontsize=14, y=1.02)
    out_path = f"comparison_{text}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison saved as '{out_path}'")


if __name__ == "__main__":
    # Parse command line arguments
    text_to_preview = sys.argv[1] if len(sys.argv) > 1 else "C"
    font_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    num_points = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    # Generate previews
    preview_text(text_to_preview, font_size, num_points)

    # Also generate comparison panel
    compare_different_approaches(text_to_preview)
