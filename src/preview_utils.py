#!/usr/bin/env python3
"""
Preview and visualization utilities for centerline extraction.
"""


def preview_extracted_points(
    extractor, text, font_size=100, num_points=500, save_path=None
):
    """Render outline and extracted centerline for each character and save image."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    print(f"Generating preview for text: '{text}'")
    paths = extractor.text_to_paths(text, font_size)
    if not paths:
        print("No paths generated for preview")
        return

    n = len(paths)
    fig, axes = plt.subplots(1, max(1, n), figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    for i, path in enumerate(paths):
        ax = axes[i]
        # Draw outline and set axis limits from glyph bounds
        _plot_outline_with_bounds(ax, path, color="lightgray", alpha=0.8, label="Outline")

        # Get all traces
        traces = extractor.extract_skeleton_from_path(path)
        print(f"Character {i}: {len(traces)} separate traces")

        # Debug stats: trace lengths
        lengths = []
        for t in traces:
            if len(t) >= 2:
                seg = np.sqrt(np.sum((t[1:] - t[:-1]) ** 2, axis=1))
                lengths.append(float(seg.sum()))
            else:
                lengths.append(0.0)
        if lengths:
            arr = np.array(lengths)
            print(
                f"Trace length stats -> min: {arr.min():.3f}, med: {np.median(arr):.3f}, max: {arr.max():.3f}, total: {arr.sum():.3f}"
            )

        # Build a LineCollection from all traces
        segments = [t.astype(float) for t in traces if len(t) >= 2]
        if segments:
            # Color by index to distinguish paths
            cmap = plt.cm.get_cmap("tab20")
            colors = [cmap(j % 20) for j in range(len(segments))]
            lc = LineCollection(segments, colors=colors, linewidths=1.8, alpha=0.9, zorder=5)
            ax.add_collection(lc)
        else:
            print("Warning: No trace segments with >=2 points to draw.")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Preview saved to: {save_path}")
    plt.close("all")


def preview_skeleton_extraction_steps(extractor, text, font_size=100, save_path=None):
    """Minimal step preview: show outline and final skeleton per character."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    print(f"Generating detailed skeleton preview for: '{text}'")
    paths = extractor.text_to_paths(text, font_size)
    if not paths:
        print("No paths generated for skeleton preview")
        return

    n = len(paths)
    fig, axes = plt.subplots(1, max(1, n), figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for i, path in enumerate(paths):
        ax = axes[i]
        # Draw outline and set axis limits from glyph bounds
        _plot_outline_with_bounds(ax, path, color="black", alpha=0.5, label="Outline")

        # Get all traces for detailed view
        traces = extractor.extract_skeleton_from_path(path)
        print(f"Character {i}: {len(traces)} separate traces")

        # Debug stats: trace lengths
        lengths = []
        for t in traces:
            if len(t) >= 2:
                seg = np.sqrt(np.sum((t[1:] - t[:-1]) ** 2, axis=1))
                lengths.append(float(seg.sum()))
            else:
                lengths.append(0.0)
        if lengths:
            arr = np.array(lengths)
            print(
                f"Trace length stats -> min: {arr.min():.3f}, med: {np.median(arr):.3f}, max: {arr.max():.3f}, total: {arr.sum():.3f}"
            )

        # Plot each trace via a LineCollection (better for many polylines)
        segments = [t.astype(float) for t in traces if len(t) >= 2]
        if segments:
            cmap = plt.cm.get_cmap("Set3")
            colors = [cmap(j % cmap.N) for j in range(len(segments))]
            lc = LineCollection(segments, colors=colors, linewidths=2.2, alpha=0.95, zorder=6)
            ax.add_collection(lc)
        else:
            print("Warning: No trace segments with >=2 points to draw.")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Skeleton preview saved to: {save_path}")
    plt.close("all")


def _plot_outline_with_bounds(ax, path, color="blue", alpha=0.5, label="Outline"):
    """Plot the outline as a PathPatch and set axis limits from bounding box."""
    import numpy as np
    from matplotlib.patches import PathPatch

    # Add the path as a patch (handles Beziers correctly)
    patch = PathPatch(path, facecolor="none", edgecolor=color, alpha=alpha, linewidth=1, label=label, zorder=3)
    ax.add_patch(patch)

    # Set limits based on path bounding box with padding
    vertices = path.vertices
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    w = max_x - min_x
    h = max_y - min_y
    pad = 0.05 * max(w, h) if max(w, h) > 0 else 1.0
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
