#!/usr/bin/env python3
"""
Preview and visualization utilities for centerline extraction.
"""


def preview_extracted_points(
    extractor, text, font_size=100, num_points=500, save_path=None
):
    """Render outline and extracted centerline for each character and save image."""
    import matplotlib.pyplot as plt

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
        plot_path_outline(ax, path, color="lightgray", alpha=0.8, label="Outline")
        
        # Get all traces
        traces = extractor.extract_skeleton_from_path(path)
        print(f"Character {i}: {len(traces)} separate traces")
        
        # Plot each trace with a different color
        colors = plt.cm.tab10(range(len(traces)))
        for j, trace in enumerate(traces):
            if len(trace) > 0:
                color = colors[j % len(colors)]
                ax.plot(
                    trace[:, 0],
                    trace[:, 1],
                    "-",
                    color=color,
                    linewidth=1.5,
                    label=f"Trace {j+1}" if j < 5 else None,  # Only label first 5 for readability
                    alpha=0.8
                )
                # Mark start point
                ax.plot(
                    trace[0, 0],
                    trace[0, 1],
                    "o",
                    color=color,
                    markersize=4,
                    alpha=0.9
                )
        
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
        plot_path_outline(ax, path, color="black", alpha=0.5, label="Outline")
        
        # Get all traces for detailed view
        traces = extractor.extract_skeleton_from_path(path)
        print(f"Character {i}: {len(traces)} separate traces")
        
        # Plot each trace with a different color and style
        colors = plt.cm.Set1(range(len(traces)))
        for j, trace in enumerate(traces):
            if len(trace) > 0:
                color = colors[j % len(colors)]
                ax.plot(
                    trace[:, 0],
                    trace[:, 1],
                    "-",
                    color=color,
                    linewidth=2,
                    label=f"Trace {j+1}" if j < 8 else None,
                    alpha=0.9
                )
                # Mark endpoints
                ax.plot(trace[0, 0], trace[0, 1], "o", color=color, markersize=6, alpha=0.9)
                ax.plot(trace[-1, 0], trace[-1, 1], "s", color=color, markersize=5, alpha=0.9)
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


def plot_path_outline(ax, path, color="blue", alpha=0.5, label="Outline"):
    """Plot the outline of a Path on provided axes, honoring codes if present."""
    vertices = path.vertices
    codes = path.codes
    if codes is None:
        ax.plot(vertices[:, 0], vertices[:, 1], color=color, alpha=alpha, label=label)
        return

    from matplotlib.path import Path as MPLPath

    label_used = False
    current = None
    for v, c in zip(vertices, codes):
        if c == MPLPath.MOVETO:
            current = v
        elif c == MPLPath.LINETO:
            if current is not None:
                ax.plot(
                    [current[0], v[0]],
                    [current[1], v[1]],
                    color=color,
                    alpha=alpha,
                    linewidth=1,
                    label=(label if not label_used else None),
                )
                label_used = True
            current = v
        elif c == MPLPath.CLOSEPOLY:
            # Close back to last MOVETO
            pass
