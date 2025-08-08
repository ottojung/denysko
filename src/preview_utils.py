#!/usr/bin/env python3
"""
Preview and visualization utilities for centerline extraction.
"""


def preview_extracted_points(extractor, text, font_size=100, num_points=500, save_path=None):
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
        contours = extractor.extract_contour_points(path, num_points)
        for j, contour in enumerate(contours):
            if len(contour) > 0:
                ax.plot(
                    contour[:, 0],
                    contour[:, 1],
                    "r-",
                    linewidth=1.2,
                    label="Centerline" if j == 0 else None,
                )
                ax.plot(
                    contour[0, 0],
                    contour[0, 1],
                    "go",
                    markersize=6,
                    label="Start" if j == 0 else None,
                )
                ax.plot(
                    contour[-1, 0],
                    contour[-1, 1],
                    "bo",
                    markersize=6,
                    label="End" if j == 0 else None,
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
        skel = extractor.extract_skeleton_from_path(path)
        if len(skel) > 0:
            ax.plot(
                skel[:, 0],
                skel[:, 1],
                "r.-",
                markersize=3,
                linewidth=1.2,
                label="Skeleton",
            )
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
