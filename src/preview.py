"""Deterministic preview rendering from the *emitted* Denysko equations.

This helper renders an image of text by evaluating the exact public
serialization produced by :mod:`src.denysko` (the same ``y=f(x)`` lines a
user pastes into Desmos). It never renders the font outline, raster mask,
skeleton, route corridor, or internal ``PathFit`` objects.

The intended pipeline is::

    public Denysko CLI
        -> emitted y=f(x) equation lines
        -> evaluate those emitted equations
        -> sample them over the text viewport
        -> Matplotlib PNG

Use :func:`render_text_preview` for the mandatory ``Hello, World!`` smoke
test artifact.
"""

from __future__ import annotations

import numpy as np

from src import denysko as _d


def _body(line: str) -> str:
    """Strip an optional ``y=`` prefix, matching the public output form."""
    return line[2:] if line.startswith("y=") else line


def evaluate_line(line: str, xs: np.ndarray) -> np.ndarray:
    """Evaluate one emitted equation line at the given x samples.

    Uses the same tolerant evaluator the validators use, so the preview is
    faithful to what users receive (including the stable nested-Horner form
    for high-degree curves).
    """
    return _d.eval_expression(_body(line), xs)


def render_text_preview(
    text: str,
    out_path: str,
    *,
    seed: int = 42,
    letter_spacing: float = 0.15,
    space_width: float = 0.50,
    samples_per_unit: int = 600,
    y_pad: float = 0.35,
):
    """Render ``text`` from its emitted equations and save a PNG to ``out_path``.

    The y viewport includes the glyph band plus a margin and clips the
    intentionally unbounded escape tails after they leave the visible
    region. No domain restrictions are added to the equations themselves.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = _d.generate_text(
        text,
        seed=seed,
        letter_spacing=letter_spacing,
        space_width=space_width,
    )
    lines = _d.serialize_text(result)

    # Global x-extent of every drawn curve (its own translated corridor
    # window). Tails extend beyond this, but we sample exactly the drawn
    # window so the preview shows the recognizable letter body.
    xs_min = np.inf
    xs_max = -np.inf
    for placed, line in zip(result.placed_fits, lines):
        c = placed.fit.corridor
        a = c.xa + placed.dx
        b = c.xb + placed.dx
        xs_min = min(xs_min, a)
        xs_max = max(xs_max, b)

    fig, ax = plt.subplots(figsize=(max(6.0, (xs_max - xs_min) * 2.2), 3.2))
    ax.set_facecolor("white")
    for placed, line in zip(result.placed_fits, lines):
        c = placed.fit.corridor
        a = c.xa + placed.dx
        b = c.xb + placed.dx
        n = max(64, int(round((b - a) * samples_per_unit)))
        xs = np.linspace(a, b, n)
        ys = evaluate_line(line, xs)
        ax.plot(xs, ys, color="black", linewidth=2.0)

    ax.set_xlim(xs_min - 0.2, xs_max + 0.2)
    ax.set_ylim(-y_pad, 1.0 + y_pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    return lines
