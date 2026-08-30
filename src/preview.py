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

    Uses the same evaluation path the validators use, so the preview is
    faithful to what a user pastes into Desmos:

    * raw power-basis lines (``a*x^2+b*x+c``) are parsed to their ordinary
      polynomial and evaluated directly;
    * the stable nested-Horner form used for high-degree curves (ordinary
      polynomial in ``x`` with explicit ``*``) is evaluated by the tolerant
      expression evaluator.

    No domain restriction is added to the equations themselves.
    """
    parsed = _d.parse_line(line)
    if parsed is not None:
        return parsed.poly(xs)
    return _d.eval_expression(_body(line), xs)


def text_viewport_xs(result, samples_per_unit: int = 600) -> np.ndarray:
    """Shared text-wide x viewport for every emitted curve.

    Returns a single, globally-increasing x sample array spanning the whole
    laid-out text: from the minimum global corridor start to the maximum
    global corridor end across all placed fits. Every curve is evaluated on
    this *same* array so the unbounded escape tails remain part of the drawn
    picture and are removed only by the fixed y viewport, never by trimming x
    per curve.
    """
    xs_min = np.inf
    xs_max = -np.inf
    for placed in result.placed_fits:
        c = placed.fit.corridor
        xs_min = min(xs_min, c.xa + placed.dx)
        xs_max = max(xs_max, c.xb + placed.dx)
    n = max(64, int(round((xs_max - xs_min) * samples_per_unit)))
    return np.linspace(xs_min, xs_max, n)


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

    Every emitted globally-unbounded equation is evaluated over ONE common
    text-wide x viewport (see :func:`text_viewport_xs`) and drawn on the same
    axes. The intentionally unbounded escape tails are clipped only by the
    fixed y viewport ``[-y_pad, 1 + y_pad]``; no per-curve x trimming is
    applied and no domain restriction is added to the equations themselves.
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

    # One shared x viewport for the whole laid-out text.
    xs = text_viewport_xs(result, samples_per_unit)
    xs_min, xs_max = float(xs[0]), float(xs[-1])

    fig, ax = plt.subplots(figsize=(max(6.0, (xs_max - xs_min) * 2.2), 3.2))
    ax.set_facecolor("white")
    for line in lines:
        ys = evaluate_line(line, xs)
        ax.plot(xs, ys, color="black", linewidth=2.0)

    # Keep BOTH requested data limits fixed. ``adjustable='datalim'`` would
    # silently expand one of them to satisfy equal aspect, violating the
    # preview contract's fixed y viewport. ``box`` changes only the axes box.
    ax.set_xlim(xs_min, xs_max)
    ax.set_ylim(-y_pad, 1.0 + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    return lines
