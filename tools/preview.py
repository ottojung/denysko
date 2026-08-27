"""Render a PNG preview of text from equations emitted by the public CLI.

Usage:
    uv run python tools/preview.py "Hello, World!" out.png [--seed 42]

Pipeline: public Denysko CLI -> emitted y=f(x) lines -> normalize into
safe Python arithmetic (implicit multiplication, ^ -> **) -> evaluate ->
dense x sampling -> Matplotlib PNG. The x viewport is derived
deterministically from the same layout semantics the CLI uses (glyph
visible widths, letter spacing, space width); the y viewport is fixed to
the glyph band. Escape tails are clipped by the viewports only; the
equations themselves stay globally unbounded.
"""

import argparse
import re
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.denysko import (
    DEFAULT_LETTER_SPACING,
    DEFAULT_SPACE_WIDTH,
    glyph_visible_width,
)

Y_MIN, Y_MAX = -0.5, 2.0
X_SAMPLE_STEP = 0.002


def emit_equations(text: str, seed: int) -> list[str]:
    result = subprocess.run(
        ["uv", "run", "denysko", text, "--seed", str(seed), "-q"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def layout_x_bounds(text: str) -> tuple[float, float]:
    """Deterministic x bounds from the CLI's own layout semantics."""
    cursor = 0.0
    x_hi = 0.0
    for ch in text:
        if ch == " ":
            cursor += DEFAULT_SPACE_WIDTH
            continue
        width = glyph_visible_width(ch)
        x_hi = max(x_hi, cursor + width)
        cursor += width + DEFAULT_LETTER_SPACING
    return 0.0, x_hi


_NUM_X = re.compile(r"(\d(?:[\d.]*(?:[eE][+-]?\d+)?)?)x")  # 94.3x -> 94.3*x
_POW = re.compile(r"\^")  # x^2 -> x**2


def normalize(expr: str) -> str:
    """Normalize a CLI display expression into Python arithmetic.

    Purely syntactic: inserts explicit multiplication between a numeric
    literal and x, and rewrites ^ as **. Semantics are unchanged, so the
    nested shifted Horner forms evaluate identically.
    """
    return _POW.sub("**", _NUM_X.sub(r"\1*x", expr))


def evaluate_line(line: str, xs: np.ndarray) -> np.ndarray:
    env = {"__builtins__": {}}
    env["x"] = xs
    body = line[2:] if line.startswith("y=") else line
    return eval(normalize(body), env)  # noqa: S307


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = emit_equations(args.text, args.seed)
    print(f"curves: {len(lines)}")

    x_lo, x_hi = layout_x_bounds(args.text)
    pad = 0.05 * (x_hi - x_lo) or 0.5
    xs = np.arange(x_lo - pad, x_hi + pad + X_SAMPLE_STEP, X_SAMPLE_STEP)
    points = []
    for line in lines:
        ys = evaluate_line(line, xs)
        mask = (ys >= Y_MIN) & (ys <= Y_MAX)
        points.append((xs[mask], ys[mask]))

    fig, ax = plt.subplots(figsize=(16, 4))
    for x, y in points:
        ax.plot(x, y, linewidth=2.5)
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
