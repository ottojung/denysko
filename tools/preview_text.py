"""Deterministic preview: render a PNG of text from the equations the
public Denysko CLI actually emits.

Pipeline (per docs/constitution.md):
  public denysko CLI -> emitted y=f(x) lines -> parse/evaluate ->
  dense x sampling -> Matplotlib PNG

Usage:
    uv run python tools/preview_text.py "Hello, World!" out.png [--seed N]

The y viewport is fixed to the glyph band; escape tails are clipped
only by the viewport, never by domain restrictions. Each emitted
equation is sampled over its own placed corridor window (plus a small
margin so the escape direction stays visible); outside that window the
unbounded polynomial is not part of the glyph, and sampling it there
would draw out-of-domain oscillation artifacts.
"""

import argparse
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.denysko import eval_expression, generate_text, parse_line


def emitted_lines(text: str, seed: int) -> list[str]:
    """Exact stdout of the public CLI for this text and seed."""
    proc = subprocess.run(
        ["uv", "run", "denysko", text, "--seed", str(seed), "-q"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip())
    return proc.stdout.splitlines()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines = emitted_lines(args.text, args.seed)
    result = generate_text(args.text, seed=args.seed)
    placed = list(result.placed_fits)
    if len(placed) != len(lines):
        raise SystemExit(
            f"CLI emitted {len(lines)} lines for {len(placed)} curves")

    if placed:
        lo = min(p.dx for p in placed)
        hi = max(p.dx + float(np.asarray(p.fit.corridor.xs).max())
                 for p in placed)
    else:
        lo, hi = 0.0, 1.0
    x0, x1 = lo - 0.3, hi + 1.3   # right margin covers escape tails
    Y_LO, Y_HI = -0.15, 1.15      # glyph band plus small margin

    fig_w = max(8.0, 1.4 * (hi - lo))
    fig, ax = plt.subplots(figsize=(fig_w, 2.2))
    for line, p in zip(lines, placed):
        if not line.startswith("y="):
            raise SystemExit(f"unexpected emitted line: {line[:60]!r}")
        cxs = np.asarray(p.fit.corridor.xs, dtype=float) + p.dx
        xs = np.linspace(float(cxs.min()) - 0.12,
                         float(cxs.max()) + 0.12, 4000)
        curve = parse_line(line)
        try:
            if curve is not None:
                ys = curve.poly(xs)
            else:
                ys = eval_expression(line[2:], xs)
        except SyntaxError:
            raise SystemExit(
                f"unexpected emitted line: {line[:60]!r}") from None
        seg_lo = np.maximum(ys[:-1], ys[1:])
        seg_hi = np.minimum(ys[:-1], ys[1:])
        visible = ((seg_hi <= Y_HI) & (seg_lo >= Y_LO)
                   & np.isfinite(seg_hi) & np.isfinite(seg_lo)
                   & (np.abs(np.diff(ys)) <= 5.0))
        for i in np.nonzero(visible)[0]:
            ax.plot(xs[i:i + 2], ys[i:i + 2], "k-", lw=1.0)

    ax.set_xlim(x0, x1)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.png, dpi=150)
    print(f"wrote {args.png} ({len(lines)} curves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
