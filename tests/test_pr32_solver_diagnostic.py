"""Temporary focused diagnostic for PR #32; remove after solver fix."""

import numpy as np

from src import denysko as d
from src import fitting as f


def test_h_known_corridor_fits_with_original_cap_when_basis_contains_tails(monkeypatch):
    """Keep issue #28 corridor geometry unchanged, but normalize the
    numerical Chebyshev basis across the stroke plus mandatory tails. The
    original degree-24 budget should then be tested directly instead of
    inflating the cap as a conditioning workaround.
    """
    _, _, _, _, _, selected = d.build_phase1("H")
    assert selected
    corridor = selected[0]

    pad = max(f.ESC_OFFSETS) * 1.02

    def stable_zmap(x, xa, xb):
        a = xa - pad
        b = xb + pad
        return (2.0 * np.asarray(x, dtype=float) - a - b) / (b - a)

    monkeypatch.setattr(f, "_zmap", stable_zmap)
    fit = f.fit_route(corridor, hi=24)
    assert fit is not None, "H path 0 still exceeds the original degree-24 cap"
