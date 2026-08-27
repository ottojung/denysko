"""Temporary focused diagnostic for PR #32; remove after solver fix."""

import numpy as np

from src import denysko as d
from src import fitting as f


def test_h_known_corridor_fits_when_chebyshev_basis_contains_tails(monkeypatch):
    """Issue #28 made corridor xa/xb semantic. Those endpoints should not
    also force the numerical Chebyshev basis to evaluate far outside [-1,1].
    Keep corridor geometry unchanged but normalize the numerical basis across
    the stroke plus mandatory tail checkpoints.
    """
    _, _, _, _, _, selected = d.build_phase1("H")
    assert selected
    corridor = selected[0]
    orientation = f.preferred_tail_orientation(corridor)

    pad = max(f.ESC_OFFSETS)

    def stable_zmap(x, xa, xb):
        a = xa - pad
        b = xb + pad
        return (2.0 * np.asarray(x, dtype=float) - a - b) / (b - a)

    monkeypatch.setattr(f, "_zmap", stable_zmap)
    fit = f.fit_degree(corridor, 60, *orientation)
    print("H path 0 degree 60 tail-inclusive basis fit:", fit is not None)
    assert fit is not None
