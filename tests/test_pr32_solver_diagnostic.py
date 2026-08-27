"""Temporary focused diagnostic for PR #32; remove after solver fix."""

import numpy as np

from src import denysko as d
from src import fitting as f


def test_h_fits_when_chebyshev_basis_contains_tail_constraints(monkeypatch):
    """Issue #28 made corridor xa/xb semantic.  Those endpoints should not
    also force the numerical Chebyshev basis to evaluate far outside [-1,1].
    Expanding only the basis map to contain the mandatory tail checkpoints
    should make Cormorant H feasible without changing corridor geometry.
    """
    _, _, _, _, _, selected = d.build_phase1("H")
    assert selected

    pad = max(f.ESC_OFFSETS)

    def stable_zmap(x, xa, xb):
        a = xa - pad
        b = xb + pad
        return (2.0 * np.asarray(x, dtype=float) - a - b) / (b - a)

    monkeypatch.setattr(f, "_zmap", stable_zmap)
    fits = [f.fit_route(c, hi=80) for c in selected]
    print("H degrees with tail-inclusive Chebyshev domain:",
          [None if fit is None else fit.degree for fit in fits])
    assert all(fit is not None for fit in fits)
