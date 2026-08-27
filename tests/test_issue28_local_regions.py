"""Regression tests for issue #28: curves may occupy local vertical
regions within a glyph.

Before the fix, the Chebyshev normalization domain was padded wider than
the corridor's real slice-interval x-range (`xa/xb` exceeded `xs`). That
collapsed the meaningful interior into a tiny central slice of z in [-1,1]
while leaving the escape tails in the same central region, so a polynomial
had almost no z-resolution across the actual stroke and swung far outside
the stroke's natural vertical extent - e.g. the dot of `!` rose/dove well
beyond its small localized band. The fix makes `xa/xb` equal the corridor
constraint region and records each corridor's own vertical band
(`ylo_local`, `yhi_local`), so every curve preserves local vertical
occupancy instead of being forced across the whole glyph height.

The mechanism is generic (no character-specific hacks): it applies to any
route, and the issue's punctuation examples (. , : ; and the dot of !) are
covered both by a generic local-occupancy check and by a specific example.
"""

import numpy as np
import pytest

from src import denysko as d
from src import topology as d_topology
from src import fitting as _fitting
from src.topology import (
    glyph_geometry,
    build_stroke_route_graph,
    enumerate_complete_routes,
    select_routes_min_cover,
    build_route_corridor,
)
from src.denysko import corridor_adherence_violation


def _local_occupancy_max_violation(fit, corr):
    """Largest amount by which the emitted curve leaves its corridor's OWN
    local vertical band [ylo_local, yhi_local] over the corridor x-window.

    Escape tails live outside the x-window and are checked separately by
    V3; this isolates the "local vertical region" property of issue #28.
    """
    xs = np.linspace(corr.xs[0], corr.xs[-1], 400)
    ys = np.polynomial.Polynomial(np.asarray(fit.poly.coef))(xs)
    return float(max(float(corr.ylo_local - ys.min()),
                     float(ys.max() - corr.yhi_local), 0.0))


def test_corridor_chebyshev_domain_is_constraint_region():
    """The Chebyshev normalization domain must equal the corridor's real
    slice-interval x-range (xa == xs[0], xb == xs[-1]). Padding the domain
    wider than `xs` is the bug that let curves span the whole glyph."""
    for ch in ["A", "H", "o", ".", "!", ":", ";"]:
        geom = glyph_geometry(ch)
        graph = build_stroke_route_graph(geom)
        cands = enumerate_complete_routes(graph)
        idx = select_routes_min_cover(graph, cands)
        for j in idx:
            c = build_route_corridor(graph, cands[j], geom)
            assert abs(c.xa - float(c.xs[0])) < 1e-9, (ch, j, c.xa, c.xs[0])
            assert abs(c.xb - float(c.xs[-1])) < 1e-9, (ch, j, c.xb, c.xs[-1])
            # local vertical region is recorded per corridor
            assert c.ylo_local <= c.yhi_local + 1e-9


@pytest.mark.parametrize("ch", [".", ",", ":", ";", "!"])
def test_punctuation_curves_preserve_local_vertical_regions(ch):
    """Punctuation features (. , : ; and the dot of !) occupy only their
    own local vertical interval, not the whole glyph. Each emitted curve
    stays inside its corridor's local band over the corridor x-window and
    still passes V2 (corridor adherence) and V3 (permanent tail escape)."""
    from src.fitting import tail_reentry_violation_cheb

    geom, graph, cands, chosen, sigs, sel = d.build_phase1(ch)
    fits, corrs, routes = d.generate_letter(ch, min_curves=1)
    assert len(fits) == len(corrs) == len(sel)
    for fit, corr in zip(fits, corrs):
        # generic mechanism: curve occupies only its local vertical region
        assert _local_occupancy_max_violation(fit, corr) < 1e-6, (
            ch, fit.poly.coef)
        # existing validators are preserved (no weakening)
        v2 = corridor_adherence_violation(np.asarray(fit.poly.coef), corr)
        assert v2 <= d_topology.CORRIDOR_EPS, (ch, v2)
        v3 = tail_reentry_violation_cheb(
            np.asarray(fit.coef_cheb), corr, fit.orientation)
        assert v3 == 0.0, (ch, v3)


def test_exclamation_dot_stays_in_glyph_bottom_region():
    """Issue #28's explicit example: the dot of `!` should not rise above
    roughly the bottom 10% of the glyph. The dot is the route whose local
    band sits lowest; its emitted curve's window-maximum y must stay
    within the glyph's bottom 10%."""
    geom, graph, cands, chosen, sigs, sel = d.build_phase1("!")
    fits, corrs, routes = d.generate_letter("!", min_curves=1)
    glyph_top = geom.ymax
    # the dot route is the one with the smallest local upper bound
    dot_i = min(range(len(corrs)), key=lambda k: corrs[k].yhi_local)
    xs = np.linspace(corrs[dot_i].xs[0], corrs[dot_i].xs[-1], 400)
    ys = np.polynomial.Polynomial(
        np.asarray(fits[dot_i].poly.coef))(xs)
    assert float(ys.max()) <= 0.10 * glyph_top + 1e-9, (
        float(ys.max()), glyph_top)
