"""Regression tests for issue #37: dense family generation must not
rediscover topology. Every family member emitted for a structural path must
share the exact tail orientation that the ordinary single-curve (base) fit
selects for that path. The optimizer may never flip an escape direction merely
because another orientation admits a cheaper polynomial family.
"""

from collections import defaultdict

import numpy as np

from src import denysko as d
from src.fitting import INITIAL_FIT_DEGREE, fit_route


def _base_orientations(letter):
    geom, graph, candidates, chosen, signatures, selected = \
        d.build_phase1(letter)
    base = [fit_route(c, hi=INITIAL_FIT_DEGREE).orientation for c in selected]
    return geom, graph, chosen, selected, base


def _family_members_match_base_orientation(letter, min_curves, seed):
    """Return True when every emitted family member for each path matches
    that path's base (single-curve) fit orientation."""
    geom, graph, chosen, selected, base = _base_orientations(letter)
    K = len(selected)
    counts = d.allocate_counts(K, max(K, min_curves),
                               np.random.default_rng(seed))
    fits, corrs, routes = d.realize_variants(
        graph, chosen, selected, counts, seed, geom)

    # Map each emitted fit to its structural path via route identity, then
    # compare the emitted orientation to the base orientation for that path.
    path_index = {id(r): i for i, r in enumerate(chosen)}
    by_path = defaultdict(list)
    for f, r in zip(fits, routes):
        by_path[path_index[id(r)]].append(f.orientation)
    for i, oris in by_path.items():
        if any(o != base[i] for o in oris):
            return False
    return True


def test_family_members_constrained_to_base_orientation():
    """Mechanism regression: for every structural path the dense family
    search may only produce curves whose orientation equals the path's
    geometry-selected base fit. The family search is not allowed to
    independently rediscover a different (topology-changing) orientation."""
    # Letters whose structural paths have clearly distinguishable base
    # orientations (opposite escape directions on different routes) so a
    # flipped orientation would be visible.
    for letter in ["t", "e", "C", "A", "r", "i", "j", "f", "l", "h", "k"]:
        ok = _family_members_match_base_orientation(
            letter, min_curves=12, seed=42)
        assert ok, (
            f"issue #37: dense family for {letter!r} produced a family "
            f"member whose tail orientation differs from the path's base "
            f"fit orientation")


def test_issue37_t_min_curves_family_keeps_base_orientation():
    """Real-glyph regression: uv run denysko t --seed 42 --min-curves 10
    must emit family members that keep the same per-route orientation as the
    ordinary base fits (uv run denysko t --seed 42)."""
    geom, graph, chosen, selected, base = _base_orientations("t")
    K = len(selected)

    # Ordinary base fits establish the geometry-selected orientation for each
    # structural path. The canonical font may change K and the concrete
    # orientation vector; issue #37 only requires dense members to preserve
    # whatever the ordinary fit selected for that same path.
    fits_plain, _, _ = d.realize_variants(
        graph, chosen, selected,
        d.allocate_counts(K, K, np.random.default_rng(42)),
        seed=42, geom=geom)
    plain_ori = sorted(f.orientation for f in fits_plain)
    assert plain_ori == sorted(base)

    fits_dense, corrs_dense, routes_dense = d.realize_variants(
        graph, chosen, selected,
        d.allocate_counts(K, 10, np.random.default_rng(42)),
        seed=42, geom=geom)
    assert len(fits_dense) == 10

    path_index = {id(r): i for i, r in enumerate(chosen)}
    by_path = defaultdict(list)
    for f, r in zip(fits_dense, routes_dense):
        by_path[path_index[id(r)]].append(f.orientation)
    for i, oris in by_path.items():
        assert all(o == base[i] for o in oris), (
            f"t path {i}: dense family orientation {sorted(oris)} != "
            f"base orientation {base[i]}")


def test_solve_family_anchors_honors_required_orientation():
    """Mechanism regression: the family solver receives exactly one
    geometry-selected orientation and must only ever return a family in
    that orientation (never rediscover one of the other three). It takes
    the actual base fit's degree and orientation, not a value derived from
    orientation signs."""
    geom, graph, candidates, chosen, signatures, selected = \
        d.build_phase1("t")
    # Use a real structural path, but do not pin its concrete orientation to
    # one font. The mechanism contract is that the family solver honors the
    # exact orientation supplied by that path's ordinary fit.
    path_index = min(1, len(selected) - 1)
    c = selected[path_index]
    base_fit = fit_route(c, hi=INITIAL_FIT_DEGREE)
    base = base_fit.orientation
    fam = d.solve_family_anchors(
        graph, chosen[path_index], c, 42, path_index,
        max(1, base_fit.degree), required_orientation=base)
    assert fam is not None
    assert fam[3] == base
