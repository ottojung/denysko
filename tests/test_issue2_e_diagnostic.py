import numpy as np

from src import denysko as d
from src.fitting import (
    INITIAL_FIT_DEGREE,
    fit_degree,
    preferred_tail_orientation,
    tail_reentry_violation_cheb,
)
from src.topology import route_join_score


def _first_feasible(corr, ori):
    for degree in range(INITIAL_FIT_DEGREE + 1):
        fit = fit_degree(corr, degree, *ori)
        if fit is None:
            continue
        if tail_reentry_violation_cheb(fit.coef_cheb, corr, ori) == 0.0:
            return degree
    return None


def test_dump_issue2_route_geometry():
    for letter in ("e", "r", "C", "A", "!", "i"):
        geom, graph, candidates, chosen, sigs, selected = d.build_phase1(letter)
        print(f"LETTER {letter!r} glyph=[{geom.ymin:.6f},{geom.ymax:.6f}]")
        for i, (route, corr) in enumerate(zip(chosen, selected)):
            center = 0.5 * (corr.lower + corr.upper)
            endpoint_ori = preferred_tail_orientation(corr)
            join = route_join_score(graph, route)
            dominant_sigma = 1 if float(np.median(center)) >= 0.5 * (corr.ylo + corr.yhi) else -1
            dominant_ori = (dominant_sigma, dominant_sigma)
            print(
                "ROUTE",
                i,
                "ori=", endpoint_ori,
                "join=", join,
                "dominant_ori=", dominant_ori,
                "dominant_degree=", _first_feasible(corr, dominant_ori) if join else None,
                "end_lo_hi=", (
                    float(corr.lower[0]), float(corr.upper[0]),
                    float(corr.lower[-1]), float(corr.upper[-1]),
                ),
                "local=", (float(corr.ylo_local), float(corr.yhi_local)),
                "center_mean=", float(np.mean(center)),
                "center_median=", float(np.median(center)),
                "center_q25_q75=", tuple(float(x) for x in np.quantile(center, [0.25, 0.75])),
                "steps=", [graph.physical_atom(s.edge_id) for s in route.steps],
            )
    assert False, "diagnostic only"
