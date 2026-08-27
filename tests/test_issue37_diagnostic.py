import numpy as np

from src import denysko as d
from src.fitting import preferred_tail_orientation
from src.topology import route_join_score


def test_dump_issue37_route_geometry():
    for letter in ("t", "e", "r", "C", "A", "H"):
        geom, graph, candidates, chosen, sigs, selected = d.build_phase1(letter)
        print(f"LETTER {letter!r} glyph=[{geom.ymin:.6f},{geom.ymax:.6f}]")
        for i, (route, corr) in enumerate(zip(chosen, selected)):
            center = 0.5 * (corr.lower + corr.upper)
            pts = corr.path.points
            xspan = float(np.ptp(pts[:, 0]))
            yspan = float(np.ptp(pts[:, 1]))
            print(
                "ROUTE", i,
                "ori=", preferred_tail_orientation(corr),
                "join=", route_join_score(graph, route),
                "xspan=", xspan,
                "yspan=", yspan,
                "verticality=", yspan / max(xspan, 1e-12),
                "center_mean=", float(np.mean(center)),
                "center_median=", float(np.median(center)),
                "center_q25_q75=", tuple(float(x) for x in np.quantile(center, [0.25, 0.75])),
                "end_lo_hi=", (
                    float(corr.lower[0]), float(corr.upper[0]),
                    float(corr.lower[-1]), float(corr.upper[-1]),
                ),
                "local=", (float(corr.ylo_local), float(corr.yhi_local)),
                "steps=", [graph.physical_atom(s.edge_id) for s in route.steps],
            )
    assert False, "diagnostic only"
