"""Stroke-junction topology via medial-axis skeletonization.

The vertical fill sweep sees holes but misses stroke junctions inside
connected slabs (H, T, E, F). This module derives a generic stroke
graph from the canonical fill mask:

    fill mask -> Zhang-Suen skeleton -> pixel graph
              -> endpoints / junction nodes + chain edges

The skeleton is ONLY for route discovery: routes remain corridors
expanded by local stroke thickness elsewhere. No letter-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StrokeNode:
    id: int
    xy: tuple            # (x, y) in mask pixel coordinates (col, row)
    kind: str            # 'end' or 'junction'
    radius: float        # local stroke half-thickness (distance transform)


@dataclass
class StrokeEdge:
    id: int
    a: int               # node ids
    b: int
    points: np.ndarray   # polyline (n, 2) pixel coords along the chain


@dataclass
class StrokeGraph:
    nodes: list
    edges: list


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to a one-pixel-wide medial skeleton."""
    img = np.pad(mask.astype(np.uint8), 1)
    # neighbour offsets in cyclic order p2..p9 (row, col)
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)]

    def shift(a, dr, dc):
        return a[1 + dr: a.shape[0] - 1 + dr,
                 1 + dc: a.shape[1] - 1 + dc]

    while True:
        changed = False
        core = img[1:-1, 1:-1].astype(bool)
        for phase in (0, 1):
            ps = [shift(img, dr, dc) for dr, dc in offs]
            stacked = np.stack(ps)                     # 8 x H x W
            b = stacked.sum(axis=0)                    # neighbour count
            # number of 0->1 transitions around the cycle
            trans = np.zeros_like(b)
            for i in range(8):
                trans += ((ps[i] == 0) & (ps[(i + 1) % 8] == 1)).astype(np.uint8)
            c1 = (ps[0] & ps[2] & ps[4]) == 0          # p2 p4 p6
            c2 = (ps[2] & ps[4] & ps[6]) == 0          # p4 p6 p8
            if phase == 1:
                c1 = (ps[0] & ps[2] & ps[6]) == 0      # p2 p4 p8
                c2 = (ps[0] & ps[4] & ps[6]) == 0      # p2 p6 p8
            remove = (
                core & (b >= 2) & (b <= 6) & (trans == 1) & c1 & c2
            )
            if remove.any():
                view = img[1:-1, 1:-1]
                view[remove] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    padded = np.pad(skel.astype(np.uint8), 1)
    n = np.zeros_like(padded, dtype=np.uint8)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            n[1:-1, 1:-1] += padded[1 + dr: padded.shape[0] - 1 + dr,
                                    1 + dc: padded.shape[1] - 1 + dc]
    return n[1:-1, 1:-1]


def stroke_graph(mask: np.ndarray, min_spur: int = 4) -> StrokeGraph:
    """Clean skeleton graph: endpoint/junction nodes joined by chains.

    Junction pixel clusters collapse into single nodes; spurs shorter
    than `min_spur` pixels are pruned (rasterization artifacts).
    """
    from scipy import ndimage  # lazy: only needed for real glyphs

    skel = skeletonize(mask)
    ncount = _neighbour_count(skel)

    radius = ndimage.distance_transform_edt(mask)

    end_mask = skel & (ncount == 1)
    junc_mask = skel & (ncount >= 3)

    # collapse adjacent junction pixels into clusters
    lab, n_j = ndimage.label(junc_mask, structure=np.ones((3, 3)))
    centers = ndimage.center_of_mass(junc_mask, lab, range(1, n_j + 1))

    node_id = np.full(skel.shape, -1, dtype=int)
    nodes = []
    for cid, (r, c) in enumerate(centers):
        node_id[int(round(r)), int(round(c))] = len(nodes)
        rr, cc = np.nonzero(lab == cid + 1)
        rad = float(radius[rr, cc].max())
        nodes.append(StrokeNode(len(nodes), (float(c), float(r)),
                                "junction", rad))

    def ensure_end_node(r, c):
        nonlocal node_id
        nid = node_id[r, c]
        if nid < 0:
            node_id[r, c] = len(nodes)
            nid = node_id[r, c]
            nodes.append(StrokeNode(
                nid, (float(c), float(r)), "end",
                float(radius[r, c])))
        return int(nid)

    # ---- chain tracing -------------------------------------------------
    starts = [(int(r), int(c)) for r, c in zip(*np.nonzero(end_mask))]
    for r, c in starts:
        ensure_end_node(int(r), int(c))
    def neighbours(r, c):
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1]:
                    if skel[rr, cc]:
                        out.append((rr, cc))
        return out

    def is_node(p):
        return junc_mask[p] or end_mask[p]

    used = np.zeros(skel.shape, dtype=bool)
    raw_edges = []          # (node_a_id, entry_pixel_a, pts)

    def walk(sr, sc):
        """From node pixel (sr,sc) step into a plain chain pixel and
        follow unvisited skeleton pixels to the next node pixel."""
        firsts = [p for p in neighbours(sr, sc) if not used[p]
                  and not is_node(p)]
        for f in firsts:
            if used[f]:
                continue
            pts = [(sr, sc), f]
            used[f] = True
            r, c = f
            prev = (sr, sc)
            while not is_node((r, c)):
                nxt = [p for p in neighbours(r, c)
                       if p != prev and not is_node(p) and not used[p]]
                if not nxt:
                    # chain abuts a node pixel without stepping on it
                    node_nbrs = [p for p in neighbours(r, c) if is_node(p)]
                    if len(node_nbrs) == 1:
                        r, c = node_nbrs[0]
                        pts.append((r, c))
                    break
                # prefer the straight continuation
                dr, dc = r - prev[0], c - prev[1]
                nxt.sort(key=lambda p: abs(p[0] - r - dr) + abs(p[1] - c - dc))
                prev = (r, c)
                r, c = nxt[0]
                used[r, c] = True
                pts.append((r, c))
            yield pts

    # endpoints first (their chains end at junctions or other endpoints)
    for r, c in starts:
        for pts in walk(r, c):
            er, ec = pts[-1]
            if junc_mask[er, ec]:
                b = int(lab[er, ec]) - 1
                raw_edges.append((int(node_id[r, c]), b, pts))
            elif end_mask[er, ec]:
                b = ensure_end_node(er, ec)
                if b != int(node_id[r, c]):
                    raw_edges.append((int(node_id[r, c]), b, pts))

    # junction-to-junction chains via unvisited junction boundary pixels
    for r, c in zip(*np.nonzero(junc_mask)):
        nid_a = int(lab[int(r), int(c)]) - 1
        if node_id[int(r), int(c)] < 0:
            node_id[int(r), int(c)] = nid_a
        for pts in walk(int(r), int(c)):
            er, ec = pts[-1]
            if junc_mask[er, ec]:
                b = int(lab[er, ec]) - 1
            elif end_mask[er, ec]:
                b = ensure_end_node(er, ec)
            else:
                b = None
            if b is not None and b != nid_a:
                raw_edges.append((nid_a, b, pts))

    # prune short spurs (endpoint edges below min_spur)
    degree = {n.id: 0 for n in nodes}
    for a, b, _ in raw_edges:
        degree[a] += 1
        degree[b] += 1
    kept = []
    for a, b, pts in raw_edges:
        ea, eb = nodes[a].kind, nodes[b].kind
        length = len(pts)
        if ((ea == "end" and length < min_spur and degree[b] > 1)
                or (eb == "end" and length < min_spur and degree[a] > 1)):
            continue
        kept.append((a, b, pts))

    out_edges = []
    for a, b, pts in kept:
        arr = np.asarray([(float(cc), float(rr)) for rr, cc in pts])
        out_edges.append(StrokeEdge(len(out_edges), a, b, arr))

    used_nodes = sorted({e.a for e in out_edges} | {e.b for e in out_edges})
    remap = {old: new for new, old in enumerate(used_nodes)}
    final_nodes = [
        StrokeNode(remap[n.id], n.xy, n.kind, n.radius)
        for n in nodes if n.id in remap
    ]
    for e in out_edges:
        e.a = remap[e.a]
        e.b = remap[e.b]
    return StrokeGraph(nodes=final_nodes, edges=out_edges)
