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


_OFFS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
         (1, 0), (1, -1), (0, -1), (-1, -1)]


def _shift(base, dr, dc):
    return base[1 + dr: base.shape[0] - 1 + dr,
                1 + dc: base.shape[1] - 1 + dc]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to a one-pixel-wide medial skeleton."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        core = img[1:-1, 1:-1].astype(bool)
        for phase in (0, 1):
            ps = [_shift(img, dr, dc) for dr, dc in _OFFS]
            b = ps[0] + ps[1] + ps[2] + ps[3] + ps[4] + ps[5] + ps[6] + ps[7]
            trans = np.zeros_like(b)
            for i in range(8):
                trans += ((ps[i] == 0)
                          & (ps[(i + 1) % 8] == 1)).astype(np.uint8)
            if phase == 0:
                c1 = (ps[0] & ps[2] & ps[4]) == 0
                c2 = (ps[2] & ps[4] & ps[6]) == 0
            else:
                c1 = (ps[0] & ps[2] & ps[6]) == 0
                c2 = (ps[0] & ps[4] & ps[6]) == 0
            remove = core & (b >= 2) & (b <= 6) & (trans == 1) & c1 & c2
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _pixel_stats(skel: np.ndarray):
    """Return (end_mask, junc_mask, junc_labels, n_junction_clusters).

    True junctions have >=3 neighbour GROUPS around the pixel cycle, so
    staircase corners (one contiguous arc) do not count."""
    base = np.pad(skel.astype(np.uint8), 1)
    ps = [_shift(base, dr, dc) for dr, dc in _OFFS]
    ncount = np.stack(ps).sum(axis=0)
    end_mask = skel & (ncount == 1)
    groups = np.zeros(skel.shape, dtype=np.uint8)
    for i in range(8):
        groups += ((ps[i] == 0) & (ps[(i + 1) % 8] == 1)).astype(np.uint8)
    junc_mask = skel & (groups >= 3)

    from scipy import ndimage  # lazy

    lab, n_j = ndimage.label(junc_mask, structure=np.ones((3, 3)))
    return end_mask, junc_mask, lab, n_j


def _polyline_len(pts):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    seg = np.diff(pts, axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def _contract(nodes, edges, min_len):
    """Merge the endpoints of every edge shorter than min_len pixels
    (junction representative wins); drop resulting self-loops."""
    changed = True
    while changed:
        changed = False
        for e in sorted(edges, key=lambda t: t[0]):
            _, a, b, pts = e
            if a not in nodes or b not in nodes:
                edges.remove(e)
                changed = True
                break
            if a == b or _polyline_len(pts) >= min_len:
                continue
            ka = nodes[a].kind if a in nodes else None
            kb = nodes[b].kind if b in nodes else None
            keep, drop = a, b
            if ka != "junction" and kb == "junction":
                keep, drop = b, a
            del nodes[drop]
            for f in edges:
                if f[1] == drop:
                    f[1] = keep
                if f[2] == drop:
                    f[2] = keep
            edges[:] = [f for f in edges
                        if not (f[0] == e[0] or f[1] == f[2])]
            changed = True
            break
    return nodes, edges


def stroke_graph(mask: np.ndarray, min_spur: int = 6,
                 min_edge_px: int = 14) -> StrokeGraph:
    """Clean skeleton graph: endpoint/junction nodes joined by chains.

    Junction pixel clusters collapse into single nodes; rasterization
    noise is removed by contracting edges shorter than `min_edge_px`
    and pruning endpoint spurs shorter than `min_spur`.
    """
    from scipy import ndimage  # lazy

    skel = skeletonize(mask)
    radius = ndimage.distance_transform_edt(mask)

    # A compact filled component can have a perfectly legitimate
    # zero-dimensional medial axis: thinning a round dot may leave one
    # pixel (or, for an even/symmetric component, no pixel at all).  Such
    # a component is still real glyph geometry, but an edge-based stroke
    # graph cannot route it.  Give only these otherwise-unroutable fill
    # components a deterministic interior chord through a maximally-inside
    # row.  This is a topology fallback, not glyph-specific dilation: normal
    # components with a usable skeleton are left byte-for-byte unchanged.
    fill_lab, fill_n = ndimage.label(mask, structure=np.ones((3, 3)))
    for cid in range(1, int(fill_n) + 1):
        rr, cc = np.nonzero(fill_lab == cid)
        if len(rr) == 0 or int(np.count_nonzero(skel[rr, cc])) >= 2:
            continue
        k = int(np.argmax(radius[rr, cc]))
        r0, c_mid = int(rr[k]), int(cc[k])
        row = fill_lab[r0, :] == cid
        lo = c_mid
        while lo > 0 and row[lo - 1]:
            lo -= 1
        hi = c_mid
        while hi + 1 < row.size and row[hi + 1]:
            hi += 1
        width = hi - lo + 1
        if width < 3:
            continue
        inset = max(1, width // 5)
        a, b = lo + inset, hi - inset
        if b - a < 2:
            a, b = lo, hi
        skel[r0, a:b + 1] = True

    end_mask, junc_mask, lab, _ = _pixel_stats(skel)

    nodes: dict[int, StrokeNode] = {}
    node_id = np.full(skel.shape, -1, dtype=int)

    # junction cluster representatives
    centers = ndimage.center_of_mass(junc_mask, lab,
                                     range(1, int(lab.max()) + 1))
    for cid, (r, c) in enumerate(centers):
        rr, cc = np.nonzero(lab == cid + 1)
        nid = len(nodes)
        nodes[nid] = StrokeNode(nid, (float(c), float(r)), "junction",
                                float(radius[rr, cc].max()))
        for r1, c1 in zip(rr, cc):
            node_id[r1, c1] = nid

    def ensure_end_node(r, c):
        nid = int(node_id[r, c])
        if nid < 0:
            nid = max(nodes) + 1 if nodes else 0
            node_id[r, c] = nid
            nodes[nid] = StrokeNode(nid, (float(c), float(r)), "end",
                                    float(radius[r, c]))
        return nid

    def neighbours(r, c):
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1]:
                    if skel[rr, cc]:
                        out.append((int(rr), int(cc)))
        return out

    # ring cuts: components of the skeleton without any node pixel are
    # cycles; their leftmost pixel becomes a cut carrying TWO node ids,
    # so walking the ring yields exactly two arcs between them.
    cut_pair: dict[tuple, tuple] = {}

    def pixel_node_id(p, src_id=None):
        if p in cut_pair:
            a, b = cut_pair[p]
            return b if src_id == a else a
        r, c = p
        if junc_mask[p]:
            return int(lab[p]) - 1
        if end_mask[p]:
            return ensure_end_node(int(r), int(c))
        return -1

    starts = [(int(r), int(c)) for r, c in zip(*np.nonzero(end_mask))]
    for r, c in starts:
        ensure_end_node(r, c)

    comp_lab, _ = ndimage.label(skel, structure=np.ones((3, 3)))
    comps_with_nodes = set()
    for rr, cc in zip(*np.nonzero(end_mask)):
        comps_with_nodes.add(int(comp_lab[rr, cc]))
    for rr, cc in zip(*np.nonzero(junc_mask)):
        comps_with_nodes.add(int(comp_lab[rr, cc]))
    for cid in range(1, int(comp_lab.max()) + 1):
        if cid in comps_with_nodes:
            continue
        rr, cc = np.nonzero(comp_lab == cid)
        k = int(np.lexsort((rr, cc))[0])       # min column, then row
        r0, c0 = int(rr[k]), int(cc[k])
        ia = max(nodes) + 1 if nodes else 0
        nodes[ia] = StrokeNode(ia, (float(c0), float(r0)), "end",
                               float(radius[r0, c0]))
        ib = ia + 1
        nodes[ib] = StrokeNode(ib, (float(c0), float(r0)), "end",
                               float(radius[r0, c0]))
        cut_pair[(r0, c0)] = (ia, ib)
        starts.append((r0, c0))

    used = np.zeros(skel.shape, dtype=bool)
    raw_edges = []          # [eid, a, b, pts]

    def is_node(p):
        return junc_mask[p] or end_mask[p] or p in cut_pair

    def walk(sr, sc, src_id):
        """Yield chains leaving node pixel (sr, sc); each terminates at
        the first node pixel whose id differs from src_id."""
        firsts = [p for p in neighbours(sr, sc) if not used[p]
                  and not is_node(p)]
        for f in firsts:
            if used[f]:
                continue
            pts = [(sr, sc), f]
            used[f] = True
            r, c = f
            prev = (sr, sc)
            while True:
                stop = None
                if len(pts) > 3:   # never stop right at the seed itself
                    for p in sorted(neighbours(r, c)):
                        if is_node(p) and pixel_node_id(p, src_id) != src_id:
                            stop = p
                            break
                if stop is not None:
                    r, c = stop
                    pts.append((r, c))
                    break
                nxt = [p for p in neighbours(r, c)
                       if p != prev and not is_node(p) and not used[p]]
                if not nxt:
                    break
                dr_, dc_ = r - prev[0], c - prev[1]
                nxt.sort(key=lambda q: abs(q[0] - r - dr_)
                         + abs(q[1] - c - dc_))
                prev = (r, c)
                r, c = nxt[0]
                used[r, c] = True
                pts.append((r, c))
            yield pts

    eid = 0
    for r, c in starts:
        sid = pixel_node_id((r, c))
        for pts in walk(r, c, sid):
            b = pixel_node_id(pts[-1], sid)
            if b not in (-1, sid):
                raw_edges.append([eid, sid, b, pts])
                eid += 1
    for r, c in zip(*np.nonzero(junc_mask)):
        sid = pixel_node_id((int(r), int(c)))
        for pts in walk(int(r), int(c), sid):
            b = pixel_node_id(pts[-1], sid)
            if b not in (-1, sid):
                raw_edges.append([eid, sid, b, pts])
                eid += 1

    import os
    if os.environ.get("DENYSKO_SKEL_DEBUG"):
        print("DBG starts:", starts, "cut_pair:", cut_pair,
              "raw:", [(e[1], e[2], len(e[3])) for e in raw_edges])
    raw_edges = [e for e in raw_edges
                 if e[1] in nodes and e[2] in nodes]
    nodes, raw_edges = _contract(nodes, raw_edges, min_edge_px)

    degree: dict[int, int] = {nid: 0 for nid in nodes}
    for _, a, b, _ in raw_edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    kept = []
    for _, a, b, pts in raw_edges:
        ea = nodes[a].kind if a in nodes else None
        eb = nodes[b].kind if b in nodes else None
        spur = ((ea == "end" and len(pts) < min_spur
                 and degree.get(b, 0) > 1)
                or (eb == "end" and len(pts) < min_spur
                    and degree.get(a, 0) > 1))
        if spur or a == b or len(pts) < 2:
            continue
        kept.append((a, b, pts))

    out_edges = []
    for a, b, pts in kept:
        arr = np.asarray([(float(cc), float(rr)) for rr, cc in pts])
        out_edges.append(StrokeEdge(len(out_edges), a, b, arr))

    used_nodes = sorted({e.a for e in out_edges} | {e.b for e in out_edges})
    remap = {old: new for new, old in enumerate(used_nodes)}
    final_nodes = [
        StrokeNode(remap[nid], nodes[nid].xy, nodes[nid].kind,
                   nodes[nid].radius)
        for nid in used_nodes
    ]
    for e in out_edges:
        e.a = remap[e.a]
        e.b = remap[e.b]
    return StrokeGraph(nodes=final_nodes, edges=out_edges)
