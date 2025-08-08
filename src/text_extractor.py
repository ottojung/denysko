#!/usr/bin/env python3
"""
Text extraction module - handles font path extraction and contour point sampling.
"""

import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt


class TextExtractor:
    """Handles extracting character paths and contour points from text."""

    def __init__(self):
        self.font_path = self._get_font_path()

    def _get_font_path(self):
        """Get a suitable font path for text rendering."""
        try:
            # Try to find a common system font
            fonts = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
            for font_name in fonts:
                font_path = font_manager.findfont(
                    font_manager.FontProperties(family=font_name)
                )
                if font_path:
                    return font_path
        except Exception:
            pass

        # Fallback to default font
        return font_manager.findfont(font_manager.FontProperties())

    def text_to_paths(self, text, font_size=100):
        """
        Convert text to Path objects using ZERO stroke width (outline only).
        This eliminates thickness and creates clean single-line letter shapes.

        Args:
            text (str): Input text
            font_size (int): Font size for rendering

        Returns:
            list: List of simplified Path objects for each character (stroke width = 0)
        """
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties

        paths = []
        x_offset = 0

        # Use a font that works well for outlines
        font_props = FontProperties(family="sans-serif", weight="normal")

        for char in text:
            if char == " ":
                x_offset += font_size * 0.3  # Space width
                continue

            try:
                # Create text path with specific font properties for clean outlines
                path = TextPath((x_offset, 0), char, size=font_size, prop=font_props)

                if len(path.vertices) > 0:
                    # Keep original outline path for preview; do not skeletonize here
                    paths.append(path)

                # Calculate character width for next character positioning
                bbox = path.get_extents()
                char_width = bbox.width if bbox.width > 0 else font_size * 0.5
                x_offset += char_width + font_size * 0.05  # Small spacing between chars

            except Exception as e:
                print(f"Warning: Could not process character '{char}': {e}")
                x_offset += font_size * 0.5  # Default spacing

        return paths

    def simplify_path_for_zero_stroke(self, path):
        """
        Extract the TRUE CENTERLINE (medial axis) from a font path.
        This creates a skeletal representation using only the midpoints of the letter strokes.

        Args:
            path: matplotlib Path object

        Returns:
            Path: Skeletonized path representing only the centerline
        """
        if len(path.vertices) == 0:
            return None

        # Convert path to a rasterized representation for skeletonization
        skeleton_points = self.extract_skeleton_from_path(path)

        if len(skeleton_points) < 3:
            return path  # Fallback to original if skeleton extraction fails

        # Create a new path from the skeleton points
        from matplotlib.path import Path as MPLPath

        # Create path codes for the skeleton (all LINETO except first MOVETO)
        codes = [MPLPath.MOVETO] + [MPLPath.LINETO] * (len(skeleton_points) - 1)

        return MPLPath(skeleton_points, codes)

    def extract_skeleton_from_path(self, path):
        """
        Extract a medial-like centerline by finding true midpoints between opposing
        boundaries along many directions (0°, 45°, 90°, 135°). This is robust for
        horizontal, vertical, and diagonal strokes. Falls back to thinning/ridge
        when insufficient.
        """
        vertices = path.vertices
        if len(vertices) < 6:
            return vertices

        pts = self._scanline_midpoint_skeleton(path)
        if pts is not None and len(pts) >= 10:
            return pts

        # Fallback: morphological thinning, then chamfer ridge
        mask, x_grid, y_grid = self._rasterize_path(path, resolution=512)
        if mask.sum() == 0:
            return vertices
        skel = self._zhang_suen_thinning(mask)
        if skel.sum() < 5:
            dist = self._chamfer_distance_transform(mask)
            skel = self._medial_ridge(mask, dist)
            if skel.sum() < 5:
                return self.create_simple_stroke_approximation(path)
        comps = self._connected_components(skel)
        if not comps:
            return self.create_simple_stroke_approximation(path)
        best_trace = None
        best_len = -1.0
        for comp in comps:
            trace = self._trace_component_path(comp)
            if len(trace) < 3:
                continue
            rs = np.array([p[0] for p in trace])
            cs = np.array([p[1] for p in trace])
            xs = x_grid[0, cs]
            ys = y_grid[rs, 0]
            path_pts = np.stack([xs, ys], axis=1)
            L = np.sum(np.linalg.norm(np.diff(path_pts, axis=0), axis=1))
            if L > best_len:
                best_len = L
                best_trace = trace
        if not best_trace:
            return self.create_simple_stroke_approximation(path)
        rs = np.array([p[0] for p in best_trace])
        cs = np.array([p[1] for p in best_trace])
        xs = x_grid[0, cs]
        ys = y_grid[rs, 0]
        path_pts = np.stack([xs, ys], axis=1)
        path_pts = self._smooth_polyline(path_pts, window=9)
        path_pts = self._dedupe_close_points(path_pts, tol=1e-2)
        return path_pts

    def extract_skeleton_from_path_old(self, path):
        """
        Compute a true centerline using a rasterized medial-axis approximation.
        Steps:
          1) Rasterize the glyph into a binary mask.
          2) Compute approximate Euclidean distance transform (two-pass chamfer).
          3) Detect ridge (local maxima) pixels of the distance map inside the mask.
          4) Keep the largest connected ridge component and trace it into an ordered polyline.
          5) Map pixels back to glyph coordinates and smooth.
        """
        vertices = path.vertices
        if len(vertices) < 6:
            return vertices

        # 1) Rasterize
        mask, x_grid, y_grid = self._rasterize_path(path)
        if mask.sum() == 0:
            return vertices

        # 2) Distance transform
        dist = self._chamfer_distance_transform(mask)
        if np.isinf(dist).all():
            return vertices

        # 3) Ridge detection (local maxima) + pruning
        ridge = self._medial_ridge(mask, dist)
        if ridge.sum() == 0:
            return self.create_simple_stroke_approximation(path)

        # 4) Largest connected component and path tracing
        components = self._connected_components(ridge)
        if not components:
            return self.create_simple_stroke_approximation(path)
        # Pick the largest component
        comp = max(components, key=lambda c: len(c))
        ordered_pixels = self._trace_component_path(comp)
        if len(ordered_pixels) < 3:
            return self.create_simple_stroke_approximation(path)

        # 5) Map to glyph coordinates
        # Pixels are (r, c) indices into y_grid/x_grid
        rs = np.array([p[0] for p in ordered_pixels])
        cs = np.array([p[1] for p in ordered_pixels])
        xs = x_grid[0, cs]
        ys = y_grid[rs, 0]
        pts = np.stack([xs, ys], axis=1)

        # Smooth and sparsify
        pts = self._smooth_polyline(pts, window=7)
        pts = self._dedupe_close_points(pts, tol=1e-2)
        return pts

    # === Medial-axis helpers ===
    def _rasterize_path(self, path, resolution=256):
        """
        Rasterize a Path to a binary mask with given resolution.
        Returns (mask, x_grid, y_grid).
        """
        vertices = path.vertices
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        # Avoid zero-size
        if max_x <= min_x:
            max_x = min_x + 1.0
        if max_y <= min_y:
            max_y = min_y + 1.0

        # Keep aspect by basing resolution on max dimension
        width = max_x - min_x
        height = max_y - min_y
        base = float(max(width, height))
        # Scale resolution to size (cap between 200 and 600)
        res = int(np.clip((resolution * base / max(base, 1e-6)), 200, 600))
        # Create grid
        xs = np.linspace(min_x, max_x, res)
        ys = np.linspace(min_y, max_y, res)
        x_grid, y_grid = np.meshgrid(xs, ys)
        pts = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

        inside = path.contains_points(pts)
        mask = inside.reshape(res, res)
        return mask, x_grid, y_grid

    def _chamfer_distance_transform(self, mask):
        """
        Approximate Euclidean distance to the boundary using a two-pass chamfer.
        Distance is in pixel units. Only defined for mask==True; others set to 0.
        """
        h, w = mask.shape
        # Identify boundary pixels: inside with any 4-neighbor outside
        boundary = np.zeros_like(mask, dtype=bool)
        # Pad to handle edges
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        for r in range(1, h + 1):
            for c in range(1, w + 1):
                if not padded[r, c]:
                    continue
                if not (
                    padded[r - 1, c]
                    and padded[r + 1, c]
                    and padded[r, c - 1]
                    and padded[r, c + 1]
                ):
                    boundary[r - 1, c - 1] = True

        inf = np.inf
        dist = np.full((h, w), inf, dtype=float)
        dist[boundary] = 0.0

        # We only propagate inside the mask
        # Forward pass
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or dist[r, c] == 0.0:
                    continue
                best = dist[r, c]
                # up
                if r - 1 >= 0:
                    best = min(best, dist[r - 1, c] + 1.0)
                # left
                if c - 1 >= 0:
                    best = min(best, dist[r, c - 1] + 1.0)
                # up-left
                if r - 1 >= 0 and c - 1 >= 0:
                    best = min(best, dist[r - 1, c - 1] + np.sqrt(2))
                # up-right
                if r - 1 >= 0 and c + 1 < w:
                    best = min(best, dist[r - 1, c + 1] + np.sqrt(2))
                dist[r, c] = best

        # Backward pass
        for r in range(h - 1, -1, -1):
            for c in range(w - 1, -1, -1):
                if not mask[r, c] or dist[r, c] == 0.0:
                    continue
                best = dist[r, c]
                # down
                if r + 1 < h:
                    best = min(best, dist[r + 1, c] + 1.0)
                # right
                if c + 1 < w:
                    best = min(best, dist[r, c + 1] + 1.0)
                # down-right
                if r + 1 < h and c + 1 < w:
                    best = min(best, dist[r + 1, c + 1] + np.sqrt(2))
                # down-left
                if r + 1 < h and c - 1 >= 0:
                    best = min(best, dist[r + 1, c - 1] + np.sqrt(2))
                dist[r, c] = best

        # Keep distance only inside mask; zeros elsewhere
        dist[~mask] = 0.0
        return dist

    def _medial_ridge(self, mask, dist):
        """
        Detect ridge pixels as 8-neighborhood local maxima of the distance transform.
        Prune weak ridges using a global threshold.
        """
        h, w = mask.shape
        ridge = np.zeros_like(mask, dtype=bool)
        max_d = dist.max() if np.isfinite(dist).any() else 0.0
        if max_d <= 0:
            return ridge
        # Threshold to remove tiny spurs (2% of max radius)
        thr = max(0.75, 0.02 * max_d)
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not mask[r, c]:
                    continue
                d = dist[r, c]
                if d < thr:
                    continue
                neighborhood = dist[r - 1 : r + 2, c - 1 : c + 2]
                if d >= neighborhood.max() and (neighborhood == d).sum() <= 3:
                    ridge[r, c] = True
        return ridge

    def _connected_components(self, mask):
        """
        Find 8-connected components of True pixels. Returns list of lists of (r,c).
        """
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or visited[r, c]:
                    continue
                # BFS
                queue = [(r, c)]
                visited[r, c] = True
                comp = []
                while queue:
                    rr, cc = queue.pop(0)
                    comp.append((rr, cc))
                    for dr, dc in neighbors:
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and mask[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                comps.append(comp)
        return comps

    def _trace_component_path(self, comp_pixels):
        """
        Order component pixels into a path by greedy traversal from an endpoint.
        """
        if not comp_pixels:
            return []
        # Build set for fast lookup
        pix_set = set((int(r), int(c)) for r, c in comp_pixels)
        # Degree per pixel using 8-neighborhood
        neighs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        deg = {}
        for r, c in comp_pixels:
            d = 0
            for dr, dc in neighs:
                if (r + dr, c + dc) in pix_set:
                    d += 1
            deg[(r, c)] = d
        # Find endpoints (degree 1); if none (loop), pick arbitrary start
        endpoints = [p for p, d in deg.items() if d == 1]
        start = endpoints[0] if endpoints else comp_pixels[0]

        ordered = [start]
        visited = set([start])
        current = start
        while True:
            candidates = []
            for dr, dc in neighs:
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in pix_set and nxt not in visited:
                    candidates.append(nxt)
            if not candidates:
                nn = self._nearest_unvisited(current, pix_set, visited, radius=2)
                if nn is None:
                    break
                candidates = [nn]
            if len(ordered) >= 2:
                prev = np.array(ordered[-2])
                cur = np.array(current)
                best = None
                best_cos = -np.inf
                v1 = cur - prev
                v1 = v1 / (np.linalg.norm(v1) + 1e-9)
                for cand in candidates:
                    v2 = np.array(cand) - cur
                    v2 = v2 / (np.linalg.norm(v2) + 1e-9)
                    cosang = float(v1 @ v2)
                    if cosang > best_cos:
                        best_cos = cosang
                        best = cand
                nxt = best
            else:
                nxt = candidates[0]
            ordered.append(nxt)
            visited.add(nxt)
            current = nxt
        return ordered

    def _nearest_unvisited(self, current, pix_set, visited, radius=2):
        r0, c0 = current
        for rad in range(1, radius + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr == 0 and dc == 0:
                        continue
                    cand = (r0 + dr, c0 + dc)
                    if cand in pix_set and cand not in visited:
                        return cand
        return None

    def _smooth_polyline(self, pts, window=7):
        if len(pts) < 3 or window < 3:
            return pts
        w = window if window % 2 == 1 else window + 1
        k = w // 2
        pad = np.vstack([pts[0:1].repeat(k, axis=0), pts, pts[-1:].repeat(k, axis=0)])
        sm = []
        for i in range(k, k + len(pts)):
            sm.append(pad[i - k : i + k + 1].mean(axis=0))
        return np.array(sm)

    def _dedupe_close_points(self, pts, tol=1e-3):
        if len(pts) <= 1:
            return pts
        out = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - out[-1]) > tol:
                out.append(p)
        return np.array(out)

    def _scanline_midpoint_skeleton(self, path, angles=(0.0, 45.0, 90.0, 135.0)):
        """
        Multi-angle scanline midpoint extraction. Returns ordered centerline points
        or None if too few midpoints found.
        """
        vertices = path.vertices
        if len(vertices) < 3:
            return None
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return None

        base = max(width, height)
        spacing = max(base / 200.0, 0.5)
        min_span = max(base * 0.01, spacing)

        all_midpoints = []
        center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=float)
        for ang in angles:
            c = np.cos(np.deg2rad(ang))
            s = np.sin(np.deg2rad(ang))
            d = np.array([c, s], dtype=float)
            n = np.array([-s, c], dtype=float)  # normal

            # Offsets along normal that cover the bbox corners
            corners = np.array(
                [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]],
                dtype=float,
            )
            proj = (corners - center) @ n
            omin, omax = float(np.min(proj)), float(np.max(proj))
            offsets = np.arange(omin - spacing, omax + spacing, spacing)

            for off in offsets:
                p0 = center + off * n
                ts = self._intersections_with_infinite_line(vertices, p0, d)
                if len(ts) < 2:
                    continue
                ts.sort()
                for i in range(0, len(ts) - 1, 2):
                    t1 = ts[i]
                    t2 = ts[i + 1]
                    if (t2 - t1) < min_span:
                        continue
                    m = p0 + 0.5 * (t1 + t2) * d
                    if path.contains_point(m):
                        all_midpoints.append(m)

        if len(all_midpoints) < 10:
            return None

        pts = np.array(all_midpoints)
        pts = self.remove_duplicate_points(pts, tolerance=1e-3)

        # Order points via PCA axis then nearest-neighbor
        mean = pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)
            axis = Vt[0]
            proj = (pts - mean) @ axis
            start_idx = int(np.argmin(proj))
        except Exception:
            start_idx = 0
        ordered = self._order_points_nearest_neighbor(pts, start_idx)
        ordered = self._smooth_polyline(ordered, window=7)
        ordered = self._dedupe_close_points(ordered, tol=1e-2)
        return ordered

    def _intersections_with_infinite_line(self, vertices, p0, d):
        """
        Compute parameter t for intersections of line p = p0 + t d with polygon edges.
        Returns list of t values where the infinite line crosses each segment.
        """
        ts = []
        n = len(vertices)
        if n < 2:
            return ts
        for i in range(n - 1):
            a = vertices[i]
            b = vertices[i + 1]
            t = self._line_seg_intersection_param(p0, d, a, b)
            if t is not None:
                ts.append(t)
        return ts

    def _line_seg_intersection_param(self, p0, d, a, b, eps=1e-9):
        """
        Solve for t,u in a + u(b-a) = p0 + t d with u in [0,1]. If intersect, return t.
        """
        ab = b - a
        A = np.array([[ab[0], -d[0]], [ab[1], -d[1]]], dtype=float)
        rhs = np.array([p0[0] - a[0], p0[1] - a[1]], dtype=float)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < eps:
            return None
        invA = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], dtype=float) / det
        u, t = invA @ rhs
        if -eps <= u <= 1.0 + eps:
            return float(t)
        return None

    def _order_points_nearest_neighbor(self, pts, start_idx=0):
        if len(pts) <= 2:
            return pts
        used = np.zeros(len(pts), dtype=bool)
        order = []
        idx = start_idx
        for _ in range(len(pts)):
            order.append(idx)
            used[idx] = True
            diffs = pts - pts[idx]
            d2 = np.sum(diffs * diffs, axis=1)
            d2[used] = np.inf
            if not np.isfinite(d2).any():
                break
            idx = int(np.argmin(d2))
        return pts[order]

    def _zhang_suen_thinning(self, img):
        """
        Perform Zhang-Suen thinning on a binary mask. Returns a boolean skeleton mask.
        img: boolean or 0/1 numpy array, True/1 for foreground.
        """
        bin_img = img.astype(np.uint8).copy()
        changed = True
        h, w = bin_img.shape
        neighbors = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]

        def neighbor_vals(r, c):
            return [bin_img[r + dr, c + dc] for dr, dc in neighbors]

        def transitions(p):
            # number of 0->1 transitions in circular sequence p2..p9..p2
            c = 0
            for i in range(8):
                if p[i] == 0 and p[(i + 1) % 8] == 1:
                    c += 1
            return c

        while changed:
            changed = False
            to_remove = []
            # Sub-iteration 1
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if bin_img[r, c] == 0:
                        continue
                    p = neighbor_vals(r, c)
                    nz = sum(p)
                    if nz < 2 or nz > 6:
                        continue
                    if transitions(p) != 1:
                        continue
                    if p[0] * p[2] * p[4] != 0:
                        continue
                    if p[2] * p[4] * p[6] != 0:
                        continue
                    to_remove.append((r, c))
            if to_remove:
                for r, c in to_remove:
                    bin_img[r, c] = 0
                changed = True
            to_remove = []
            # Sub-iteration 2
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if bin_img[r, c] == 0:
                        continue
                    p = neighbor_vals(r, c)
                    nz = sum(p)
                    if nz < 2 or nz > 6:
                        continue
                    if transitions(p) != 1:
                        continue
                    if p[0] * p[2] * p[6] != 0:
                        continue
                    if p[0] * p[4] * p[6] != 0:
                        continue
                    to_remove.append((r, c))
            if to_remove:
                for r, c in to_remove:
                    bin_img[r, c] = 0
                changed = True
        return bin_img.astype(bool)

    def find_letter_key_points(self, vertices):
        """
        Find the key structural points that define the letter's main features.
        For 'A': bottom-left corner, top peak, bottom-right corner, crossbar points.

        Args:
            vertices: Path vertices

        Returns:
            list: Key structural points
        """
        if len(vertices) < 4:
            return vertices

        # Find bounding box
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)

        key_points = []

        # Find extremal points
        # 1. Topmost point (peak of 'A')
        top_idx = np.argmax(vertices[:, 1])
        key_points.append(vertices[top_idx])

        # 2. Bottommost points (base of 'A')
        bottom_y = np.min(vertices[:, 1])
        bottom_threshold = bottom_y + (max_y - min_y) * 0.1  # Bottom 10%

        bottom_indices = np.where(vertices[:, 1] <= bottom_threshold)[0]
        if len(bottom_indices) > 0:
            # Find leftmost and rightmost bottom points
            bottom_vertices = vertices[bottom_indices]
            left_bottom_idx = bottom_indices[np.argmin(bottom_vertices[:, 0])]
            right_bottom_idx = bottom_indices[np.argmax(bottom_vertices[:, 0])]

            key_points.append(vertices[left_bottom_idx])
            key_points.append(vertices[right_bottom_idx])

        # 3. Middle height points (for crossbar of 'A')
        mid_y = min_y + (max_y - min_y) * 0.4  # 40% up from bottom
        mid_threshold = (max_y - min_y) * 0.15  # Tolerance around middle

        mid_indices = np.where(np.abs(vertices[:, 1] - mid_y) <= mid_threshold)[0]
        if len(mid_indices) > 1:
            # Find leftmost and rightmost middle points
            mid_vertices = vertices[mid_indices]
            left_mid_idx = mid_indices[np.argmin(mid_vertices[:, 0])]
            right_mid_idx = mid_indices[np.argmax(mid_vertices[:, 0])]

            # Only add if they're significantly apart (actual crossbar)
            if (
                abs(vertices[left_mid_idx][0] - vertices[right_mid_idx][0])
                > (max_x - min_x) * 0.3
            ):
                key_points.append(vertices[left_mid_idx])
                key_points.append(vertices[right_mid_idx])

        print(f"            Found {len(key_points)} key structural points")
        return key_points

    def create_structural_strokes(self, key_points):
        """
        Create clean stroke lines from key structural points.
        For 'A': Create left diagonal, right diagonal, and horizontal crossbar.

        Args:
            key_points: List of key structural points

        Returns:
            list: List of stroke line dictionaries with start/endpoint
        """
        if len(key_points) < 3:
            return []

        strokes = []

        # Sort points by y-coordinate (top to bottom) then by x-coordinate
        sorted_points = sorted(key_points, key=lambda p: (-p[1], p[0]))

        if len(sorted_points) >= 3:
            top_point = sorted_points[0]

            # Find bottom points
            bottom_points = [
                p for p in sorted_points if p[1] < top_point[1] - 10
            ]  # Significantly below top

            if len(bottom_points) >= 2:
                # Sort bottom points by x-coordinate
                bottom_points.sort(key=lambda p: p[0])
                left_bottom = bottom_points[0]
                right_bottom = bottom_points[-1]

                # Create main structural strokes
                # Left diagonal: top to left bottom
                strokes.append(
                    {"start": top_point, "end": left_bottom, "type": "diagonal_left"}
                )

                # Right diagonal: top to right bottom
                strokes.append(
                    {"start": top_point, "end": right_bottom, "type": "diagonal_right"}
                )

                # Crossbar: find middle-height points
                mid_points = [
                    p
                    for p in sorted_points
                    if abs(p[1] - (top_point[1] + left_bottom[1]) / 2) < 20
                ]
                if len(mid_points) >= 2:
                    mid_points.sort(key=lambda p: p[0])
                    strokes.append(
                        {
                            "start": mid_points[0],
                            "end": mid_points[-1],
                            "type": "crossbar",
                        }
                    )

        print("            Created {len(strokes)} structural strokes")
        return strokes

    def connect_strokes_intelligently(self, stroke_lines):
        """
        Connect stroke lines in a logical pen-drawing order to minimize unwanted connecting lines.
        For letter 'A': Connect left-diagonal -> right-diagonal -> crossbar with minimal jumps.

        Args:
            stroke_lines: List of stroke line dictionaries

        Returns:
            list: Connected sequence of points with intelligent stroke ordering
        """
        if not stroke_lines:
            return []

        print(f"            Connecting {len(stroke_lines)} strokes intelligently...")

        # Convert each stroke to points first
        stroke_segments = []
        for line in stroke_lines:
            line_points = self.generate_line_points(line["start"], line["end"], 15)
            if len(line_points) > 1:
                stroke_segments.append(
                    {
                        "points": line_points,
                        "type": line["type"],
                        "start": line["start"],
                        "end": line["end"],
                    }
                )

        if not stroke_segments:
            return []

        # For letter 'A': Connect in a logical pen-drawing order
        # 1. Left diagonal: bottom-left to top
        # 2. Right diagonal: top to bottom-right
        # 3. Crossbar: left to right

        connected_points = []

        # Find strokes by type
        left_diag = None
        right_diag = None
        crossbar = None

        for stroke in stroke_segments:
            if stroke["type"] == "diagonal_left":
                left_diag = stroke
            elif stroke["type"] == "diagonal_right":
                right_diag = stroke
            elif stroke["type"] == "crossbar":
                crossbar = stroke

        # Connect strokes in optimal order to minimize jumps
        if left_diag and right_diag:
            # Start with left diagonal (bottom to top)
            left_points = left_diag["points"]

            # Check if we need to reverse to go bottom->top
            if (
                left_points[0][1] > left_points[-1][1]
            ):  # First point is higher than last
                left_points = left_points[::-1]  # Reverse to go bottom->top

            connected_points.extend(left_points)

            # Continue with right diagonal (top to bottom)
            # The right diagonal should start near where left diagonal ended (the top)
            right_points = right_diag["points"]

            # Check if right diagonal should be reversed to connect smoothly
            left_end = left_points[-1]
            right_start = right_points[0]
            right_end = right_points[-1]

            # Use the right diagonal orientation that connects better to left diagonal end
            start_dist = np.linalg.norm(np.array(left_end) - np.array(right_start))
            end_dist = np.linalg.norm(np.array(left_end) - np.array(right_end))

            if end_dist < start_dist:
                right_points = right_points[::-1]  # Reverse to connect better

            connected_points.extend(right_points)

            # Add crossbar as a separate stroke by connecting to nearest point
            if crossbar:
                print("            Adding crossbar stroke with smart connection...")
                crossbar_points = crossbar["points"]

                # Find the closest point on existing strokes to crossbar start
                right_end = connected_points[-1]  # End of right diagonal
                crossbar_start = crossbar_points[0]
                crossbar_end = crossbar_points[-1]

                # Choose crossbar orientation based on proximity to right diagonal end
                dist_to_start = np.linalg.norm(
                    np.array(right_end) - np.array(crossbar_start)
                )
                dist_to_end = np.linalg.norm(
                    np.array(right_end) - np.array(crossbar_end)
                )

                # Use the crossbar orientation that's closer to our current position
                if dist_to_end < dist_to_start:
                    crossbar_points = crossbar_points[::-1]  # Reverse crossbar

                # Add a single connecting line to crossbar (minimal jump)
                crossbar_connection_start = crossbar_points[0]
                connected_points.append(crossbar_connection_start)

                # Add all crossbar points
                connected_points.extend(crossbar_points)

                print("            Crossbar included with minimal connection jump")

        # If we only have one or two strokes, use only the main stroke
        elif left_diag:
            connected_points.extend(left_diag["points"])
        elif right_diag:
            connected_points.extend(right_diag["points"])
        elif crossbar:
            connected_points.extend(crossbar["points"])

        # Fallback: if stroke types don't match expected patterns, use first stroke only
        if not connected_points and stroke_segments:
            connected_points.extend(stroke_segments[0]["points"])
            print("            FALLBACK: Using only first stroke to avoid jumps")

        print(
            f"            Connected strokes into {len(connected_points)} points (avoided unwanted jumps)"
        )
        return connected_points

    def generate_line_points(self, start_point, end_point, num_points=20):
        """
        Generate evenly spaced points along a straight line.

        Args:
            start_point: [x, y] start coordinates
            end_point: [x, y] end coordinates
            num_points: Number of points to generate

        Returns:
            list: List of (x, y) points along the line
        """
        if num_points < 2:
            return [start_point, end_point]

        # Generate parameter values from 0 to 1
        t_values = np.linspace(0, 1, num_points)

        # Interpolate along the line
        points = []
        for t in t_values:
            point = start_point + t * (end_point - start_point)
            points.append(point)

        return points

    def create_simple_stroke_approximation(self, path):
        """
        Fallback method: Create a simple approximation when geometric detection fails.

        Args:
            path: matplotlib Path object

        Returns:
            np.array: Simple stroke approximation points
        """
        vertices = path.vertices

        if len(vertices) < 6:
            return vertices

        # Use every 10th vertex to create a simplified representation
        step = max(1, len(vertices) // 10)
        simplified = vertices[::step]

        # Ensure we have at least a few points
        if len(simplified) < 3:
            simplified = vertices[[0, len(vertices) // 2, -1]]

        return simplified

    def find_x_intersections_at_y(self, path, y):
        """
        Find x-coordinates where a horizontal line at y intersects the path.

        Args:
            path: matplotlib Path object
            y: y-coordinate of horizontal line

        Returns:
            list: List of x-coordinates of intersections
        """
        vertices = path.vertices
        intersections = []

        # Check each edge of the path
        for i in range(len(vertices) - 1):
            x1, y1 = vertices[i]
            x2, y2 = vertices[i + 1]

            # Check if the edge crosses the horizontal line y
            if ((y1 <= y <= y2) or (y2 <= y <= y1)) and y1 != y2:
                # Calculate intersection point
                t = (y - y1) / (y2 - y1)
                x_intersect = x1 + t * (x2 - x1)
                intersections.append(x_intersect)

        return intersections

    def extract_medial_axis_points(self, path, resolution=200):
        """
        Extract true medial axis points using distance transform approach.
        Only keeps points that are local maxima of the distance field - the true centerline.

        Args:
            path: matplotlib Path object
            resolution: Grid resolution for distance field computation

        Returns:
            np.array: Medial axis points (true stroke centers)
        """
        vertices = path.vertices
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)

        # Create high-resolution sampling grid
        x_samples = np.linspace(min_x, max_x, resolution)
        y_samples = np.linspace(min_y, max_y, resolution)

        # Compute distance field for all interior points
        distance_field = {}
        interior_points = []

        for i, x in enumerate(x_samples):
            for j, y in enumerate(y_samples):
                point = np.array([x, y])

                # Only process points inside the shape
                if path.contains_point(point):
                    dist = self.distance_to_boundary(point, path)
                    if dist > 0.5:  # Minimum distance to avoid boundary noise
                        distance_field[(i, j)] = dist
                        interior_points.append((i, j, x, y, dist))

        if not interior_points:
            return np.array([])

        # Find local maxima in the distance field (medial axis points)
        medial_points = []

        for i, j, x, y, dist in interior_points:
            # Check if this point is a local maximum in its neighborhood
            if self.is_local_distance_maximum(i, j, dist, distance_field, radius=2):
                medial_points.append([x, y])

        if not medial_points:
            return np.array([])

        return np.array(medial_points)

    def is_local_distance_maximum(self, i, j, dist, distance_field, radius=2):
        """
        Check if a point is a local maximum in the distance field.
        This identifies points that are furthest from boundaries in their neighborhood.

        Args:
            i, j: Grid coordinates
            dist: Distance value at this point
            distance_field: Dictionary of (i,j) -> distance
            radius: Neighborhood radius to check

        Returns:
            bool: True if point is local maximum (medial axis point)
        """
        # Check all neighbors within radius
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue

                neighbor_key = (i + di, j + dj)
                if neighbor_key in distance_field:
                    neighbor_dist = distance_field[neighbor_key]

                    # If any neighbor has higher distance, this is not a maximum
                    if (
                        neighbor_dist > dist + 0.1
                    ):  # Small tolerance for numerical precision
                        return False

        return True

    def thin_skeleton_points(self, skeleton_points, min_distance=1.0):
        """
        Thin skeleton points by removing points that are too close together.
        This creates a truly single-pixel-width skeleton.

        Args:
            skeleton_points: Array of skeleton points
            min_distance: Minimum distance between skeleton points

        Returns:
            np.array: Thinned skeleton points
        """
        if len(skeleton_points) <= 1:
            return skeleton_points

        # Start with the first point
        thinned_points = [skeleton_points[0]]

        for point in skeleton_points[1:]:
            # Check distance to all existing thinned points
            too_close = False
            for existing in thinned_points:
                if np.linalg.norm(point - existing) < min_distance:
                    too_close = True
                    break

            # Only add if far enough from existing points
            if not too_close:
                thinned_points.append(point)

        return np.array(thinned_points)

    def find_interior_stroke_points(self, path, resolution=100):
        """
        Find points that are truly inside the letter shape and represent stroke centers.
        Uses distance transform approach to find points equidistant from boundaries.

        Args:
            path: matplotlib Path object
            resolution: Grid resolution for sampling

        Returns:
            np.array: Interior points that likely represent stroke centers
        """
        vertices = path.vertices
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)

        # Create dense sampling grid
        x_samples = np.linspace(min_x, max_x, resolution)
        y_samples = np.linspace(min_y, max_y, resolution)

        interior_points = []

        # For each grid point, check if it's inside and calculate distance to boundary
        for x in x_samples:
            for y in y_samples:
                point = np.array([x, y])

                # Check if point is inside the path
                if path.contains_point(point):
                    # Calculate distance to nearest boundary
                    dist_to_boundary = self.distance_to_boundary(point, path)

                    # Only keep points that are not too close to boundary (stroke centers)
                    if dist_to_boundary > 2.0:  # Minimum distance from boundary
                        interior_points.append([x, y, dist_to_boundary])

        if not interior_points:
            return np.array([])

        # Convert to numpy array and sort by distance (furthest from boundary first)
        interior_points = np.array(interior_points)
        # Sort by distance to boundary (descending)
        interior_points = interior_points[interior_points[:, 2].argsort()[::-1]]

        # Return only x,y coordinates
        return interior_points[:, :2]

    def distance_to_boundary(self, point, path):
        """
        Calculate minimum distance from a point to the path boundary.
        Uses a more accurate method for precise distance calculation.

        Args:
            point: [x, y] coordinates
            path: matplotlib Path object

        Returns:
            float: Distance to nearest boundary point
        """
        vertices = path.vertices

        # Handle closed paths properly
        if len(vertices) > 2 and np.allclose(vertices[0], vertices[-1]):
            # Path is closed, check all edges including wrap-around
            edges_to_check = len(vertices) - 1
        else:
            # Open path
            edges_to_check = len(vertices) - 1

        min_dist = float("inf")

        # Check distance to each edge of the path
        for i in range(edges_to_check):
            edge_start = vertices[i]
            edge_end = vertices[(i + 1) % len(vertices)]

            # Skip degenerate edges
            if np.allclose(edge_start, edge_end):
                continue

            # Calculate distance from point to line segment
            dist = self.point_to_line_distance(point, edge_start, edge_end)
            min_dist = min(min_dist, dist)

        # Also check distance to vertices themselves
        for vertex in vertices:
            dist = np.linalg.norm(point - vertex)
            min_dist = min(min_dist, dist)

        return min_dist

    def point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate distance from a point to a line segment.

        Args:
            point: [x, y] coordinates of point
            line_start: [x, y] start of line segment
            line_end: [x, y] end of line segment

        Returns:
            float: Distance from point to line segment
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        # Vector from line_start to point
        point_vec = point - line_start

        # Calculate projection of point_vec onto line_vec
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:
            # Line is actually a point
            return np.linalg.norm(point - line_start)

        # Parameter t represents position along line (0 = start, 1 = end)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))

        # Find closest point on line segment
        closest_point = line_start + t * line_vec

        # Return distance
        return np.linalg.norm(point - closest_point)

    def filter_stroke_centerline_points(self, path, interior_points):
        """
        Filter interior points to keep only those that represent actual stroke centerlines.
        Removes points that are in hollow areas (like inside the triangle of 'A').

        Args:
            path: Original path
            interior_points: Points inside the shape

        Returns:
            np.array: Filtered points representing stroke centerlines
        """
        if len(interior_points) == 0:
            return interior_points

        filtered_points = []

        for point in interior_points:
            x, y = point

            # Check if this point is part of a stroke by examining its neighborhood
            if self.is_stroke_center_point(path, point):
                filtered_points.append(point)

        return np.array(filtered_points) if filtered_points else interior_points

    def is_stroke_center_point(self, path, point):
        """
        Determine if a point represents the center of a stroke.
        A stroke center point should have roughly equal distances to boundaries
        in perpendicular directions.

        Args:
            path: Original path
            point: [x, y] coordinates to test

        Returns:
            bool: True if point is likely a stroke center
        """
        x, y = point

        # Sample in 8 directions around the point to find boundary distances
        directions = [
            [1, 0],
            [-1, 0],  # horizontal
            [0, 1],
            [0, -1],  # vertical
            [1, 1],
            [-1, -1],  # diagonal
            [1, -1],
            [-1, 1],  # diagonal
        ]

        boundary_distances = []

        for dx, dy in directions:
            # Cast ray in this direction until we hit the boundary
            dist = self.ray_to_boundary_distance(path, point, [dx, dy])
            if dist > 0:
                boundary_distances.append(dist)

        if len(boundary_distances) < 4:
            return False

        # For a stroke center, we expect roughly equal distances in opposite directions
        # Check horizontal and vertical balance
        horizontal_dists = boundary_distances[0:2]  # left, right
        vertical_dists = boundary_distances[2:4]  # up, down

        # A good stroke center has balanced distances
        h_ratio = (
            min(horizontal_dists) / max(horizontal_dists)
            if max(horizontal_dists) > 0
            else 0
        )
        v_ratio = (
            min(vertical_dists) / max(vertical_dists) if max(vertical_dists) > 0 else 0
        )

        # Point is likely a stroke center if distances are reasonably balanced
        return h_ratio > 0.3 or v_ratio > 0.3

    def ray_to_boundary_distance(self, path, start_point, direction, max_dist=50):
        """
        Cast a ray from start_point in given direction until hitting boundary.

        Args:
            path: Path object
            start_point: [x, y] starting coordinates
            direction: [dx, dy] direction vector
            max_dist: Maximum distance to search

        Returns:
            float: Distance to boundary, or 0 if not found
        """
        x, y = start_point
        dx, dy = direction

        # Normalize direction
        length = np.sqrt(dx * dx + dy * dy)
        if length == 0:
            return 0
        dx, dy = dx / length, dy / length

        # Step along ray until we exit the shape
        step_size = 0.5
        for distance in np.arange(step_size, max_dist, step_size):
            test_point = [x + dx * distance, y + dy * distance]

            if not path.contains_point(test_point):
                return distance

        return 0

    def connect_stroke_segments(self, stroke_points):
        """
        Connect stroke centerline points into continuous segments.
        This creates connected paths that follow the letter strokes without filling holes.

        Args:
            stroke_points: Array of stroke center points

        Returns:
            np.array: Connected stroke centerline points
        """
        if len(stroke_points) <= 2:
            return stroke_points

        # Identify separate stroke segments first
        stroke_segments = self.identify_stroke_segments(stroke_points)

        # Connect each segment separately and combine
        all_connected_points = []

        for segment in stroke_segments:
            if len(segment) > 1:
                # Sort points within each segment by connectivity
                connected_segment = self.sort_points_by_stroke_connectivity(segment)
                all_connected_points.extend(connected_segment)

        if len(all_connected_points) == 0:
            return stroke_points

        return np.array(all_connected_points)

    def identify_stroke_segments(self, points, max_segment_gap=5.0):
        """
        Identify separate stroke segments by clustering nearby points.
        This prevents connecting across holes or between separate strokes.

        Args:
            points: Array of points
            max_segment_gap: Maximum distance to consider points in same segment

        Returns:
            list: List of point arrays, each representing a separate stroke segment
        """
        if len(points) <= 1:
            return [points]

        # Use simple clustering based on distance
        segments = []
        unassigned = list(range(len(points)))

        while unassigned:
            # Start new segment with first unassigned point
            current_segment = [unassigned.pop(0)]
            segment_points = [points[current_segment[0]]]

            # Keep adding nearby points to this segment
            added_any = True
            while added_any:
                added_any = False
                to_remove = []

                for idx in unassigned:
                    point = points[idx]

                    # Check if point is close to any point in current segment
                    min_dist = min(
                        np.linalg.norm(point - seg_point)
                        for seg_point in segment_points
                    )

                    if min_dist <= max_segment_gap:
                        current_segment.append(idx)
                        segment_points.append(point)
                        to_remove.append(idx)
                        added_any = True

                # Remove assigned points
                for idx in to_remove:
                    unassigned.remove(idx)

            # Add this segment
            if len(current_segment) > 0:
                segment_points_array = points[[i for i in current_segment]]
                segments.append(segment_points_array)

        return segments

    def sort_points_by_stroke_connectivity(self, points):
        """
        Sort points to follow stroke patterns rather than just nearest neighbor.

        Args:
            points: Array of stroke points

        Returns:
            np.array: Points sorted to follow stroke structure
        """
        if len(points) <= 2:
            return points

        # Find stroke endpoints (points that are furthest from other points)
        endpoint_indices = self.find_stroke_endpoints(points)

        if len(endpoint_indices) < 2:
            # Fallback to simple connectivity sorting
            return self.sort_points_by_connectivity(points)

        # Start from one endpoint and trace to others
        sorted_points = []
        remaining_indices = set(range(len(points)))

        # Start with the first endpoint
        current_idx = endpoint_indices[0]
        sorted_points.append(points[current_idx])
        remaining_indices.remove(current_idx)

        # Trace path by following nearest neighbors
        while remaining_indices:
            current_point = sorted_points[-1]

            # Find nearest remaining point
            min_dist = float("inf")
            next_idx = None

            for idx in remaining_indices:
                dist = np.linalg.norm(current_point - points[idx])
                if dist < min_dist:
                    min_dist = dist
                    next_idx = idx

            if next_idx is not None:
                sorted_points.append(points[next_idx])
                remaining_indices.remove(next_idx)
            else:
                break

        return np.array(sorted_points)

    def find_stroke_endpoints(self, points):
        """
        Find points that are likely endpoints of strokes.

        Args:
            points: Array of points

        Returns:
            list: Indices of points that are likely stroke endpoints
        """
        if len(points) <= 3:
            return [0, len(points) - 1]

        endpoint_indices = []

        # Calculate average distance to nearest neighbors for each point
        for i, point in enumerate(points):
            distances = []
            for j, other_point in enumerate(points):
                if i != j:
                    distances.append(np.linalg.norm(point - other_point))

            # Sort distances and take the 3 nearest neighbors
            distances.sort()
            avg_near_dist = np.mean(distances[: min(3, len(distances))])

            # Points with higher average distances to neighbors are likely endpoints
            if len(endpoint_indices) == 0 or avg_near_dist > distances[0] * 1.5:
                endpoint_indices.append(i)

        # Limit to a reasonable number of endpoints
        if len(endpoint_indices) > 4:
            # Sort by distance score and take top 4
            endpoint_scores = []
            for idx in endpoint_indices:
                distances = [
                    np.linalg.norm(points[idx] - points[j])
                    for j in range(len(points))
                    if j != idx
                ]
                distances.sort()
                score = np.mean(distances[:3])
                endpoint_scores.append((score, idx))

            endpoint_scores.sort(reverse=True)
            endpoint_indices = [idx for _, idx in endpoint_scores[:4]]

        return endpoint_indices

    def find_y_intersections_at_x(self, path, x):
        """
        Find y-coordinates where a vertical line at x intersects the path.

        Args:
            path: matplotlib Path object
            x: x-coordinate of vertical line

        Returns:
            list: List of y-coordinates of intersections
        """
        vertices = path.vertices
        intersections = []

        # Check each edge of the path
        for i in range(len(vertices) - 1):
            x1, y1 = vertices[i]
            x2, y2 = vertices[i + 1]

            # Check if the edge crosses the vertical line x
            if ((x1 <= x <= x2) or (x2 <= x <= x1)) and x1 != x2:
                # Calculate intersection point
                t = (x - x1) / (x2 - x1)
                y_intersect = y1 + t * (y2 - y1)
                intersections.append(y_intersect)

        return intersections

    def simplify_outline_to_centerline(self, vertices):
        """
        Fallback method: simplify outline to approximate centerline.

        Args:
            vertices: Array of path vertices

        Returns:
            np.array: Simplified centerline points
        """
        if len(vertices) < 4:
            return vertices

        # Use every nth point to create a simplified centerline
        step = max(1, len(vertices) // 20)
        simplified = vertices[::step]

        return simplified

    def remove_duplicate_points(self, points, tolerance=1e-6):
        """
        Remove duplicate points from an array.

        Args:
            points: Array of (x, y) points
            tolerance: Distance tolerance for considering points duplicate

        Returns:
            np.array: Array with duplicates removed
        """
        if len(points) <= 1:
            return points

        unique_points = [points[0]]

        for point in points[1:]:
            # Check if this point is too close to any existing point
            is_duplicate = False
            for existing in unique_points:
                if np.linalg.norm(point - existing) < tolerance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_points.append(point)

        return np.array(unique_points)

    def sort_points_by_connectivity(self, points):
        """
        Sort points to create a connected path (simple nearest-neighbor ordering).

        Args:
            points: Array of (x, y) points

        Returns:
            np.array: Points sorted for connectivity
        """
        if len(points) <= 2:
            return points

        # Start with first point
        sorted_points = [points[0]]
        remaining = list(points[1:])

        while remaining:
            current = sorted_points[-1]

            # Find nearest remaining point
            distances = [np.linalg.norm(current - p) for p in remaining]
            nearest_idx = np.argmin(distances)

            sorted_points.append(remaining.pop(nearest_idx))

        return np.array(sorted_points)

    def extract_path_contours(self, path):
        """
        Extract individual contours from a path.

        Args:
            path: matplotlib Path object

        Returns:
            list: List of contour dictionaries with 'vertices' and 'codes'
        """
        vertices = path.vertices
        codes = path.codes

        if codes is None:
            # If no codes, treat as single contour
            return [{"vertices": vertices, "codes": None}]

        contours = []
        current_vertices = []
        current_codes = []

        for i, (vertex, code) in enumerate(zip(vertices, codes)):
            from matplotlib.path import Path as MPLPath

            if code == MPLPath.MOVETO and current_vertices:
                # Start of new contour, save previous one
                contours.append(
                    {
                        "vertices": np.array(current_vertices),
                        "codes": np.array(current_codes),
                    }
                )
                current_vertices = []
                current_codes = []

            current_vertices.append(vertex)
            current_codes.append(code)

        # Add the final contour
        if current_vertices:
            contours.append(
                {
                    "vertices": np.array(current_vertices),
                    "codes": np.array(current_codes),
                }
            )

        return contours

    def find_main_contour(self, contours):
        """
        Find the main (largest area) contour from a list of contours.
        This represents the outer boundary for zero stroke width.

        Args:
            contours (list): List of contour dictionaries

        Returns:
            dict: Main contour dictionary, or None if no suitable contour found
        """
        if not contours:
            return None

        if len(contours) == 1:
            return contours[0]

        # Calculate area for each contour using shoelace formula
        main_contour = None
        max_area = 0

        for contour in contours:
            vertices = contour["vertices"]
            if len(vertices) < 3:
                continue

            # Calculate area using shoelace formula
            area = (
                abs(
                    sum(
                        vertices[i][0] * vertices[(i + 1) % len(vertices)][1]
                        - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]
                        for i in range(len(vertices))
                    )
                )
                / 2.0
            )

            if area > max_area:
                max_area = area
                main_contour = contour

        return main_contour

    def extract_contour_points(self, path, num_points=500):
        """
        Extract HUNDREDS of points along the MIDPOINT/CENTERLINE of the letter path.
        This creates a zero-thickness trace through the center of the letter strokes.

        Args:
            path: matplotlib Path object
            num_points (int): Number of points to extract (default: 500 for high detail)

        Returns:
            list: List of contour arrays, each representing the centerline of the letter
        """
        if not path or len(path.vertices) == 0:
            return []

        vertices = path.vertices
        codes = path.codes

        print(f"    Extracting {num_points} centerline points from letter path...")

        # Extract the centerline by tracing through the middle of the letter strokes
        centerline_points = self.extract_centerline_from_path(
            vertices, codes, num_points
        )

        if len(centerline_points) < 10:
            print(f"    Warning: Only {len(centerline_points)} centerline points found")
            return [centerline_points] if len(centerline_points) > 0 else []

        print(f"    SUCCESS: Generated {len(centerline_points)} centerline points")
        return [centerline_points]

    def extract_centerline_from_path(self, vertices, codes, num_points=500):
        """
        Extract TRUE CENTERLINE points by finding midpoints between opposing boundaries.
        This creates a skeletal representation with zero thickness.

        Args:
            vertices: Path vertices
            codes: Path codes
            num_points (int): Target number of points to generate

        Returns:
            np.array: Array of (x, y) points representing the true letter centerline
        """
        if len(vertices) < 3:
            return np.array(vertices)

        print(
            f"        SKELETON EXTRACTION: Finding true centerline from {len(vertices)} outline points"
        )

        # Create a temporary path object for the skeleton extraction
        from matplotlib.path import Path as MPLPath

        temp_path = MPLPath(vertices, codes)

        # Extract skeleton using the new midpoint-based approach
        skeleton_points = self.extract_skeleton_from_path(temp_path)

        if len(skeleton_points) < 3:
            # Fallback to simplified outline approach
            print(
                f"        FALLBACK: Using simplified outline ({len(skeleton_points)} skeleton points)"
            )
            skeleton_points = self.simplify_outline_to_centerline(vertices)

        # Resample to get exactly the requested number of points
        if len(skeleton_points) >= num_points:
            # Downsample using even spacing
            indices = np.linspace(0, len(skeleton_points) - 1, num_points, dtype=int)
            resampled_points = skeleton_points[indices]
        else:
            # Upsample by interpolating between existing points
            resampled_points = self.upsample_centerline(skeleton_points, num_points)

        print(
            f"        SKELETON SUCCESS: Generated {len(resampled_points)} true centerline points"
        )
        return resampled_points

    def upsample_centerline(self, points, target_count):
        """
        Upsample a centerline by interpolating between existing points.

        Args:
            points: Existing centerline points
            target_count: Desired number of points

        Returns:
            np.array: Upsampled centerline points
        """
        if len(points) < 2:
            return points

        # Calculate cumulative distances along the centerline
        distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)

        total_length = distances[-1]
        if total_length == 0:
            return points

        # Create evenly spaced target distances
        target_distances = np.linspace(0, total_length, target_count)

        # Interpolate points at target distances
        upsampled_points = []
        for target_dist in target_distances:
            # Find the segment containing this distance
            idx = np.searchsorted(distances, target_dist)
            if idx == 0:
                upsampled_points.append(points[0])
            elif idx >= len(points):
                upsampled_points.append(points[-1])
            else:
                # Interpolate between points[idx-1] and points[idx]
                t = (target_dist - distances[idx - 1]) / (
                    distances[idx] - distances[idx - 1]
                )
                interpolated = points[idx - 1] + t * (points[idx] - points[idx - 1])
                upsampled_points.append(interpolated)

        return np.array(upsampled_points)

    def extract_path_segments(self, vertices, codes):
        """
        Extract continuous segments from path vertices and codes.

        Args:
            vertices: Path vertices
            codes: Path codes

        Returns:
            list: List of segments, each being an array of points
        """
        from matplotlib.path import Path as MPLPath

        if codes is None:
            return [vertices]

        segments = []
        current_segment = []

        for i, (vertex, code) in enumerate(zip(vertices, codes)):
            if code == MPLPath.MOVETO:
                if current_segment:
                    segments.append(np.array(current_segment))
                current_segment = [vertex]
            elif code in [MPLPath.LINETO, MPLPath.CURVE3, MPLPath.CURVE4]:
                current_segment.append(vertex)
            elif code == MPLPath.CLOSEPOLY:
                if current_segment and len(current_segment) > 1:
                    current_segment.append(current_segment[0])  # Close the path
                    segments.append(np.array(current_segment))
                current_segment = []

        # Add final segment
        if current_segment and len(current_segment) > 1:
            segments.append(np.array(current_segment))

        return segments

    def calculate_segment_length(self, segment):
        """
        Calculate the total length of a path segment.

        Args:
            segment: Array of (x, y) points

        Returns:
            float: Total length of the segment
        """
        if len(segment) < 2:
            return 0

        distances = np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1))
        return np.sum(distances)

    def interpolate_along_segment(self, segment, ratio):
        """
        Interpolate a point at a given ratio along a segment.

        Args:
            segment: Array of (x, y) points
            ratio: Position along segment (0.0 to 1.0)

        Returns:
            np.array: Interpolated (x, y) point
        """
        if len(segment) < 2:
            return segment[0] if len(segment) > 0 else np.array([0, 0])

        if ratio <= 0:
            return segment[0]
        if ratio >= 1:
            return segment[-1]

        # Calculate cumulative distances
        distances = np.cumsum(np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)

        total_length = distances[-1]
        if total_length == 0:
            return segment[0]

        # Find target distance
        target_distance = ratio * total_length

        # Find the segment index
        idx = np.searchsorted(distances, target_distance)
        if idx == 0:
            return segment[0]
        if idx >= len(segment):
            return segment[-1]

        # Interpolate between segment[idx-1] and segment[idx]
        seg_start = distances[idx - 1]
        seg_end = distances[idx]

        if seg_end == seg_start:
            return segment[idx - 1]

        local_ratio = (target_distance - seg_start) / (seg_end - seg_start)

        return segment[idx - 1] + local_ratio * (segment[idx] - segment[idx - 1])

    def preview_extracted_points(
        self, text, font_size=100, num_points=500, save_path=None
    ):
        """
        Preview how the extracted centerline points look on the plane.
        Shows both the original outline and the extracted skeleton points.

        Args:
            text (str): Text to extract and preview
            font_size (int): Font size for rendering
            num_points (int): Number of centerline points to extract
            save_path (str, optional): Path to save the preview image

        Returns:
            None: Displays the preview plot
        """
        print(f"Generating preview for text: '{text}'")

        # Extract paths for the text
        paths = self.text_to_paths(text, font_size)

        if not paths:
            print("No paths generated for preview")
            return

        # Create figure with subplots for each character
        num_chars = len(paths)
        fig, axes = plt.subplots(1, max(1, num_chars), figsize=(4 * num_chars, 6))

        if num_chars == 1:
            axes = [axes]  # Make it a list for consistency

        for i, path in enumerate(paths):
            ax = axes[i] if num_chars > 1 else axes[0]

            # Plot original outline
            self.plot_path_outline(
                ax, path, color="lightgray", label="Original Outline"
            )

            # Extract and plot centerline points
            contours = self.extract_contour_points(path, num_points)

            for j, contour in enumerate(contours):
                if len(contour) > 0:
                    # Plot centerline points as connected line
                    ax.plot(
                        contour[:, 0],
                        contour[:, 1],
                        "ro-",
                        markersize=2,
                        linewidth=1,
                        label=f"Centerline {j + 1}" if j == 0 else "",
                    )

                    # Plot first and last points differently
                    if len(contour) > 1:
                        ax.plot(
                            contour[0, 0],
                            contour[0, 1],
                            "go",
                            markersize=6,
                            label="Start" if j == 0 else "",
                        )
                        ax.plot(
                            contour[-1, 0],
                            contour[-1, 1],
                            "bo",
                            markersize=6,
                            label="End" if j == 0 else "",
                        )

            # Customize subplot
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"Character {i + 1}: Centerline Extraction\n({num_points} points)"
            )
            if i == 0:  # Only show legend on first subplot
                ax.legend()

            # Invert y-axis to match typical text orientation
            ax.invert_yaxis()

        plt.tight_layout()
        plt.suptitle(
            f"Centerline Preview: '{text}' (Zero-Width Skeleton)", fontsize=14, y=1.02
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Preview saved to: {save_path}")

        # Close figure instead of show to avoid backend warnings
        plt.close("all")

    def plot_path_outline(self, ax, path, color="blue", alpha=0.5, label="Outline"):
        """
        Plot the outline of a matplotlib Path object.

        Args:
            ax: Matplotlib axes object
            path: matplotlib Path object
            color: Color for the outline
            alpha: Transparency level
            label: Label for the plot
        """
        vertices = path.vertices
        codes = path.codes

        if codes is None:
            # Simple line plot if no codes
            ax.plot(
                vertices[:, 0], vertices[:, 1], color=color, alpha=alpha, label=label
            )
            return

        # Plot path segments according to codes
        from matplotlib.path import Path as MPLPath

        # Track if we've added a label yet
        label_used = False

        current_pos = None
        for i, (vertex, code) in enumerate(zip(vertices, codes)):
            if code == MPLPath.MOVETO:
                current_pos = vertex
            elif code == MPLPath.LINETO:
                if current_pos is not None:
                    # Only add label to first line segment
                    line_label = label if not label_used else None
                    ax.plot(
                        [current_pos[0], vertex[0]],
                        [current_pos[1], vertex[1]],
                        color=color,
                        alpha=alpha,
                        linewidth=1,
                        label=line_label,
                    )
                    label_used = True
                current_pos = vertex
            elif code == MPLPath.CLOSEPOLY:
                # Close the path back to the start of current segment
                if current_pos is not None and i > 0:
                    # Find the most recent MOVETO
                    start_vertex = None
                    for j in range(i - 1, -1, -1):
                        if codes[j] == MPLPath.MOVETO:
                            start_vertex = vertices[j]
                            break
                    if start_vertex is not None:
                        line_label = label if not label_used else None
                        ax.plot(
                            [current_pos[0], start_vertex[0]],
                            [current_pos[1], start_vertex[1]],
                            color=color,
                            alpha=alpha,
                            linewidth=1,
                            label=line_label,
                        )
                        label_used = True

    def preview_skeleton_extraction_steps(self, text, font_size=100, save_path=None):
        """
        Preview the skeleton extraction process step by step.
        Shows outline, intersection grid, and final skeleton.

        Args:
            text (str): Text to process
            font_size (int): Font size for rendering
            save_path (str, optional): Path to save the preview
        """
        print(f"Generating detailed skeleton preview for: '{text}'")

        paths = self.text_to_paths(text, font_size)

        if not paths:
            print("No paths generated for skeleton preview")
            return

        fig, axes = plt.subplots(2, len(paths), figsize=(6 * len(paths), 12))

        # Ensure axes is always 2D for consistent indexing
        if len(paths) == 1:
            axes = axes.reshape(2, 1)

        for i, path in enumerate(paths):
            # Top subplot: Original outline + intersection grid
            ax_top = axes[0, i]

            # Plot original outline
            self.plot_path_outline(
                ax_top, path, color="black", alpha=0.8, label="Original Outline"
            )

            # Show intersection grid
            self.plot_intersection_grid(ax_top, path, alpha=0.3)

            ax_top.set_aspect("equal")
            ax_top.grid(True, alpha=0.2)
            ax_top.set_title("Step 1: Outline + Sampling Grid")
            # Only show legend if there are labeled elements
            handles, labels = ax_top.get_legend_handles_labels()
            if handles:
                ax_top.legend()
            ax_top.invert_yaxis()

            # Bottom subplot: Skeleton extraction result
            ax_bottom = axes[1, i]

            # Plot faded outline for reference
            self.plot_path_outline(
                ax_bottom, path, color="lightgray", alpha=0.5, label="Original (faded)"
            )

            # Extract and plot skeleton
            skeleton_points = self.extract_skeleton_from_path(path)

            if len(skeleton_points) > 0:
                ax_bottom.plot(
                    skeleton_points[:, 0],
                    skeleton_points[:, 1],
                    "ro-",
                    markersize=3,
                    linewidth=1.5,
                    label="Skeleton Points",
                )

                # Show midpoint calculation examples
                self.plot_midpoint_examples(ax_bottom, path, skeleton_points[:10])

            ax_bottom.set_aspect("equal")
            ax_bottom.grid(True, alpha=0.2)
            ax_bottom.set_title(
                f"Step 2: Extracted Skeleton\n({len(skeleton_points)} points)"
            )
            # Only show legend if there are labeled elements
            handles, labels = ax_bottom.get_legend_handles_labels()
            if handles:
                ax_bottom.legend()
            ax_bottom.invert_yaxis()

        plt.tight_layout()
        plt.suptitle(f"Skeleton Extraction Process: '{text}'", fontsize=16, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Skeleton preview saved to: {save_path}")

        # Close figure instead of show to avoid backend warnings
        plt.close("all")

    def plot_intersection_grid(self, ax, path, resolution=20, alpha=0.3):
        """
        Plot the sampling grid used for intersection finding.

        Args:
            ax: Matplotlib axes
            path: matplotlib Path object
            resolution: Grid resolution
            alpha: Transparency for grid lines
        """
        vertices = path.vertices
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)

        # Create sampling grid
        x_samples = np.linspace(min_x, max_x, resolution)
        y_samples = np.linspace(min_y, max_y, resolution)

        # Plot vertical grid lines
        for x in x_samples[::3]:  # Show every 3rd line to avoid clutter
            ax.axvline(x, color="blue", alpha=alpha, linewidth=0.5, linestyle="--")

        # Plot horizontal grid lines
        for y in y_samples[::3]:  # Show every 3rd line to avoid clutter
            ax.axhline(y, color="red", alpha=alpha, linewidth=0.5, linestyle="--")

    def plot_midpoint_examples(self, ax, path, sample_skeleton_points, num_examples=3):
        """
        Plot examples showing how midpoints are calculated.

        Args:
            ax: Matplotlib axes
            path: Original path
            sample_skeleton_points: Sample skeleton points to demonstrate
            num_examples: Number of examples to show
        """
        if len(sample_skeleton_points) < num_examples:
            return

        # Show a few example midpoint calculations
        for i in range(min(num_examples, len(sample_skeleton_points))):
            point = sample_skeleton_points[i]
            x, y = point

            # Find intersections at this y-level
            x_intersections = self.find_x_intersections_at_y(path, y)

            if len(x_intersections) >= 2:
                x_intersections = sorted(x_intersections)

                # Find the pair that this midpoint belongs to
                for j in range(0, len(x_intersections) - 1, 2):
                    if j + 1 < len(x_intersections):
                        left_x = x_intersections[j]
                        right_x = x_intersections[j + 1]
                        mid_x = (left_x + right_x) / 2

                        # If this is close to our skeleton point, show the calculation
                        if abs(mid_x - x) < 0.1:  # Tolerance for matching
                            # Draw line between intersection points
                            ax.plot(
                                [left_x, right_x], [y, y], "g-", alpha=0.7, linewidth=2
                            )
                            # Mark intersection points
                            ax.plot([left_x, right_x], [y, y], "gs", markersize=4)
                            # Mark midpoint
                            ax.plot(mid_x, y, "ro", markersize=5)
                            break
