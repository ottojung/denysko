#!/usr/bin/env python3
"""
Text extraction module - handles font path extraction and contour point sampling.
"""

import numpy as np
from matplotlib import font_manager


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
        Extract a medial-like centerline that collapses each stroke to a single path.
        Prioritizes morphological thinning (single-pixel skeleton), falls back to
        chamfer-distance ridge, and only then to scanline midpoints as a last resort.
        """
        vertices = path.vertices
        if len(vertices) < 6:
            return vertices

        # 1) Rasterize once for all downstream steps
        mask, x_grid, y_grid = self._rasterize_path(path, resolution=512)
        if mask.sum() == 0:
            return vertices

        # 2) Primary: Zhang–Suen thinning yields true single-pixel skeletons
        skel = self._zhang_suen_thinning(mask)

        # 3) Fallback to medial ridge if thinning is too sparse
        if skel.sum() < 5:
            dist = self._chamfer_distance_transform(mask)
            skel = self._medial_ridge(mask, dist)

        # 4) If we still don't have enough, last resort: scanline midpoints
        if skel.sum() < 5:
            pts = self._scanline_midpoint_skeleton(path)
            if pts is not None and len(pts) >= 10:
                pts = self._smooth_polyline(pts, window=9)
                pts = self._dedupe_close_points(pts, tol=1e-2)
                return pts
            return self.create_simple_stroke_approximation(path)

        # 5) Trace skeleton components to ordered polylines and pick the longest
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
        Multi-angle scanline midpoint extraction that returns a SINGLE path.
        For each angle, we keep at most one midpoint (widest span) per scanline,
        then choose the single best angle path (longest, smoothest) and return it.
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

        center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=float)

        candidates = []
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

            midpoints = []
            for off in offsets:
                p0 = center + off * n
                ts = self._intersections_with_infinite_line(vertices, p0, d)
                if len(ts) < 2:
                    continue
                ts.sort()
                # Choose the widest interior span only
                best_pair = None
                best_width = -np.inf
                for i in range(0, len(ts) - 1, 2):
                    t1 = ts[i]
                    t2 = ts[i + 1]
                    w = t2 - t1
                    if w > best_width:
                        best_width = w
                        best_pair = (t1, t2)
                if best_pair is None or best_width < min_span:
                    continue
                t1, t2 = best_pair
                m = p0 + 0.5 * (t1 + t2) * d
                if path.contains_point(m):
                    midpoints.append(m)

            if len(midpoints) < 10:
                continue
            pts = np.array(midpoints)
            pts = self.remove_duplicate_points(pts, tolerance=1e-3)
            # Order and smooth
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
            if len(ordered) >= 3:
                length = float(np.sum(np.linalg.norm(np.diff(ordered, axis=0), axis=1)))
                candidates.append((length, ordered))

        if not candidates:
            return None
        # Return the longest candidate (single path)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def extract_contour_points(self, path, num_points=500):
        """Extract a single centerline polyline from a Path and resample to num_points."""
        if not path or len(path.vertices) == 0:
            return []
        vertices = path.vertices
        codes = path.codes
        pts = self.extract_centerline_from_path(vertices, codes, num_points)
        return [pts] if len(pts) > 0 else []

    def extract_centerline_from_path(self, vertices, codes, num_points=500):
        """Build a Path from vertices/codes, extract skeleton, and resample to num_points."""
        if len(vertices) < 3:
            return np.array(vertices)
        from matplotlib.path import Path as MPLPath
        temp = MPLPath(vertices, codes)
        skel = self.extract_skeleton_from_path(temp)
        if len(skel) < 3:
            skel = self.create_simple_stroke_approximation(temp)
        # Resample
        if len(skel) >= num_points:
            idx = np.linspace(0, len(skel) - 1, num_points, dtype=int)
            return skel[idx]
        return self.upsample_centerline(skel, num_points)

    def upsample_centerline(self, points, target_count):
        """Upsample centerline to target_count via arc-length interpolation."""
        if len(points) < 2:
            return points
        seg = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        d = np.insert(np.cumsum(seg), 0, 0.0)
        total = d[-1]
        if total == 0:
            return points
        targets = np.linspace(0, total, target_count)
        out = []
        j = 1
        for t in targets:
            while j < len(d) and d[j] < t:
                j += 1
            if j == 0:
                out.append(points[0])
            elif j >= len(d):
                out.append(points[-1])
            else:
                t0, t1 = d[j-1], d[j]
                if t1 == t0:
                    out.append(points[j-1])
                else:
                    a = (t - t0) / (t1 - t0)
                    out.append(points[j-1] + a * (points[j] - points[j-1]))
        return np.array(out)

    def preview_extracted_points(self, text, font_size=100, num_points=500, save_path=None):
        """Render outline and extracted centerline for each character and save image."""
        import matplotlib.pyplot as plt
        print(f"Generating preview for text: '{text}'")
        paths = self.text_to_paths(text, font_size)
        if not paths:
            print("No paths generated for preview")
            return
        n = len(paths)
        fig, axes = plt.subplots(1, max(1, n), figsize=(4*n, 6))
        if n == 1:
            axes = [axes]
        for i, path in enumerate(paths):
            ax = axes[i]
            self.plot_path_outline(ax, path, color='lightgray', alpha=0.8, label='Outline')
            contours = self.extract_contour_points(path, num_points)
            for j, contour in enumerate(contours):
                if len(contour) > 0:
                    ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1.2, label='Centerline' if j == 0 else None)
                    ax.plot(contour[0, 0], contour[0, 1], 'go', markersize=6, label='Start' if j == 0 else None)
                    ax.plot(contour[-1, 0], contour[-1, 1], 'bo', markersize=6, label='End' if j == 0 else None)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
            ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Preview saved to: {save_path}")
        plt.close('all')

    def preview_skeleton_extraction_steps(self, text, font_size=100, save_path=None):
        """Minimal step preview: show outline and final skeleton per character."""
        import matplotlib.pyplot as plt
        print(f"Generating detailed skeleton preview for: '{text}'")
        paths = self.text_to_paths(text, font_size)
        if not paths:
            print("No paths generated for skeleton preview")
            return
        n = len(paths)
        fig, axes = plt.subplots(1, max(1, n), figsize=(6*n, 6))
        if n == 1:
            axes = [axes]
        for i, path in enumerate(paths):
            ax = axes[i]
            self.plot_path_outline(ax, path, color='black', alpha=0.5, label='Outline')
            skel = self.extract_skeleton_from_path(path)
            if len(skel) > 0:
                ax.plot(skel[:, 0], skel[:, 1], 'r.-', markersize=3, linewidth=1.2, label='Skeleton')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
            ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Skeleton preview saved to: {save_path}")
        plt.close('all')

    def plot_path_outline(self, ax, path, color='blue', alpha=0.5, label='Outline'):
        """Plot the outline of a Path on provided axes, honoring codes if present."""
        vertices = path.vertices
        codes = path.codes
        if codes is None:
            ax.plot(vertices[:, 0], vertices[:, 1], color=color, alpha=alpha, label=label)
            return
        from matplotlib.path import Path as MPLPath
        label_used = False
        current = None
        for v, c in zip(vertices, codes):
            if c == MPLPath.MOVETO:
                current = v
            elif c == MPLPath.LINETO:
                if current is not None:
                    ax.plot([current[0], v[0]], [current[1], v[1]], color=color, alpha=alpha, linewidth=1,
                            label=(label if not label_used else None))
                    label_used = True
                current = v
            elif c == MPLPath.CLOSEPOLY:
                # Close back to last MOVETO
                pass
