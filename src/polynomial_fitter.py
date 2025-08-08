#!/usr/bin/env python3
"""
Polynomial fitting module - REIMPLEMENTED FROM SCRATCH

Two guiding principles:
1. Exact fitting to as many points as possible
2. Separation into multiple curves when multiple strokes cover same horizontal domain

Algorithm:
- Detect horizontal overlap (multiple y values at similar x coordinates)
- Split overlapping regions into separate curves (vertical split: upper/lower)
- Fit ONE exact polynomial through ALL points in each curve (no x-domain pieces)
- Use degree = n - 1 for n points (exact interpolation)
- IMPORTANT: No domain restrictions in the output; functions are for all real x
"""

import numpy as np
from numpy.polynomial import polynomial as P  # for polyadd, polymul, polypow


class PolynomialFitter:
    """Fits exact polynomials to letter coordinate points."""

    def __init__(self):
        """Initialize fitter with piecewise LS configuration (no domain restrictions)."""
        # Stroke split
        self.min_points_per_stroke = 5
        # Segmenting & fitting
        self.min_points_per_segment = 10
        self.max_points_per_segment = 80
        self.min_seg_ratio = 0.08  # initial random window width as fraction of x-span
        self.max_seg_ratio = 0.18
        self.seg_jitter_ratio = 0.0  # no jitter
        self.r2_threshold = 0.97
        self.max_expand_steps = 8
        # Guarding to "go away" after the segment (small-weight penalties outside window)
        self.guard_weight = 0.05
        self.guard_margin_ratio = 0.03  # how far from window edges to place guard x's

    def fit_contour_polynomials(self, contour):
        """
        Main fitting method - implements the two core principles.

        Args:
            contour: Array of (x, y) letter centerline points

        Returns:
            list: Polynomial function strings (global, no domain restrictions)
        """
        if len(contour) < 10:  # Require many points for quality fitting
            print(
                f"Warning: Only {len(contour)} points, need at least 10 for quality fitting"
            )
            return []

        print(f"Fitting {len(contour)} letter points...")

        # Split overlapping horizontal strokes (e.g., crossbars)
        curves = self._detect_overlapping_strokes(contour)
        print(f"Found {len(curves)} separate curves")

        functions = []
        for i, curve in enumerate(curves):
            print(f"Curve {i + 1}: {len(curve)} points")
            funcs = self._fit_piecewise_poly_segments(curve)
            functions.extend(funcs)

        print(f"Generated {len(functions)} polynomials (piecewise, global)")
        return functions

    def _detect_overlapping_strokes(self, points):
        """
        Detect if multiple strokes occupy the same horizontal space.
        Key for separating letter diagonals from crossbar.

        Args:
            points: Array of (x, y) coordinates

        Returns:
            list: Separated stroke arrays
        """
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Check for significant y-variation within x-ranges (indicates overlap)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        x_span = x_max - x_min

        if x_span < 1e-6:
            return [points]  # All same x

        # Divide into x-segments and check y-variation in each
        num_segments = 8
        overlap_found = False

        for i in range(num_segments):
            seg_start = x_min + i * x_span / num_segments
            seg_end = x_min + (i + 1) * x_span / num_segments

            # Points in this x-segment
            in_segment = (x_coords >= seg_start) & (x_coords <= seg_end)
            if np.sum(in_segment) < 2:
                continue

            seg_y = y_coords[in_segment]
            y_variation = np.max(seg_y) - np.min(seg_y)

            # If y varies significantly, we have overlapping strokes
            if y_variation > x_span * 0.2:  # 20% threshold (heuristic)
                overlap_found = True
                break

        if not overlap_found:
            return [points]  # Single stroke

        # Separate into upper and lower strokes
        y_median = np.median(y_coords)

        upper = points[y_coords >= y_median]
        lower = points[y_coords < y_median]

        strokes = []
        if len(upper) >= self.min_points_per_stroke:
            strokes.append(upper)
        if len(lower) >= self.min_points_per_stroke:
            strokes.append(lower)

        return strokes if strokes else [points]

    def _fit_exact_polynomial_single(self, points):
        """Exact interpolation on the entire curve (no x-domain split).
        Returns a list with a single function string, valid for all real x.
        The only allowed split is vertical (upper/lower strokes handled elsewhere).
        """
        # Sort by x and handle duplicates
        x_data, y_data = points[:, 0], points[:, 1]
        sort_idx = np.argsort(x_data)
        x_sorted, y_sorted = x_data[sort_idx], y_data[sort_idx]

        # Average duplicate x-values to ensure unique x's for interpolation
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        unique_y = np.array(
            [np.mean(y_sorted[inverse == i]) for i in range(len(unique_x))]
        )

        n = len(unique_x)
        if n < self.min_points_per_stroke:
            print(
                f"  Skipping: only {n} unique x-coordinates, need ≥{self.min_points_per_stroke}"
            )
            return []

        func = self._fit_exact_single(unique_x, unique_y)
        return [func] if func else []

    def _fit_exact_single(self, x_vals, y_vals):
        """Fit a single exact polynomial of degree len(x_vals)-1 to the segment.
        Uses affine scaling to [-1,1] to stabilize the Vandermonde solve, then
        composes back to the x-basis to emit a global polynomial (no domain).
        """
        n_points = len(x_vals)
        degree = max(2, n_points - 1)
        print(f"  Points: {n_points}, Natural degree: {n_points - 1}, Using: {degree}")
        try:
            xmin = float(np.min(x_vals))
            xmax = float(np.max(x_vals))
            span = xmax - xmin if xmax > xmin else 1.0
            # Affine map: x in [xmin,xmax] -> z in [-1,1]: z = a*x + b
            a = 2.0 / span
            b = -(xmax + xmin) / span
            z_vals = a * x_vals + b

            # Build stable Vandermonde in ascending powers of z and solve for exact coefficients
            Vz = np.vander(
                z_vals, N=degree + 1, increasing=True
            )  # columns: z^0, z^1, ..., z^degree
            cz = np.linalg.solve(Vz, y_vals)  # cz[k] is coeff for z^k

            # Compose p(z) with z=a*x+b in coefficient space (ascending order)
            # px(x) = sum_k cz[k] * (a*x + b)^k
            z_poly = np.array([b, a], dtype=float)  # coefficients of b + a*x
            px = np.array([0.0], dtype=float)
            for k, ck in enumerate(cz):
                if abs(ck) < 1e-18:
                    continue
                # (a*x + b)^k
                zk = P.polypow(z_poly, k)
                term = zk * ck
                # pad and add
                px = P.polyadd(px, term)

            # Verify exactness back on original x
            y_fit = P.polyval(x_vals, px)  # ascending order
            max_err = float(np.max(np.abs(y_fit - y_vals)))
            print(f"  Max error = {max_err:.8e}")
            if not np.isfinite(max_err) or max_err > 1e-8:
                print("  WARNING: Residual detected; expected exact fit")

            # Convert to string (need high->low order)
            func = self._coeffs_to_string(
                px[::-1]
            )  # reverse to high->low for string builder
            if func is None:
                return None
            print("  Generated function (no domain)")
            return func
        except np.linalg.LinAlgError as e:
            print(f"  Linear solve failed: {e}")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None

    def _encode_exponent(self, power):
        """Encode multi-digit exponents by chaining digits with carets for Desmos.
        Example: power=1234 -> '1^2^3^4'. Single-digit powers remain unchanged.
        """
        p = int(power)
        if p <= 9:
            return str(p)
        return "^".join(str(p))

    def _format_number(self, value, precision: int = 15) -> str:
        """Format a float without scientific notation (no 'e'/'E'), trimming trailing zeros.
        Ensures outputs like 0.0000001 instead of 1e-7 for better Desmos compatibility.
        """
        try:
            s = np.format_float_positional(
                float(value), precision=precision, unique=False, trim="-"
            )
            # Normalize negative zero to plain zero
            if s.startswith("-0") and float(value) == 0.0:
                return "0"
            return s
        except Exception:
            return str(value)

    def _coeffs_to_string(self, coeffs):
        """Convert coefficients (high->low) to clean function string (no domain constraints)."""
        if len(coeffs) < 3:
            print(
                f"  Error: Polynomial degree too low - got {len(coeffs) - 1}, need ≥ 2"
            )
            return None
        terms = []
        degree = len(coeffs) - 1
        print(f"  Converting degree {degree} polynomial to string")
        for i, c in enumerate(coeffs):
            power = degree - i
            if not np.isfinite(c) or abs(c) < 1e-15:
                continue

            if power == 0:
                c_str = self._format_number(c, precision=15)
                terms.append(c_str)
            elif power == 1:
                if abs(c - 1.0) < 1e-15:
                    terms.append("x")
                elif abs(c + 1.0) < 1e-15:
                    terms.append("-x")
                else:
                    c_str = self._format_number(c, precision=15)
                    terms.append(f"{c_str}*x")
            else:
                power_str = self._encode_exponent(power)
                if abs(c - 1.0) < 1e-15:
                    terms.append(f"x^{power_str}")
                elif abs(c + 1.0) < 1e-15:
                    terms.append(f"-x^{power_str}")
                else:
                    c_str = self._format_number(c, precision=15)
                    terms.append(f"{c_str}*x^{power_str}")

        if not terms:
            print("  Warning: All coefficients were essentially zero")
            return "y = 0"
        result = "y = " + terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += " - " + term[1:]
            else:
                result += " + " + term
        result = result.replace(" + -", " - ")
        print(f"  Generated function: {result}")
        return result

    def fit_contours_to_polynomials(self, contours):
        """Fit polynomials to multiple contours (exact fit; one per stroke)."""
        all_functions = []

        for i, contour in enumerate(contours):
            print(f"\nContour {i + 1}/{len(contours)}:")
            functions = self.fit_contour_polynomials(contour)
            all_functions.extend(functions)

        return all_functions

    def _fit_piecewise_poly_segments(self, points):
        """Fit degree-2..4 polynomials to random local segments, ensuring coverage.
        Returns a list of function strings (each global; no domain suffix).
        """
        # Sort and deduplicate x by averaging y at duplicates
        x, y = points[:, 0], points[:, 1]
        order = np.argsort(x)
        x, y = x[order], y[order]
        ux, inv = np.unique(x, return_inverse=True)
        uy = np.array([np.mean(y[inv == i]) for i in range(len(ux))])

        n = len(ux)
        if n < self.min_points_per_segment:
            print(
                f"  Skipping: only {n} unique x-coordinates, need ≥{self.min_points_per_segment}"
            )
            return []

        x_min, x_max = float(np.min(ux)), float(np.max(ux))
        span = max(x_max - x_min, 1e-9)
        total_y_range = float(np.max(uy) - np.min(uy)) if n > 1 else 1.0
        covered = np.zeros(n, dtype=bool)
        result = []

        rng = np.random.default_rng()

        i = 0
        while not np.all(covered):
            # pick leftmost uncovered
            i = int(np.argmax(~covered))
            x0 = ux[i]
            # random window width (deterministic start, no jitter)
            w_ratio = float(rng.uniform(self.min_seg_ratio, self.max_seg_ratio))
            w = w_ratio * span
            # deterministic start: clamp so window fits within [x_min, x_max]
            start = min(max(x0, x_min), x_max - w)
            end = start + w
            if end - start < (self.min_seg_ratio * span * 0.5):
                end = min(x_max, start + self.min_seg_ratio * span)

            seg_mask = (ux >= start) & (ux <= end)
            # ensure we have enough points; if not, expand rightwards
            expand_steps = 0
            while np.sum(seg_mask) < self.min_points_per_segment and expand_steps < self.max_expand_steps:
                end = min(x_max, end + 0.5 * w)
                seg_mask = (ux >= start) & (ux <= end)
                expand_steps += 1
            indices = np.where(seg_mask)[0]
            if len(indices) < self.min_points_per_segment:
                # fallback: take next block of min points
                j2 = min(i + self.min_points_per_segment, n)
                indices = np.arange(i, j2)
                start, end = ux[indices[0]], ux[indices[-1]]

            best = self._fit_best_degree_window(
                ux[indices], uy[indices], x_min, x_max, total_y_range
            )
            # try expanding to cover more points if R^2 stays high
            expand_steps = 0
            while expand_steps < self.max_expand_steps:
                # attempt to extend right by a small block
                j_end = indices[-1]
                block = min(self.min_points_per_segment // 2, n - j_end - 1)
                if block <= 0:
                    break
                cand_idx = np.arange(indices[0], j_end + 1 + block)
                cand = self._fit_best_degree_window(
                    ux[cand_idx], uy[cand_idx], x_min, x_max, total_y_range
                )
                if cand and cand["r2"] >= self.r2_threshold:
                    indices = cand_idx
                    best = cand
                    expand_steps += 1
                else:
                    break

            if best is None:
                print("  WARNING: could not fit a satisfactory segment; proceeding")
            else:
                result.append(best["func"])  # already formatted

            # mark covered
            covered[indices] = True

        return result

    def _fit_best_degree_window(self, xw, yw, x_min, x_max, total_y_range):
        """Fit degrees 2..4 by weighted LS in normalized z, pick best by R^2.
        Adds low-weight guard points just outside window to discourage overlap.
        Returns dict with {func, r2} or None.
        """
        if len(xw) < self.min_points_per_segment:
            return None
        # normalize to z = a*x + b mapping x in [x_min,x_max] to roughly [-1,1]
        span = max(x_max - x_min, 1e-9)
        a = 2.0 / span
        b = -(x_max + x_min) / span
        zw = a * xw + b

        # Build guards (left and right of window)
        left_edge, right_edge = float(xw[0]), float(xw[-1])
        margin = self.guard_margin_ratio * span
        guard_x = []
        if left_edge - margin > x_min:
            guard_x.extend(np.linspace(x_min, left_edge - margin, 3))
        if right_edge + margin < x_max:
            guard_x.extend(np.linspace(right_edge + margin, x_max, 3))
        guard_x = np.array(guard_x, dtype=float) if guard_x else np.array([], dtype=float)
        zg = a * guard_x + b if guard_x.size else guard_x
        # push predictions away from stroke y-range (repulsion)
        if total_y_range <= 0:
            y_center = float(np.mean(yw))
            y_far = y_center + 10.0
        else:
            y_mid = 0.5 * (float(np.min(yw)) + float(np.max(yw)))
            y_far = y_mid + 5.0 * total_y_range
        yg = np.full_like(guard_x, y_far)

        best = None
        for deg in (2, 3, 4):
            # Design matrices in ascending powers of z
            Vw = np.vander(zw, N=deg + 1, increasing=True)
            if zg.size:
                Vg = np.vander(zg, N=deg + 1, increasing=True)
                V = np.vstack([Vw, Vg])
                y_all = np.concatenate([yw, yg])
                weights = np.concatenate([
                    np.ones_like(yw),
                    self.guard_weight * np.ones_like(yg),
                ])
            else:
                V = Vw
                y_all = yw
                weights = np.ones_like(yw)

            # Apply weights via row scaling
            Wsqrt = np.sqrt(weights)[:, None]
            Vwtd = V * Wsqrt
            ywtd = y_all * np.sqrt(weights)

            try:
                cz, *_ = np.linalg.lstsq(Vwtd, ywtd, rcond=None)
            except np.linalg.LinAlgError as e:
                print(f"  LS failed (deg={deg}): {e}")
                continue

            # Compose cz in z back to x coefficients px (ascending)
            z_poly = np.array([b, a], dtype=float)
            px = np.array([0.0], dtype=float)
            for k, ck in enumerate(cz):
                if abs(ck) < 1e-18:
                    continue
                zk = P.polypow(z_poly, k)
                px = P.polyadd(px, zk * ck)

            # Evaluate R^2 on the real window points only
            y_pred = P.polyval(xw, px)
            y_mean = float(np.mean(yw))
            ss_res = float(np.sum((yw - y_pred) ** 2))
            ss_tot = float(np.sum((yw - y_mean) ** 2)) if len(yw) > 1 else 0.0
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

            func = self._coeffs_to_string(px[::-1])  # high->low for string
            if func is None:
                continue
            cand = {"func": func, "r2": r2, "deg": deg}
            if (best is None) or (cand["r2"] > best["r2"]) or (
                abs(cand["r2"] - best["r2"]) < 1e-6 and cand["deg"] < best["deg"]
            ):
                best = cand

        if best and best["r2"] >= self.r2_threshold:
            return best
        return best  # might be below threshold; caller may still accept
