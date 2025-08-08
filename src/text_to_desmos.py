#!/usr/bin/env python3
"""
TextToDesmos: minimal end-to-end pipeline (from text to Desmos-ready functions).
- Only produces y = f(x) functions.
- Uses TextExtractor to get centerline points.
- Uses PolynomialFitter to fit piecewise polynomials.
- Uses FunctionTransformer to apply origin/scale and simplify strings.
"""

from typing import List, Tuple, Optional

from .text_extractor import TextExtractor
from .polynomial_fitter import PolynomialFitter
from .function_transformer import FunctionTransformer


class TextToDesmos:
    """Convert text to Desmos polynomial functions (y = f(x) only)."""

    def __init__(
        self,
        origin: Tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        extractor: Optional[TextExtractor] = None,
        fitter: Optional[PolynomialFitter] = None,
        transformer: Optional[FunctionTransformer] = None,
    ) -> None:
        self.extractor = extractor or TextExtractor()
        self.fitter = fitter or PolynomialFitter()
        self.transformer = transformer or FunctionTransformer(origin=origin, scale=scale)

    def text_to_desmos_functions(
        self,
        text: str,
        font_size: int = 100,
        points_per_char: int = 500,
    ) -> List[str]:
        """Generate Desmos-compatible y = f(x) functions for the given text.

        Args:
            text: Input text to render.
            font_size: Font size used to build glyph outlines.
            points_per_char: Number of centerline points per character.
        Returns:
            List of function strings, each in the form 'y = ...'.
        """
        if not text:
            return []

        # 1) Outline paths per character
        paths = self.extractor.text_to_paths(text, font_size)
        if not paths:
            return []

        # 2) Centerline points per path
        contours_all: List[List[Tuple[float, float]]] = []
        for path in paths:
            contours = self.extractor.extract_contour_points(path, points_per_char)
            # filter out empties
            contours = [c for c in contours if hasattr(c, "__len__") and len(c) > 1]
            if contours:
                # current extractor returns a single centerline per glyph
                contours_all.extend(contours)

        if not contours_all:
            return []

        # 3) Fit polynomials (y=f(x) only)
        y_functions: List[str] = []
        for contour in contours_all:
            funcs = self.fitter.fit_contour_polynomials(contour)
            # keep only y = ...
            for f in funcs:
                if isinstance(f, str) and f.startswith("y ="):
                    y_functions.append(f)

        if not y_functions:
            return []

        # 4) Transform origin/scale and simplify
        transformed = self.transformer.transform_functions(y_functions)
        simplified = [self.transformer.simplify_function_string(f) for f in transformed]

        # 5) Ensure final validation (y = ... only)
        return [f for f in simplified if isinstance(f, str) and f.startswith("y =")] 

    def save_functions(self, functions: List[str], filename: str = "desmos_functions.txt") -> None:
        """Save function strings to a text file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Desmos Functions (y = f(x) only)\n\n")
            for i, func in enumerate(functions, 1):
                f.write(f"# Function {i}\n{func}\n\n")
