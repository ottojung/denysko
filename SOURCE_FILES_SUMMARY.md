# Source Files in src/ Directory

## File Overview

### 1. `__init__.py` (12 lines)
```python
"""
Text to Desmos source package
"""

from .text_extractor import TextExtractor
from .polynomial_fitter import PolynomialFitter
from .function_transformer import FunctionTransformer
from .text_to_desmos import TextToDesmos
from .main import main

__all__ = ['TextExtractor', 'PolynomialFitter', 'FunctionTransformer', 'TextToDesmos', 'main']
```
**Purpose**: Package initialization with all module imports

### 2. `main.py` (52 lines)
**Purpose**: Main entry point for testing the system
**Key Features**:
- Tests letter 'A' conversion
- Verifies only y = f(x) functions are generated
- Saves results to `letter_A_functions.txt`
- Uses relative imports (`.text_to_desmos`)

### 3. `text_extractor.py` (890+ lines)
**Purpose**: Font path extraction and zero-width centerline extraction
**Key Features**:
- **Skeleton-Based Extraction**: `extract_skeleton_from_path()` - finds true centerlines using midpoint analysis
- **Zero Stroke Width**: Converts thick font outlines to skeletal centerlines
- **High Resolution**: Extracts 500 centerline points per letter
- **Preview Functionality**: 
  - `preview_extracted_points()` - basic centerline preview
  - `preview_skeleton_extraction_steps()` - detailed process visualization
  - `plot_path_outline()`, `plot_intersection_grid()`, `plot_midpoint_examples()`
- **Midpoint Algorithm**: Uses 50x50 grid to find intersections and calculate midpoints
- **Fallback Systems**: Simplified outline extraction if skeleton fails

**Current Status**: ✅ Complete with skeleton extraction and preview functionality

### 4. `polynomial_fitter.py` (459 lines) 
**Purpose**: Fits y=f(x) polynomials to centerline points with shape accuracy
**Key Features**:
- **Shape-Accuracy Strategy**: Prioritizes accuracy at letter centerline points
- **Weighted Least-Squares**: 1000x weight on actual letter points
- **Clean Output**: Generates fewer curves (5+) for reduced visual clutter
- **No Domain Restrictions**: Polynomials defined over entire real line as requested
- **Adaptive Degree**: Automatically adjusts polynomial degree based on error
- **Methods**:
  - `fit_contour_polynomials()` - main fitting strategy
  - `fit_polynomial_for_shape_accuracy()` - weighted fitting
  - `split_contour_into_x_monotonic_segments()` - ensures y=f(x) compatibility

**Current Status**: ✅ Complete with shape-accuracy focus and no domain restrictions

### 5. `function_transformer.py` (109 lines)
**Purpose**: Coordinate transformations and function string cleanup
**Key Features**:
- **Y=f(x) Only**: Only processes y = f(x) functions, skips others
- **Coordinate Transforms**: Applies origin offset and scaling
- **String Cleanup**: Simplifies polynomial expressions
- **Safety Checks**: Warns about any non-y functions (shouldn't exist)

**Current Status**: ✅ Complete

### 6. `text_to_desmos.py` (112 lines)
**Purpose**: Main coordinator class integrating all modules
**Key Features**:
- **Pipeline Coordination**: text → paths → centerlines → polynomials → functions
- **Y=f(x) Enforcement**: Multiple verification steps ensure only y=f(x) output
- **High-Resolution Processing**: Uses 500 centerline points per character
- **File Output**: `save_functions()` method for saving results
- **Filtering**: Removes any non-y functions at multiple pipeline stages

**Current Status**: ✅ Complete

## Architecture Summary

### Data Flow:
1. **text_extractor.py**: Text → Font paths → Skeleton centerlines (500 points)
2. **polynomial_fitter.py**: Centerlines → Shape-accurate polynomials (5+ curves)
3. **function_transformer.py**: Polynomials → Coordinate transforms → Cleanup
4. **text_to_desmos.py**: Coordinates entire pipeline
5. **main.py**: Entry point for testing

### Key Improvements Implemented:
- ✅ **Zero-Width Skeleton**: True centerline extraction using midpoint analysis
- ✅ **Shape Accuracy**: Weighted fitting prioritizing letter centerline points
- ✅ **No Domain Restrictions**: Polynomials over entire real line
- ✅ **Preview System**: Visual verification of centerline extraction
- ✅ **Reduced Clutter**: Fewer curves for cleaner output
- ✅ **Y=f(x) Only**: Strict enforcement throughout pipeline

### Current Architecture Status:
- **Modular Design**: ✅ Clean separation of concerns
- **Zero-Width Letters**: ✅ Skeleton-based centerline extraction
- **High-Resolution Input**: ✅ 500 points per letter
- **Shape-Accurate Fitting**: ✅ Weighted least-squares at centerline points
- **Clean Output**: ✅ Fewer polynomials, no domain restrictions
- **Preview System**: ✅ Visual verification tools
- **Y=f(x) Enforcement**: ✅ Multiple validation stages

All source files are complete and implement the requested zero-width skeleton extraction with shape-accurate polynomial fitting and preview functionality.
