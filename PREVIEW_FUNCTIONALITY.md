# Centerline Preview Functionality

## Overview
The preview functionality allows you to visualize how the zero-width skeleton extraction works. You can see both the original letter outlines and the extracted centerline points to verify that the algorithm is correctly finding the midpoints of letter strokes.

## Preview Methods Added to TextExtractor

### 1. `preview_extracted_points(text, font_size=100, num_points=500, save_path=None)`
**Basic Preview**: Shows the original outline and extracted centerline points.

- **Features**:
  - Original letter outline in light gray
  - Centerline points as connected red dots
  - Start point marked in green, end point in blue
  - Separate subplot for each character

- **Usage**:
```python
from src.text_extractor import TextExtractor

extractor = TextExtractor()

# Preview letter 'A' with 500 centerline points
extractor.preview_extracted_points('A', num_points=500)

# Save the preview to a file
extractor.preview_extracted_points('A', save_path='letter_A_preview.png')
```

### 2. `preview_skeleton_extraction_steps(text, font_size=100, save_path=None)`
**Detailed Process Preview**: Shows the skeleton extraction process step-by-step.

- **Features**:
  - Top panel: Original outline + sampling grid
  - Bottom panel: Extracted skeleton with midpoint examples
  - Visual demonstration of how midpoints are calculated
  - Shows intersection points and their corresponding midpoints

- **Usage**:
```python
# See the detailed skeleton extraction process
extractor.preview_skeleton_extraction_steps('A')

# Save detailed preview
extractor.preview_skeleton_extraction_steps('A', save_path='skeleton_process.png')
```

### 3. Helper Methods
- `plot_path_outline()`: Plots the original letter outline
- `plot_intersection_grid()`: Shows the sampling grid used for finding intersections  
- `plot_midpoint_examples()`: Demonstrates midpoint calculations with visual examples

## Standalone Preview Script

### `preview_centerlines.py`
A complete script for generating centerline previews without running the full system.

**Usage**:
```bash
# Preview letter 'A' with default settings
python preview_centerlines.py

# Preview custom text with specific settings
python preview_centerlines.py "Hello" 120 300

# Arguments: [text] [font_size] [num_points]
```

**Features**:
- Basic centerline preview
- Detailed skeleton extraction process
- Point count comparison (50, 200, 500 points)
- Automatic image saving
- Error handling for missing dependencies

## What You'll See in the Previews

### Basic Preview
- **Light gray outline**: Original letter shape with thickness
- **Red connected dots**: Extracted centerline points (zero thickness)
- **Green dot**: Start of centerline trace
- **Blue dot**: End of centerline trace

### Detailed Skeleton Preview
- **Top panel**: 
  - Black outline of original letter
  - Dashed grid lines showing sampling pattern
  - Blue vertical lines, red horizontal lines
  
- **Bottom panel**:
  - Faded gray outline for reference
  - Red skeleton points showing true centerline
  - Green example lines showing midpoint calculations
  - Green squares marking boundary intersections

### Point Count Comparison
- Side-by-side comparison of different point densities
- Shows how more points capture finer details
- Helps optimize the balance between accuracy and performance

## Verification Points

Use the preview to verify:

1. **Zero Thickness**: Centerline runs through middle of strokes, not along edges
2. **Connectivity**: Points form a connected path tracing letter shape
3. **Coverage**: All major letter features are captured
4. **Start/End**: Points begin and end at logical locations
5. **Smoothness**: Sufficient point density for smooth curves

## Example Output Description

For letter 'A':
- Centerline should run up the left stroke, across the crossbar, and down the right stroke
- No thickness artifacts from original font outline
- Points should be densely packed enough to capture the sharp apex
- Crossbar should be clearly defined as horizontal centerline segment

## Files Generated
- `preview_basic_[text].png`: Basic centerline preview
- `preview_skeleton_[text].png`: Detailed skeleton extraction process  
- `comparison_[text].png`: Point count comparison

This preview functionality helps validate that the zero-width skeleton extraction is working correctly and produces true centerlines rather than thick outlines.
