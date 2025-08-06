# Text to Desmos Polynomial Plotter

A Python program that converts text into polynomial approximations that can be plotted on the Desmos graphing calculator. The program extracts character outlines from fonts and fits polynomial curves to approximate the text shapes.

## Features

- **Text to Polynomial Conversion**: Converts any text into polynomial functions
- **Customizable Parameters**: Adjustable origin point, scale factor, and polynomial complexity
- **Desmos Compatible Output**: Generates functions that work directly in Desmos
- **Multiple Character Support**: Handles letters, numbers, and symbols
- **Automatic Font Detection**: Uses system fonts automatically
- **Domain Constraints**: Functions include domain restrictions for proper display

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy >= 1.20.0`
- `matplotlib >= 3.3.0`

## Quick Start

### Method 1: Interactive Mode
```bash
python3 main.py
```
Follow the prompts to enter your text, origin point, and scale factor.

### Method 2: Programmatic Usage
```python
from main import TextToDesmos

# Create converter
converter = TextToDesmos(origin=(0, 0), scale=1.0)

# Generate functions
functions = converter.text_to_desmos_functions(
    text="HELLO",
    font_size=60,
    points_per_char=30,
    max_degree=4
)

# Save to file
converter.save_functions(functions, "hello_functions.txt")
```

### Method 3: Run Examples
```bash
python3 example_usage.py
```
This creates several example files with different configurations.

## Using the Output in Desmos

1. **Generate Functions**: Run the program to create a `.txt` file with polynomial functions
2. **Open Desmos**: Go to [desmos.com/calculator](https://www.desmos.com/calculator)
3. **Copy Functions**: Open the generated `.txt` file and copy the function lines (starting with `y =` or `x =`)
4. **Paste in Desmos**: Paste each function into a separate expression in Desmos
5. **Adjust View**: Zoom and pan to see your text outline

## Parameters

### TextToDesmos Constructor
- `origin` (tuple): Origin point (x, y) for positioning the text. Default: `(0, 0)`
- `scale` (float): Scale factor for the text size. Default: `1.0`

### text_to_desmos_functions Method
- `text` (str): Input text to convert
- `font_size` (int): Font size for rendering. Default: `100`
- `points_per_char` (int): Number of points to extract per character. Default: `50`
- `max_degree` (int): Maximum polynomial degree. Default: `6`

## Algorithm Overview

1. **Text to Paths**: Convert text characters to vector paths using matplotlib's TextPath
2. **Contour Extraction**: Extract discrete points along character outlines
3. **Segmentation**: Split contours into manageable segments
4. **Polynomial Fitting**: Fit polynomial curves to each segment using numpy.polyfit
5. **Domain Constraints**: Add domain restrictions to ensure proper display
6. **Transformation**: Apply origin translation and scaling transformations

## Example Outputs

The program generates functions like:
```
y = (-0.001234*x^3 + 0.123456*x^2 - 1.234567*x + 12.345678) * {x >= 10.000 AND x <= 25.000}
x = (0.002345*y^2 - 0.234567*y + 23.456789) * {y >= -5.000 AND y <= 10.000}
```

These functions combine to form the complete text outline when plotted in Desmos.

## Tips for Better Results

1. **Start Simple**: Begin with short text (1-3 characters)
2. **Adjust Complexity**: Use lower `max_degree` (3-4) for simpler, smoother curves
3. **Optimize Points**: Fewer `points_per_char` (20-40) can reduce function count
4. **Scale Appropriately**: Use `scale` parameter to make text fit nicely in Desmos
5. **Position Strategically**: Use `origin` to place text where you want it

## File Structure

```
├── main.py                 # Main program with TextToDesmos class
├── example_usage.py        # Example usage demonstrations
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── generated files:
    ├── desmos_functions.txt    # Default output from interactive mode
    ├── example1_hi.txt         # Example outputs
    ├── example2_ok_scaled.txt
    ├── example3_letter_A_detailed.txt
    ├── example4_math_optimized.txt
    └── DESMOS_INSTRUCTIONS.md  # Detailed Desmos usage instructions
```

## Troubleshooting

### Functions Don't Display in Desmos
- Try reducing `max_degree` to 3 or 4
- Decrease `points_per_char` to reduce function complexity
- Check that you copied the complete function including domain constraints

### Text Appears Distorted
- Adjust the `scale` parameter
- Try different `font_size` values
- Increase `points_per_char` for more detail

### Too Many Functions Generated
- Use shorter text
- Reduce `points_per_char`
- Increase segment length (modify `segment_length` parameter in the code)

### Program Crashes
- Ensure numpy and matplotlib are installed
- Try with simpler characters (letters/numbers only)
- Check that your system has fonts available

## Limitations

- Complex characters may generate many functions
- Very high polynomial degrees may cause numerical instability
- Some font styles may not convert well
- Desmos has limits on the number of expressions

## License

This project is open source. Feel free to modify and distribute.

## Contributing

Contributions are welcome! Areas for improvement:
- Better polynomial fitting algorithms
- Support for more complex curves (splines, Bezier)
- Optimization to reduce function count
- Better handling of complex characters
- GUI interface
