# Shape & Contour Analyzer

An interactive Streamlit dashboard for analyzing geometric shapes in images using computer vision techniques.

## Features

- **Shape Detection**: Automatically detects triangles, squares, rectangles, pentagons, hexagons, and circles
- **Object Counting**: Counts the number of each shape type
- **Feature Extraction**: Displays area and perimeter for each detected object
- **Contour Visualization**: Shows contours drawn on the original image

## Learning Outcomes

- Understanding of image contours and their detection
- Feature extraction techniques (area, perimeter, shape approximation)
- Computer vision basics with OpenCV

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run main.py
```

Upload an image (JPG, PNG, JPEG) and the app will analyze it for geometric shapes.

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Pillow