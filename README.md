# python
work in python

This repository contains assorted scripts. The `fractal_webgl.py` module provides
helpers for generating a WebGL fractal viewer. It includes a utility function
`validate_colormap` that maps a variety of color scheme names to valid
Matplotlib color maps and an HTML template that shades points using theme
colors. The `generate_html` helper now accepts a `colormap` argument and
precomputes a full color matrix to avoid washed‑out gradients.

Additional helpers:

- `apply_matrix_transform(matrix, vectors)`: multiply a transformation matrix by one or more vectors using NumPy.
- `prepare_point_matrix(points)`: normalize points into a vertex matrix for the viewer.
