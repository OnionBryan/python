import json
import numpy as np
from numpy.typing import ArrayLike
from matplotlib import colormaps, cm


def validate_colormap(name: str) -> str:
    """Return a valid matplotlib colormap name."""
    if name in colormaps:
        return name
    # common aliases used by user scripts
    aliases = {
        'earth': 'YlOrBr',
        'electric': 'plasma',
    }
    return aliases.get(name, 'plasma')


def apply_matrix_transform(matrix: ArrayLike, vectors: ArrayLike) -> np.ndarray:
    """Multiply a matrix by one or more vectors using numpy."""
    m = np.asarray(matrix, dtype=float)
    v = np.asarray(vectors, dtype=float)
    if v.ndim == 1:
        return m @ v
    return v @ m.T


def prepare_point_matrix(points: ArrayLike) -> np.ndarray:
    """Normalize 2D point data and return an Nx3 vertex matrix."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.zeros((1, 3))
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("points must be Nx2 or NxM array")

    mins = arr[:, :2].min(axis=0)
    maxs = arr[:, :2].max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm = (arr[:, :2] - mins) / ranges - 0.5
    scaled = norm @ np.array([[4.0, 0.0], [0.0, 4.0]])
    zeros = np.zeros((scaled.shape[0], 1))
    return np.hstack((scaled, zeros))


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>Fractal Viewer</title>
<style>
 body {{ margin: 0; background: {bg}; overflow: hidden; }}
 canvas {{ display: block; }}
</style>
</head>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const points = {points};
const colors = {colors};
function init() {{
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('{bg}');
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight,1,1000);
  camera.position.z = 5;
  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);
  const geometry = new THREE.BufferGeometry();
  const vertices = new Float32Array(points.length * 3);
  const pointColors = new Float32Array(points.length * 3);
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  points.forEach(p=>{{minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);}});
  const rangeX = maxX-minX || 1;
  const rangeY = maxY-minY || 1;
  const primary = new THREE.Color('{primary}');
  const accent = new THREE.Color('{accent}');
  const useCustomColors = Array.isArray(colors) && colors.length === points.length;
  points.forEach((pt,i)=>{
    vertices[i*3] = ((pt[0]-minX)/rangeX-0.5)*4;
    vertices[i*3+1] = ((pt[1]-minY)/rangeY-0.5)*4;
    vertices[i*3+2] = 0;
    if (useCustomColors) {
      pointColors[i*3] = colors[i][0];
      pointColors[i*3+1] = colors[i][1];
      pointColors[i*3+2] = colors[i][2];
    } else {
      const t=i/points.length;
      const r=primary.r*(1-t)+accent.r*t;
      const g=primary.g*(1-t)+accent.g*t;
      const b=primary.b*(1-t)+accent.b*t;
      pointColors[i*3]=r;
      pointColors[i*3+1]=g;
      pointColors[i*3+2]=b;
    }
  });
  geometry.setAttribute('position',new THREE.BufferAttribute(vertices,3));
  geometry.setAttribute('color',new THREE.BufferAttribute(pointColors,3));
  const material = new THREE.PointsMaterial({{size:0.05,vertexColors:true}});
  const mesh = new THREE.Points(geometry,material);
  scene.add(mesh);
  renderer.setAnimationLoop(()=>{{renderer.render(scene,camera);}});
}}
window.addEventListener('load',init);
</script>
</body>
</html>
"""


def generate_html(points: ArrayLike, theme_colors: dict, colormap: str = 'plasma') -> str:
    """Return an HTML string visualizing the points with Three.js."""
    vertices = prepare_point_matrix(points)
    cmap = cm.get_cmap(validate_colormap(colormap))
    n = len(vertices)
    t = np.linspace(0, 1, n)
    color_vals = cmap(t)[:, :3]

    data = HTML_TEMPLATE.format(
        points=json.dumps(vertices.tolist()),
        colors=json.dumps(color_vals.tolist()),
        bg=theme_colors.get('background', '#000'),
        primary=theme_colors.get('primary', '#ff6b6b'),
        accent=theme_colors.get('accent', '#4ecdc4'),
    )
    return data


__all__ = [
    "validate_colormap",
    "apply_matrix_transform",
    "prepare_point_matrix",
    "generate_html",
]
