import json
from matplotlib import colormaps


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
  points.forEach((pt,i)=>{{
    vertices[i*3] = ((pt[0]-minX)/rangeX-0.5)*4;
    vertices[i*3+1] = ((pt[1]-minY)/rangeY-0.5)*4;
    vertices[i*3+2] = 0;
    const t=i/points.length;
    const r=primary.r*(1-t)+accent.r*t;
    const g=primary.g*(1-t)+accent.g*t;
    const b=primary.b*(1-t)+accent.b*t;
    pointColors[i*3]=r;
    pointColors[i*3+1]=g;
    pointColors[i*3+2]=b;
  }});
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


def generate_html(points, theme_colors):
    data = HTML_TEMPLATE.format(
        points=json.dumps(points),
        bg=theme_colors.get('background', '#000'),
        primary=theme_colors.get('primary', '#ff6b6b'),
        accent=theme_colors.get('accent', '#4ecdc4'),
    )
    return data


__all__ = ["validate_colormap", "generate_html"]
