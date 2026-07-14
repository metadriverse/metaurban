"""Visualize and compare asset annotations before/after reannotation.

For every GLB with an annotation in both --before and --after, renders the
asset offscreen under each annotation — with the physics chassis box, ground
plane at z=0, and origin axes drawn in — and writes a single self-contained
``report.html`` with the images side by side plus per-field numeric deltas.
A well-canonicalized asset shows the wireframe box hugging the model, the
base resting on the grid, and the axes at the footprint center.

Usage:
    # 1) reannotate into a fresh dir, keeping the originals for comparison
    python scripts/reannotate_assets.py --out /tmp/adj_new
    # 2) compare
    python scripts/visualize_reannotation.py --before metaurban/assets/adj_parameter_folder \\
        --after /tmp/adj_new --out reannotation_report

    # no assets downloaded? synthetic end-to-end demo:
    python scripts/visualize_reannotation.py --demo
"""
import argparse
import base64
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reannotate_assets import REPO_ROOT, _get_base, load_annotation_index, reannotate

IMG_W, IMG_H = 640, 480
FIELDS = ("pos0", "pos1", "pos2", "length", "width", "height", "hshift", "scale")


def _get_render_base():
    """ShowBase with an offscreen buffer; falls back to the software renderer."""
    from panda3d.core import loadPrcFileData
    # Panda latches display modules when the first pipe is created, so the
    # software fallback must be registered before the first ShowBase.
    loadPrcFileData("", f"win-size {IMG_W} {IMG_H}\naux-display p3tinydisplay")
    base = _get_base("offscreen")
    if base.win is None:
        raise RuntimeError(
            "ShowBase was already created without a render buffer; "
            "call _get_render_base() before any other loader use in this process"
        )
    base.disableMouse()
    base.setBackgroundColor(0.94, 0.94, 0.97, 1)
    try:
        # PBR pipeline so textured GLB materials render shaded (the sim's vendored
        # simplepbr needs engine PSSM shader inputs, so use the standalone package);
        # skipped on the software fallback, which cannot run shaders
        import simplepbr
        simplepbr.init(msaa_samples=4)
    except Exception as e:
        print(f"[warn] simplepbr unavailable ({e}); rendering with the fixed-function pipeline")
    return base


def _flat_meta(meta):
    """Annotation dict -> {field: value}, resolving the general/top-level split."""
    g = meta.get("general", meta)
    return {
        "pos0": meta.get("pos0", 0.0),
        "pos1": meta.get("pos1", 0.0),
        "pos2": meta.get("pos2", 0.0),
        "length": g.get("length", meta.get("length", 0.0)),
        "width": g.get("width", meta.get("width", 0.0)),
        "height": meta.get("height", 0.0),
        "hshift": meta.get("hshift", 0.0),
        "scale": meta.get("scale", 1.0),
    }


def _lines(parent, segs, color, thickness=1.5):
    from panda3d.core import LineSegs
    ls = LineSegs()
    ls.setThickness(thickness)
    ls.setColor(*color)
    for a, b in segs:
        ls.moveTo(*a)
        ls.drawTo(*b)
    np = parent.attachNewNode(ls.create())
    # keep overlay lines out of the PBR pipeline so they render their flat colors
    np.setShaderOff(1)
    np.setLightOff(1)


def _draw_chassis_box(parent, l, w, h):
    """Wireframe of the Bullet chassis box: (l, w, h) centered at (0, 0, h/2)."""
    x, y = l / 2.0, w / 2.0
    corners = [(sx * x, sy * y, z) for z in (0.0, h) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    edges += [(corners[4 + i], corners[4 + (i + 1) % 4]) for i in range(4)]
    edges += [(corners[i], corners[4 + i]) for i in range(4)]
    _lines(parent, edges, (1.0, 0.35, 0.1, 1), 2.5)


def _draw_grid_and_axes(parent, extent):
    n = max(2, int(math.ceil(extent)))
    grid = []
    for i in range(-n, n + 1):
        grid.append(((i, -n, 0), (i, n, 0)))
        grid.append(((-n, i, 0), (n, i, 0)))
    _lines(parent, grid, (0.55, 0.55, 0.6, 1), 1)
    axis_len = max(1.0, extent / 3.0)
    _lines(parent, [((0, 0, 0), (axis_len, 0, 0))], (0.9, 0.1, 0.1, 1), 3)
    _lines(parent, [((0, 0, 0), (0, axis_len, 0))], (0.1, 0.7, 0.1, 1), 3)
    _lines(parent, [((0, 0, 0), (0, 0, axis_len))], (0.15, 0.25, 0.95, 1), 3)


def _build_scene(base, model_path, flat):
    """Asset under its annotation transform + chassis box, in the annotation frame.

    Note: at spawn TestObject additionally lifts the model by half the sidewalk
    thickness (PGDrivableAreaProperty.SIDEWALK_THICKNESS / 2 = 0.15 m) relative
    to the chassis; the canonical annotation frame shown here is pre-lift.
    """
    root = base.render.attachNewNode("scene")
    model = base.loader.loadModel(model_path, noCache=True)
    model.reparentTo(root)
    model.setH(flat["hshift"])
    model.setPos(flat["pos0"], flat["pos1"], flat["pos2"])
    model.setScale(flat["scale"])
    _draw_chassis_box(root, flat["length"], flat["width"], flat["height"])
    bounds = model.getTightBounds(root)
    if bounds is None:
        lo = hi = type("P", (), {"x": 0, "y": 0, "z": 0})()
    else:
        lo, hi = bounds
    ext = max(abs(v) for v in (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
                               flat["length"] / 2, flat["width"] / 2, flat["height"])) or 1.0
    return root, ext


def _add_lights(base):
    from panda3d.core import AmbientLight, DirectionalLight
    amb = AmbientLight("amb")
    amb.setColor((0.55, 0.55, 0.55, 1))
    base.render.setLight(base.render.attachNewNode(amb))
    sun = DirectionalLight("sun")
    sun.setColor((0.8, 0.8, 0.75, 1))
    sun_np = base.render.attachNewNode(sun)
    sun_np.setHpr(-40, -35, 0)
    base.render.setLight(sun_np)


def _shoot(base, out_png, cam_dist, cam_focus_z):
    from panda3d.core import Filename
    base.camera.setPos(cam_dist * 0.75, -cam_dist * 0.75, cam_dist * 0.55 + cam_focus_z)
    base.camera.lookAt(0, 0, cam_focus_z)
    # step the task manager (not bare renderFrame) so simplepbr's per-frame
    # shader-input updates run
    base.taskMgr.step()
    base.taskMgr.step()
    base.win.saveScreenshot(Filename.fromOsSpecific(out_png))


def render_pair(base, model_path, before_flat, after_flat, out_dir, stem):
    """Render before/after with an identical camera; returns the two PNG paths."""
    paths = []
    scenes = []
    ext = 1.0
    for flat in (before_flat, after_flat):
        root, e = _build_scene(base, model_path, flat)
        root.hide()
        scenes.append(root)
        ext = max(ext, e)
    cam_dist = ext * 2.6 + 1.0
    focus_z = max(before_flat["height"], after_flat["height"]) / 3.0
    for root, tag in zip(scenes, ("before", "after")):
        _draw_grid_and_axes(root, ext + 0.5)
        png = os.path.join(out_dir, f"{stem}_{tag}.png")
        root.show()
        _shoot(base, png, cam_dist, focus_z)
        root.removeNode()
        paths.append(png)
    return paths


# --- report -------------------------------------------------------------------


def _img_tag(path):
    # downscale to display size before embedding: at catalog scale (hundreds of
    # assets) full-res base64 PNGs would make the report hundreds of MB
    import io
    from PIL import Image
    img = Image.open(path)
    img.thumbnail((IMG_W // 2, IMG_H // 2))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" width="{IMG_W // 2}">'


def _asset_section(name, before_flat, after_flat, before_png, after_png):
    rows = []
    for k in FIELDS:
        b, a = before_flat[k], after_flat[k]
        delta = a - b
        cls = ' class="changed"' if abs(delta) > 1e-6 else ""
        rows.append(f"<tr{cls}><td>{k}</td><td>{b:.4f}</td><td>{a:.4f}</td><td>{delta:+.4f}</td></tr>")
    return f"""
<section>
  <h2>{name}</h2>
  <div class="pair">
    <figure>{_img_tag(before_png)}<figcaption>before</figcaption></figure>
    <figure>{_img_tag(after_png)}<figcaption>after</figcaption></figure>
  </div>
  <table>
    <tr><th>field</th><th>before</th><th>after</th><th>&Delta;</th></tr>
    {''.join(rows)}
  </table>
</section>"""


def write_report(sections, out_html):
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Reannotation comparison</title><style>
  body {{ font-family: system-ui, sans-serif; margin: 2em auto; max-width: 900px; color: #222; }}
  .pair {{ display: flex; gap: 12px; }}
  figure {{ margin: 0; }}
  figcaption {{ text-align: center; color: #666; font-size: 0.9em; }}
  img {{ border: 1px solid #ccc; border-radius: 4px; }}
  table {{ border-collapse: collapse; margin-top: 8px; font-size: 0.9em; }}
  td, th {{ border: 1px solid #ddd; padding: 3px 10px; text-align: right; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr.changed td {{ background: #fff3e0; }}
  section {{ margin-bottom: 3em; }}
</style></head><body>
<h1>Asset reannotation: before vs after</h1>
<p>Orange wireframe = physics chassis box; grid = ground plane (z=0); red/green/blue = X/Y/Z axes.
A canonical asset is centered on X/Y with its base on the grid and the box hugging the model.
Images show the annotation frame; at spawn the sim additionally lifts the visual model by half
the sidewalk thickness (0.15&thinsp;m) relative to the chassis box.</p>
{''.join(sections)}
</body></html>"""
    with open(out_html, "w") as f:
        f.write(html)


def compare(models_dir, before_dir, after_dir, out_dir):
    out_real = os.path.realpath(out_dir)
    for d in (before_dir, after_dir):
        d_real = os.path.realpath(d)
        if os.path.commonpath([out_real, d_real]) == d_real:
            raise SystemExit(f"--out must not be inside an annotation dir ({d}): the sim's "
                             "annotation loader json-parses every file it finds there")
    base = _get_render_base()
    _add_lights(base)
    os.makedirs(out_dir, exist_ok=True)
    before_idx = load_annotation_index(before_dir)
    after_idx = load_annotation_index(after_dir)

    sections = []
    for fname in sorted(set(before_idx) & set(after_idx)):
        model_path = os.path.join(models_dir, fname)
        if not os.path.isfile(model_path):
            print(f"[skip] {fname}: GLB not found in {models_dir}")
            continue
        before_flat = _flat_meta(before_idx[fname][1])
        after_flat = _flat_meta(after_idx[fname][1])
        stem = os.path.splitext(fname)[0]
        before_png, after_png = render_pair(base, model_path, before_flat, after_flat, out_dir, stem)
        sections.append(_asset_section(fname, before_flat, after_flat, before_png, after_png))
        print(f"[ok] {fname}")

    if not sections:
        print("nothing to compare (need the same GLB annotated in both --before and --after)")
        return None
    out_html = os.path.join(out_dir, "report.html")
    write_report(sections, out_html)
    print(f"report: {out_html}")
    return out_html


# --- demo ---------------------------------------------------------------------


def _write_box_glb(path, size, offset, color):
    """Hand-built GLB: a lit, colored box of `size` at `offset` (glTF Y-up)."""
    sx, sy, sz = size
    ox, oy, oz = offset
    faces = [  # (normal, 4 corners CCW seen from outside)
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
        ((0, 0, -1), [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)]),
        ((1, 0, 0), [(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]),
        ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        ((0, 1, 0), [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)]),
        ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ]
    positions, normals, indices = [], [], []
    for norm, quad in faces:
        i0 = len(positions)
        for u, v, w in quad:
            positions.append((ox + u * sx, oy + v * sy, oz + w * sz))
            normals.append(norm)
        indices += [i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]
    pos_bin = b"".join(struct.pack("<fff", *p) for p in positions)
    nrm_bin = b"".join(struct.pack("<fff", *n) for n in normals)
    idx_bin = b"".join(struct.pack("<H", i) for i in indices)
    idx_bin += b"\x00" * (-len(idx_bin) % 4)
    blob = pos_bin + nrm_bin + idx_bin
    mins = [min(p[i] for p in positions) for i in range(3)]
    maxs = [max(p[i] for p in positions) for i in range(3)]
    gltf_json = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorFactor": list(color) + [1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                    "indices": 2, "material": 0}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bin), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bin), "byteLength": len(nrm_bin), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bin) + len(nrm_bin), "byteLength": len(idx_bin), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": mins, "max": maxs},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
    }
    json_bin = json.dumps(gltf_json).encode()
    json_bin += b" " * (-len(json_bin) % 4)
    total = 12 + 8 + len(json_bin) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bin), 0x4E4F534A) + json_bin)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


DEMO_ASSETS = [
    # name, glTF size (x, y-up, z), glTF offset, color, junk "before" annotation
    ("bench-aaaa1111.glb", (2.0, 0.9, 0.8), (1.3, 0.0, 0.6), (0.72, 0.53, 0.34),
     {"hshift": 0.0, "scale": 1.0}),
    ("hydrant-bbbb2222.glb", (0.4, 1.1, 0.4), (-0.6, -0.2, 0.3), (0.85, 0.15, 0.12),
     {"hshift": 30.0, "scale": 1.0}),
    ("kiosk-cccc3333.glb", (2.2, 4.5, 2.2), (0.8, 0.0, -1.1), (0.25, 0.45, 0.75),
     {"hshift": 0.0, "scale": 0.5}),
]


def demo(out_dir):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="reannotate_viz_")
    models = os.path.join(tmp, "models")
    adj_old = os.path.join(tmp, "adj_old")
    adj_new = os.path.join(tmp, "adj_new")
    os.makedirs(models)
    os.makedirs(adj_old)

    for fname, size, offset, color, before in DEMO_ASSETS:
        _write_box_glb(os.path.join(models, fname), size, offset, color)
        meta = {
            "CLASS_NAME": fname.split("-")[0],
            "filename": fname,
            "hshift": before["hshift"],
            "scale": before["scale"],
            "pos0": 0.0, "pos1": 0.0, "pos2": 0.0,
            "height": 1.0,
            "general": {"detail_type": fname.split("-")[0], "length": 1.0, "width": 1.0},
        }
        with open(os.path.join(adj_old, fname.replace(".glb", ".json")), "w") as f:
            json.dump(meta, f)

    # rendering buffer must exist before reannotate() grabs the shared ShowBase
    _get_render_base()
    assert reannotate(models, adj_old, adj_new) == len(DEMO_ASSETS)
    out_html = compare(models, adj_old, adj_new, out_dir)
    assert out_html is not None

    from PIL import Image
    for fname, *_ in DEMO_ASSETS:
        png = os.path.join(out_dir, fname.replace(".glb", "_after.png"))
        img = Image.open(png).convert("RGB")
        assert len(img.getcolors(maxcolors=4096) or [0] * 100) > 4, f"{png} looks blank"
    print("demo passed")
    return out_html


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=os.path.join(REPO_ROOT, "metaurban/assets/models/test"))
    p.add_argument("--before", help="dir of original annotation JSONs")
    p.add_argument("--after", help="dir of reannotated JSONs")
    p.add_argument("--out", default="reannotation_report")
    p.add_argument("--demo", action="store_true", help="synthetic end-to-end demo, no assets needed")
    args = p.parse_args()

    if args.demo:
        demo(args.out)
    elif args.before and args.after:
        compare(args.models, args.before, args.after, args.out)
    else:
        p.error("need --before and --after (or --demo)")


if __name__ == "__main__":
    main()
