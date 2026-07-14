"""Reannotate GLB assets to canonical frames.

Recomputes the per-asset adjustment annotations (``adj_parameter_folder``
JSONs) so that every asset, once TestObject applies ``hshift``/``pos``/
``scale``, sits in the canonical frame assumed by its physics chassis:
bounding box centered on X/Y, base resting at z=0, length along X, meters.

Semantic fields that cannot be derived from geometry (``hshift``, ``scale``,
``detail_type``, ``CLASS_NAME``, ...) are preserved from the existing
annotation; assets without one get defaults (hshift=0, scale=1, type from
filename).

Usage:
    python scripts/reannotate_assets.py                 # reannotate in place
    python scripts/reannotate_assets.py --dry-run       # preview changes
    python scripts/reannotate_assets.py --models DIR --annotations DIR --out DIR
    python scripts/reannotate_assets.py --self-test
"""
import argparse
import json
import math
import os
import struct
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_LOADER = None


def _make_loader():
    """Headless Panda3D loader with the same GLB plugin the sim uses."""
    global _LOADER
    if _LOADER is None:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "window-type none\naudio-library-name null\nnotify-level fatal")
        from direct.showbase.ShowBase import ShowBase
        import gltf
        base = ShowBase()
        gltf.patch_loader(base.loader)
        _LOADER = base.loader
    return _LOADER


def canonicalize(model, hshift=0.0, scale=1.0):
    """Compute canonical-frame corrections for a loaded model.

    Returns (pos0, pos1, pos2, length, width, height) such that applying
    setH(hshift); setPos(pos0, pos1, pos2); setScale(scale) — the exact
    transform TestObject applies — grounds the asset at z=0 with its
    bounding box centered on X/Y.
    """
    from panda3d.core import NodePath
    root = NodePath("canonical")
    model.reparentTo(root)
    model.setH(hshift)
    model.setScale(scale)
    bounds = model.getTightBounds(root)
    model.detachNode()
    if bounds is None:
        raise ValueError("model has no geometry")
    lo, hi = bounds
    return (
        -(lo.x + hi.x) / 2.0,
        -(lo.y + hi.y) / 2.0,
        -lo.z,
        hi.x - lo.x,
        hi.y - lo.y,
        hi.z - lo.z,
    )


def load_annotation_index(annotation_dir):
    """Map GLB filename -> (json path, annotation dict)."""
    index = {}
    if not os.path.isdir(annotation_dir):
        return index
    for root, _, files in os.walk(annotation_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            path = os.path.join(root, f)
            with open(path, "r") as fh:
                try:
                    meta = json.load(fh)
                except ValueError:
                    print(f"[skip] unreadable annotation: {path}")
                    continue
            if "filename" in meta:
                index[meta["filename"]] = (path, meta)
    return index


def reannotate(models_dir, annotation_dir, out_dir, dry_run=False):
    loader = _make_loader()
    index = load_annotation_index(annotation_dir)
    os.makedirs(out_dir, exist_ok=True)

    glbs = sorted(
        f for f in os.listdir(models_dir) if f.lower().endswith((".glb", ".gltf"))
    ) if os.path.isdir(models_dir) else []
    if not glbs:
        print(f"no GLB assets found in {models_dir}")
        return 0

    done = 0
    for fname in glbs:
        model = loader.loadModel(os.path.join(models_dir, fname), noCache=True)
        if model is None:
            print(f"[skip] failed to load {fname}")
            continue

        json_path, meta = index.get(fname, (None, None))
        if meta is None:
            # Fresh annotation; filenames look like "<type>-<uid>.glb".
            detail_type = os.path.splitext(fname)[0].rsplit("-", 1)[0]
            meta = {
                "CLASS_NAME": detail_type,
                "filename": fname,
                "hshift": 0.0,
                "scale": 1.0,
                "general": {"detail_type": detail_type, "length": 0, "width": 0},
            }
            json_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".json")
        elif out_dir != annotation_dir:
            json_path = os.path.join(out_dir, os.path.basename(json_path))

        try:
            pos0, pos1, pos2, length, width, height = canonicalize(
                model, meta.get("hshift", 0.0), meta.get("scale", 1.0)
            )
        except ValueError:
            print(f"[skip] no geometry in {fname}")
            continue

        meta.update(pos0=pos0, pos1=pos1, pos2=pos2, height=height)
        if "general" in meta:
            meta["general"].update(length=length, width=width)
        else:
            meta.update(length=length, width=width)

        if dry_run:
            print(f"[dry-run] {fname}: pos=({pos0:.4f}, {pos1:.4f}, {pos2:.4f}) "
                  f"lwh=({length:.4f}, {width:.4f}, {height:.4f}) -> {json_path}")
        else:
            with open(json_path, "w") as fh:
                json.dump(meta, fh, indent=2)
            print(f"[ok] {fname} -> {json_path}")
        done += 1
    return done


# --- self test ---------------------------------------------------------------


def _write_minimal_glb(path):
    """A hand-built GLB: one triangle spanning x:[0,1] y:[0,2] z:[0,3] (Y-up)."""
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
    pos_bin = b"".join(struct.pack("<fff", *p) for p in positions)
    idx_bin = struct.pack("<HHH", 0, 1, 2) + b"\x00\x00"  # pad to 4 bytes
    blob = pos_bin + idx_bin
    gltf_json = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bin), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bin), "byteLength": 6, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [0.0, 0.0, 0.0], "max": [1.0, 2.0, 3.0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    json_bin = json.dumps(gltf_json).encode()
    json_bin += b" " * (-len(json_bin) % 4)
    total = 12 + 8 + len(json_bin) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bin), 0x4E4F534A) + json_bin)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def self_test():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="reannotate_test_")
    models, ann = os.path.join(tmp, "models"), os.path.join(tmp, "adj")
    os.makedirs(models)
    _write_minimal_glb(os.path.join(models, "testbox-cafe0000.glb"))
    assert reannotate(models, ann, ann) == 1

    with open(os.path.join(ann, "testbox-cafe0000.json")) as f:
        meta = json.load(f)
    # glTF is Y-up, Panda is Z-up: extents (1, 2, 3) -> (1, 3, 2).
    assert math.isclose(meta["general"]["length"], 1.0, abs_tol=1e-5), meta
    assert math.isclose(meta["general"]["width"], 3.0, abs_tol=1e-5), meta
    assert math.isclose(meta["height"], 2.0, abs_tol=1e-5), meta
    assert meta["general"]["detail_type"] == "testbox"
    # Applying the annotation must center X/Y and ground the base at z=0.
    from panda3d.core import NodePath
    model = _make_loader().loadModel(os.path.join(models, meta["filename"]), noCache=True)
    root = NodePath("origin")
    model.reparentTo(root)
    model.setH(meta["hshift"])
    model.setPos(meta["pos0"], meta["pos1"], meta["pos2"])
    model.setScale(meta["scale"])
    lo, hi = model.getTightBounds(root)
    assert math.isclose(lo.x + hi.x, 0.0, abs_tol=1e-5), (lo, hi)
    assert math.isclose(lo.y + hi.y, 0.0, abs_tol=1e-5), (lo, hi)
    assert math.isclose(lo.z, 0.0, abs_tol=1e-5), (lo, hi)
    print("self-test passed")


def main():
    with open(os.path.join(REPO_ROOT, "path_config.yaml")) as f:
        cfg = yaml.safe_load(f)["path"]
    parent = os.path.join(REPO_ROOT, cfg["parentfolder"])

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=os.path.join(REPO_ROOT, cfg["metaurbanasset"]))
    p.add_argument("--annotations", default=os.path.join(parent, cfg["subfolders"]["adj_parameter_folder"]))
    p.add_argument("--out", default=None, help="output dir for JSONs (default: --annotations, in place)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    n = reannotate(args.models, args.annotations, args.out or args.annotations, args.dry_run)
    print(f"reannotated {n} assets")


if __name__ == "__main__":
    sys.exit(main())
