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
import shutil
import sys

import yaml

from metaurban.asset_metainfo import _get_base, _write_minimal_glb, canonicalize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_loader():
    return _get_base().loader


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
    ann_real, out_real = os.path.realpath(annotation_dir), os.path.realpath(out_dir)
    in_place = out_real == ann_real
    if not in_place and os.path.commonpath([out_real, ann_real]) == ann_real:
        raise SystemExit(f"--out must not be nested inside the annotations dir ({annotation_dir}): "
                         "the sim's annotation loader walks it recursively and would load both copies")
    os.makedirs(out_dir, exist_ok=True)

    glbs = sorted(
        f for f in os.listdir(models_dir) if f.lower().endswith((".glb", ".gltf"))
    ) if os.path.isdir(models_dir) else []
    if not glbs:
        print(f"no GLB assets found in {models_dir}")
        return 0

    done = 0
    rewritten_sources = set()
    for fname in glbs:
        try:
            model = loader.loadModel(os.path.join(models_dir, fname), noCache=True)

            src_path, meta = index.get(fname, (None, None))
            json_path = src_path
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
            elif not in_place:
                # keep subdir layout so same-basename JSONs can't clobber
                json_path = os.path.join(out_dir, os.path.relpath(json_path, annotation_dir))

            pos0, pos1, pos2, length, width, height = canonicalize(
                model, meta.get("hshift", 0.0), meta.get("scale", 1.0)
            )

            meta.update(pos0=pos0, pos1=pos1, pos2=pos2, height=height)
            if "general" in meta:
                meta["general"].update(length=length, width=width)
                if "height" in meta["general"]:
                    meta["general"]["height"] = height
                if "bounding_box" in meta["general"]:
                    x, y = length / 2, width / 2
                    meta["general"]["bounding_box"] = [[x, y], [x, -y], [-x, -y], [-x, y]]
                    meta["general"]["center"] = [0.0, 0.0]
            else:
                meta.update(length=length, width=width)

            if dry_run:
                print(f"[dry-run] {fname}: pos=({pos0:.4f}, {pos1:.4f}, {pos2:.4f}) "
                      f"lwh=({length:.4f}, {width:.4f}, {height:.4f}) -> {json_path}")
            else:
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w") as fh:
                    json.dump(meta, fh, indent=2)
                print(f"[ok] {fname} -> {json_path}")
            if src_path is not None:
                rewritten_sources.add(os.path.realpath(src_path))
            done += 1
        except Exception as e:
            print(f"[skip] {fname}: {e}")

    if not in_place and not dry_run:
        # Copy everything not rewritten (car assets, orphan annotations) through
        # unchanged so --out is a complete replacement for the annotations dir.
        copied = 0
        for root, _, files in os.walk(annotation_dir):
            for f in files:
                src = os.path.join(root, f)
                if os.path.realpath(src) in rewritten_sources:
                    continue
                dst = os.path.join(out_dir, os.path.relpath(src, annotation_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
        if copied:
            print(f"copied {copied} annotation files without a reannotated GLB through unchanged")
    return done


# --- self test ---------------------------------------------------------------


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
