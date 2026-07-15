"""Auto-generated asset metainfo: bounding boxes and default headings from GLBs.

Replaces the hand-maintained per-asset JSONs in ``adj_parameter_folder`` for
static objects. Geometry (position corrections, length/width/height) is
derived from each GLB's tight bounds and cached next to the models; only the
curated semantics that cannot be derived from geometry live in
``asset_semantics.json`` at the repo root:

    filename -> {detail_type, hshift (default heading, deg), scale}

New GLBs need no annotation at all: geometry is derived on first use and the
detail type falls back to the filename stem (``Bench-<uid>.glb`` -> ``Bench``).

Usage:
    python -m metaurban.asset_metainfo                  # (re)build the cache
    python -m metaurban.asset_metainfo --seed-from DIR  # extract semantics from legacy adj JSONs
    python -m metaurban.asset_metainfo --self-test
"""
import argparse
import json
import os
import struct

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANTICS_FILE = os.path.join(REPO_ROOT, "asset_semantics.json")
CACHE_NAME = ".metainfo_cache.json"

_BASE = None


def _get_base(window_type="none"):
    """Headless Panda3D ShowBase with the same GLB plugin the sim uses."""
    global _BASE
    if _BASE is None:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", f"window-type {window_type}\naudio-library-name null\nnotify-level fatal")
        from direct.showbase.ShowBase import ShowBase
        import gltf
        _BASE = ShowBase()
        gltf.patch_loader(_BASE.loader)
    return _BASE


def _ensure_gltf(loader):
    """Patch GLB support onto a loader unless it is already patched.

    The engine only patches its loader in on/offscreen modes; in
    RENDER_MODE_NONE (headless training) it stays unpatched.
    """
    if getattr(loader.loadModel, "__module__", "") != "gltf":
        import gltf
        gltf.patch_loader(loader)
    return loader


def _resolve_loader():
    """The running engine's loader when one exists, else a headless one.

    The engine subclasses ShowBase, so spawning our own base while it is
    alive would raise "Attempt to spawn multiple ShowBase instances!".
    """
    import builtins
    running = getattr(builtins, "base", None)  # EngineCore or any ShowBase
    if running is not None:
        return _ensure_gltf(running.loader)
    return _ensure_gltf(_get_base().loader)


def canonicalize(model, hshift=0.0, scale=1.0):
    """Compute canonical-frame corrections for a loaded model.

    Returns (pos0, pos1, pos2, length, width, height) such that applying
    setH(hshift); setPos(pos0, pos1, pos2); setScale(scale) — the exact
    transform TestObject applies — grounds the asset at z=0 with its
    bounding box centered on X/Y.
    """
    from panda3d.core import NodePath
    if scale <= 0:
        raise ValueError(f"invalid scale {scale}")
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


def load_semantics():
    try:
        with open(SEMANTICS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{SEMANTICS_FILE} not found. It holds the curated per-asset semantics "
            "(detail_type, default heading, scale) and ships with the repo; regenerate it with "
            "'python -m metaurban.asset_metainfo --seed-from <adj_parameter_folder>'"
        ) from None


def load_asset_metainfo(models_dir, semantics=None):
    """Metainfo dicts (legacy adj_parameter_folder schema) for annotated GLBs in models_dir.

    Geometry comes from a per-directory cache keyed by (mtime, size, hshift,
    scale); new or changed GLBs are re-derived from their tight bounds. GLBs
    without an asset_semantics.json entry are skipped: detail_type, default
    heading and scale are curated decisions geometry cannot supply, and raw
    (e.g. objaverse) units at scale 1 are arbitrary.
    """
    if semantics is None:
        semantics = load_semantics()
    if not os.path.isdir(models_dir):
        print(f"[asset_metainfo] models dir not found, no static assets: {models_dir}")
        return []
    cache_file = os.path.join(models_dir, CACHE_NAME)
    try:
        with open(cache_file) as f:
            cache = json.load(f)
    except (FileNotFoundError, ValueError):
        cache = {}

    entries, unlisted = [], 0
    for fname in sorted(os.listdir(models_dir)):
        if not fname.lower().endswith((".glb", ".gltf")):
            continue
        sem = semantics.get(fname)
        if sem is None:
            unlisted += 1
            continue
        hshift, scale = sem.get("hshift", 0.0), sem.get("scale", 1.0)
        detail_type = sem.get("detail_type", os.path.splitext(fname)[0].rsplit("-", 1)[0])
        try:
            st = os.stat(os.path.join(models_dir, fname))
        except OSError as e:  # dangling symlink, deleted mid-scan
            print(f"[asset_metainfo] skip {fname}: {e}")
            continue
        entries.append((fname, hshift, scale, detail_type, [st.st_mtime, st.st_size, hshift, scale]))
    if unlisted:
        print(f"[asset_metainfo] {unlisted} GLBs in {models_dir} have no {os.path.basename(SEMANTICS_FILE)} "
              "entry and will not spawn; add entries (detail_type/hshift/scale) to include them")

    stale = [e for e in entries if cache.get(e[0], {}).get("key") != e[4]]
    if stale:
        print(f"[asset_metainfo] deriving geometry for {len(stale)} new/changed GLBs in {models_dir} ...")
        ldr = _resolve_loader()
        for fname, hshift, scale, _, key in stale:
            try:
                model = ldr.loadModel(os.path.join(models_dir, fname), noCache=True)
                pos0, pos1, pos2, length, width, height = canonicalize(model, hshift, scale)
            except Exception as e:
                print(f"[asset_metainfo] skip {fname}: {e}")
                cache.pop(fname, None)  # never serve stale geometry for a changed file
                continue
            cache[fname] = {"key": key, "pos": [pos0, pos1, pos2], "lwh": [length, width, height]}
        current = {e[0] for e in entries}
        cache = {k: v for k, v in cache.items() if k in current}  # prune deleted/renamed GLBs
        tmp = f"{cache_file}.{os.getpid()}.tmp"  # per-process: parallel env workers race on cold start
        try:
            with open(tmp, "w") as f:
                json.dump(cache, f)
            os.replace(tmp, cache_file)
        except OSError as e:  # read-only assets dir, concurrent replace: keep in-memory result
            print(f"[asset_metainfo] cache not persisted ({e}); geometry will be re-derived next run")

    metainfos = []
    for fname, hshift, scale, detail_type, _ in entries:
        entry = cache.get(fname)
        if entry is None:  # unreadable GLB, skipped above
            continue
        (pos0, pos1, pos2), (length, width, height) = entry["pos"], entry["lwh"]
        metainfos.append(
            {
                "CLASS_NAME": fname,
                "filename": fname,
                "hshift": hshift,
                "scale": scale,
                "pos0": pos0,
                "pos1": pos1,
                "pos2": pos2,
                "height": height,
                "general": {"detail_type": detail_type, "length": length, "width": width},
            }
        )
    return metainfos


def load_annotation_index(annotation_dir):
    """Map GLB filename -> (json path, annotation dict) for legacy adj JSONs."""
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
                except (ValueError, UnicodeDecodeError):
                    print(f"[skip] unreadable annotation: {path}")
                    continue
            if "filename" in meta:
                index[meta["filename"]] = (path, meta)
    return index


def seed_semantics(adj_dir):
    """Extract the curated fields from legacy adj_parameter_folder JSONs."""
    try:
        semantics = load_semantics()
    except FileNotFoundError:  # --seed-from is exactly the bootstrap path
        semantics = {}
    n = 0
    for fname, (path, meta) in load_annotation_index(adj_dir).items():
        if os.path.basename(path).lower().startswith("car") or "general" not in meta:
            continue
        semantics[fname] = {
            "detail_type": meta["general"]["detail_type"],
            "hshift": meta.get("hshift", 0.0),
            "scale": meta.get("scale", 1.0),
        }
        n += 1
    with open(SEMANTICS_FILE, "w") as f:
        json.dump(semantics, f, indent=1, sort_keys=True)
    print(f"seeded {n} entries -> {SEMANTICS_FILE}")


# --- self test ----------------------------------------------------------------


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
    _write_glb(path, gltf_json, blob)


def _write_glb(path, gltf_json, blob):
    """Write a binary glTF container with one JSON and one BIN chunk."""
    json_bin = json.dumps(gltf_json).encode()
    json_bin += b" " * (-len(json_bin) % 4)
    total = 12 + 8 + len(json_bin) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bin), 0x4E4F534A) + json_bin)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def self_test():
    import math
    import tempfile
    tmp = tempfile.mkdtemp(prefix="asset_metainfo_test_")
    _write_minimal_glb(os.path.join(tmp, "testbox-cafe0000.glb"))
    sem = {"testbox-cafe0000.glb": {"detail_type": "testbox"}}
    metas = load_asset_metainfo(tmp, semantics=sem)
    assert len(metas) == 1, metas
    m = metas[0]
    # glTF is Y-up, Panda is Z-up: extents (1, 2, 3) -> (1, 3, 2).
    assert math.isclose(m["general"]["length"], 1.0, abs_tol=1e-5), m
    assert math.isclose(m["general"]["width"], 3.0, abs_tol=1e-5), m
    assert math.isclose(m["height"], 2.0, abs_tol=1e-5), m
    assert m["general"]["detail_type"] == "testbox"
    assert os.path.exists(os.path.join(tmp, CACHE_NAME))
    assert load_asset_metainfo(tmp, semantics=sem) == metas  # second pass served from cache
    assert load_asset_metainfo(tmp, semantics={}) == []  # GLBs without curated semantics don't spawn
    print("self-test passed")


def default_models_dir():
    import yaml
    with open(os.path.join(REPO_ROOT, "path_config.yaml")) as f:
        return os.path.join(REPO_ROOT, yaml.safe_load(f)["path"]["metaurbanasset"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=default_models_dir())
    p.add_argument("--seed-from", help="legacy adj_parameter_folder to extract semantics from")
    p.add_argument("--force", action="store_true", help="discard the cache and re-derive everything")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.seed_from:
        seed_semantics(args.seed_from)
    if args.force:
        cache = os.path.join(args.models, CACHE_NAME)
        if os.path.exists(cache):
            os.remove(cache)
    metainfos = load_asset_metainfo(args.models)
    print(f"metainfo ready for {len(metainfos)} assets")


if __name__ == "__main__":
    main()
