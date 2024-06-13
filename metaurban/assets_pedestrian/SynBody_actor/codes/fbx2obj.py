import sys
from pathlib import Path

import bpy


def main(fbx_path: str):
    # open empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    root = Path(fbx_path).parent

    bpy.ops.import_scene.fbx(filepath=fbx_path)

    bpy.ops.file.find_missing_files(directory=(root / "textures").as_posix())
    bpy.ops.file.make_paths_absolute()

    for img in bpy.data.images:
        filepath = img.filepath
        filepath = Path(filepath)
        img.filepath = '//textures/' + filepath.name

    bpy.ops.export_scene.obj(filepath=str(root / "SMPL-XL-Tpose.obj"))


if __name__ == "__main__":
    path = sys.argv[-1]
    main(fbx_path=path)
