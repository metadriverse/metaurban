import sys
from pathlib import Path

import bpy


def main(obj_path: str):
    root = Path(obj_path).parent

    bpy.ops.file.find_missing_files(directory=str(root / "textures"))
    bpy.ops.file.make_paths_absolute()

    for img in bpy.data.images:
        filepath = img.filepath
        filepath = Path(filepath)
        img.filepath = '//textures/' + filepath.name

    # join all meshes
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()

    bpy.ops.wm.obj_export(filepath=obj_path)


if __name__ == "__main__":
    obj_path = sys.argv[-1]
    main(obj_path=obj_path)
