import sys
from pathlib import Path

import bpy


def main(fbx_path: str):
    bpy.ops.file.make_paths_absolute()
    root = Path(fbx_path).parent

    bpy.ops.file.find_missing_files(directory=(root / "textures").as_posix())

    for img in bpy.data.images:
        filepath = img.filepath
        filepath = Path(filepath)
        img.filepath = '//textures/' + filepath.name

    # deselect
    bpy.ops.object.select_all(action='DESELECT')

    for hair in ['HairBase', 'current_hair', 'current_beard']:
        hair = bpy.data.objects.get(hair, None)
        if hair is None:
            continue
        hair.hide_render = False
        hair.hide_viewport = False
        hair.select_set(state=True)

        bpy.context.view_layer.objects.active = hair
        bpy.ops.object.delete()

    
    bpy.ops.export_scene.gltf(
            filepath=fbx_path,
            export_format='GLTF_EMBEDDED',
            #export_selected=True (I got a warning here)
            # use_selection=True
     )

if __name__ == "__main__":
    path = sys.argv[-1]
    main(fbx_path=path)
