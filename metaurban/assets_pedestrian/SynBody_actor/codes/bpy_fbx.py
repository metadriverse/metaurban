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

    print(fbx_path)
    bpy.ops.better_export.fbx(
        filepath=fbx_path,
        # check_existing=True,
        my_file_type='.fbx',
        filter_glob="*.fbx",
        # my_scale=1,
        # use_animation=True,
        # use_optimize_for_game_engine=False,
        use_embed_media=True,
        use_copy_texture=True,
    )


if __name__ == "__main__":
    path = sys.argv[-1]
    main(fbx_path=path)
