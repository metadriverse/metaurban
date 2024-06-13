import subprocess
from pathlib import Path

from rich.progress import track

# blender_path = "D:/Program Files/Blender/3.3/blender.exe"
blender_path = 'C:/Program Files/Blender Foundation/Blender 3.6/blender.exe'
# blender_path = '/home/SENSETIME/meihaiyi/programs/blender-3.3.5-linux-x64/blender'

# blend_paths = sorted(Path("X:/DressingPeople/Synbody_v1_1/SMPL-XL100").glob("*/scene.blend"))
blend_paths = sorted(Path("X:/DressingPeople/Synbody_v1_1/0006000-0007383_baked").glob("*/scene.blend"))


def rename():
    for blend_path in track(blend_paths):
        root = blend_path.parent
        backup_dir = root / "_backup"
        backup_dir.mkdir(exist_ok=True)

        # fbxs = sorted(root.glob("*.fbx"))
        # npzs = sorted(root.glob("*.npz"))
        # abcs = sorted(root.glob("*.abc"))
        objs = sorted(root.glob("*.obj"))
        mtls = sorted(root.glob("*.mtl"))
        # for file in fbxs + npzs + abcs:
        for file in objs + mtls:
            new_path = backup_dir / file.name
            file.rename(new_path)


def rename_new():
    fbx_paths = sorted(
        Path("/mnt/SenseFeitoria/DressingPeople/Synbody_v1_1/0006000-0007383_baked").glob("*/SMPL-XL-baked.fbx")
    )
    for fbx_path in track(fbx_paths):
        print(fbx_path)
        fbx_path.rename(fbx_path.parent / "people_baked.fbx")


def blend2obj():
    for blend_path in track(blend_paths):
        root = blend_path.parent
        fbx_path = root / "SMPL-XL-baked.fbx"
        obj_path = root / "SMPL-XL-Tpose.obj"

        subprocess.run(
            [
                blender_path,
                blend_path.as_posix(),
                "--background",
                "-P",
                "bpy_obj.py",
                "--",
                fbx_path.as_posix(),
            ]
        )


def blend2fbx():
    for blend_path in track(blend_paths):
        root = blend_path.parent
        fbx_path = root / "SMPL-XL-baked.fbx"
        obj_path = root / "SMPL-XL-Tpose.obj"

        print(fbx_path.as_posix())
        subprocess.run(
            [
                blender_path,
                blend_path.as_posix(),
                "--background",
                "-P",
                "bpy_fbx.py",
                "--",
                fbx_path.as_posix(),
            ]
        )


def fbx2obj():
    fbx_paths = sorted(Path("X:/DressingPeople/Synbody_v1_1/0006000-0007383_baked").glob("*/people_baked.fbx"))
    for fbx_path in track(fbx_paths):
        subprocess.run(
            [
                blender_path,
                "-b",
                "-P",
                "fbx2obj.py",
                "--",
                fbx_path.as_posix(),
            ]
        )


if __name__ == "__main__":
    # rename()
    # rename_new()
    # blend2obj()
    blend2fbx()
    # fbx2obj()
