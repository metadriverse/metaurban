import shutil
from pathlib import Path

import xrfeitoria as xf
from xrfeitoria.sequence.sequence_unreal import SequenceUnreal
from xrfeitoria.factory import XRFeitoriaUnreal
from rich.progress import track

anim_path = Path("Subject_75_F_12.fbx").resolve()
obj_paths = sorted(Path("SMPL-XL100").resolve().glob("*/_backup/people_tpose.obj"))
fbx_paths = sorted(Path("SMPL-XL100").resolve().glob("*/SMPL-XL-baked.fbx"))
unreal_project = "E:/UE/XRFeitoriaUnreal_Sample/XRFeitoriaUnreal_Sample.uproject"

def import_obj(xf_runner: XRFeitoriaUnreal):
    for idx, obj_path in enumerate(track(obj_paths)):
        new_path = obj_path.with_stem(f'{obj_path.parent.parent.name}_Tpose')
        shutil.copyfile(obj_path, new_path)
        location = (idx // 10 * 2, idx % 10 * 2, 0)
        try:
            print(f"Importing {new_path}")
            # SMPL_XL_path = xf_runner.utils.import_asset(path=new_path)
            xf_runner.Actor.import_from_file(file_path=new_path, location=location)
        except Exception as e:
            print(f"Failed to import {new_path}")


def import_fbx(xf_runner: XRFeitoriaUnreal):
    for idx, fbx_path in enumerate(track(fbx_paths)):
        new_path = fbx_path.with_stem(fbx_path.parent.name)
        shutil.copyfile(fbx_path, new_path)
        try:
            print(f"Importing {new_path}")
            SMPL_XL_path = xf_runner.utils.import_asset(path=new_path)
            animation_path = xf_runner.utils.import_anim(
                path=anim_path, skeleton_path=f"{SMPL_XL_path}_Skeleton"
            )[0]
        except Exception as e:
            print(f"Failed to import {new_path}")


def add_to_seq(xf_runner: XRFeitoriaUnreal):
    root = "/Game/XRFeitoriaUnreal/Assets"
    with xf_runner.Sequence.new(
        seq_name="SMPL-XL100",
        level="/Game/Levels/NewMap",
        replace=True,
        seq_length=300,
    ) as seq:
        seq.show()
        seq.spawn_camera(location=(1000, 3000, 600), rotation=(0, -20, -90))
        for idx, fbx_path in enumerate(track(fbx_paths)):
            _name = fbx_path.parent.name
            _loc = (idx // 10 * 2, idx % 10 * 2, 0)
            _actor_path = f"{root}/{_name}/{_name}"
            _anim_path = f"{root}/{_name}/Animation/{anim_path.stem}"
            seq.spawn_actor(
                actor_asset_path=_actor_path,
                actor_name=_name,
                location=_loc,
                rotation=(0, 0, 0),
                stencil_value=idx,
                anim_asset_path=_anim_path,
            )


with xf.init_unreal(project_path=unreal_project) as xf_runner:
    import_obj(xf_runner)
    # import_fbx(xf_runner)
    # add_to_seq(xf_runner)
