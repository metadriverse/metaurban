#!/usr/bin/env python
from panda3d.core import loadPrcFileData
import argparse
import sys
import cv2
import os


from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban import MetaUrban_PACKAGE_DIR
from metaurban.component.sensors.mini_map import MiniMap
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.dashboard import DashBoard
from metaurban.envs import SidewalkStaticMetaUrbanEnv


def capture_headless_image(cuda, image_source="main_camera"):
    if image_source == "main_camera":
        sensors = {"main_camera": ()}
    elif image_source == "rgb_camera":
        sensors = {"rgb_camera": (RGBCamera, 256, 256)}
    elif image_source == "depth_camera":
        sensors = {"depth_camera": (DepthCamera, 256, 256)}
    else:
        sensors = {}
    env = SidewalkStaticMetaUrbanEnv(
        dict(
            crswalk_density=1,
            walk_on_all_regions=False,
            map='X',
            drivable_area_extension=55,
            height_scale=1,
            show_mid_block_map=False,
            show_ego_navigation=False,
            debug=False,
            horizon=300,
            on_continuous_line_done=False,
            out_of_route_done=True,
            show_sidewalk=True,
            show_crosswalk=True,
            
            use_render=False,
            start_seed=666,
            image_on_cuda=cuda,
            object_density=0.1,
            image_observation=True,
            window_size=(600, 400),
            sensors=sensors,
            interface_panel=[],
            vehicle_config={
                "image_source": image_source,
            },
        )
    )
    try:
        env.reset()
        for i in range(10):
            o, r, tm, tc, i = env.step([0, 1])
        assert isinstance(o, dict)
        # print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        o = o["image"][..., -1] * 255 if not cuda else o["image"].get()[..., -1] * 255
        cv2.imwrite(
            os.path.join(
                MetaUrban_PACKAGE_DIR, "examples",
                "{}_from_observation{}.png".format(image_source, "_cuda" if cuda else "")
            ), o
        )
        cam = env.engine.get_sensor(image_source)
        cam.save_image(
            env.agent,
            os.path.join(
                MetaUrban_PACKAGE_DIR, "examples", "{}_from_buffer{}.png".format(image_source, "_cuda" if cuda else "")
            )
        )

        print(
            "{} Test result: \nHeadless mode Offscreen render launched successfully! \n"
            "images named \'{}_from_observation.png\' and \'{}_from_buffer.png\' are saved to {}. "
            "Open it to check if offscreen mode works well".format(
                image_source, image_source, image_source, os.path.join(MetaUrban_PACKAGE_DIR, "examples")
            )
        )
    finally:
        env.close()


def verify_installation(cuda=False, camera="main"):
    env = SidewalkStaticMetaUrbanEnv({"use_render": False, "image_observation": False})
    try:
        env.reset()
        for i in range(1, 100):
            o, r, tm, tc, info = env.step([0, 1])
    except:
        print("Error happens in Bullet physics world !")
        sys.exit()
    else:
        print("Bullet physics world is launched successfully!")
    finally:
        env.close()
        if camera == "main":
            capture_headless_image(cuda)
        elif camera == "rgb":
            capture_headless_image(cuda, image_source="rgb_camera")
        elif camera == "depth":
            capture_headless_image(cuda, image_source="depth_camera")
        else:
            raise ValueError("Can not find camera: {}, please select from [rgb, depth, main]".format(camera))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--camera", type=str, default="main", choices=["main", "rgb", "depth"])
    args = parser.parse_args()
    loadPrcFileData("", "notify-level-task fatal")
    verify_installation(args.cuda, args.camera)
