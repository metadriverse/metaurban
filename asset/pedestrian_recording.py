# Test sidewalk manager that can spawn all types of object on, near, or outside sidewalk
# Please refer to  metaurban.manager.sidewalk_manager for implementation detail
#
import cv2
import numpy as np
import sys
sys.path.append('E:\\Projects\\metavqa_main_branch')
from metaurban.component.traffic_participants.pedestrian import Pedestrian
from metaurban.envs.metaurban_env import metaurbanEnv
from metaurban.component.static_object.test_new_object import TestObject
from metaurban.envs.test_pede_metaurban_env import TestPedemetaurbanEnv
# from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.vehicle.vehicle_type import CustomizedCar
def try_pedestrian(render=False):
    env = TestPedemetaurbanEnv(
        {
            "num_scenarios": 5,
            "traffic_density": 0.1,
            "traffic_mode": "hybrid",
            "start_seed": 3,
            "debug": True,
            # "cull_scene": True,
            "manual_control": True,
            "use_render": render,
            "decision_repeat": 5,
            "need_inverse_traffic": True,
            # "rgb_clip": True,
            "map": 4,
            # "agent_policy": IDMPolicy,
            "height_scale": 1,
            "random_traffic": True,
            "random_lane_width": True,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "force_destroy": True,
            "window_size": (2400, 1600),
            "vehicle_config": {
                "enable_reverse": True,
                "image_source": "main_camera "
            },
            # "sensors": dict(
            #     rgb=(RGBCamera, 960, 640),
            # )
        }
    )
    env.reset()
    env.agent.expert_takeover = True
    frames = []
    camera = env.engine.get_sensor("main_camera")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    frame_rate = 20  # or any other frame rate
    frame_size = (2400, 1600)  # Width, Height - Adjust based on your camera settings
    out = cv2.VideoWriter('E:\\sidewalk_output_video.mp4', fourcc, frame_rate, frame_size)

    try:
        for s in range(1, 200):
            o, r, tm, tc, info = env.step([0, 0])
            
            image = env.engine.get_sensor("rgb_camera").perceive(to_float=False)
            image = image[..., [2, 1, 0]]

            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()

            ego = env.vehicle
            print(ego.crashed_objects)
            rgb = camera.perceive()

            # Convert RGB data for video saving. OpenCV uses BGR format.
            rgb = (rgb * 255).astype(np.uint8)  # Ensure it's scaled to [0, 255]
            # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            out.write(rgb)  # Write frame to video

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True

    finally:
        out.release()  # Release the video writer
        cv2.destroyAllWindows()  # Close all OpenCV windows
        env.close()


if __name__ == "__main__":
    try_pedestrian(True)
