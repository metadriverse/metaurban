# - Added a test case for the DepthCamera

# Previous version:
# 1. first frame is empty
# 2. fail to use pos and hpr to control the depth camera

# Current version:
# 1. first frame is not empty
# 2. use pos and hpr to control the depth camera successfully

# Changes:
# 1.
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import gymnasium as gym
import mediapy
import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation

from simgen import SimGenPipeline


def postprocess_semantic_image(image):
    """
    In order to align with the Segformer's output, we modify the output color of the semantic image from MetaDrive.
    """
    # unique_elements = np.unique(image.reshape(-1, 3), axis=0)
    # print("Unique elements: ", unique_elements)

    # customized
    old_LANE_LINE = (255, 255, 255)
    old_CROSSWALK = (55, 176, 189)
    old_SIDEWALK = (244, 35, 232)

    # These color might be prettier?
    new_LANE_LINE = (128, 64, 128)
    new_CROSSWALK = (128, 64, 128)
    new_SIDEWALK = (152, 251, 152)

    # Change the color of the lane line and crosswalk
    assert image.dtype == np.uint8

    is_lane_line = (
        (image[..., 0] == old_LANE_LINE[0]) & (image[..., 1] == old_LANE_LINE[1]) & (image[..., 2] == old_LANE_LINE[2])
    )
    image[is_lane_line] = new_LANE_LINE

    is_crosswalk = (
        (image[..., 0] == old_CROSSWALK[0]) & (image[..., 1] == old_CROSSWALK[1]) & (image[..., 2] == old_CROSSWALK[2])
    )
    image[is_crosswalk] = new_CROSSWALK
    
    is_sidewalk = (
        (image[..., 0] == old_SIDEWALK[0]) & (image[..., 1] == old_SIDEWALK[1]) & (image[..., 2] == old_SIDEWALK[2])
    )
    image[is_sidewalk] = new_SIDEWALK

    return image


def add_text(image, text_prompt):
    # Convert the image to RGBA mode
    image = Image.fromarray(image, mode="RGB").convert("RGBA")

    # Create a transparent overlay
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Define the text and font
    font_path = "Arial.ttf"  # Replace with your font path
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    # Get text size
    text_width = draw.textlength(text_prompt, font=font)

    # Make the text align in left bottom corner
    position = (50, image.height - font_size - 50)

    # Draw the semi-transparent rectangle
    off = 15
    rect_start = (position[0] - off, position[1] - off)
    rect_end = (position[0] + text_width + off, position[1] + font_size + off)
    draw.rectangle([rect_start, rect_end], fill=(255, 255, 255, int(255 * 0.5)))  # Alpha = 0.5

    # Draw the text
    draw.text(position, text_prompt, font=font, fill="black")

    # Merge the overlay with the original image
    image = Image.alpha_composite(image, overlay)

    # Convert back to RGB mode if needed
    image = image.convert("RGB")

    # Get back the image
    image = np.array(image)
    return image


if __name__ == "__main__":

    # ===== MetaUrban Setup =====
    map_type = 'X'
    config = dict(
        crswalk_density=1,
        object_density=0.4,
        use_render=False,
        map=map_type,
        manual_control=False,
        drivable_area_extension=55,
        height_scale=1,
        spawn_deliveryrobot_num=2,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            lidar=dict(num_lasers=120, distance=50),
            lane_line_detector=dict(num_lasers=0, distance=50),
            side_detector=dict(num_lasers=12, distance=50)
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        traffic_density=0.2,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=1,
        accident_prob=0,
        window_size=(800, 450),
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        image_observation=True,
        norm_pixel=False,
        stack_size=1,
        sensors=dict(
            rgb_camera=(RGBCamera, 800, 450),
            depth_camera=(DepthCamera, 800, 450),
            semantic_camera=(SemanticCamera, 800, 450),
        ),
        agent_observation=ThreeSourceMixObservation,
        interface_panel=[],
        
        camera_dist = 0.8,  # 0.8, 1.71
        camera_height = 1.5,  # 1.5
        camera_pitch = None,
        camera_fov = 66,  # 60, 66
    )

    env = SidewalkStaticMetaUrbanEnv(config)
    o, _ = env.reset(seed=0)
    for t in range(5):
        env.step([0., 0.])
    # env.agents['default_agent'].set_position([160, 5, env.agents['default_agent'].HEIGHT / 2])
    # env.agents['default_agent'].set_heading_theta(45 / 180 * np.pi)
    
    env.agents['default_agent'].set_position([-5, 0, env.agents['default_agent'].HEIGHT / 2])
    env.agents['default_agent'].set_heading_theta(0 / 180 * np.pi)

    # ===== SimGen Setup =====
    pipeline = SimGenPipeline()
    ddim_steps = 100
    region_candidates = [
        "Los Angeles, United States",
        "Beijing, China",
        "Pretoria, South Africa",
        "London, England",
        "Riyadh, Saudi Arabia",
        "Moscow, Russia",
        "Zurich, Switzerland",
        "Kyoto, Japan",
        "Vancouver, Canada",
        "Seoul, Korea",
        "Delhi, India",
    ]

    prefix_candidates = [
        # " in a lego style",
        # " in a ukiyo-e style",
        # " in a minecraft style",
        " "
    ]

    # ===== Prepare input =====
    camera = env.engine.get_sensor("rgb_camera")
    rgb_front = camera.perceive(
        clip=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, 2, 2], hpr=[0, 0, 0]
    )
    max_rgb_value = rgb_front.max()
    rgb = rgb_front[..., ::-1]
    if max_rgb_value > 1:
        rgb = rgb.astype(np.uint8)
    else:
        rgb = (rgb * 255).astype(np.uint8)

    camera = env.engine.get_sensor("depth_camera")
    depth_front = camera.perceive(
        clip=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, 2, 2], hpr=[0, 0, 0]
    ).reshape(450, 800, -1)[..., -1]
    depth = depth_front
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    depth_img = cv2.bitwise_not(depth_front)
    depth_img = depth_img[..., None]

    camera = env.engine.get_sensor("semantic_camera")
    semantic_front = camera.perceive(
        clip=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, 2, 2], hpr=[0, 0, 0]
    )
    max_rgb_value = semantic_front.max()
    semantic = semantic_front
    if max_rgb_value > 1:
        semantic = semantic.astype(np.uint8)
    else:
        semantic = (semantic * 255).astype(np.uint8)
    semantic = postprocess_semantic_image(semantic[..., ::-1])
    # semantic = semantic

    simgen_input = {
        'rgb': rgb,
        'depth': depth_img,
        'seg': semantic,
    }

    # ===== Close the environment to reduce GPU Memory =====
    env.close()

    # ===== Run simulation =====
    depth_img = Image.fromarray(simgen_input["depth"].repeat(3, axis=-1), mode="RGB")
    seg_img = Image.fromarray(simgen_input["seg"], mode="RGB")
    rgb_img = Image.fromarray(simgen_input["rgb"], mode="RGB")

    generated_imgs = []
    for ep in range(3):
        seed = np.random.randint(0, 100000)
        np_random = np.random.RandomState(seed)
        sampled_region_name = np_random.choice(region_candidates)
        sampled_prefix = np_random.choice(prefix_candidates)
        sampled_prefix = sampled_prefix if np_random.rand() < 0.5 else ""
        text_prompt = "An image of a city street in {}{}.".format(sampled_region_name, sampled_prefix)
        print("Text prompt: ", text_prompt)

        # Run SimGen
        with torch.no_grad():
            output = pipeline(
                depth_image=depth_img,
                seg_image=seg_img,
                content_image=rgb_img,
                prompt=text_prompt,
                seed=seed,
                num_inference_steps=ddim_steps,
            )
            images = output.images
            image = images[0]
            h, w = image.shape[:2]
            vis_w = rgb.shape[1]
            image = cv2.resize(image, (vis_w, int(h * vis_w / w)))
            image = add_text(image, text_prompt)
            generated_imgs.append(image)

    rgbds = np.concatenate((rgb, depth_colored, semantic), axis=1)
    generated_imgs = np.concatenate(generated_imgs, axis=1)

    displayed_imgs = np.concatenate((rgbds, generated_imgs), axis=0)

    # ===== Display the raw observation and generated image =====
    plt.imshow(displayed_imgs)
    plt.title('Paired RGB-Depth-Semantic-Generated Image (RGB-Depth-Semantic || Sample 1-2-3)')
    plt.axis('off')
    plt.show()
