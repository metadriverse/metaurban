from metaurban.envs.metaurban_env import metaurbanEnv

if __name__ == "__main__":
    env = metaurbanEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "start_seed": 4,
            "vehicle_config": {
                "stack_size": 5,
                "norm_pixel": False,
                "rgb_camera": (800, 500),
                # "rgb_to_grayscale": True
            },
            "manual_control": True,
            "use_render": False,
            "image_observation": True,  # it is a switch telling metaurban to use rgb as observation
            "norm_pixel": False,  # clip rgb to range(0,1) instead of (0, 255)
            # "pstats": True,
        }
    )
    env.reset()
    # # print m to capture rgb observation
    env.engine.accept("m", env.agent.get_camera(env.agent.config["image_source"]).save_image, extraArgs=[env.agent])
    import cv2

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        # save
        rgb_cam = env.agent.get_camera(env.agent.config["image_source"])
        # rgb_cam.save_image(env.agent, name="{}.png".format(i))
        cv2.imshow('img', o["image"][..., -1] / 255)
        cv2.waitKey(0)

        # if env.config["use_render"]:
        # for i in range(ImageObservation.STACK_SIZE):
        #      ObservationType.show_gray_scale_array(o["image"][:, :, i])
        # image = env.render(mode="any str except human", text={"can you see me": i})
        # ObservationType.show_gray_scale_array(image)
        if tm or tc:
            # print("Reset")
            env.reset()
    env.close()