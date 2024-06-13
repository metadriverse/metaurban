from metaurban.envs.metaurban_env import metaurbanEnv
from metaurban.policy.idm_policy import IDMPolicy
from IPython.display import Image

env=metaurbanEnv(dict(map="X",
                      agent_policy=IDMPolicy,
                      log_level=50,
                      accident_prob=1.0,
                      traffic_density=0.3))
try:
    # run several episodes
    env.reset()
    for step in range(600):
        # simulation
        _,_,_,_,info = env.step([0, 3])
        env.render(mode="topdown", 
                   window=True,
                   screen_record=True,
                   screen_size=(700, 870),
                   camera_position=(60,0)
                  )
        if info["arrive_dest"]:
            break
    env.top_down_renderer.generate_gif()
finally:
    env.close()

