import torch
import numpy as np
import os.path as osp
from metaurban.engine.engine_utils import get_global_config
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.engine.logger import get_logger
from stable_baselines3 import PPO

from metaurban.utils.math import panda_vector

ckpt_path = osp.join(osp.dirname(__file__), "expert_weights.npz")
_expert_weights = None
_expert_observation = None

logger = get_logger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from panda3d.core import Point3, Vec2, LPoint3f
dests = [(10.0, 14.0), (63.5, -45.0), (8.0, 14.0), (10.0, 18.0), (18.0, 18.0)]


# dests = [(63.0 45.0), (8.0, 14.0), (10.0, 18.0), (18.0, 18.0)]
def rule_expert(vehicle, deterministic=False, need_obs=False):
    dest_pos = vehicle.navigation.get_checkpoints()[0]
    position = vehicle.position

    dest = panda_vector(dest_pos[0], dest_pos[1])
    vec_to_2d = dest - position
    dist_to = vec_to_2d.length()

    heading = Vec2(*vehicle.heading).signedAngleDeg(vec_to_2d) * 3

    if dist_to > 2:
        vehicle._body.setAngularMovement(heading)
        vehicle._body.setLinearMovement(LPoint3f(0, 1, 0) * 6, True)
    else:
        vehicle._body.setLinearMovement(LPoint3f(0, 1, 0) * 1, True)
    return None


def get_dest_heading(obj, dest_pos):
    position = obj.position

    dest = panda_vector(dest_pos[0], dest_pos[1])
    vec_to_2d = dest - position
    # dist_to = vec_to_2d.length()
    ####

    heading = Vec2(*obj.heading).signedAngleDeg(vec_to_2d)
    #####
    return heading
