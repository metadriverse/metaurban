#!/usr/bin/env python
import argparse
import time

import numpy as np
import logging
import tqdm
from metaurban import SidewalkStaticMetaUrbanEnv
from metadrive.utils import setup_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", "-n", default=1000, type=int, help="Total steps of profiling.")
    args = parser.parse_args()

    setup_logger(debug=False)
    env = SidewalkStaticMetaUrbanEnv(dict(num_scenarios=1000, start_seed=1010, object_density=0.05))
    obs, _ = env.reset()
    start = time.time()
    reset_used_time = 0
    action = [0.0, 0.]
    total_steps = args.num_steps
    for s in tqdm.trange(total_steps):
        o, r, tm, tc, i = env.step(action)
        if tm or tc:
            start_reset = time.time()
            env.reset()
            reset_used_time += time.time() - start_reset
    print(
        "Total Time Elapse: {:.3f}, average FPS: {:.3f}.".format(
            time.time() - start, total_steps / (time.time() - start - reset_used_time)
        )
    )
