# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Qmini.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # Robot articulation handle for height diagnostics
    robot = env.unwrapped.scene["robot"]
    # The configured spawn height (z component of init_state.pos in qmini.py)
    spawn_height_cfg = float(robot.cfg.init_state.pos[2])
    # Per-env terrain origin z (height of the ground under each env)
    terrain_origins_z = env.unwrapped.scene.env_origins[:, 2]

    step_idx = 0
    print_every = 50  # print one diagnostic line per 50 sim steps

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

            if step_idx % print_every == 0:
                # base_link world-frame position
                base_pos_w = robot.data.root_pos_w  # (num_envs, 3)
                # Height of base_link above the local terrain under each env
                height_above_ground = base_pos_w[:, 2] - terrain_origins_z

                mean_h = height_above_ground.mean().item()
                min_h = height_above_ground.min().item()
                max_h = height_above_ground.max().item()
                suggested = round(mean_h + 0.02, 3)  # ~2 cm clearance above resting height

                print(
                    f"[height @ step {step_idx:>5d}] configured pos.z = {spawn_height_cfg:.3f} m  |  "
                    f"actual base_link height above ground: mean = {mean_h:.3f} m, "
                    f"min = {min_h:.3f} m, max = {max_h:.3f} m  |  "
                    f"suggested pos.z ≈ {suggested:.3f} m"
                )

            step_idx += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
