# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless behavioral eval for rsl_rl checkpoints: does the policy step and track commands?

Prints commanded vs actual forward velocity and swing rate — a two-minute answer to
"is it walking?" without opening the viewer.

Run::

    python scripts/rsl_rl/eval_policy.py --headless
    python scripts/rsl_rl/eval_policy.py --headless --load_run <run> --checkpoint model_1200.pt
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--task", type=str, default="qmini-velocity-play")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import importlib.metadata as metadata

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import Qmini.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = (
        args_cli.checkpoint
        if args_cli.checkpoint and os.path.isfile(args_cli.checkpoint)
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact = unwrapped.scene.sensors["contact_forces"]
    foot_ids = [i for i, n in enumerate(contact.body_names) if n.startswith("ankle")]

    obs = env.get_observations()
    swings = torch.zeros(args_cli.num_envs, len(foot_ids), device=unwrapped.device)
    was_air = torch.zeros_like(swings, dtype=torch.bool)
    vx_sum, cmd_sum, n = 0.0, 0.0, 0
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            air = contact.data.current_air_time[:, foot_ids] > 0.05
            swings += (air & ~was_air).float()
            was_air = air
            cmd = unwrapped.command_manager.get_command("base_velocity")
            vx_sum += robot.data.root_lin_vel_b[:, 0].mean().item()
            cmd_sum += cmd[:, 0].mean().item()
            n += 1
    dur = args_cli.steps * unwrapped.step_dt
    print(f"\n==== BEHAVIOR over {dur:.1f} s ====")
    print(f"mean commanded vx: {cmd_sum / n:+.3f} m/s")
    print(f"mean actual    vx: {vx_sum / n:+.3f} m/s")
    print(f"swings per foot per second: {(swings.mean() / dur).item():.2f} (target ~1.5 when walking)")
    print(f"envs with >5 swings: {(swings.mean(dim=1) > 5).sum().item()}/{args_cli.num_envs}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
