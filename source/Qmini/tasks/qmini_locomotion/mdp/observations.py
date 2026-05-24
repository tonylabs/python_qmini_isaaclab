# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Qmini bipedal locomotion task.

Adds a gait-phase clock signal and a static-flag indicator. These let the
policy learn coherent swing/stance scheduling and a distinct standing mode —
both features were present in the RoboTamer4Qmini (Isaac Gym) reference and
were missing from the original Isaac Lab port.

Stateless: phase is derived from ``env.episode_length_buf`` so no per-env
counter has to be maintained across resets.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _phase_per_leg(env: "ManagerBasedRLEnv", frequency: float) -> torch.Tensor:
    """Return per-env, per-leg phase ∈ [0, 1) as a tensor of shape (num_envs, 2).

    Left leg phase advances from 0 each reset; right leg is offset by 0.5
    (legs in anti-phase, the natural pattern for a steady-state walking gait).
    """
    # step_dt is the policy-step interval (sim.dt * decimation) — 0.02 s by default
    step_dt = env.step_dt
    t = env.episode_length_buf.float() * step_dt  # (num_envs,)
    phase_l = (t * frequency) % 1.0
    phase_r = (phase_l + 0.5) % 1.0
    return torch.stack([phase_l, phase_r], dim=1)  # (num_envs, 2)


def gait_phase_sincos(env: "ManagerBasedRLEnv", frequency: float = 1.5) -> torch.Tensor:
    """Return ``[sin(2π φ_L), cos(2π φ_L), sin(2π φ_R), cos(2π φ_R)]`` per env.

    Args:
        env: the Isaac Lab manager-based RL env.
        frequency: gait clock frequency in Hz. 1.5 Hz ≈ comfortable bipedal cadence
            and matches the RoboTamer4Qmini reference framework's typical
            policy-output frequency for this robot. Adjust by retraining if the
            real-robot gait looks wrong.

    Returns:
        Tensor of shape (num_envs, 4).
    """
    phase = _phase_per_leg(env, frequency)  # (num_envs, 2)
    two_pi_phase = 2.0 * math.pi * phase
    return torch.cat([torch.sin(two_pi_phase), torch.cos(two_pi_phase)], dim=1)


def static_flag(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", threshold: float = 0.15) -> torch.Tensor:
    """Return 1.0 when the commanded velocity is below ``threshold`` (standing), else 0.0.

    Per-env tensor of shape ``(num_envs, 1)``. Used to let the policy condition
    on "I'm being told to stand" vs "I'm being told to walk", and to gate
    reward terms appropriately.

    Threshold matches the RoboTamer4Qmini reference (`norm(cmd) < 0.15`).
    """
    cmd = env.command_manager.get_command(command_name)
    norm = torch.linalg.norm(cmd, dim=1, keepdim=True)
    return (norm < threshold).float()
