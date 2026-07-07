# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find a statically-stable standing pose for Qmini and write it into the robot cfg.

Why this exists
---------------
In this project the action term is ``JointPositionActionCfg(..., use_default_offset=True)``,
so a **zero action** commands the PD controller to hold exactly ``init_state.joint_pos``.
"Does the robot stand when I run ``zero_agent.py``?" therefore reduces to one question:

    Is ``init_state.joint_pos`` a statically-stable standing pose, spawned at a height where
    the feet just rest on the ground?

This script answers that question *physically* and, if the answer is no, **searches for a pose
that is** — using the simulator itself as a forward-kinematics engine.

How it works
------------
1. VERIFY: spawn the robot alone on flat ground with its real actuators, command the candidate
   ``joint_pos`` as the PD hold target (identical to a zero action), drop it, and let it settle.
   If it stays upright with both feet planted, great — report the rest height.

2. SEARCH (if the candidate is unstable): pin the base in the air and
   use the sim as forward kinematics. Run a left/right-symmetric coordinate descent over the leg
   joints to satisfy the three conditions of a static stand:
       * feet level (equal height),
       * feet centred under the base (CoM over the support polygon),
       * flat soles, at a target crouch height.
   Then drop the winning pose free-base to confirm it stands and measure the true rest height.

3. REPORT + WRITE: print the resulting ``joint_pos`` dict + spawn ``pos.z`` and patch them
   straight into the robot cfg file.

This tool is Qmini-specific: it always loads ``BDX_R_CFG`` from ``Qmini.robots.qmini`` and
always writes the verified pose back into ``source/Qmini/robots/qmini.py``. Left/right joints are paired by
their ``_l`` / ``_r`` suffix and mirrored with the sign convention found in the existing
default pose.

Examples
--------
Verify the current pose — and, if it is unstable, search for a stable one — then write
the result into the robot cfg::

    python scripts/tune_home_pose.py --headless

Same, but watch it in the viewer::

    python scripts/tune_home_pose.py

"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Find a stable standing pose for Qmini and write it into the robot cfg.")
parser.add_argument("--settle-time", type=float, default=5.0, help="Seconds to let the robot settle (free-base verify).")
parser.add_argument("--clearance", type=float, default=0.02, help="Extra metres above measured rest height for spawn pos.z.")
parser.add_argument("--upright-tol-deg", type=float, default=10.0, help="Max base tilt (deg) still considered standing.")
parser.add_argument(
    "--target-height",
    type=float,
    default=0.38,
    help="Desired base standing height (m). Defaults to the env's base_height_deviation target so "
    "the stand pose matches the height the rewards expect. Guides the search only; "
    "the reported spawn height is measured from the real free-base settle.",
)
parser.add_argument("--sim-dt", type=float, default=0.005, help="Physics timestep (default matches the env: 200 Hz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import re
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat


def load_robot_cfg() -> ArticulationCfg:
    from Qmini.robots.qmini import BDX_R_CFG

    return BDX_R_CFG


def upright_angle_deg(robot: Articulation) -> torch.Tensor:
    """Tilt of the base from vertical (deg); 0 = perfectly upright."""
    g_b = robot.data.projected_gravity_b  # gravity in base frame, unit length; (0,0,-1) when upright
    cos = (-g_b[:, 2]).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def reset_state(robot: Articulation, spawn_z: float, target: torch.Tensor):
    device = robot.device
    root = robot.data.default_root_state.clone()
    root[:, 0:3] = torch.tensor([0.0, 0.0, spawn_z], device=device)
    root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    root[:, 7:13] = 0.0
    robot.write_root_pose_to_sim(root[:, 0:7])
    robot.write_root_velocity_to_sim(root[:, 7:13])
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.reset()


def free_settle(robot: Articulation, sim: SimulationContext, spawn_z: float, target: torch.Tensor) -> dict:
    """Drop the robot free-base, hold `target` with the PD, settle, and average the last 0.5 s."""
    dt = args_cli.sim_dt
    n = max(1, int(args_cli.settle_time / dt))
    win = max(1, int(0.5 / dt))
    reset_state(robot, spawn_z, target)
    sj = torch.zeros_like(target)
    sh = st = 0.0
    vmax = 0.0
    for step in range(n):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if step >= n - win:
            sj += robot.data.joint_pos
            sh += robot.data.root_pos_w[0, 2].item()
            st += upright_angle_deg(robot)[0].item()
            vmax = max(vmax, robot.data.joint_vel.abs().max().item())
    return {"joint_pos": sj / win, "height": sh / win, "tilt_deg": st / win, "max_joint_speed": vmax}


def _wrap(a):
    """Wrap an angle tensor to [-pi, pi]."""
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


def free_eval(robot, sim, masses, foot_ids, target, settle_s: float = 3.0) -> dict:
    """Drop the robot FREE-BASE, hold `target` with the PD, and measure how well it stands.

    This is evaluated under load, so the legs bear the full body weight and sag exactly as they
    will under a zero action — the effect a base-pinned FK pass misses. We track how long the base
    stays upright (the survival signal that gives the search a gradient even among falling poses)
    and, over the TRAILING window, the loaded CoM-over-feet offset, foot levelness, sole flatness,
    and height. Sampling the trailing window (not the back half) means a slow topple shows up as a
    growing CoM offset / rising tilt rather than being averaged away by the stable early seconds.
    """
    dt = args_cli.sim_dt
    n = max(1, int(settle_s / dt))
    sample_from = n - max(1, int(1.0 / dt))  # last ~1.0 s
    spawn_z = args_cli.target_height + 0.02
    fl, fr = foot_ids
    reset_state(robot, spawn_z, target)
    upright = 0
    acc = {"com_dx": 0.0, "com_dy": 0.0, "level": 0.0, "flat": 0.0, "height": 0.0}
    nsamp = 0
    for step in range(n):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        tilt = upright_angle_deg(robot)[0].item()
        if tilt < 30.0:
            upright += 1
            if step >= sample_from:  # sample only the trailing window
                pos = robot.data.body_pos_w[0]
                quat = robot.data.body_quat_w[0]
                com = (masses.view(-1, 1) * pos).sum(0) / masses.sum()
                fc = 0.5 * (pos[fl] + pos[fr])
                rl, pl, _ = euler_xyz_from_quat(quat[fl : fl + 1])
                rr, pr, _ = euler_xyz_from_quat(quat[fr : fr + 1])
                acc["com_dx"] += abs(com[0].item() - fc[0].item())
                acc["com_dy"] += abs(com[1].item() - fc[1].item())
                acc["level"] += abs(pos[fl, 2].item() - pos[fr, 2].item())
                acc["flat"] += float(_wrap(pl).abs() + _wrap(pr).abs() + _wrap(rl).abs() + _wrap(rr).abs())
                acc["height"] += robot.data.root_pos_w[0, 2].item()
                nsamp += 1
    final_tilt = upright_angle_deg(robot)[0].item()
    frac_up = upright / n
    if nsamp == 0:  # never settled upright — return worst-case CoM terms but keep the survival signal
        return {"frac_up": frac_up, "final_tilt": final_tilt, "height": robot.data.root_pos_w[0, 2].item(),
                "com_dx": 1.0, "com_dy": 1.0, "level": 1.0, "flat": 4.0, "samples": 0}
    for k in acc:
        acc[k] /= nsamp
    return {"frac_up": frac_up, "final_tilt": final_tilt, "samples": nsamp, **acc}


def standing_cost(m: dict) -> float:
    """Lower is better. Survival dominates (a pose that stays up beats one that falls); then the
    loaded CoM-over-feet offset, foot levelness/flatness, and a mild crouch-height preference."""
    fell = 1.0 - m["frac_up"]
    height_err = abs(m["height"] - args_cli.target_height)
    return (
        120.0 * fell           # stay upright the whole window
        + 15.0 * m["com_dx"]   # loaded CoM fore/aft offset — the topple axis
        + 8.0 * m["com_dy"]    # lateral CoM offset (≈0 by symmetry)
        + 4.0 * m["level"]     # feet at equal height
        + 1.0 * m["flat"]      # soles horizontal
        + 3.0 * height_err     # prefer the intended height, but never at the cost of stability
        + 0.6 * m["final_tilt"]  # the end state matters most — penalise a slow topple
    )


def build_symmetry(joint_names: list[str], default: torch.Tensor):
    """Map a per-leg parameter vector to a full joint vector using _l/_r pairs and the sign
    convention present in the existing default pose (right = sign * left)."""
    base_to_idx: dict[str, dict[str, int]] = {}
    for i, n in enumerate(joint_names):
        if n.endswith("_l"):
            base_to_idx.setdefault(n[:-2], {})["l"] = i
        elif n.endswith("_r"):
            base_to_idx.setdefault(n[:-2], {})["r"] = i
    params = []  # (base_name, left_idx, right_idx_or_None, right_sign)
    for base, d in base_to_idx.items():
        li = d.get("l")
        ri = d.get("r")
        if li is None:
            continue
        sign = -1.0
        if ri is not None and abs(default[0, li].item()) > 1e-6:
            ratio = default[0, ri].item() / default[0, li].item()
            sign = 1.0 if ratio >= 0 else -1.0
        params.append((base, li, ri, sign))
    return params


def expand(params, p_vals: torch.Tensor, nj: int, device) -> torch.Tensor:
    full = torch.zeros((1, nj), device=device)
    for (base, li, ri, sign), v in zip(params, p_vals.tolist()):
        full[0, li] = v
        if ri is not None:
            full[0, ri] = sign * v
    return full


def search_pose(robot, sim, foot_ids, masses, limits, default) -> torch.Tensor:
    """Pattern search over the leg joints to minimise standing_cost (evaluated free-base, under load).

    Beyond single-joint moves it includes COUPLED moves among the sagittal joints (hip_pitch,
    knee, ankle). Shifting a foot fore/aft while keeping the sole flat needs those joints to move
    together; pure coordinate descent stalls because any single-joint step tips the foot and looks
    worse. The coupled directions let the search translate the stance and bring the CoM over the feet.
    """
    device = robot.device
    nj = len(robot.joint_names)
    # legs are always optimised as left/right-symmetric pairs (see build_symmetry)
    params = build_symmetry(list(robot.joint_names), default)
    p_vals = torch.tensor([default[0, li].item() for (_, li, _, _) in params], device=device)
    lo = torch.tensor([limits[li, 0].item() for (_, li, _, _) in params], device=device)
    hi = torch.tensor([limits[li, 1].item() for (_, li, _, _) in params], device=device)
    to_full = lambda pv: expand(params, pv, nj, device)
    labels = [b for (b, _, _, _) in params]

    npar = len(p_vals)
    sag = [i for i, l in enumerate(labels) if "pitch" in l]  # hip_pitch, knee_pitch, ankle_pitch
    directions = []
    for i in range(npar):  # single-joint moves
        e = torch.zeros(npar, device=device)
        e[i] = 1.0
        directions.append(e)
    for a in range(len(sag)):  # coupled sagittal pairs (both sign combos)
        for b in range(a + 1, len(sag)):
            for s in (1.0, -1.0):
                e = torch.zeros(npar, device=device)
                e[sag[a]] = 1.0
                e[sag[b]] = s
                directions.append(e)

    def cost_of(pv):
        return standing_cost(free_eval(robot, sim, masses, foot_ids, to_full(pv)))

    best_cost = cost_of(p_vals)
    print(f"\n[search] start cost={best_cost:.3f}  params={[round(x,3) for x in p_vals.tolist()]} ({labels})")
    print(f"[search] {len(directions)} search directions ({npar} single + coupled sagittal)")
    evals = 1
    for delta in (0.25, 0.12, 0.06, 0.03, 0.015):
        improved = True
        while improved:
            improved = False
            for d in directions:
                for s in (delta, -delta):
                    trial = torch.clamp(p_vals + s * d, lo, hi)
                    if torch.allclose(trial, p_vals):
                        continue
                    c = cost_of(trial)
                    evals += 1
                    if c < best_cost - 1e-2:
                        best_cost, p_vals, improved = c, trial, True
            print(f"[search] delta={delta:.2f}  cost={best_cost:.3f}  params={[round(x,3) for x in p_vals.tolist()]}")
    print(f"[search] done after {evals} evaluations, final cost={best_cost:.3f}")
    return to_full(p_vals)


def fmt_joint_dict(joint_names, values) -> str:
    width = max(len(n) for n in joint_names) + 3
    lines = []
    for n, v in zip(joint_names, values.tolist()):
        key = f'"{n}":'
        lines.append(f"            {key:<{width}} {v: .4f},")
    return "\n".join(lines)


def patch_source_file(path: Path, joint_names, values, spawn_z: float):
    text = path.read_text()
    original = text
    for name, val in zip(joint_names, values.tolist()):
        pat = re.compile(rf'("{re.escape(name)}"\s*:\s*)(-?\d+\.?\d*)')
        if not pat.search(text):
            print(f"  [warn] joint '{name}' not found in {path.name}; skipped")
            continue
        text = pat.sub(lambda m, v=val: f"{m.group(1)}{v:.4f}", text, count=1)
    pos_pat = re.compile(r"(pos\s*=\s*\(\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*)([-\d.]+)(\s*\))")
    if pos_pat.search(text):
        text = pos_pat.sub(lambda m: f"{m.group(1)}{spawn_z:.4f}{m.group(3)}", text, count=1)
    else:
        print(f"  [warn] no 'pos=(x, y, z)' tuple found in {path.name}; spawn height not patched")
    if text == original:
        print(f"  [warn] nothing changed in {path.name}")
        return
    path.write_text(text)
    print(f"  [ok] patched {path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, 1.8, 1.0], target=[0.0, 0.0, 0.3])
    # match the env's ground friction (~1.0) so contact behaviour is representative
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
    )
    ground.func("/World/ground", ground)
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    robot_cfg = load_robot_cfg()
    spawn_z = float(robot_cfg.init_state.pos[2])
    robot = Articulation(robot_cfg.replace(prim_path="/World/Robot"))
    sim.reset()

    joint_names = list(robot.joint_names)
    print("\n[INFO] joint order:", joint_names)

    # locate the foot bodies (leaf links of each leg) for the FK objective
    # accept both bare (`ankle_pitch_l`) and ROS-style (`ankle_pitch_l_link`) leaf-link names
    foot_l = robot.body_names.index([b for b in robot.body_names if b.startswith("ankle") and re.search(r"_l(_link)?$", b)][0])
    foot_r = robot.body_names.index([b for b in robot.body_names if b.startswith("ankle") and re.search(r"_r(_link)?$", b)][0])
    foot_ids = [foot_l, foot_r]
    limits = robot.data.joint_pos_limits[0]  # (nj, 2)
    default = robot.data.default_joint_pos.clone()
    masses = robot.root_physx_view.get_masses()[0].to(robot.device)  # (nb,) per-body mass for CoM

    # ---- 1. verify the current pose -----------------------------------------
    target = default.clone()
    res = free_settle(robot, sim, spawn_z, target)
    stable = res["tilt_deg"] < args_cli.upright_tol_deg
    print(
        f"\n[verify current] spawn_z={spawn_z:.3f} -> rest_height={res['height']:.3f} m  "
        f"tilt={res['tilt_deg']:.1f} deg  {'STABLE ✓' if stable else 'UNSTABLE ✗ (tips over)'}"
    )

    # ---- 2. search if needed ------------------------------------------------
    if not stable:
        print("\n[INFO] current pose is unstable — searching for a stable standing pose ...")
        target = search_pose(robot, sim, foot_ids, masses, limits, default)
        # full-length free-base verify from just above the target crouch height (gentle landing)
        res = free_settle(robot, sim, args_cli.target_height + 0.02, target)
        stable = res["tilt_deg"] < args_cli.upright_tol_deg
        print(
            f"\n[verify search] rest_height={res['height']:.3f} m  tilt={res['tilt_deg']:.1f} deg  "
            f"residual_speed={res['max_joint_speed']:.3f} rad/s  "
            f"{'STABLE ✓' if stable else 'STILL UNSTABLE ✗'}"
        )

    if not stable:
        print("\n[RESULT] Could not find a statically-stable standing pose. Try a larger --target-height,")
        print("         check the URDF joint signs/limits, or relax --upright-tol-deg.")
        return

    final = target[0]
    final_spawn_z = round(res["height"] + args_cli.clearance, 4)
    print("\n" + "=" * 78)
    print("STABLE STANDING POSE FOUND")
    print("=" * 78)
    print(f"\nSuggested spawn height:  pos=(0.0, 0.0, {final_spawn_z})")
    print("\nSuggested joint_pos (ready to paste into init_state):\n")
    print("        joint_pos={")
    print(fmt_joint_dict(joint_names, final))
    print("        },")

    # the verified pose is always written into the Qmini robot cfg
    src = Path(__file__).resolve().parents[1] / "source/Qmini/robots/qmini.py"
    if not src.exists():
        print(f"\n[ERROR] robot cfg not found: {src}")
    else:
        print(f"\n[WRITE] patching {src} ...")
        patch_source_file(src, joint_names, final, final_spawn_z)


if __name__ == "__main__":
    main()
    simulation_app.close()
