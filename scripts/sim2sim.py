"""Sim2sim: run the Isaac-Lab-trained Qmini ONNX policy inside MuJoCo.

Loads ``<repo>/policy.onnx`` (exported by ``scripts/export_onnx.py`` — always the
project root, see CLAUDE.md) and drives the qmini_description MJCF with the same
observation layout, per-term scales, default joint pose, action scale, control
rate, and PD gains used by the ``qmini-velocity`` training task.

Single source of truth for every constant below:
    - JOINT_ORDER (Isaac BFS)            ← printed by scripts/rsl_rl/play.py
    - KP/KD, TAU_LIMIT, ARMATURE         ← source/Qmini/robots/qmini.py (BDX_R_CFG)
    - DEFAULT_JOINT_POS                  ← BDX_R_CFG.init_state.joint_pos (same file)
    - ACTION_SCALE                       ← qmini_env_cfg.ActionsCfg.joint_pos.scale (= 0.5)
    - Obs order + per-term scales        ← qmini_env_cfg.ObservationsCfg.PolicyCfg
      (44 dims, single step, no history)
    - Gait clock / static flag           ← Qmini mdp/observations.py
      (1.5 Hz, right leg +0.5 phase; static threshold 0.15 on the command norm)
    - SIM_DT, DECIMATION                 ← LocomotionVelocityRoughEnvCfg defaults
                                           (sim.dt = 1/200, decimation 4 → 50 Hz)

Quick start:

    # 1. Export the trained checkpoint to ONNX (writes <repo>/policy.onnx):
    python scripts/export_onnx.py <date-folder>
    # 2. Run the policy in MuJoCo (always opens the viewer):
    python scripts/sim2sim.py --vx 0.4
    python scripts/sim2sim.py --vx 0.4 --duration 20   # fixed-length run

Asset paths default to the in-repo submodule (override with --mjcf / --onnx):
    - MJCF : source/Qmini/assets/qmini/mjcf/world.xml
    - ONNX : <repo>/policy.onnx

MJCF contract (world.xml already satisfies all of this):
    1. A free joint on the body named ``base_link`` (or pass ``--torso``).
    2. One ``<motor name="...">`` per leg joint, names matching JOINT_ORDER.
       The script computes τ from a Python-side PD loop and writes it into
       ``data.ctrl`` — do NOT use ``<position>`` actuators, the ctrl semantics
       differ. (ctrlrange is aligned with TAU_LIMIT at load.)
    3. Per-joint ``armature`` matching ARMATURE below (already in the MJCF).
    4. An ``imu`` site with a ``<gyro name="imu_gyro">`` — Qmini's IMU mount is
       identity rotation w.r.t. base_link (translation-only offset), so site and
       base share orientation.

Command deadband (the joystick contract):
    if the command norm is below 0.15 the whole command is zeroed and the
    static_flag observation is 1.0 — exactly how training represents "stand".

Keyboard control (viewer mode only; WASD is reserved by the MuJoCo viewer):
    * ``UP`` / ``DOWN``    — vx ±0.1 m/s;  ``LEFT`` / ``RIGHT`` — vy ±0.1 m/s
    * ``Q`` / ``E``        — target heading ±15° (drives wz via the heading
      controller, exactly as during training with heading_command=True)
    * ``0``                — zero vx and vy (stand);  ``R`` — reset to CLI values

Debugging:
    * ``--dump-imu HZ`` — print imu.read() at the requested rate (max 50 Hz).
    * ``--duration``    — wall-clock seconds; 0 (default) = run until Ctrl-C /
      the viewer window is closed.
    * IMU axes overlay — X red / Y green / Z blue, anchored at the IMU site.
      Bright, longer arrows = the frame perceived through ``imu.read()``;
      dim, shorter arrows = the MuJoCo ``imu`` site frame (ground truth).
      They coincide when the IMU obs frame matches training.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

import sys

sys.path.append(str(Path(__file__).resolve().parent))
from imu import DEFAULT_BAUD, DEFAULT_PORT, Imu, MujocoImu  # noqa: E402

# Project root (<repo>/scripts/sim2sim.py -> parents[1]).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = _PROJECT_ROOT / "source/Qmini/assets/qmini/mjcf/world.xml"
DEFAULT_ONNX = _PROJECT_ROOT / "policy.onnx"

# Isaac Lab LocomotionVelocityRoughEnvCfg defaults (qmini_env_cfg inherits them):
# physics 200 Hz, decimation 4 → policy 50 Hz.
SIM_DT = 1.0 / 200.0
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION  # 1/50 s

ACTION_SCALE = 0.5  # qmini_env_cfg.ActionsCfg.joint_pos.scale

# MUST match Isaac Lab's articulation joint order — BFS pairs joints by family
# (L,R per joint type), NOT by leg. Verified by scripts/rsl_rl/play.py's
# ">>> POLICY JOINT ORDER" printout. MJCF joint/actuator names are identical.
JOINT_ORDER = [
    "hip_yaw_l",     "hip_yaw_r",
    "hip_roll_l",    "hip_roll_r",
    "hip_pitch_l",   "hip_pitch_r",
    "knee_pitch_l",  "knee_pitch_r",
    "ankle_pitch_l", "ankle_pitch_r",
]

# Default joint positions (BDX_R_CFG.init_state.joint_pos): the bent-knee stand
# tuned by scripts/tune_home_pose.py (statically stable under pure PD; the slight
# toe-in hip_yaw widens the support polygon). Right-side joints are the sign-mirror
# of the left (mirrored joint axes). This is BOTH the JointPositionAction offset
# (use_default_offset=True) and the joint_pos_rel reference, so it must match the
# robot cfg exactly — re-sync whenever tune_home_pose.py rewrites qmini.py.
DEFAULT_JOINT_POS = {
    "hip_yaw_l": -0.115,   "hip_yaw_r": 0.115,
    "hip_roll_l": 0.015,   "hip_roll_r": -0.015,
    "hip_pitch_l": 0.25,   "hip_pitch_r": -0.25,
    "knee_pitch_l": 0.4,   "knee_pitch_r": -0.4,
    "ankle_pitch_l": 0.18, "ankle_pitch_r": -0.18,
}

Q_DEFAULT = np.array([DEFAULT_JOINT_POS[n] for n in JOINT_ORDER], dtype=np.float32)

# Rotor-side kp/kd × reduction² (6.33² = 40.07; hip_roll has an
# extra 3:1 gear → 360.62). Order matches JOINT_ORDER (L/R pair per family:
# hip_yaw, hip_roll, hip_pitch, knee, ankle).
KP = np.array([40.07] * 2 + [180.31] * 2 + [60.10] * 2 + [40.07] * 2 + [40.07] * 2, dtype=np.float32)
KD = np.array([2.0] * 2 + [18.03] * 2 + [2.0] * 2 + [2.0] * 2 + [2.0] * 2, dtype=np.float32)

# Peak torque (effort_limit_sim): 20 N·m everywhere except hip_roll 60 N·m.
# The MJCF motor ctrlrange already matches; the load-time override below just
# guarantees it stays aligned if the MJCF changes.
TAU_LIMIT = np.array([20.0] * 2 + [60.0] * 2 + [20.0] * 2 + [20.0] * 2 + [20.0] * 2, dtype=np.float32)

# Reflected rotor inertia (BDX_R_CFG armature = I_rotor × reduction²). All joints
# are Unitree GO-M8010-6 motors with a 6.33:1 internal reduction (→ 0.02); the
# hip_roll joints have an extra 3:1 external gear (qmini_official_sdk), so 3²·0.02
# = 0.18. Applied onto the MuJoCo model at load — the submodule MJCF still ships
# the older, physically inconsistent values.
ARMATURE = np.array([0.02] * 2 + [0.18] * 2 + [0.02] * 6, dtype=np.float32)

# Observation scales (ObsTerm `scale=` kwargs in qmini_env_cfg.PolicyCfg).
SCALE_IMU_ANG_VEL = 0.2
SCALE_JOINT_VEL = 0.05

# Gait clock + static flag (Qmini mdp/observations.py):
#   gait_phase_sincos: [sin 2πφL, sin 2πφR, cos 2πφL, cos 2πφR], φR = φL + 0.5,
#   1.5 Hz, phase from policy-step count since episode start — NOT gated.
#   static_flag: 1.0 when ||cmd|| < 0.15 (the stand/walk switch).
GAIT_FREQ = 1.5
GAIT_PHASE_OFFSET = 0.5
STATIC_CMD_THRESHOLD = 0.15

# Heading controller (qmini_env_cfg CommandsCfg: heading_command=True,
# heading_control_stiffness=0.5, ang_vel_z range ±1.0). Training recomputes wz
# from the heading error every step; sim2sim must do the same or cmd[2] would
# be an observation distribution the policy never saw.
HEADING_STIFFNESS = 0.5
ANG_VEL_Z_MIN = -1.0
ANG_VEL_Z_MAX = 1.0

# Single-step observation dim: 3+3+10+10+10+3+4+1 = 44 (no history stacking).
OBS_DIM = 44


# ---------------------------------------------------------------------------
def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _build_obs(
    ang_vel_imu: np.ndarray,
    proj_grav_imu: np.ndarray,
    q_rel: np.ndarray,
    qd: np.ndarray,
    last_action: np.ndarray,
    cmd: np.ndarray,
    phase_sincos: np.ndarray,
    static_flag: float,
) -> np.ndarray:
    """One 44-dim observation in PolicyCfg term order + scales.

    Term order mirrors ``qmini_env_cfg.ObservationsCfg.PolicyCfg``:
    imu_ang_vel(0.2), imu_projected_gravity, joint_pos_rel, joint_vel(0.05),
    actions, velocity_commands, gait_phase_sincos(4), static_flag(1).
    """
    obs = np.concatenate([
        SCALE_IMU_ANG_VEL * ang_vel_imu,        # 3
        proj_grav_imu,                          # 3
        q_rel,                                  # 10 (q - q_default)
        SCALE_JOINT_VEL * qd,                   # 10
        last_action,                            # 10 (raw, pre-scale)
        cmd,                                    # 3 (vx, vy, wz)
        phase_sincos,                           # 4
        np.array([static_flag], np.float32),    # 1
    ]).astype(np.float32)
    assert obs.shape == (OBS_DIM,), obs.shape
    return obs[None, :]


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qmini Isaac-Lab ONNX policy in MuJoCo.")
    parser.add_argument("--vx", type=float, default=0.4, help="Forward velocity command (m/s).")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s).")
    parser.add_argument(
        "--heading",
        type=float,
        default=0.0,
        help=(
            "Target heading (rad, world frame). Training uses heading_command=True, so wz is "
            "recomputed each step from this target. Default 0.0 = face world +X."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Wall-clock seconds to run. 0 (default) = until Ctrl-C / viewer close.",
    )
    parser.add_argument("--torso", default="base_link", help="MJCF body name of the torso/IMU mount.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MJCF world to load.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Policy ONNX (default: <repo>/policy.onnx).")
    parser.add_argument("--dump-imu", type=float, default=0.0, metavar="HZ",
                        help="Print imu.read() at this rate in Hz (0 = off, max 50).")
    parser.add_argument("--imu", choices=("sim", "hardware"), default="sim",
                        help="IMU source: MuJoCo-simulated (default) or real IMU over serial.")
    parser.add_argument("--imu-port", default=DEFAULT_PORT, help="Serial port for --imu hardware.")
    parser.add_argument("--imu-baud", type=int, default=DEFAULT_BAUD, help="Baud rate for --imu hardware.")
    args = parser.parse_args()

    if not Path(args.mjcf).exists():
        raise SystemExit(f"MJCF not found: {args.mjcf}")
    if not Path(args.onnx).exists():
        raise SystemExit(
            f"ONNX policy not found: {args.onnx}\n"
            "Export it first (writes <repo>/policy.onnx):\n"
            "  python scripts/export_onnx.py <date-folder-under-logs/rsl_rl/qmini>"
        )
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    # Resolve joint qpos/qvel addresses and actuator ids in policy order.
    joint_qadr = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_ORDER]
    )
    joint_vadr = np.array(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_ORDER]
    )
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_ORDER]
    )
    if (actuator_ids < 0).any():
        missing = [n for n, i in zip(JOINT_ORDER, actuator_ids) if i < 0]
        raise RuntimeError(f"MJCF is missing actuators for: {missing}")

    # Keep MuJoCo's motor ctrlrange aligned with the training effort limits so the
    # motors never clip torque tighter than the policy was trained with.
    for aid, tau in zip(actuator_ids, TAU_LIMIT):
        model.actuator_ctrlrange[aid] = (-float(tau), float(tau))

    # Align joint armature with training (the submodule MJCF predates the
    # gear-corrected values: hip_roll 0.18 from its extra 3:1 gear, ankle 0.02).
    for vadr, arm in zip(joint_vadr, ARMATURE):
        model.dof_armature[vadr] = float(arm)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.torso)
    if torso_id < 0:
        raise RuntimeError(f"Body {args.torso!r} not found in MJCF; pass --torso <body_name>.")

    # Foot bodies — used by the contact diagnostic below.
    foot_body_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("ankle_pitch_l_link", "ankle_pitch_r_link")
        ]
    )
    if (foot_body_ids < 0).any():
        raise RuntimeError("MJCF is missing ankle_pitch_l_link / ankle_pitch_r_link foot bodies.")

    # IMU site — anchor + reference frame for the axes overlay.
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")

    # Start from the bent-knee default pose AT REST HEIGHT, policy engaged from
    # step 0 — exactly how training resets work. Do NOT freeze-and-drop instead:
    # Qmini cannot passively hold the crouch under pure PD (it topples before
    # touchdown), which is also why there is no "wait for landing" gate here.
    free_qadr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")])
    # tune_home_pose.py measured rest height 0.398 (it writes rest+0.02 = 0.418
    # into qmini.py as Isaac's spawn); here we spawn at the rest height directly
    SPAWN_Z = 0.398
    data.qpos[joint_qadr] = Q_DEFAULT
    data.qpos[free_qadr + 2] = SPAWN_Z
    mujoco.mj_forward(model, data)  # make xpos/site/sensor data valid for the first obs

    if args.imu == "hardware":
        print(f"[INFO] Opening hardware IMU on {args.imu_port} @ {args.imu_baud} baud …")
        imu = Imu(port=args.imu_port, baudrate=args.imu_baud)
    else:
        imu = MujocoImu(model, data, body_name=args.torso)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    last_action = np.zeros(len(JOINT_ORDER), dtype=np.float32)
    target_heading = float(args.heading)
    cmd = np.array([args.vx, args.vy, 0.0], dtype=np.float32)  # cmd[2] recomputed per step
    gait_step = 0  # policy-step counter for the gait clock

    # Keyboard control (viewer mode) — clamped to the training command ranges.
    KEY_VX_STEP, KEY_VY_STEP = 0.1, 0.1
    KEY_HEADING_STEP = math.radians(15.0)
    KEY_VX_RANGE = (-0.4, 0.7)  # qmini_env_cfg __post_init__ command ranges
    KEY_VY_RANGE = (-0.4, 0.4)
    _K_LEFT, _K_RIGHT, _K_UP, _K_DOWN = 263, 262, 265, 264
    _K_Q, _K_E, _K_R = ord("Q"), ord("E"), ord("R")
    _K_ZERO = ord("0")

    def _key_callback(keycode: int) -> None:
        nonlocal target_heading
        changed = True
        if keycode == _K_UP:
            cmd[0] = float(np.clip(cmd[0] + KEY_VX_STEP, *KEY_VX_RANGE))
        elif keycode == _K_DOWN:
            cmd[0] = float(np.clip(cmd[0] - KEY_VX_STEP, *KEY_VX_RANGE))
        elif keycode == _K_LEFT:
            cmd[1] = float(np.clip(cmd[1] + KEY_VY_STEP, *KEY_VY_RANGE))
        elif keycode == _K_RIGHT:
            cmd[1] = float(np.clip(cmd[1] - KEY_VY_STEP, *KEY_VY_RANGE))
        elif keycode == _K_Q:
            target_heading += KEY_HEADING_STEP
        elif keycode == _K_E:
            target_heading -= KEY_HEADING_STEP
        elif keycode == _K_ZERO:
            cmd[0] = 0.0
            cmd[1] = 0.0
        elif keycode == _K_R:
            cmd[0], cmd[1] = float(args.vx), float(args.vy)
            target_heading = float(args.heading)
        else:
            changed = False
        if changed:
            print(f"[KEY] vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} heading={target_heading:+.2f} rad")

    # Contact diagnostic — training disables self-collisions, so the only expected
    # contacts are foot↔floor and foot↔foot. Anything else means the MJCF collision
    # setup diverges from training (or the robot fell). Printed at most 1 Hz.
    foot_body_set = set(int(b) for b in foot_body_ids)
    last_contact_print_t = [-1.0]

    def _check_contacts() -> None:
        if data.time - last_contact_print_t[0] < 1.0:
            return
        bad: list[tuple[str, str]] = []
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            foot1, foot2 = b1 in foot_body_set, b2 in foot_body_set
            world1, world2 = b1 == 0, b2 == 0
            if (foot1 and world2) or (foot2 and world1) or (foot1 and foot2):
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"b{b1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"b{b2}"
            bad.append((n1, n2))
        if bad:
            print(f"[CONTACT t={data.time:.2f}] unexpected pairs ({len(bad)}): {sorted(set(bad))}")
        last_contact_print_t[0] = data.time

    dump_every = 0
    if args.dump_imu > 0.0:
        dump_every = max(1, int(round((1.0 / POLICY_DT) / args.dump_imu)))
    imu_print_counter = [0]

    def _maybe_print_imu(ang_vel, proj_g, yaw, lin_acc) -> None:
        if dump_every == 0:
            return
        imu_print_counter[0] += 1
        if imu_print_counter[0] % dump_every != 0:
            return
        print(
            f"[IMU t={data.time:6.3f}s] ang_vel=[{ang_vel[0]:+.4f},{ang_vel[1]:+.4f},{ang_vel[2]:+.4f}] "
            f"proj_g=[{proj_g[0]:+.4f},{proj_g[1]:+.4f},{proj_g[2]:+.4f}] "
            f"lin_acc=[{lin_acc[0]:+.3f},{lin_acc[1]:+.3f},{lin_acc[2]:+.3f}] yaw={yaw:+.4f}"
        )

    def _step_once() -> None:
        nonlocal last_action, gait_step

        ang_vel_imu, proj_grav_imu, base_yaw, lin_acc_imu = imu.read()
        _maybe_print_imu(ang_vel_imu, proj_grav_imu, base_yaw, lin_acc_imu)
        q_rel = (data.qpos[joint_qadr] - Q_DEFAULT).astype(np.float32)
        qd = data.qvel[joint_vadr].astype(np.float32)

        # Heading-driven wz (heading_command=True during training), clipped to the
        # training ang_vel_z range ±1.0.
        heading_err = _wrap_to_pi(target_heading - base_yaw)
        cmd[2] = float(np.clip(HEADING_STIFFNESS * heading_err, ANG_VEL_Z_MIN, ANG_VEL_Z_MAX))

        # Command deadband: below the static threshold the whole command is zero
        # and static_flag = 1 — this is the trained joystick contract (zero
        # joystick → stand). Matches training where standing envs hold cmd = 0.
        cmd_step = cmd.copy()
        if float(np.linalg.norm(cmd_step)) < STATIC_CMD_THRESHOLD:
            cmd_step[:] = 0.0
            static_flag = 1.0
        else:
            static_flag = 0.0

        # Gait clock (always running, matching training's episode-time clock):
        # [sin 2πφL, sin 2πφR, cos 2πφL, cos 2πφR], φR = φL + 0.5.
        phi_l = math.fmod(gait_step * POLICY_DT * GAIT_FREQ, 1.0)
        phi_r = math.fmod(phi_l + GAIT_PHASE_OFFSET, 1.0)
        phase_sincos = np.array(
            [
                math.sin(2.0 * math.pi * phi_l),
                math.sin(2.0 * math.pi * phi_r),
                math.cos(2.0 * math.pi * phi_l),
                math.cos(2.0 * math.pi * phi_r),
            ],
            dtype=np.float32,
        )
        gait_step += 1

        obs = _build_obs(ang_vel_imu, proj_grav_imu, q_rel, qd, last_action, cmd_step, phase_sincos, static_flag)
        action = sess.run(["action"], {"obs": obs})[0][0].astype(np.float32)
        last_action = action

        # Recompute the PD torque every sim substep (matching Isaac's actuator),
        # not once per policy tick — zero-order-held velocity feedback acts as
        # negative damping on the low-inertia ankles.
        q_target = Q_DEFAULT + ACTION_SCALE * action
        for _ in range(DECIMATION):
            tau = KP * (q_target - data.qpos[joint_qadr]) - KD * data.qvel[joint_vadr]
            data.ctrl[actuator_ids] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
            mujoco.mj_step(model, data)

    # ~1 Hz telemetry (posture / progress). "fallen" mirrors the training
    # terminations: base z < 0.15 m or tilt > ~0.9 rad.
    last_telem_t = [-1.0]

    def _telemetry() -> None:
        if data.time - last_telem_t[0] < 1.0:
            return
        last_telem_t[0] = data.time
        bx, by, bz = (float(v) for v in data.xpos[torso_id])
        _, proj_g, yaw, _ = imu.read()
        gz = float(np.clip(-proj_g[2], -1.0, 1.0))
        tilt = math.acos(gz)
        fallen = bz < 0.15 or tilt > 0.9
        print(
            f"[TELEM t={data.time:5.2f}s] base=({bx:+.2f},{by:+.2f},{bz:.3f})m "
            f"yaw={yaw:+.2f} tilt={math.degrees(tilt):4.1f}° "
            f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})"
            + ("  <<< FALLEN" if fallen else "")
        )

    run_forever = args.duration <= 0.0
    if run_forever:
        print("[INFO] --duration <= 0: running indefinitely (Ctrl-C to stop).")

    # ------------------------------------------------------------------
    # IMU axes overlay: two frames anchored at the IMU site, standard colors
    # X = red, Y = green, Z = blue.
    #   * bright R/G/B (longer)  — the IMU frame as perceived through imu.read()
    #                              (projected gravity + yaw), what the POLICY sees
    #   * dim R/G/B (shorter)    — the MuJoCo `imu` site frame (ground-truth mount)
    # They coincide when the IMU obs frame matches training; with a hardware
    # IMU (--imu hardware) a mismatch exposes a wrong mount orientation.
    # ------------------------------------------------------------------
    AXIS_WIDTH = 0.008
    MJ_AXIS_LEN = 0.15
    IMU_AXIS_LEN = 0.19
    IMU_AXIS_COLORS = np.array(  # X red, Y green, Z blue
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32
    )
    MJ_AXIS_COLORS = IMU_AXIS_COLORS * np.array([[0.45, 0.45, 0.45, 1.0]], dtype=np.float32)

    def _imu_rot_world_from_read(proj_g: np.ndarray, yaw: float) -> np.ndarray:
        """R_world_from_imu from (projected_gravity_imu, yaw): ZYX yaw-pitch-roll."""
        g = np.asarray(proj_g, dtype=np.float64)
        n = float(np.linalg.norm(g))
        if not np.isfinite(n) or n < 1e-6:
            return np.eye(3)
        g = g / n
        roll = math.atan2(-g[1], -g[2])
        pitch = math.atan2(g[0], math.sqrt(g[1] ** 2 + g[2] ** 2))
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return rz @ ry @ rx

    def _draw_arrow(scn, origin: np.ndarray, end: np.ndarray, rgba: np.ndarray) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, AXIS_WIDTH, origin, end)
        scn.ngeom += 1

    def _draw_imu_axes(viewer) -> None:
        scn = viewer.user_scn
        if scn is None:
            return
        scn.ngeom = 0
        if imu_site_id >= 0:
            origin = np.asarray(data.site_xpos[imu_site_id], dtype=np.float64)
            r_site = np.asarray(data.site_xmat[imu_site_id], dtype=np.float64).reshape(3, 3)
        else:
            origin = np.asarray(data.xpos[torso_id], dtype=np.float64)
            r_site = np.asarray(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
        # ground-truth IMU site frame (R/G/B)
        for i in range(3):
            _draw_arrow(scn, origin, origin + r_site[:, i] * MJ_AXIS_LEN, MJ_AXIS_COLORS[i])
        # frame as perceived through imu.read() (orange/cyan/magenta)
        _, proj_g_dbg, yaw_dbg, _ = imu.read()
        r_imu = _imu_rot_world_from_read(proj_g_dbg, yaw_dbg)
        for i in range(3):
            _draw_arrow(scn, origin, origin + r_imu[:, i] * IMU_AXIS_LEN, IMU_AXIS_COLORS[i])

    print(
        "[INFO] Keyboard control (arrow keys — WASD is reserved by MuJoCo viewer):\n"
        "         UP / DOWN    — vx ±0.1 m/s\n"
        "         LEFT / RIGHT — vy ±0.1 m/s\n"
        "         Q / E        — heading ±15°\n"
        "         0            — stop (zero vx, vy) → robot stands\n"
        "         R            — reset cmd + heading to CLI values\n"
        "[INFO] IMU axes — X red / Y green / Z blue. Bright=perceived (imu.read), dim=site frame."
    )
    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data, key_callback=_key_callback)
    except RuntimeError as e:
        if "mjpython" not in str(e):
            raise
        raise SystemExit(
            "macOS GUI viewer requires `mjpython`. Run:\n"
            f"    mjpython {' '.join(sys.argv)}"
        ) from None

    try:
        with viewer_ctx as viewer:
            t_end = None if run_forever else time.time() + args.duration
            while viewer.is_running() and (t_end is None or time.time() < t_end):
                t0 = time.time()
                _step_once()
                _check_contacts()
                _telemetry()
                _draw_imu_axes(viewer)
                viewer.sync()
                sleep = POLICY_DT - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print(f"[OK] Stopped on Ctrl-C; sim time {data.time:.2f}s.")
    finally:
        imu.close()


if __name__ == "__main__":
    main()
