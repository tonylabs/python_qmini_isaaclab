# Qmini Bipedal Robotic Locomotion

## 🎯 Current Focus: Walking and Sim2Real

The project is currently in its initial phase, with the core focus on mastering bipedal locomotion. The immediate objectives are:

-   **Achieve Stable Walking:** Train a robust walking policy using reinforcement learning.
-   **Cross the Sim2Real Gap:** Successfully transfer the policy trained in a simulated environment to the physical robot.

*At this stage, the project is concentrated on the fundamental mechanics of the body's movement. Expressiveness and the integration of a head are future goals to be explored after mastering stable locomotion.*

---

## 🛠️ Hardware

The Qmini is built with a focus on high-performance components that are accessible to the robotics community. The entire build is being developed with a target budget under **$3,000**.

-   **Unitree GO-M8010-6 Motors:** These motors provide the necessary torque and precision for dynamic and controlled leg movements.
-   **NVIDIA Jetson Orin Nano:** Serving as the onboard computer, the Jetson Orin Nano has the computational power required to run the trained RL policy in real-time.

---

## 🤖 Software and Training: Reinforcement Learning with Isaac Lab

The robot's ability to walk is being developed through reinforcement learning within the **NVIDIA Isaac Lab** simulation environment.

A policy is trained in this virtual space, allowing the Qmini to learn and adapt its movements to maintain balance and achieve forward motion. This process is critical for developing a robust control system before deploying it on the physical hardware.

---

## 🚀 Installation

To install the necessary packages for this project, after cloning the repo, run the following command:

```bash
cd python_qmini_isaaclab
python -m pip install -e source/Qmini
```

```bash
python scripts/rsl_rl/train.py --task=qmini-velocity --headless

# resume training
python scripts/rsl_rl/train.py --task=qmini-velocity --headless --resume --load_run <date> --checkpoint model_*.pt
```

```bash
python scripts/rsl_rl/play.py --task=qmini-velocity-play --num_envs 100
```

---

## 🧪 Sanity-Checking the Robot with `zero_agent.py`

Before training, it's worth verifying that the robot, URDF, actuator gains, and initial pose are all wired up correctly. The `zero_agent.py` script does exactly that: it launches the environment and pushes a **zero action vector** every step, so the only forces acting on the robot are gravity and the PD controller pulling each joint toward its `default_joint_pos` (i.e. the crouch pose defined in `source/Qmini/robots/qmini.py`).

If everything is set up correctly, you should see the robot spawn ~45 cm above the ground, fall briefly, and then stand stably in the crouched pose — no exploding joints, no penetration through the floor, no immediate falls.

### Usage

```bash
python scripts/zero_agent.py --task=qmini-velocity --num_envs 1
```

### Useful flags

- `--task` — Gym task ID. Use `qmini-velocity` (training env) or `qmini-velocity-play` (smaller scene, no randomization).
- `--num_envs` — number of parallel environments. Use `1` for visual inspection, larger values to stress-test stability.
- `--headless` — run without the viewer window (faster; useful if you only want to confirm there are no crashes).
- `--disable_fabric` — fall back to USD I/O instead of Fabric. Only needed if Fabric throws unexpected errors.

### What it tells you

- **Robot stands stably in the crouched pose** — URDF, `stiffness`/`damping`/`armature`, and `init_state` are consistent. Safe to start training.
- **Robot drifts and slowly tips over** — likely `stiffness` too low, `armature` too low, or the crouch pose is not statically stable for this geometry. Re-tune PD gains.
- **Joints visibly jitter or oscillate** — `stiffness` too high relative to `armature`. Increase `armature` or reduce `stiffness` (see `LEARN.md`).
- **Feet penetrate the ground or the robot launches into the air** — `init_state.pos` z is too low, or the URDF collision meshes are wrong.
- **Simulation crashes with NaN** — almost always an actuator-config issue (effort/velocity limits, or `armature = 0` with very stiff PD).

---

## 📦 Exporting a Policy to ONNX with `export_onnx.py`

To deploy a trained policy on the Jetson, export it to ONNX with `scripts/export_onnx.py`. Pass the **date folder** of the training run; the script picks the latest `model_*.pt` in `logs/rsl_rl/qmini/<date>/` and exports it:

```bash
python scripts/export_onnx.py 2026-05-24
```

This writes the result to `policy.onnx` in the project root. The exported graph is the deterministic actor — it maps raw observations directly to action means (the Qmini PPO config uses no observation normalization).

### Why a standalone script (instead of `play.py`)

`play.py` also exports ONNX via the built-in Isaac Lab exporter, but that exporter expects the legacy `ActorCritic` layout and silently **skips** the export for the newer rsl-rl `mlp.*` weight layout this project trains with. `export_onnx.py` sidesteps that: it reconstructs the actor MLP directly from the checkpoint's `actor_state_dict`, so it does **not** boot Isaac Sim and does **not** depend on the installed rsl-rl version.

### Verification

If `onnxruntime` is installed (it is a project dependency), the script runs the exported graph and compares it against the PyTorch model on random inputs, printing the max absolute difference:

```
[INFO] PyTorch/ONNX max abs diff: 5.364e-07 (OK)
```

A difference below `1e-5` confirms the export is faithful and safe to deploy.

### Useful flags

- `--activation` — hidden activation, default `elu` (matches `PPORunnerCfg`).
- `--opset` — ONNX opset version, default `12`.
- `--key` — state-dict key inside the checkpoint, default `actor_state_dict`.

> **Note:** The script always exports the **latest** checkpoint in the folder, which is not necessarily the **best**-performing one — RL reward can be noisy near the end of training. Validate with `play.py` rollouts before deploying.

> **Tip:** `zero_agent.py` is also the fastest way to validate any change to `qmini.py` or the URDF before you spend GPU hours training.