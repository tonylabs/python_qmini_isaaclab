"""IMU sources for sim2sim.py — a common ``read()`` contract over two backends.

``read()`` returns ``(ang_vel, projected_gravity, yaw, lin_acc)``:

* ``ang_vel``           — angular velocity in the IMU/base frame (rad/s), (3,)
* ``projected_gravity`` — unit gravity vector in the IMU/base frame; ≈ (0, 0, -1)
  when upright (Isaac Lab convention), (3,)
* ``yaw``               — base yaw in the world frame (rad); used only by the
  heading controller, never fed to the policy directly
* ``lin_acc``           — linear acceleration in the IMU frame (m/s²), (3,);
  diagnostic only

Qmini's IMU is mounted with **identity rotation** relative to ``base_link``
(``ImuCfg`` offset in ``qmini_env_cfg.py`` is translation-only), so the IMU
sensor frame and the base frame share orientation — angular velocity and
projected gravity are translation-invariant, meaning base-frame values are
exactly what training saw.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200


class MujocoImu:
    """Simulated IMU: reads the ``imu`` site (gyro + orientation) from MuJoCo.

    Uses the MJCF's named sensors when present (``imu_gyro``, ``imu_acc`` on the
    ``imu`` site of the qmini_description MJCF); orientation comes from the
    site's world rotation matrix.
    """

    def __init__(self, model, data, body_name: str = "base_link", site_name: str = "imu"):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise RuntimeError(f"Body {body_name!r} not found in MJCF")
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        # named sensor addresses (fall back to body-frame math if absent)
        self._gyro_adr = self._sensor_adr("imu_gyro")
        self._acc_adr = self._sensor_adr("imu_acc")

    def _sensor_adr(self, name: str) -> int | None:
        sid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SENSOR, name)
        return None if sid < 0 else int(self.model.sensor_adr[sid])

    def _rot_world_from_imu(self) -> np.ndarray:
        if self.site_id >= 0:
            return np.asarray(self.data.site_xmat[self.site_id], dtype=np.float64).reshape(3, 3)
        return np.asarray(self.data.xmat[self.body_id], dtype=np.float64).reshape(3, 3)

    def read(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        rot = self._rot_world_from_imu()
        # angular velocity in the IMU frame
        if self._gyro_adr is not None:
            ang_vel = np.asarray(self.data.sensordata[self._gyro_adr : self._gyro_adr + 3], dtype=np.float32)
        else:
            # free-joint angular velocity is expressed in the child (body) frame
            dof = int(self.model.body_dofadr[self.body_id])
            ang_vel = np.asarray(self.data.qvel[dof + 3 : dof + 6], dtype=np.float32)
        # unit gravity direction rotated into the IMU frame
        proj_g = (rot.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
        # base yaw in the world frame (from the body x-axis)
        r_body = np.asarray(self.data.xmat[self.body_id], dtype=np.float64).reshape(3, 3)
        yaw = float(np.arctan2(r_body[1, 0], r_body[0, 0]))
        if self._acc_adr is not None:
            lin_acc = np.asarray(self.data.sensordata[self._acc_adr : self._acc_adr + 3], dtype=np.float32)
        else:
            lin_acc = np.zeros(3, dtype=np.float32)
        return ang_vel, proj_g, yaw, lin_acc

    def close(self) -> None:
        pass


class Imu:
    """Hardware IMU over serial (HI226/HI229-style). Placeholder.

    Wire in the real serial driver when running ``--imu hardware``; sim2sim
    only needs the same ``read()``/``close()`` contract as :class:`MujocoImu`.
    """

    def __init__(self, port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUD):
        raise NotImplementedError(
            "Hardware IMU support is not wired up in this project yet — "
            "run sim2sim.py with the default --imu sim."
        )
