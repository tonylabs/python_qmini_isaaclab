from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg # noqa: F401
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# Dynamically get the directory where this qmini.py script is located
QMINI_ASSETS_DIR = Path(__file__).resolve().parent

BDX_R_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        # The URDF lives in the qmini_description submodule (checked out at assets/qmini) and
        # references its meshes as package://qmini_description/... . The importer's resolveXrefPath
        # strips the package name and searches the URDF's parent directories, so meshes/ inside
        # assets/qmini is found without any qmini_description directory on disk.
        asset_path=str(Path(__file__).resolve().parents[1] / "assets/qmini/urdf/qmini_description.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Statically-stable standing pose for the qmini_description submodule URDF (zero-pose
        # convention with straight legs; joint origins carry no baked-in rpy offsets). Derived with
        # scripts/tune_home_pose.py, which drops the robot free-base under its own PD actuators and
        # searches for a pose that stays upright with both feet planted. A zero action holds this
        # pose, so zero_agent.py shows a stable stand.
        pos=(0.0, 0.0, 0.4173),
        joint_pos={
            "hip_yaw_l":      -0.1150,
            "hip_roll_l":     0.0150,
            "hip_pitch_l":    0.2500,
            "knee_pitch_l":   0.4000,
            "ankle_pitch_l":  0.1800,

            "hip_yaw_r":      0.1150,
            "hip_roll_r":     -0.0150,
            "hip_pitch_r":    -0.2500,
            "knee_pitch_r":   -0.4000,
            "ankle_pitch_r":  -0.1800,
        },
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["hip_yaw_.*", "hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*", "ankle_pitch_.*"],
            # 刚性/阻尼 — FAITHFUL to the deployed hardware (qmini_official_sdk
            # config.yaml). The GO-M8010-6 PD runs in ROTOR coordinates and the SDK
            # passes kp/kd through unscaled while converting q/dq by the reduction,
            # so effective joint-side gain = config value × ratio²
            # (6.33² = 40.07 normal joints; (6.33·3)² = 360.62 for the geared hip_roll):
            #   kp: hip_yaw 1.0→40.07, hip_roll 0.5→180.31, hip_pitch 1.5→60.10,
            #       knee 1.0→40.07, ankle 1.0→40.07
            #   kd: 0.05 everywhere → 2.00 (hip_roll → 18.03)
            # The previous 55/105/75/45/30 + 0.25-2.5 values were NOT what the robot
            # actually runs — training with these closes that sim2real gap.
            stiffness={
                "hip_yaw_.*": 40.07,
                "hip_roll_.*": 180.31,
                "hip_pitch_.*": 60.10,
                "knee_pitch_.*": 40.07,
                "ankle_pitch_.*": 40.07,
            },
            damping={
                "hip_yaw_.*": 2.0,
                "hip_roll_.*": 18.03,
                "hip_pitch_.*": 2.0,
                "knee_pitch_.*": 2.0,
                "ankle_pitch_.*": 2.0,
            },
            # 电枢惯量 (reflected rotor inertia = I_rotor × reduction²).
            # Every joint is a Unitree GO-M8010-6 with a 6.33:1 internal reduction
            # (armature 0.02 ↔ I_rotor ≈ 5.0e-4 kg·m²). The hip_roll joints carry an
            # ADDITIONAL 3:1 external gear (qmini_official_sdk Motor_thread.hpp:
            # Gear_Ratio=3 on motors 1/6), so their reflected inertia is 3² = 9×
            # larger: 0.18. The ankle is the same motor as the rest — its old 0.0042
            # implied a 5× lighter rotor and was physically inconsistent.
            armature={
                "hip_yaw_.*": 0.02,
                "hip_roll_.*": 0.18,
                "hip_pitch_.*": 0.02,
                "knee_pitch_.*": 0.02,
                "ankle_pitch_.*": 0.02,
            },
            # Effort/velocity limits mirror the proven Isaac Gym setup (qmini_official_rl):
            # its q1.urdf declares effort=20 (hip_roll 60) and velocity=30 (hip_roll 10),
            # which Isaac Gym clipped torques to at every step. The qmini_description URDF
            # carries the same values; stating them here keeps the sim actuator explicit.
            effort_limit_sim={
                "hip_yaw_.*": 20.0,
                "hip_roll_.*": 60.0,
                "hip_pitch_.*": 20.0,
                "knee_pitch_.*": 20.0,
                "ankle_pitch_.*": 20.0,
            },
            velocity_limit_sim={
                "hip_yaw_.*": 30.0,
                "hip_roll_.*": 10.0,
                "hip_pitch_.*": 30.0,
                "knee_pitch_.*": 30.0,
                "ankle_pitch_.*": 30.0,
            },
            # Sim-to-real: USB→RS-485 polling + motor command-execution lag on the Pi 5
            # is empirically several ms; randomize 2–8 sim steps at 200 Hz physics = 10–40 ms
            # per episode, matching the RoboTamer4Qmini (IG) reference framework.
            min_delay=2,
            max_delay=8
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Disney BD-X robot with implicit actuator model."""