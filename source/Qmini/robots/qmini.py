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
        asset_path=str(Path(__file__).resolve().parents[1] / "assets/descriptions/qmini/qmini.urdf"),
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
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            # Mirrored bipedal crouch pose migrated from RoboTamer4Qmini (Isaac Gym)
            "hip_yaw_l":      0.4,
            "hip_roll_l":    -0.1,
            "hip_pitch_l":   -1.5,
            "knee_pitch_l":   1.0,
            "ankle_pitch_l": -1.3,
            "hip_yaw_r":     -0.4,
            "hip_roll_r":    0.1,
            "hip_pitch_r":    1.5,
            "knee_pitch_r":  -1.0,
            "ankle_pitch_r":  1.3,
        },
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["hip_yaw_.*", "hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*", "ankle_pitch_.*"],
            stiffness={                 #刚性
                "hip_yaw_.*": 55.0,
                "hip_roll_.*": 105.0,
                "hip_pitch_.*": 75.0,
                "knee_pitch_.*": 45.0,
                "ankle_pitch_.*": 30.0,
            },
            damping={                   #阻尼
                "hip_yaw_.*": 0.3,
                "hip_roll_.*": 2.5,
                "hip_pitch_.*": 0.3,
                "knee_pitch_.*": 0.5,
                "ankle_pitch_.*": 0.25,
            },
            armature={                  #电枢惯量
                "hip_yaw_.*": 0.02,
                "hip_roll_.*": 0.02,
                "hip_pitch_.*": 0.02,
                "knee_pitch_.*": 0.02,
                "ankle_pitch_.*": 0.0042,
            },
            effort_limit_sim={
                "hip_yaw_.*": 42.0,
                "hip_roll_.*": 42.0,
                "hip_pitch_.*": 42.0,
                "knee_pitch_.*": 42.0,
                "ankle_pitch_.*": 11.9,
            },
            velocity_limit_sim={
                "hip_yaw_.*": 18.849,
                "hip_roll_.*": 18.849,
                "hip_pitch_.*": 18.849,
                "knee_pitch_.*": 18.849,
                "ankle_pitch_.*": 37.699,
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