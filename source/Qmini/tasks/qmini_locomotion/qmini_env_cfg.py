# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ImuCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
#from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv  # <<<<<<<<<< MOVE THE IMPORT HERE
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    # It's fine to keep this import here for type checkers if needed, but the one above is essential for runtime.
    from isaaclab.managers import RewardTermCfg

# TODO: Clean up all non used imports

##
# Pre-defined configs
##

from Qmini.robots.qmini import BDX_R_CFG # isort:skip

import Qmini.tasks.qmini_locomotion.mdp as mdp  # isort:skip

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class CriticCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        actions = ObsTerm(func=mdp.last_action)
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        feet_contact_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["ankle_pitch_.*"]
                )
            },
        )
        # Joint positions and velocities with less noise (privileged accurate state)
        joint_pos_accurate = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        joint_vel_accurate = ObsTerm(
            func=mdp.joint_vel_rel,
            scale =0.05,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        # Mirror policy's gait clock so the value function can credit phase-conditioned behaviour
        gait_phase_sincos = ObsTerm(func=mdp.gait_phase_sincos, params={"frequency": 1.5})
        static_flag = ObsTerm(func=mdp.static_flag, params={"command_name": "base_velocity", "threshold": 0.15})

        # No noise for the critic
        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, scale=0.2, noise=Unoise(n_min=-0.35, n_max=0.35))

        # TODO: Adds IMu Projected Gravity
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # Gait phase clock (sin/cos per leg, anti-phase) — gives the policy a coherent
        # swing/stance reference. Frequency is fixed at 1.5 Hz; retrain if changing.
        gait_phase_sincos = ObsTerm(func=mdp.gait_phase_sincos, params={"frequency": 1.5})
        # Static flag — 1.0 when the robot is being told to stand still, 0.0 otherwise.
        # Lets the policy switch behaviour modes; paired with static-aware rewards.
        static_flag = ObsTerm(func=mdp.static_flag, params={"command_name": "base_velocity", "threshold": 0.15})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups:
    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()





@configclass
class QminiRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    air_time = RewTerm(
        func=mdp.bipedal_air_time_reward,
        weight=7.5,
        params={
            "mode_time": 0.5,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
        },
    )
    # penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="ankle_pitch_.*")},
    )
    # Keep hip_yaw inside a band around its default (±0.4) so the legs can float a
    # little but cannot curl inward / toe in. deadzone=0.1 ⇒ allowed range
    # [0.3, 0.5] (left) and [-0.5, -0.3] (right). No penalty inside the band, linear
    # penalty on the excess outside it.
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deadzone_deviation_l1,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_yaw_.*"]),
            "deadzone": 0.1,
        },
    )
    # stronger penalty on hip_roll to prevent knock-knee / inward-bowed legs
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_roll_.*"])},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=2,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "threshold": 1.0,
        },
    )
    base_height_deviation = RewTerm(
        func=mdp.base_height_l2,
        weight=-2,  # Tune this weight as needed
        params={
            "target_height": 0.30846,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["base_link"]),
        },
    )
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    feet_alternate = RewTerm(
        func=mdp.feet_alternate_reward,
        weight=2.0,
        params={
            "std": 0.1,
            "max_err": 0.5,
            "velocity_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
        },
    )
    feet_double_support = RewTerm(
        func=mdp.feet_double_support_penalty,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "velocity_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
        },
    )

    # --- Phase-conditioned gait rewards (added 2026-05-23 for RoboTamer4Qmini parity) ---
    # Frequency MUST match the gait_phase_sincos observation term in PolicyCfg.
    foot_phase_clearance = RewTerm(
        func=mdp.foot_phase_clearance,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "target_height": 0.08,
            "std": 0.005,
            "frequency": 1.5,
            "static_threshold": 0.15,
        },
    )
    foot_phase_support = RewTerm(
        func=mdp.foot_phase_support,
        weight=0.7,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "threshold": 1.0,
            "frequency": 1.5,
            "static_threshold": 0.15,
        },
    )
    # Positive standing-mode bonus complementing joint_position_penalty's static scale.
    posture_hold = RewTerm(
        func=mdp.posture_hold,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
            "static_threshold": 0.15,
        },
    )





@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.2, 0.2),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
#
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Episode ends if any non-foot body touches the ground.
    # The previous version only checked `base_link`, which let the robot lie on its
    # hip/knee/ankle bodies indefinitely without triggering reset.
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "base_link",
                    "hip_yaw_.*",
                    "hip_roll_.*",
                    "hip_pitch_.*",
                    "knee_pitch_.*",
                ],
            ),
            "threshold": 1.0,
        },
    )

    # Episode ends if the base tilts more than ~50° from upright (≈0.9 rad).
    # Catches "fallen but still floating on a leg" failures the contact check misses.
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.9},
    )

    # Episode ends if the base drops below ~15 cm (true crouch is ~30-45 cm).
    # Catches collapsed poses where the contact and orientation checks both lag.
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.15},
    )

@configclass
class QminiEnvCfg(LocomotionVelocityRoughEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: QminiRewards = QminiRewards()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = BDX_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        
        # CORRECT IMU CONFIGURATION
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",  
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(  
                pos=(-0.04718, 0.0663, 0.11094),    
                rot=(1.0, 0.0, 0.0, 0.0),           
            )
        )
# actions
        self.actions.joint_pos.scale = 0.5

        # events
        self.events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        #self.events.push_robot = None
        self.rewards.feet_air_time = None 
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.physics_material.params["static_friction_range"] = (0.1, 2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 2)
        self.events.physics_material.params["asset_cfg"].body_names = "ankle_pitch_.*"
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        #self.events.randomize_imu_mount = EventTerm(
        #    func=randomize_imu_mount,
        #    mode="reset",
        #    params={
        #        "sensor_cfg": SceneEntityCfg("imu"),
        #        "pos_range": {
        #            "x": (-0.05, 0.05),
        #            "y": (-0.05, 0.05),
        #            "z": (-0.05, 0.05),
        #        },
        #        "rot_range": {
        #            "roll": (-0.1, 0.1),
        #            "pitch": (-0.1, 0.1),
        #            "yaw": (-0.1, 0.1),
        #        },
        #    },
        #)
        

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        # (illegal_contact, bad_orientation, base_too_low are configured directly
        # in TerminationsCfg above — no overrides needed here.)

        # rewards
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        self.rewards.action_rate_l2.weight =-0.05     
        self.rewards.dof_acc_l2.weight =-1.25e-7      
        self.rewards.flat_orientation_l2.weight = -2

        # Walk
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4,0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1, 1)


@configclass
class QminiEnvCfg_Play(QminiEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.00, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # remove random pushing event