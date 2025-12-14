# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from dataclasses import MISSING
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import ContactSensorCfg

from . import mdp

##
# Pre-defined configs
##

from Iiwa14_DEXEE_Grasp.Robots.iiwa14_dexee import IIWA14_DEXEE_CFG  # noqa: F401


##
# Scene definition
##


@configclass
class Iiwa14DexeeGraspSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robot
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Contact sensors
    F0_J0_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F0_J0_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F0_J1_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F0_J1_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F0_J2_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F0_J2_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F0_J3_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F0_J3_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F1_J0_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F1_J0_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F1_J1_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F1_J1_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F1_J2_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F1_J2_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F1_J3_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F1_J3_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F2_J0_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F2_J0_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F2_J1_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F2_J1_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F2_J2_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F2_J2_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)
    F2_J3_jointbody = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/F2_J3_jointbody", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],update_period=0.0, history_length=6, debug_vis= True)

#
# MDP settings
##
@configclass
class CommandsCfg:
    """Command terms for the MDP."""



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | mdp.JointActionCfg | mdp.JointPositionActionCfg | mdp.JointPositionToLimitsActionCfg | mdp.EMAJointPositionToLimitsActionCfg | None = None



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pos_RRF = ObsTerm (func = mdp.frame_in_robot_root_frame, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_quat_RRF = ObsTerm(func= mdp.quat_in_robot_frame, noise=Unoise(n_min=-0.01, n_max=0.01))
        desired_pose = ObsTerm(func = mdp.desired_config_robot_frame)
        desired_hand_joints = ObsTerm(func = mdp.desired_hand_joints)
        object_position = ObsTerm(func = mdp.object_position_in_robot_root_frame, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_orientation = ObsTerm(func = mdp.object_rotation_in_robot_root_frame, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_manipulator_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:

    position_tracking = RewTerm (func=mdp.position_tracking, weight=-0.8)
    position_enhance = RewTerm (func=mdp.position_enhance, weight=0.1)
    object_moved = RewTerm( func= mdp.object_moving, weight = -0.3)
    object_moved2 = RewTerm( func= mdp.object_moving2, weight = -0.3)
    test5 = RewTerm(func = mdp.test_5, weight =0.65)
    test6 = RewTerm(func = mdp.test_6, weight =-0.4)
    lifting_object = RewTerm(func = mdp.lifting, weight = 100)
 
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
@configclass
class Iiwa14DexeeGraspEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Iiwa14DexeeGraspSceneCfg = Iiwa14DexeeGraspSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()

    # curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation