# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from Iiwa14_DEXEE_Grasp.Robots.iiwa14_dexee import IIWA14_DEXEE_CFG  # noqa: F401
from Iiwa14_DEXEE_Grasp.tasks.manager_based.iiwa14_dexee_grasp.iiwa14_dexee_grasp_env_cfg import Iiwa14DexeeGraspEnvCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import math
from pathlib import Path
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
import numpy as np
import torch
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab.utils.math import quat_from_euler_xyz
##
# Scene definition
##
import pickle

def rotate_point_with_quaternion(q, point):
    """
    Rotate point using quaternion - compact version.
    """
    # q = quat_from_euler_xyz(E[0], E[1], E[2]).numpy()

    w, x, y, z = q
    x *= -1
    y *= -1
    z *= -1
    px, py, pz = point
    
    # Convert to quaternions
    q_vec = np.array([w, x, y, z])
    p_vec = np.array([0, px, py, pz])
    
    # Quaternion multiplication: result = q * p * q_conj
    q_conj = np.array([w, -x, -y, -z])
    
    # First multiplication: temp = q * p
    temp_w = -x*px - y*py - z*pz
    temp_x = w*px + y*pz - z*py
    temp_y = w*py + z*px - x*pz
    temp_z = w*pz + x*py - y*px
    
    # Second multiplication: result = temp * q_conj
    result_x = temp_w*(-x) + temp_x*w + temp_y*(-z) - temp_z*(-y)
    result_y = temp_w*(-y) + temp_y*w + temp_z*(-x) - temp_x*(-z)
    result_z = temp_w*(-z) + temp_z*w + temp_x*(-y) - temp_y*(-x)
    
    return np.array([result_x, result_y, result_z])



@configclass
class Iiwa14DEXEEGraspEnvCfg(Iiwa14DexeeGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = IIWA14_DEXEE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint.*"],
        )

        # self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        # asset_name="robot",
        # joint_names=["joint.*"],
        # body_name="joint7_jointbody",
        # controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        # scale=0.1,
        # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0,0.0,0.045]),
        # )

     
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
            "F0_J0",
            "F1_J0",
            "F2_J0",

            "F0_J1",
            "F1_J1",
            "F2_J1",

            "F0_J2",
            "F1_J2",
            "F2_J2",

            "F0_J3",
            "F1_J3",
            "F2_J3",
            ],
            # scale=0.0
        )


        # override actions

        # override command generator body
        # end-effector is along z-direction
        # self.commands.object_pose.body_name = "joint7_jointbody"
        # self.commands.object_pose.ranges.pitch = (0, 0)

        # self.commands.object_pose_reset.body_name = "joint7_jointbody"
        # self.commands.object_pose_reset.ranges.pitch = (0, 0)
        
        # Set the body name for the end effector
        # self.commands.object_pose.body_name = "joint7"
        # Change this:
# data = np.load(filename, allow_pickle=True)
# To this:


# /home/casper-3/Iiwa14_DEXEE_Grasp/source/Iiwa14_DEXEE_Grasp/Data/sem-Hammer-405f308492a6f40d2c3380317c2cc450/coacd/sem-Hammer-405f308492a6f40d2c3380317c2cc450.npy
# /home/casper-3/Iiwa14_DEXEE_Grasp/source/Iiwa14_DEXEE_Grasp/Data/dex_ee_pose_dataset/poses/sem-Hammer-405f308492a6f40d2c3380317c2cc450.npy
# /home/casper-3/Iiwa14_DEXEE_Grasp/source/Iiwa14_DEXEE_Grasp/Data/sem-Hammer-405f308492a6f40d2c3380317c2cc450/coacd/sem_hammer1.usd


        RELATIVE_PATH = "../../source/Iiwa14_DEXEE_Grasp/Data/sem-Hammer-369593e48bdb2208419a349e9c699f76/coacd/sem-Hammer-369593e48bdb2208419a349e9c699f76.npy"
        ABSOLUTE_PATH = Path(RELATIVE_PATH).resolve().as_posix()

        data = np.load(
            ABSOLUTE_PATH, 
            allow_pickle=True
        )
        grasp = data[19]  
        final_pose = grasp["qpos"]
        q = quat_from_euler_xyz(torch.tensor(final_pose['WRJRx']), torch.tensor(final_pose['WRJRy']), torch.tensor(final_pose['WRJRz'])).numpy()
        point = [final_pose['WRJTx'], final_pose['WRJTy'], final_pose['WRJTz']]
        # Usage

        result = rotate_point_with_quaternion(q, point)
        pos_x = float(-result[0])
        pos_y = float(-result[1])
        pos_z = float(1.306 - result[2])
        rot_w = float(q[0])
        rot_x = float(-q[1])
        rot_y = float(-q[2])
        rot_z = float(-q[3])

        print(f"Result: [{result[0]:.6f}, {result[1]:.6f}, {result[2]:.6f}]")
        scale0 =grasp["scale"]
        RELATIVE_PATH2 = "../../source/Iiwa14_DEXEE_Grasp/Data/sem-Hammer-369593e48bdb2208419a349e9c699f76/coacd/sem_hammer2.usd"
        ABSOLUTE_PATH2 = Path(RELATIVE_PATH2).resolve().as_posix()
        self.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[pos_x, pos_y, pos_z], rot=[rot_w, rot_x, rot_y, rot_z]),  #to check grasp pose and also set kinematic_enabled to True
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6,0.0,0.08], rot=[0.7071,0.7071,0.0,0.0]), 

        spawn=UsdFileCfg(
            usd_path= ABSOLUTE_PATH2,
            scale=(scale0, scale0, scale0),

            semantic_tags=[("class", "object"), ("color", "red")],
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                # max_angular_velocity=100,
                # max_linear_velocity=100,
                max_depenetration_velocity=5.0,
                linear_damping= 10,
                angular_damping = 10,
                disable_gravity=False,
                # kinematic_enabled= True,
            ),
            # mass_props= MassPropertiesCfg(mass = 5),
        ),
    )
        
        # Set Cube as object
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.85, 0.0, 0.2], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/converted_robot/Base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/converted_robot/joint7_jointbody",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos = [0.0,0.0,0.045]
                    ),
                ),
            ],
        )

@configclass
class Iiwa14DEXEEGraspEnvCfg_PLAY(Iiwa14DexeeGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



