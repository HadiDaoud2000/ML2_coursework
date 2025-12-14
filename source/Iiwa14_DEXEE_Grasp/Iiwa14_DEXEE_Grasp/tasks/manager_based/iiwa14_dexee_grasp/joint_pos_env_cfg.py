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
        )



        RELATIVE_PATH = "../../source/Iiwa14_DEXEE_Grasp/Data/ddg-gd_drill_poisson_000/coacd/ddg-gd_drill_poisson_000.npy"
        ABSOLUTE_PATH = Path(RELATIVE_PATH).resolve().as_posix()

        data = np.load(
            ABSOLUTE_PATH, 
            allow_pickle=True
        )
        grasp = data[12]  
        # Usage

        scale0 =grasp["scale"]
        RELATIVE_PATH2 = "../../source/Iiwa14_DEXEE_Grasp/Data/ddg-gd_drill_poisson_000/coacd/drill2.usd"
        ABSOLUTE_PATH2 = Path(RELATIVE_PATH2).resolve().as_posix()
        self.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[pos_x, pos_y, pos_z], rot=[rot_w, rot_x, rot_y, rot_z]),  #to check grasp pose and also set kinematic_enabled to True
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6,0.0,0.04], rot=[0.0,-0.0,0.7071,-0.7071]), 

        spawn=UsdFileCfg(
            usd_path= ABSOLUTE_PATH2,
            scale=(scale0, scale0, scale0),

            semantic_tags=[("class", "object"), ("color", "red")],
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,

                disable_gravity=False,
            ),
        ),
    )
        

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



