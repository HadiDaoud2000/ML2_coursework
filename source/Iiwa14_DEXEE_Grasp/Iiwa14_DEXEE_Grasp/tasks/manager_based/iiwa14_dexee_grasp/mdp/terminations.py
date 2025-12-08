# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import random
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm
from .observations import initial_object_pose_robot_root_frame
from isaaclab.utils.math import quat_error_magnitude
import numpy as np

"""
MDP terminations.
"""


# def time_out2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Terminate the episode when the episode length exceeds the maximum episode length."""
#     c = random.randint(10,150)
#     b = random.randint(10,150)
#     a =  env.episode_length_buf >= torch.tensor([c,b]).to(env.device)
#     print('terminated terminated:' ,a )
#     return a



def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)



def object_falling_from_the_table(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def object_oriented_badly(
    env: ManagerBasedRLEnv, thr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Terminate when the object's current orientation is a lot different than its initial orientation.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    current_quat = asset.data.root_quat_w[:, :4] 
    initial_quat = initial_object_pose_robot_root_frame(env)[:,3:7]
    diff = quat_error_magnitude(current_quat, initial_quat)/np.pi

    return (diff >=thr)
