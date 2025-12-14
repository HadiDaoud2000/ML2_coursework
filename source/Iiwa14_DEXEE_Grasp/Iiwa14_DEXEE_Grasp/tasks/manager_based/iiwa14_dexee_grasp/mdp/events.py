
from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# def reset_root_state_uniform2(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     pose_range: dict[str, tuple[float, float]],
#     velocity_range: dict[str, tuple[float, float]],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     command_name: str | None = None
# ):
#     """Reset the asset root state to a random position and velocity uniformly within the given ranges.

#     This function randomizes the root position and velocity of the asset.

#     * It samples the root position from the given ranges and adds them to the default root position, before setting
#       them into the physics simulation.
#     * It samples the root orientation from the given ranges and sets them into the physics simulation.
#     * It samples the root velocity from the given ranges and sets them into the physics simulation.

#     The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
#     dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
#     ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     # get default root state
#     root_states = asset.data.default_root_state[env_ids].clone()
#     print('blbbllblblblblblblbl', env.command_manager.get_command(command_name))

#     # poses
#     range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#     ranges = torch.tensor(range_list, device=asset.device)
#     # rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
#     rand_samples= env.command_manager.get_command(command_name)
#     positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
#     orientations_delta = rand_samples[:,3:7]
#     orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
#     # velocities
#     range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#     ranges = torch.tensor(range_list, device=asset.device)
#     rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

#     velocities = root_states[:, 7:13] + rand_samples
#     print('in events' ,torch.cat([positions, orientations], dim=-1))
#     print('fsssssssssssssssssssssdsssssssssssssssfdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd' \
#     'sfdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd' \
#     'sfdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd' \
#     'sfddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
#     # set into the physics simulation

#     asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
#     asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)





def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)




# def reset_manipulator_joints_by_scale(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     position_range: tuple[float, float],
#     velocity_range: tuple[float, float],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Reset the robot joints by scaling the default position and velocity by the given ranges.

#     This function samples random values from the given ranges and scales the default joint positions and velocities
#     by these values. The scaled values are then set into the physics simulation.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]

#     # cast env_ids to allow broadcasting
#     if asset_cfg.joint_ids != slice(None):
#         iter_env_ids = env_ids[:, None]
#     else:
#         iter_env_ids = env_ids
#     print('hhhhhhhhhhhhhh',asset_cfg.joint_ids)

#     # get default joint state
#     joint_pos = asset.data.default_joint_pos[iter_env_ids, [asset_cfg.joint_ids]].clone()
#     joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

#     # scale these values randomly
#     joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
#     joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

#     # clamp joint pos to limits
#     joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
#     joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
#     # clamp joint vel to limits
#     joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
#     joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

#     # set into the physics simulation
#     asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

def reset_manipulator_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    
    Note: This function only affects the first 7 joints of the robot.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the total number of joints
    num_joints = asset.num_joints
    
    # Get default joint state for ALL joints
    joint_pos = asset.data.default_joint_pos[env_ids, :].clone()
    joint_vel = asset.data.default_joint_vel[env_ids, :].clone()

    # Create scaling factors - only scale first 7 joints, keep others at 1.0
    scale_pos = torch.ones((len(env_ids), num_joints), device=joint_pos.device)
    scale_vel = torch.ones((len(env_ids), num_joints), device=joint_vel.device)
    
    # Apply random scaling only to first 7 joints
    pos_scale_samples = math_utils.sample_uniform(*position_range, (len(env_ids), 7), joint_pos.device)
    vel_scale_samples = math_utils.sample_uniform(*velocity_range, (len(env_ids), 7), joint_vel.device)
    
    scale_pos[:, :7] = pos_scale_samples
    scale_vel[:, :7] = vel_scale_samples

    # Scale the joint positions and velocities
    joint_pos *= scale_pos
    joint_vel *= scale_vel

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, :]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids, :]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation (all joints)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=None, env_ids=env_ids)