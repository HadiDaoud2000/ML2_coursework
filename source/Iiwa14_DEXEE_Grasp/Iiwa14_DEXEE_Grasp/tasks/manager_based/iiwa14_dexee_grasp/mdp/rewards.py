
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_from_euler_xyz, subtract_frame_transforms
from pathlib import Path
import numpy as np

from .observations import ( 
    get_grasp_config,
    frame_in_robot_root_frame,
    quat_in_robot_frame,
    initial_object_pose_robot_root_frame,
    desired_config_robot_frame,
    get_open_gripper_config
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection


def position_tracking(
        env: ManagerBasedRLEnv,

) -> torch.Tensor:

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)/1

    return (object_ee_distance )


def position_enhance(
        env: ManagerBasedRLEnv,

) -> torch.Tensor:

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)
    quat_RF = quat_in_robot_frame(env)

    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    return (1 - torch.tanh(object_ee_distance))




def orientation_tracking(
        env: ManagerBasedRLEnv, 
) -> torch.Tensor:
    quat_RF = quat_in_robot_frame(env)
    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi

    return diff

def test_4(
        env: ManagerBasedRLEnv,

) -> torch.Tensor:

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)
    quat_RF = quat_in_robot_frame(env)

    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi


    asset: Articulation = env.scene["robot"]
    # print(asset.data.joint_names)
    ids=[7,8,9,10,11,12,13,14,15,16,17,18]
    actual_joints_position = asset.data.joint_pos[:, ids]

    grasp_pos= get_open_gripper_config(env)
    grasp_joints_position = grasp_pos[:, [7,8,9,10,11,12,13,14,15,16,17,18]]

    diff0 = torch.mean(torch.abs(asset.data.joint_pos[:, [7,8,9]]*180/np.pi - grasp_pos[:, [7,8,9]]*180/np.pi) , dim=1)/100
    diff1 = torch.mean(torch.abs(asset.data.joint_pos[:, [10,11,12]]*180/np.pi - grasp_pos[:, [10,11,12]]*180/np.pi) , dim=1)/125
    diff2 = torch.mean(torch.abs(asset.data.joint_pos[:, [13,14,15]]*180/np.pi - grasp_pos[:, [13,14,15]]*180/np.pi) , dim=1)/80
    diff3 = torch.mean(torch.abs(asset.data.joint_pos[:, [16,17,18]]*180/np.pi - grasp_pos[:, [16,17,18]]*180/np.pi) , dim=1)/115

    # b = torch.mean(torch.abs(diff) , dim=1)
    b = (4*diff0+3*diff1+2*diff2+diff3)/10


    return (1 - torch.tanh(object_ee_distance))*(object_ee_distance<0.1)*(diff<0.06)*(b<0.05)*(object_ee_distance>=0.05) + (1 - torch.tanh(object_ee_distance))*(object_ee_distance<0.1)*(diff<0.06)*(object_ee_distance<0.05)

def grasp_joints_difference(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    # print(asset.data.joint_names)
    ids=[7,8,9,10,11,12,13,14,15,16,17,18]
    actual_joints_position = asset.data.joint_pos[:, ids]

    grasp_pos= get_grasp_config(env)
    grasp_joints_position = grasp_pos[:, [7,8,9,10,11,12,13,14,15,16,17,18]]

    diff0 = torch.mean(torch.abs(asset.data.joint_pos[:, [7,8,9]]*180/np.pi - grasp_pos[:, [7,8,9]]*180/np.pi) , dim=1)/100
    diff1 = torch.mean(torch.abs(asset.data.joint_pos[:, [10,11,12]]*180/np.pi - grasp_pos[:, [10,11,12]]*180/np.pi) , dim=1)/125
    diff2 = torch.mean(torch.abs(asset.data.joint_pos[:, [13,14,15]]*180/np.pi - grasp_pos[:, [13,14,15]]*180/np.pi) , dim=1)/80
    diff3 = torch.mean(torch.abs(asset.data.joint_pos[:, [16,17,18]]*180/np.pi - grasp_pos[:, [16,17,18]]*180/np.pi) , dim=1)/115

    # b = torch.mean(torch.abs(diff) , dim=1)
    b = (4*diff0+3*diff1+2*diff2+diff3)/10

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    # print('ee_pos_RF is:', ee_pos_RF)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)/1

 
    quat_RF = quat_in_robot_frame(env)

    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi

    # return b*(object_ee_distance>0.1) + b*(diff_rot>0.1) + 0.0001*(object_ee_distance<=0.1)*(diff_rot<=0.1)
    # return torch.exp(-(b**0.5))
    return b



def enhance_grasp_joints_difference(##############################
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    # print(asset.data.joint_names)
    ids=[7,8,9,10,11,12,13,14,15,16,17,18]
    actual_joints_position = asset.data.joint_pos[:, ids]

    grasp_pos= get_grasp_config(env)#############################################
    grasp_joints_position = grasp_pos[:, [7,8,9,10,11,12,13,14,15,16,17,18]]

    diff0 = torch.mean(torch.abs(asset.data.joint_pos[:, [7,8,9]]*180/np.pi - grasp_pos[:, [7,8,9]]*180/np.pi) , dim=1)/100
    diff1 = torch.mean(torch.abs(asset.data.joint_pos[:, [10,11,12]]*180/np.pi - grasp_pos[:, [10,11,12]]*180/np.pi) , dim=1)/125
    diff2 = torch.mean(torch.abs(asset.data.joint_pos[:, [13,14,15]]*180/np.pi - grasp_pos[:, [13,14,15]]*180/np.pi) , dim=1)/80
    diff3 = torch.mean(torch.abs(asset.data.joint_pos[:, [16,17,18]]*180/np.pi - grasp_pos[:, [16,17,18]]*180/np.pi) , dim=1)/115

    # b = torch.mean(torch.abs(diff) , dim=1)
    b = (1*diff0+5*diff1+3*diff2+4*diff3)/13

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    # print('ee_pos_RF is:', ee_pos_RF)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)/1

 
    quat_RF = quat_in_robot_frame(env)

    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi


    return 200*(1-b)*(object_ee_distance<0.08)*(diff<0.08)
    # return b*(object_ee_distance<=0.08)  + (object_ee_distance>0.08)


def keep_gripper_open(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ids=[7,8,9,10,11,12,13,14,15,16,17,18]
    actual_joints_position = asset.data.joint_pos[:, ids]

    grasp_pos= get_open_gripper_config(env)
    grasp_joints_position = grasp_pos[:, [7,8,9,10,11,12,13,14,15,16,17,18]]
    diff0 = torch.mean(torch.abs(asset.data.joint_pos[:, [7,8,9]]*180/np.pi - grasp_pos[:, [7,8,9]]*180/np.pi) , dim=1)/100
    diff1 = torch.mean(torch.abs(asset.data.joint_pos[:, [10,11,12]]*180/np.pi - grasp_pos[:, [10,11,12]]*180/np.pi) , dim=1)/125
    diff2 = torch.mean(torch.abs(asset.data.joint_pos[:, [13,14,15]]*180/np.pi - grasp_pos[:, [13,14,15]]*180/np.pi) , dim=1)/80
    diff3 = torch.mean(torch.abs(asset.data.joint_pos[:, [16,17,18]]*180/np.pi - grasp_pos[:, [16,17,18]]*180/np.pi) , dim=1)/115

    # b = torch.mean(torch.abs(diff) , dim=1)
    b = (4*diff0+3*diff1+2*diff2+diff3)/10


    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    # print('ee_pos_RF is:', ee_pos_RF)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)/1

    # quat_RF = quat_in_robot_frame(env)
    # grasp_quat = desired_config_robot_frame(env)[:,3:7]
    # # grasp_quat = quat_from_euler_xyz(grasp_pos[:, 3], grasp_pos[:, 4], grasp_pos[:, 5])
    # diff_rot = quat_error_magnitude(quat_RF, grasp_quat)/np.pi
    # print('actual joints are', actual_joints_position*np.pi/180)
    # return b*(object_ee_distance>0.1) + b*(diff_rot>0.1) + 0.0001*(object_ee_distance<=0.1)*(diff_rot<=0.1)
    # return 1-torch.exp(-5*b)*(10**torch.log10(torch.exp(-b)))
    return b*(object_ee_distance >0.08)


    # return b


def object_finger_contact(
    env: ManagerBasedRLEnv,
    J0_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J0_jointbody"),
    J1_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J1_jointbody"),
    J2_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J2_jointbody"),
    J3_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J3_jointbody"),

) -> torch.Tensor:
    """"""
    threshold = 0.1
    # extract the used quantities (to enable type-hinting)
    sensor0 = env.scene.sensors[J0_cfgs.name]
    f0 = torch.norm(sensor0.data.net_forces_w[:, :, ], dim=-1)
    i0 = ( f0> threshold)
    sensor1 = env.scene.sensors[J1_cfgs.name]
    f1 = torch.norm(sensor1.data.net_forces_w[:, :, ], dim=-1)
    i1 = (f1 > threshold)
    sensor2 = env.scene.sensors[J2_cfgs.name]
    f2 = torch.norm(sensor2.data.net_forces_w[:, :, ], dim=-1)
    i2 = ( f2 > threshold) 
    sensor3 = env.scene.sensors[J3_cfgs.name]
    i3 = (torch.norm(sensor3.data.net_forces_w[:, :, ], dim=-1) > threshold) *(torch.norm(sensor3.data.net_forces_w[:, :, ], dim=-1) <3000)
    is_contact_finger = (i0+i1+i2+i3).squeeze(1)

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)/1

    quat_RF = quat_in_robot_frame(env)
    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi


    return 1*(is_contact_finger)*(diff<0.03)*(object_ee_distance<0.04)

def object_all_fingers_contact(
    env: ManagerBasedRLEnv,
    J0_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J0_jointbody"),
    J1_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J1_jointbody"),
    J2_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J2_jointbody"),
    J3_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J3_jointbody"),

) -> torch.Tensor:
    """"""
    F0 = object_finger_contact(env)
    F1 = object_finger_contact(env, SceneEntityCfg("F1_J0_jointbody"), SceneEntityCfg("F1_J1_jointbody"),
                                    SceneEntityCfg("F1_J2_jointbody"), SceneEntityCfg("F1_J3_jointbody"))
    F2 = object_finger_contact(env, SceneEntityCfg("F2_J0_jointbody"), SceneEntityCfg("F2_J1_jointbody"),
                                    SceneEntityCfg("F2_J2_jointbody"), SceneEntityCfg("F2_J3_jointbody"))
    return 1*(F0*F1*F2)



# This penalty function is to ensure that the object didn't move without grasping
def object_moving(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.01,  # Small tolerance 
) -> torch.Tensor:
    
    device = env.device
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    robot_pos_w = robot.data.root_pos_w[:, :3]
    object_pos_w = obj.data.root_pos_w[:, :3]
    
    # Transform object position to robot's frame
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        object_pos_w
    )
    
    # Expected position in robot's frame (adjust as needed)
    expected_pos = initial_object_pose_robot_root_frame(env)[:,:3]
    pos_diff = torch.norm(object_pos_b - expected_pos, dim=1)
    
    return pos_diff


def undesired_object_finger_contact(
    env: ManagerBasedRLEnv,
    J0_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J0_jointbody"),
    J1_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J1_jointbody"),
    J2_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J2_jointbody"),
    J3_cfgs: SceneEntityCfg = SceneEntityCfg("F0_J3_jointbody"),

) -> torch.Tensor:
    """"""
    threshold = 0.1
    # extract the used quantities (to enable type-hinting)
    sensor0 = env.scene.sensors[J0_cfgs.name]
    f0 = torch.norm(sensor0.data.net_forces_w[:, :, ], dim=-1)
    i0 = ( f0> threshold)
    sensor1 = env.scene.sensors[J1_cfgs.name]
    f1 = torch.norm(sensor1.data.net_forces_w[:, :, ], dim=-1)
    i1 = (f1 > threshold)
    sensor2 = env.scene.sensors[J2_cfgs.name]
    f2 = torch.norm(sensor2.data.net_forces_w[:, :, ], dim=-1)
    i2 = ( f2 > threshold) 
    sensor3 = env.scene.sensors[J3_cfgs.name]
    i3 = (torch.norm(sensor3.data.net_forces_w[:, :, ], dim=-1) > threshold) *(torch.norm(sensor3.data.net_forces_w[:, :, ], dim=-1) <3000)
    is_contact_finger = (i0+i1+i2+i3).squeeze(1)
    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    # print('ee_pos_RF is:', ee_pos_RF)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)/1

    quat_RF = quat_in_robot_frame(env)

    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi


    return 1*(is_contact_finger)*(object_ee_distance>0.1) + 1*(is_contact_finger)*(diff>0.1)



