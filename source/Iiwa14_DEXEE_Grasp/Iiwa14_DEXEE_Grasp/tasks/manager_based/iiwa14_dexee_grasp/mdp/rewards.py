
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_from_euler_xyz, subtract_frame_transforms, quat_slerp
from pathlib import Path
import numpy as np

from .observations import ( 
    get_grasp_config,
    frame_in_robot_root_frame,
    quat_in_robot_frame,
    initial_object_pose_robot_root_frame,
    desired_config_robot_frame,
    get_open_gripper_config,
    initial_distance,
    initial_EE_pose_robot_root_frame,
    object_rotation_in_robot_root_frame,
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
    print('distance is:' , object_ee_distance[:10])

    return (object_ee_distance )


def position_enhance(
        env: ManagerBasedRLEnv,

) -> torch.Tensor:

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = frame_in_robot_root_frame(env)
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)
    quat_RF = quat_in_robot_frame(env)

    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi

    return (1 - torch.tanh(object_ee_distance))*(diff<=0.1)




def orientation_tracking(
        env: ManagerBasedRLEnv, 
) -> torch.Tensor:
    quat_RF = quat_in_robot_frame(env)
    grasp_quat = desired_config_robot_frame(env)[:,3:7]
    diff = quat_error_magnitude(quat_RF, grasp_quat)/np.pi
    print('orientation  is:' , diff[:10])

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


def test_5(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Extract essential info
    ee_pos_RF = frame_in_robot_root_frame(env)  # (N,3)
    grasp_target_RF = desired_config_robot_frame(env)[:, :3]
    quat_RF = quat_in_robot_frame(env)
    grasp_quat = desired_config_robot_frame(env)[:, 3:7]

    d = torch.norm(ee_pos_RF - grasp_target_RF, dim=1)   

    ori_err = quat_error_magnitude(quat_RF, grasp_quat) / np.pi  

    # Robot articulation
    asset: Articulation = env.scene["robot"]
    joint_ids = [7,8,9,10,11,12,13,14,15,16,17,18]
    q_t = asset.data.joint_pos[:, joint_ids]  

    q_open = get_open_gripper_config(env)[:, joint_ids]  

    # Desired grasp configuration (target when very close)
    grasp_pos =   get_grasp_config(env)
    q_goal = grasp_pos[:, [7,8,9,10,11,12,13,14,15,16,17,18]]


    # --- Smooth trajectory schedule ---
    # when d >= 0.1m   → stay open
    # when d <= 0.02m → be fully grasp pose


    d_start = 1
    d_end = 0.02
    denom = (d_start - d_end)
    if isinstance(denom, tuple):
        denom = float(denom[0])   # fallback for tuple param bug

    s = torch.clamp((d_start - d) / denom, 0.0, 1.0)



    # d_start= 0.2,
    # d_end =  0.02
    # s = torch.clamp((d_start - d) / (d_start - d_end), 0.0, 1.0)  

    # Interpolated joint target: trajectory
    q_traj = q_open + s[:, None] * (q_goal - q_open)  

    # Joint tracking error (deg for scaling)
    joint_err0 = torch.abs((q_t - q_traj) * 180/np.pi)
    joint_err0[:,:3] = joint_err0[:,:3]/100
    joint_err0[:,3:6] = joint_err0[:,3:6]/125
    joint_err0[:,6:9] = joint_err0[:,6:9]/80
    joint_err0[:,9:] = joint_err0[:,9:]/115

    joint_err = torch.mean(joint_err0, dim=1)
    # Normalize ranges
    reward = -joint_err   

    # Optional success bonus if really close
    # reward += (d < 0.015) * 2.0
    print('joint error is:', joint_err[:10] )
    return reward



def test_6(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Extract essential info
    ee_pos_RF = frame_in_robot_root_frame(env)  # (N,3)
    grasp_target_RF = desired_config_robot_frame(env)[:, :3]
    initial_d = initial_distance(env)
    initial_quat = initial_EE_pose_robot_root_frame(env)[:,3:7]
    quat_RF = quat_in_robot_frame(env)
    grasp_quat = desired_config_robot_frame(env)[:, 3:7]
    # print('initial d is kkkkk', initial_d)
    d = (torch.clamp(torch.norm(ee_pos_RF - grasp_target_RF, dim=1), torch.zeros(env.num_envs,).to(env.device) ,initial_d)/ initial_d).to(env.device)
    orientation_trajectory = torch.zeros(env.num_envs,4).to(env.device)
    # print( ' initial quat hghghgh is:', initial_quat[0])
    # print('dddddddddddddddddddddis:', d[0])
    for i in range(env.num_envs):
        
        orientation_trajectory[i]= quat_slerp( grasp_quat[i], initial_quat[i], d[i])
        # print('grasp quanterion is :', grasp_quat[i])
        # print('slerp wuanterion is :', orientation_trajectory[i])
    ori_err = quat_error_magnitude(quat_RF, orientation_trajectory) / np.pi  

    print('quat tracking error is:', ori_err [:10])
    return ori_err



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
    actual_joints_position = asset.data.joint_pos[:, [0,1,2,3,4,5,6]]
    print('actual joints are:', actual_joints_position)
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


    return 2*(1-b)*(object_ee_distance<0.08)*(diff<0.08)
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
    print(' fingers contact :', 1*(F0*F1*F2))
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




def object_moving2(
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
    object_pos_b = object_rotation_in_robot_root_frame(env)
    # print('quat in robot frame is :', object_pos_b)
    
    # Expected position in robot's frame (adjust as needed)
    expected_pos = initial_object_pose_robot_root_frame(env)[:,3:7]
    # print(' expected pos is:', expected_pos)
    pos_diff = quat_error_magnitude(object_pos_b, expected_pos)/np.pi
    print('move diff2:', pos_diff[:20])
    
    return pos_diff
