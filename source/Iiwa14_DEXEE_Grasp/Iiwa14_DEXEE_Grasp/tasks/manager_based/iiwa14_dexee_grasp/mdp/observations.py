from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from pathlib import Path
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz, subtract_frame_transforms, matrix_from_quat, quat_from_matrix
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
import numpy as np
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def frame_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    ee_pos_b0 = ee_frame.data.target_pos_source[:, 0] 
    
    # ee_pos_b=cfg.func(ee_pos_b0, cfg)

    # print('end effector position in robot frame is:', ee_pos_b0)
    return ee_pos_b0.to(env.device)



def quat_in_robot_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
     
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    ee_quat_b = ee_frame.data.target_quat_source[:, 0]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w[:, :4]
    object_pos_w = object.data.root_pos_w[:, :3]

    _, rot0 = subtract_frame_transforms(
           object_pos_w, object_quat_w, ee_w, ee_quat_b
    )
    # print('quat in robot frame', rot0)
    # rot=cfg.func(rot0, cfg)
    return torch.tensor(rot0, dtype=torch.float32, device=env.device)

# /home/casper-3/Iiwa14_DEXEE_Grasp/source/Iiwa14_DEXEE_Grasp/Data/mujoco-Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker/coacd/mujoco-Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker.npy
# /home/casper-3/Iiwa14_DEXEE_Grasp/source/Iiwa14_DEXEE_Grasp/Data/mujoco-Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker/coacd/coffeemaker.usd
def get_grasp_config(env: ManagerBasedRLEnv,
                    ) -> torch.Tensor:
    RELATIVE_PATH = "../../source/Iiwa14_DEXEE_Grasp/Data/sem-Hammer-369593e48bdb2208419a349e9c699f76/coacd/sem-Hammer-369593e48bdb2208419a349e9c699f76.npy"
    ABSOLUTE_PATH = Path(RELATIVE_PATH).resolve().as_posix()
    data = np.load(
        ABSOLUTE_PATH, 
        allow_pickle=True
    )
    grasp = data[19]    
    scale = grasp["scale"]
                      
    final_pose = grasp["qpos"]
    grasp_quat = quat_from_euler_xyz(torch.tensor(final_pose['WRJRx']), torch.tensor(final_pose['WRJRy']), torch.tensor(final_pose['WRJRz'])).numpy()

    pose_tensor = torch.tensor([

        final_pose['WRJTx'], final_pose['WRJTy'], final_pose['WRJTz'],
        grasp_quat[0],grasp_quat[1],grasp_quat[2],grasp_quat[3],

        final_pose['F0_J0'],
        final_pose["F1_J0"],
        final_pose["F2_J0"],
        final_pose["F0_J1"],
        final_pose['F1_J1'],
        final_pose["F2_J1"],
        final_pose["F0_J2"],
        final_pose["F1_J2"],
        final_pose["F2_J2"],
        final_pose["F0_J3"],
        final_pose["F1_J3"],
        final_pose["F2_J3"],
    ], dtype=torch.float32).unsqueeze(0).to(env.device)
    pose_tensor2 = pose_tensor.repeat(env.num_envs, 1)
    # print('final pose is :', final_pose)
    # print('pose tensor is:', pose_tensor)
    return pose_tensor2


def get_open_gripper_config(env: ManagerBasedRLEnv,
                    ) -> torch.Tensor:
    RELATIVE_PATH = "../../source/Iiwa14_DEXEE_Grasp/Data/sem-Hammer-369593e48bdb2208419a349e9c699f76/coacd/sem-Hammer-369593e48bdb2208419a349e9c699f76.npy"
    ABSOLUTE_PATH = Path(RELATIVE_PATH).resolve().as_posix()
    data = np.load(
        ABSOLUTE_PATH, 
        allow_pickle=True
    )
    grasp = data[19]    
    scale = grasp["scale"]
                      
    final_pose = grasp["qpos"]
    grasp_quat = quat_from_euler_xyz(torch.tensor(final_pose['WRJRx']), torch.tensor(final_pose['WRJRy']), torch.tensor(final_pose['WRJRz'])).numpy()

    pose_tensor = torch.tensor([

        final_pose['WRJTx'], final_pose['WRJTy'], final_pose['WRJTz'],
        grasp_quat[0],grasp_quat[1],grasp_quat[2],grasp_quat[3],

        final_pose['F0_J0'],
        final_pose["F1_J0"],
        final_pose["F2_J0"],
        # final_pose['F0_J1'],
        # final_pose["F1_J1"],
        # final_pose["F2_J1"],
        -1.394,
        -1.394,
        -1.394,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], dtype=torch.float32).unsqueeze(0).to(env.device)
    pose_tensor2 = pose_tensor.repeat(env.num_envs, 1)

    return pose_tensor2


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    
    positions = initial_object_pose_robot_root_frame(env)[:,:3]
    # print('positions are:', positions)

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position in world frame
    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_quat_w[:, :4]
    # print('object_ quat is', object_quat_w)
    # Transform to robot's root frame
    object_pos_b,_ = subtract_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        object_pos_w,
        # object_quat_w
    )
    return object_pos_b.to(env.device)


# Return the orientation of the object in robot frame taken from one shot 
def object_rotation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    rotations = initial_object_pose_robot_root_frame(env)[:,3:]
    # print('rotations are:', rotations)

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w[:, :4]
    # print('object_ quat is', object_quat_w)
    # Get object position in world frame
    object_pos_w = object.data.root_pos_w[:, :3]
    
    # Transform to robot's root frame
    _,object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        object_pos_w
    )
    # print( 'what is actually returned is:',object_quat_w)
    return object_quat_w.to(env.device)



def desired_config_robot_frame(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    grasp_pos = get_grasp_config(env)               
    grasp_quat = grasp_pos[:, 3:7]                  
    grasp_p = grasp_pos[:, :3]                      
    grasp_RM = matrix_from_quat(grasp_quat)         

    # 2. object pose in robot frame
    object_P_RF = object_position_in_robot_root_frame(env)  
    object_quat_RF = object_rotation_in_robot_root_frame(env)

    object_RM = matrix_from_quat(object_quat_RF)   
    # print('object rotation matrix in robot frame:', object_RM)
    # print('desired rotation matrix in object frame', grasp_RM)
    # print('desired quat in object frame:', grasp_quat)
    # Rotation
    EE_grasp_RM_RF = torch.bmm(object_RM, grasp_RM)     
    EE_grasp_quat_RF = quat_from_matrix(EE_grasp_RM_RF) 

    # Translation: p_re = R_ro * p_oe + p_ro
    EE_grasp_P_RF = torch.bmm(object_RM, grasp_p.unsqueeze(-1)).squeeze(-1) + object_P_RF 


    EE_pose_RF0 = torch.cat([EE_grasp_P_RF, EE_grasp_quat_RF], dim=-1)  
    return EE_pose_RF0.to(env.device)



initial_pose = None

def initial_object_pose_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    
    global initial_pose
    
    # Initialize initial_pose if it's None
    # if initial_pose is None:
    #     n = env.num_envs
    #     # Initialize with default pose [0,0,0,1,0,0,0] for all environments
    #     initial_pose = torch.tensor([0, 0, 0, 1, 0, 0, 0], device='cuda').repeat(n, 1)


    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position in world frame
    object_pos_w = object.data.root_pos_w[:, :3].to(env.device)
    object_quat_w = object.data.root_quat_w[:, :4].to(env.device)

    # Transform to robot's root frame
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        object_pos_w, object_quat_w, 
        
    )
    threshold = 1.5 * env.step_dt


    # Initialize global storage
    if initial_pose is None:
        initial_pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=object.data.root_pos_w.dtype)

    t = current_time_s(env)  
    
    # Update initial_pose only in the first few timesteps
    env_ids = torch.nonzero(t <= threshold, as_tuple=True)[0].long()  # ensure LongTensor for indexing

    if env_ids.numel() > 0:
        initial_pose[env_ids, :] =torch.cat([object_pos_b[env_ids, :], object_quat_b[env_ids, :]], dim=-1).clone().detach().to(initial_pose.dtype)
    # print('in observation initial object position is::', initial_pose)
    return initial_pose

 

def current_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current time in the episode (in seconds)."""
    # print('current time is:',env.episode_length_buf.unsqueeze(1) * env.step_dt )
    return env.episode_length_buf.unsqueeze(1) * env.step_dt


# Global storage for initial configuration
initial_confff = None

def initial_robot_configuration(
    env: "ManagerBasedRLEnv",
    robot_cfg: "SceneEntityCfg" = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Returns the initial robot joint configuration for all environments.
    Updates only once per environment at the very first timestep of each episode.
    """
    global initial_confff

    asset = env.scene[robot_cfg.name]
    n_envs = env.num_envs
    t = current_time_s(env)  # shape: [num_envs]

    # Initialize storage if not yet done
    if initial_confff is None:
        num_joints = asset.data.joint_pos.shape[1]
        initial_confff = torch.zeros((n_envs, num_joints), device=asset.data.joint_pos.device, dtype=asset.data.joint_pos.dtype)

    # Find environments whose time is below threshold (first timestep)
    threshold = 1.5 * env.step_dt
    env_ids = torch.nonzero(t <= threshold, as_tuple=True)[0].long()  # ensure LongTensor for indexing

    if env_ids.numel() > 0:
        # Update only these envs and clone/detach to freeze values
        initial_confff[env_ids, :] = asset.data.joint_pos[env_ids, :].clone().detach().to(initial_confff.dtype)
   
    return initial_confff




initial_EE_pose_RF = None

def initial_EE_pose_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    
    global initial_EE_pose_RF
    robot: RigidObject = env.scene[robot_cfg.name]


    position_RF = frame_in_robot_root_frame(env)
    quat_RF = quat_in_robot_frame(env)

    threshold = 1.5 * env.step_dt


    # Initialize global storage
    if initial_EE_pose_RF is None:
        initial_EE_pose_RF = torch.zeros((env.num_envs, 7), device=env.device, dtype=robot.data.root_pos_w.dtype)

    t = current_time_s(env)  
    
    # Update initial_pose only in the first few timesteps
    env_ids = torch.nonzero(t <= threshold, as_tuple=True)[0].long()  # ensure LongTensor for indexing

    if env_ids.numel() > 0:
        initial_EE_pose_RF[env_ids, :] =torch.cat([position_RF[env_ids, :], quat_RF[env_ids, :]], dim=-1).clone().detach().to(initial_EE_pose_RF.dtype)
    # print('in observation initial robot position is::', initial_EE_pose_RF)
    return initial_EE_pose_RF


def initial_distance(
        env: ManagerBasedRLEnv,

) -> torch.Tensor:

    ggg = desired_config_robot_frame(env)[:,:3]
    ee_pos_RF = initial_EE_pose_robot_root_frame(env)[:,:3]
    object_ee_distance = torch.norm(ee_pos_RF - ggg, dim=1)/1
    # print('intial distance is:' , object_ee_distance[:10].unsqueeze(-1))

    return object_ee_distance


