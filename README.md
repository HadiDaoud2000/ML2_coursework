## Student Hadi Daoud ISU: 408037
## Target-driven dexterous grasp using Shadow Dex-ee mounted on KUKA Iiwa14

## Training scheme: 
<img width="818" height="631" alt="Screenshot from 2025-12-14 20-35-14" src="https://github.com/user-attachments/assets/70bf5133-8124-4497-a978-b5e6ef88eaf8" />

## Stages of work: 

<img width="462" height="346" alt="Screenshot from 2025-12-14 12-30-54" src="https://github.com/user-attachments/assets/0a6d6a77-8558-4d22-9ebb-26e00799dee5" />

# Reward Function Components

The reward function \( R \) consists of weighted components:

\[
\begin{aligned}
R = &\ w_{\text{reach desired position}} \cdot r_{\text{reach desired position}} \\
    + &\ w_{\text{reach desired orientation}} \cdot r_{\text{reach desired orientation}} \\
    + &\ w_{\text{reach desired pose}} \cdot r_{\text{reach desired pose}} \\
    + &\ w_{\text{reach gripper's desired joints positions}} \cdot r_{\text{reach gripper's desired joints positions}} \\
    + &\ w_{\text{all fingers-object contact}} \cdot r_{\text{all fingers-object contact}}
\end{aligned}
\]

## Component Descriptions:

- **Position Reward** (\( r_{\text{reach desired position}} \)): Rewards reaching the target position
- **Orientation Reward** (\( r_{\text{reach desired orientation}} \)): Rewards achieving the desired orientation
- **Pose Reward** (\( r_{\text{reach desired pose}} \)): Combined position and orientation reward
- **Gripper Joints Reward** (\( r_{\text{reach gripper's desired joints positions}} \)): Rewards proper gripper joint configuration
- **Contact Reward** (\( r_{\text{all fingers-object contact}} \)): Rewards establishing contact between all fingers and the object

## Weights:

Each component is weighted by its corresponding \( w \) parameter, allowing for tuning of the reward function's emphasis on different aspects of the grasping task.
