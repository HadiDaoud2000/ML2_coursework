## Student Hadi Daoud ISU: 408037
## Target-driven dexterous grasp using Shadow Dex-ee mounted on KUKA Iiwa14

## Training scheme: 
<img width="818" height="631" alt="Screenshot from 2025-12-14 20-35-14" src="https://github.com/user-attachments/assets/70bf5133-8124-4497-a978-b5e6ef88eaf8" />

## Stages of work: 

<img width="462" height="346" alt="Screenshot from 2025-12-14 12-30-54" src="https://github.com/user-attachments/assets/0a6d6a77-8558-4d22-9ebb-26e00799dee5" />

# Reward Function

\[
R = w_p \cdot r_p + w_o \cdot r_o + w_{\text{pose}} \cdot r_{\text{pose}} + w_g \cdot r_g + w_c \cdot r_c
\]

Where:
- \( r_p \): Reach desired position reward
- \( r_o \): Reach desired orientation reward  
- \( r_{\text{pose}} \): Reach desired pose reward
- \( r_g \): Reach gripper's desired joints positions reward
- \( r_c \): All fingers-object contact reward
- \( w_* \): Corresponding weight parameters
