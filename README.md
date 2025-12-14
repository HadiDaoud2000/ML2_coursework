## Student Hadi Daoud ISU: 408037
## Target-driven dexterous grasp using Shadow Dex-ee mounted on KUKA Iiwa14

## Key concept:
We're training a robot to grasp objects using a simple but powerful approach using vanilla model-free RL: instead of making it figure out grasping from cameras (which is complicated and noisy), we give it the exact answers upfront—the precise hand position, orientation, and finger arrangement needed for a good grip—and use reinforcement learning to teach it how to reach that perfect grasp configuration reliably and efficiently, using only its own joint sensors without any visual input. This can make grasping learning more generalized and efficient, and can be integrated with functional grasp configurations generator away from diffusion models or vision-based that requires huge datasets and resources.
## Training scheme: 
<img width="409" height="315" alt="Screenshot from 2025-12-14 20-35-14" src="https://github.com/user-attachments/assets/70bf5133-8124-4497-a978-b5e6ef88eaf8" />

## Stages of work: 

<img width="231" height="173" alt="Screenshot from 2025-12-14 12-30-54" src="https://github.com/user-attachments/assets/0a6d6a77-8558-4d22-9ebb-26e00799dee5" />

## Results: 
some photos from the environment with different objects:

<img width="700" height="213" alt="image" src="https://github.com/user-attachments/assets/69e78002-f01e-40a0-a857-28deb9eea169" />


Reward plots for the drill with randomized position in a rectangle of 30*40 cm, and randomized initial joints positions for the manipulator:

<img width="700" height="213" alt="Screenshot from 2025-12-14 19-52-58" src="https://github.com/user-attachments/assets/506279a5-bb8d-4ee8-b247-d2ce95a8ad85" />
<img width="700" height="213" alt="Screenshot from 2025-12-14 19-46-46" src="https://github.com/user-attachments/assets/e495b87c-c01a-4c97-9cac-f7ddfeb8c0e9" />

<img width="700" height="213" alt="Screenshot from 2025-12-14 19-45-12" src="https://github.com/user-attachments/assets/d936c3f9-544a-493f-9069-b62156e1e8ce" />
<img width="700" height="213" alt="Screenshot from 2025-12-14 19-44-57" src="https://github.com/user-attachments/assets/2f8284ba-c2a4-4a1d-bc8e-1eb2450101c6" />

<img width="700" height="213" alt="Screenshot from 2025-12-14 19-44-28" src="https://github.com/user-attachments/assets/7a85f862-9cf9-4331-be5e-694eb28f6eb7" /><img width="700" height="213" alt="Screenshot from 2025-12-14 19-39-31" src="https://github.com/user-attachments/assets/30286986-fa8a-4d41-abf8-295fd12bf923" />
<img width="700" height="213" alt="Screenshot from 2025-12-14 19-39-06" src="https://github.com/user-attachments/assets/5d370a14-2522-40ba-a468-dfadbed27940" />

![demo](https://github.com/user-attachments/assets/fdc7ea4d-e8d6-4030-ab4a-94952748cd0c)

## Limitations: 
Lack of functional grasps.

Generalization.

Sim 2 real transfer.

## Future work:
Using Teacher-Student Training for Object Grasping with Only Initial Pose Knowledge. 

Generalize with more objects with different scales.

Usage of functional configurations generator.

## Installation
1- First install Isaaclab framework from the link: 
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

2- make sure to activate the environment with IsaacLab.

3- clone and build this repo 

``` bash
git clone https://github.com/HadiDaoud2000/ML2_coursework.git
cd ML2_coursework
python -m pip install -e source/Iiwa14_DEXEE_Grasp/
```

## Training 
```bash
python scripts/skrl/train.py --task Template-Iiwa14-Dexee-Grasp-v0 --num_envs 1000 --headless
```
you can change number of environments if you want

## Playing 
```bash
python scripts/skrl/play.py --task Template-Iiwa14-Dexee-Grasp-v0 --num_envs 10 --checkpoints example.pt
```
replace example.pt with the real checkpoint path.


