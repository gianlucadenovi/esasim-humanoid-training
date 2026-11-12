<div align="center">

<!-- Banner: upload a picture named banner.png under Data › Readme pictures (the G1 coming down the ramp, from the Side camera, works well), then: ![Humanoid training](pictures/banner.png) -->

# Humanoid training

**A Unitree G1 humanoid learns to walk by reinforcement learning, then walks a test course live in 3D: rough ground, a slope up, a slope down.**

Thousands of G1s learn at once on your GPU, in MuJoCo physics: first to walk straight on a flat floor at the speed asked, then on a course with bumps, blocks and a ramp. Then one robot walks the course live, lap after lap, steered and pushed from the view, seen from its own head camera and with every sensor and joint on a telemetry window.

![type](https://img.shields.io/badge/type-Robotics-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![method](https://img.shields.io/badge/method-Reinforcement%20learning%20(PPO)-orange) ![physics](https://img.shields.io/badge/physics-MuJoCo%20%7C%20MuJoCo%20Warp-2b6cb0) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

</div>

## Overview

Walking is hard for a humanoid: it is never statically stable. The G1 stands on two feet 18 cm long and must keep catching itself, 50 times a second, with 29 motors that only follow position targets through their own stiff loops.

Nobody writes the walking here. A neural network (the **policy**) reads what a real G1 can sense (its speed and turn rates from the IMU, which way is down, the angle and speed of every joint, what it did last, the speed it is asked for and the beat of its steps) and sets the targets for its 29 joints. It learns by trial and error (PPO, proximal policy optimisation) over hundreds of millions of steps, with thousands of robots at once on the GPU.

The policy is **blind**: it does not see the ground. On the course it feels the bumps, the blocks and the slopes only through its feet and its joints, as a G1 walking without its cameras would.

## Flows

- **1 · Learn to walk**: thousands of G1s at once on the GPU learn to walk straight on a flat floor, at speeds from standing to 1 m/s, pushed now and then and started in odd poses. One robot's episode is replayed in 3D every 20 seconds, with charts of the walk score, the falls, the episode length and the speed error. The policy goes to the **humanoid-policy** dataset.
- **2 · Walk the course**: the newest policy walks the G1 live, in real time, along the test course: flat, bumps, a slope up, the plateau, a slope down, blocks, then it turns at the end and walks back. From the main view's bar: **Push**, **Reset** (back to the start line), **speed** and **heading**. Next to it: the robot's own head camera (colour and depth), a fixed camera beside it, and the **telemetry** window.
- **3 · Learn the course**: carries on from the best flat-floor policy, now on the course: robots start anywhere along it, walking either way, so the policy learns the rough ground and the slopes. The policy goes to the same dataset (`humanoid_walk_course_…pt`), and flow 2 then walks with it.

## The robot

| Part | What is simulated |
|---|---|
| **Body** | Unitree G1, 29 joints, 30 links with their masses and inertias (about 35 kg, 1.3 m), from MuJoCo Menagerie |
| **Motors** | position loops on every joint (stiffness 75, ankles 20, ankle roll and wrists 2), each with its torque limit (up to 139 N·m at the knees) and rotor inertia |
| **Feet** | boxes 18 × 6 cm with friction on the floor (0.6) and on the course (0.8); the legs, hands, pelvis, torso and head touch the ground too when it falls |
| **Sensors** | the pelvis IMU (gyro, accelerometer, gravity), joint angles and speeds, foot contacts and loads; what the policy reads carries sensor noise while it learns |
| **Head camera** | a RealSense D435i in the head, looking down 48° (69° × 42° in colour, 0.2–5 m of depth shown): drawn in the 3D view, not used by the policy |
| **Timing** | physics at 500 Hz, the policy at 50 Hz |

## The course

| From | To | What | |
|---|---|---|---|
| 0 m | 4 m | flat floor, the start line | |
| 4 m | 8 m | **bumps**: a 4 × 4 m square of rubber granulate, 0–4 cm | a 0.4 m swell and a 0.1 m grain |
| 10 m | 14 m | **slope up**: 12.5 % (7°) | concrete with grooves |
| 14 m | 17 m | **plateau**, 0.5 m up | |
| 17 m | 21 m | **slope down**: 12.5 % | |
| 24 m | 28 m | **blocks**: a 4 × 4 m square of 0.4 m tiles, 0–5 cm high | edges a foot can catch |
| 28 m | | flat floor | Walk turns back at 32.5 m |

The physics uses it as one heightfield (a point every 5 cm); the 3D view builds the same ground from the same numbers, so what you see is what the feet touch.

## The learning

- **Rewards** (after MuJoCo Playground's G1 joystick task): walking at the speed and turn asked, lifting the feet in step with a gait clock, staying up; penalties for tilting, slipping, jerky actions, joints near their limits and straying from the standing pose. A step never pays less than nothing, so staying up always beats falling.
- **Commands**: mostly along a line (a guide steers it back onto the line, as a driver with a joystick would), sometimes random turns and side steps, sometimes standing still.
- **Robustness**: pushes of up to 1.5 m/s every 5–10 s, starting poses with every joint off by up to 50 %, noise on what it senses.
- **Score**: the walk score is how well it keeps the speed asked times the share of episodes it does not fall (100 % = always on its feet, always at the speed asked).

## Where it runs

- **NVIDIA GPU**: the way to train. MuJoCo Warp runs the whole simulation on the card. On an RTX 3070 with 4,096 robots, flow 1 ran at 25,000–50,000 steps a second. The robots stayed on their feet for most of an episode after about 45 minutes, and the walk score levelled off at 84–85 % after about an hour (165 million steps).
- **CPU**: works, with at most 512 robots at once; about 1,000–3,000 steps a second, fine to try it, far too slow to learn a walk.
- Flow 2 runs on any machine: one robot, in real time, on the CPU.

## What you need

- A machine with the EsaSim worker (Project › Compute), with an NVIDIA GPU for training.
- Python packages, installed by the worker the first time: `torch` (with CUDA on NVIDIA machines), `mujoco`, `mujoco-warp`, `warp-lang`. Nothing else is downloaded and no account or key is needed.
- Somewhere for the policies: the **humanoid-policy** dataset, made by the first training run.

## How to run it

**If you took this project from Demos**, first tell it where your policies live (a copy keeps no link to anyone else's storage):

1. Open **1 · Learn to walk** under **Blueprint**, select the training block and set **Outputs go to** to a place in your workspace, with the dataset name `humanoid-policy`. Do the same in **3 · Learn the course**.
2. Once the dataset exists, set the **Policies** block of flows 2 and 3 to it.

Then:

1. Under **Compute**, check that a machine with an NVIDIA GPU is available for this project.
2. Run **1 · Learn to walk** (90 minutes by default; **Finish now** ends it early and still keeps the best policy).
3. Run **2 · Walk the course** and watch the flat-floor policy on the course. Use the bar to push it, change its speed or reset it.
4. Run **3 · Learn the course** to teach it the rough ground and the slopes, then run flow 2 again: it walks with the newest policy.

## Settings worth knowing

- **Learn to walk / Learn the course**: **Ground** (Flat floor or Test course), **Robots learning at once** (4,096 on a GPU), **Minutes** (90 and 60), **Top speed asked** (1 m/s), **Pushes up to** (1.5 m/s), **Replay every** (20 s), **Start from scratch** (off: it carries on from the best policy on the same ground, or else the best one).
- **Walk**: **Ground**, **Walking speed** (0.6 m/s to start), **Heading**, **Laps** (turn back at each end of the course), **Policy file** (empty: the newest), **Minutes** (30).
- In the 3D view: the bar switches cameras (Director, Follow, Side, Front, Orbit, Free, Robot's eyes, Depth camera), and turns Sound and Sensors on or off. Sensors shows the speed asked and measured, the turn, how far off the line it is, what it walks on, its height above the ground and the load on each foot.
- Fixed views for a wall of windows: `humanoid: follow`, `side`, `front`, `eyes`, `depth` and `telemetry` (Camera in the view's settings).

## What you get

- `humanoid_walk_<date>_<NN>.pt` and `humanoid_walk_course_<date>_<NN>.pt` in the **humanoid-policy** dataset: the policy (network, input scaling, the ground it learned on, its walk score NN %, its steps of experience).
- **training.csv**: the learning curve (walk score, falls, speed error, episode length, steps per second).
- Live in the views: the replayed episodes while learning; the walk on the course with the head camera, the depth camera and the telemetry window (speed and turn asked and measured, height above the ground, attitude, gyro, accelerometer, foot loads and contacts, leg angles and torques, and a board of all 29 joints); captions as it reaches each part of the course, falls, or finishes a lap.

## Data and licences

- No dataset is read: every step is simulated.
- G1 physics model: MuJoCo Menagerie (Unitree G1, BSD-3-Clause), with the collision set and task design of MuJoCo Playground (Apache 2.0), both by Google DeepMind. MuJoCo and MuJoCo Warp by Google DeepMind (Apache 2.0).
- 3D model: "Unitree G1 Humanoid (rigged)" by **KenzoDkohno22T**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Sketchfab); simplified to 214,000 triangles, compressed, animation removed.
- The code of the blocks is EsaSim's.

## Notes

- A policy that learned only on the flat floor already walks the course at 0.6 m/s; at 1 m/s it sometimes trips on the blocks. **3 · Learn the course** is there to close that gap.
- The policy is blind: a version that reads the ground ahead (a height scan, or the depth camera) is the natural next skill, and so are stairs.
- Falls in **Walk** are decided by the physics: it gets back on its feet 3 s later where it fell, a step further on.
- The course, the head camera and the telemetry window need the 3D view 1.50 or later.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to walk](blueprints/1-learn-to-walk.json) · 9 blocks · version 3
- [2 · Walk the circuit](blueprints/2-walk-the-circuit.json) · 9 blocks · version 4
- [3 · Learn the circuit](blueprints/3-learn-the-circuit.json) · 11 blocks · version 2

### Workspace blocks it uses

- [Humanoid G1](blocks/ws-humanoid-g1) · version 3

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
