<div align="center">

<!-- Banner: upload a picture named banner.png under Data › Readme pictures (the G1 climbing the circuit's stairs, from the Side camera, works well), then: ![Humanoid training](pictures/banner.png) -->

# Humanoid training

**A humanoid robot (Unitree G1 by default, or a Unitree H1 or an Apptronik Apollo) learns to walk by reinforcement learning, then walks a circuit live in 3D: rough ground, slopes, moving tiles and a staircase of 10 steps up and 10 down.**

Thousands of G1s learn at once on your GPU, in MuJoCo physics: first to walk straight on a flat floor at the speed asked, then round a rectangular circuit, following its line, reading the ground ahead from the depth camera in its head. Then one robot walks the circuit live, lap after lap, steered and pushed from the view, seen from its own head camera and with every sensor and joint on a telemetry window.

![type](https://img.shields.io/badge/type-Robotics-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![method](https://img.shields.io/badge/method-Reinforcement%20learning%20(PPO)-orange) ![physics](https://img.shields.io/badge/physics-MuJoCo%20%7C%20MuJoCo%20Warp-2b6cb0) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

</div>

## Overview

Walking is hard for a humanoid: it is never statically stable. The G1 stands on two feet 18 cm long and must keep catching itself, 50 times a second, with 29 motors that only follow position targets through their own stiff loops.

Nobody writes the walking here. A neural network (the **policy**) reads what a real G1 can sense (its speed and turn rates from the IMU, which way is down, the angle and speed of every joint, what it did last, the speed it is asked for and the beat of its steps) and sets the targets for its 29 joints. It learns by trial and error (PPO, proximal policy optimisation) over hundreds of millions of steps, with thousands of robots at once on the GPU.

The policy also **sees the ground ahead**: an elevation map of 13 × 5 points, from 0.3 m behind its feet to 1.5 m ahead and 0.3 m either side, as the depth camera in its head gives it (with a couple of centimetres of noise and offset, as a real one would). On a flat floor the map is flat and changes nothing; on stairs it is what lets it place its feet on the treads instead of finding the risers with its toes.

## Flows

- **1 · Learn to walk**: thousands of G1s at once on the GPU learn to walk straight on a flat floor, at speeds from standing to 1 m/s, pushed now and then and started in odd poses. One robot's episode is replayed in 3D every 20 seconds, with charts of the walk score, the falls, the episode length and the speed error. The policy goes to the **humanoid-policy** dataset.
- **2 · Learn the circuit**: carries on from the best policy so far, now on the circuit: robots start anywhere on its line (more of them near the stairs and the moving tiles), either way round, and a third of them in the practice yard. The policy goes to the same dataset (`humanoid_walk_circuit_…pt`), and flow 3 then tests it.
- **3 · Circuit test**: the newest policy walks the G1 live, in real time, round the circuit, lap after lap: the rough ground and the ramp on side 1, the moving tiles on side 2, the staircase on side 3. From the main view's bar: **Push**, **Reset** (back to the start line), **speed** and **heading** (beyond ±90° it turns round and walks the circuit clockwise). Next to it: the robot's own head camera (colour and depth), cameras behind and in front of it, and the **telemetry** window. Captions say what it walks on, when it falls and each lap's time.
- **4 · Learn to stand up**: robots lying where a fall left them learn to get up, with an upward help on the torso that is taken away as they succeed. The stand-up policy goes to the same dataset, and Circuit test uses it to get up on its own after a fall.

## The robot

| Part | What is simulated |
|---|---|
| **Body** | Unitree G1, 29 joints, 30 links with their masses and inertias (about 35 kg, 1.3 m), from MuJoCo Menagerie |
| **Motors** | position loops on every joint (stiffness 75, ankles 20, ankle roll and wrists 2), each with its torque limit (up to 139 N·m at the knees) and rotor inertia |
| **Feet** | boxes 18 × 6 cm with friction on the floor (0.6), on the circuit and on the moving tiles (0.8); the legs, hands, pelvis, torso and head touch the ground too when it falls |
| **Sensors** | the pelvis IMU (gyro, accelerometer, gravity), joint angles and speeds, foot contacts and loads; what the policy reads carries sensor noise while it learns |
| **Head camera** | a RealSense D435i in the head, looking down 48° (69° × 42° in colour, 0.2–5 m of depth shown); the policy reads the ground from it as the elevation map, shown as dots under **Sensors** in the 3D view |
| **Timing** | physics at 500 Hz, the policy at 50 Hz |

### Choosing the robot

The G1 is the default. Every **Humanoid** block has a **Robot** setting (select the block in the blueprint of each flow) with three choices:

| Robot | Size | Joints |
|---|---|---|
| **Unitree G1** (default) | 1.3 m, 35 kg | 29 |
| **Unitree H1** | 1.8 m, 51 kg | 19 |
| **Apptronik Apollo** | 1.7 m, 81 kg | 32 |

Each robot keeps its own policies, and the file names say whose they are: `humanoid_walk_…` and `humanoid_standup_…` are the G1's, `humanoid_h1_walk_…` and `humanoid_apollo_standup_…` the others'. They all go to the same **humanoid-policy** dataset; a run uses only the policies of the robot it is set to and leaves the others as they are, so you can switch back and forth and lose nothing. Set the same robot in the flow that learns and in the flow that walks it. The 3D view draws whichever robot the run sends.

The rewards and settings were tuned on the G1. The H1 and the Apollo learn the same task scaled to their height and weight: expect them to need longer, and perhaps some tuning, to walk as well.

## The circuit

A 38 × 18 m rectangle with rounded corners (2.5 m), 108 m round, walked counterclockwise from the start line (or clockwise). A grey lane marks its line; the robot follows it with a guide, as a driver with a joystick would.

| Where | What | |
|---|---|---|
| **Side 1** | flat start, then a 4 × 4 m square of **bumps** (0–4 cm), a **slope up** of 12.5 % (7°) to a **plateau** 0.5 m up, a **slope down**, a 4 × 4 m square of **blocks** (0.4 m tiles, 0–5 cm) | rubber granulate, grooved concrete, edges a foot can catch |
| **Side 2** | **moving tiles**: a 4 × 4 m square of 64 tiles, 0.5 m each, every one rising and falling on its own rhythm (2–8 cm, every 1.5–4 s) | in training a new pattern every episode; in Walk always the same one |
| **Side 3** | a **staircase** 3 m wide: **10 steps up** (12 cm risers, 30 cm treads), a 2 m landing 1.2 m up, **10 steps down** | yellow nosings and handrails |
| **Side 4** | flat, back to the start line | |
| **Inside** | the **practice yard**: four staircases of 10 steps up and 10 down, with 6, 9, 12 and 15 cm risers, 1.8 m wide | labelled "PRACTICE n cm"; used only in training |

The physics uses the ground as one heightfield (a point every 5 cm) and the moving tiles as 64 boxes moved each step; the 3D view builds the same ground from the same numbers and moves the tiles on the same clock, so what you see is what the feet touch.

## The learning

- **Rewards** (after MuJoCo Playground's G1 joystick task): walking at the speed and turn asked, lifting the feet in step with a gait clock, staying up; penalties for tilting, slipping, jerky actions, joints near their limits and straying from the standing pose. A step never pays less than nothing, so staying up always beats falling.
- **Commands**: mostly along its line (the circuit's, or a yard staircase's), sometimes random turns and side steps, sometimes standing still.
- **Robustness**: pushes of up to 1.5 m/s every 5–10 s, starting poses with every joint off by up to 50 %, noise on what it senses and on the elevation map.
- **Carrying on**: flow 2 starts from the best flat-floor policy. A policy that learned blind gets the elevation map as new inputs that start at zero, so it walks exactly as before; the critic first learns what the new ground is worth (25 iterations), then the policy adapts.
- **Score**: the walk score is how well it keeps the speed asked times the share of episodes it does not fall (100 % = always on its feet, always at the speed asked).

## Where it runs

- **NVIDIA GPU**: the way to train. MuJoCo Warp runs the whole simulation on the card, moving tiles included. On an RTX 3070 with 4,096 robots, flow 1 ran at 25,000–50,000 steps a second, and the walk score levelled off at 84–85 % after about an hour (165 million steps). Flow 2 has much more to learn (stairs above all): give it 2 hours or more and watch the falls chart.
- **CPU**: works, with at most 512 robots at once; about 1,000–3,000 steps a second, fine to try it, far too slow to learn a walk.
- Flow 3 runs on any machine: one robot, in real time, on the CPU.

## What you need

- A machine with the EsaSim worker (Project › Compute), with an NVIDIA GPU for training.
- Python packages, installed by the worker the first time: `torch` (with CUDA on NVIDIA machines), `mujoco`, `mujoco-warp`, `warp-lang`. Nothing else is downloaded and no account or key is needed.
- Somewhere for the policies: the **humanoid-policy** dataset, made by the first training run.

## How to run it

**If you took this project from Demos**, first tell it where your policies live (a copy keeps no link to anyone else's storage):

1. Open **1 · Learn to walk** under **Blueprint**, select the training block and set **Outputs go to** to a place in your workspace, with the dataset name `humanoid-policy`. Do the same in **2 · Learn the circuit**.
2. Once the dataset exists, set the **Policies** block of flows 2 and 3 to it.

Then:

1. Under **Compute**, check that a machine with an NVIDIA GPU is available for this project.
2. Run **1 · Learn to walk** (90 minutes by default; **Finish now** ends it early and still keeps the best policy).
3. Run **2 · Learn the circuit** (120 minutes) to teach it the circuit: the stairs, the moving tiles, the slopes and the rough ground.
4. Run **3 · Circuit test** and watch it go round, lap after lap. It always walks with the newest policy, so you can also run it between 1 and 2: the flat-floor policy manages the corners, the slopes and the tiles, but falls on the stairs, having never seen one.

To train another robot, set **Robot** on the Humanoid block of each flow you run (the same one everywhere), then follow the same steps.

## Settings worth knowing

- **All flows**: **Robot** (Unitree G1, Unitree H1 or Apptronik Apollo).
- **Learn to walk / Learn the circuit**: **Ground** (Flat floor or Circuit), **Robots learning at once** (4,096 on a GPU), **Minutes** (90 and 120), **Top speed asked** (1 m/s), **Pushes up to** (1.5 m/s), **Replay every** (20 s), **Training** (Continue: it carries on from this robot's policy with the most experience; Start over: from nothing).
- **Walk** (Circuit test): **Ground**, **Walking speed** (0.6 m/s to start), **Heading** (beyond ±90°: clockwise), **Policy file** (empty: the newest), **After a fall**, **Minutes** (30).
- In the 3D view: the bar switches cameras (Director, Follow, Side, Front, Orbit, Free, Robot's eyes, Depth camera), and turns Sound and Sensors on or off. Sound is the footsteps and a faint motor hum; switched off, it is silent. Sensors shows the speed asked and measured, the turn, how far off the line it is, what it walks on, its height above the ground, the load on each foot and the elevation map's dots.
- Fixed views for a wall of windows: `humanoid: follow`, `side`, `front`, `eyes`, `depth` and `telemetry` (Camera in the view's settings). They are silent.

## What you get

- `humanoid_walk_<date>_<NN>.pt` and `humanoid_walk_circuit_<date>_<NN>.pt` in the **humanoid-policy** dataset (with `h1_` or `apollo_` after `humanoid_` for the other robots): the policy (network, input scaling, the robot, the ground it learned on, its walk score NN %, its steps of experience).
- **training.csv**: the learning curve (walk score, falls, speed error, episode length, steps per second).
- Live in the views: the replayed episodes while learning (the stairs, the yard, the tiles); the walk round the circuit with the head camera, the depth camera and the telemetry window (speed and turn asked and measured, height above the ground, attitude, gyro, accelerometer, foot loads and contacts, leg angles and torques, and a board of all its joints); captions as it reaches each part of the circuit (stairs up or down, the tiles, the slopes), falls, or finishes a lap.

## Data and licences

- No dataset is read: every step is simulated.
- G1 physics model: MuJoCo Menagerie (Unitree G1, BSD-3-Clause), with the collision set and task design of MuJoCo Playground (Apache 2.0), both by Google DeepMind. MuJoCo and MuJoCo Warp by Google DeepMind (Apache 2.0).
- Unitree H1 (Unitree Robotics, BSD-3-Clause) and Apptronik Apollo (Apptronik, Apache 2.0): MuJoCo Menagerie models, their 3D look simplified from the same meshes.
- 3D model: "Unitree G1 Humanoid (rigged)" by **KenzoDkohno22T**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Sketchfab); simplified to 214,000 triangles, compressed, animation removed.
- The code of the blocks is EsaSim's.

## Notes

- A policy that learned only on the flat floor already goes round the corners and over the moving tiles at 0.6 m/s, but it trips on the stairs. **2 · Learn the circuit** is there to close that gap; the practice yard lets it learn low steps before the 12 cm ones.
- The elevation map is read from the simulation's ground under the points the camera sees, with noise; a real robot would build it from the depth images.
- Falls in **Walk** are decided by the physics: it gets back on its feet 3 s later on the circuit's line, a step further on.
- The circuit, the stairs, the moving tiles and the elevation map need the 3D view 1.51 or later; the H1 and the Apollo need 1.56.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to walk](blueprints/1-learn-to-walk.json) · 9 blocks · version 5
- [2 · Learn the circuit](blueprints/2-learn-the-circuit.json) · 11 blocks · version 5
- [3 · Circuit test](blueprints/3-circuit-test.json) · 9 blocks · version 5
- [4 · Learn to stand up](blueprints/4-learn-to-stand-up.json) · 10 blocks · version 1

### Workspace blocks it uses

- [Humanoid](blocks/ws-humanoid-g1) · version 7

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
