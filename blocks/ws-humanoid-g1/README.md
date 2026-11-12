# Humanoid G1

A Unitree G1 humanoid (29 joints) learns to walk in MuJoCo, thousands at once on the GPU, on a flat floor or a test course (bumps, blocks, a ramp), then walks the course live in 3D.

A Unitree G1 humanoid (1.3 m, 35 kg, 29 joints) in MuJoCo: its links with their masses and inertias, its joints with their limits, its motors with their torque limits and position loops, its feet on the ground with friction (MuJoCo Menagerie and Playground).

Ground: Flat floor, or Test course: flat to 4 m, a 4 x 4 m square of bumps (0-4 cm), a 12.5 % slope up to a 0.5 m plateau at 14-17 m and down to 21 m, a 4 x 4 m square of blocks (0.4 m tiles, 0-5 cm) at 24-28 m, flat after (one heightfield; the 3D view builds the same ground).

Learn to walk: a policy (a neural network that, 50 times a second, turns what the robot senses into targets for its 29 joints: speed and turn rates, which way is down, joint angles and speeds, the speed asked, the beat of its steps; it does not see the ground) learns by trial and error (PPO) to walk at the speed asked along a line, and to turn when asked. 4096 robots on an NVIDIA GPU with MuJoCo Warp, up to 512 on the CPU (to try, not to learn). Pushes, odd starting poses and sensor noise make it robust; on the course robots start anywhere along it, walking either way. It carries on from the best policy wired in that learned on the same ground (else the best one, its critic first learning the new ground), unless Start from scratch. One episode is replayed in 3D every few seconds (channel robot); numbers go to channel train (walk_score, fell, speed_err, episode_s, distance). Finish now (control channel) stops and keeps the best policy: humanoid_walk_[course_]<date>_<NN>.pt.

Walk: the newest policy wired in (or Policy file) walks the robot live, in real time (esasim.humanoid/1). On the course it turns back at each end (Laps). From the view's bar: speed, heading, Push (0.8 m/s from the side), Reset (back to the start line). After a fall it gets up where it fell, 3 s later. Events (channel event): each part of the course, falls, laps. Frames carry its pose, the command, the IMU (gyro, accelerometer), foot contacts and loads, the ground under it and every motor's torque, for the view's head camera (humanoid: eyes, depth) and telemetry window (humanoid: telemetry).

Packages: torch, mujoco, and for the GPU mujoco-warp and warp-lang. About 100-165 million steps for a good walk: about an hour on an RTX 3070.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-humanoid-g1` · version 2 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | Walking policies (.pt) from Learn to walk: Walk drives the robot with the newest; Learn to walk carries on from the best Several connections. Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | Learn to walk: humanoid_walk_[course_]<date>_<NN>.pt, the best policy learned (network, input scaling, ground, walk score) |
| `curve` Learning curve | Table (csv) | Learn to walk: training.csv |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to walk"` |  |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU |
| `terrain` Ground | select | `"Flat floor"` | Test course: bumps, a slope up, a plateau, a slope down, blocks |
| `robots` Robots learning at once | number | `4096` | Learn to walk; at most 512 on the CPU |
| `minutes` Minutes | number | `90` | How long it learns (Learn to walk) or walks (Walk) |
| `max_speed` Top speed asked (m/s) | number | `1` |  |
| `push` Pushes up to (m/s) | number | `1.5` | A shove every 5 to 10 s while it learns; 0: none |
| `replay_every` Replay every (s) | number | `20` | How often an episode is replayed in the 3D view |
| `seed` Seed | number | `1` |  |
| `from_scratch` Start from scratch | bool | `false` | Ignore a policy wired in |
| `speed` Walking speed (m/s) | number | `0.6` | Walk: the speed it starts at (the view's slider changes it) |
| `heading` Heading (degrees) | number | `0` | Walk |
| `laps` Laps | bool | `true` | Walk on the course: turn back at each end |
| `policy_file` Policy file | text | `""` | Walk: the name of the .pt to use among those wired in (empty: the newest) |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
