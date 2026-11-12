# Humanoid G1

A Unitree G1 humanoid (29 joints) learns to walk in MuJoCo, thousands at once on the GPU, on a flat floor or round a circuit with rough ground, slopes, moving tiles and stairs, seeing the ground ahead

A Unitree G1 humanoid (1.3 m, 35 kg, 29 joints) in MuJoCo: links with their masses and inertias, joints with their limits, motors with torque limits and position loops, feet on the ground with friction (MuJoCo Menagerie and Playground).

Ground: Flat floor, or Circuit: a 38 x 18 m rectangle with rounded corners whose line the robot follows. Side 1: bumps (0-4 cm), a 12.5 % slope up to a 0.5 m plateau and down, blocks (0-5 cm). Side 2: 64 tiles that rise and fall each on its own rhythm by 2-8 cm. Side 3: a staircase, 10 steps up (12 cm), a landing, 10 down. Inside: a yard of practice stairs (6, 9, 12, 15 cm). One heightfield, the tiles as moving boxes; the 3D view builds the same ground.

Learn to walk: a policy (a neural network that, 50 times a second, turns what the robot senses, its speed and turn rates, which way is down, joint angles and speeds, the speed asked, the beat of its steps, and an elevation map of the ground from 0.3 m behind to 1.5 m ahead, as its head depth camera gives it, into targets for its 29 joints) learns by trial and error (PPO) to walk at the speed asked along its line. 4096 robots on an NVIDIA GPU with MuJoCo Warp, up to 512 on the CPU (to try, not to learn). Pushes, odd starts and sensor noise make it robust. It carries on from the best policy wired in that learned on the same ground (else the best one; a blind one gets the elevation map, its new inputs starting at zero, and the critic learns first). It ends three minutes before the run's Stop after, so the policy is always written. One episode is replayed in 3D every few seconds (channel robot); numbers go to channel train. Finish now (control channel) stops and keeps the best: humanoid_walk_[circuit_]<date>_<NN>.pt.

Walk: the newest policy wired in (or Policy file) walks the robot live round the circuit (esasim.humanoid/1), lap after lap; heading beyond 90° walks it the other way round. From the view's bar: speed, heading, Push, Reset (back to the start line). After a fall it gets up on its line, 3 s later. Events (channel event): each part, falls, laps. Frames carry its pose, command, IMU, foot loads, the ground under it, motor torques and the tiles' clock.

Packages: torch, mujoco, and for the GPU mujoco-warp and warp-lang.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-humanoid-g1` · version 4 |
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
| `policy` Policy | Files (pt) | Learn to walk: humanoid_walk_[circuit_]<date>_<NN>.pt, the best policy learned (network, input scaling, ground, walk score) |
| `curve` Learning curve | Table (csv) | Learn to walk: training.csv |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to walk"` |  |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU |
| `terrain` Ground | select | `"Flat floor"` | Circuit: rough ground, slopes, moving tiles and stairs round a rectangle, and a yard of practice stairs |
| `robots` Robots learning at once | number | `4096` | Learn to walk; at most 512 on the CPU |
| `minutes` Minutes | number | `90` | How long it learns (Learn to walk) or walks (Walk); the run's Stop after should be a little longer |
| `max_speed` Top speed asked (m/s) | number | `1` |  |
| `push` Pushes up to (m/s) | number | `1.5` | A shove every 5 to 10 s while it learns; 0: none |
| `replay_every` Replay every (s) | number | `20` | How often an episode is replayed in the 3D view |
| `seed` Seed | number | `1` |  |
| `from_scratch` Start from scratch | bool | `false` | Ignore a policy wired in |
| `speed` Walking speed (m/s) | number | `0.6` | Walk: the speed it starts at (the view's slider changes it) |
| `heading` Heading (degrees) | number | `0` | Walk: on the circuit, beyond ±90° it goes round the other way (clockwise); on the flat floor, the way it walks |
| `policy_file` Policy file | text | `""` | Walk: the name of the .pt to use among those wired in (empty: the newest) |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
