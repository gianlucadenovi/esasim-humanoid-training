# Humanoid G1

A Unitree G1 humanoid (29 joints) learns to walk in MuJoCo, thousands at once on the GPU, then walks live in 3D where you steer and push it.

A Unitree G1 humanoid (1.3 m, 35 kg, 29 joints) in MuJoCo, the physics engine of robotics research: its links with their masses and inertias, its joints with their limits, its motors with their torque limits and position loops, its feet touching the ground with friction (the model of MuJoCo Menagerie and MuJoCo Playground).

Learn to walk: a policy (a neural network that, 50 times a second, turns what the robot senses into targets for its 29 joints: its speed and turn rates, which way is down, the angle and speed of each joint, the speed it is asked for and the beat of its steps) learns by trial and error (PPO) to walk at the speed asked, straight along a line, and to turn when asked. Many robots learn at once: 4096 on an NVIDIA GPU with MuJoCo Warp (the whole simulation on the card), up to 512 on the CPU (much slower: use it to try, not to learn). They are pushed now and then and start in odd poses, so the policy learns to catch itself. It is rewarded for walking at the speed asked, lifting its feet in step and keeping upright, and penalised for falling, slipping, jerking and straining its joints. Every few seconds one robot's episode is replayed in the 3D view (channel robot); the numbers go to channel train: walk_score (the share of a straight walk at the speed asked it does), fell (% of episodes ending in a fall), speed_err, episode_s, distance. Finish now (control channel) stops and keeps the best policy. The policy (.pt) goes to the policy output, the learning curve to curve. With a policy wired in, it carries on learning from it (unless Start from scratch).

Walk: the newest policy wired in (or the one named in Policy file) walks the robot live, in real time, in the 3D view (esasim.humanoid/1) along a lane; from the view's bar: speed, heading, Push (a shove of 0.8 m/s from the side) and Reset. A fall gets it back on its feet after 3 s. What happens goes to channel event.

Packages: torch, mujoco, and for the GPU mujoco-warp (with warp-lang). Expect about 100 million steps for a good walk: about an hour on a recent NVIDIA GPU.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-humanoid-g1` · version 1 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | A walking policy (.pt) from Learn to walk: Walk drives the robot with it; Learn to walk carries on from it Several connections. Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | Learn to walk: humanoid_walk_<date>_<NN>.pt, the best policy learned (network, input scaling, walk score) |
| `curve` Learning curve | Table (csv) | Learn to walk: training.csv |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to walk"` |  |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU |
| `robots` Robots learning at once | number | `4096` | Learn to walk; at most 512 on the CPU |
| `minutes` Minutes | number | `90` | How long it learns (Learn to walk) or walks (Walk) |
| `max_speed` Top speed asked (m/s) | number | `1` |  |
| `push` Pushes up to (m/s) | number | `1.5` | A shove every 5 to 10 s while it learns; 0: none |
| `replay_every` Replay every (s) | number | `20` | How often an episode is replayed in the 3D view |
| `seed` Seed | number | `1` |  |
| `from_scratch` Start from scratch | bool | `false` | Ignore a policy wired in |
| `speed` Walking speed (m/s) | number | `0.6` | Walk: the speed it starts at (the view's slider changes it) |
| `heading` Heading (degrees) | number | `0` | Walk |
| `policy_file` Policy file | text | `""` | Walk: the name of the .pt to use among those wired in (empty: the newest) |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
