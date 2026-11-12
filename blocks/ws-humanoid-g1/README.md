# Humanoid G1

A Unitree G1 humanoid (29 joints) learns to walk in MuJoCo, thousands at once on the GPU, on a flat floor or round a circuit seeing the ground ahead, and learns to get up on its own after a fall

A Unitree G1 humanoid (1.3 m, 35 kg, 29 joints) in MuJoCo: links with their masses and inertias, joints with their limits, motors with torque limits and position loops, feet on the ground with friction (MuJoCo Menagerie and Playground).

Ground: Flat floor, or Circuit: a 38 x 18 m rectangle with rounded corners whose line the robot follows. Side 1: bumps (0-4 cm), a 12.5 % slope up to a 0.5 m plateau and down, blocks (0-5 cm). Side 2: 64 tiles that rise and fall each on its own rhythm by 2-8 cm. Side 3: a staircase, 10 steps up (12 cm), a landing, 10 down. Inside: a yard of practice stairs (6, 9, 12, 15 cm). One heightfield, the tiles as moving boxes; the 3D view builds the same ground.

Learn to walk: a policy (a neural network that, 50 times a second, turns what the robot senses, its speed and turn rates, which way is down, joint angles and speeds, the speed asked, the beat of its steps, and an elevation map of the ground from 0.3 m behind to 1.5 m ahead, as its head depth camera gives it, into targets for its 29 joints) learns by trial and error (PPO) to walk at the speed asked along its line. 4096 robots on an NVIDIA GPU with MuJoCo Warp, up to 512 on the CPU (to try, not to learn). Pushes, odd starts and sensor noise make it robust. Training: Continue carries on from the policy with the most experience, among those wired in and the checkpoints stopped runs left on this machine (those that learned the same ground first; a blind one gets the elevation map, its new inputs starting at zero, and the critic learns first), with its optimizer's state; Start over learns from nothing. It ends three minutes before the run's Stop after, so the policy is always written. One episode is replayed in 3D every few seconds (channel robot); numbers go to channel train. Finish now (control channel) stops and keeps the best: humanoid_walk_[circuit_]<date>_<NN>.pt.

Learn to stand up: each try starts with the robot lying on the flat floor where a fall left it (poses made by dropping it from a random pose, any way round), with 10 s to get up and stay up; rewards: pelvis high, torso upright, then the standing pose, both feet down and stillness. At first an upward pull on its torso of 80 % of its weight (while it is low) makes it possible; each time it gets up in more than 60 % of tries the help goes down by 5 %, to none. Stand-up score: the share of tries it gets up from, times how little help it still needs. Numbers go to channel train (stand_score, up_pct, time_to_up, assist); a try is replayed in 3D every few seconds: humanoid_standup_<date>_<NN>.pt. Several hours on a GPU for it to get up with no help.

Walk: the newest policy wired in (or Policy file) walks the robot live round the circuit (esasim.humanoid/1), lap after lap; heading beyond 90° walks it the other way round. From the view's bar: speed, heading, Push, Reset (back to the start line). After a fall, with a stand-up policy wired in, it gets up on its own and walks on from there (put back on its feet i

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-humanoid-g1` · version 5 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | Policies (.pt): walking ones (Learn to walk) and stand-up ones (Learn to stand up). Walk drives the robot with the newest walking one and gets it up with the newest stand-up one; each training carries Several connections. Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | humanoid_walk_[circuit_]<date>_<NN>.pt (Learn to walk) or humanoid_standup_<date>_<NN>.pt (Learn to stand up): the best policy learned (network, input scaling, optimizer, ground, score) |
| `curve` Learning curve | Table (csv) | training.csv (Learn to walk) or standup.csv (Learn to stand up) |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to walk"` | Learn to walk: walking along its line at the speed asked. Learn to stand up: getting up from the floor after a fall (flat floor). Walk: live, with the newest policies |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU |
| `terrain` Ground | select | `"Flat floor"` | Circuit: rough ground, slopes, moving tiles and stairs round a rectangle, and a yard of practice stairs (Learn to stand up always uses the flat floor) |
| `robots` Robots learning at once | number | `4096` | Learn to walk, Learn to stand up; at most 512 on the CPU |
| `minutes` Minutes | number | `90` | How long it learns or walks; it stops 3 minutes before the run's Stop after so the policy is kept |
| `max_speed` Top speed asked (m/s) | number | `1` |  |
| `push` Pushes up to (m/s) | number | `1.5` | A shove every 5 to 10 s while it learns; 0: none |
| `replay_every` Replay every (s) | number | `20` | How often an episode is replayed in the 3D view |
| `seed` Seed | number | `1` |  |
| `training` Training | select | `"Continue"` | Continue: carry on from the policy with the most experience (those wired in, and the checkpoints that stopped runs left on this machine), with its optimizer; Start over: a new policy that learns from nothing |
| `speed` Walking speed (m/s) | number | `0.6` | Walk: the speed it starts at (the view's slider changes it) |
| `heading` Heading (degrees) | number | `0` | Walk: on the circuit, beyond ±90° it goes round the other way (clockwise); on the flat floor, the way it walks |
| `policy_file` Policy file | text | `""` | Walk: the name of the .pt to use among those wired in (empty: the newest) |
| `after_fall` After a fall | select | `"Get up on its own"` | Walk: with a stand-up policy wired in, it gets up on its own (put back on its feet if still down after 10 s); else, or with Put back on its feet, it is set on its feet 3 s later |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
