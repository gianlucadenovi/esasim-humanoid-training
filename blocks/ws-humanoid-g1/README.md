# Humanoid

A humanoid robot (Unitree G1, Unitree H1 or Apptronik Apollo) learns to walk in MuJoCo, thousands at once on a GPU, on a flat floor or round a circuit, and to get up after a fall

A humanoid robot in MuJoCo (models from MuJoCo Menagerie): Robot sets which. Unitree G1 (the default; 1.3 m, 35 kg, 29 joints), Unitree H1 (1.8 m, 51 kg, 19 joints) or Apptronik Apollo (1.7 m, 81 kg, 32 joints). Each robot has its own policies: their file names say which (humanoid_walk_… is the G1's, humanoid_h1_walk_…, humanoid_apollo_standup_…), a run uses only the chosen robot's and leaves the others as they are, so switching robot loses nothing.

Ground: Flat floor, or Circuit: a 38 x 18 m rectangle whose line the robot follows: bumps, a slope to a 0.5 m plateau, blocks, 64 moving tiles, 10 steps up and 10 down, and a yard of practice stairs.

Learn to walk: a policy (a neural network that, 50 times a second, turns what the robot senses, its speed and turn rates, which way is down, joint angles and speeds, the speed asked, the beat of its steps, and an elevation map of the ground ahead as its head depth camera gives it, into targets for its joints) learns by trial and error (PPO) to walk at the speed asked along its line. 4096 robots on an NVIDIA GPU with MuJoCo Warp, up to 512 on the CPU (to try, not to learn). Pushes, odd starts and sensor noise make it robust. Training: Continue carries on from this robot's policy with the most experience, among those wired in and the checkpoints stopped runs left on this machine, with its optimizer's state; Start over learns from nothing. It ends three minutes before the run's Stop after, so the policy is always written. An episode is replayed in 3D every few seconds (channel robot); numbers go to channel train. Finish now (control channel) stops and keeps the best.

Learn to stand up: each try starts with the robot lying on the flat floor where a fall left it, with 10 s to get up and stay up. At first an upward pull on its torso of 80 % of its weight makes it possible; each time it gets up in more than 60 % of tries the help goes down by 5 %, to none. Stand-up score: the share of tries it gets up from, times how little help it still needs. Several hours on a GPU.

Walk: the robot's newest policy wired in (or Policy file) walks it live round the circuit (esasim.humanoid/1), lap after lap; heading beyond 90° walks it the other way. From the view's bar: speed, heading, Push, Reset. After a fall, with a stand-up policy of this robot wired in, it gets up on its own (put back on its feet if still down after 10 s); without one, it is set on its feet 3 s later. Events (channel event): falls, laps.

The G1's rewards and settings were tuned by learning; the H1's and the Apollo's use the same task scaled to their height and weight, and may need longer or some tuning to walk as well.

Packages: torch, mujoco, and for the GPU mujoco-warp and warp-lang.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a machine, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-humanoid-g1` · version 8 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any machine, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | Policies (.pt), walking and stand-up, of any robot: each run uses only those of the robot chosen (Walk: the newest of each kind) Several connections. Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Policy | Files (pt) | humanoid_[h1_\|apollo_]walk_[circuit_]<date>_<NN>.pt or humanoid_[h1_\|apollo_]standup_<date>_<NN>.pt: the best policy (network, input scaling, optimizer, robot, ground, score) |
| `curve` Learning curve | Table (csv) | training.csv (Learn to walk) or standup.csv (Learn to stand up) |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to walk"` | Learn to walk: walking along its line at the speed asked. Learn to stand up: getting up from the floor after a fall (flat floor). Walk: live, with the newest policies |
| `robot` Robot | select | `"Unitree G1"` | G1: 1.3 m, 35 kg, 29 joints. H1: 1.8 m, 51 kg, 19 joints. Apollo: 1.7 m, 81 kg, 32 joints. Each learns and walks with its own policies only |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU |
| `terrain` Ground | select | `"Flat floor"` | Circuit: rough ground, slopes, moving tiles and stairs round a rectangle, and a yard of practice stairs (Learn to stand up always uses the flat floor) |
| `robots` Robots learning at once | number | `4096` | Learn to walk, Learn to stand up; at most 512 on the CPU |
| `minutes` Minutes | number | `90` | How long it learns or walks; it stops 3 minutes before the run's Stop after so the policy is kept |
| `max_speed` Top speed asked (m/s) | number | `1` |  |
| `push` Pushes up to (m/s) | number | `1.5` | A shove every 5 to 10 s while it learns; 0: none |
| `replay_every` Replay every (s) | number | `20` | How often an episode is replayed in the 3D view |
| `seed` Seed | number | `1` |  |
| `training` Training | select | `"Continue"` | Continue: carry on from this robot's policy with the most experience (wired in, or checkpoints stopped runs left on this machine), with its optimizer; Start over: learn from nothing |
| `speed` Walking speed (m/s) | number | `0.6` | Walk: the speed it starts at (the view's slider changes it) |
| `heading` Heading (degrees) | number | `0` | Walk: on the circuit, beyond ±90° it goes round the other way (clockwise); on the flat floor, the way it walks |
| `policy_file` Policy file | text | `""` | Walk: the name of the .pt to use among this robot's policies wired in (empty: the newest) |
| `after_fall` After a fall | select | `"Get up on its own"` | Walk: with a stand-up policy wired in, it gets up on its own (put back on its feet if still down after 10 s); else, or with Put back on its feet, it is set on its feet 3 s later |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.
