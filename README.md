# Humanoid training

A Unitree G1 humanoid (1.3 m, 35 kg, 29 joints) learns skills in a physics simulation, starting with the first one every humanoid needs: **walking straight**. Further skills (turning on the spot, walking to a point, standing up after a fall, carrying, stairs…) will be added as new policies on the same robot.

## The flows

**1 · Learn to walk** – thousands of G1s learn at once on the NVIDIA GPU (MuJoCo Warp: the whole physics runs on the card). Each one is asked to walk along a line at a speed between 0 and 1 m/s, is pushed now and then and starts in odd poses, so the policy learns to catch itself. Every 20 s one robot's attempt is replayed in the 3D view, so you can watch the gait form: first stumbles, then shuffling, then a regular walk. The charts show:

- **Walk score**: the share of a straight walk at the speed asked that the robot achieves (100% = it walked the whole episode at the speed asked, on the line)
- **Falls**: % of episodes that end with the robot on the ground
- **Speed error** (m/s) and **episode length** (s)

**Finish now** stops early and keeps the best policy. The policy goes to the dataset **humanoid-policy**.

**2 · Walk** – the policy drives the G1 live, in real time. In the 3D view's bar: the **speed** and **heading** sliders, **Push** (a shove from the side) and **Reset**. In the Dataset block, pick **humanoid-policy** (made by flow 1).

## What it takes

- A worker with an NVIDIA GPU and `torch` (CUDA build), `mujoco`, `mujoco-warp` and `warp-lang` (installed by the block before it runs).
- About 100 million steps for a good walk: roughly an hour on a recent NVIDIA GPU with 4096 robots. The CPU works for a quick try (up to 512 robots) but is far too slow to learn a walk.

## How it learns

- **Simulation**: MuJoCo, 500 Hz physics; the policy acts 50 times a second. The G1 model comes from MuJoCo Menagerie with the collision set of MuJoCo Playground: box feet with friction on the floor, capsules for the legs, hands, pelvis, torso and head when it falls.
- **Policy**: an MLP (512-256-128) that reads what a real G1 can sense (pelvis speed and turn rates, which way is down, the angle and speed of every joint, its last action, the speed asked and the phase of its gait) and sets targets for the 29 joint position loops.
- **Learning**: PPO with a critic that also sees the simulation's hidden state (foot contacts, true velocities). Rewards follow MuJoCo Playground's G1 joystick task: track the speed and turn asked, lift the feet in step, stay upright; penalties for falling, slipping, jerky actions and joints near their limits. Sensor noise, pushes and random starts make it robust.

## Credits

- 3D model: "Unitree G1 Humanoid (rigged)" by **KenzoDkohno22T**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Sketchfab); simplified to 214,000 triangles, compressed, animation removed.
- Physics model: MuJoCo Menagerie (Unitree G1, BSD-3-Clause) and MuJoCo Playground (Apache 2.0) by Google DeepMind.
- MuJoCo and MuJoCo Warp by Google DeepMind (Apache 2.0).

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to walk](blueprints/1-learn-to-walk.json) · 9 blocks · version 1
- [2 · Walk](blueprints/2-walk.json) · 6 blocks · version 1

### Workspace blocks it uses

- [Humanoid G1](blocks/ws-humanoid-g1) · version 1

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.
