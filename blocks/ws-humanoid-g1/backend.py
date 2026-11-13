# ── EsaSim block ──────────────────────────────────────────────────────
# Humanoid (ws-humanoid-g1) · version 7 · machine block, Python: def run(ctx)
# A humanoid robot (Unitree G1, Unitree H1 or Apptronik Apollo) learns to walk in MuJoCo, thousands at once on the GPU, on a flat floor or round a circuit seeing the ground ahead, and learns to get up o
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-13
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Policy · several · optional · Policies (.pt), walking and stand-up, of any robot: each run uses only those of the robot chosen (Walk: the newest of each kind)
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Policy · humanoid_[h1_|apollo_]walk_[circuit_]<date>_<NN>.pt or humanoid_[h1_|apollo_]standup_<date>_<NN>.pt: the best policy (network, input scaling, optimizer, robot, ground, score)
#   curve       Table (csv)               Learning curve · training.csv (Learn to walk) or standup.csv (Learn to stand up)
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    What to do = "Learn to walk" of Learn to walk | Learn to stand up | Walk
#   robot       select    Robot = "Unitree G1" of Unitree G1 | Unitree H1 | Apptronik Apollo
#   accelerator select    Run on = "auto" of auto | NVIDIA GPU | CPU
#   terrain     select    Ground = "Flat floor" of Flat floor | Circuit
#   robots      number    Robots learning at once = 4096
#   minutes     number    Minutes = 90
#   max_speed   number    Top speed asked (m/s) = 1
#   push        number    Pushes up to (m/s) = 1.5
#   replay_every number    Replay every (s) = 20
#   seed        number    Seed = 1
#   training    select    Training = "Continue" of Continue | Start over
#   speed       number    Walking speed (m/s) = 0.6
#   heading     number    Heading (degrees) = 0
#   policy_file text      Policy file = ""
#   after_fall  select    After a fall = "Get up on its own" of Get up on its own | Put back on its feet
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch, mujoco>=3.3, mujoco-warp, warp-lang (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Humanoid training: a humanoid robot (by default the Unitree G1, 29 joints, 35 kg; with Robot, the Unitree H1 or the
Apptronik Apollo) learns to walk in MuJoCo, the physics engine robotics
research uses, on a flat floor or round a circuit (bumps, a ramp up to a plateau and down, blocks, moving tiles, a
staircase of 10 steps up and 10 down, and a yard of practice stairs). Many robots learn at once on the NVIDIA GPU (MuJoCo
Warp), or on the CPU. The robot's model: its 30 links with their masses and inertias, its joints with their limits, the
motors with their torque limits and the gains of their position loops (from MuJoCo Menagerie and MuJoCo Playground,
Apache 2.0 / BSD); the feet touch the ground as boxes with friction; the legs, the hands, the pelvis, the torso and the
head touch it too when it falls.

What to do: Learn to walk: the policy (a neural network that, 50 times a second, turns what the robot senses, and an
elevation map of the ground ahead from its head camera, into targets for its 29 joints) learns by trial and error (PPO)
to walk at the speed asked, along its line. Walk: the newest policy drives the robot live in the 3D view
(esasim.humanoid/1) round the circuit, lap after lap: set the speed and the way round, push it, reset it; its frames
carry its IMU, foot loads, motor torques and the tiles' clock for the head camera and telemetry views. Learn to stand
up: lying where a fall left it, it learns to get up on its own (with an upward help on its torso at first, taken away as
it succeeds); Walk then uses that policy after each fall. Training: Continue carries on from the newest policy (also one
a stopped run left on the machine), Start over learns from nothing."""
import math


# the robot: Unitree G1, 29 joints (MuJoCo Menagerie, BSD-3; collision set of MuJoCo Playground, Apache 2.0), meshes left out
MODEL_XML = r'''<mujoco model="g1_29dof"><compiler angle="radian" /><option iterations="3" ls_iterations="5" timestep=".002" integrator="Euler"><flag eulerdamp="disable" /></option><default><default class="g1"><geom condim="1" contype="0" conaffinity="0" /><joint frictionloss="0.1" damping="2" /><position inheritrange="1" kp="75" /><default class="hip"><default class="hip_pitch"><joint axis="0 1 0" range="-2.5307 2.8798" actuatorfrcrange="-88 88" armature="0.01017752004" /></default><default class="hip_roll"><joint axis="1 0 0" range="-0.5236 2.9671" actuatorfrcrange="-139 139" armature="0.025101925" /></default><default class="hip_yaw"><joint axis="0 0 1" range="-2.7576 2.7576" actuatorfrcrange="-88 88" armature="0.01017752004" /></default></default><default class="knee"><joint axis="0 1 0" range="-0.087267 2.8798" actuatorfrcrange="-139 139" armature="0.025101925" /></default><default class="ankle"><default class="ankle_pitch"><position kp="20" /><joint axis="0 1 0" range="-0.87267 0.5236" actuatorfrcrange="-50 50" damping="1" armature="0.00721945" /></default><default class="ankle_roll"><position kp="2" /><joint axis="1 0 0" range="-0.2618 0.2618" actuatorfrcrange="-50 50" damping=".2" armature="0.00721945" /></default></default><default class="waist_yaw"><joint axis="0 0 1" range="-2.618 2.618" actuatorfrcrange="-88 88" armature="0.01017752004" /></default><default class="waist_pitch"><joint axis="0 1 0" range="-0.52 0.52" actuatorfrcrange="-50 50" armature="0.00721945" /></default><default class="waist_roll"><joint axis="1 0 0" range="-0.52 0.52" actuatorfrcrange="-50 50" armature="0.00721945" /></default><default class="shoulder"><default class="shoulder_pitch"><joint axis="0 1 0" range="-3.0892 2.6704" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="shoulder_roll"><joint axis="1 0 0" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="shoulder_yaw"><joint axis="0 0 1" range="-2.618 2.618" actuatorfrcrange="-25 25" armature="0.003609725" /></default></default><default class="elbow"><joint axis="0 1 0" range="-1.0472 2.0944" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="wrist"><position kp="2" /><joint damping=".2" /><default class="wrist_roll"><joint axis="1 0 0" range="-1.97222 1.97222" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="wrist_pitch"><joint axis="0 1 0" range="-1.61443 1.61443" actuatorfrcrange="-5 5" armature="0.00425" /></default><default class="wrist_yaw"><joint axis="0 0 1" range="-1.61443 1.61443" actuatorfrcrange="-5 5" armature="0.00425" /></default></default><default class="visual"><geom group="2" type="mesh" density="0" material="silver" /></default><default class="collision"><geom group="3" rgba=".2 .6 .2 .3" /><default class="foot"><geom size="0.085 0.03 0.005" /></default></default><site group="5" rgba="1 0 0 1" /></default></default><asset><hfield name="terrain" nrow="581" ncol="921" size="23 14.5 2 0.1" /><material name="silver" rgba="0.7 0.7 0.7 1" /><material name="black" rgba="0.2 0.2 0.2 1" /></asset><worldbody><geom name="floor" size="0 0 0.01" type="plane" pos="0 0 -0.001" /><geom name="terrain" type="hfield" hfield="terrain" pos="15 7.5 -0.05" /><light name="spotlight" mode="targetbodycom" target="pelvis" pos="0 -2 4" exponent="20" /><body name="pelvis" pos="0 0 0.793" childclass="g1"><geom name="pelvis_col" class="collision" type="capsule" size="0.07" fromto="0 -0.07 -0.06 0 0.07 -0.06" /><camera name="track" pos="1.734 -1.135 .35" xyaxes="0.552 0.834 -0.000 -0.170 0.112 0.979" mode="trackcom" /><inertial pos="0 0 -0.07605" quat="1 0 -0.000399148 0" mass="3.813" diaginertia="0.010549 0.0093089 0.0079184" /><freejoint name="floating_base_joint" /><site name="imu_in_pelvis" size="0.01" pos="0.04525 0 -0.08339" /><body name="left_hip_pitch_link" pos="0 0.064452 -0.1027"><inertial pos="0.002741 0.047791 -0.02606" quat="0.954862 0.293964 0.0302556 0.030122" mass="1.35" diaginertia="0.00181517 0.00153422 0.00116212" /><joint name="left_hip_pitch_joint" class="hip_pitch" /><body name="left_hip_roll_link" pos="0 0.052 -0.030465" quat="0.996179 0 -0.0873386 0"><inertial pos="0.029812 -0.001045 -0.087934" quat="0.977808 -1.97119e-05 0.205576 -0.0403793" mass="1.52" diaginertia="0.00254986 0.00241169 0.00148755" /><joint name="left_hip_roll_joint" class="hip_roll" /><geom name="left_thigh" class="collision" type="capsule" size="0.06" fromto="0.02 0 0 0.02 0 -0.2" /><body name="left_hip_yaw_link" pos="0.025001 0 -0.12412"><inertial pos="-0.057709 -0.010981 -0.15078" quat="0.600598 0.15832 0.223482 0.751181" mass="1.702" diaginertia="0.00776166 0.00717575 0.00160139" /><joint name="left_hip_yaw_joint" class="hip_yaw" /><body name="left_knee_link" pos="-0.078273 0.0021489 -0.17734" quat="0.996179 0 0.0873386 0"><inertial pos="0.005457 0.003964 -0.12074" quat="0.923418 -0.0327699 0.0158246 0.382067" mass="1.932" diaginertia="0.0113804 0.0112778 0.00146458" /><joint name="left_knee_joint" class="knee" /><geom name="left_shin" class="collision" type="capsule" size="0.08" fromto="0.02 0 0 0.02 0 -0.25" /><body name="left_ankle_pitch_link" pos="0 -9.4445e-05 -0.30001"><inertial pos="-0.007269 0 0.011137" quat="0.603053 0.369225 0.369225 0.603053" mass="0.074" diaginertia="1.89e-05 1.40805e-05 6.9195e-06" /><joint name="left_ankle_pitch_joint" class="ankle_pitch" /><body name="left_ankle_roll_link" pos="0 0 -0.017558"><site name="left_foot" rgba="1 0 0 1" /><inertial pos="0.026505 0 -0.016425" quat="-0.000481092 0.728482 -0.000618967 0.685065" mass="0.608" diaginertia="0.00167218 0.0016161 0.000217621" /><joint name="left_ankle_roll_joint" class="ankle_roll" /><geom name="left_foot" class="foot" pos="0.04 0 -0.029" size="0.09 0.03 0.008" type="box" conaffinity="2" friction="0.8 0.02 0.01" /></body></body></body></body></body></body><body name="right_hip_pitch_link" pos="0 -0.064452 -0.1027"><inertial pos="0.002741 -0.047791 -0.02606" quat="0.954862 -0.293964 0.0302556 -0.030122" mass="1.35" diaginertia="0.00181517 0.00153422 0.00116212" /><joint name="right_hip_pitch_joint" class="hip_pitch" /><body name="right_hip_roll_link" pos="0 -0.052 -0.030465" quat="0.996179 0 -0.0873386 0"><inertial pos="0.029812 0.001045 -0.087934" quat="0.977808 1.97119e-05 0.205576 0.0403793" mass="1.52" diaginertia="0.00254986 0.00241169 0.00148755" /><joint name="right_hip_roll_joint" class="hip_roll" /><geom name="right_thigh" class="collision" type="capsule" size="0.06" fromto="0.02 0 0 0.02 0 -0.2" /><body name="right_hip_yaw_link" pos="0.025001 0 -0.12412"><inertial pos="-0.057709 0.010981 -0.15078" quat="0.751181 0.223482 0.15832 0.600598" mass="1.702" diaginertia="0.00776166 0.00717575 0.00160139" /><joint name="right_hip_yaw_joint" class="hip_yaw" /><body name="right_knee_link" pos="-0.078273 -0.0021489 -0.17734" quat="0.996179 0 0.0873386 0"><inertial pos="0.005457 -0.003964 -0.12074" quat="0.923439 0.0345276 0.0116333 -0.382012" mass="1.932" diaginertia="0.011374 0.0112843 0.00146452" /><joint name="right_knee_joint" class="knee" /><geom name="right_shin" class="collision" type="capsule" size="0.08" fromto="0.02 0 0 0.02 0 -0.25" /><body name="right_ankle_pitch_link" pos="0 9.4445e-05 -0.30001"><inertial pos="-0.007269 0 0.011137" quat="0.603053 0.369225 0.369225 0.603053" mass="0.074" diaginertia="1.89e-05 1.40805e-05 6.9195e-06" /><joint name="right_ankle_pitch_joint" class="ankle_pitch" /><body name="right_ankle_roll_link" pos="0 0 -0.017558"><site name="right_foot" pos="0 0 0" /><inertial pos="0.026505 0 -0.016425" quat="0.000481092 0.728482 0.000618967 0.685065" mass="0.608" diaginertia="0.00167218 0.0016161 0.000217621" /><joint name="right_ankle_roll_joint" class="ankle_roll" /><geom name="right_foot" class="foot" pos="0.04 0 -0.029" size="0.09 0.03 0.008" type="box" conaffinity="2" friction="0.8 0.02 0.01" /></body></body></body></body></body></body><body name="waist_yaw_link"><inertial pos="0.003494 0.000233 0.018034" quat="0.289697 0.591001 -0.337795 0.672821" mass="0.214" diaginertia="0.000163531 0.000107714 0.000102205" /><joint name="waist_yaw_joint" class="waist_yaw" /><body name="waist_roll_link" pos="-0.0039635 0 0.044"><inertial pos="0 2.3e-05 0" quat="0.5 0.5 -0.5 0.5" mass="0.086" diaginertia="8.245e-06 7.079e-06 6.339e-06" /><joint name="waist_roll_joint" class="waist_roll" /><body name="torso_link"><geom name="torso_col" class="collision" type="capsule" size="0.1" fromto="0 0 0.08 0 0 0.34" /><geom name="head_col" class="collision" type="sphere" size="0.09" pos="0.01 0 0.45" /><inertial pos="0.00203158 0.000339683 0.184568" quat="0.999803 -6.03319e-05 0.0198256 0.00131986" mass="7.818" diaginertia="0.121847 0.109825 0.0273735" /><joint name="waist_pitch_joint" class="waist_pitch" /><site name="imu_in_torso" size="0.01" pos="-0.03959 -0.00224 0.14792" /><body name="left_shoulder_pitch_link" pos="0.0039563 0.10022 0.24778" quat="0.990264 0.139201 1.38722e-05 -9.86868e-05"><inertial pos="0 0.035892 -0.011628" quat="0.654152 0.0130458 -0.326267 0.68225" mass="0.718" diaginertia="0.000465864 0.000432842 0.000406394" /><joint name="left_shoulder_pitch_joint" class="shoulder_pitch" /><geom size="0.03 0.025" pos="0 0.04 -0.01" quat="0.707107 0 0.707107 0" type="cylinder" class="visual" /><body name="left_shoulder_roll_link" pos="0 0.038 -0.013831" quat="0.990268 -0.139172 0 0"><inertial pos="-0.000227 0.00727 -0.063243" quat="0.701256 -0.0196223 -0.00710317 0.712604" mass="0.643" diaginertia="0.000691311 0.000618011 0.000388977" /><joint name="left_shoulder_roll_joint" range="-1.5882 2.2515" class="shoulder_roll" /><geom size="0.03 0.015" pos="-0.004 0.006 -0.053" type="cylinder" class="visual" /><body name="left_shoulder_yaw_link" pos="0 0.00624 -0.1032"><inertial pos="0.010773 -0.002949 -0.072009" quat="0.716879 -0.0964829 -0.0679942 0.687134" mass="0.734" diaginertia="0.00106187 0.00103217 0.000400661" /><joint name="left_shoulder_yaw_joint" class="shoulder_yaw" /><body name="left_elbow_link" pos="0.015783 0 -0.080518"><inertial pos="0.064956 0.004454 -0.010062" quat="0.541765 0.636132 0.388821 0.388129" mass="0.6" diaginertia="0.000443035 0.000421612 0.000259353" /><joint name="left_elbow_joint" class="elbow" /><body name="left_wrist_roll_link" pos="0.1 0.00188791 -0.01"><inertial pos="0.0171394 0.000537591 4.8864e-07" quat="0.575338 0.411667 -0.574906 0.411094" mass="0.085445" diaginertia="5.48211e-05 4.96646e-05 3.57798e-05" /><joint name="left_wrist_roll_joint" class="wrist_roll" /><body name="left_wrist_pitch_link" pos="0.038 0 0"><inertial pos="0.0229999 -0.00111685 -0.00111658" quat="0.249998 0.661363 0.293036 0.643608" mass="0.48405" diaginertia="0.000430353 0.000429873 0.000164648" /><joint name="left_wrist_pitch_joint" class="wrist_pitch" /><body name="left_wrist_yaw_link" pos="0.046 0 0"><inertial pos="0.0708244 0.000191745 0.00161742" quat="0.510571 0.526295 0.468078 0.493188" mass="0.254576" diaginertia="0.000646113 0.000559993 0.000147566" /><joint name="left_wrist_yaw_joint" class="wrist_yaw" /><site name="left_palm" pos="0.08 0 0" size="0.01" /><geom name="left_hand_collision" class="collision" type="capsule" size="0.05" fromto="0.05 0 0 0.1 0 0" rgba=".2 .6 .2 .2" /></body></body></body></body></body></body></body><body name="right_shoulder_pitch_link" pos="0.0039563 -0.10021 0.24778" quat="0.990264 -0.139201 1.38722e-05 9.86868e-05"><inertial pos="0 -0.035892 -0.011628" quat="0.68225 -0.326267 0.0130458 0.654152" mass="0.718" diaginertia="0.000465864 0.000432842 0.000406394" /><joint name="right_shoulder_pitch_joint" class="shoulder_pitch" /><geom size="0.03 0.025" pos="0 -0.04 -0.01" quat="0.707107 0 0.707107 0" type="cylinder" class="visual" /><body name="right_shoulder_roll_link" pos="0 -0.038 -0.013831" quat="0.990268 0.139172 0 0"><inertial pos="-0.000227 -0.00727 -0.063243" quat="0.712604 -0.00710317 -0.0196223 0.701256" mass="0.643" diaginertia="0.000691311 0.000618011 0.000388977" /><joint name="right_shoulder_roll_joint" range="-2.2515 1.5882" class="shoulder_roll" /><geom size="0.03 0.015" pos="-0.004 -0.006 -0.053" type="cylinder" class="visual" /><body name="right_shoulder_yaw_link" pos="0 -0.00624 -0.1032"><inertial pos="0.010773 0.002949 -0.072009" quat="0.687134 -0.0679942 -0.0964829 0.716879" mass="0.734" diaginertia="0.00106187 0.00103217 0.000400661" /><joint name="right_shoulder_yaw_joint" class="shoulder_yaw" /><body name="right_elbow_link" pos="0.015783 0 -0.080518"><inertial pos="0.064956 -0.004454 -0.010062" quat="0.388129 0.388821 0.636132 0.541765" mass="0.6" diaginertia="0.000443035 0.000421612 0.000259353" /><joint name="right_elbow_joint" class="elbow" /><body name="right_wrist_roll_link" pos="0.1 -0.00188791 -0.01"><inertial pos="0.0171394 -0.000537591 4.8864e-07" quat="0.411667 0.575338 -0.411094 0.574906" mass="0.085445" diaginertia="5.48211e-05 4.96646e-05 3.57798e-05" /><joint name="right_wrist_roll_joint" class="wrist_roll" /><body name="right_wrist_pitch_link" pos="0.038 0 0"><inertial pos="0.0229999 0.00111685 -0.00111658" quat="0.643608 0.293036 0.661363 0.249998" mass="0.48405" diaginertia="0.000430353 0.000429873 0.000164648" /><joint name="right_wrist_pitch_joint" class="wrist_pitch" /><body name="right_wrist_yaw_link" pos="0.046 0 0"><inertial pos="0.0708244 -0.000191745 0.00161742" quat="0.493188 0.468078 0.526295 0.510571" mass="0.254576" diaginertia="0.000646113 0.000559993 0.000147566" /><joint name="right_wrist_yaw_joint" class="wrist_yaw" /><site name="right_palm" pos="0.08 0 0" size="0.01" /><geom name="right_hand_collision" class="collision" type="capsule" size="0.05" fromto="0.05 0 0 0.1 0 0" rgba=".2 .6 .2 .2" /></body></body></body></body></body></body></body></body></body></body></body></worldbody><contact><pair name="left_foot_floor" geom1="left_foot" geom2="floor" condim="3" friction="0.6 0.6" /><pair name="right_foot_floor" geom1="right_foot" geom2="floor" condim="3" friction="0.6 0.6" /><pair name="left_foot_terrain" geom1="left_foot" geom2="terrain" condim="3" friction="0.8 0.8" /><pair name="right_foot_terrain" geom1="right_foot" geom2="terrain" condim="3" friction="0.8 0.8" /><pair name="left_hand_thigh" geom1="left_hand_collision" geom2="left_thigh" condim="1" /><pair name="right_hand_thigh" geom1="right_hand_collision" geom2="right_thigh" condim="1" /><pair name="left_foot_right_foot" geom1="left_foot" geom2="right_foot" condim="1" /><pair geom1="floor" geom2="left_shin" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="right_shin" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="left_thigh" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="right_thigh" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="left_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="right_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="pelvis_col" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="torso_col" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="head_col" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="left_shin" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="right_shin" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="left_thigh" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="right_thigh" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="left_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="right_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="pelvis_col" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="torso_col" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="head_col" condim="3" friction="0.6 0.6" /></contact><actuator><position class="hip_pitch" name="left_hip_pitch_joint" joint="left_hip_pitch_joint" /><position class="hip_roll" name="left_hip_roll_joint" joint="left_hip_roll_joint" /><position class="hip_yaw" name="left_hip_yaw_joint" joint="left_hip_yaw_joint" /><position class="knee" name="left_knee_joint" joint="left_knee_joint" /><position class="ankle_pitch" name="left_ankle_pitch_joint" joint="left_ankle_pitch_joint" /><position class="ankle_roll" name="left_ankle_roll_joint" joint="left_ankle_roll_joint" /><position class="hip_pitch" name="right_hip_pitch_joint" joint="right_hip_pitch_joint" /><position class="hip_roll" name="right_hip_roll_joint" joint="right_hip_roll_joint" /><position class="hip_yaw" name="right_hip_yaw_joint" joint="right_hip_yaw_joint" /><position class="knee" name="right_knee_joint" joint="right_knee_joint" /><position class="ankle_pitch" name="right_ankle_pitch_joint" joint="right_ankle_pitch_joint" /><position class="ankle_roll" name="right_ankle_roll_joint" joint="right_ankle_roll_joint" /><position class="waist_yaw" name="waist_yaw_joint" joint="waist_yaw_joint" /><position class="waist_roll" name="waist_roll_joint" joint="waist_roll_joint" /><position class="waist_pitch" name="waist_pitch_joint" joint="waist_pitch_joint" /><position class="shoulder_pitch" name="left_shoulder_pitch_joint" joint="left_shoulder_pitch_joint" /><position class="shoulder_roll" name="left_shoulder_roll_joint" joint="left_shoulder_roll_joint" /><position class="shoulder_yaw" name="left_shoulder_yaw_joint" joint="left_shoulder_yaw_joint" /><position class="elbow" name="left_elbow_joint" joint="left_elbow_joint" /><position class="wrist_roll" name="left_wrist_roll_joint" joint="left_wrist_roll_joint" /><position class="wrist_pitch" name="left_wrist_pitch_joint" joint="left_wrist_pitch_joint" /><position class="wrist_yaw" name="left_wrist_yaw_joint" joint="left_wrist_yaw_joint" /><position class="shoulder_pitch" name="right_shoulder_pitch_joint" joint="right_shoulder_pitch_joint" /><position class="shoulder_roll" name="right_shoulder_roll_joint" joint="right_shoulder_roll_joint" /><position class="shoulder_yaw" name="right_shoulder_yaw_joint" joint="right_shoulder_yaw_joint" /><position class="elbow" name="right_elbow_joint" joint="right_elbow_joint" /><position class="wrist_roll" name="right_wrist_roll_joint" joint="right_wrist_roll_joint" /><position class="wrist_pitch" name="right_wrist_pitch_joint" joint="right_wrist_pitch_joint" /><position class="wrist_yaw" name="right_wrist_yaw_joint" joint="right_wrist_yaw_joint" /></actuator><sensor><framezaxis objtype="site" objname="imu_in_torso" name="upvector_torso" /><velocimeter site="imu_in_torso" name="local_linvel_torso" /><velocimeter site="imu_in_pelvis" name="local_linvel_pelvis" /><accelerometer site="imu_in_torso" name="accelerometer_torso" /><accelerometer site="imu_in_pelvis" name="accelerometer_pelvis" /><gyro site="imu_in_torso" name="gyro_torso" /><gyro site="imu_in_pelvis" name="gyro_pelvis" /><framezaxis objtype="site" objname="imu_in_pelvis" name="upvector_pelvis" /><framexaxis objtype="site" objname="imu_in_torso" name="forwardvector_torso" /><framexaxis objtype="site" objname="imu_in_pelvis" name="forwardvector_pelvis" /><framequat objtype="site" objname="imu_in_torso" name="orientation_torso" /><framequat objtype="site" objname="imu_in_pelvis" name="orientation_pelvis" /><framelinvel objtype="site" objname="imu_in_torso" name="global_linvel_torso" /><framelinvel objtype="site" objname="imu_in_pelvis" name="global_linvel_pelvis" /><frameangvel objtype="site" objname="imu_in_torso" name="global_angvel_torso" /><frameangvel objtype="site" objname="imu_in_pelvis" name="global_angvel_pelvis" /><framelinvel objtype="site" objname="left_foot" name="left_foot_global_linvel" /><framelinvel objtype="site" objname="right_foot" name="right_foot_global_linvel" /><framezaxis objtype="site" objname="left_foot" name="left_foot_upvector" /><framezaxis objtype="site" objname="right_foot" name="right_foot_upvector" /><force name="left_foot_force" site="left_foot" /><force name="right_foot_force" site="right_foot" /><contact name="left_foot_floor_found" geom1="left_foot" geom2="floor" reduce="mindist" num="1" data="found" /><contact name="right_foot_floor_found" geom1="right_foot" geom2="floor" reduce="mindist" num="1" data="found" /><contact name="left_foot_terrain_found" geom1="left_foot" geom2="terrain" reduce="mindist" num="1" data="found" /><contact name="right_foot_terrain_found" geom1="right_foot" geom2="terrain" reduce="mindist" num="1" data="found" /><contact name="right_foot_left_foot_found" geom1="right_foot" geom2="left_foot" reduce="mindist" num="1" data="found" /><contact name="left_foot_right_shin_found" geom1="left_foot" geom2="right_shin" reduce="mindist" num="1" data="found" /><contact name="right_foot_left_shin_found" geom1="right_foot" geom2="left_shin" reduce="mindist" num="1" data="found" /><contact name="left_hand_left_thigh_found" geom1="left_hand_collision" geom2="left_thigh" reduce="mindist" num="1" data="found" /><contact name="right_hand_right_thigh_found" geom1="right_hand_collision" geom2="right_thigh" reduce="mindist" num="1" data="found" /><framepos objtype="site" objname="left_foot" name="left_foot_pos" /><framepos objtype="site" objname="right_foot" name="right_foot_pos" /><framezaxis objtype="body" objname="torso_link" name="torso_zaxis" /></sensor><keyframe><key name="home" qpos=" 0 0 0.785 1 0 0 0 -0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0 " ctrl=" -0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0 " /><key name="knees_bent" qpos=" 0 0 0.755 1 0 0 0 -0.312 0 0 0.669 -0.363 0 -0.312 0 0 0.669 -0.363 0 0 0 0.073 0.2 0.2 0 0.6 0 0 0 0.2 -0.2 0 0.6 0 0 0 " ctrl=" -0.312 0 0 0.669 -0.363 0 -0.312 0 0 0.669 -0.363 0 0 0 0.073 0.2 0.2 0 0.6 0 0 0 0.2 -0.2 0 0.6 0 0 0 " /></keyframe></mujoco>'''

# the Unitree H1 (MuJoCo Menagerie, Unitree Robotics, BSD-3) and the Apptronik Apollo (MuJoCo Menagerie, Apptronik, Apache 2.0),
# in the conventions above (meshes left out), compressed (zlib, base64)
H1_XML_Z = '''
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
qTXqk/nnv37+D5SaPy8=
'''
APOLLO_XML_Z = '''
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
'''


# ── the robots: the Unitree G1 (the default), the Unitree H1 and the Apptronik Apollo ─────────────────────────────────
# Each robot's MuJoCo model follows the same conventions (made from MuJoCo Menagerie by the same converter): its root
# body is "pelvis", its torso "torso_link"; the IMU sites imu_in_pelvis and imu_in_torso; the feet are boxes named
# left_foot / right_foot (with sites of the same name at the ankles); the shapes that may touch the ground are named
# left_shin, left_thigh, left_hand_collision, pelvis_col, torso_col, head_col (and right_…); the sensors, the contact
# pairs and the keyframes "home" and "knees_bent" are those of the G1. The joints keep the G1's names where they match
# (left_hip_pitch_joint, left_knee_joint…). So the task, the rewards and the view work for each; what differs is said
# here: the standing pelvis height h0 (the falls, the standing test, the elevation map scale with it), the head camera,
# the 3D model, the credit.
def _xml(z):
    import base64
    import zlib
    return zlib.decompress(base64.b64decode(z)).decode()


ROBOTS = {
    "g1": {"name": "Unitree G1", "short": "G1", "xml": lambda: MODEL_XML, "h0": 0.755, "view": "g1",
           "camera": {"body": "torso_link", "pos": [0.0576, 0.0175, 0.4299], "pitch": 0.8308, "fov_h": 69.0, "aspect": 1.7778,
                      "name": "Intel RealSense D435i in the head"},
           "credit": "Unitree G1: MuJoCo Menagerie (BSD-3) and MuJoCo Playground (Apache 2.0); 3D model KenzoDkohno22T, CC BY 4.0"},
    "h1": {"name": "Unitree H1", "short": "H1", "xml": lambda: _xml(H1_XML_Z), "h0": 0.982, "view": "robot-h1",
           "camera": {"body": "torso_link", "pos": [0.095, 0.0, 0.62], "pitch": 0.8308, "fov_h": 69.0, "aspect": 1.7778,
                      "name": "Intel RealSense D435i in the head"},
           "credit": "Unitree H1: MuJoCo Menagerie, Unitree Robotics (BSD-3)"},
    "apollo": {"name": "Apptronik Apollo", "short": "Apollo", "xml": lambda: _xml(APOLLO_XML_Z), "h0": 1.023, "view": "robot-apollo",
               "camera": {"body": "neck_pitch_link", "pos": [0.15, 0.0, 0.0], "pitch": 0.70, "fov_h": 69.0, "aspect": 1.7778,
                          "name": "a depth camera in the head"},
               "credit": "Apptronik Apollo: MuJoCo Menagerie, Apptronik (Apache 2.0)"},
}
ROBOT_OF = {"unitree g1": "g1", "g1": "g1", "unitree h1": "h1", "h1": "h1", "apptronik apollo": "apollo", "apollo": "apollo"}
ROBOT = "g1"            # the robot in use (use_robot sets it, and the sizes that follow from it)
NU = 29                 # its joints (motors)
H0 = 0.755              # its pelvis' height standing (knees bent)


def robot_id(p):
    """the robot a block's settings ask for (the G1 when none)"""
    return ROBOT_OF.get(str(p.get("robot") or "Unitree G1").strip().lower(), "g1")


def use_robot(rid, mujoco):
    """sets the robot: its model's size decides what the policy reads and does"""
    global ROBOT, NU, H0, OBS_BASE, PRIV_BASE, OBS_DIM, PRIV_DIM, MODEL_IN_USE, FIELDS
    ROBOT, R = rid, ROBOTS[rid]
    MODEL_IN_USE = R["xml"]()
    m = mujoco.MjModel.from_xml_string(MODEL_IN_USE)
    NU, H0 = int(m.nu), float(R["h0"])
    OBS_BASE = 3 + 3 + 3 + 3 + NU + NU + NU + 4
    PRIV_BASE = OBS_BASE + 1 + 2 + 2 + 6 + 3 + 3
    OBS_DIM, PRIV_DIM = OBS_BASE + N_SCAN, PRIV_BASE + N_SCAN
    FIELDS = make_fields(NU)
    return R


def hs(v):
    """a height set for the G1 (pelvis 0.755 m standing), for the robot in use"""
    return v * H0 / 0.755


MODEL_IN_USE = None


# ── the physics: MuJoCo, many robots at once ──────────────────────────────────────────────────────────────────────
# Two ways to run the same model: MuJoCo Warp on the NVIDIA GPU (thousands of robots, the whole step on the card, one
# CUDA graph per control step) or MuJoCo on the CPU (one simulation per robot, spread over the cores). Both give the
# same views as PyTorch tensors: qpos, qvel, ctrl and the sensors, on the device the learning runs on.
CTRL_DT, SIM_DT = 0.02, 0.002           # the policy at 50 Hz, the physics at 500 Hz
N_SUB = int(round(CTRL_DT / SIM_DT))


# ── the ground: a rectangular circuit (x along its first side from the start line, y to the left, metres) ──────────
# The robot follows the circuit's line round a 38 x 18 m rectangle with rounded corners (2.5 m), counterclockwise:
#   side 1 (y = 0, x from -4 to 34): a 4 x 4 m square of bumps (0-4 cm) at 4-8 m · a 12.5 % slope (7°) up to a 0.5 m
#     plateau at 14-17 m and down to 21 m · a 4 x 4 m square of blocks (0.4 m tiles, 0-5 cm) at 24-28 m
#   side 3 (y = 18, walked toward -x): a staircase 3 m wide, 10 steps up (12 cm, 30 cm deep) from x = 22, a 2 m
#     landing 1.2 m up, 10 steps down
#   sides 2 and 4: flat
# Inside it, a training yard: four staircases of 10 steps up and 10 down, with 6, 9, 12 and 15 cm risers.
# The same few numbers go to the 3D view, which builds the same ground from them. In MuJoCo: one heightfield, 5 cm a
# point (the stairs' edges on its points).
HF_X0, HF_X1, HF_Y0, HF_Y1, HF_RES, HF_BASE = -8.0, 38.0, -7.0, 22.0, 0.05, 0.05
CIRCUIT = {"x0": -4.0, "x1": 34.0, "y0": 0.0, "y1": 18.0, "r": 2.5}


def make_path(c=CIRCUIT, ds=0.1):
    """the circuit's line: points every ds metres, counterclockwise from the start line (0, 0); heading and curvature"""
    import numpy as np
    x0, x1, y0, y1, r = c["x0"], c["x1"], c["y0"], c["y1"], c["r"]
    pts = []
    def line(a, b):
        for t in np.linspace(0, 1, 400, endpoint=False):
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    def arc(cx, cy, a0):
        for t in np.linspace(0, 1, 200, endpoint=False):
            a = a0 + t * np.pi / 2
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    line((x0 + r, y0), (x1 - r, y0)); arc(x1 - r, y0 + r, -np.pi / 2)
    line((x1, y0 + r), (x1, y1 - r)); arc(x1 - r, y1 - r, 0.0)
    line((x1 - r, y1), (x0 + r, y1)); arc(x0 + r, y1 - r, np.pi / 2)
    line((x0, y1 - r), (x0, y0 + r)); arc(x0 + r, y0 + r, np.pi)
    P = np.array(pts)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    S = np.concatenate([[0.0], np.cumsum(seg)])
    L = S[-1]
    s0 = np.interp(-x0 - r, [0, x1 - x0 - 2 * r], [0, x1 - x0 - 2 * r])    # the start line: x = 0 on side 1
    s = (s0 + np.arange(0, L, ds)) % L
    Pc = np.vstack([P, P[:1]])
    Q = np.stack([np.interp(s, S, Pc[:, 0]), np.interp(s, S, Pc[:, 1])], -1)
    d = np.roll(Q, -1, axis=0) - np.roll(Q, 1, axis=0)
    head = np.arctan2(d[:, 1], d[:, 0])
    dh = np.angle(np.exp(1j * (np.roll(head, -1) - np.roll(head, 1))))
    curv = dh / (2 * ds)
    return Q, head, curv, float(L)


def make_course():
    import numpy as np
    r = np.random.default_rng(20261112)
    coarse = r.uniform(-1, 1, (11, 11))                       # bumps: a 0.4 m swell and a 0.1 m grain
    xi = np.linspace(0, 10, 41)
    up = np.array([[np.interp(x, np.arange(11), row) for x in xi] for row in coarse])
    up = np.array([np.interp(xi, np.arange(11), up[:, j]) for j in range(41)]).T
    bumps = 16 * up + r.uniform(-6, 6, (41, 41))
    bumps = np.round(bumps - bumps.min()).astype(int)            # (all above the floor: 0 to about 4 cm)
    blocks = r.integers(0, 51, (10, 10))
    stairs = [{"x0": 22.0, "y": 18.0, "dir": -1, "n": 10, "rise": 0.12, "tread": 0.3, "landing": 2.0, "width": 3.0, "name": "circuit"}]
    for k, rise in enumerate((0.06, 0.09, 0.12, 0.15)):          # the training yard, inside the circuit
        stairs.append({"x0": 2.0, "y": round(6.6 + 2.2 * k, 2), "dir": 1, "n": 10, "rise": rise, "tread": 0.3, "landing": 1.2, "width": 1.8, "name": "yard"})
    Q, head, curv, L = make_path()
    k4 = max(1, int(round(0.25 / 0.1)))
    tiles = {"x0": 32.0, "y0": 7.0, "cell": 0.5, "n": 8, "half": 0.15, "amp": [0.01, 0.04], "base": [0.0, 0.03], "period": [1.5, 4.0]}
    return {"ramp": {"x": [10.0, 14.0, 17.0, 21.0], "h": 0.5, "half": 3.0, "bank": 1.5},
            "patches": [{"kind": "bumps", "x0": 4.0, "y0": -2.0, "size": 4.0, "cell": 0.1, "n": 41, "mm": bumps.ravel().tolist()},
                        {"kind": "blocks", "x0": 24.0, "y0": -2.0, "size": 4.0, "cell": 0.4, "n": 10, "mm": blocks.ravel().tolist()}],
            "stairs": stairs, "tiles": tiles, "circuit": dict(CIRCUIT, length=round(L, 2)),
            "path": [[round(float(x), 3), round(float(y), 3)] for x, y in Q[::k4]]}


COURSE = make_course()
PATH = make_path()              # (points every 10 cm, heading, curvature, length)


def stair_profile(u, st):
    """a staircase's height at distance u (m) from its first riser, along the way it climbs"""
    import numpy as np
    n, r, t, L = st["n"], st["rise"], st["tread"], st["landing"]
    up = (n - 1) * t
    return np.where(u < 0, 0.0, np.where(u < up, (np.floor(u / t + 1e-6) + 1) * r, np.where(u < up + L, n * r,
                    np.where(u < 2 * up + L, (n - 1 - np.floor((u - up - L) / t + 1e-6)) * r, 0.0))))


def course_height(X, Y, course=COURSE):
    """the ground's height (m) at X, Y (numpy arrays)"""
    import numpy as np
    X, Y = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))
    H = np.zeros(X.shape)
    rp = course["ramp"]
    xs = rp["x"]
    prof = np.interp(X, xs, [0.0, rp["h"], rp["h"], 0.0], left=0.0, right=0.0)
    H += prof * np.clip((rp["half"] + rp["bank"] - np.abs(Y)) / rp["bank"], 0.0, 1.0)
    for pa in course["patches"]:
        n, c = pa["n"], pa["cell"]
        g = np.array(pa["mm"], dtype=float).reshape(n, n) / 1000.0        # [row along y, column along x]
        u, v = (X - pa["x0"]) / c, (Y - pa["y0"]) / c
        if pa["kind"] == "bumps":
            inside = (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1)
            u0 = np.clip(np.floor(u), 0, n - 2).astype(int); v0 = np.clip(np.floor(v), 0, n - 2).astype(int)
            fu, fv = np.clip(u - u0, 0, 1), np.clip(v - v0, 0, 1)
            h = (g[v0, u0] * (1 - fu) * (1 - fv) + g[v0, u0 + 1] * fu * (1 - fv) + g[v0 + 1, u0] * (1 - fu) * fv + g[v0 + 1, u0 + 1] * fu * fv)
        else:
            inside = (u >= 0) & (u < n) & (v >= 0) & (v < n)
            h = g[np.clip(np.floor(v), 0, n - 1).astype(int), np.clip(np.floor(u), 0, n - 1).astype(int)]
        H = H + np.where(inside, h, 0.0)
    for st in course.get("stairs", []):
        inside = np.abs(Y - st["y"]) <= st["width"] / 2 + 1e-6
        H = H + np.where(inside, stair_profile((X - st["x0"]) * st["dir"], st), 0.0)
    return H


def ground_grid(terrain):
    """the heightfield's heights (rows along y, columns along x)"""
    import numpy as np
    xs = np.linspace(HF_X0, HF_X1, int(round((HF_X1 - HF_X0) / HF_RES)) + 1)
    ys = np.linspace(HF_Y0, HF_Y1, int(round((HF_Y1 - HF_Y0) / HF_RES)) + 1)
    if terrain != "circuit":
        return np.zeros((len(ys), len(xs)))
    X, Y = np.meshgrid(xs, ys)
    return course_height(X, Y)


def tiles_xml(t=None):
    """the moving tiles: one box each, moved up and down by the task (mocap bodies; only the feet touch them)"""
    t = t or COURSE["tiles"]
    out = []
    for j in range(t["n"]):
        for i in range(t["n"]):
            x, y = t["x0"] + (i + 0.5) * t["cell"], t["y0"] + (j + 0.5) * t["cell"]
            out.append('<body name="tile_%d_%d" mocap="true" pos="%.3f %.3f %.3f"><geom type="box" size="%.3f %.3f %.3f" contype="2" conaffinity="0" '
                       'condim="3" friction="0.8 0.02 0.01" group="1" /></body>' % (i, j, x, y, -t["half"], t["cell"] / 2 - 0.005, t["cell"] / 2 - 0.005, t["half"]))
    return "".join(out)


def load_model(mujoco, terrain="circuit"):
    import numpy as np
    m = mujoco.MjModel.from_xml_string((MODEL_IN_USE or MODEL_XML).replace("</worldbody>", tiles_xml() + "</worldbody>"))
    m.opt.timestep = SIM_DT
    H = ground_grid(terrain)
    hid = m.hfield("terrain").id
    assert H.shape == (int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid]))
    top = float(m.hfield_size[hid][2])
    a = int(m.hfield_adr[hid])
    m.hfield_data[a:a + H.size] = np.clip((H + HF_BASE) / top, 0.0, 1.0).ravel().astype(np.float32)
    if terrain != "circuit":            # the flat floor only: the course's heightfield out of the way (fewer contacts)
        m.geom_pos[m.geom("terrain").id][2] = -5.0
    return m


class WarpSim:
    """MuJoCo Warp: n worlds on the GPU"""

    def __init__(self, mjm, n, device="cuda:0", log=None):
        import warp as wp
        import mujoco_warp as mjw
        import torch
        wp.config.quiet = True
        wp.init()
        self.wp, self.mjw, self.dev, self.torch = wp, mjw, device, torch
        self.n = n
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(mjm)
            try:
                self.m.opt.warn_overflow = 0
            except Exception:
                pass
            self.d = mjw.make_data(mjm, nworld=n, nconmax=96, njmax=480)    # (room for the contacts on rough ground)
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.sens = wp.to_torch(self.d.sensordata)
        self.tq = wp.to_torch(self.d.actuator_force)         # each motor's torque (N m)
        self.mocap = wp.to_torch(self.d.mocap_pos) if mjm.nmocap else None      # the moving tiles' places (nworld, ntiles, 3)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)                          # forces applied to bodies (nworld, nbody, 6: force, torque)
        self.tdev = self.qpos.device
        self._qacc = wp.to_torch(self.d.qacc_warmstart) if hasattr(self.d, "qacc_warmstart") else None
        self.graph = None
        self.cuda = str(device).startswith("cuda")
        t = __import__("time").time()
        with wp.ScopedDevice(device):
            self.mjw.forward(self.m, self.d)
            for _ in range(2):
                self.mjw.step(self.m, self.d)          # (the kernels compile here, the first time on a machine)
            wp.synchronize()
            if self.cuda:
                with wp.ScopedCapture() as cap:
                    for _ in range(N_SUB):
                        self.mjw.step(self.m, self.d)
                self.graph = cap.graph
        if log:
            log("MuJoCo Warp ready on %s: %d robots (kernels built in %.0f s)." % (device, n, __import__("time").time() - t))
        torch.cuda.synchronize() if self.cuda else None

    def step(self):
        wp = self.wp
        if self.cuda:
            self.torch.cuda.synchronize()        # (what PyTorch wrote into the arrays first, then the physics)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            with wp.ScopedDevice(self.dev):
                for _ in range(N_SUB):
                    self.mjw.step(self.m, self.d)
        if self.cuda:
            wp.synchronize_device(self.dev)

    def kick(self, dv):
        """add dv (n, 2) to the robots' horizontal speed: a push"""
        self.qvel[:, :2] += dv.to(self.qvel.dtype)

    def set_tiles(self, z):
        """the moving tiles' centres' heights (n, ntiles), for the next step"""
        if self.mocap is not None:
            self.mocap[:, :, 2] = z.to(self.mocap.dtype)

    def set_lift(self, body, fz):
        """an upward force fz (n, newtons) on one body of each robot, until changed (a help while it learns to stand up)"""
        self.xfrc[:, body, 2] = fz.to(self.xfrc.dtype)

    def set_rows(self, idx, qpos, qvel, ctrl):
        """robots idx start again from qpos, qvel (their motors aimed at ctrl)"""
        self.qpos[idx] = qpos.to(self.qpos.dtype)
        self.qvel[idx] = qvel.to(self.qvel.dtype)
        self.ctrl[idx] = ctrl.to(self.ctrl.dtype)
        if self._qacc is not None:
            self._qacc[idx] = 0.0

    def refresh(self):
        """the sensors after set_rows (they are computed by the step; a forward pass brings them up to date)"""
        if self.cuda:
            self.torch.cuda.synchronize()
        with self.wp.ScopedDevice(self.dev):
            self.mjw.forward(self.m, self.d)
        if self.cuda:
            self.wp.synchronize_device(self.dev)

    def sync(self):
        if self.cuda:
            self.wp.synchronize()


class CpuSim:
    """MuJoCo on the CPU: one simulation per robot, the robots shared out over threads (MuJoCo releases the GIL)"""

    def __init__(self, mjm, n, threads=None, log=None):
        import os
        import mujoco
        import numpy as np
        import torch
        from concurrent.futures import ThreadPoolExecutor
        self.mj, self.np, self.torch = mujoco, np, torch
        self.m = mjm
        self.n = n
        self.datas = [mujoco.MjData(mjm) for _ in range(n)]
        nt = max(1, min(n, threads or (os.cpu_count() or 2)))
        self.ex = ThreadPoolExecutor(nt)
        self.parts = [list(range(i * n // nt, (i + 1) * n // nt)) for i in range(nt)]
        self.qpos_np = np.zeros((n, mjm.nq))
        self.qvel_np = np.zeros((n, mjm.nv))
        self.ctrl_np = np.zeros((n, mjm.nu))
        self.sens_np = np.zeros((n, mjm.nsensordata))
        self.tq_np = np.zeros((n, mjm.nu))
        self.tq = torch.zeros(n, mjm.nu)
        self.tile_z = None
        self.lift = None             # (body, forces) an upward help on one body
        self.qpos = torch.zeros(n, mjm.nq)
        self.qvel = torch.zeros(n, mjm.nv)
        self.ctrl = torch.zeros(n, mjm.nu)
        self.sens = torch.zeros(n, mjm.nsensordata)
        self.tdev = torch.device("cpu")
        if log:
            log("MuJoCo on the CPU: %d robots over %d threads." % (n, nt))

    def _work(self, part):
        mj, m, D = self.mj, self.m, self.datas
        tz, lift = self.tile_z, self.lift
        for i in part:
            d = D[i]
            d.ctrl[:] = self.ctrl_np[i]
            if tz is not None:
                d.mocap_pos[:, 2] = tz[i]
            if lift is not None:
                d.xfrc_applied[lift[0], 2] = lift[1][i]
            for _ in range(N_SUB):
                mj.mj_step(m, d)
            self.qpos_np[i] = d.qpos
            self.qvel_np[i] = d.qvel
            self.sens_np[i] = d.sensordata
            self.tq_np[i] = d.actuator_force

    def step(self):
        self.ctrl_np[:] = self.ctrl.numpy()
        list(self.ex.map(self._work, self.parts))
        self._pull()

    def _pull(self):
        self.qpos.copy_(self.torch.from_numpy(self.qpos_np))
        self.qvel.copy_(self.torch.from_numpy(self.qvel_np))
        self.sens.copy_(self.torch.from_numpy(self.sens_np))
        self.tq.copy_(self.torch.from_numpy(self.tq_np))

    def set_rows(self, idx, qpos, qvel, ctrl):
        qp, qv, cu = qpos.double().numpy(), qvel.double().numpy(), ctrl.double().numpy()
        for k, i in enumerate(idx.tolist()):
            d = self.datas[i]
            self.mj.mj_resetData(self.m, d)
            d.qpos[:] = qp[k]
            d.qvel[:] = qv[k]
            d.ctrl[:] = cu[k]
            self.mj.mj_forward(self.m, d)
            self.qpos_np[i] = d.qpos
            self.qvel_np[i] = d.qvel
            self.sens_np[i] = d.sensordata
            self.ctrl_np[i] = cu[k]
        self.ctrl[idx] = ctrl.float()
        self._pull()

    def refresh(self):
        pass

    def kick(self, dv):
        d_ = dv.double().numpy()
        for i in (abs(d_).sum(-1) > 0).nonzero()[0].tolist():
            self.datas[i].qvel[0:2] += d_[i]
            self.qvel_np[i] = self.datas[i].qvel
        self._pull()

    def set_tiles(self, z):
        self.tile_z = z.double().numpy() if self.m.nmocap else None

    def set_lift(self, body, fz):
        self.lift = (int(body), fz.double().numpy())

    def sync(self):
        pass


def make_sim(mjm, n, want, log=None):
    """(sim, label): the GPU through MuJoCo Warp when it can, else the CPU"""
    import torch
    w = str(want or "auto").lower()
    if w.startswith("nvidia") or w == "auto":
        err = None
        if torch.cuda.is_available():
            try:
                sim = WarpSim(mjm, n, "cuda:0", log)
                return sim, "%s (MuJoCo Warp)" % torch.cuda.get_device_name(0)
            except ImportError as e:
                err = "MuJoCo Warp is not installed on this worker (pip install mujoco-warp): %s" % e
            except Exception as e:      # noqa: BLE001
                err = "MuJoCo Warp could not start on the GPU: %s" % e
        else:
            err = "this PyTorch cannot use an NVIDIA GPU (install the CUDA build: pip install torch --index-url https://download.pytorch.org/whl/cu124)"
        if w.startswith("nvidia"):
            raise RuntimeError("NVIDIA GPU asked, but " + err)
        if log:
            log("On the CPU instead: " + err)
    n = min(n, 512)
    return CpuSim(mjm, n, log=log), "CPU (MuJoCo, %d robots)" % n


# ── the task: walk, following a command (straight ahead along a line, as fast as asked) ──────────────────────────────
# What the policy sees (as the robot itself could): its speed and turning (from the IMU in the pelvis), where gravity
# points, the command, its 29 joint angles and speeds, what it did last, and a clock for the gait (the two feet half a
# step apart). What it does: 29 joint targets around a standing pose (the motors' own PD loops follow them).
# The command: forward speed, sideways speed and turn rate. To walk straight it follows a line: a guide (like a driver
# holding a joystick) turns and slides it back onto the line and the heading; in training the commands also come at
# random, so it learns to turn and to step sideways when the guide asks.
# The rewards (after MuJoCo Playground's G1 joystick task): follow the commanded speed and turn, stay upright, step with
# a gait (feet in the air in turn, lifted to a set height, no slipping), stay near the standing pose, no falling.
import math

ACTION_SCALE = 0.5
OBS_BASE = 3 + 3 + 3 + 3 + 29 + 29 + 29 + 4      # what it senses of itself (103)
PRIV_BASE = OBS_BASE + 1 + 2 + 2 + 6 + 3 + 3      # and what only the critic knows (120)
# what it sees of the ground: an elevation map from its head depth camera, 13 x 5 points from 0.3 m behind to 1.5 m
# ahead and 0.3 m either side, turned with it (the ground under its feet comes from what the camera saw a moment ago)
SCAN_X = [-0.3 + 0.15 * i for i in range(13)]
SCAN_Y = [-0.3, -0.15, 0.0, 0.15, 0.3]
N_SCAN = len(SCAN_X) * len(SCAN_Y)
OBS_DIM = OBS_BASE + N_SCAN
PRIV_DIM = PRIV_BASE + N_SCAN
EP_STEPS = 1000                      # 20 s
SENSORS = ["local_linvel_pelvis", "gyro_pelvis", "global_angvel_torso", "torso_zaxis", "global_linvel_pelvis",
           "global_angvel_pelvis", "left_foot_global_linvel", "right_foot_global_linvel", "left_foot_force", "right_foot_force",
           "left_foot_floor_found", "right_foot_floor_found", "right_foot_left_foot_found", "left_hand_left_thigh_found",
           "right_hand_right_thigh_found", "left_foot_pos", "right_foot_pos", "left_foot_terrain_found", "right_foot_terrain_found",
           "accelerometer_pelvis"]
SCALES = {"tracking_lin_vel": 1.0, "tracking_ang_vel": 0.75, "ang_vel_xy": -0.05, "alive": 0.5, "orientation": -2.0, "feet_air_time": 2.0,
          "feet_slip": -0.25, "feet_phase": 1.0, "stand_still": -1.0, "termination": -100.0, "collision": -0.1,
          "contact_force": -0.01, "joint_deviation_knee": -0.1, "joint_deviation_hip": -0.25, "dof_pos_limits": -1.0,
          "pose": -0.1, "action_rate": -0.01}


def quat_rotate_inv(q, v):
    """v (world) into the frame of the unit quaternions q (w, x, y, z)"""
    import torch
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def yaw_of(q):
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rz_bezier(phi, h):
    """the wanted foot height along the gait cycle (lift and set down smoothly)"""
    import torch
    x = (phi + math.pi) / (2 * math.pi)
    bez = lambda a, b, s: a + (b - a) * (s ** 3 + 3 * (s ** 2 * (1 - s)))   # noqa: E731
    return torch.where(x <= 0.5, bez(0.0, h, 2 * x), bez(h, 0.0, 2 * x - 1))


class WalkEnv:
    def __init__(self, sim, mjm, seed=0, vmax=1.0, push=1.5, noise=1.0, train=True, terrain="circuit", scan=True):
        import torch
        self.sim, self.m, self.n = sim, mjm, sim.n
        dev = self.dev = sim.tdev
        self.g = torch.Generator(device=dev).manual_seed(seed)
        f = dict(device=dev, dtype=torch.float32)
        key = mjm.key("knees_bent")
        self.q0 = torch.tensor(key.qpos, **f)
        self.default = self.q0[7:].clone()
        self.vmax, self.push_max, self.noise, self.train = vmax, push, noise, train
        lo, hi = torch.tensor(mjm.jnt_range[1:, 0], **f), torch.tensor(mjm.jnt_range[1:, 1], **f)
        c, r = (lo + hi) / 2, hi - lo
        self.soft_lo, self.soft_hi = c - 0.5 * r * 0.95, c + 0.5 * r * 0.95
        names = [mjm.joint(i).name for i in range(1, mjm.njnt)]
        idx = lambda keys: torch.tensor([i for i, nm in enumerate(names) if any(k in nm for k in keys)], device=dev)   # noqa: E731
        self.hip_i = idx(["hip_roll", "hip_yaw"])
        self.knee_i = idx(["knee"])
        self.sadr = {}
        for s in SENSORS:
            sid = mjm.sensor(s).id
            a, d = int(mjm.sensor_adr[sid]), int(mjm.sensor_dim[sid])
            self.sadr[s] = slice(a, a + d)
        # what differs with the robot: the torso's lean standing (the G1's as Playground has it), the feet's load limit
        if ROBOT == "g1":
            tz0 = [0.073, 0.0, 1.0]
        else:
            import mujoco
            d0 = mujoco.MjData(mjm)
            mujoco.mj_resetDataKeyframe(mjm, d0, key.id)
            mujoco.mj_forward(mjm, d0)
            tz0 = [float(x) for x in d0.sensordata[self.sadr["torso_zaxis"]]]
        self.tz0 = torch.tensor(tz0, **f)
        self.f_max = 500.0 * float(mjm.body_subtreemass[mjm.body("pelvis").id]) / 33.34
        n = self.n
        z = lambda *s: torch.zeros(n, *s, **f)   # noqa: E731
        self.cmd = z(3)
        self.last_act, self.act = z(NU), z(NU)
        self.phase, self.phase_dt = z(2), z()
        self.air = z(2)
        self.last_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.swing_peak = z(2)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.push_every = torch.zeros(n, dtype=torch.long, device=dev)
        self.mode_line = torch.ones(n, dtype=torch.bool, device=dev)       # following the line (or free joystick commands)
        self.line_y, self.line_yaw = z(), z()                              # the line: a heading and an offset, in the world
        self.speed = z()
        self.ep_ret, self.ep_dist = z(), z()
        self.t_cmd = torch.zeros(n, dtype=torch.long, device=dev)
        self.fell = torch.zeros(n, dtype=torch.bool, device=dev)
        # the ground (the same heights as the heightfield the physics uses), to measure heights from it
        self.terrain = terrain
        self.Hg = torch.tensor(ground_grid(terrain), device=dev, dtype=torch.float32)
        # the circuit's line (robots on it follow it, either way round) and the elevation map's points
        Q, head, curv, L = PATH
        self.P, self.PH, self.PK = torch.tensor(Q, **f), torch.tensor(head, **f), torch.tensor(curv, **f)
        self.M = self.P.shape[0]
        self.on_path = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pidx = torch.zeros(n, dtype=torch.long, device=dev)
        self.pdir = torch.ones(n, **f)
        self.win = torch.arange(-10, 31, device=dev)
        self.scan = scan
        gx, gy = torch.meshgrid(torch.tensor(SCAN_X, **f), torch.tensor(SCAN_Y, **f), indexing="ij")
        self.sx, self.sy = gx.reshape(-1), gy.reshape(-1)
        self.scan_bias = z()
        self.yard = [s_ for s_ in COURSE["stairs"] if s_["name"] == "yard"]
        # the moving tiles: each its own rhythm (new for every episode in training), on a clock of their own
        T = self.tiles = COURSE["tiles"]
        nt = T["n"] * T["n"]
        self.tile_a, self.tile_b, self.tile_w, self.tile_p = z(nt), z(nt), z(nt), z(nt)
        self.tile_h = z(nt)
        self.tt = z()
        self.tiles_on = terrain == "circuit"
        self.stairs_main = [s_ for s_ in COURSE["stairs"] if s_["name"] == "circuit"][0]

    def ground(self, x, y, w=None):
        """the ground's height under x, y (bilinear between the heightfield's points; flat outside it), the moving tiles
        included (x, y: one row per robot, or w: the robots they belong to)"""
        import torch
        g = self.ground_static(x, y)
        if not self.tiles_on:
            return g
        T = self.tiles
        n = T["n"]
        u, v = (x - T["x0"]) / T["cell"], (y - T["y0"]) / T["cell"]
        inside = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        tid = (v.clamp(0, n - 1e-4).floor().long() * n + u.clamp(0, n - 1e-4).floor().long())
        if w is None:
            th = self.tile_h if tid.dim() == 2 else self.tile_h
            th = th.gather(1, tid) if tid.dim() == 2 else th.gather(1, tid.unsqueeze(-1)).squeeze(-1)
        else:
            th = self.tile_h[w].gather(1, tid.reshape(len(w), -1)).reshape(tid.shape)
        return torch.where(inside, torch.maximum(g, th), g)

    def tiles_now(self, idx=None):
        """the tiles' tops at the tiles' clock: a + a sin(w t + p) above their resting heights b (0 to 2a over b)"""
        import torch
        h = self.tile_b + self.tile_a * (1.0 + torch.sin(self.tile_w * self.tt.unsqueeze(-1) + self.tile_p))
        self.tile_h = h if self.tiles_on else torch.full_like(h, -5.0)
        self.sim.set_tiles(self.tile_h - self.tiles["half"])

    def new_tiles(self, idx, seed=None):
        """robots idx get a new pattern of tiles (in training: every episode)"""
        import torch
        T, k = self.tiles, idx.numel()
        if k == 0:
            return
        nt = T["n"] * T["n"]
        g = self.g if seed is None else torch.Generator(device=self.dev).manual_seed(seed)
        R = lambda a, b: a + (b - a) * torch.rand(k, nt, device=self.dev, generator=g)   # noqa: E731
        self.tile_a[idx] = R(*T["amp"])
        self.tile_b[idx] = R(*T["base"])
        self.tile_w[idx] = 2 * math.pi / R(*T["period"])
        self.tile_p[idx] = R(0.0, 2 * math.pi)
        self.tt[idx] = 0.0

    def ground_static(self, x, y):
        """the heightfield's ground under x, y"""
        import torch
        H = self.Hg
        nr, nc = H.shape
        gx = ((x - HF_X0) / HF_RES).clamp(0, nc - 1.0001)
        gy = ((y - HF_Y0) / HF_RES).clamp(0, nr - 1.0001)
        i0, j0 = gx.floor().long(), gy.floor().long()
        fx, fy = gx - i0, gy - j0
        return (H[j0, i0] * (1 - fx) * (1 - fy) + H[j0, i0 + 1] * fx * (1 - fy) + H[j0 + 1, i0] * (1 - fx) * fy + H[j0 + 1, i0 + 1] * fx * fy)

    def contacts(self):
        """each foot on the ground (the floor or the course)"""
        import torch
        l = (self.S("left_foot_floor_found")[:, 0] > 0) | (self.S("left_foot_terrain_found")[:, 0] > 0)
        r = (self.S("right_foot_floor_found")[:, 0] > 0) | (self.S("right_foot_terrain_found")[:, 0] > 0)
        if self.tiles_on:       # on a moving tile: its sole within a centimetre of the tile's top
            fh = self.feet_height()
            T = self.tiles
            S = T["cell"] * T["n"]
            lp, rp = self.S("left_foot_pos"), self.S("right_foot_pos")
            inl = (lp[:, 0] >= T["x0"]) & (lp[:, 0] < T["x0"] + S) & (lp[:, 1] >= T["y0"]) & (lp[:, 1] < T["y0"] + S)
            inr = (rp[:, 0] >= T["x0"]) & (rp[:, 0] < T["x0"] + S) & (rp[:, 1] >= T["y0"]) & (rp[:, 1] < T["y0"] + S)
            l, r = l | (inl & (fh[:, 0] < 0.049)), r | (inr & (fh[:, 1] < 0.049))
        return torch.stack([l, r], -1)

    def feet_height(self):
        """each foot's height above the ground under it"""
        import torch
        lp, rp = self.S("left_foot_pos"), self.S("right_foot_pos")
        return torch.stack([lp[:, 2] - self.ground(lp[:, 0], lp[:, 1]), rp[:, 2] - self.ground(rp[:, 0], rp[:, 1])], -1)

    def height(self):
        """the pelvis' height above the ground under it"""
        q = self.sim.qpos.float()
        return q[:, 2] - self.ground(q[:, 0], q[:, 1])

    # ---- helpers
    def S(self, name):
        return self.sim.sens[:, self.sadr[name]].float()

    def U(self, k, a, b, *shape):
        import torch
        return a + (b - a) * torch.rand(k, *shape, device=self.dev, generator=self.g)

    # ---- commands
    def sample_cmd(self, idx):
        import torch
        k = idx.numel()
        if k == 0:
            return
        self.speed[idx] = self.U(k, 0.0, self.vmax) * (torch.rand(k, device=self.dev, generator=self.g) > 0.08)
        free = torch.rand(k, device=self.dev, generator=self.g) < (0.2 if self.train else 0.0)
        self.mode_line[idx] = ~free
        rnd = torch.stack([self.U(k, -0.5, self.vmax), self.U(k, -0.4, 0.4) * (torch.rand(k, device=self.dev, generator=self.g) < 0.4),
                           self.U(k, -1.0, 1.0)], -1)
        rnd = rnd * (torch.rand(k, 1, device=self.dev, generator=self.g) > 0.08)
        self.cmd[idx] = torch.where(free.unsqueeze(-1), rnd, self.cmd[idx])
        self.t_cmd[idx] = 0

    def path_heading(self):
        """the way to walk: along the circuit's line (either way round) or along its straight line"""
        import torch
        ph = self.PH[self.pidx] + (self.pdir < 0).float() * math.pi
        return torch.where(self.on_path, ph, self.line_yaw)

    def guide(self):
        """the line follower: the turn and the sideways step that bring it back onto its line and its heading
        (a straight line, or the circuit's line: its nearest point found near the last one, the heading a little ahead)"""
        import torch
        q = self.sim.qpos
        x, y = q[:, 0].float(), q[:, 1].float()
        yaw = yaw_of(q[:, 3:7].float())
        cand = (self.pidx.unsqueeze(-1) + self.win.unsqueeze(0) * self.pdir.long().unsqueeze(-1)) % self.M
        d2 = (self.P[cand, 0] - x.unsqueeze(-1)) ** 2 + (self.P[cand, 1] - y.unsqueeze(-1)) ** 2
        self.pidx = torch.where(self.on_path, cand.gather(1, d2.argmin(-1, keepdim=True)).squeeze(-1), self.pidx)
        ahead = (self.pidx + 3 * self.pdir.long()) % self.M
        p_head = self.PH[ahead] + (self.pdir < 0).float() * math.pi
        nx, ny = -torch.sin(self.PH[self.pidx]), torch.cos(self.PH[self.pidx])
        p_off = ((x - self.P[self.pidx, 0]) * nx + (y - self.P[self.pidx, 1]) * ny) * self.pdir     # left of the way it walks (m)
        cy, sy = torch.cos(self.line_yaw), torch.sin(self.line_yaw)
        l_off = -(x * sy) + y * cy - self.line_y
        want = torch.where(self.on_path, p_head, self.line_yaw)
        off = torch.where(self.on_path, p_off, l_off)
        dyaw = torch.atan2(torch.sin(want - yaw), torch.cos(want - yaw))
        ff = torch.where(self.on_path, self.PK[self.pidx] * self.pdir * self.speed, torch.zeros_like(off))   # (round the corners)
        wz = (2.0 * dyaw - 0.6 * off * torch.cos(dyaw) + ff).clamp(-1.0, 1.0)
        vy = (-0.8 * off).clamp(-0.3, 0.3) * (self.speed > 0.05)
        wz = wz * ((self.speed > 0.05) | (dyaw.abs() > 0.15))
        line_cmd = torch.stack([self.speed, vy, wz], -1)
        self.cmd = torch.where(self.mode_line.unsqueeze(-1), line_cmd, self.cmd)
        self.yaw_err, self.line_off = dyaw, off

    # ---- reset
    def reset(self, idx, at=None):
        """robots idx start again (at: [x, y, yaw] to put one there, standing, instead of the usual start)"""
        import torch
        k = idx.numel()
        if k == 0:
            return
        qpos = self.q0.unsqueeze(0).repeat(k, 1)
        qvel = torch.zeros(k, self.m.nv, device=self.dev)
        if self.train:
            if self.terrain == "circuit":      # on the circuit (anywhere, near the stairs or on side 1, either way round) or in the yard
                yaw = self.spawn_circuit(idx, qpos)
            else:
                qpos[:, 0:2] += self.U(k, -0.5, 0.5, 2)
                yaw = self.U(k, -math.pi, math.pi)
            qpos[:, 7:] = qpos[:, 7:] * self.U(k, 0.5, 1.5, NU)
            qvel[:, 0:6] = self.U(k, -0.5, 0.5, 6)
        else:
            yaw = torch.zeros(k, device=self.dev)
        if not self.train:
            self.on_path[idx] = False
        if at is not None:
            qpos[:, 0], qpos[:, 1], yaw = float(at[0]), float(at[1]), torch.full((k,), float(at[2]), device=self.dev)
        # standing on the ground where it starts (the highest point under its feet, a moving tile's top included, and a centimetre)
        if self.train:
            self.new_tiles(idx)
        self.tiles_now()
        gmax = self.ground(qpos[:, 0], qpos[:, 1], w=idx)
        for dx, dy in ((0.15, 0.12), (0.15, -0.12), (-0.1, 0.12), (-0.1, -0.12)):
            gmax = torch.maximum(gmax, self.ground(qpos[:, 0] + dx * torch.cos(yaw) - dy * torch.sin(yaw), qpos[:, 1] + dx * torch.sin(yaw) + dy * torch.cos(yaw), w=idx))
        qpos[:, 2] = qpos[:, 2] + gmax + 0.01
        qpos[:, 3] = torch.cos(yaw / 2)
        qpos[:, 6] = torch.sin(yaw / 2)
        qpos[:, 7:] = torch.maximum(torch.minimum(qpos[:, 7:], self.soft_hi), self.soft_lo)
        self.sim.set_rows(idx, qpos, qvel, qpos[:, 7:])
        # the line: through where it starts, along a heading near its own
        yard_line = getattr(self, "_yard_line", None)
        if yard_line is not None:          # in the yard: the line of its staircase, either way
            self.line_yaw[idx] = yard_line[1]
            self.line_y[idx] = yard_line[0]
            self._yard_line = None
        else:
            self.line_yaw[idx] = yaw + (self.U(k, -0.4, 0.4) if self.train else 0.0)    # (Walk sets its own line after)
            self.line_y[idx] = -(qpos[:, 0] * torch.sin(self.line_yaw[idx])) + qpos[:, 1] * torch.cos(self.line_yaw[idx]) + (self.U(k, -0.5, 0.5) if self.train else 0.0)
        self.scan_bias[idx] = self.U(k, -0.02, 0.02) if self.train else 0.0
        f = self.U(k, 1.25, 1.5) if self.train else torch.full((k,), 1.4, device=self.dev)
        self.phase_dt[idx] = 2 * math.pi * CTRL_DT * f
        self.phase[idx] = torch.tensor([0.0, math.pi], device=self.dev)
        self.last_act[idx] = 0.0
        self.act[idx] = 0.0
        self.air[idx] = 0.0
        self.last_contact[idx] = False
        self.swing_peak[idx] = 0.0
        self.steps[idx] = 0
        self.push_every[idx] = (self.U(k, 5.0, 10.0) / CTRL_DT).long()
        self.ep_ret[idx] = 0.0
        self.ep_dist[idx] = 0.0
        self.fell[idx] = False
        self.sample_cmd(idx)

    def spawn_circuit(self, idx, qpos):
        """where robots idx start in training on the circuit: 25 % anywhere on its line, 20 % near its stairs, 12 % near the
        moving tiles, 8 % on side 1's rough ground and slopes, 35 % on a yard staircase; on the circuit either way round"""
        import torch
        k = idx.numel()
        u = torch.rand(k, device=self.dev, generator=self.g)
        M = self.M
        # on the circuit: a point of its line
        i_any = (torch.rand(k, device=self.dev, generator=self.g) * M).long() % M
        st = self.stairs_main
        mid = st["x0"] + st["dir"] * ((st["n"] - 1) * st["tread"] + st["landing"] / 2)
        s_st = self.nearest_index(torch.tensor([mid], device=self.dev), torch.tensor([st["y"]], device=self.dev))
        i_st = (s_st + (self.U(k, -1.0, 1.0) * 80).long()) % M          # within 8 m of the landing: the stairs and their approaches
        i_s1 = (self.nearest_index(torch.tensor([2.0], device=self.dev), torch.tensor([0.0], device=self.dev)) + (torch.rand(k, device=self.dev, generator=self.g) * 280).long()) % M
        tl = self.tiles
        i_tl = (self.nearest_index(torch.tensor([tl["x0"] + tl["cell"] * tl["n"] / 2], device=self.dev), torch.tensor([tl["y0"] + tl["cell"] * tl["n"] / 2], device=self.dev))
                + (self.U(k, -1.0, 1.0) * 50).long()) % M                # within 5 m of the moving tiles' centre
        pi = torch.where(u < 0.25, i_any, torch.where(u < 0.45, i_st, torch.where(u < 0.57, i_tl, i_s1)))
        dirn = torch.where(torch.rand(k, device=self.dev, generator=self.g) < 0.5, 1.0, -1.0)
        side = self.U(k, -0.3, 0.3)
        hx, hy = torch.cos(self.PH[pi]), torch.sin(self.PH[pi])
        px, py = self.P[pi, 0] - side * hy, self.P[pi, 1] + side * hx
        yaw = self.PH[pi] + (dirn < 0).float() * math.pi + self.U(k, -0.3, 0.3)
        # or in the yard: on one of its staircases, walking along it either way
        yard = u >= 0.65
        ys = torch.tensor([s_["y"] for s_ in self.yard], device=self.dev)
        which = (torch.rand(k, device=self.dev, generator=self.g) * len(self.yard)).long() % len(self.yard)
        y0 = self.yard[0]
        ln = 2 * (y0["n"] - 1) * y0["tread"] + y0["landing"]
        yx = self.U(k, y0["x0"] - 2.0, y0["x0"] + ln + 2.0)
        yy = ys[which] + self.U(k, -0.25, 0.25)
        yback = torch.rand(k, device=self.dev, generator=self.g) < 0.5
        yyaw = yback.float() * math.pi + self.U(k, -0.2, 0.2)
        qpos[:, 0] = torch.where(yard, yx, px)
        qpos[:, 1] = torch.where(yard, yy, py)
        self.on_path[idx] = ~yard
        self.pidx[idx] = pi
        self.pdir[idx] = dirn
        lyaw = yback.float() * math.pi
        self._yard_line = (torch.where(yard, -(0.0) * torch.sin(lyaw) + ys[which] * torch.cos(lyaw), self.line_y[idx]),
                           torch.where(yard, lyaw, yaw))
        return torch.where(yard, yyaw, yaw)

    def nearest_index(self, x, y):
        """the circuit's point nearest to each x, y"""
        d2 = (self.P[:, 0].unsqueeze(0) - x.unsqueeze(-1)) ** 2 + (self.P[:, 1].unsqueeze(0) - y.unsqueeze(-1)) ** 2
        return d2.argmin(-1)

    def put_on_path(self, idx, i, dirn):
        """robots idx follow the circuit from its point i, either way round (dirn +1 counterclockwise)"""
        self.on_path[idx] = True
        self.pidx[idx] = i
        self.pdir[idx] = float(dirn)

    def ground_scan(self, noisy):
        """the elevation map: under and ahead of it, turned with it, as heights below the pelvis' usual height"""
        import torch
        q = self.sim.qpos.float()
        yaw = yaw_of(q[:, 3:7])
        c, s_ = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        px = q[:, 0:1] + c * self.sx - s_ * self.sy
        py = q[:, 1:2] + s_ * self.sx + c * self.sy
        h = (q[:, 2:3] - hs(0.75) - self.ground(px, py)).clamp(-1.0, 1.0)
        if noisy and self.noise > 0:
            h = h + self.scan_bias.unsqueeze(-1) + (2 * torch.rand(h.shape, device=self.dev, generator=self.g) - 1) * 0.02 * self.noise
        return h

    def reset_all(self):
        import torch
        self.reset(torch.arange(self.n, device=self.dev))
        self.sim.refresh()

    # ---- what it senses
    def obs(self, noisy=True):
        import torch
        q, v = self.sim.qpos.float(), self.sim.qvel.float()
        quat = q[:, 3:7]
        grav = quat_rotate_inv(quat, torch.tensor([[0.0, 0.0, -1.0]], device=self.dev).expand(self.n, 3))
        lin, gyro = self.S("local_linvel_pelvis"), self.S("gyro_pelvis")
        jp, jv = q[:, 7:] - self.default, v[:, 6:]
        k = self.noise if noisy else 0.0
        nz = lambda x, s: x + (2 * torch.rand(x.shape, device=self.dev, generator=self.g) - 1) * s * k if k > 0 else x   # noqa: E731
        ph = torch.cat([torch.cos(self.phase), torch.sin(self.phase)], -1)
        actor = torch.cat([nz(lin, 0.1), nz(gyro, 0.2), nz(grav, 0.05), self.cmd, nz(jp, 0.03), nz(jv, 1.5), self.last_act, ph], -1)
        sc = self.ground_scan(noisy=False)
        if self.scan:
            actor = torch.cat([actor, self.ground_scan(noisy=noisy) if (noisy and k > 0) else sc], -1)
        contact = self.contacts().float()
        feet_v = torch.cat([self.S("left_foot_global_linvel"), self.S("right_foot_global_linvel")], -1)
        priv = torch.cat([torch.cat([lin, gyro, grav, self.cmd, jp, jv, self.last_act, ph], -1), self.height().unsqueeze(-1), contact, self.air,
                          feet_v, self.S("global_angvel_pelvis"), self.S("global_linvel_pelvis"), sc], -1)
        return actor, priv

    # ---- one control step
    def step(self, action):
        import torch
        sim, n = self.sim, self.n
        a = action.clamp(-5.0, 5.0)
        self.guide()
        # a push now and then (a shove of up to push m/s)
        if self.train and self.push_max > 0:
            due = (self.steps + 1) % self.push_every.clamp(min=1) == 0
            if bool(due.any()):
                th = self.U(n, 0.0, 2 * math.pi)
                mag = self.U(n, 0.1, self.push_max) * due
                sim.kick(torch.stack([torch.cos(th) * mag, torch.sin(th) * mag], -1))
        sim.ctrl.copy_((self.default + a * ACTION_SCALE).to(sim.ctrl.dtype))
        if self.tiles_on:
            self.tt += CTRL_DT
            self.tiles_now()
        sim.step()
        q, v = sim.qpos.float(), sim.qvel.float()
        contact = self.contacts()
        contact_f = contact | self.last_contact
        first = (self.air > 0) & contact_f
        self.air += CTRL_DT
        feet_z = self.feet_height()
        self.swing_peak = torch.maximum(self.swing_peak, feet_z)
        # falling, or feet tangled: the end
        tz = self.S("torso_zaxis")
        fell = (self.height() < hs(0.45)) | (tz[:, 2] < 0.5) | (self.S("right_foot_left_foot_found")[:, 0] > 0) | torch.isnan(q).any(-1) | torch.isnan(v).any(-1)
        # the rewards
        lin, gyro = self.S("local_linvel_pelvis"), self.S("gyro_pelvis")
        cmd = self.cmd
        yaw = yaw_of(q[:, 3:7])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        tzh = torch.stack([tz[:, 0] * cy + tz[:, 1] * sy, -tz[:, 0] * sy + tz[:, 1] * cy, tz[:, 2]], -1)
        jq = q[:, 7:]
        r = {}
        r["tracking_lin_vel"] = torch.exp(-((cmd[:, :2] - lin[:, :2]) ** 2).sum(-1) / 0.25)
        r["tracking_ang_vel"] = torch.exp(-(cmd[:, 2] - gyro[:, 2]) ** 2 / 0.25)
        r["ang_vel_xy"] = (self.S("global_angvel_torso")[:, :2] ** 2).sum(-1)
        r["orientation"] = ((tzh - self.tz0) ** 2).sum(-1)
        air = ((self.air - 0.2) * first).clamp(max=0.3)
        r["feet_air_time"] = air.sum(-1)
        body_v = self.S("global_linvel_pelvis")[:, :2].norm(dim=-1)
        r["feet_slip"] = (body_v.unsqueeze(-1) * contact).sum(-1)
        rz = rz_bezier(self.phase, 0.15)
        moving = (self.S("global_linvel_pelvis")[:, :2].norm(dim=-1) > 0.1) | (self.S("global_angvel_pelvis")[:, 2].abs() > 0.1) | (cmd.norm(dim=-1) > 0.01)
        r["feet_phase"] = torch.exp(-((feet_z - rz) ** 2).sum(-1) / 0.01) * moving
        still = cmd.norm(dim=-1) < 0.01
        r["stand_still"] = (jq - self.default).abs().sum(-1) * still
        r["termination"] = fell.float()
        r["alive"] = (~fell).float()                      # staying up is worth something from the very first step
        r["collision"] = ((self.S("left_hand_left_thigh_found")[:, 0] > 0) | (self.S("right_hand_right_thigh_found")[:, 0] > 0)).float()
        fl, fr = self.S("left_foot_force")[:, 2].abs(), self.S("right_foot_force")[:, 2].abs()
        r["contact_force"] = (fl - self.f_max).clamp(min=0) + (fr - self.f_max).clamp(min=0)
        r["joint_deviation_knee"] = (jq[:, self.knee_i] - self.default[self.knee_i]).abs().sum(-1)
        hip = (jq[:, self.hip_i] - self.default[self.hip_i]).abs()
        lat = (cmd[:, 1].abs() > 0.1).unsqueeze(-1)
        w_hip = torch.tensor([0.0, 1.0, 0.0, 1.0], device=self.dev)
        r["joint_deviation_hip"] = torch.where(lat, hip * w_hip, hip).sum(-1)
        r["dof_pos_limits"] = ((self.soft_lo - jq).clamp(min=0) + (jq - self.soft_hi).clamp(min=0)).sum(-1)
        r["pose"] = ((jq - self.default) ** 2).sum(-1)
        r["action_rate"] = ((a - self.last_act) ** 2).sum(-1)
        # never below zero (as Playground and legged_gym do): staying up always pays at least as well as falling
        rew = (sum(SCALES[k_] * v_ for k_, v_ in r.items()) * CTRL_DT).clamp(min=0.0)
        rew = torch.nan_to_num(rew, nan=-1.0)
        # the clock, the bookkeeping
        self.steps += 1
        self.t_cmd += 1
        self.phase = torch.fmod(self.phase + self.phase_dt.unsqueeze(-1) + math.pi, 2 * math.pi) - math.pi
        self.last_act = a
        self.air = self.air * (~contact)
        self.last_contact = contact
        self.swing_peak = self.swing_peak * (~contact)
        self.ep_ret += rew
        wh = self.path_heading()
        self.ep_dist += (v[:, 0] * torch.cos(wh) + v[:, 1] * torch.sin(wh)) * CTRL_DT
        timeout = self.steps >= EP_STEPS
        if self.train:
            new_cmd = (self.t_cmd >= 500).nonzero(as_tuple=False).flatten()
            self.sample_cmd(new_cmd)
        self.fell = fell
        info = {"fell": fell, "timeout": timeout & ~fell, "terms": r, "lin": lin, "gyro": gyro, "contact": contact, "feet_z": feet_z,
                "forces": torch.stack([fl, fr], -1)}
        return rew, fell | timeout, info


def quat_mul(a, b):
    """the product of quaternions a and b (w, x, y, z), row by row"""
    import torch
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


STAND_EP = 500                       # 10 s to get up and stay up
STAND_SCALES = {"height": 3.0, "upright": 2.0, "pose": 2.0, "feet": 0.5, "still": 1.0, "action_rate": -0.01, "joint_vel": -1e-4,
                "dof_pos_limits": -1.0}


def is_standing(h, tz):
    """on its feet: the pelvis at walking height and the torso upright"""
    return (h > hs(0.65)) & (tz > 0.85)


class StandEnv(WalkEnv):
    """Learning to stand up: each episode starts with the robot lying on the floor where a fall left it (falls the physics
    made: dropped from a random pose, any way round), and it has 10 s to get up and stay up. Rewards: its pelvis high,
    its torso upright, and once it stands, the standing pose, both feet down and stillness. At first an upward help on
    its torso (a share of its weight) makes getting up possible; the help is taken away as it succeeds."""

    def __init__(self, sim, mjm, seed=0, noise=1.0, train=True):
        super().__init__(sim, mjm, seed=seed, vmax=0.0, push=0.0, noise=noise, train=train, terrain="flat", scan=True)
        import torch
        self.torso = mjm.body("torso_link").id
        self.assist = 0.0
        self.weight = float(mjm.body_subtreemass[mjm.body("pelvis").id]) * 9.81      # the robot alone (not the floor tiles)
        self.bank = None
        n = self.n
        self.up_steps = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.t_up = torch.full((n,), -1.0, device=self.dev)
        self.start_standing = torch.zeros(n, dtype=torch.bool, device=self.dev)

    def build_bank(self, rounds=3, log=None):
        """lying poses: every robot dropped from 1 m in a random pose and turn, holding random joint targets, 2 s to land and settle"""
        import torch
        n, dev = self.n, self.dev
        idx = torch.arange(n, device=dev)
        keep = []
        span = self.soft_hi - self.soft_lo
        for _ in range(rounds):
            qpos = self.q0.unsqueeze(0).repeat(n, 1)
            qpos[:, 0:2] = self.U(n, -0.3, 0.3, 2)
            qpos[:, 2] = self.U(n, hs(1.0), hs(1.2))
            quat = torch.randn(n, 4, device=dev, generator=self.g)
            qpos[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True)
            qpos[:, 7:] = torch.maximum(torch.minimum(self.default + self.U(n, -0.3, 0.3, NU) * span, self.soft_hi), self.soft_lo)
            qvel = torch.zeros(n, self.m.nv, device=dev)
            qvel[:, 0:6] = self.U(n, -1.0, 1.0, 6)
            targets = torch.maximum(torch.minimum(self.default + self.U(n, -0.25, 0.25, NU) * span, self.soft_hi), self.soft_lo)
            self.sim.set_rows(idx, qpos, qvel, targets)
            self.sim.set_lift(self.torso, torch.zeros(n, device=dev))
            for _ in range(100):
                self.sim.step()
            q = self.sim.qpos.float().clone()
            ok = ~torch.isnan(q).any(-1) & (self.height() < hs(0.45)) & (self.sim.qvel.float().abs().amax(-1) < 5.0)
            keep.append(q[ok])
        self.bank = torch.cat(keep)
        if log:
            log("%d lying poses made by dropping the robot (%d rounds)." % (len(self.bank), rounds))
        return len(self.bank)

    def reset(self, idx, at=None):
        import torch
        k = idx.numel()
        if k == 0:
            return
        dev = self.dev
        if at is not None or self.bank is None or not len(self.bank):
            qpos = self.q0.unsqueeze(0).repeat(k, 1)
            stand = torch.ones(k, dtype=torch.bool, device=dev)
        else:
            qpos = self.bank[torch.randint(len(self.bank), (k,), device=dev, generator=self.g)].clone()
            stand = torch.rand(k, device=dev, generator=self.g) < (0.12 if self.train else 0.0)    # some start on their feet: staying up is part of it
            qpos[stand] = self.q0
        # any way round, anywhere near the middle
        yaw = self.U(k, -math.pi, math.pi)
        qz = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        qpos[:, 3:7] = quat_mul(qz, qpos[:, 3:7])
        qpos[:, 0:2] = self.U(k, -0.5, 0.5, 2)
        qvel = torch.zeros(k, self.m.nv, device=dev)
        self.sim.set_rows(idx, qpos, qvel, qpos[:, 7:])
        self.cmd[idx] = 0.0
        self.mode_line[idx] = False
        self.speed[idx] = 0.0
        self.phase[idx] = torch.tensor([0.0, math.pi], device=dev)
        self.phase_dt[idx] = 0.0          # (no gait clock while getting up)
        self.last_act[idx] = 0.0
        self.act[idx] = 0.0
        self.air[idx] = 0.0
        self.last_contact[idx] = False
        self.swing_peak[idx] = 0.0
        self.steps[idx] = 0
        self.ep_ret[idx] = 0.0
        self.ep_dist[idx] = 0.0
        self.fell[idx] = False
        self.up_steps[idx] = 0
        self.t_up[idx] = -1.0
        self.start_standing[idx] = stand
        self.scan_bias[idx] = self.U(k, -0.02, 0.02) if self.train else 0.0

    def step(self, action):
        import torch
        sim = self.sim
        a = action.clamp(-5.0, 5.0)
        sim.ctrl.copy_((self.default + a * ACTION_SCALE).to(sim.ctrl.dtype))
        # the help: a share of its weight lifting the torso while it is low (none once it is up)
        sim.set_lift(self.torso, self.assist * self.weight * (self.height() < hs(0.7)).float())
        sim.step()
        q, v = sim.qpos.float(), sim.qvel.float()
        h = self.height()
        tz = self.S("torso_zaxis")[:, 2]
        standing = is_standing(h, tz)
        self.up_steps = torch.where(standing, self.up_steps + 1, torch.zeros_like(self.up_steps))
        newly = (self.up_steps == 25) & (self.t_up < 0)
        self.t_up = torch.where(newly, self.steps.float() * CTRL_DT, self.t_up)
        contact = self.contacts()
        jq, jv = q[:, 7:], v[:, 6:]
        st = standing.float()
        r = {}
        r["height"] = ((h - hs(0.2)) / hs(0.55)).clamp(0.0, 1.0)
        r["upright"] = tz.clamp(0.0, 1.0) ** 2
        r["pose"] = torch.exp(-((jq - self.default) ** 2).sum(-1) / 2.0) * st
        r["feet"] = contact.all(-1).float() * st
        r["still"] = torch.exp(-(self.S("global_linvel_pelvis") ** 2).sum(-1) / 0.1) * st
        r["action_rate"] = ((a - self.last_act) ** 2).sum(-1)
        r["joint_vel"] = (jv ** 2).sum(-1)
        r["dof_pos_limits"] = ((self.soft_lo - jq).clamp(min=0) + (jq - self.soft_hi).clamp(min=0)).sum(-1)
        rew = sum(STAND_SCALES[k_] * v_ for k_, v_ in r.items()) * CTRL_DT
        bad = torch.isnan(q).any(-1) | torch.isnan(v).any(-1)
        rew = torch.nan_to_num(rew, nan=0.0)
        self.steps += 1
        self.last_act = a
        self.last_contact = contact
        self.ep_ret += rew
        timeout = self.steps >= STAND_EP
        self.fell = bad
        info = {"fell": bad, "timeout": timeout & ~bad, "terms": r, "standing": standing, "height": h, "upright": tz,
                "lin": self.S("local_linvel_pelvis"), "gyro": self.S("gyro_pelvis"), "contact": contact}
        return rew, timeout | bad, info


# ── learning: PPO with an actor (the policy: what the robot senses → joint targets) and a critic (how good a moment is,
# from more than the robot can sense: the privileged state of the simulation) ─────────────────────────────────────────
import math


class Norm:
    """running mean and variance of what goes into a network"""

    def __init__(self, torch, n, dev):
        self.t = torch
        self.mean = torch.zeros(n, device=dev)
        self.var = torch.ones(n, device=dev)
        self.count = torch.full((n,), 1e-4, device=dev)          # (one count per input: new inputs learn their scale fast)

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-2)).clamp(-5.0, 5.0)     # (what never changes stays near 0)

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count.cpu()}

    def load(self, st, dev):
        torch = self.t
        n = self.mean.shape[0]
        m, v, c = st["mean"].to(dev), st["var"].to(dev), st["count"]
        c = c.to(dev) if torch.is_tensor(c) else torch.full((m.shape[0],), float(c), device=dev)
        k = m.shape[0]               # (a policy with fewer inputs: the new ones start unscaled, with no count)
        self.mean[:k], self.var[:k], self.count[:k] = m, v, c
        if k < n:
            self.mean[k:], self.var[k:], self.count[k:] = 0.0, 1.0, 1e-4


def mlp(nn, i, sizes, o):
    layers, d = [], i
    for h in sizes:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def make_ac(torch, obs, priv, act, hidden=(512, 256, 128)):
    import torch.nn as nn

    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = mlp(nn, obs, hidden, act)
            self.critic = mlp(nn, priv, hidden, 1)
            self.log_std = nn.Parameter(torch.full((act,), math.log(0.25)))    # exploration: joint targets jitter by 0.25 x 0.5 rad at first (more and even a good walker falls)
            with torch.no_grad():     # a new policy starts by holding the default pose (its last layer near zero)
                self.actor[-1].weight.mul_(0.01)
                self.actor[-1].bias.zero_()
    return AC()


class PPO:
    def __init__(self, torch, dev, n_env, obs, priv, act, horizon=24, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5,
                 minibatches=4, entropy=0.001, desired_kl=0.01, schedule="fixed"):
        self.t, self.dev = torch, dev
        self.ac = make_ac(torch, obs, priv, act).to(dev)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.lr = lr
        self.on = Norm(torch, obs, dev)
        self.pn = Norm(torch, priv, dev)
        self.n, self.T, self.obs_dim, self.priv_dim, self.act_dim = n_env, horizon, obs, priv, act
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.ent, self.kl_want = gamma, lam, clip, epochs, minibatches, entropy, desired_kl
        self.schedule = schedule
        T, n = horizon, n_env
        z = lambda *s: torch.zeros(*s, device=dev)   # noqa: E731
        self.b_obs, self.b_priv, self.b_act = z(T, n, obs), z(T, n, priv), z(T, n, act)
        self.b_lp, self.b_val, self.b_rew, self.b_done, self.b_mu = z(T, n), z(T, n), z(T, n), z(T, n), z(T, n, act)

    def act(self, obs, priv, t=None, deterministic=False):
        torch = self.t
        with torch.no_grad():
            x = self.on(obs)
            mu = self.ac.actor(x)
            if deterministic:
                return mu
            std = self.ac.log_std.exp()
            a = mu + std * torch.randn_like(mu)
            if t is not None:
                v = self.ac.critic(self.pn(priv)).squeeze(-1)
                lp = (-0.5 * ((a - mu) / std) ** 2 - self.ac.log_std - 0.9189385).sum(-1)
                self.b_obs[t], self.b_priv[t], self.b_act[t], self.b_lp[t], self.b_val[t], self.b_mu[t] = obs, priv, a, lp, v, mu
            return a

    def record(self, t, rew, done, timeout, priv_next):
        """the reward of step t; a run cut by the clock (not a fall) is worth what the critic says of where it stopped"""
        torch = self.t
        with torch.no_grad():
            if bool(timeout.any()):
                v_next = self.ac.critic(self.pn(priv_next)).squeeze(-1)
                rew = rew + self.gamma * v_next * timeout.float()
            self.b_rew[t], self.b_done[t] = rew, done.float()

    def update(self, priv_last, critic_only=False):
        """one learning step on the batch gathered (critic_only: the policy holds still while the critic learns new ground)"""
        torch = self.t
        T, n = self.T, self.n
        with torch.no_grad():
            last_v = self.ac.critic(self.pn(priv_last)).squeeze(-1)
            adv = torch.zeros_like(self.b_rew)
            gae = torch.zeros(n, device=self.dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else self.b_val[t + 1]
                nonterm = 1.0 - self.b_done[t]
                delta = self.b_rew[t] + self.gamma * nv * nonterm - self.b_val[t]
                gae = delta + self.gamma * self.lam * nonterm * gae
                adv[t] = gae
            ret = adv + self.b_val
        N = T * n
        obs, priv, act = self.b_obs.reshape(N, -1), self.b_priv.reshape(N, -1), self.b_act.reshape(N, -1)
        lp_old, val_old, mu_old = self.b_lp.reshape(N), self.b_val.reshape(N), self.b_mu.reshape(N, -1)
        adv, ret = adv.reshape(N), ret.reshape(N)
        xo, xp = self.on(obs), self.pn(priv)
        size = N // self.mb
        stats = {"kl": 0.0, "vloss": 0.0, "ploss": 0.0}
        std_old = self.ac.log_std.detach().exp()
        stop, stop_ok = False, True
        for _ in range(self.epochs):
            if stop:
                break
            perm = torch.randperm(N, device=self.dev)
            for i in range(self.mb):
                idx = perm[i * size:(i + 1) * size]
                mu = self.ac.actor(xo[idx])
                ls = self.ac.log_std
                std = ls.exp()
                v = self.ac.critic(xp[idx]).squeeze(-1)
                # how far the policy has moved from the one that gathered this batch (KL)
                with torch.no_grad():
                    kl = (torch.log(std / std_old) + (std_old ** 2 + (mu_old[idx] - mu) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean()
                    if self.schedule == "adaptive":          # the step size follows it (as rsl_rl can)
                        if kl > 2 * self.kl_want:
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif 0 < kl < self.kl_want / 2:
                            self.lr = min(1e-2, self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                if kl > 4 * self.kl_want and stop_ok and not critic_only:
                    stop = True                                 # moved far enough on this batch: no more steps on it
                    break
                a_ = adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                nlp = (-0.5 * ((act[idx] - mu) / std) ** 2 - ls - 0.9189385).sum(-1)
                ratio = (nlp - lp_old[idx]).exp()
                pl = -torch.min(ratio * a_, ratio.clamp(1 - self.clip, 1 + self.clip) * a_).mean()
                vc = val_old[idx] + (v - val_old[idx]).clamp(-self.clip, self.clip)
                vl = torch.max((v - ret[idx]) ** 2, (vc - ret[idx]) ** 2).mean()
                ent = (ls + 1.4189385).sum()
                loss = vl if critic_only else pl + 1.0 * vl - self.ent * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()
                stats["kl"], stats["vloss"], stats["ploss"] = float(kl), float(vl.detach()), float(pl.detach())
        with torch.no_grad():     # (the input scaling follows what it sees, after the update that used the old one)
            self.on.update(obs)
            self.pn.update(priv)
        stats["std"] = float(self.ac.log_std.detach().exp().mean())
        stats["lr"] = self.lr
        return stats

    def state(self):
        return {"ac": self.ac.state_dict(), "obs_norm": self.on.state(), "priv_norm": self.pn.state(), "lr": self.lr, "opt": self.opt.state_dict()}

    def load(self, st):
        sd = st["ac"]
        same = sd["actor.0.weight"].shape[1] == self.obs_dim and sd["critic.0.weight"].shape[1] == self.priv_dim
        self.ac.load_state_dict(widen(self.t, sd, self.obs_dim, self.priv_dim))
        self.on.load(st["obs_norm"], self.dev)
        self.pn.load(st["priv_norm"], self.dev)
        self.lr = float(st.get("lr") or self.lr)
        # the optimizer's running moments too, when the network is the same shape: training carries on where it was
        if same and st.get("opt"):
            try:
                self.opt.load_state_dict(st["opt"])
            except Exception:      # noqa: BLE001
                pass


def widen(torch, sd, obs, priv):
    """a policy's weights for more inputs than it had: the new inputs' weights start at zero (it acts as before)"""
    sd = dict(sd)
    for key, want in (("actor.0.weight", obs), ("critic.0.weight", priv)):
        w = sd[key]
        if w.shape[1] < want:
            sd[key] = torch.cat([w, torch.zeros(w.shape[0], want - w.shape[1], dtype=w.dtype)], 1)
    return sd


def load_policy(path, dev="cpu"):
    """the actor of a saved policy: obs → joint targets (deterministic), with its input scaling"""
    import torch
    st = torch.load(path, map_location=dev, weights_only=False)
    ac = make_ac(torch, st["obs_dim"], st["priv_dim"], st["act_dim"]).to(dev)
    ac.load_state_dict(st["ac"])
    ac.eval()
    on = Norm(torch, st["obs_dim"], dev)
    on.load(st["obs_norm"], dev)

    def act(obs):
        with torch.no_grad():
            return ac.actor(on(obs))
    return act, st


# ── what the 3D view gets (esasim.humanoid/1): the robot once (its body tree, to pose its 3D model), then frames ──────
FORMAT = "esasim.humanoid-policy/1"
TASK = "walk"
def make_fields(nu):
    return (["t", "x", "y", "z", "qw", "qx", "qy", "qz"] + ["j%d" % k for k in range(nu)]
            + ["cvx", "cvy", "cwz", "vx", "vy", "wz", "cl", "cr", "fl", "fr"]
            + ["h", "gx", "gy", "gz", "ax", "ay", "az"] + ["tq%d" % k for k in range(nu)] + ["tt"])


FIELDS = make_fields(29)
# h: the ground under the pelvis (m); gx..gz: the pelvis' gyro (rad/s); ax..az: its accelerometer (m/s²); tq: each motor's torque (N m)


GROUND_NAME = {"flat": "the flat floor", "course": "the straight test course", "circuit": "the circuit"}


def make_zones(c=COURSE):
    """the parts of the circuit, as boxes: (key, x0, x1, y0, y1, kind, what it is; %s: up or down)"""
    z = []
    for pa in c["patches"]:
        mx = max(pa["mm"]) / 10.0
        z.append((pa["kind"], pa["x0"], pa["x0"] + pa["size"], pa["y0"], pa["y0"] + pa["size"], "rough",
                  "Rough ground: %s up to %.0f cm" % (pa["kind"], mx)))
    rp = c["ramp"]
    pct = 100.0 * rp["h"] / (rp["x"][1] - rp["x"][0])
    w = rp["half"]
    z.append(("slope_a", rp["x"][0], rp["x"][1], -w, w, "slope", "Slope %s: " + "%.1f %%%% (%.0f°)" % (pct, math.degrees(math.atan(pct / 100)))))
    z.append(("plateau", rp["x"][1], rp["x"][2], -w, w, "flat", "On the plateau, %.1f m up" % rp["h"]))
    z.append(("slope_b", rp["x"][2], rp["x"][3], -w, w, "slope", "Slope %s: " + "%.1f %%%%" % pct))
    for st in c["stairs"]:
        if st["name"] != "circuit":
            continue
        up = (st["n"] - 1) * st["tread"]
        u_edges = [0.0, up, up + st["landing"], 2 * up + st["landing"]]
        xs = [st["x0"] + st["dir"] * u for u in u_edges]
        y0, y1 = st["y"] - st["width"] / 2, st["y"] + st["width"] / 2
        txt = "Stairs %s: " + "%d steps of %.0f cm" % (st["n"], 100 * st["rise"])
        z.append(("stairs_a", min(xs[0], xs[1]), max(xs[0], xs[1]), y0, y1, "stairs", txt))
        z.append(("landing", min(xs[1], xs[2]), max(xs[1], xs[2]), y0, y1, "flat", "On the landing, %.1f m up" % (st["n"] * st["rise"])))
        z.append(("stairs_b", min(xs[2], xs[3]), max(xs[2], xs[3]), y0, y1, "stairs", txt))
    tl = c.get("tiles")
    if tl:
        S = tl["cell"] * tl["n"]
        z.append(("tiles", tl["x0"], tl["x0"] + S, tl["y0"], tl["y0"] + S, "rough",
                  "Moving tiles: each rises and falls by up to %.0f cm" % (200 * tl["amp"][1])))
    return z


ZONES = make_zones()


def terrain_of(p, default):
    t = str(p.get("terrain") or default).lower()
    return "circuit" if ("circuit" in t or "course" in t or "rough" in t or "stairs" in t) else "flat"


def robot_info(m, terrain="circuit", scan=True):
    """the body tree for the view: each body's parent, its place and turn on its parent, its joint (axis, slot in qpos)"""
    bodies = []
    for b in range(1, m.nbody):
        if m.body_mocapid[b] >= 0:
            continue                   # (the moving tiles: the view draws them from their rhythms)
        j = int(m.body_jntadr[b]) if m.body_jntnum[b] > 0 else -1
        jt = int(m.jnt_type[j]) if j >= 0 else -1
        bodies.append({"name": m.body(b).name, "parent": int(m.body_parentid[b]) - 1,
                       "pos": [round(float(x), 6) for x in m.body_pos[b]], "quat": [round(float(x), 6) for x in m.body_quat[b]],
                       "jtype": "free" if jt == 0 else "hinge" if jt == 3 else None,
                       "axis": [round(float(x), 4) for x in m.jnt_axis[j]] if jt == 3 else None,
                       "qadr": int(m.jnt_qposadr[j]) if j >= 0 else None})
    joints = [m.joint(i).name for i in range(1, m.njnt)]
    limits = [[round(float(m.jnt_range[i][0]), 3), round(float(m.jnt_range[i][1]), 3)] for i in range(1, m.njnt)]
    tmax = [round(float(abs(m.jnt_actfrcrange[int(m.actuator_trnid[a][0])][1])), 1) for a in range(m.nu)]
    R = ROBOTS[ROBOT]
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    feet = []
    for side in ("left_foot", "right_foot"):   # the link each sole is on: how high its origin stands over the floor, how far ahead the sole's middle is
        g = m.geom(side).id
        b = int(m.geom_bodyid[g])
        feet.append({"body": m.body(b).name, "sole": round(float(d.xpos[b][2] - d.geom_xpos[g][2] + m.geom_size[g][2]), 3),
                     "ahead": round(float(d.geom_xpos[g][0] - d.xpos[b][0]), 3), "size": [round(2 * float(m.geom_size[g][0]), 3), round(2 * float(m.geom_size[g][1]), 3)]})
    return {"model": ROBOT, "feet": feet, "name": R["name"], "short": R["short"], "glb": R["view"], "credit": R["credit"], "h0": H0,
            "mass": round(float(m.body_subtreemass[m.body("pelvis").id]), 1),
            "bodies": bodies, "joints": joints, "limits": limits, "torque_max": tmax,
            "default": [round(float(x), 4) for x in m.key("knees_bent").qpos[7:]], "fields": FIELDS,
            "course": COURSE if terrain == "circuit" else None, "scan": {"x": SCAN_X, "y": SCAN_Y, "base": round(hs(0.75), 3)} if scan else None,
            "camera": R["camera"]}


def frame_row(env, i, t):
    import torch
    q = env.sim.qpos[i].float().tolist()
    c = env.cmd[i].tolist()
    lin = env.S("local_linvel_pelvis")[i].tolist()
    gy = env.S("gyro_pelvis")[i].tolist()
    ct = env.contacts()[i].tolist()
    cl, cr = int(ct[0]), int(ct[1])
    wt = env.f_max / 500.0 * 340.0                                 # its weight (N): the loads in body weights
    fl = float(env.S("left_foot_force")[i, 2].abs()) / wt
    fr = float(env.S("right_foot_force")[i, 2].abs()) / wt
    r3, r4 = (lambda v: round(v, 3)), (lambda v: round(v, 4))
    return ([round(t, 3), r3(q[0]), r3(q[1]), r3(q[2])] + [r4(v) for v in q[3:7]] + [r3(v) for v in q[7:7 + NU]]
            + [round(c[0], 2), round(c[1], 2), round(c[2], 2), round(lin[0], 2), round(lin[1], 2), round(gy[2], 2), cl, cr, round(fl, 2), round(fr, 2)]
            + [r3(float(env.ground(env.sim.qpos[i:i + 1, 0].float(), env.sim.qpos[i:i + 1, 1].float(), w=torch.tensor([i], device=env.dev))[0]))] + [round(v, 2) for v in gy]
            + [round(v, 1) for v in env.S("accelerometer_pelvis")[i].tolist()] + [round(v, 1) for v in env.sim.tq[i].float().tolist()] + [round(float(env.tt[i]), 3)])


def tiles_item(env, i):
    """robot i's moving tiles, for the view (it moves them on the frames' tile clock tt)"""
    r = lambda t: [round(v, 4) for v in t[i].tolist()]   # noqa: E731
    return {"a": r(env.tile_a), "b": r(env.tile_b), "w": r(env.tile_w), "p": r(env.tile_p)}


def chunks(rows, budget=40000):
    """the frames in pieces small enough for a telemetry item (at most 60 KB each)"""
    import json
    per = max(1, len(json.dumps(rows[0]))) * 1.15 if rows else 1
    size = max(10, int(budget / per))
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size], i + size >= len(rows)


# ── policies to carry on from ─────────────────────────────────────────────────────────────────────────────────────
def policy_stamp(path, name):
    """when a policy was made: the date in its name (humanoid_walk_20261112_1926_64.pt), else its file's time"""
    import os
    import re
    import time
    m = re.search(r"(\d{8})_(\d{4})", str(name))
    if m:
        try:
            return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))
        except ValueError:
            pass
    return os.path.getmtime(path)


def find_policies(ctx, task, terrain, checkpoint):
    """The policies a training can carry on from: those wired in (the dataset), and those that stopped runs left on this
    machine (the checkpoint in their task folder, next to this one). Those that learned this ground come first, then the
    one with the most experience (the furthest along), then the newest."""
    import glob
    import os
    import torch
    out, others, foreign = [], [0], [0]

    def consider(path, name, where):
        try:
            st = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:     # noqa: BLE001
            ctx.log("Skipped %s: %s" % (name, e))
            return
        if st.get("format") != FORMAT:
            foreign[0] += 1         # (a checkpoint of another block, or of an older version of this one)
            return
        if (st.get("task") or TASK) != task:
            return                  # (a policy of the other skill: walking, or standing up)
        if (st.get("robot") or "g1") != ROBOT:
            others[0] += 1          # (another robot's: kept as it is, for that robot)
            return
        out.append({"same": 1 if (st.get("terrain") or "flat") == terrain else 0, "steps": int(st.get("steps") or 0),
                    "stamp": policy_stamp(path, name), "name": str(name), "where": where, "st": st, "path": path})
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            consider(f.path, f.name, "dataset")
    here = os.getcwd()
    jobs = os.path.dirname(os.path.dirname(here))
    if os.path.basename(jobs).lower() == "jobs":
        paths = sorted(glob.glob(os.path.join(jobs, "*", "*", checkpoint)), key=os.path.getmtime, reverse=True)[:30]
        for pth in paths:
            if os.path.dirname(os.path.abspath(pth)) == os.path.abspath(here):
                continue
            consider(pth, "the policy run #%s left on this machine" % os.path.basename(os.path.dirname(os.path.dirname(pth))), "machine")
    out.sort(key=lambda x: (x["same"], x["steps"], x["stamp"]), reverse=True)
    if others[0]:
        ctx.log("%d policies of other robots left as they are (each robot carries on from its own)." % others[0])
    if foreign[0]:
        ctx.log("Skipped %d checkpoints made by other blocks or older versions of this one." % foreign[0])
    return out


def start_over(p):
    return str(p.get("training") or "").lower().startswith("start") or bool(p.get("from_scratch"))


# ── learning to walk ─────────────────────────────────────────────────────────────────────────────────────────────────
def learn_run(ctx):
    import collections
    import csv
    import os
    import re
    import shutil
    import time
    import mujoco
    import torch
    p = ctx.params
    n = int(p.get("robots") or 4096)
    minutes = float(p.get("minutes") or 90)
    vmax = float(p.get("max_speed") or 1.0)
    replay_every = float(p.get("replay_every") or 20)
    seed = int(p.get("seed") or 1)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    terrain = terrain_of(p, "flat")
    torch.manual_seed(seed)
    mjm = load_model(mujoco, terrain)
    sim, label = make_sim(mjm, n, p.get("accelerator") or "auto", ctx.log)
    n = sim.n
    dev = sim.tdev
    env = WalkEnv(sim, mjm, seed=seed, vmax=vmax, push=float(p.get("push") if p.get("push") not in (None, "") else 1.5), terrain=terrain, scan=True)
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, NU, horizon=24)
    steps_done, warm = 0, 0
    found = [] if start_over(p) else find_policies(ctx, TASK, terrain, "policy_checkpoint.pt")
    if start_over(p):
        ctx.log("Starting over (Training: Start over): a new policy that learns from nothing.")
    elif found:
        pick = found[0]
        st = pick["st"]
        algo.load(st)
        steps_done = int(st.get("steps") or 0)
        if (st.get("terrain") or "flat") != terrain or int(st.get("obs_dim") or 0) < OBS_DIM:
            warm = 25          # new ground or new inputs: the critic learns what they are worth first, the walk is left as it is
        if int(st.get("obs_dim") or 0) < OBS_DIM:
            ctx.log("It learned blind: it now gets an elevation map of the ground ahead (from its head camera), which it learns to use.")
        ctx.log("Continuing from %s (walk score %.0f%% on %s, %.0f million steps of experience)." % (pick["name"], float(st.get("score") or 0), GROUND_NAME.get(st.get("terrain") or "flat", "?"), steps_done / 1e6))
    else:
        ctx.log("No policy to carry on from yet: a new one learns from nothing.")
    # The run's time limit (Stop after): learning ends a few minutes before it, so the policy is written and kept
    # rather than lost when the worker stops the block.
    left = ctx.time_left() if hasattr(ctx, "time_left") else None
    if left is not None and left / 60.0 - 3.0 < minutes:
        cut = max(1.0, left / 60.0 - 3.0)
        ctx.log("The run stops after %.0f minutes (Stop after): learning ends at %.0f minutes instead of %.0f, so the policy is kept. Raise Stop after to learn longer." % (ctx.max_minutes, cut, minutes))
        minutes = cut
    ctx.log("The %s learning to walk on %s, on %s: %d robots at once, for up to %.0f minutes." % (ROBOTS[ROBOT]["name"], GROUND_NAME[terrain], label, n, minutes))
    env.reset_all()
    a_obs, a_priv = env.obs()
    ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, terrain)})
    curve = open(ctx.output("curve", "training.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "steps", "walk_score", "fell_pct", "speed_err", "episode_s", "reward", "steps_per_s"])
    hist = collections.deque(maxlen=max(200, min(4000, n)))    # finished episodes: (fell, length s, return, distance)
    track = collections.deque(maxlen=200)                       # (tracking, speed error) per iteration
    t_start, last_tel, last_replay = time.time(), 0.0, -1e9
    it, best, stop_reason = 0, -1.0, "time"
    out_path = os.path.join(os.getcwd(), "policy_checkpoint.pt")
    rec, rec_ep = [], 0
    sim_t0 = 0.0

    def score():
        if len(hist) < 50 or not track:
            return 0.0
        fall = sum(h[0] for h in hist) / len(hist)
        tr = sum(x[0] for x in track) / len(track)
        return 100.0 * tr * (1.0 - fall)

    def save(note):
        st = algo.state()
        st.update({"format": FORMAT, "task": TASK, "robot": ROBOT, "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": NU, "score": score(),
                   "note": note, "steps": steps_done, "device": label, "minutes": (time.time() - t_start) / 60.0,
                   "action_scale": ACTION_SCALE, "default": env.default.cpu(), "vmax": vmax, "ctrl_dt": CTRL_DT, "terrain": terrain})
        torch.save(st, out_path)

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed > minutes * 60:
                break
            if any(isinstance(m.get("data") if isinstance(m, dict) else None, dict) and str(m["data"].get("control", "")).lower() in ("finish now", "finish")
                   for m in ctx.messages(ch_ctl)):
                stop_reason = "asked"
                break
            t0 = time.time()
            tr_sum, err_sum = 0.0, 0.0
            for t in range(algo.T):
                act = algo.act(a_obs, a_priv, t)
                rew, done, info = env.step(act)
                o2, p2 = env.obs()
                algo.record(t, rew, done, info["timeout"], p2)
                tr_sum += float(info["terms"]["tracking_lin_vel"].mean())
                err_sum += float((env.cmd[:, 0] - info["lin"][:, 0]).abs().mean())
                # one robot, replayed in 3D from time to time
                if it % 1 == 0:
                    if int(env.steps[0]) % 2 == 0:
                        rec.append(frame_row(env, 0, sim_t0 + float(env.steps[0]) * CTRL_DT))
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=False).flatten()
                    fell = info["fell"][idx].tolist()
                    lens = env.steps[idx].tolist()
                    rets = env.ep_ret[idx].tolist()
                    dist = env.ep_dist[idx].tolist()
                    for a_, b_, c_, d_ in zip(fell, lens, rets, dist):
                        hist.append((1.0 if a_ else 0.0, b_ * CTRL_DT, c_, d_))
                    if bool(done[0]):
                        if rec and time.time() - last_replay > replay_every and len(rec) > 25:
                            last_replay = time.time()
                            rec_ep += 1
                            res = {"fell": bool(info["fell"][0]), "seconds": round(float(env.steps[0]) * CTRL_DT, 1), "distance": round(float(env.ep_dist[0]), 1)}
                            lab = "Episode %d: %s after %.0f s, %.1f m along its line" % (rec_ep, "fell" if res["fell"] else "still walking", res["seconds"], res["distance"])
                            for k, part, last in chunks(rec):
                                ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "training", "ep": rec_ep, "seq": k, "last": last, "frames": part,
                                                       "result": res, "label": lab, "tiles": tiles_item(env, 0) if env.tiles_on else None, **({"path": int(env.pdir[0])} if bool(env.on_path[0]) else {"line": [round(float(env.line_y[0]), 3), round(float(env.line_yaw[0]), 4)]})})
                        rec = []
                    env.reset(idx)
                    sim.refresh()
                    o2, p2 = env.obs()
                a_obs, a_priv = o2, p2
            stats = algo.update(a_priv, critic_only=it < warm)
            if warm and it == warm - 1:
                ctx.log("The critic has learned the new ground (%d iterations): the policy starts to adapt." % warm)
            it += 1
            steps_done += algo.T * n
            sps = algo.T * n / max(1e-6, time.time() - t0)
            track.append((tr_sum / algo.T, err_sum / algo.T))
            sc = score()
            if sc > best and len(hist) >= 50 and it > 20:
                best = sc
                save("best so far")
            if time.time() - last_tel > 2.0 or it == 1:
                last_tel = time.time()
                fall = 100.0 * sum(h[0] for h in hist) / max(1, len(hist))
                eps = sum(h[1] for h in hist) / max(1, len(hist))
                ret = sum(h[2] for h in hist) / max(1, len(hist))
                dist = sum(h[3] for h in hist) / max(1, len(hist))
                err = track[-1][1]
                row = {"iteration": it, "minutes": round(elapsed / 60, 2), "steps_m": round(steps_done / 1e6, 2), "walk_score": round(sc, 1),
                       "fell": round(fall, 1), "speed_err": round(err, 3), "episode_s": round(eps, 1), "reward": round(ret, 2), "distance": round(dist, 1),
                       "steps_per_s": round(sps), "std": round(stats["std"], 3), "kl": round(stats["kl"], 4), "lr": stats["lr"]}
                ctx.telemetry(ch_train, row)
                cw.writerow([row["minutes"], it, steps_done, row["walk_score"], row["fell"], row["speed_err"], row["episode_s"], row["reward"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "walk score %.0f%% · falls %.0f%% · %.0f M steps" % (sc, fall, steps_done / 1e6))
    finally:
        curve.close()
    if best < 0 or score() >= best:
        save("final")
    name = "humanoid_%swalk_%s%s_%02d.pt" % ("" if ROBOT == "g1" else ROBOT + "_", "circuit_" if terrain == "circuit" else "", time.strftime("%Y%m%d_%H%M"), int(round(max(0.0, best, score()))))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s (walk score %.0f%%: how well it keeps the asked speed, times the share of runs it does not fall)." % (name, max(0.0, best, score())))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ({"time": "time is up", "asked": "finished on request"}[stop_reason], it, steps_done / 1e6, (time.time() - t_start) / 60))
    return {"walk_score": round(max(0.0, best, score()), 1), "steps_m": round(steps_done / 1e6, 1)}


# ── learning to stand up after a fall ───────────────────────────────────────────────────────────────────────────────
ASSIST0 = 0.8          # the help it starts with: an upward pull on its torso of 80% of its weight, while it is low


def stand_run(ctx):
    import collections
    import csv
    import math
    import os
    import shutil
    import time
    import mujoco
    import torch
    p = ctx.params
    n = int(p.get("robots") or 4096)
    minutes = float(p.get("minutes") or 90)
    replay_every = float(p.get("replay_every") or 20)
    seed = int(p.get("seed") or 1)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    torch.manual_seed(seed)
    mjm = load_model(mujoco, "flat")
    sim, label = make_sim(mjm, n, p.get("accelerator") or "auto", ctx.log)
    n = sim.n
    dev = sim.tdev
    env = StandEnv(sim, mjm, seed=seed)
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, NU, horizon=32)
    steps_done, assist = 0, ASSIST0
    found = [] if start_over(p) else find_policies(ctx, "standup", "flat", "standup_checkpoint.pt")
    if start_over(p):
        ctx.log("Starting over (Training: Start over): a new policy that learns from nothing.")
    elif found:
        st = found[0]["st"]
        algo.load(st)
        steps_done = int(st.get("steps") or 0)
        assist = float(st.get("assist") if st.get("assist") is not None else ASSIST0)
        ctx.log("Continuing from %s (stand-up score %.0f%%, help %.0f%% of its weight, %.0f million steps of experience)." % (found[0]["name"], float(st.get("score") or 0), 100 * assist, steps_done / 1e6))
    else:
        ctx.log("No stand-up policy to carry on from yet: a new one learns from nothing.")
    left = ctx.time_left() if hasattr(ctx, "time_left") else None
    if left is not None and left / 60.0 - 3.0 < minutes:
        cut = max(1.0, left / 60.0 - 3.0)
        ctx.log("The run stops after %.0f minutes (Stop after): learning ends at %.0f minutes instead of %.0f, so the policy is kept. Raise Stop after to learn longer." % (ctx.max_minutes, cut, minutes))
        minutes = cut
    rounds = max(2, min(12, int(math.ceil(6000.0 / n))))
    env.build_bank(rounds, ctx.log)
    env.assist = assist
    ctx.log("The " + ROBOTS[ROBOT]["name"] + " learning to stand up on the flat floor, on %s: %d robots at once, for up to %.0f minutes. Each try starts lying where a fall left it, with %.0f s to get up and stay up." % (label, n, minutes, STAND_EP * CTRL_DT))
    ctx.log("At first an upward help on its torso (%.0f%% of its weight) makes it possible; the help goes down as it succeeds, to none." % (100 * assist))
    env.reset_all()
    a_obs, a_priv = env.obs()
    ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, "flat")})
    curve = open(ctx.output("curve", "standup.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "steps", "stand_score", "up_pct", "time_to_up_s", "assist_pct", "reward", "steps_per_s"])
    hist = collections.deque(maxlen=max(300, min(4000, n)))     # finished tries from the floor: (up at the end, seconds to get up, return)
    t_start, last_tel, last_replay, last_cut = time.time(), 0.0, -1e9, 0
    it, best, stop_reason = 0, -1.0, "time"
    out_path = os.path.join(os.getcwd(), "standup_checkpoint.pt")
    rec, rec_ep = [], 0

    def up_rate():
        return sum(h[0] for h in hist) / len(hist) if hist else 0.0

    def score():
        if len(hist) < 100:
            return 0.0
        return 100.0 * up_rate() * (1.0 - env.assist / ASSIST0)

    def save(note):
        st = algo.state()
        st.update({"format": FORMAT, "task": "standup", "robot": ROBOT, "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": NU, "score": score(),
                   "note": note, "steps": steps_done, "device": label, "minutes": (time.time() - t_start) / 60.0, "assist": env.assist,
                   "up_pct": 100.0 * up_rate(), "action_scale": ACTION_SCALE, "default": env.default.cpu(), "ctrl_dt": CTRL_DT, "terrain": "flat"})
        torch.save(st, out_path)

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed > minutes * 60:
                break
            if any(isinstance(m.get("data") if isinstance(m, dict) else None, dict) and str(m["data"].get("control", "")).lower() in ("finish now", "finish")
                   for m in ctx.messages(ch_ctl)):
                stop_reason = "asked"
                break
            t0 = time.time()
            for t in range(algo.T):
                act = algo.act(a_obs, a_priv, t)
                rew, done, info = env.step(act)
                o2, p2 = env.obs()
                algo.record(t, rew, done, info["timeout"], p2)
                if int(env.steps[0]) % 2 == 0:
                    rec.append(frame_row(env, 0, float(env.steps[0]) * CTRL_DT))
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=False).flatten()
                    up = (env.up_steps[idx] >= 25) & ~info["fell"][idx]
                    for u_, tu, r_, s_ in zip(up.tolist(), env.t_up[idx].tolist(), env.ep_ret[idx].tolist(), env.start_standing[idx].tolist()):
                        if not s_:
                            hist.append((1.0 if u_ else 0.0, tu if tu >= 0 else None, r_))
                    if bool(done[0]):
                        if rec and time.time() - last_replay > replay_every and len(rec) > 25 and not bool(env.start_standing[0]):
                            last_replay = time.time()
                            rec_ep += 1
                            ok, tu = bool(up[0]), float(env.t_up[0])
                            res = {"fell": not ok, "seconds": round(float(env.steps[0]) * CTRL_DT, 1), "distance": 0.0}
                            lab = ("Try %d: up %sin %.1f s" % (rec_ep, "" if env.assist > 0.005 else "on its own ", tu) if ok and tu >= 0 else "Try %d: still down after %.0f s" % (rec_ep, res["seconds"])) + \
                                  (" (help %.0f%% of its weight)" % (100 * env.assist) if env.assist > 0.005 else "")
                            for k, part, last in chunks(rec):
                                ctx.telemetry(ch_bot, {"format": "esasim.humanoid/1", "kind": "training", "ep": rec_ep, "seq": k, "last": last, "frames": part,
                                                       "result": res, "label": lab})
                        rec = []
                    env.reset(idx)
                    sim.refresh()
                    o2, p2 = env.obs()
                a_obs, a_priv = o2, p2
            stats = algo.update(a_priv)
            it += 1
            steps_done += algo.T * n
            sps = algo.T * n / max(1e-6, time.time() - t0)
            # less help as it succeeds: more than 60% of the tries from the floor up and staying up
            if env.assist > 0 and len(hist) >= min(300, max(100, n)) and it - last_cut >= 10 and up_rate() > 0.6:
                env.assist = max(0.0, env.assist - 0.05) if env.assist > 0.06 else 0.0
                last_cut = it
                hist.clear()
                ctx.log("It gets up in most tries: the help goes down to %.0f%% of its weight." % (100 * env.assist) if env.assist > 0 else
                        "It gets up with almost no help: from now on it gets up on its own, with no help at all.")
            sc = score()
            if sc > best and len(hist) >= 100 and it > 20:
                best = sc
                save("best so far")
            if time.time() - last_tel > 2.0 or it == 1:
                last_tel = time.time()
                ups = [h[1] for h in hist if h[0] and h[1] is not None]
                row = {"iteration": it, "minutes": round(elapsed / 60, 2), "steps_m": round(steps_done / 1e6, 2), "stand_score": round(sc, 1),
                       "up_pct": round(100.0 * up_rate(), 1), "time_to_up": round(sum(ups) / len(ups), 2) if ups else None,
                       "assist": round(100.0 * env.assist, 1), "reward": round(sum(h[2] for h in hist) / max(1, len(hist)), 2),
                       "steps_per_s": round(sps), "std": round(stats["std"], 3), "kl": round(stats["kl"], 4), "lr": stats["lr"]}
                ctx.telemetry(ch_train, row)
                cw.writerow([row["minutes"], it, steps_done, row["stand_score"], row["up_pct"], row["time_to_up"], row["assist"], row["reward"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * elapsed / (minutes * 60)), "stand-up score %.0f%% · up %.0f%% · help %.0f%%" % (sc, row["up_pct"], row["assist"]))
    finally:
        curve.close()
    if best < 0 or score() >= best:
        save("final")
    final = max(0.0, best, score())
    name = "humanoid_%sstandup_%s_%02d.pt" % ("" if ROBOT == "g1" else ROBOT + "_",time.strftime("%Y%m%d_%H%M"), int(round(final)))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s (stand-up score %.0f%%: the share of falls it gets up from and stays up, less while it still needs help; help now %.0f%% of its weight)." % (name, final, 100 * env.assist))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ({"time": "time is up", "asked": "finished on request"}[stop_reason], it, steps_done / 1e6, (time.time() - t_start) / 60))
    return {"stand_score": round(final, 1), "help_pct": round(100 * env.assist, 1), "steps_m": round(steps_done / 1e6, 1)}


# ── walking with a learned policy, live ─────────────────────────────────────────────────────────────────────────────
def zone_at(x, y):
    """the part of the circuit at x, y: (key, kind, what it is)"""
    for z in ZONES:
        if z[1] <= x <= z[2] and z[3] <= y <= z[4]:
            return z
    return None


def zone_text(z, climb):
    kind = z[5]
    if kind in ("slope", "stairs"):
        return z[6] % ("up" if climb > 0 else "down")
    return z[6]


def walk_run(ctx):
    import os
    import re
    import time
    import math
    import mujoco
    import torch
    p = ctx.params
    torch.set_num_threads(1)
    pols, ups, others = [], [], 0
    R = ROBOTS[ROBOT]
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                h = torch.load(f.path, map_location="cpu", weights_only=False)
                if h.get("format") == FORMAT and (h.get("robot") or "g1") != ROBOT:
                    others += 1
                    continue
                if h.get("format") == FORMAT and h.get("task") == "standup":
                    ups.append((policy_stamp(f.path, f.name), f))
                    continue
                if h.get("format") != FORMAT or h.get("task") != TASK:
                    ctx.log("Skipped %s: made for another task or version." % f.name)
                    continue
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                pols.append((m_.group(1) if m_ else "", os.path.getmtime(f.path), f))
            except Exception as e:     # noqa: BLE001
                ctx.log("Skipped %s: %s" % (f.name, e))
    if others:
        ctx.log("%d policies of other robots left aside: the %s walks with its own." % (others, R["name"]))
    if not pols:
        raise RuntimeError("No walking policy for the %s yet: run Learn to walk with Robot set to %s first (it writes humanoid_%swalk_*.pt to its dataset), and wire that dataset to Policies." % (R["name"], R["name"], "" if ROBOT == "g1" else ROBOT + "_"))
    pols.sort(key=lambda x: (x[0], x[1]), reverse=True)
    want = str(p.get("policy_file") or "").strip()
    pick = [x for x in pols if want and want in str(x[2].name)]
    pol = (pick or pols)[0][2]
    act, st = load_policy(pol.path)
    sees = int(st.get("obs_dim") or OBS_BASE) > OBS_BASE
    walk_dim = int(st.get("obs_dim") or OBS_BASE)
    up_act, up_st = None, None
    if ups and str(p.get("after_fall") or "Get up on its own").lower().startswith("get up"):
        ups.sort(key=lambda x: x[0], reverse=True)
        up_act, up_st = load_policy(ups[0][1].path)
        ctx.log("Stand-up policy %s: stand-up score %.0f%%%s. After a fall it gets up on its own (put back on its feet if it is still down after 10 s)." % (
            ups[0][1].name, float(up_st.get("score") or 0), (", learned with help (%.0f%% of its weight) still on" % (100 * float(up_st.get("assist") or 0))) if float(up_st.get("assist") or 0) > 0.005 else ""))
    elif not ups:
        ctx.log("No stand-up policy among the policies: after a fall it is put back on its feet (Learn to stand up teaches it to get up on its own).")
    terrain = terrain_of(p, "circuit")
    ctx.log("Walking policy %s: walk score %.0f%% on %s, %.0f million steps of experience on %s; %s." % (
        pol.name, float(st.get("score") or 0), GROUND_NAME.get(st.get("terrain") or "flat", "?"), float(st.get("steps") or 0) / 1e6,
        st.get("device") or "?", "it sees the ground ahead (elevation map from its head camera)" if sees else "it is blind (it feels the ground only through its feet)"))
    if terrain == "circuit" and (st.get("terrain") or "flat") != "circuit":
        ctx.log("This policy has not learned on the circuit: the rough ground, the slopes and the stairs are new to it (Learn the circuit teaches them).")
    ch, ch_ev, ch_ctl = p.get("robot_channel") or "robot", p.get("events_channel") or "event", p.get("control_channel") or "control"
    mjm = load_model(mujoco, terrain)
    sim = CpuSim(mjm, 1, threads=1)
    env = WalkEnv(sim, mjm, seed=int(time.time()) % 100000, vmax=1.5, push=0.0, noise=0.0, train=False, terrain=terrain, scan=sees or up_act is not None)
    env.new_tiles(torch.zeros(1, dtype=torch.long), seed=20261112)       # (the same tiles every walk)
    ctx.telemetry(ch, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, terrain, sees)})
    circuit = terrain == "circuit"
    speed = float(p.get("speed") if p.get("speed") not in (None, "") else 0.6)
    heading = math.radians(float(p.get("heading") or 0.0))
    minutes = float(p.get("minutes") or 30)
    dirn = 1 if math.cos(heading) >= 0 else -1          # on the circuit: counterclockwise, or (heading beyond 90°) the other way
    one = torch.zeros(1, dtype=torch.long)

    def set_line(h):
        q = sim.qpos[0]
        env.line_yaw[:] = h
        env.line_y[:] = -float(q[0]) * math.sin(h) + float(q[1]) * math.cos(h)

    def start(at=None, i=None):
        """on its feet: at the start line, or (at) where it fell, on its line"""
        if circuit:
            i = 0 if i is None else i
            px, py = float(env.P[i, 0]), float(env.P[i, 1])
            env.reset(one, at=[px, py, float(env.PH[i]) + (math.pi if dirn < 0 else 0.0)])
            env.put_on_path(one, i, dirn)
        else:
            env.reset(one, at=at)
            if at is None:
                set_line(heading)
            else:
                env.line_yaw[:] = heading
        env.mode_line[:] = True
        env.speed[:] = 0.0
    start()
    t_wall0 = time.time()
    sim_t, run, frames, last_send, fell_at = 0.0, 1, [], time.time(), None
    getting_up, up_n = False, 0

    def to_ground():
        """getting up: no command and no gait clock (as it learned it)"""
        env.mode_line[:] = False
        env.speed[:] = 0.0
        env.cmd[:] = 0.0
        env.phase_dt[:] = 0.0
        env.phase[:] = torch.tensor([0.0, math.pi])

    def back_to_walking():
        """up: walking again, from where it got up (on its line, or on the circuit from the point it reached)"""
        env.phase_dt[:] = 2 * math.pi * CTRL_DT * 1.4
        env.phase[:] = torch.tensor([0.0, math.pi])
        if circuit:
            env.put_on_path(one, int(env.pidx[0]), dirn)        # (its point of the circuit, followed while it got up)
        else:
            ly, lyaw = float(env.line_y[0]), float(env.line_yaw[0])
            env.line_yaw[:] = lyaw
            env.line_y[:] = ly
        env.mode_line[:] = True
    zone, lap, lap_t0, falls_lap, s_last = None, 1, 0.0, 0, 0
    hold = 1.5        # s standing still before it sets off
    ctx.telemetry(ch_ev, {"text": ("The %s is on the circuit (%s): it sets off at %.1f m/s" % (R["short"], "counterclockwise" if dirn > 0 else "clockwise", speed)) if circuit
                          else "The %s is on the floor: it sets off at %.1f m/s" % (R["short"], speed)})
    o, _ = env.obs(noisy=False)
    while time.time() - t_wall0 < minutes * 60:
        # the controls
        for msg in ctx.messages(ch_ctl):
            d = msg.get("data") if isinstance(msg, dict) else None
            if not isinstance(d, dict):
                continue
            c, v = str(d.get("control", "")).strip().lower(), d.get("value")
            if c == "speed":
                try:
                    speed = max(0.0, min(1.5, float(v)))
                except (TypeError, ValueError):
                    pass
            elif c in ("heading", "turn", "direction"):
                try:
                    heading = math.radians(float(v))
                except (TypeError, ValueError):
                    continue
                if circuit:
                    nd = 1 if math.cos(heading) >= 0 else -1
                    if nd != dirn:
                        dirn = nd
                        env.pdir[:] = float(dirn)
                        ctx.telemetry(ch_ev, {"text": "Turning round: now %s" % ("counterclockwise" if dirn > 0 else "clockwise")})
                else:
                    set_line(heading)
                    ctx.telemetry(ch_ev, {"text": "New heading: %.0f°" % math.degrees(heading)})
            elif c == "push":
                ang = env.U(1, 0.0, 2 * math.pi)
                sim.kick(torch.stack([torch.cos(ang), torch.sin(ang)], -1) * 0.8)
                wh = float(env.path_heading()[0])
                ctx.telemetry(ch_ev, {"text": "A push (0.8 m/s, from the %s)" % ("left" if math.sin(float(ang) - wh) > 0 else "right")})
            elif c in ("reset", "stand up", "restart", "back to the start"):
                getting_up, fell_at = False, None
                start()
                sim_t += 0.5
                run += 1
                hold = 1.5
                zone, lap, lap_t0, falls_lap, s_last = None, 1, sim_t, 0, 0
                ctx.telemetry(ch_ev, {"text": "Back at the start line"})
        t_due = t_wall0 + sim_t
        now = time.time()
        if now < t_due:
            time.sleep(min(0.01, t_due - now))
            continue
        # one control step: 20 ms of the robot's time
        if getting_up:
            to_ground()
            a = up_act(o)
        else:
            env.speed[:] = 0.0 if hold > 0 else speed
            hold -= CTRL_DT
            a = act(o[:, :walk_dim])
        rew, done, info = env.step(a)
        sim_t += CTRL_DT
        o, _ = env.obs(noisy=False)
        if int(env.steps[0]) % 2 == 0:
            frames.append(frame_row(env, 0, sim_t))
        if bool(info["fell"][0]) and fell_at is None:
            fell_at = sim_t
            falls_lap += 1
            where = (" on " + zone_text(zone, 0).split(":")[0].split(" (")[0].lower().replace(" down", "").replace(" up", "")) if zone else ""
            if up_act is not None:
                getting_up, up_n = True, 0
                to_ground()
                ctx.telemetry(ch_ev, {"text": "It fell%s: it gets up on its own" % where})
            else:
                ctx.telemetry(ch_ev, {"text": "It fell%s: back up in 3 s" % where})
        if getting_up:
            hq = float(env.height()[0])
            tz_ = float(env.S("torso_zaxis")[0, 2])
            up_n = up_n + 1 if (hq > hs(0.65) and tz_ > 0.85) else 0
            if up_n >= 25:           # up and steady for half a second: walking again
                ctx.telemetry(ch_ev, {"text": "Up on its own in %.1f s: walking on" % (sim_t - fell_at)})
                getting_up, fell_at = False, None
                back_to_walking()
                hold = 0.5
        if fell_at is not None and sim_t - fell_at > (10.0 if getting_up else 3.0):
            if getting_up:
                ctx.telemetry(ch_ev, {"text": "Still down after 10 s: put back on its feet"})
            getting_up = False
            back_to_walking()
            fell_at = None
            if circuit:      # back on its feet on the circuit's line, a step further on
                start(i=(int(env.pidx[0]) + 4 * dirn) % env.M)
            else:
                q = sim.qpos[0]
                ly, lyaw = float(env.line_y[0]), float(env.line_yaw[0])
                s_ = float(q[0]) * math.cos(lyaw) + float(q[1]) * math.sin(lyaw) + 0.4
                start([s_ * math.cos(lyaw) - ly * math.sin(lyaw), s_ * math.sin(lyaw) + ly * math.cos(lyaw), lyaw])
                env.line_y[:] = ly
            run += 1
            hold = 1.0
        # round the circuit: what it walks on now (said once as it gets there), a lap each time it crosses the start line
        if circuit and fell_at is None:
            q = sim.qpos[0]
            x_, y_ = float(q[0]), float(q[1])
            z_ = zone_at(x_, y_)
            if z_ is not zone:
                zone = z_
                if z_ is not None:
                    wh = float(env.path_heading()[0])
                    climb = float(env.ground(q[0:1].float() + 0.6 * math.cos(wh), q[1:2].float() + 0.6 * math.sin(wh))[0] - env.ground(q[0:1].float(), q[1:2].float())[0])
                    ctx.telemetry(ch_ev, {"text": zone_text(z_, climb)})
            s_now = int(env.pidx[0])
            if hold <= 0 and abs(s_now - s_last) > env.M // 2:        # it went over the start line
                ctx.telemetry(ch_ev, {"text": "Lap %d in %.0f s, %s" % (lap, sim_t - lap_t0, "no falls" if not falls_lap else "%d fall%s" % (falls_lap, "s" if falls_lap > 1 else ""))})
                lap, lap_t0, falls_lap = lap + 1, sim_t, 0
            s_last = s_now
        if int(env.steps[0]) >= EP_STEPS - 1:
            env.steps[:] = 0          # (walking on: the episode clock only matters in training)
        if time.time() - last_send > 0.25 and frames:
            last_send = time.time()
            item = {"format": "esasim.humanoid/1", "kind": "live", "run": run, "t0": round(t_wall0, 3), "frames": frames, "label": "Walking: %s" % pol.name}
            if env.tiles_on:
                item["tiles"] = tiles_item(env, 0)
            if circuit:
                item["path"] = dirn
            else:
                item["line"] = [round(float(env.line_y[0]), 3), round(float(env.line_yaw[0]), 4)]
            ctx.telemetry(ch, item)
            frames = []
    return {"minutes": minutes}


def run(ctx):
    import mujoco
    what = str(ctx.params.get("mode") or "Learn to walk").lower()
    use_robot(robot_id(ctx.params), mujoco)
    if what.startswith("walk"):
        return walk_run(ctx)
    if "stand" in what:
        return stand_run(ctx)
    return learn_run(ctx)
