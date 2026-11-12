# ── EsaSim block ──────────────────────────────────────────────────────
# Humanoid G1 (ws-humanoid-g1) · version 2 · machine block, Python: def run(ctx)
# A Unitree G1 humanoid (29 joints) learns to walk in MuJoCo, thousands at once on the GPU, on a flat floor or a test course (bumps, blocks, a ramp), then walks the course live in 3D.
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-12
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Policy · several · optional · Walking policies (.pt) from Learn to walk: Walk drives the robot with the newest; Learn to walk carries on from the best
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Policy · Learn to walk: humanoid_walk_[course_]<date>_<NN>.pt, the best policy learned (network, input scaling, ground, walk score)
#   curve       Table (csv)               Learning curve · Learn to walk: training.csv
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    What to do = "Learn to walk" of Learn to walk | Walk
#   accelerator select    Run on = "auto" of auto | NVIDIA GPU | CPU
#   terrain     select    Ground = "Flat floor" of Flat floor | Test course
#   robots      number    Robots learning at once = 4096
#   minutes     number    Minutes = 90
#   max_speed   number    Top speed asked (m/s) = 1
#   push        number    Pushes up to (m/s) = 1.5
#   replay_every number    Replay every (s) = 20
#   seed        number    Seed = 1
#   from_scratch bool      Start from scratch = false
#   speed       number    Walking speed (m/s) = 0.6
#   heading     number    Heading (degrees) = 0
#   laps        bool      Laps = true
#   policy_file text      Policy file = ""
# Runs on  Any machine, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch, mujoco>=3.3, mujoco-warp, warp-lang (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Humanoid training: a Unitree G1 humanoid (29 joints, 35 kg) learns to walk in MuJoCo, the physics engine robotics
research uses, on a flat floor or on a test course (bumps, a ramp up to a plateau and down, blocks). Many robots learn at
once on the NVIDIA GPU (MuJoCo Warp), or on the CPU. The robot's model: its 30 links with their masses and inertias, its
joints with their limits, the motors with their torque limits and the gains of their position loops (from MuJoCo
Menagerie and MuJoCo Playground, Apache 2.0 / BSD); the feet touch the ground as boxes with friction; the legs, the
hands, the pelvis, the torso and the head touch it too when it falls.

What to do: Learn to walk: the policy (a neural network that, 50 times a second, turns what the robot senses into
targets for its 29 joints) learns by trial and error (PPO) to walk at the speed asked, straight along a line. Walk: the
newest policy drives the robot live in the 3D view (esasim.humanoid/1): set the speed and the heading, push it, reset it;
on the course it walks lap after lap; its frames carry its IMU, foot loads and motor torques for the telemetry window."""
import math


# the robot: Unitree G1, 29 joints (MuJoCo Menagerie, BSD-3; collision set of MuJoCo Playground, Apache 2.0), meshes left out
MODEL_XML = r'''<mujoco model="g1_29dof"><compiler angle="radian" /><option iterations="3" ls_iterations="5" timestep=".002" integrator="Euler"><flag eulerdamp="disable" /></option><default><default class="g1"><geom condim="1" contype="0" conaffinity="0" /><joint frictionloss="0.1" damping="2" /><position inheritrange="1" kp="75" /><default class="hip"><default class="hip_pitch"><joint axis="0 1 0" range="-2.5307 2.8798" actuatorfrcrange="-88 88" armature="0.01017752004" /></default><default class="hip_roll"><joint axis="1 0 0" range="-0.5236 2.9671" actuatorfrcrange="-139 139" armature="0.025101925" /></default><default class="hip_yaw"><joint axis="0 0 1" range="-2.7576 2.7576" actuatorfrcrange="-88 88" armature="0.01017752004" /></default></default><default class="knee"><joint axis="0 1 0" range="-0.087267 2.8798" actuatorfrcrange="-139 139" armature="0.025101925" /></default><default class="ankle"><default class="ankle_pitch"><position kp="20" /><joint axis="0 1 0" range="-0.87267 0.5236" actuatorfrcrange="-50 50" damping="1" armature="0.00721945" /></default><default class="ankle_roll"><position kp="2" /><joint axis="1 0 0" range="-0.2618 0.2618" actuatorfrcrange="-50 50" damping=".2" armature="0.00721945" /></default></default><default class="waist_yaw"><joint axis="0 0 1" range="-2.618 2.618" actuatorfrcrange="-88 88" armature="0.01017752004" /></default><default class="waist_pitch"><joint axis="0 1 0" range="-0.52 0.52" actuatorfrcrange="-50 50" armature="0.00721945" /></default><default class="waist_roll"><joint axis="1 0 0" range="-0.52 0.52" actuatorfrcrange="-50 50" armature="0.00721945" /></default><default class="shoulder"><default class="shoulder_pitch"><joint axis="0 1 0" range="-3.0892 2.6704" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="shoulder_roll"><joint axis="1 0 0" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="shoulder_yaw"><joint axis="0 0 1" range="-2.618 2.618" actuatorfrcrange="-25 25" armature="0.003609725" /></default></default><default class="elbow"><joint axis="0 1 0" range="-1.0472 2.0944" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="wrist"><position kp="2" /><joint damping=".2" /><default class="wrist_roll"><joint axis="1 0 0" range="-1.97222 1.97222" actuatorfrcrange="-25 25" armature="0.003609725" /></default><default class="wrist_pitch"><joint axis="0 1 0" range="-1.61443 1.61443" actuatorfrcrange="-5 5" armature="0.00425" /></default><default class="wrist_yaw"><joint axis="0 0 1" range="-1.61443 1.61443" actuatorfrcrange="-5 5" armature="0.00425" /></default></default><default class="visual"><geom group="2" type="mesh" density="0" material="silver" /></default><default class="collision"><geom group="3" rgba=".2 .6 .2 .3" /><default class="foot"><geom size="0.085 0.03 0.005" /></default></default><site group="5" rgba="1 0 0 1" /></default></default><asset><hfield name="terrain" nrow="281" ncol="1041" size="26 7 1 0.1" /><material name="silver" rgba="0.7 0.7 0.7 1" /><material name="black" rgba="0.2 0.2 0.2 1" /></asset><worldbody><geom name="floor" size="0 0 0.01" type="plane" pos="0 0 -0.001" /><geom name="terrain" type="hfield" hfield="terrain" pos="20 0 -0.05" /><light name="spotlight" mode="targetbodycom" target="pelvis" pos="0 -2 4" exponent="20" /><body name="pelvis" pos="0 0 0.793" childclass="g1"><geom name="pelvis_col" class="collision" type="capsule" size="0.07" fromto="0 -0.07 -0.06 0 0.07 -0.06" /><camera name="track" pos="1.734 -1.135 .35" xyaxes="0.552 0.834 -0.000 -0.170 0.112 0.979" mode="trackcom" /><inertial pos="0 0 -0.07605" quat="1 0 -0.000399148 0" mass="3.813" diaginertia="0.010549 0.0093089 0.0079184" /><freejoint name="floating_base_joint" /><site name="imu_in_pelvis" size="0.01" pos="0.04525 0 -0.08339" /><body name="left_hip_pitch_link" pos="0 0.064452 -0.1027"><inertial pos="0.002741 0.047791 -0.02606" quat="0.954862 0.293964 0.0302556 0.030122" mass="1.35" diaginertia="0.00181517 0.00153422 0.00116212" /><joint name="left_hip_pitch_joint" class="hip_pitch" /><body name="left_hip_roll_link" pos="0 0.052 -0.030465" quat="0.996179 0 -0.0873386 0"><inertial pos="0.029812 -0.001045 -0.087934" quat="0.977808 -1.97119e-05 0.205576 -0.0403793" mass="1.52" diaginertia="0.00254986 0.00241169 0.00148755" /><joint name="left_hip_roll_joint" class="hip_roll" /><geom name="left_thigh" class="collision" type="capsule" size="0.06" fromto="0.02 0 0 0.02 0 -0.2" /><body name="left_hip_yaw_link" pos="0.025001 0 -0.12412"><inertial pos="-0.057709 -0.010981 -0.15078" quat="0.600598 0.15832 0.223482 0.751181" mass="1.702" diaginertia="0.00776166 0.00717575 0.00160139" /><joint name="left_hip_yaw_joint" class="hip_yaw" /><body name="left_knee_link" pos="-0.078273 0.0021489 -0.17734" quat="0.996179 0 0.0873386 0"><inertial pos="0.005457 0.003964 -0.12074" quat="0.923418 -0.0327699 0.0158246 0.382067" mass="1.932" diaginertia="0.0113804 0.0112778 0.00146458" /><joint name="left_knee_joint" class="knee" /><geom name="left_shin" class="collision" type="capsule" size="0.08" fromto="0.02 0 0 0.02 0 -0.25" /><body name="left_ankle_pitch_link" pos="0 -9.4445e-05 -0.30001"><inertial pos="-0.007269 0 0.011137" quat="0.603053 0.369225 0.369225 0.603053" mass="0.074" diaginertia="1.89e-05 1.40805e-05 6.9195e-06" /><joint name="left_ankle_pitch_joint" class="ankle_pitch" /><body name="left_ankle_roll_link" pos="0 0 -0.017558"><site name="left_foot" rgba="1 0 0 1" /><inertial pos="0.026505 0 -0.016425" quat="-0.000481092 0.728482 -0.000618967 0.685065" mass="0.608" diaginertia="0.00167218 0.0016161 0.000217621" /><joint name="left_ankle_roll_joint" class="ankle_roll" /><geom name="left_foot" class="foot" pos="0.04 0 -0.029" size="0.09 0.03 0.008" type="box" /></body></body></body></body></body></body><body name="right_hip_pitch_link" pos="0 -0.064452 -0.1027"><inertial pos="0.002741 -0.047791 -0.02606" quat="0.954862 -0.293964 0.0302556 -0.030122" mass="1.35" diaginertia="0.00181517 0.00153422 0.00116212" /><joint name="right_hip_pitch_joint" class="hip_pitch" /><body name="right_hip_roll_link" pos="0 -0.052 -0.030465" quat="0.996179 0 -0.0873386 0"><inertial pos="0.029812 0.001045 -0.087934" quat="0.977808 1.97119e-05 0.205576 0.0403793" mass="1.52" diaginertia="0.00254986 0.00241169 0.00148755" /><joint name="right_hip_roll_joint" class="hip_roll" /><geom name="right_thigh" class="collision" type="capsule" size="0.06" fromto="0.02 0 0 0.02 0 -0.2" /><body name="right_hip_yaw_link" pos="0.025001 0 -0.12412"><inertial pos="-0.057709 0.010981 -0.15078" quat="0.751181 0.223482 0.15832 0.600598" mass="1.702" diaginertia="0.00776166 0.00717575 0.00160139" /><joint name="right_hip_yaw_joint" class="hip_yaw" /><body name="right_knee_link" pos="-0.078273 -0.0021489 -0.17734" quat="0.996179 0 0.0873386 0"><inertial pos="0.005457 -0.003964 -0.12074" quat="0.923439 0.0345276 0.0116333 -0.382012" mass="1.932" diaginertia="0.011374 0.0112843 0.00146452" /><joint name="right_knee_joint" class="knee" /><geom name="right_shin" class="collision" type="capsule" size="0.08" fromto="0.02 0 0 0.02 0 -0.25" /><body name="right_ankle_pitch_link" pos="0 9.4445e-05 -0.30001"><inertial pos="-0.007269 0 0.011137" quat="0.603053 0.369225 0.369225 0.603053" mass="0.074" diaginertia="1.89e-05 1.40805e-05 6.9195e-06" /><joint name="right_ankle_pitch_joint" class="ankle_pitch" /><body name="right_ankle_roll_link" pos="0 0 -0.017558"><site name="right_foot" pos="0 0 0" /><inertial pos="0.026505 0 -0.016425" quat="0.000481092 0.728482 0.000618967 0.685065" mass="0.608" diaginertia="0.00167218 0.0016161 0.000217621" /><joint name="right_ankle_roll_joint" class="ankle_roll" /><geom name="right_foot" class="foot" pos="0.04 0 -0.029" size="0.09 0.03 0.008" type="box" /></body></body></body></body></body></body><body name="waist_yaw_link"><inertial pos="0.003494 0.000233 0.018034" quat="0.289697 0.591001 -0.337795 0.672821" mass="0.214" diaginertia="0.000163531 0.000107714 0.000102205" /><joint name="waist_yaw_joint" class="waist_yaw" /><body name="waist_roll_link" pos="-0.0039635 0 0.044"><inertial pos="0 2.3e-05 0" quat="0.5 0.5 -0.5 0.5" mass="0.086" diaginertia="8.245e-06 7.079e-06 6.339e-06" /><joint name="waist_roll_joint" class="waist_roll" /><body name="torso_link"><geom name="torso_col" class="collision" type="capsule" size="0.1" fromto="0 0 0.08 0 0 0.34" /><geom name="head_col" class="collision" type="sphere" size="0.09" pos="0.01 0 0.45" /><inertial pos="0.00203158 0.000339683 0.184568" quat="0.999803 -6.03319e-05 0.0198256 0.00131986" mass="7.818" diaginertia="0.121847 0.109825 0.0273735" /><joint name="waist_pitch_joint" class="waist_pitch" /><site name="imu_in_torso" size="0.01" pos="-0.03959 -0.00224 0.14792" /><body name="left_shoulder_pitch_link" pos="0.0039563 0.10022 0.24778" quat="0.990264 0.139201 1.38722e-05 -9.86868e-05"><inertial pos="0 0.035892 -0.011628" quat="0.654152 0.0130458 -0.326267 0.68225" mass="0.718" diaginertia="0.000465864 0.000432842 0.000406394" /><joint name="left_shoulder_pitch_joint" class="shoulder_pitch" /><geom size="0.03 0.025" pos="0 0.04 -0.01" quat="0.707107 0 0.707107 0" type="cylinder" class="visual" /><body name="left_shoulder_roll_link" pos="0 0.038 -0.013831" quat="0.990268 -0.139172 0 0"><inertial pos="-0.000227 0.00727 -0.063243" quat="0.701256 -0.0196223 -0.00710317 0.712604" mass="0.643" diaginertia="0.000691311 0.000618011 0.000388977" /><joint name="left_shoulder_roll_joint" range="-1.5882 2.2515" class="shoulder_roll" /><geom size="0.03 0.015" pos="-0.004 0.006 -0.053" type="cylinder" class="visual" /><body name="left_shoulder_yaw_link" pos="0 0.00624 -0.1032"><inertial pos="0.010773 -0.002949 -0.072009" quat="0.716879 -0.0964829 -0.0679942 0.687134" mass="0.734" diaginertia="0.00106187 0.00103217 0.000400661" /><joint name="left_shoulder_yaw_joint" class="shoulder_yaw" /><body name="left_elbow_link" pos="0.015783 0 -0.080518"><inertial pos="0.064956 0.004454 -0.010062" quat="0.541765 0.636132 0.388821 0.388129" mass="0.6" diaginertia="0.000443035 0.000421612 0.000259353" /><joint name="left_elbow_joint" class="elbow" /><body name="left_wrist_roll_link" pos="0.1 0.00188791 -0.01"><inertial pos="0.0171394 0.000537591 4.8864e-07" quat="0.575338 0.411667 -0.574906 0.411094" mass="0.085445" diaginertia="5.48211e-05 4.96646e-05 3.57798e-05" /><joint name="left_wrist_roll_joint" class="wrist_roll" /><body name="left_wrist_pitch_link" pos="0.038 0 0"><inertial pos="0.0229999 -0.00111685 -0.00111658" quat="0.249998 0.661363 0.293036 0.643608" mass="0.48405" diaginertia="0.000430353 0.000429873 0.000164648" /><joint name="left_wrist_pitch_joint" class="wrist_pitch" /><body name="left_wrist_yaw_link" pos="0.046 0 0"><inertial pos="0.0708244 0.000191745 0.00161742" quat="0.510571 0.526295 0.468078 0.493188" mass="0.254576" diaginertia="0.000646113 0.000559993 0.000147566" /><joint name="left_wrist_yaw_joint" class="wrist_yaw" /><site name="left_palm" pos="0.08 0 0" size="0.01" /><geom name="left_hand_collision" class="collision" type="capsule" size="0.05" fromto="0.05 0 0 0.1 0 0" rgba=".2 .6 .2 .2" /></body></body></body></body></body></body></body><body name="right_shoulder_pitch_link" pos="0.0039563 -0.10021 0.24778" quat="0.990264 -0.139201 1.38722e-05 9.86868e-05"><inertial pos="0 -0.035892 -0.011628" quat="0.68225 -0.326267 0.0130458 0.654152" mass="0.718" diaginertia="0.000465864 0.000432842 0.000406394" /><joint name="right_shoulder_pitch_joint" class="shoulder_pitch" /><geom size="0.03 0.025" pos="0 -0.04 -0.01" quat="0.707107 0 0.707107 0" type="cylinder" class="visual" /><body name="right_shoulder_roll_link" pos="0 -0.038 -0.013831" quat="0.990268 0.139172 0 0"><inertial pos="-0.000227 -0.00727 -0.063243" quat="0.712604 -0.00710317 -0.0196223 0.701256" mass="0.643" diaginertia="0.000691311 0.000618011 0.000388977" /><joint name="right_shoulder_roll_joint" range="-2.2515 1.5882" class="shoulder_roll" /><geom size="0.03 0.015" pos="-0.004 -0.006 -0.053" type="cylinder" class="visual" /><body name="right_shoulder_yaw_link" pos="0 -0.00624 -0.1032"><inertial pos="0.010773 0.002949 -0.072009" quat="0.687134 -0.0679942 -0.0964829 0.716879" mass="0.734" diaginertia="0.00106187 0.00103217 0.000400661" /><joint name="right_shoulder_yaw_joint" class="shoulder_yaw" /><body name="right_elbow_link" pos="0.015783 0 -0.080518"><inertial pos="0.064956 -0.004454 -0.010062" quat="0.388129 0.388821 0.636132 0.541765" mass="0.6" diaginertia="0.000443035 0.000421612 0.000259353" /><joint name="right_elbow_joint" class="elbow" /><body name="right_wrist_roll_link" pos="0.1 -0.00188791 -0.01"><inertial pos="0.0171394 -0.000537591 4.8864e-07" quat="0.411667 0.575338 -0.411094 0.574906" mass="0.085445" diaginertia="5.48211e-05 4.96646e-05 3.57798e-05" /><joint name="right_wrist_roll_joint" class="wrist_roll" /><body name="right_wrist_pitch_link" pos="0.038 0 0"><inertial pos="0.0229999 0.00111685 -0.00111658" quat="0.643608 0.293036 0.661363 0.249998" mass="0.48405" diaginertia="0.000430353 0.000429873 0.000164648" /><joint name="right_wrist_pitch_joint" class="wrist_pitch" /><body name="right_wrist_yaw_link" pos="0.046 0 0"><inertial pos="0.0708244 -0.000191745 0.00161742" quat="0.493188 0.468078 0.526295 0.510571" mass="0.254576" diaginertia="0.000646113 0.000559993 0.000147566" /><joint name="right_wrist_yaw_joint" class="wrist_yaw" /><site name="right_palm" pos="0.08 0 0" size="0.01" /><geom name="right_hand_collision" class="collision" type="capsule" size="0.05" fromto="0.05 0 0 0.1 0 0" rgba=".2 .6 .2 .2" /></body></body></body></body></body></body></body></body></body></body></body></worldbody><contact><pair name="left_foot_floor" geom1="left_foot" geom2="floor" condim="3" friction="0.6 0.6" /><pair name="right_foot_floor" geom1="right_foot" geom2="floor" condim="3" friction="0.6 0.6" /><pair name="left_foot_terrain" geom1="left_foot" geom2="terrain" condim="3" friction="0.8 0.8" /><pair name="right_foot_terrain" geom1="right_foot" geom2="terrain" condim="3" friction="0.8 0.8" /><pair name="left_hand_thigh" geom1="left_hand_collision" geom2="left_thigh" condim="1" /><pair name="right_hand_thigh" geom1="right_hand_collision" geom2="right_thigh" condim="1" /><pair name="left_foot_right_foot" geom1="left_foot" geom2="right_foot" condim="1" /><pair geom1="floor" geom2="left_shin" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="right_shin" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="left_thigh" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="right_thigh" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="left_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="right_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="pelvis_col" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="torso_col" condim="3" friction="0.6 0.6" /><pair geom1="floor" geom2="head_col" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="left_shin" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="right_shin" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="left_thigh" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="right_thigh" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="left_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="right_hand_collision" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="pelvis_col" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="torso_col" condim="3" friction="0.6 0.6" /><pair geom1="terrain" geom2="head_col" condim="3" friction="0.6 0.6" /></contact><actuator><position class="hip_pitch" name="left_hip_pitch_joint" joint="left_hip_pitch_joint" /><position class="hip_roll" name="left_hip_roll_joint" joint="left_hip_roll_joint" /><position class="hip_yaw" name="left_hip_yaw_joint" joint="left_hip_yaw_joint" /><position class="knee" name="left_knee_joint" joint="left_knee_joint" /><position class="ankle_pitch" name="left_ankle_pitch_joint" joint="left_ankle_pitch_joint" /><position class="ankle_roll" name="left_ankle_roll_joint" joint="left_ankle_roll_joint" /><position class="hip_pitch" name="right_hip_pitch_joint" joint="right_hip_pitch_joint" /><position class="hip_roll" name="right_hip_roll_joint" joint="right_hip_roll_joint" /><position class="hip_yaw" name="right_hip_yaw_joint" joint="right_hip_yaw_joint" /><position class="knee" name="right_knee_joint" joint="right_knee_joint" /><position class="ankle_pitch" name="right_ankle_pitch_joint" joint="right_ankle_pitch_joint" /><position class="ankle_roll" name="right_ankle_roll_joint" joint="right_ankle_roll_joint" /><position class="waist_yaw" name="waist_yaw_joint" joint="waist_yaw_joint" /><position class="waist_roll" name="waist_roll_joint" joint="waist_roll_joint" /><position class="waist_pitch" name="waist_pitch_joint" joint="waist_pitch_joint" /><position class="shoulder_pitch" name="left_shoulder_pitch_joint" joint="left_shoulder_pitch_joint" /><position class="shoulder_roll" name="left_shoulder_roll_joint" joint="left_shoulder_roll_joint" /><position class="shoulder_yaw" name="left_shoulder_yaw_joint" joint="left_shoulder_yaw_joint" /><position class="elbow" name="left_elbow_joint" joint="left_elbow_joint" /><position class="wrist_roll" name="left_wrist_roll_joint" joint="left_wrist_roll_joint" /><position class="wrist_pitch" name="left_wrist_pitch_joint" joint="left_wrist_pitch_joint" /><position class="wrist_yaw" name="left_wrist_yaw_joint" joint="left_wrist_yaw_joint" /><position class="shoulder_pitch" name="right_shoulder_pitch_joint" joint="right_shoulder_pitch_joint" /><position class="shoulder_roll" name="right_shoulder_roll_joint" joint="right_shoulder_roll_joint" /><position class="shoulder_yaw" name="right_shoulder_yaw_joint" joint="right_shoulder_yaw_joint" /><position class="elbow" name="right_elbow_joint" joint="right_elbow_joint" /><position class="wrist_roll" name="right_wrist_roll_joint" joint="right_wrist_roll_joint" /><position class="wrist_pitch" name="right_wrist_pitch_joint" joint="right_wrist_pitch_joint" /><position class="wrist_yaw" name="right_wrist_yaw_joint" joint="right_wrist_yaw_joint" /></actuator><sensor><framezaxis objtype="site" objname="imu_in_torso" name="upvector_torso" /><velocimeter site="imu_in_torso" name="local_linvel_torso" /><velocimeter site="imu_in_pelvis" name="local_linvel_pelvis" /><accelerometer site="imu_in_torso" name="accelerometer_torso" /><accelerometer site="imu_in_pelvis" name="accelerometer_pelvis" /><gyro site="imu_in_torso" name="gyro_torso" /><gyro site="imu_in_pelvis" name="gyro_pelvis" /><framezaxis objtype="site" objname="imu_in_pelvis" name="upvector_pelvis" /><framexaxis objtype="site" objname="imu_in_torso" name="forwardvector_torso" /><framexaxis objtype="site" objname="imu_in_pelvis" name="forwardvector_pelvis" /><framequat objtype="site" objname="imu_in_torso" name="orientation_torso" /><framequat objtype="site" objname="imu_in_pelvis" name="orientation_pelvis" /><framelinvel objtype="site" objname="imu_in_torso" name="global_linvel_torso" /><framelinvel objtype="site" objname="imu_in_pelvis" name="global_linvel_pelvis" /><frameangvel objtype="site" objname="imu_in_torso" name="global_angvel_torso" /><frameangvel objtype="site" objname="imu_in_pelvis" name="global_angvel_pelvis" /><framelinvel objtype="site" objname="left_foot" name="left_foot_global_linvel" /><framelinvel objtype="site" objname="right_foot" name="right_foot_global_linvel" /><framezaxis objtype="site" objname="left_foot" name="left_foot_upvector" /><framezaxis objtype="site" objname="right_foot" name="right_foot_upvector" /><force name="left_foot_force" site="left_foot" /><force name="right_foot_force" site="right_foot" /><contact name="left_foot_floor_found" geom1="left_foot" geom2="floor" reduce="mindist" num="1" data="found" /><contact name="right_foot_floor_found" geom1="right_foot" geom2="floor" reduce="mindist" num="1" data="found" /><contact name="left_foot_terrain_found" geom1="left_foot" geom2="terrain" reduce="mindist" num="1" data="found" /><contact name="right_foot_terrain_found" geom1="right_foot" geom2="terrain" reduce="mindist" num="1" data="found" /><contact name="right_foot_left_foot_found" geom1="right_foot" geom2="left_foot" reduce="mindist" num="1" data="found" /><contact name="left_foot_right_shin_found" geom1="left_foot" geom2="right_shin" reduce="mindist" num="1" data="found" /><contact name="right_foot_left_shin_found" geom1="right_foot" geom2="left_shin" reduce="mindist" num="1" data="found" /><contact name="left_hand_left_thigh_found" geom1="left_hand_collision" geom2="left_thigh" reduce="mindist" num="1" data="found" /><contact name="right_hand_right_thigh_found" geom1="right_hand_collision" geom2="right_thigh" reduce="mindist" num="1" data="found" /><framepos objtype="site" objname="left_foot" name="left_foot_pos" /><framepos objtype="site" objname="right_foot" name="right_foot_pos" /><framezaxis objtype="body" objname="torso_link" name="torso_zaxis" /></sensor><keyframe><key name="home" qpos=" 0 0 0.785 1 0 0 0 -0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0 " ctrl=" -0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0 " /><key name="knees_bent" qpos=" 0 0 0.755 1 0 0 0 -0.312 0 0 0.669 -0.363 0 -0.312 0 0 0.669 -0.363 0 0 0 0.073 0.2 0.2 0 0.6 0 0 0 0.2 -0.2 0 0.6 0 0 0 " ctrl=" -0.312 0 0 0.669 -0.363 0 -0.312 0 0 0.669 -0.363 0 0 0 0.073 0.2 0.2 0 0.6 0 0 0 0.2 -0.2 0 0.6 0 0 0 " /></keyframe></mujoco>'''


# ── the physics: MuJoCo, many robots at once ──────────────────────────────────────────────────────────────────────
# Two ways to run the same model: MuJoCo Warp on the NVIDIA GPU (thousands of robots, the whole step on the card, one
# CUDA graph per control step) or MuJoCo on the CPU (one simulation per robot, spread over the cores). Both give the
# same views as PyTorch tensors: qpos, qvel, ctrl and the sensors, on the device the learning runs on.
CTRL_DT, SIM_DT = 0.02, 0.002           # the policy at 50 Hz, the physics at 500 Hz
N_SUB = int(round(CTRL_DT / SIM_DT))


# ── the ground: a test course along the lane (x forward from the start line, y to the left, metres) ────────────────
# flat to 4 m · a 4 x 4 m square of bumps (0-4.5 cm) · flat · a slope up (12.5 %, 7°) to a 0.5 m plateau at 14-17 m · a
# slope down to 21 m · flat · a 4 x 4 m square of blocks (0.4 m tiles, 0-5 cm high) at 24-28 m · flat on. The same few
# numbers go to the 3D view, which builds the same ground from them. In MuJoCo: one heightfield, 5 cm a point.
HF_X0, HF_X1, HF_Y0, HF_Y1, HF_RES, HF_BASE = -6.0, 46.0, -7.0, 7.0, 0.05, 0.05


def make_course():
    import numpy as np
    r = np.random.default_rng(20261112)
    coarse = r.uniform(-1, 1, (11, 11))                       # bumps: a 0.4 m swell and a 0.1 m grain
    xi = np.linspace(0, 10, 41)
    up = np.array([[np.interp(x, np.arange(11), row) for x in xi] for row in coarse])
    up = np.array([np.interp(xi, np.arange(11), up[:, j]) for j in range(41)]).T
    bumps = 16 * up + r.uniform(-6, 6, (41, 41))
    bumps = np.round(bumps - bumps.min()).astype(int)            # (all above the floor: 0 to about 4.5 cm)
    blocks = r.integers(0, 51, (10, 10))
    return {"ramp": {"x": [10.0, 14.0, 17.0, 21.0], "h": 0.5, "half": 3.0, "bank": 1.5},
            "patches": [{"kind": "bumps", "x0": 4.0, "y0": -2.0, "size": 4.0, "cell": 0.1, "n": 41, "mm": bumps.ravel().tolist()},
                        {"kind": "blocks", "x0": 24.0, "y0": -2.0, "size": 4.0, "cell": 0.4, "n": 10, "mm": blocks.ravel().tolist()}],
            "length": 30.0}


COURSE = make_course()


def course_height(X, Y, course=COURSE):
    """the ground's height (m) at X, Y (numpy arrays)"""
    import numpy as np
    H = np.zeros(np.broadcast(X, Y).shape)
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
    return H


def ground_grid(terrain):
    """the heightfield's heights (rows along y, columns along x)"""
    import numpy as np
    xs = np.linspace(HF_X0, HF_X1, int(round((HF_X1 - HF_X0) / HF_RES)) + 1)
    ys = np.linspace(HF_Y0, HF_Y1, int(round((HF_Y1 - HF_Y0) / HF_RES)) + 1)
    if terrain != "course":
        return np.zeros((len(ys), len(xs)))
    X, Y = np.meshgrid(xs, ys)
    return course_height(X, Y)


def load_model(mujoco, terrain="course"):
    import numpy as np
    m = mujoco.MjModel.from_xml_string(MODEL_XML)
    m.opt.timestep = SIM_DT
    H = ground_grid(terrain)
    hid = m.hfield("terrain").id
    assert H.shape == (int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid]))
    top = float(m.hfield_size[hid][2])
    a = int(m.hfield_adr[hid])
    m.hfield_data[a:a + H.size] = np.clip((H + HF_BASE) / top, 0.0, 1.0).ravel().astype(np.float32)
    if terrain != "course":             # the flat floor only: the course's heightfield out of the way (fewer contacts)
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
        self.qpos = torch.zeros(n, mjm.nq)
        self.qvel = torch.zeros(n, mjm.nv)
        self.ctrl = torch.zeros(n, mjm.nu)
        self.sens = torch.zeros(n, mjm.nsensordata)
        self.tdev = torch.device("cpu")
        if log:
            log("MuJoCo on the CPU: %d robots over %d threads." % (n, nt))

    def _work(self, part):
        mj, m, D = self.mj, self.m, self.datas
        for i in part:
            d = D[i]
            d.ctrl[:] = self.ctrl_np[i]
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
OBS_DIM = 3 + 3 + 3 + 3 + 29 + 29 + 29 + 4
PRIV_DIM = OBS_DIM + 1 + 2 + 2 + 6 + 3 + 3
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
    def __init__(self, sim, mjm, seed=0, vmax=1.0, push=1.5, noise=1.0, train=True, terrain="course"):
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
        n = self.n
        z = lambda *s: torch.zeros(n, *s, **f)   # noqa: E731
        self.cmd = z(3)
        self.last_act, self.act = z(29), z(29)
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

    def ground(self, x, y):
        """the ground's height under x, y (bilinear between the heightfield's points; flat outside it)"""
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
        free = torch.rand(k, device=self.dev, generator=self.g) < (0.3 if self.train else 0.0)
        self.mode_line[idx] = ~free
        rnd = torch.stack([self.U(k, -0.5, self.vmax), self.U(k, -0.4, 0.4) * (torch.rand(k, device=self.dev, generator=self.g) < 0.4),
                           self.U(k, -1.0, 1.0)], -1)
        rnd = rnd * (torch.rand(k, 1, device=self.dev, generator=self.g) > 0.08)
        self.cmd[idx] = torch.where(free.unsqueeze(-1), rnd, self.cmd[idx])
        self.t_cmd[idx] = 0

    def guide(self):
        """the line follower: the turn and the sideways step that bring it back onto the line and its heading"""
        import torch
        q = self.sim.qpos
        yaw = yaw_of(q[:, 3:7].float())
        dyaw = torch.atan2(torch.sin(self.line_yaw - yaw), torch.cos(self.line_yaw - yaw))
        cy, sy = torch.cos(self.line_yaw), torch.sin(self.line_yaw)
        off = -(q[:, 0].float() * sy) + q[:, 1].float() * cy - self.line_y      # left of the line (m)
        wz = (2.0 * dyaw - 0.6 * off * torch.cos(dyaw)).clamp(-1.0, 1.0)
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
            if self.terrain == "course":       # anywhere along the course, mostly walking along it (either way)
                qpos[:, 0] = self.U(k, -1.0, COURSE["length"])
                qpos[:, 1] = self.U(k, -2.5, 2.5)
                along = torch.rand(k, device=self.dev, generator=self.g) < 0.75
                back = (torch.rand(k, device=self.dev, generator=self.g) < 0.5).float() * math.pi
                yaw = torch.where(along, back + self.U(k, -0.3, 0.3), self.U(k, -math.pi, math.pi))
            else:
                qpos[:, 0:2] += self.U(k, -0.5, 0.5, 2)
                yaw = self.U(k, -math.pi, math.pi)
            qpos[:, 7:] = qpos[:, 7:] * self.U(k, 0.5, 1.5, 29)
            qvel[:, 0:6] = self.U(k, -0.5, 0.5, 6)
        else:
            yaw = torch.zeros(k, device=self.dev)
        if at is not None:
            qpos[:, 0], qpos[:, 1], yaw = float(at[0]), float(at[1]), torch.full((k,), float(at[2]), device=self.dev)
        # standing on the ground where it starts (the highest point under its feet, and a centimetre)
        gmax = self.ground(qpos[:, 0], qpos[:, 1])
        for dx, dy in ((0.15, 0.12), (0.15, -0.12), (-0.1, 0.12), (-0.1, -0.12)):
            gmax = torch.maximum(gmax, self.ground(qpos[:, 0] + dx * torch.cos(yaw) - dy * torch.sin(yaw), qpos[:, 1] + dx * torch.sin(yaw) + dy * torch.cos(yaw)))
        qpos[:, 2] = qpos[:, 2] + gmax + 0.01
        qpos[:, 3] = torch.cos(yaw / 2)
        qpos[:, 6] = torch.sin(yaw / 2)
        qpos[:, 7:] = torch.maximum(torch.minimum(qpos[:, 7:], self.soft_hi), self.soft_lo)
        self.sim.set_rows(idx, qpos, qvel, qpos[:, 7:])
        # the line: through where it starts, along a heading near its own
        self.line_yaw[idx] = yaw + (self.U(k, -0.4, 0.4) if self.train else 0.0)    # (Walk sets its own line after)
        self.line_y[idx] = -(qpos[:, 0] * torch.sin(self.line_yaw[idx])) + qpos[:, 1] * torch.cos(self.line_yaw[idx]) + (self.U(k, -0.5, 0.5) if self.train else 0.0)
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
        contact = self.contacts().float()
        feet_v = torch.cat([self.S("left_foot_global_linvel"), self.S("right_foot_global_linvel")], -1)
        priv = torch.cat([torch.cat([lin, gyro, grav, self.cmd, jp, jv, self.last_act, ph], -1), self.height().unsqueeze(-1), contact, self.air,
                          feet_v, self.S("global_angvel_pelvis"), self.S("global_linvel_pelvis")], -1)
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
        fell = (self.height() < 0.45) | (tz[:, 2] < 0.5) | (self.S("right_foot_left_foot_found")[:, 0] > 0) | torch.isnan(q).any(-1) | torch.isnan(v).any(-1)
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
        r["orientation"] = ((tzh - torch.tensor([0.073, 0.0, 1.0], device=self.dev)) ** 2).sum(-1)
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
        r["contact_force"] = (fl - 500.0).clamp(min=0) + (fr - 500.0).clamp(min=0)
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
        self.ep_dist += (v[:, 0] * torch.cos(self.line_yaw) + v[:, 1] * torch.sin(self.line_yaw)) * CTRL_DT
        timeout = self.steps >= EP_STEPS
        if self.train:
            new_cmd = (self.t_cmd >= 500).nonzero(as_tuple=False).flatten()
            self.sample_cmd(new_cmd)
        self.fell = fell
        info = {"fell": fell, "timeout": timeout & ~fell, "terms": r, "lin": lin, "gyro": gyro, "contact": contact, "feet_z": feet_z,
                "forces": torch.stack([fl, fr], -1)}
        return rew, fell | timeout, info


# ── learning: PPO with an actor (the policy: what the robot senses → joint targets) and a critic (how good a moment is,
# from more than the robot can sense: the privileged state of the simulation) ─────────────────────────────────────────
import math


class Norm:
    """running mean and variance of what goes into a network"""

    def __init__(self, torch, n, dev):
        self.t = torch
        self.mean = torch.zeros(n, device=dev)
        self.var = torch.ones(n, device=dev)
        self.count = 1e-4

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
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load(self, st, dev):
        self.mean, self.var, self.count = st["mean"].to(dev), st["var"].to(dev), st["count"]


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
        return {"ac": self.ac.state_dict(), "obs_norm": self.on.state(), "priv_norm": self.pn.state(), "lr": self.lr}

    def load(self, st):
        self.ac.load_state_dict(st["ac"])
        self.on.load(st["obs_norm"], self.dev)
        self.pn.load(st["priv_norm"], self.dev)
        self.lr = float(st.get("lr") or self.lr)


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
FIELDS = (["t", "x", "y", "z", "qw", "qx", "qy", "qz"] + ["j%d" % k for k in range(29)]
          + ["cvx", "cvy", "cwz", "vx", "vy", "wz", "cl", "cr", "fl", "fr"]
          + ["h", "gx", "gy", "gz", "ax", "ay", "az"] + ["tq%d" % k for k in range(29)])
# h: the ground under the pelvis (m); gx..gz: the pelvis' gyro (rad/s); ax..az: its accelerometer (m/s²); tq: each motor's torque (N m)


GROUND_NAME = {"flat": "the flat floor", "course": "the test course"}
ZONES = [(4.0, 8.0, "Rough ground: bumps up to 4.5 cm"), (10.0, 14.0, "Slope up: 12.5 % (7°)"), (14.0, 17.0, "On the plateau, 0.5 m up"),
         (17.0, 21.0, "Slope down"), (24.0, 28.0, "Rough ground: blocks up to 5 cm")]


def terrain_of(p, default):
    t = str(p.get("terrain") or default).lower()
    return "course" if ("course" in t or "rough" in t) else "flat"


def robot_info(m, terrain="course"):
    """the body tree for the view: each body's parent, its place and turn on its parent, its joint (axis, slot in qpos)"""
    bodies = []
    for b in range(1, m.nbody):
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
    return {"model": "unitree_g1", "glb": "g1", "bodies": bodies, "joints": joints, "limits": limits, "torque_max": tmax,
            "default": [round(float(x), 4) for x in m.key("knees_bent").qpos[7:]], "fields": FIELDS,
            "course": COURSE if terrain == "course" else None,
            "camera": {"body": "torso_link", "pos": [0.0576, 0.0175, 0.4299], "pitch": 0.8308, "fov_h": 69.0, "aspect": 1.7778,
                       "name": "Intel RealSense D435i in the head"}}


def frame_row(env, i, t):
    q = env.sim.qpos[i].float().tolist()
    c = env.cmd[i].tolist()
    lin = env.S("local_linvel_pelvis")[i].tolist()
    gy = env.S("gyro_pelvis")[i].tolist()
    ct = env.contacts()[i].tolist()
    cl, cr = int(ct[0]), int(ct[1])
    fl = float(env.S("left_foot_force")[i, 2].abs()) / 340.0       # in body weights (34 kg)
    fr = float(env.S("right_foot_force")[i, 2].abs()) / 340.0
    r3, r4 = (lambda v: round(v, 3)), (lambda v: round(v, 4))
    return ([round(t, 3), r3(q[0]), r3(q[1]), r3(q[2])] + [r4(v) for v in q[3:7]] + [r3(v) for v in q[7:36]]
            + [round(c[0], 2), round(c[1], 2), round(c[2], 2), round(lin[0], 2), round(lin[1], 2), round(gy[2], 2), cl, cr, round(fl, 2), round(fr, 2)]
            + [r3(float(env.ground(env.sim.qpos[i:i + 1, 0].float(), env.sim.qpos[i:i + 1, 1].float())[0]))] + [round(v, 2) for v in gy]
            + [round(v, 1) for v in env.S("accelerometer_pelvis")[i].tolist()] + [round(v, 1) for v in env.sim.tq[i].float().tolist()])


def chunks(rows, budget=40000):
    """the frames in pieces small enough for a telemetry item (at most 60 KB each)"""
    import json
    per = max(1, len(json.dumps(rows[0]))) * 1.15 if rows else 1
    size = max(10, int(budget / per))
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size], i + size >= len(rows)


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
    env = WalkEnv(sim, mjm, seed=seed, vmax=vmax, push=float(p.get("push") if p.get("push") not in (None, "") else 1.5), terrain=terrain)
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, 29, horizon=24)
    found = []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                st_ = torch.load(f.path, map_location="cpu", weights_only=False)
                if st_.get("format") != FORMAT or st_.get("task") != TASK:
                    ctx.log("Skipped %s: made for another task or version." % f.name)
                    continue
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                same = 1 if (st_.get("terrain") or "flat") == terrain else 0     # one that learned this ground first, then the best
                found.append((same, float(st_.get("score") or 0), m_.group(1) if m_ else "", os.path.getmtime(f.path), f, st_))
            except Exception as e:     # noqa: BLE001
                ctx.log("Skipped %s: %s" % (f.name, e))
    steps_done, warm = 0, 0
    if found and not bool(p.get("from_scratch")):
        found.sort(key=lambda x: x[:4], reverse=True)
        f, st = found[0][4], found[0][5]
        algo.load(st)
        steps_done = int(st.get("steps") or 0)
        if (st.get("terrain") or "flat") != terrain:
            warm = 25          # new ground: the critic learns what it is worth first, the walk is left as it is
        ctx.log("Continuing from %s (walk score %.0f%% on %s, %.0f million steps of experience)." % (f.name, float(st.get("score") or 0), GROUND_NAME[st.get("terrain") or "flat"], steps_done / 1e6))
    ctx.log("Learning to walk on %s, on %s: %d robots at once, for up to %.0f minutes." % (GROUND_NAME[terrain], label, n, minutes))
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
        st.update({"format": FORMAT, "task": TASK, "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": 29, "score": score(),
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
                                                       "result": res, "label": lab, "line": [round(float(env.line_y[0]), 3), round(float(env.line_yaw[0]), 4)]})
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
    name = "humanoid_walk_%s%s_%02d.pt" % ("course_" if terrain == "course" else "", time.strftime("%Y%m%d_%H%M"), int(round(max(0.0, best, score()))))
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Policy written: %s (walk score %.0f%%: how well it keeps the asked speed, times the share of runs it does not fall)." % (name, max(0.0, best, score())))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ({"time": "time is up", "asked": "finished on request"}[stop_reason], it, steps_done / 1e6, (time.time() - t_start) / 60))
    return {"walk_score": round(max(0.0, best, score()), 1), "steps_m": round(steps_done / 1e6, 1)}


# ── walking with a learned policy, live ─────────────────────────────────────────────────────────────────────────────
def walk_run(ctx):
    import os
    import re
    import time
    import math
    import mujoco
    import torch
    p = ctx.params
    torch.set_num_threads(1)
    pols = []
    for f in (ctx.inputs.get("policy") or []):
        if str(f.name).endswith(".pt"):
            try:
                h = torch.load(f.path, map_location="cpu", weights_only=False)
                if h.get("format") != FORMAT or h.get("task") != TASK:
                    ctx.log("Skipped %s: made for another task or version." % f.name)
                    continue
                m_ = re.search(r"(\d{8}_\d{4})", str(f.name))
                pols.append((m_.group(1) if m_ else "", os.path.getmtime(f.path), f))
            except Exception as e:     # noqa: BLE001
                ctx.log("Skipped %s: %s" % (f.name, e))
    if not pols:
        raise RuntimeError("No walking policy yet: run the flow that learns to walk first (it writes humanoid_walk_*.pt to its dataset), and wire that dataset to Policies.")
    pols.sort(key=lambda x: (x[0], x[1]), reverse=True)
    want = str(p.get("policy_file") or "").strip()
    pick = [x for x in pols if want and want in str(x[2].name)]
    pol = (pick or pols)[0][2]
    act, st = load_policy(pol.path)
    terrain = terrain_of(p, "course")
    ctx.log("Walking policy %s: walk score %.0f%% on %s, %.0f million steps of experience on %s." % (pol.name, float(st.get("score") or 0), GROUND_NAME[st.get("terrain") or "flat"], float(st.get("steps") or 0) / 1e6, st.get("device") or "?"))
    if terrain == "course" and (st.get("terrain") or "flat") != "course":
        ctx.log("This policy learned on the flat floor only: the rough squares and the slopes are new to it (Learn the course teaches them).")
    ch, ch_ev, ch_ctl = p.get("robot_channel") or "robot", p.get("events_channel") or "event", p.get("control_channel") or "control"
    mjm = load_model(mujoco, terrain)
    sim = CpuSim(mjm, 1, threads=1)
    env = WalkEnv(sim, mjm, seed=int(time.time()) % 100000, vmax=1.5, push=0.0, noise=0.0, train=False, terrain=terrain)
    ctx.telemetry(ch, {"format": "esasim.humanoid/1", "kind": "robot", "robot": robot_info(mjm, terrain)})
    laps = terrain == "course" and str(p.get("laps", True)).lower() not in ("false", "0", "no", "")
    zone, lap, lap_t0, falls_lap = None, 1, 0.0, 0
    speed = float(p.get("speed") if p.get("speed") not in (None, "") else 0.6)
    heading = math.radians(float(p.get("heading") or 0.0))
    minutes = float(p.get("minutes") or 30)

    def set_line(h):
        q = sim.qpos[0]
        env.line_yaw[:] = h
        env.line_y[:] = -float(q[0]) * math.sin(h) + float(q[1]) * math.cos(h)

    def start(at=None):
        """back on its feet: at the start line, or (at) where it fell, on its line"""
        env.reset(torch.zeros(1, dtype=torch.long), at=at)
        env.mode_line[:] = True
        if at is None:
            set_line(heading)
        else:
            env.line_yaw[:] = heading          # (the same line as before)
        env.speed[:] = 0.0
        return 0
    start()
    t_wall0 = time.time()
    sim_t, run, frames, last_send, fell_at, pushes, walking = 0.0, 1, [], time.time(), None, 0, False
    hold = 1.5        # s standing still before it sets off
    ctx.telemetry(ch_ev, {"text": "The G1 is on the track: it starts walking at %.1f m/s" % speed})
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
            elif c in ("heading", "turn"):
                try:
                    heading = math.radians(float(v))
                except (TypeError, ValueError):
                    continue
                set_line(heading)
                ctx.telemetry(ch_ev, {"text": "New heading: %.0f°" % math.degrees(heading)})
            elif c == "push":
                ang = env.U(1, 0.0, 2 * math.pi)
                sim.kick(torch.stack([torch.cos(ang), torch.sin(ang)], -1) * 0.8)
                pushes += 1
                ctx.telemetry(ch_ev, {"text": "A push (0.8 m/s, from the %s)" % ("left" if math.sin(float(ang) - heading) > 0 else "right")})
            elif c in ("reset", "stand up", "restart", "back to the start"):
                heading = 0.0 if laps else heading
                start()
                sim_t += 0.5
                run += 1
                hold = 1.5
                zone, lap, lap_t0, falls_lap = None, 1, sim_t, 0
                ctx.telemetry(ch_ev, {"text": "Back at the start line"})
        t_due = t_wall0 + sim_t
        now = time.time()
        if now < t_due:
            time.sleep(min(0.01, t_due - now))
            continue
        # one control step: 20 ms of the robot's time
        env.speed[:] = 0.0 if hold > 0 else speed
        hold -= CTRL_DT
        a = act(o)
        rew, done, info = env.step(a)
        sim_t += CTRL_DT
        o, _ = env.obs(noisy=False)
        if int(env.steps[0]) % 2 == 0:
            row = frame_row(env, 0, sim_t)
            frames.append(row)
        if bool(info["fell"][0]) and fell_at is None:
            fell_at = sim_t
            falls_lap += 1
            ctx.telemetry(ch_ev, {"text": "It fell%s: back up in 3 s" % ((" on " + zone[2].split(":")[0].lower()) if zone else "")})
        if fell_at is not None and sim_t - fell_at > 3.0:
            fell_at = None
            q = sim.qpos[0]          # back on its feet where it fell, on its line, a step further on
            ly, lyaw = float(env.line_y[0]), float(env.line_yaw[0])
            s_ = float(q[0]) * math.cos(lyaw) + float(q[1]) * math.sin(lyaw) + 0.4
            start([s_ * math.cos(lyaw) - ly * math.sin(lyaw), s_ * math.sin(lyaw) + ly * math.cos(lyaw), lyaw])
            env.line_y[:] = ly
            run += 1
            hold = 1.0
        # along the course: what it walks on now, and at either end it turns back
        if terrain == "course" and fell_at is None:
            q = sim.qpos[0]
            x_, y_ = float(q[0]), float(q[1])
            z_ = next((zz for zz in ZONES if zz[0] <= x_ < zz[1] and abs(y_) < 2.0), None)
            if z_ is not zone:
                zone = z_
                if z_ is not None:
                    ctx.telemetry(ch_ev, {"text": z_[2]})
            if laps and hold <= 0:
                fwd = math.cos(heading)
                if (fwd > 0.7 and x_ > COURSE["length"] + 2.5) or (fwd < -0.7 and x_ < -2.5):
                    ctx.telemetry(ch_ev, {"text": "Lap %d done in %.0f s, %s: turning back" % (lap, sim_t - lap_t0, "no falls" if not falls_lap else "%d fall%s" % (falls_lap, "s" if falls_lap > 1 else ""))})
                    heading = math.atan2(math.sin(heading + math.pi), math.cos(heading + math.pi))
                    set_line(heading)
                    lap, lap_t0, falls_lap = lap + 1, sim_t, 0
        if int(env.steps[0]) >= EP_STEPS - 1:
            env.steps[:] = 0          # (walking on: the episode clock only matters in training)
        if time.time() - last_send > 0.25 and frames:
            last_send = time.time()
            ctx.telemetry(ch, {"format": "esasim.humanoid/1", "kind": "live", "run": run, "t0": round(t_wall0, 3), "frames": frames,
                               "line": [round(float(env.line_y[0]), 3), round(float(env.line_yaw[0]), 4)], "label": "Walking: %s" % pol.name})
            frames = []
    return {"minutes": minutes}


def run(ctx):
    what = str(ctx.params.get("mode") or "Learn to walk").lower()
    if what.startswith("walk"):
        return walk_run(ctx)
    return learn_run(ctx)
