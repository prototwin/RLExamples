import asyncio
import stable_baselines3.ppo
import stable_baselines3.ppo.ppo
import torch
import gymnasium
import numpy as np
import math
import random
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.vec_env
import stable_baselines3.common.vec_env.base_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

import prototwin
import prototwin_gymnasium

# Signal addresses (obtained from ProtoTwin).
left_hip_target_position = 2
left_hip_current_position = 5
left_hip_current_velocity = 6
left_hip_current_force = 7

right_hip_target_position = 9
right_hip_current_position = 12
right_hip_current_velocity = 13
right_hip_current_force = 14

left_thigh_target_position = 16
left_thigh_current_position = 19
left_thigh_current_velocity = 20
left_thigh_current_force = 21

right_thigh_target_position = 23
right_thigh_current_position = 26
right_thigh_current_velocity = 27
right_thigh_current_force = 28

left_calf_target_position = 30
left_calf_current_position = 33
left_calf_current_velocity = 34
left_calf_current_force = 35

right_calf_target_position = 37
right_calf_current_position = 40
right_calf_current_velocity = 41
right_calf_current_force = 42

left_foot_target_position = 44
left_foot_current_position = 47
left_foot_current_velocity = 48
left_foot_current_force = 49

left_foot_position_x = 74
left_foot_position_y = 75
left_foot_position_z = 76

left_foot_linear_velocity_x = 77
left_foot_linear_velocity_y = 78
left_foot_linear_velocity_z = 79

right_foot_target_position = 51
right_foot_current_position = 54
right_foot_current_velocity = 55
right_foot_current_force = 56

right_foot_position_x = 80
right_foot_position_y = 81
right_foot_position_z = 82

right_foot_linear_velocity_x = 83
right_foot_linear_velocity_y = 84
right_foot_linear_velocity_z = 85

neck_target_position = 58
neck_current_position = 61
neck_current_velocity = 62
neck_current_force = 63

base_orientation_x = 64
base_orientation_y = 65
base_orientation_z = 66
base_orientation_w = 67

base_linear_velocity_x = 68
base_linear_velocity_y = 69
base_linear_velocity_z = 70

base_angular_velocity_x = 71
base_angular_velocity_y = 72
base_angular_velocity_z = 73

base_position_x = 86
base_position_y = 87
base_position_z = 88

action_addresses = [left_hip_target_position, right_hip_target_position,
                    left_thigh_target_position, right_thigh_target_position,
                    left_calf_target_position, right_calf_target_position,
                    left_foot_target_position, right_foot_target_position,
                    neck_target_position]

observation_addresses = [left_hip_current_position, right_hip_current_position,
                         left_thigh_current_position, right_thigh_current_position,
                         left_calf_current_position, right_calf_current_position,
                         left_foot_current_position, right_foot_current_position,
                         neck_current_position,
                         left_hip_current_velocity, right_hip_current_velocity,
                         left_thigh_current_velocity, right_thigh_current_velocity,
                         left_calf_current_velocity, right_calf_current_velocity,
                         left_foot_current_velocity, right_foot_current_velocity,
                         neck_current_velocity,
                         left_hip_current_force, right_hip_current_force,
                         left_thigh_current_force, right_thigh_current_force,
                         left_calf_current_force, right_calf_current_force,
                         left_foot_current_force, right_foot_current_force,
                         neck_current_force,
                         base_orientation_x, base_orientation_y, base_orientation_z, base_orientation_w,
                         base_linear_velocity_x, base_linear_velocity_y, base_linear_velocity_z,
                         base_angular_velocity_x, base_angular_velocity_y, base_angular_velocity_z,
                         left_foot_position_y, right_foot_position_y,
                         base_position_x]

duration = 20

class BipedalEnv(prototwin_gymnasium.VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int) -> None:
        super().__init__(client, instance)
        self.has_reset = False
        self.duration = random.uniform(0, duration)
        self.previous_action = [0] * 9
        self.previous_joint_velocities = [0] * 9

    def observations(self):
        obs = [0] * 49
        for i in range(40):
            obs[i] = self.get(observation_addresses[i])
        for i in range(9):
            obs[40 + i] = self.previous_action[i]
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if (self.has_reset):
            self.duration = duration
        self.has_reset = True
        self.previous_action = [0] * 9
        self.previous_joint_velocities = [0] * 9
        obs = self.observations()
        return obs, {}
    
    def phi(self, measured, target):
        diff = target - measured
        norm = diff * diff
        return math.exp(-norm / 0.125)
    
    def _reward_forward_velocity(self, target_velocity):
        return self.phi(self.get(base_linear_velocity_z), target_velocity)
    
    def _reward_angular_velocity(self):
        return self.phi(self.get(base_angular_velocity_y), 0)
    
    def _reward_base_linear_velocity_penalty(self):
        vy = self.get(base_linear_velocity_y)
        return -(vy * vy)
    
    def _reward_angular_velocity_penalty(self):
        vx = self.get(base_angular_velocity_x)
        vz = self.get(base_angular_velocity_z)
        return -(vx * vx + vz * vz)
    
    def _reward_action_rate_penalty(self, action):
        reward = 0
        for i in range(9):
            q = action[i] - self.previous_action[i]
            reward += q * q
        return -reward
    
    def _reward_joint_velocity_penalty(self):
        reward = 0
        for i in range(9):
            v = self.get(observation_addresses[9 + i])
            reward += v * v
        return -reward
    
    def _reward_joint_acceleration_penalty(self, dt):
        reward = 0
        for i in range(9):
            a = (self.get(observation_addresses[9 + i]) - self.previous_joint_velocities[i]) / dt
            reward += a * a
        return -reward
    
    def _reward_joint_torque_penalty(self):
        reward = 0
        for i in range(12):
            t = self.get(observation_addresses[18 + i])
            reward += t * t
        return -reward
    
    def _reward_joint_limit_penalty(self):
        space = self.action_space
        low = space.low
        high = space.high
        reward = 0
        for i in range(8):
            center = (high[i] + low[i]) * 0.5
            max_distance = math.fabs(high[i] - center)
            distance = min(max_distance, math.fabs(self.get(action_addresses[i]) - center))
            fraction = distance / max_distance
            reward += fraction * fraction * fraction * fraction * fraction
        return -reward / 12
    
    def _reward_up_axis_penalty(self):
        x = self.get(base_orientation_x)
        z = self.get(base_orientation_z)
        x2 = x + x
        z2 = z + z
        xx2 = x * x2
        zz2 = z * z2
        up = 1.0 - xx2 - zz2
        return -math.acos(max(-1, min(1, up)))
    
    def _reward_forward_axis_penalty(self):
        x = self.get(base_orientation_x)
        y = self.get(base_orientation_y)
        x2 = x + x
        y2 = y + y
        xx2 = x * x2
        yy2 = y * y2
        forward = 1.0 - xx2 - yy2
        return -math.acos(max(1, min(1, forward)))
    
    def _reward_drift_penalty(self):
        return -math.fabs(self.get(base_position_x))

    def reward(self, action, dt):
        forward_velocity_reward = self._reward_forward_velocity(0.5)
        forward_velocity_scale = max(0, forward_velocity_reward)

        reward = 0
        reward += self._reward_forward_velocity(0.5) * dt
        reward += self._reward_angular_velocity() * forward_velocity_scale * dt
        reward += self._reward_base_linear_velocity_penalty() * 2.0 * dt
        reward += self._reward_angular_velocity_penalty() * 0.05 * dt
        reward += self._reward_action_rate_penalty(action) * 0.01 * dt
        reward += self._reward_joint_acceleration_penalty(dt) * 0.00001 * dt
        reward += self._reward_joint_torque_penalty() * 0.00006 * dt
        reward += self._reward_joint_limit_penalty() * dt
        reward += self._reward_up_axis_penalty() * 0.1 * dt
        reward += self._reward_forward_axis_penalty() * 0.1 * dt
        reward += self._reward_drift_penalty() * 0.1 * dt

        return max(0, reward)
    
    def terminal(self):
        return self.get(base_position_y) < -0.01 or math.fabs(self.get(base_position_x)) > 4
    
    def apply(self, action):
        self.action = action
        for i in range(9):
            self.set(action_addresses[i], action[i])

    def step(self):
        reward = self.reward(self.action, 0.01)
        obs = self.observations()
        self.previous_action = self.action
        self.previous_joint_velocities = obs[9:18]
        terminated = self.terminal()
        truncated = self.time >= self.duration
        return obs, reward, terminated, truncated, {}
    
async def main():
    client = await prototwin.start()
    await client.load("bipedal.ptm")

    action_high = np.array([0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3], dtype=np.float32)
    action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    observation_high = np.array([np.finfo(np.float32).max] * 49, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    gymnasium.vector.SyncVectorEnv

    save_freq = 4000 # Number of timesteps per instance
    policy_kwargs = dict(activation_fn=torch.nn.ELU, net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))
    instances = 100
    steps = 400
    minibatch_size = 50 * steps
    learning_rate = 0.0003
    env = prototwin_gymnasium.VecEnv(BipedalEnv, client, "Main", instances, observation_space, action_space, pattern=prototwin.Pattern.LINEAR_X, spacing=0.5)
    monitored = stable_baselines3.common.vec_env.VecMonitor(env)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs/checkpoints/", name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)
    model = stable_baselines3.PPO(stable_baselines3.ppo.MlpPolicy, monitored, verbose=1, ent_coef=0.001, learning_rate=learning_rate, batch_size=minibatch_size, n_steps=steps, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=1_000_000_000, callback=checkpoint_callback)

asyncio.run(main())