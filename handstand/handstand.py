import asyncio
import stable_baselines3.ppo
import stable_baselines3.ppo.ppo
import torch
import prototwin
import prototwin_gymnasium
import gymnasium
import numpy as np
import math
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.vec_env
import stable_baselines3.common.vec_env.base_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Signal addresses (obtained from ProtoTwin).
base_orientation_x = 85
base_orientation_y = 86
base_orientation_z = 87
base_orientation_w = 88

base_angular_velocity_x = 89
base_angular_velocity_y = 90
base_angular_velocity_z = 91

base_linear_velocity_x = 92
base_linear_velocity_y = 93
base_linear_velocity_z = 94

base_position_x = 107
base_position_y = 108
base_position_z = 109

bl_hip_target_position = 65
bl_hip_position = 68
bl_hip_velocity = 69
bl_hip_torque = 70

br_hip_target_position = 44
br_hip_position = 47
br_hip_velocity = 48
br_hip_torque = 49

fl_hip_target_position = 30
fl_hip_position = 33
fl_hip_velocity = 34
fl_hip_torque = 35

fr_hip_target_position = 2
fr_hip_position = 5
fr_hip_velocity = 6
fr_hip_torque = 7

bl_thigh_target_position = 72
bl_thigh_position = 75
bl_thigh_velocity = 76
bl_thigh_torque = 77

br_thigh_target_position = 51
br_thigh_position = 54
br_thigh_velocity = 55
br_thigh_torque = 56

fl_thigh_target_position = 23
fl_thigh_position = 26
fl_thigh_velocity = 27
fl_thigh_torque = 28

fr_thigh_target_position = 9
fr_thigh_position = 12
fr_thigh_velocity = 13
fr_thigh_torque = 14

bl_calf_target_position = 79
bl_calf_position = 82
bl_calf_velocity = 83
bl_calf_torque = 84

br_calf_target_position = 58
br_calf_position = 61
br_calf_velocity = 62
br_calf_torque = 63

fl_calf_target_position = 37
fl_calf_position = 40
fl_calf_velocity = 41
fl_calf_torque = 42

fr_calf_target_position = 16
fr_calf_position = 19
fr_calf_velocity = 20
fr_calf_torque = 21

bl_foot_distance = 96
br_foot_distance = 99
fl_foot_distance = 102
fr_foot_distance = 105

bl_foot_velocity_x = 110
bl_foot_velocity_z = 112
br_foot_velocity_x = 113
br_foot_velocity_z = 115
fl_foot_velocity_x = 116
fl_foot_velocity_z = 118
fr_foot_velocity_x = 119
fr_foot_velocity_z = 121

head_distance = 123

foot_radius = 0.02

torques = [bl_hip_torque, br_hip_torque, fl_hip_torque, fr_hip_torque,
           bl_thigh_torque, br_thigh_torque, fl_thigh_torque, fr_thigh_torque,
           bl_calf_torque, br_calf_torque, fl_calf_torque, fr_calf_torque]

states = [base_orientation_x, base_orientation_y, base_orientation_z, base_orientation_w,
          base_angular_velocity_x, base_angular_velocity_y, base_angular_velocity_z,
          base_linear_velocity_x, base_linear_velocity_y, base_linear_velocity_z,
          bl_hip_position, br_hip_position, fl_hip_position, fr_hip_position,
          bl_thigh_position, br_thigh_position, fl_thigh_position, fr_thigh_position,
          bl_calf_position, br_calf_position, fl_calf_position, fr_calf_position,
          bl_hip_velocity, br_hip_velocity, fl_hip_velocity, fr_hip_velocity,
          bl_thigh_velocity, br_thigh_velocity, fl_thigh_velocity, fr_thigh_velocity,
          bl_calf_velocity, br_calf_velocity, fl_calf_velocity, fr_calf_velocity,
          bl_foot_distance, br_foot_distance, fl_foot_distance, fr_foot_distance]

actions = [bl_hip_target_position, br_hip_target_position, fl_hip_target_position, fr_hip_target_position,
           bl_thigh_target_position, br_thigh_target_position, fl_thigh_target_position, fr_thigh_target_position,
           bl_calf_target_position, br_calf_target_position, fl_calf_target_position, fr_calf_target_position]

class HandstandEnv(prototwin_gymnasium.VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.time = 0
        self.previous_action = [0] * 12
        self.feet_air_time = 0
    
    def _reward_back_feet_height(self):
        return max(0, min(0.4, min(self.get(bl_foot_distance), self.get(br_foot_distance)) - foot_radius))
    
    def _reward_back_feet_air_time(self, dt):
        self.feet_air_time += dt
        l = self.get(bl_foot_distance) - foot_radius
        r = self.get(br_foot_distance) - foot_radius
        if l < 0.2 or r < 0.2:
            self.feet_air_time = 0
        return min(1, self.feet_air_time)
    
    def _reward_joint_limit_penalty(self, obs):
        space = self.action_space
        low = space.low
        high = space.high
        reward = 0
        for i in range(12):
            center = (high[i] + low[i]) * 0.5
            max_distance = math.fabs(high[i] - center)
            distance = min(max_distance, math.fabs(obs[10 + i] - center))
            fraction = distance / max_distance
            reward += fraction * fraction * fraction * fraction * fraction
        return -reward / 12
    
    def _reward_drift_penalty(self):
        x = self.get(base_position_x)
        y = self.get(base_position_y)
        return -math.sqrt(x * x + y * y)
            
    def _reward_fn(self, obs):
        dt = 0.01
        reward = 0
        reward += self._reward_back_feet_height() * 0.5 * dt
        reward += self._reward_back_feet_air_time(dt) * dt
        reward += self._reward_joint_limit_penalty(obs) * dt
        reward += self._reward_drift_penalty() * 0.1 * dt
        return max(0, reward)
    
    def _terminal(self):
        dx = self.get(base_position_x)
        dy = self.get(base_position_y)
        drift = math.sqrt(dx * dx + dy * dy)
        return self.get(head_distance) < 0.1 or self.get(head_distance) > 0.35 or self.get(base_position_z) < 0.04 or drift > 1
    
    def reset(self, seed = None):
        super().reset(seed=seed)
        self.previous_action = [0] * 12
        self.feet_air_time = 0
        return np.array([0] * 50), {}
    
    def apply(self, action):
        self.action = action
        for i in range(12):
            self.set(actions[i], action[i])

    def step(self):
        obs = [0] * 50
        for i in range(38):
            obs[i] = self.get(states[i])
            if (i >= 34):
                obs[i] -= foot_radius

        for i in range(12):
            obs[38 + i] = self.previous_action[i]

        reward = self._reward_fn(obs)
        self.previous_action = self.action
        done = self._terminal()
        truncated = self.time > 20
        return obs, reward, done, truncated, {}

async def main():
    client = await prototwin.start()
    await client.load("handstand.ptm")

    action_high = np.array([0.2, 0.2, 0.2, 0.2, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    observation_high = np.array([np.finfo(np.float32).max] * 50, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    save_freq = 4000 # Number of timesteps per instance

    policy_kwargs = dict(activation_fn=torch.nn.ELU, net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))
    instances = 64
    steps = 2000
    learning_rate = 0.003
    ent_coef = 0.001
    env = prototwin_gymnasium.VecEnv(HandstandEnv, client, "Robot", instances, observation_space, action_space, pattern=prototwin.Pattern.GRID, spacing=1)
    monitored = stable_baselines3.common.vec_env.VecMonitor(env)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs/checkpoints/", name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)
    model = stable_baselines3.PPO(stable_baselines3.ppo.MlpPolicy, monitored, verbose=1, ent_coef=ent_coef, learning_rate=learning_rate, batch_size=instances*steps, n_steps=steps, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=1_000_000_000, callback=checkpoint_callback)

asyncio.run(main())