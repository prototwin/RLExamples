# STEP 1: Import dependencies
import os
import asyncio
import math
import torch
import numpy as np
import prototwin
import gymnasium
from prototwin_gymnasium import VecEnvInstance, VecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# STEP 2: Signal addresses (copy/paste these values from ProtoTwin)
base_orientation_sensor_x = 85
base_orientation_sensor_y = 86
base_orientation_sensor_z = 87
base_orientation_sensor_w = 88
base_angular_velocity_sensor_x = 89
base_angular_velocity_sensor_y = 90
base_angular_velocity_sensor_z = 91
base_linear_velocity_sensor_x = 92
base_linear_velocity_sensor_y = 93
base_linear_velocity_sensor_z = 94
base_position_sensor_x = 95
base_position_sensor_y = 96
base_position_sensor_z = 97
head_position_sensor_x = 98
head_position_sensor_y = 99
head_position_sensor_z = 100
bl_hip_motor_state = 1
bl_hip_motor_target_position = 2
bl_hip_motor_target_velocity = 3
bl_hip_motor_force_limit = 4
bl_hip_motor_current_position = 5
bl_hip_motor_current_velocity = 6
bl_hip_motor_current_force = 7
br_hip_motor_state = 29
br_hip_motor_target_position = 30
br_hip_motor_target_velocity = 31
br_hip_motor_force_limit = 32
br_hip_motor_current_position = 33
br_hip_motor_current_velocity = 34
br_hip_motor_current_force = 35
fl_hip_motor_state = 15
fl_hip_motor_target_position = 16
fl_hip_motor_target_velocity = 17
fl_hip_motor_force_limit = 18
fl_hip_motor_current_position = 19
fl_hip_motor_current_velocity = 20
fl_hip_motor_current_force = 21
fr_hip_motor_state = 43
fr_hip_motor_target_position = 44
fr_hip_motor_target_velocity = 45
fr_hip_motor_force_limit = 46
fr_hip_motor_current_position = 47
fr_hip_motor_current_velocity = 48
fr_hip_motor_current_force = 49
bl_thigh_motor_state = 57
bl_thigh_motor_target_position = 58
bl_thigh_motor_target_velocity = 59
bl_thigh_motor_force_limit = 60
bl_thigh_motor_current_position = 61
bl_thigh_motor_current_velocity = 62
bl_thigh_motor_current_force = 63
br_thigh_motor_state = 78
br_thigh_motor_target_position = 79
br_thigh_motor_target_velocity = 80
br_thigh_motor_force_limit = 81
br_thigh_motor_current_position = 82
br_thigh_motor_current_velocity = 83
br_thigh_motor_current_force = 84
fl_thigh_motor_state = 64
fl_thigh_motor_target_position = 65
fl_thigh_motor_target_velocity = 66
fl_thigh_motor_force_limit = 67
fl_thigh_motor_current_position = 68
fl_thigh_motor_current_velocity = 69
fl_thigh_motor_current_force = 70
fr_thigh_motor_state = 71
fr_thigh_motor_target_position = 72
fr_thigh_motor_target_velocity = 73
fr_thigh_motor_force_limit = 74
fr_thigh_motor_current_position = 75
fr_thigh_motor_current_velocity = 76
fr_thigh_motor_current_force = 77
bl_calf_motor_state = 8
bl_calf_motor_target_position = 9
bl_calf_motor_target_velocity = 10
bl_calf_motor_force_limit = 11
bl_calf_motor_current_position = 12
bl_calf_motor_current_velocity = 13
bl_calf_motor_current_force = 14
bl_foot_linear_velocity_sensor_x = 119
bl_foot_linear_velocity_sensor_y = 120
bl_foot_linear_velocity_sensor_z = 121
bl_foot_position_sensor_x = 122
bl_foot_position_sensor_y = 123
bl_foot_position_sensor_z = 124
br_calf_motor_state = 36
br_calf_motor_target_position = 37
br_calf_motor_target_velocity = 38
br_calf_motor_force_limit = 39
br_calf_motor_current_position = 40
br_calf_motor_current_velocity = 41
br_calf_motor_current_force = 42
br_foot_linear_velocity_sensor_x = 113
br_foot_linear_velocity_sensor_y = 114
br_foot_linear_velocity_sensor_z = 115
br_foot_position_sensor_x = 116
br_foot_position_sensor_y = 117
br_foot_position_sensor_z = 118
fl_calf_motor_state = 22
fl_calf_motor_target_position = 23
fl_calf_motor_target_velocity = 24
fl_calf_motor_force_limit = 25
fl_calf_motor_current_position = 26
fl_calf_motor_current_velocity = 27
fl_calf_motor_current_force = 28
fl_foot_linear_velocity_sensor_x = 101
fl_foot_linear_velocity_sensor_y = 102
fl_foot_linear_velocity_sensor_z = 103
fl_foot_position_sensor_x = 104
fl_foot_position_sensor_y = 105
fl_foot_position_sensor_z = 106
fr_calf_motor_state = 50
fr_calf_motor_target_position = 51
fr_calf_motor_target_velocity = 52
fr_calf_motor_force_limit = 53
fr_calf_motor_current_position = 54
fr_calf_motor_current_velocity = 55
fr_calf_motor_current_force = 56
fr_foot_linear_velocity_sensor_x = 107
fr_foot_linear_velocity_sensor_y = 108
fr_foot_linear_velocity_sensor_z = 109
fr_foot_position_sensor_x = 110
fr_foot_position_sensor_y = 111
fr_foot_position_sensor_z = 112

current_positions = [bl_hip_motor_current_position, br_hip_motor_current_position, fl_hip_motor_current_position, fr_hip_motor_current_position,
                     bl_thigh_motor_current_position, br_thigh_motor_current_position, fl_thigh_motor_current_position, fr_thigh_motor_current_position,
                     bl_calf_motor_current_position, br_calf_motor_current_position, fl_calf_motor_current_position, fr_calf_motor_current_position]

torques = [bl_hip_motor_current_force, br_hip_motor_current_force, fl_hip_motor_current_force, fr_hip_motor_current_force,
           bl_thigh_motor_current_force, br_thigh_motor_current_force, fl_thigh_motor_current_force, fr_thigh_motor_current_force,
           bl_calf_motor_current_force, br_calf_motor_current_force, fl_calf_motor_current_force, fr_calf_motor_current_force]

states = [base_position_sensor_x, base_position_sensor_y, base_position_sensor_z,
          base_orientation_sensor_x, base_orientation_sensor_y, base_orientation_sensor_z, base_orientation_sensor_w,
          base_angular_velocity_sensor_x, base_angular_velocity_sensor_y, base_angular_velocity_sensor_z,
          base_linear_velocity_sensor_x, base_linear_velocity_sensor_y, base_linear_velocity_sensor_z,
          bl_hip_motor_current_position, br_hip_motor_current_position, fl_hip_motor_current_position, fr_hip_motor_current_position,
          bl_thigh_motor_current_position, br_thigh_motor_current_position, fl_thigh_motor_current_position, fr_thigh_motor_current_position,
          bl_calf_motor_current_position, br_calf_motor_current_position, fl_calf_motor_current_position, fr_calf_motor_current_position,
          bl_hip_motor_current_velocity, br_hip_motor_current_velocity, fl_hip_motor_current_velocity, fr_hip_motor_current_velocity,
          bl_thigh_motor_current_velocity, br_thigh_motor_current_velocity, fl_thigh_motor_current_velocity, fr_thigh_motor_current_velocity,
          bl_calf_motor_current_velocity, br_calf_motor_current_velocity, fl_calf_motor_current_velocity, fr_calf_motor_current_velocity,
          bl_foot_position_sensor_y, br_foot_position_sensor_y, fl_foot_position_sensor_y, fr_foot_position_sensor_y]

actions = [bl_hip_motor_target_position, br_hip_motor_target_position, fl_hip_motor_target_position, fr_hip_motor_target_position,
           bl_thigh_motor_target_position, br_thigh_motor_target_position, fl_thigh_motor_target_position, fr_thigh_motor_target_position,
           bl_calf_motor_target_position, br_calf_motor_target_position, fl_calf_motor_target_position, fr_calf_motor_target_position]

current_position_size = len(current_positions)
torque_size = len(torques)
state_size = len(states)
action_size = len(actions)
observation_size = state_size + action_size

# STEP 3: Create your own vectorized environment instance by extending the VecEnvInstance base class
class HandstandEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.duration = 10
        self.previous_action = [0] * action_size
        self.feet_air_time = 0
    
    def reward_back_feet_height(self):
        '''
        Reward the agent for placing its feet higher in the air
        '''
        bl = self.get(bl_foot_position_sensor_y)
        br = self.get(br_foot_position_sensor_y)
        return max(0, min(0.8, (bl + br) * 0.5))
    
    def reward_back_feet_air_time(self, dt):
        '''
        Reward the agent for placing its feet in the air for a long amount of time
        '''
        bl = self.get(bl_foot_position_sensor_y)
        br = self.get(br_foot_position_sensor_y)
        if bl < 0.4 or br < 0.4:
            self.feet_air_time = 0
        else:
            self.feet_air_time += dt
        return min(1, self.feet_air_time)
    
    def reward_position_drift_penalty(self):
        '''
        Penalize the agent for positional drift
        '''
        dx = self.get(base_position_sensor_x)
        dy = self.get(base_position_sensor_y)
        return math.sqrt(dx * dx + dy * dy)
    
    def reward_rotation_drift_penalty(self):
        '''
        Penalize the agent for rotational drift
        '''
        qx = self.get(base_orientation_sensor_x)
        qy = self.get(base_orientation_sensor_y)
        qz = self.get(base_orientation_sensor_z)
        qw = self.get(base_orientation_sensor_w)
        dot = np.dot([qx, qy, qz, qw], [0, 0, 0, 1])
        arg = np.clip(2.0 * dot * dot - 1.0, -1.0, 1.0)
        return np.arccos(arg) / np.pi
    
    def reward_joint_limit_penalty(self):
        '''
        Penalize the agent for joint angles being too close to the joint limits
        '''
        low = self.action_space.low
        high = self.action_space.high
        penalty = 0
        for i in range(current_position_size):
            center = (low[i] + high[i]) * 0.5
            max_distance = math.fabs(high[i] - center)
            distance = min(max_distance, math.fabs(self.get(current_positions[i]) - center))
            fraction = distance / max_distance
            penalty += fraction * fraction * fraction * fraction * fraction
        return penalty / current_position_size
    
    def reward_joint_torque_penalty(self):
        '''
        Penalize the agent for high torques in the joints
        '''
        penalty = 0
        for i in range(torque_size):
            torque = self.get(torques[i])
            penalty += torque * torque
        return penalty / torque_size
            
    def reward(self):
        '''
        Reward function
        '''
        dt = 0.01

        reward = 0
        reward += self.reward_back_feet_height()
        reward += self.reward_back_feet_air_time(dt)

        penalty = 0
        penalty += self.reward_position_drift_penalty()
        penalty += self.reward_rotation_drift_penalty()
        penalty += self.reward_joint_limit_penalty() * 0.75
        penalty += self.reward_joint_torque_penalty() * 0.002
        penalty *= min(1, reward)

        return (reward - penalty) * dt
    
    def terminal(self):
        '''
        Whether the agent is in a terminal state
        '''
        drift = self.reward_position_drift_penalty()
        return self.get(head_position_sensor_y) < 0.1 or self.get(head_position_sensor_y) > 0.5 or self.get(base_position_sensor_z) < 0.02 or drift > 1
    
    def observations(self):
        '''
        The current observations
        '''
        obs = [0] * observation_size
        for i in range(state_size):
            obs[i] = self.get(states[i])
        for i in range(action_size):
            obs[state_size + i] = self.previous_action[i]
        return np.array(obs)
    
    def reset(self, seed = None):
        super().reset(seed=seed)
        self.previous_action = [0] * action_size
        self.feet_air_time = 0
        return self.observations(), {}
    
    def apply(self, action):
        self.action = action
        for i in range(action_size):
            self.set(actions[i], action[i])

    def step(self):
        obs = self.observations()
        self.previous_action = self.action
        reward = self.reward()
        done = self.terminal()
        truncated = self.time > self.duration
        return obs, reward, done, truncated, {}

# STEP 4: Setup the training session
async def main():
    # Launch ProtoTwin Connect and load the handstand model
    path = os.path.join(os.path.dirname(__file__), "handstand.ptm")
    client = await prototwin.start()
    await client.load(path)

    observation_high = np.array([np.finfo(np.float32).max] * observation_size, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    action_high = np.array([0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32)
    action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    # Create callback to regularly save the model
    save_freq = 5000
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs/checkpoints/", name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)

    # Define learning rate schedule
    def lr_schedule(progress_remaining):
        initial_lr = 0.001
        return initial_lr * (progress_remaining ** 2)
    
    # Create the vectorized environment
    entity_name = "Main"
    num_envs = 100
    pattern = prototwin.Pattern.GRID
    spacing = 1
    env = VecEnv(HandstandEnv, client, entity_name, num_envs, observation_space, action_space, pattern=pattern, spacing=spacing)
    env = VecMonitor(env) # Monitor the training progress

    # Define the ML model
    batch_size = 40000
    n_steps = 1000
    ent_coef = 0.0002
    policy_kwargs = dict(activation_fn=torch.nn.ELU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
    model = PPO(MlpPolicy, env, verbose=1, batch_size=batch_size, n_steps=n_steps, ent_coef=ent_coef, policy_kwargs=policy_kwargs, learning_rate=lr_schedule, tensorboard_log="./tensorboard/", device="cuda")

    # Start training!
    model.learn(total_timesteps=200_000_000, callback=checkpoint_callback)

asyncio.run(main())
