# STEP 1: Import dependencies
import asyncio
import random
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
quadrotor_orientation_sensor_x = 29
quadrotor_orientation_sensor_y = 30
quadrotor_orientation_sensor_z = 31
quadrotor_orientation_sensor_w = 32
quadrotor_position_sensor_x = 33
quadrotor_position_sensor_y = 34
quadrotor_position_sensor_z = 35
quadrotor_linear_velocity_sensor_x = 36
quadrotor_linear_velocity_sensor_y = 37
quadrotor_linear_velocity_sensor_z = 38
quadrotor_angular_velocity_sensor_x = 39
quadrotor_angular_velocity_sensor_y = 40
quadrotor_angular_velocity_sensor_z = 41
rotor_1_motor_state = 1
rotor_1_motor_target_position = 2
rotor_1_motor_target_velocity = 3
rotor_1_motor_force_limit = 4
rotor_1_motor_current_position = 5
rotor_1_motor_current_velocity = 6
rotor_1_motor_current_force = 7
rotor_2_motor_state = 15
rotor_2_motor_target_position = 16
rotor_2_motor_target_velocity = 17
rotor_2_motor_force_limit = 18
rotor_2_motor_current_position = 19
rotor_2_motor_current_velocity = 20
rotor_2_motor_current_force = 21
rotor_3_motor_state = 8
rotor_3_motor_target_position = 9
rotor_3_motor_target_velocity = 10
rotor_3_motor_force_limit = 11
rotor_3_motor_current_position = 12
rotor_3_motor_current_velocity = 13
rotor_3_motor_current_force = 14
rotor_4_motor_state = 22
rotor_4_motor_target_position = 23
rotor_4_motor_target_velocity = 24
rotor_4_motor_force_limit = 25
rotor_4_motor_current_position = 26
rotor_4_motor_current_velocity = 27
rotor_4_motor_current_force = 28
target_position_writer_x = 42
target_position_writer_y = 43
target_position_writer_z = 44
target_position_sensor_x = 45
target_position_sensor_y = 46
target_position_sensor_z = 47

states = [rotor_1_motor_current_velocity, rotor_2_motor_current_velocity, rotor_3_motor_current_velocity, rotor_4_motor_current_velocity,
          quadrotor_orientation_sensor_x, quadrotor_orientation_sensor_y, quadrotor_orientation_sensor_z, quadrotor_orientation_sensor_w,
          quadrotor_position_sensor_x, quadrotor_position_sensor_y, quadrotor_position_sensor_z,
          quadrotor_linear_velocity_sensor_x, quadrotor_linear_velocity_sensor_y, quadrotor_linear_velocity_sensor_z,
          quadrotor_angular_velocity_sensor_x, quadrotor_angular_velocity_sensor_y, quadrotor_angular_velocity_sensor_z,
          target_position_sensor_x, target_position_sensor_y, target_position_sensor_z]

actions = [rotor_1_motor_target_velocity, rotor_2_motor_target_velocity, rotor_3_motor_target_velocity, rotor_4_motor_target_velocity]

state_size = len(states)
action_size = len(actions)

max_speed = math.radians(3600)

# STEP 3: Create your own vectorized environment instance by extending the VecEnvInstance base class
class QuadrotorEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.duration = 20
        self.solved_time = 0

    def error(self):
        '''
        The distance from the quadrotor to the target position
        '''
        dx = self.get(target_position_sensor_x) - self.get(quadrotor_position_sensor_x)
        dy = self.get(target_position_sensor_y) - self.get(quadrotor_position_sensor_y)
        dz = self.get(target_position_sensor_z) - self.get(quadrotor_position_sensor_z)
        normsq = dx * dx + dy * dy + dz * dz
        return math.sqrt(normsq)

    def reward_position_target(self, dt):
        '''
        Reward the agent for solving to the target position
        '''
        if self.error() < 0.02:
            self.solved_time += dt
        else:
            self.solved_time = 0
        return min(1, self.solved_time)
    
    def reward_position_error(self):
        '''
        Reward the agent for how close the quadrotor is to the target position
        '''
        return 1 / (1 + 2 * self.error())
    
    def reward_angular_velocity_penalty(self):
        '''
        Penalize the agent for high angular velocity
        '''
        wx = self.get(quadrotor_angular_velocity_sensor_x)
        wy = self.get(quadrotor_angular_velocity_sensor_y)
        wz = self.get(quadrotor_angular_velocity_sensor_z)
        normsq = wx * wx + wy * wy + wz * wz
        return normsq
    
    def reward_linear_velocity_penalty(self):
        '''
        Penalize the agent for high linear velocity
        '''
        vx = self.get(quadrotor_linear_velocity_sensor_x)
        vy = self.get(quadrotor_linear_velocity_sensor_y)
        vz = self.get(quadrotor_linear_velocity_sensor_z)
        normsq = vx * vx + vy * vy + vz * vz
        return normsq

    def reward(self):
        '''
        Reward function
        '''
        dt = 0.005
        reward = 0
        reward += self.reward_position_target(dt)
        reward += self.reward_position_error()
        reward -= self.reward_angular_velocity_penalty() * 0.1
        reward -= self.reward_linear_velocity_penalty() * 0.1
        return reward * dt

    def terminal(self):
        '''
        Whether the agent is in a terminal state
        '''
        x = self.get(quadrotor_position_sensor_x)
        y = self.get(quadrotor_position_sensor_y)
        z = self.get(quadrotor_position_sensor_z)
        return x < -2 or x > 2 or y < 0.5 or y > 8 or z < -2 or z > 2

    def observations(self):
        '''
        The current observations
        '''
        obs = [0] * state_size
        for i in range(state_size):
            value = self.get(states[i])
            if i < 4:
                value /= max_speed
            obs[i] = value
        return np.array(obs)
    
    def reset_target(self):
        '''
        Reset the target position to a random position
        '''
        x = random.uniform(-1, 1)
        y = random.uniform(1, 5)
        z = random.uniform(-1, 1)
        self.set(target_position_writer_x, x)
        self.set(target_position_writer_y, y)
        self.set(target_position_writer_z, z)

    def reset(self, seed = None):
        super().reset(seed=seed)
        self.reset_target()
        self.solved_time = 0
        asyncio.run(self.client.sync())
        return self.observations(), {}
    
    def apply(self, action):
        for i in range(action_size):
            self.set(actions[i], action[i] * max_speed)

    def step(self):
        obs = self.observations()
        reward = self.reward()
        done = self.terminal()
        truncated = self.time > self.duration
        return obs, reward, done, truncated, {}
    
# STEP 4: Setup the training session
async def main():
    # Launch ProtoTwin Connect and load the quadrotor model
    client = await prototwin.start()
    await client.load("quadrotor.ptm")

    observation_high = np.array([np.finfo(np.float32).max] * state_size, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    action_low = np.array([0] * action_size, dtype=np.float32)
    action_high = np.array([1] * action_size, dtype=np.float32)
    action_space = gymnasium.spaces.Box(action_low, action_high, dtype=np.float32)

    # Create callback to regularly save the model
    save_freq = 5000
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs/checkpoints/", name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)

    # Define learning rate schedule
    def lr_schedule(progress_remaining):
        initial_lr = 0.001
        return initial_lr * (progress_remaining ** 2)
    
    # Create the vectorized environment
    entity_name = "Main"
    num_envs = 15*15
    pattern = prototwin.Pattern.GRID
    spacing = 3
    env = VecEnv(QuadrotorEnv, client, entity_name, num_envs, observation_space, action_space, pattern=pattern, spacing=spacing)
    env = VecMonitor(env) # Monitor the training progress

    # Define the ML model
    batch_size = 4500
    n_steps = 1000
    ent_coef = 0.00003
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
    model = PPO(MlpPolicy, env, verbose=1, batch_size=batch_size, n_steps=n_steps, ent_coef=ent_coef, policy_kwargs=policy_kwargs, learning_rate=lr_schedule, tensorboard_log="./tensorboard/", device="cuda")

    # Start training!
    model.learn(total_timesteps=200_000_000, callback=checkpoint_callback)

asyncio.run(main())
