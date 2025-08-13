# STEP 1: Import dependencies
import os
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
cart_motor_target_position = 2
cart_motor_target_velocity = 3
cart_motor_current_position = 5
cart_motor_current_velocity = 6
cart_motor_current_force = 7
pole_motor_current_position = 12
pole_motor_current_velocity = 13

states = [cart_motor_current_position, pole_motor_current_position, cart_motor_current_velocity, pole_motor_current_velocity]
actions = [cart_motor_target_position]

state_size = len(states)
action_size = len(actions)
observation_size = state_size + action_size

domain_randomization = True
max_cart_position = 0.35

# STEP 3: Create your own vectorized environment instance by extending the VecEnvInstance base class
class CartPoleEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.duration = 10
        self.previous_action = [0] * action_size
        if domain_randomization:
            self.set_domain_randomization_params()

    def set_domain_randomization_params(self):
        '''
        Sets the domain randomization parameters
        '''
        self.encoder_constant_error = random.uniform(math.radians(-0.25), math.radians(0.25))
        self.encoder_precision = (math.pi * 2.0) / random.choice([600, 800, 1000, 1200, 1500, 2000, 2500])
        self.encoder_max_noise = random.uniform(0, math.radians(0.5))
    
    def normalize_position(self, position):
        '''
        Normalizes the specified position so that it lies in the range [-1, 1]
        '''
        return position / max_cart_position

    def normalize_angle(self, angle):
        '''
        Normalizes the specified angle so that it lies in the range [-1, 1]
        '''
        return math.atan2(math.sin(angle), math.cos(math.pi - angle)) / math.pi
    
    def round(self, value, base):
        '''
        Rounds the given value to the nearest multiple of the specified base.
        '''
        return round(value / base) * base
    
    def pole_angle_observation(self):
        '''
        The angular distance of the pole from the upright position after domain randomization
        '''
        value = self.get(pole_motor_current_position) # Read the true angle of the pole
        if domain_randomization:
            value += self.encoder_constant_error # Add a constant error from the encoder
            value += random.uniform(-self.encoder_max_noise, self.encoder_max_noise) # Add some measurement noise from the encoder
            value = self.round(value, self.encoder_precision) # Round to the precision of the encoder
        value = self.normalize_angle(value)
        return value
    
    def reward_angle(self):
        '''
        Reward the agent for how close the angle of the pole is from the upright position
        '''
        pole_angle = self.pole_angle_observation()
        return 1 - math.fabs(pole_angle)
    
    def reward_position_penalty(self):
        '''
        Penalize the agent for moving the cart away from the center
        '''
        cart_position = self.get(cart_motor_current_position)
        return math.fabs(self.normalize_position(cart_position))
    
    def reward_force_penalty(self):
        '''
        Penalize the agent for high force in the cart's motor
        '''
        force = self.get(cart_motor_current_force)
        return math.fabs(force)

    def reward(self):
        '''
        Reward function
        '''
        dt = 0.005
        reward = 0
        reward += self.reward_angle()
        reward -= self.reward_position_penalty() * reward * 0.5
        reward -= self.reward_force_penalty() * 0.0025
        return reward * dt
    
    def observations(self):
        '''
        The current observations
        '''
        obs = [0] * observation_size
        for i in range(state_size):
            obs[i] = self.get(states[i])
        for i in range(action_size):
            obs[state_size + i] = self.previous_action[i]
        obs[0] = self.normalize_position(self.get(cart_motor_current_position))
        obs[1] = self.pole_angle_observation()
        return np.array(obs)
    
    def reset(self, seed = None):
        super().reset(seed=seed)
        self.previous_action = [0] * action_size
        if domain_randomization:
            self.set_domain_randomization_params()
        return self.observations(), {}
    
    def apply(self, action):
        self.action = action
        for i in range(action_size):
            self.set(actions[i], action[i])

    def step(self):
        obs = self.observations()
        self.previous_action = self.action
        reward = self.reward()
        done = abs(obs[0]) > 1
        truncated = self.time > self.duration
        return obs, reward, done, truncated, {}

# STEP 4: Setup the training session
async def main():
    # Launch ProtoTwin Connect and load the cartpole-v2 model
    path = os.path.join(os.path.dirname(__file__), "cartpole-v2.ptm")
    client = await prototwin.start()
    await client.load(path)

    observation_high = np.array([np.finfo(np.float32).max] * observation_size, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    action_high = np.array([max_cart_position] * action_size, dtype=np.float32)
    action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    # Create callback to regularly save the model
    save_freq = 5000
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs/checkpoints/", name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)

    # Define learning rate schedule
    def lr_schedule(progress_remaining):
        initial_lr = 0.003
        return initial_lr * (progress_remaining ** 2)
    
    # Create the vectorized environment
    entity_name = "Main"
    num_envs = 100
    pattern = prototwin.Pattern.GRID
    spacing = 1
    env = VecEnv(CartPoleEnv, client, entity_name, num_envs, observation_space, action_space, pattern=pattern, spacing=spacing)
    env = VecMonitor(env) # Monitor the training progress

    # Define the ML model
    batch_size = 10000
    n_steps = 1000
    ent_coef = 0.0001
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))
    model = PPO(MlpPolicy, env, verbose=1, batch_size=batch_size, n_steps=n_steps, ent_coef=ent_coef, policy_kwargs=policy_kwargs, learning_rate=lr_schedule, tensorboard_log="./tensorboard/", device="cuda")

    # Start training!
    model.learn(total_timesteps=20_000_000, callback=checkpoint_callback)

asyncio.run(main())
