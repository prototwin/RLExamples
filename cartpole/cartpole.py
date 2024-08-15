# STEP 1: Import dependencies
import asyncio
import os
import torch
import numpy as np
import math
import gymnasium
import prototwin
import stable_baselines3.ppo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from prototwin_gymnasium import VecEnvInstance, VecEnv

# STEP 2: Define signal addresses (obtain these values from ProtoTwin)
address_cart_target_velocity = 3
address_cart_position = 5
address_cart_velocity = 6
address_cart_force = 7
address_pole_angle = 12
address_pole_angular_velocity = 13

# STEP 3: Create your vectorized instance environment by extending the base environment
class CartPoleEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int) -> None:
        super().__init__(client, instance)
        self.dt = 0.01 # Time step
        self.x_threshold = 0.65 # Maximum cart distance

    def reward(self, obs):
        distance = 1 - math.fabs(obs[0]) # How close the cart is to the center
        angle = 1 - math.fabs(obs[1]) # How close the pole is to the upright position
        force = math.fabs(self.get(address_cart_force)) # How much force is being applied to drive the cart's motor
        reward = angle * 0.8 + distance * 0.2 - force * 0.004
        return max(reward * self.dt, 0)
    
    def observations(self):
        cart_position = self.get(address_cart_position) # Read the current cart position
        cart_velocity = self.get(address_cart_velocity) # Read the current cart velocity
        pole_angle = self.get(address_pole_angle) # Read the current pole angle
        pole_angular_velocity = self.get(address_pole_angular_velocity) # Read the current pole angular velocity
        pole_angular_distance = math.atan2(math.sin(pole_angle), math.cos(math.pi - pole_angle)) # Calculate angular distance from upright position
        return np.array([cart_position / self.x_threshold, pole_angular_distance / math.pi, cart_velocity, pole_angular_velocity])

    def reset(self, seed = None):
        super().reset(seed=seed)
        return self.observations(), {}
    
    def apply(self, action):
        self.set(address_cart_target_velocity, action[0]) # Apply action by setting the cart's target velocity

    def step(self):
        obs = self.observations()
        reward = self.reward(obs) # Calculate reward
        done = abs(obs[0]) > 1 # Terminate if cart goes beyond limits
        truncated = self.time > 10 # Truncate after 10 seconds
        return obs, reward, done, truncated, {}

# STEP 4: Setup the training session
async def main():
    # Start ProtoTwin Connect
    client = await prototwin.start()

    # Load the ProtoTwin model
    filepath = os.path.join(os.path.dirname(__file__), "CartPole.ptm")
    await client.load(filepath)

    # Create the vectorized environment
    entity_name = "Main"
    num_envs = 64

    # The observation space contains:
    # 0. A measure of the cart's distance from the center, where 0 is at the center and +/-1 is at the limit
    # 1. A measure of the pole's angular distance from the upright position, where 0 is at the upright position and +/-1 is at the down position
    # 2. The cart's current velocity (m/s)
    # 3. The pole's angular velocity (rad/s)
    observation_high = np.array([1, 1, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    # The action space contains only the cart's target velocity
    action_high = np.array([1.0], dtype=np.float32)
    action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    env = VecEnv(CartPoleEnv, client, entity_name, num_envs, observation_space, action_space)
    monitored = VecMonitor(env) # Monitor the training progress

    # Create callback to regularly save the model
    save_freq = 10000 # Number of timesteps per instance
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs/checkpoints/", name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)

    # Define learning rate schedule
    def lr_schedule(progress_remaining):
        initial_lr = 0.003
        return initial_lr * (progress_remaining ** 2)

    # Define the ML model
    model = PPO(stable_baselines3.ppo.MlpPolicy, monitored, device=torch.cuda.current_device(), verbose=1, batch_size=4096, n_steps=1000, learning_rate=lr_schedule, tensorboard_log="./tensorboard/")

    # Start training!
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

asyncio.run(main())