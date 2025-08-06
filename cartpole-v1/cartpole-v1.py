# STEP 1: Import dependencies
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
cart_motor_target_velocity = 3
cart_motor_current_position = 5
cart_motor_current_velocity = 6
cart_motor_current_force = 7
pole_motor_current_position = 12
pole_motor_current_velocity = 13

# STEP 3: Create your own vectorized environment instance by extending the VecEnvInstance base class
class CartPoleEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.duration = 10
        self.max_cart_position = 0.65

    def normalize_position(self, position):
        '''
        Normalizes the specified position so that it lies in the range [-1, 1]
        '''
        return position / self.max_cart_position

    def normalize_angle(self, angle):
        '''
        Normalizes the specified angle so that it lies in the range [-1, 1]
        '''
        return math.atan2(math.sin(angle), math.cos(math.pi - angle)) / math.pi
    
    def reward_angle(self):
        '''
        Reward the agent for how close the angle of the pole is from the upright position
        '''
        pole_angle = self.normalize_angle(self.get(pole_motor_current_position))
        return 1 - math.fabs(pole_angle)

    def reward_position_penalty(self):
        '''
        Penalize the agent for moving the cart away from the center
        '''
        cart_position = self.get(cart_motor_current_position)
        return math.fabs(self.normalize_position(cart_position))
    
    def reward_force_penalty(self):
        '''
        Penalize the agent high force in the cart's motor
        '''
        force = self.get(cart_motor_current_force)
        return math.fabs(force)

    def reward(self):
        '''
        Reward function
        '''
        dt = 0.01
        reward = 0
        reward += self.reward_angle()
        reward -= self.reward_position_penalty() * 0.5
        reward -= self.reward_force_penalty() * 0.0025
        return reward * dt
    
    def observations(self):
        '''
        The current observations
        '''
        cart_position = self.get(cart_motor_current_position) # Read the current position of the cart
        cart_velocity = self.get(cart_motor_current_velocity) # Read the current velocity of the cart
        pole_angular_velocity = self.get(pole_motor_current_velocity) # Read the current angular velocity of the pole
        pole_angular_distance = self.get(pole_motor_current_position) # Read the current angular distance of the pole
        return np.array([self.normalize_position(cart_position), self.normalize_angle(pole_angular_distance) / math.pi, cart_velocity, pole_angular_velocity])

    def reset(self, seed = None):
        super().reset(seed=seed)
        return self.observations(), {}
    
    def apply(self, action):
        self.set(cart_motor_target_velocity, action[0]) # Apply action (set the target velocity of the cart)

    def step(self):
        obs = self.observations()
        reward = self.reward()
        done = abs(obs[0]) > 1
        truncated = self.time > self.duration
        return obs, reward, done, truncated, {}

# STEP 4: Setup the training session
async def main():
    # Launch ProtoTwin Connect and load the cartpole-v1 model
    client = await prototwin.start()
    await client.load("cartpole-v1.ptm")

    # The observation space contains:
    # 0. A measure of the cart's distance from the center, where 0 is at the center and +/-1 is at the limit
    # 1. A measure of the pole's angular distance from the upright position, where 0 is at the upright position and +/-1 is at the down position
    # 2. The cart's current linear velocity (m/s)
    # 3. The pole's current angular velocity (rad/s)
    observation_high = np.array([1.0, 1.0, 1.0, np.finfo(np.float32).max], dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    # The action space contains:
    # 0. The cart's target velocity (m/s)
    action_high = np.array([1.0], dtype=np.float32)
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
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))
    model = PPO(MlpPolicy, env, verbose=1, batch_size=batch_size, n_steps=n_steps, policy_kwargs=policy_kwargs, learning_rate=lr_schedule, tensorboard_log="./tensorboard/", device="cuda")

    # Start training!
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

asyncio.run(main())
