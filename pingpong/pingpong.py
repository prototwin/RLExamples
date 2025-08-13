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
link5_motor_state = 7
link5_motor_target_position = 8
link5_motor_target_velocity = 9
link5_motor_force_limit = 10
link5_motor_current_position = 11
link5_motor_current_velocity = 12
link5_motor_current_force = 13
link1_motor_state = 14
link1_motor_target_position = 15
link1_motor_target_velocity = 16
link1_motor_force_limit = 17
link1_motor_current_position = 18
link1_motor_current_velocity = 19
link1_motor_current_force = 20
link6_motor_state = 21
link6_motor_target_position = 22
link6_motor_target_velocity = 23
link6_motor_force_limit = 24
link6_motor_current_position = 25
link6_motor_current_velocity = 26
link6_motor_current_force = 27
surface_position_sensor_x = 49
surface_position_sensor_y = 50
surface_position_sensor_z = 51
surface_orientation_sensor_x = 52
surface_orientation_sensor_y = 53
surface_orientation_sensor_z = 54
surface_orientation_sensor_w = 55
surface_linear_velocity_sensor_x = 56
surface_linear_velocity_sensor_y = 57
surface_linear_velocity_sensor_z = 58
surface_angular_velocity_sensor_x = 59
surface_angular_velocity_sensor_y = 60
surface_angular_velocity_sensor_z = 61
link3_motor_state = 28
link3_motor_target_position = 29
link3_motor_target_velocity = 30
link3_motor_force_limit = 31
link3_motor_current_position = 32
link3_motor_current_velocity = 33
link3_motor_current_force = 34
link2_motor_state = 35
link2_motor_target_position = 36
link2_motor_target_velocity = 37
link2_motor_force_limit = 38
link2_motor_current_position = 39
link2_motor_current_velocity = 40
link2_motor_current_force = 41
link4_motor_state = 42
link4_motor_target_position = 43
link4_motor_target_velocity = 44
link4_motor_force_limit = 45
link4_motor_current_position = 46
link4_motor_current_velocity = 47
link4_motor_current_force = 48
ball_position_sensor_x = 1
ball_position_sensor_y = 2
ball_position_sensor_z = 3
ball_linear_velocity_sensor_x = 4
ball_linear_velocity_sensor_y = 5
ball_linear_velocity_sensor_z = 6

torques = [link1_motor_current_force, link2_motor_current_force, link3_motor_current_force, link4_motor_current_force, link5_motor_current_force, link6_motor_current_force]

states = [link1_motor_current_position, link2_motor_current_position, link3_motor_current_position, link4_motor_current_position, link5_motor_current_position, link6_motor_current_position,
          link1_motor_current_velocity, link2_motor_current_velocity, link3_motor_current_velocity, link4_motor_current_velocity, link5_motor_current_velocity, link6_motor_current_velocity,
          surface_position_sensor_x, surface_position_sensor_y, surface_position_sensor_z,
          surface_orientation_sensor_x, surface_orientation_sensor_y, surface_orientation_sensor_z, surface_orientation_sensor_w,
          surface_linear_velocity_sensor_x, surface_linear_velocity_sensor_y, surface_linear_velocity_sensor_z,
          surface_angular_velocity_sensor_x, surface_angular_velocity_sensor_y, surface_angular_velocity_sensor_z,
          ball_position_sensor_x, ball_position_sensor_y, ball_position_sensor_z,
          ball_linear_velocity_sensor_x, ball_linear_velocity_sensor_y, ball_linear_velocity_sensor_z]

actions = [link1_motor_target_position, link2_motor_target_position, link3_motor_target_position, link4_motor_target_position, link5_motor_target_position, link6_motor_target_position]

torque_size = len(torques)
state_size = len(states)
action_size = len(actions)
observation_size = state_size + action_size

ball_x_zero = 0.0003
ball_y_zero = 0.5828
ball_z_zero = 0.2957
bat_x_zero = 0.0003
bat_y_zero = -0.2247
bat_z_zero = 0.3017

# STEP 3: Create your own vectorized environment instance by extending the VecEnvInstance base class
class PingPongEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.duration = 60
        self.previous_action = [0] * action_size
    
    def reward_ball_height(self):
        '''
        Reward the agent for bouncing the ball to a height that is close to the ball's zero position
        '''
        ball_y = self.get(ball_position_sensor_y)
        bat_y = self.get(surface_position_sensor_y)
        if ball_y - bat_y > 0.05:
            ball_y_distance = ball_y - ball_y_zero
            return math.exp(-5 * ball_y_distance * ball_y_distance)
        return 0 # The reward is zero if the ball is resting on the bat
    
    def reward_position_penalty(self):
        '''
        Penalize the agent for moving the bat away from the zero position
        '''
        bat_x = self.get(surface_position_sensor_x)
        bat_y = self.get(surface_position_sensor_y)
        bat_z = self.get(surface_position_sensor_z)
        dx = bat_x - bat_x_zero
        dy = bat_y - bat_y_zero
        dz = bat_z - bat_z_zero
        return dx * dx + dy * dy + dz * dz
    
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
        dt = 0.005
        reward = 0
        reward += self.reward_ball_height()
        reward -= self.reward_position_penalty()
        reward -= self.reward_joint_torque_penalty() * 0.002
        return reward * dt
    
    def terminal(self):
        '''
        Whether the agent is in a terminal state
        '''
        ball_x_min = ball_x_zero - 0.3
        ball_x_max = ball_x_zero + 0.3

        ball_y_min = ball_y_zero - 1.0
        ball_y_max = ball_y_zero + 0.4

        ball_z_min = ball_z_zero - 0.3
        ball_z_max = ball_z_zero + 0.3

        bat_x_min = bat_x_zero - 0.3
        bat_x_max = bat_x_zero + 0.3

        bat_y_min = bat_y_zero - 0.3
        bat_y_max = bat_y_zero + 0.3

        bat_z_min = bat_z_zero - 0.2
        bat_z_max = bat_z_zero + 0.3

        ball_x = self.get(ball_position_sensor_x)
        ball_y = self.get(ball_position_sensor_y)
        ball_z = self.get(ball_position_sensor_z)
        bat_x = self.get(surface_position_sensor_x)
        bat_y = self.get(surface_position_sensor_y)
        bat_z = self.get(surface_position_sensor_z)
        
        return (ball_x < ball_x_min or ball_x > ball_x_max or
                ball_y < ball_y_min or ball_y > ball_y_max or
                ball_z < ball_z_min or ball_z > ball_z_max or
                bat_x < bat_x_min or bat_x > bat_x_max or
                bat_y < bat_y_min or bat_y > bat_y_max or
                bat_z < bat_z_min or bat_z > bat_z_max)
    
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
    # Launch ProtoTwin Connect and load the pingpong model
    path = os.path.join(os.path.dirname(__file__), "pingpong.ptm")
    client = await prototwin.start()
    await client.load(path)

    observation_high = np.array([np.finfo(np.float32).max] * observation_size, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    action_high = np.array([0.6] * action_size, dtype=np.float32)
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
    env = VecEnv(PingPongEnv, client, entity_name, num_envs, observation_space, action_space, pattern=pattern, spacing=spacing)
    env = VecMonitor(env) # Monitor the training progress

    # Define the ML model
    batch_size = 8000
    n_steps = 4000
    ent_coef = 0.0003
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
    model = PPO(MlpPolicy, env, verbose=1, batch_size=batch_size, n_steps=n_steps, ent_coef=ent_coef, policy_kwargs=policy_kwargs, learning_rate=lr_schedule, tensorboard_log="./tensorboard/", device="cuda")

    # Start training!
    model.learn(total_timesteps=100_000_000, callback=checkpoint_callback)

asyncio.run(main())
