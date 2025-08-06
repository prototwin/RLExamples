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
left_calf_motor_state = 1
left_calf_motor_target_position = 2
left_calf_motor_target_velocity = 3
left_calf_motor_force_limit = 4
left_calf_motor_current_position = 5
left_calf_motor_current_velocity = 6
left_calf_motor_current_force = 7
right_hip_motor_state = 8
right_hip_motor_target_position = 9
right_hip_motor_target_velocity = 10
right_hip_motor_force_limit = 11
right_hip_motor_current_position = 12
right_hip_motor_current_velocity = 13
right_hip_motor_current_force = 14
left_hip_motor_state = 15
left_hip_motor_target_position = 16
left_hip_motor_target_velocity = 17
left_hip_motor_force_limit = 18
left_hip_motor_current_position = 19
left_hip_motor_current_velocity = 20
left_hip_motor_current_force = 21
right_thigh_motor_state = 22
right_thigh_motor_target_position = 23
right_thigh_motor_target_velocity = 24
right_thigh_motor_force_limit = 25
right_thigh_motor_current_position = 26
right_thigh_motor_current_velocity = 27
right_thigh_motor_current_force = 28
right_foot_motor_state = 29
right_foot_motor_target_position = 30
right_foot_motor_target_velocity = 31
right_foot_motor_force_limit = 32
right_foot_motor_current_position = 33
right_foot_motor_current_velocity = 34
right_foot_motor_current_force = 35
right_foot_position_sensor_x = 36
right_foot_position_sensor_y = 37
right_foot_position_sensor_z = 38
right_foot_linear_velocity_sensor_x = 39
right_foot_linear_velocity_sensor_y = 40
right_foot_linear_velocity_sensor_z = 41
right_calf_motor_state = 42
right_calf_motor_target_position = 43
right_calf_motor_target_velocity = 44
right_calf_motor_force_limit = 45
right_calf_motor_current_position = 46
right_calf_motor_current_velocity = 47
right_calf_motor_current_force = 48
body_orientation_sensor_x = 49
body_orientation_sensor_y = 50
body_orientation_sensor_z = 51
body_orientation_sensor_w = 52
body_linear_velocity_sensor_x = 53
body_linear_velocity_sensor_y = 54
body_linear_velocity_sensor_z = 55
body_angular_velocity_sensor_x = 56
body_angular_velocity_sensor_y = 57
body_angular_velocity_sensor_z = 58
body_position_sensor_x = 59
body_position_sensor_y = 60
body_position_sensor_z = 61
left_thigh_motor_state = 62
left_thigh_motor_target_position = 63
left_thigh_motor_target_velocity = 64
left_thigh_motor_force_limit = 65
left_thigh_motor_current_position = 66
left_thigh_motor_current_velocity = 67
left_thigh_motor_current_force = 68
back_neck_motor_state = 69
back_neck_motor_target_position = 70
back_neck_motor_target_velocity = 71
back_neck_motor_force_limit = 72
back_neck_motor_current_position = 73
back_neck_motor_current_velocity = 74
back_neck_motor_current_force = 75
left_foot_motor_state = 76
left_foot_motor_target_position = 77
left_foot_motor_target_velocity = 78
left_foot_motor_force_limit = 79
left_foot_motor_current_position = 80
left_foot_motor_current_velocity = 81
left_foot_motor_current_force = 82
left_foot_position_sensor_x = 83
left_foot_position_sensor_y = 84
left_foot_position_sensor_z = 85
left_foot_linear_velocity_sensor_x = 86
left_foot_linear_velocity_sensor_y = 87
left_foot_linear_velocity_sensor_z = 88

current_positions = [left_hip_motor_current_position, right_hip_motor_current_position,
                     left_thigh_motor_current_position, right_thigh_motor_current_position,
                     left_calf_motor_current_position, right_calf_motor_current_position,
                     left_foot_motor_current_position, right_foot_motor_current_position,
                     back_neck_motor_current_position]

torques = [left_hip_motor_current_force, right_hip_motor_current_force,
           left_thigh_motor_current_force, right_thigh_motor_current_force,
           left_calf_motor_current_force, right_calf_motor_current_force,
           left_foot_motor_current_force, right_foot_motor_current_force,
           back_neck_motor_current_force]

states = [left_hip_motor_current_position, right_hip_motor_current_position,
          left_thigh_motor_current_position, right_thigh_motor_current_position,
          left_calf_motor_current_position, right_calf_motor_current_position,
          left_foot_motor_current_position, right_foot_motor_current_position,
          back_neck_motor_current_position,
          left_hip_motor_current_velocity, right_hip_motor_current_velocity,
          left_thigh_motor_current_velocity, right_thigh_motor_current_velocity,
          left_calf_motor_current_velocity, right_calf_motor_current_velocity,
          left_foot_motor_current_velocity, right_foot_motor_current_velocity,
          back_neck_motor_current_velocity,
          body_orientation_sensor_x, body_orientation_sensor_y, body_orientation_sensor_z, body_orientation_sensor_w,
          body_linear_velocity_sensor_x, body_linear_velocity_sensor_y, body_linear_velocity_sensor_z,
          body_angular_velocity_sensor_x, body_angular_velocity_sensor_y, body_angular_velocity_sensor_z,
          left_foot_position_sensor_y, right_foot_position_sensor_y,
          body_position_sensor_x]

actions = [left_hip_motor_target_position, right_hip_motor_target_position,
           left_thigh_motor_target_position, right_thigh_motor_target_position,
           left_calf_motor_target_position, right_calf_motor_target_position,
           left_foot_motor_target_position, right_foot_motor_target_position,
           back_neck_motor_target_position]

current_position_size = len(current_positions)
torque_size = len(torques)
state_size = len(states)
action_size = len(actions)
observation_size = state_size + action_size

# STEP 3: Create your own vectorized environment instance by extending the VecEnvInstance base class
class BipedalEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.duration = 20
        self.target_velocity = 0.5
        self.previous_action = [0] * action_size
    
    def reward_forward_velocity(self):
        '''
        Reward the agent for moving with a forward velocity close to the target velocity
        '''
        diff = self.get(body_linear_velocity_sensor_z) - self.target_velocity
        return math.exp(-8 * diff * diff)
    
    def reward_angular_velocity_penalty(self):
        '''
        Penalize the agent for moving with a large angular velocity
        '''
        vx = self.get(body_angular_velocity_sensor_x)
        vy = self.get(body_angular_velocity_sensor_y)
        vz = self.get(body_angular_velocity_sensor_z)
        return vx * vx + vy * vy + vz * vz
    
    def reward_action_rate_penalty(self, action):
        '''
        Penalize the agent for taking two very different actions
        '''
        penalty = 0
        for i in range(action_size):
            q = action[i] - self.previous_action[i]
            penalty += q * q
        return penalty / action_size
    
    def reward_joint_torque_penalty(self):
        '''
        Penalize the agent for high torques in the joints
        '''
        penalty = 0
        for i in range(torque_size):
            torque = self.get(torques[i])
            penalty += torque * torque
        return penalty / torque_size
    
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
    
    def reward_position_drift_penalty(self):
        '''
        Penalize the agent for positional drift
        '''
        return math.fabs(self.get(body_position_sensor_x))
    
    def reward_rotation_drift_penalty(self):
        '''
        Penalize the agent for rotational drift
        '''
        qx = self.get(body_orientation_sensor_x)
        qy = self.get(body_orientation_sensor_y)
        qz = self.get(body_orientation_sensor_z)
        qw = self.get(body_orientation_sensor_w)
        dot = np.dot([qx, qy, qz, qw], [0, 0, 0, 1])
        arg = np.clip(2.0 * dot * dot - 1.0, -1.0, 1.0)
        return np.arccos(arg) / np.pi
    
    def reward(self):
        '''
        Reward function
        '''
        dt = 0.01

        reward = 0
        reward += self.reward_forward_velocity()

        penalty = 0
        penalty += self.reward_angular_velocity_penalty() * 0.01
        penalty += self.reward_action_rate_penalty(self.action) * 0.1
        penalty += self.reward_joint_torque_penalty() * 0.002
        penalty += self.reward_joint_limit_penalty() * 0.5
        penalty += self.reward_position_drift_penalty() * 2.0
        penalty += self.reward_rotation_drift_penalty() * 0.5
        penalty *= min(1, reward)

        return (reward - penalty) * dt
    
    def terminal(self):
        '''
        Whether the agent is in a terminal state
        '''
        return self.get(body_position_sensor_y) < -0.01 or math.fabs(self.get(body_position_sensor_x)) > 2
    
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

    def reset(self, seed=None):
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
    # Launch ProtoTwin Connect and load the bipedal model
    client = await prototwin.start()
    await client.load("bipedal.ptm")

    observation_high = np.array([np.finfo(np.float32).max] * observation_size, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)

    action_high = np.array([0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3], dtype=np.float32)
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
    pattern = prototwin.Pattern.LINEAR_X
    spacing = 0.5
    env = VecEnv(BipedalEnv, client, entity_name, num_envs, observation_space, action_space, pattern=pattern, spacing=spacing)
    env = VecMonitor(env) # Monitor the training progress

    # Define the ML model
    batch_size = 10000
    n_steps = 500
    ent_coef = 0.0002
    policy_kwargs = dict(activation_fn=torch.nn.ELU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
    model = PPO(MlpPolicy, env, verbose=1, batch_size=batch_size, n_steps=n_steps, ent_coef=ent_coef, policy_kwargs=policy_kwargs, learning_rate=lr_schedule, tensorboard_log="./tensorboard/", device="cuda")

    # Start training!
    model.learn(total_timesteps=100_000_000, callback=checkpoint_callback)

asyncio.run(main())
