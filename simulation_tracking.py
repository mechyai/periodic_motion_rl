import pandas as pd
import numpy as np
import os

import gym
import mujoco_py

from stable_baselines3 import TD3, SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy


# create environment
env = gym.make('HalfCheetah-gait-v0')
# env = gym.make('HalfCheetah-v3')

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
"""
HalfCheetah has pretrained A2C, PPO, SAC, TD3, & TQC  https://github.com/DLR-RM/rl-baselines3-zoo#mujoco-environments
Mean reward performance for HalfCheetah-V3 TQC >> TD3 >= SAC > PPO > A2C https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
"""
# load trained agent from ZIP
agent = TD3.load('Pretrained-Agents/SB3/td3_HalfCheetah-v3.zip', env=env)  # more optional params

# train the agent
# agent.learn(total_timesteps=int(1e5))
# save the agent
# agent.save('td3_HalfCheetah-trained)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)

# inits
frame_skip = 1  # const fixed in other HalfCheetahEnv class (change in half_cheetah_gait_v#.py file)
MDP_TS = 5  # agent should act/observe every 5 sim timesteps
sim_timesteps = 600  # num of total timsteps
start_time = 300  # heuristic from watching simulation
# data collection
state_list = np.empty((sim_timesteps - (start_time - 1), 19))  # properly sized for delayed start data collection
action_list = np.empty((sim_timesteps - (start_time - 1), 7))

# run simulation
obs = env.reset()
for i in range(sim_timesteps):
    if i * frame_skip % MDP_TS == 0:  # only act at 5 sim timestep increments, maintain action for entire segment
        action, _states = agent.predict(obs, deterministic=True)
    # collect data at every simulation timestep, not action, for smoother data
    obs, rewards, dones, info = env.step(action)
    env.render()
    # collect sim data
    if i > start_time - 1:  # delay start to reach steady state motion
        x_pos = info['x_position']  # removed from obs since x_pos is used to compute avg vel, add to state
        state_list[i - start_time + 1] = np.concatenate([[i*frame_skip, x_pos], obs])  # prepend timestep info
        action_list[i - start_time + 1] = np.concatenate([[i*frame_skip], action])  # prepend timestep info

# create data collection dfs
state_labels = ['Timestep',  # units in [m] and [rad]
                  'x_pos', 'z_pos', 'y_pos',  # torso body frame cartesian position
                  'bt_pos', 'bs_pos', 'bf_pos',  # back t(high), s(hin), f(oot) joint angular position
                  'ft_pos', 'fs_pos', 'ff_pos',  # front t(high), s(hin), f(oot) joint angular position
                  'x_vel', 'z_vel', 'y_vel',  # torso body frame cartesian position
                  'bt_vel', 'bs_vel', 'bf_vel',  # back t(high), s(hin), f(oot) joint angular velocity
                  'ft_vel', 'fs_vel', 'ff_vel']  # front t(high), s(hin), f(oot) joint angular velocity

action_labels =['Timestep',  # units in [N-m]
                  'bt_torq', 'bs_torq', 'bf_torq',  # back t(high), s(hin), f(oot) joint torque
                  'ft_torq', 'fs_torq', 'ff_torq']  # front t(high), s(hin), f(oot) joint torque

# convert np array to lits
state_df = pd.DataFrame(state_list, columns=state_labels)
action_df = pd.DataFrame(action_list, columns=action_labels)

save_experiment = True
if save_experiment:
    # output, experiment naming convention
    exp_name = 'smallSample'
    env_ver = 'v0'
    xml_ver = 'v0'
    algo = 'TD3'
    sim_len = (sim_timesteps - start_time)
    # naming
    file_name = f'{exp_name}_{env_ver}_{xml_ver}_{algo}_{sim_len}_{start_time}_{frame_skip}_'
    folder = 'Experimental-Data/'
    # saving experiments, df -> .csv
    state_df.to_pickle(folder + file_name + 'states.csv')
    state_df.to_pickle(folder + file_name + 'actions.csv')






