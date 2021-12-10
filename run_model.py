import gym
import mujoco_py

# see what MuJoCo Envs and RL Algos are available https://github.com/DLR-RM/rl-baselines3-zoo#mujoco-environments

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
# agent = TD3.load('td3_HalfCheetah-v3', env=env)  # more optional params

# train the agent
# agent.learn(total_timesteps=int(1e5))
# save the agent
# agent.save('td3_HalfCheetah-trained)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    # action, _states = agent.predict(obs, deterministic=True)
    # print(action)
    # obs, rewards, dones, info = env.step(action)
    env.render()
    # obs, rewards, dones, info = env.step(env.action_space.sample())
    # print(info["toe_pos"])

