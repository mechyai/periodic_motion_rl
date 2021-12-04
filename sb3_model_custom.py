import gym
import mujoco_py

# see what MuJoCo Envs and RL Algos are available https://github.com/DLR-RM/rl-baselines3-zoo#mujoco-environments

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# create environment
env = gym.make('HalfCheetah-gait-v0')
# try from https://gym.openai.com/envs/

# instantiate agent
agent = PPO('MlpPolicy', env, verbose=5)
# train the agent
agent.learn(total_timesteps=int(1e5))
# save the agent
agent.save('ppo_cheetah_test')
del agent # to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
agent = PPO.load('ppo_cheetah_test', env=env)


# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = agent.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()