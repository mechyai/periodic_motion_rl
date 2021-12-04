import mujoco_py
import gym
import os

env = gym.make('HalfCheetah-gait-v0')
# try from https://gym.openai.com/envs/

env.reset()
# Rendering our instance 300 times
for _ in range(250):
  #renders the environment
  env.render()
  #Takes a random action from its action space
  # aka the number of unique actions an agent can perform
  action = env.action_space.sample()

  obs, reward, done, info = env.step(action)

  print(f'\n\nObservation:\n', obs)
  print(f'Actuation:\n', action)

env.close()