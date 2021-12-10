import mujoco_py
import gym
import os

# os.system('source ~/.bashrc')
# os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libGLEW.so'
# os.environ['LD_LIBRARY_PATH'] = '/home/jpc/.mujoco/mujoco210/bin'
# os.environ['LD_LIBRARY_PATH'] = '/usr/lib/nvidia'


env = gym.make('HandManipulatePen-v0')
# try from https://gym.openai.com/envs/

env.reset()
# Rendering our instance 300 times
for _ in range(500):
  #renders the environment
  env.render()
  #Takes a random action from its action space
  # aka the number of unique actions an agent can perform
  env.step(env.action_space.sample())

env.close()