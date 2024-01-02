import os
import sys

import matplotlib.pyplot as plt
from matplotlib import animation
import gym

# env = gym.make('MountainCar-v0',  render_mode='human')
# env = gym.make('MountainCar-v0', render_mode="rgb_array")
env = gym.make('MountainCar-v0',  render_mode="human")
for episode in range(0, 10):
    env.reset()
    print(f'Episode finished after {episode} timesteps')

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
env.close()