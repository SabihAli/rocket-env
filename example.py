"""
Script to test functionality of the 6DOF environment.
"""

from rocket_env.envs import Rocket6DOF

# Instantiate the environment
env = Rocket6DOF()

done = False

# Initialize the environment
obs = env.reset()
env.render(mode="human")

while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render(mode="human")
    
    if done:
        env.reset()
        env.render(mode="human")

env.close()