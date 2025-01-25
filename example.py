"""
Script to test functionality of the 6DOF environment.
"""

from rocket_env.envs import Rocket6DOF

# Initialize the environment
env = Rocket6DOF(render_mode="human")  # or "rgb_array"
obs, info = env.reset(seed=123)

while True:
    obs, rew, terminated, truncated, info, = env.step(env.action_space.sample())
    env.render()
    
    if terminated or truncated:
        env.reset()
        env.render()

env.close()