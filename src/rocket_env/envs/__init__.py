import gymnasium as gym
from gymnasium.envs.registration import register

from .rocket_env import Rocket6DOF
from .rocket_env_fins import Rocket6DOF_Fins

# Register the environment
register(
    id="my_environment/Falcon6DOF-v1",
    entry_point="my_environment.envs:Rocket6DOF_Fins",
    # Optionally, you can specify kwargs, max_episode_steps, etc. here.
    # kwargs={"render_mode": "human"}  # Example
    # max_episode_steps=1000
)