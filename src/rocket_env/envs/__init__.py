from gym.envs.registration import register

from .rocket_env import Rocket6DOF
from .rocket_env_fins import Rocket6DOF_Fins

# Register the environment
register(
    id='my_environment/Falcon6DOF-v1',
    entry_point='my_environment.envs:Rocket6DOF_Fins'
)
