# if __name__ == "__main__":
#     """
#     Script to test functionality of the 6DOF environment.
#     """
#
#     from src.rocket_env.envs import Rocket6DOF
#     from src.rocket_env.envs import Rocket6DOF_Fins
#     from stable_baselines3 import SAC
#     from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
#     from functools import partial
#
#     def make_env():
#         return Rocket6DOF_Fins(render_mode=None, reward_shaping_type="velocity")
#
#     # Create 4 parallel environments
#     env = SubprocVecEnv([make_env for _ in range(4)])
#     env = VecMonitor(env)
#     obs = env.reset()
#
#
#     model = SAC("MlpPolicy", env=env, verbose=1)
#
#     try:
#         model.learn(total_timesteps=1000000)
#     except KeyboardInterrupt:
#         model.save("interrupted_model")
#
#     model.save("test_model_one")
#     while True:
#         obs, rew, terminated, truncated, info, = env.step(env.action_space.sample())
#         env.render()
#
#         if terminated or truncated:
#             env.reset()
#             env.render()
#
#     env.close()
#
#

# =========================================================================================
#                           MODEL TESTING SCRIPT
# =========================================================================================

import gymnasium as gym
from stable_baselines3 import SAC

from src.rocket_env.envs import Rocket6DOF_Fins
from src.rocket_env.envs.rocket_env import Rocket6DOF

# Create the environment (must match training env setup)
env = Rocket6DOF_Fins(render_mode="human")

# Load the trained model
model = SAC.load("test_model_one_p2", env=env)

while True:
    # Reset environment
    obs, info = env.reset()

    terminated = False
    truncated = False

    # Simulation loop (inference only)
    while not (terminated or truncated):
        # Predict action using trained model
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Optional rendering
        env.render()

        if terminated or truncated:
            obs, info = env.reset()
