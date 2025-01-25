"""
Script to test functionality of the 6DOF environment
"""

from rocket_env.envs import Rocket6DOF

def test_rocket_landing(landing_attempts):
    # Instantiate the environment
    env = Rocket6DOF()

    # Initialize the environment
    done = False
    obs = env.reset()
    env.render(mode="human")

    succesful_landings = 0

    for _ in range(landing_attempts):
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render(mode="human")

        if done:
            succesful_landings += 1
            done = False
            env.reset()
            env.render(mode="human")

    print(f"The success rate is:{succesful_landings/landing_attempts*100}%")
    env.close()
