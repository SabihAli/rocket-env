# Rocket SIM 6DOF

This is a Python Gym environment to simulate a 6 degrees of freedom rocket.

<img src="./_images/rocket_trajectory.png" alt="rocket-trajectory" width="80%">

## Features

* full 6DOF rocket landing environment
* realistic dynamics equation modeled on a rigid body assumption
* interactive 3D visualization through PyVista
* actuators available:
    1. thruster
    1. fins
* Wandb logging wrapper

### Continuous action space
The environment employes a continuous action space, with the engine allowed to throttle between `maxThrust` and `minThrust`. The thrust was normalized to lie in the range `[-1, +1]` as best practice for convergence of the algorithms suggest. The engine is gimbaled by two angles $\delta_y$ and $\delta_z$ around two hinge points, respectively moving the engine around the z and y axis.

## Installation Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/rocket-env.git
    cd rocket-env
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Examples

### Minimal Example with PPO

```python
import gym
from stable_baselines3 import PPO
from src.rocket_env.envs import Rocket6DOF

# Create the environment
env = Rocket6DOF()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Evaluate the agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
        obs = env.reset()
```