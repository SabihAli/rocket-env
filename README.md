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

<!---

% TODO: add installation instructions

% TODO: add demo instructions

-->
