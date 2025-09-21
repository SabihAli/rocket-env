import numpy as np
import logging

import gymnasium as gym
from gymnasium import spaces

from .rocket_env import Rocket6DOF


class Rocket6DOF_Fins(Rocket6DOF):
    def __init__(
        self,
        render_mode=None,
        IC=[600, 100, 100, -100, 0, 0, 1, 0, 0, 0, 0, 0, 0, 50000],
        ICRange=[100, 50, 50, 20, 5, 5, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 2000],
        timestep=0.1,
        reward_shaping_type="acceleration",
        reward_coeff={
            "alfa": -0.01,
            "beta": -1e-8,
            "eta": 2,
            "gamma": -10,
            "delta": -5,
            "kappa": 10,
            "w_r_f": 1,
            "w_v_f": 5,
            "max_r_f": 100,
            "max_v_f": 100,
        },
        trajectory_limits={"attitude_limit": [85, 85, 360]},
        landing_params={
            "landing_radius": 30,
            "maximum_velocity": 15,
            "landing_attitude_limit": [10, 10, 360],
            "omega_lim": [0.2, 0.2, 0.2],
            "waypoint": 50,
        },
    ):
        super().__init__(
            render_mode=render_mode,
            IC=IC,
            ICRange=ICRange,
            timestep=timestep,
            reward_shaping_type=reward_shaping_type,
            reward_coeff=reward_coeff,
            trajectory_limits=trajectory_limits,
            landing_params=landing_params,
        )

        # Append fins action names
        self.action_names.extend(
            ["beta_fin_1", "beta_fin_2", "beta_fin_3", "beta_fin_4"]
        )

        # Grid fins bounds
        self.max_fins_gimbal = np.deg2rad(90)

        # Redefine action space (7D: 3 thruster + 4 fins)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Reinitialize the action
        self.action = np.zeros(7, dtype=np.float32)

    def _denormalize_action(self, action):
        """
        Denormalize thruster commands and fins. 
        The first 3 are thruster-related (handled by super()), the last 4 are fins.
        """
        thruster_action = super()._denormalize_action(action)

        # action[3:7] -> fins in [-1, +1], scaled to [-max_fins_gimbal, +max_fins_gimbal]
        fins_action = np.array(action[3:7]) * self.max_fins_gimbal
        denormalized_action = np.concatenate((thruster_action, fins_action))
        assert denormalized_action.shape == (7,)
        return denormalized_action

    def step(self, normalized_action):
        """
        Step the environment, returning (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = super().step(normalized_action)

        # Add additional info if desired
        info["euler_angles"] = self.rotation_obj.as_euler("zyx", degrees=True)

        return obs, reward, terminated, truncated, info