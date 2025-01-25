__all__ = [
    "RewardAnnealing",
    "EpisodeAnalyzer",
    "RemoveMassFromObs",
    "VerticalAttitudeReward",
]

import numpy as np
import logging

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import Env

from matplotlib import pyplot as plt

import wandb
import pandas as pd

pd.options.plotting.backend = "plotly"


class RewardAnnealing(gym.Wrapper):
    def __init__(self, env: gym.Env, thrust_penalty: float = 0.01) -> None:
        super().__init__(env)
        self.xi = self.env.unwrapped.reward_coefficients.get("xi", thrust_penalty)

    def step(self, action):
        # Call parent step, which returns (obs, reward, terminated, truncated, info)
        obs, _, terminated, truncated, info = super().step(action)

        old_rewards_dict = info["rewards_dict"]
        # Only keep selected terms from old rewards dict
        new_rewards_keys = [
            "attitude_constraint",
            "goal_conditions",
            "final_position",
            "final_velocity",
        ]
        rewards_dict = {key: old_rewards_dict[key] for key in new_rewards_keys if key in old_rewards_dict}

        # Apply a new thrust penalty shaping
        # (action[2] in [-1, +1], so we do -( xi * (action[2] + 1) ) as penalty)
        rewards_dict["thrust_penalty"] = -self.xi * (action[2] + 1)

        # Final shaped reward
        reward = sum(rewards_dict.values())

        # Update info
        info["rewards_dict"] = rewards_dict

        return obs, reward, terminated, truncated, info


class EpisodeAnalyzer(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.rewards_info = []

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)

        sim_time = self.env.unwrapped.SIM.t  # Current sim time
        self.rewards_info.append({**info["rewards_dict"], "time": sim_time})

        # If the episode ended (either terminated or truncated)
        if terminated or truncated:
            fig = self.env.unwrapped.get_trajectory_plotly()
            states_dataframe = self.env.unwrapped.states_to_dataframe()
            actions_dataframe = self.env.unwrapped.actions_to_dataframe()

            rewards_dataframe = pd.DataFrame(self.rewards_info)
            shaping_type = self.env.unwrapped.shaping_type
            if shaping_type == "velocity":
                shaper_dataframe = self.env.unwrapped.vtarg_to_dataframe()
                shaper_name = "ep_history/atarg"
            elif shaping_type == "acceleration":
                shaper_dataframe = self.env.unwrapped.atarg_to_dataframe()
                shaper_name = "ep_history/atarg"
            else:
                # Fallback
                shaper_dataframe = pd.DataFrame()
                shaper_name = "ep_history/shaper"

            names = self.env.unwrapped.state_names
            final_state = states_dataframe.iloc[-1, :]
            values = np.abs(final_state)
            final_errors_dict = {"final_errors/" + n: v for n, v in zip(names, values)}

            if wandb.run is not None:
                wandb.log(
                    {
                        "ep_history/states": states_dataframe.plot(),
                        "ep_history/actions": actions_dataframe.plot(),
                        shaper_name: shaper_dataframe.plot() if not shaper_dataframe.empty else None,
                        "ep_history/rewards": rewards_dataframe.drop("time", axis=1).plot(),
                        "plots3d/atarg_trajectory": self.env.unwrapped.get_atarg_plotly(),
                        "plots3d/trajectory": fig,
                        "ep_statistic/used_mass": (
                            states_dataframe.iloc[0, -1] - states_dataframe.iloc[-1, -1]
                        ),
                        **final_errors_dict,
                    }
                )
            else:
                # In case wandb is not used, display the final 3D plot
                fig.show()

            # Reset after logging
            self.rewards_info = []

        return obs, rew, terminated, truncated, info


# Simple ObservationWrapper example to remove the last dimension (mass) from the observation
class RemoveMassFromObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # If original obs is shape (14,), remove one dimension -> (13,)
        self.observation_space = Box(low=-1, high=1, shape=(13,), dtype=np.float32)

    def observation(self, obs):
        return obs[0:13]


class VerticalAttitudeReward(gym.Wrapper):
    def __init__(self, env: Env, threshold_height=1e-3, weight=-0.5) -> None:
        super().__init__(env)
        self.threshold_height = threshold_height
        self.reward_weight = weight

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)

        state = self.env.unwrapped.get_state()
        x = state[0]  # 'x' coordinate

        # If we are near the ground and have some positive final velocity shaping
        if x < self.threshold_height and info["rewards_dict"]["final_velocity"] > 0:
            # Extract quaternion portion
            q = state[6:10]

            # Angular deviation from vertical. For a quaternion [q0, q1, q2, q3] with q0 ~ cos(theta/2)
            # degrees of deviation from perfectly vertical
            vertical_deviation_deg = np.degrees(np.arccos(q[0])) * 2
            vertical_attitude_rew = np.clip(
                vertical_deviation_deg * self.reward_weight,
                a_min=-10,
                a_max=+10,
            )

            rew += vertical_attitude_rew
            info["rewards_dict"]["vertical_attitude_reward"] = vertical_attitude_rew

        # Default to zero if it wasn't set
        info["rewards_dict"].setdefault("vertical_attitude_reward", 0)

        return obs, rew, terminated, truncated, info