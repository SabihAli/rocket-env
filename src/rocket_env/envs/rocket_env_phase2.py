import numpy as np
# import pyvista as pv
import gymnasium as gym
from gymnasium import spaces
from numpy.typing import ArrayLike
from pandas import DataFrame
from scipy.spatial.transform.rotation import Rotation as R

from ..utils.simulator import Simulator6DOF


class Rocket6DOF_Phase2(gym.Env):
    """
    Rocket environment with 6DOF dynamics.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1 / 0.1,  # default if you want ~10 FPS
    }

    def __init__(
        self,
        render_mode=None,
        IC=[1500, 300, 300, 150, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5e4],
        ICRange=[200, 100, 100, 50, 50, 50, -0.0474, 0.1768, 0.3062, 0.9340, 0.1, 0.1, 0.1, 1e3],
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
            "landing_attitude_limit": [10, 10, 360],  # [Yaw, Pitch, Roll]
            "omega_lim": [0.2, 0.2, 0.2],
            "waypoint": 50,
        },
    ) -> None:
        """
        `render_mode` can be "human", "rgb_array", or None.
        """
        super().__init__()

        self.render_mode = render_mode  # Must store this for the new render API

        self.state_names = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "q0",
            "q1",
            "q2",
            "q3",
            "omega1",
            "omega2",
            "omega3",
            "mass",
        ]
        self.action_names = ["gimbal_y", "gimbal_z", "thrust"]

        # Initial conditions mean values and ± range
        self.ICMean = np.float32(IC)
        self.ICRange = np.float32(ICRange)
        self.timestep = timestep
        self.metadata["render_fps"] = 1 / timestep
        self.reward_coefficients = reward_coeff

        # Space from which we sample initial conditions
        self.init_space = spaces.Box(
            low=self.ICMean - self.ICRange / 2,
            high=self.ICMean + self.ICRange / 2,
            dtype=np.float32,
        )

        # Actuators bounds
        self.max_gimbal = np.deg2rad(20)  # [rad]
        self.max_thrust = 981e3  # [N]

        # State normalizer and bounds
        t_free_fall = (
            -self.ICMean[3]
            + np.sqrt(self.ICMean[3] ** 2 + 2 * 9.81 * self.ICMean[0])
        ) / 9.81
        inertia = 6.04e6
        lever_arm = 15.0

        omega_max = (
            self.max_thrust
            * np.sin(self.max_gimbal)
            * lever_arm
            / (inertia)
            * t_free_fall
            / 5.0
        )
        v_max = 2 * 9.81 * t_free_fall

        self.state_normalizer = np.maximum(
            np.array(
                [
                    1.2 * abs(self.ICMean[0]),
                    1.5 * abs(self.ICMean[1]),
                    1.5 * abs(self.ICMean[2]),
                    v_max,
                    v_max,
                    v_max,
                    1.1,
                    1.1,
                    1.1,
                    1.1,
                    omega_max,
                    omega_max,
                    omega_max,
                    self.ICMean[13] + self.ICRange[13],
                ]
            ),
            1,
        )

        # Set environment bounds (for x, y, z)
        position_bounds_high = 0.9 * np.maximum(self.state_normalizer[0:3], 200)
        position_bounds_low = -0.9 * np.maximum(self.state_normalizer[1:3], 200)
        position_bounds_low = np.insert(position_bounds_low, 0, -30)
        self.position_bounds_space = spaces.Box(
            low=position_bounds_low, high=position_bounds_high, dtype=np.float32
        )

        # Define observation space (always normalized)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)

        # Action space: gimbal_y, gimbal_z, thrust ∈ [-1, +1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Environment state variable and simulator object
        self.state = None
        self.infos = []
        self.SIM: Simulator6DOF = None
        self.rotation_obj: R = None
        self.action = np.array([0.0, 0.0, 0.0])
        self.atarg_history = []
        self.vtarg_history = []

        # Trajectory constraints
        self.attitude_traj_limit = np.deg2rad(trajectory_limits["attitude_limit"])

        # Landing parameters
        self.target_r = landing_params["landing_radius"]
        self.maximum_v = landing_params["maximum_velocity"]
        self.landing_target = [0, 0, 0]
        self.landing_attitude_limit = np.deg2rad(
            landing_params["landing_attitude_limit"]
        )
        self.omega_lim = np.array(landing_params["omega_lim"])
        self.waypoint = landing_params["waypoint"]

        # Renderer variables (pyvista)
        self.rocket_body_mesh = None
        self.landing_pad_mesh = None
        self.plotter = None

        # Reward function type
        self.shaping_type = reward_shaping_type

    def reset(self, seed=None, options=None):
        """
        The reset method for Gymnasium-style environments.
        Returns (observation, info).
        """
        super().reset(seed=seed)  # This will set self.np_random

        self.atarg_history = []
        self.vtarg_history = []

        # Seed the init_space using self.np_random, so sampling is reproducible
        if self.init_space is not None and hasattr(self.init_space, "seed"):
            # seed the Box space with an integer from our RNG
            space_seed = int(self.np_random.integers(2**31 - 1))
            self.init_space.seed(space_seed)

        self.initial_condition = self.init_space.sample()

        # Normalize the quaternion portion
        norm_quat = np.linalg.norm(self.initial_condition[6:10])
        if norm_quat < 1e-8:
            # Avoid dividing by near zero, fallback to identity orientation
            self.initial_condition[6:10] = np.array([1, 0, 0, 0])
        else:
            self.initial_condition[6:10] /= norm_quat

        self.state = self.initial_condition

        # Create rotation object from quaternion
        self.rotation_obj = R.from_quat(self._scipy_quat_convention(self.state[6:10]))

        # Instantiate the simulator object
        self.SIM = Simulator6DOF(self.initial_condition, self.timestep, enable_fins=True)

        # Return normalized observation and empty info (or custom info if needed)
        return self._get_obs(), {}

    def step(self, normalized_action: ArrayLike):
        # Convert normalized action to actual physical values
        self.action = self._denormalize_action(normalized_action)

        # Simulate the next state
        next_state, isterminal = self.SIM.step(self.action, integration_method="RK45")
        self.state = next_state.astype(np.float32)

        # Update rotation object
        self.prev_rotation_obj = self.rotation_obj
        self.rotation_obj = R.from_quat(self._scipy_quat_convention(self.state[6:10]))

        # Check termination or bounding violation
        terminated = bool(isterminal) or self._check_bounds_violation(self.state)
        truncated = False  # No explicit "time limit" or other truncation in this env

        # Compute reward
        reward, rewards_dict = self._compute_reward(self.state, self.action)

        # Info dict
        info = {
            "rewards_dict": rewards_dict,
            "state_history": self.SIM.states,
            "action_history": self.SIM.actions,
            "timesteps": self.SIM.times,
            "bounds_violation": self._check_bounds_violation(self.state),
        }
        info["is_done"] = terminated

        # Penalty for bounds violation
        if info["bounds_violation"]:
            reward -= 50.0

        return self._get_obs(), reward, terminated, truncated, info

    # def render(self):
    #     """
    #     Render function for Gymnasium environment.
    #     Uses self.render_mode to decide how to render:
    #       - 'human' displays on-screen
    #       - 'rgb_array' returns an image (numpy array)
    #     If render_mode is None, do nothing.
    #     """
    #     if self.render_mode is None:
    #         return

    #     if self.plotter is None:
    #         # Setup plotter if it's not already
    #         kwargs = {}
    #         if self.render_mode == "rgb_array":
    #             kwargs["off_screen"] = True

    #         self.plotter = pv.Plotter(**kwargs)
    #         self.plotter.show_axes()
    #         self._add_meshes_to_plotter(resetting=True)
    #         # Set camera
    #         self.plotter.camera_position = [
    #             (2.0e03, 1.0e01, -5.0e03),
    #             (1.0e03, -1.0e02, 4.5e02),
    #             (1, 0, 0),
    #         ]
    #         self.plotter.show(auto_close=False, interactive=False)

    #     # Remove old rocket and thrust vector
    #     self.plotter.remove_actor(["thrust_vector", "rocket_body"], render=False)
    #     self._add_meshes_to_plotter()

    #     self.plotter.update()

    #     # If rgb_array, return the current rendered frame
    #     if self.render_mode == "rgb_array":
    #         return self.plotter.image

    #     # If "human", PyVista tries to update the existing window;
    #     # there's no separate return.

    def close(self) -> None:
        """
        Clean up viewer / plotter.
        """
        super().close()
        pv.close_all()

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _compute_reward(self, state, denormalized_action):
        reward = 0.0

        r = state[0:3]
        v = state[3:6]
        m = state[-1]
        thrust_magnitude = denormalized_action[2]
        coeff = self.reward_coefficients

        # Shaping-based reward
        if self.shaping_type == "acceleration":
            _ = self._compute_atarg(r=np.array(r), v=np.array(v), mass=m)
            thrust_vec = self.SIM.get_thrust_vector_inertial()
            a = thrust_vec / m
            a_targ = self.get_atarg()
            shaping_dict = {"atarg_tracking": coeff["alfa"] * np.linalg.norm(a - a_targ)}
        elif self.shaping_type == "velocity":
            v_targ, _ = self._compute_vtarg(r, v)
            shaping_dict = {"vtarg_tracking": coeff["alfa"] * np.linalg.norm(v - v_targ)}
        else:
            shaping_dict = {}

        # Additional terms
        rewards_dict = {
            **shaping_dict,
            "thrust_penalty": coeff["beta"] * thrust_magnitude,
            "eta": coeff["eta"],
            "attitude_constraint": self._check_attitude_limits(),
            **self._reward_goal(state),
        }

        reward = sum(rewards_dict.values())
        return reward, rewards_dict

    def _check_attitude_limits(self):
        gamma = self.reward_coefficients["gamma"]
        attitude_euler_angles = self.rotation_obj.as_euler("zyx")
        # If any angle exceeds the limit, penalty is triggered
        return gamma * np.any(np.abs(attitude_euler_angles) > self.attitude_traj_limit)

    def _reward_goal(self, state):
        r = np.linalg.norm(state[0:3])
        v = np.linalg.norm(state[3:6])
        q = state[6:10]
        omega = state[10:13]
        attitude_euler_angles = self.rotation_obj.as_euler("zyx")
        assert q.shape == (4,) and omega.shape == (3,)

        landing_conditions = {
            "zero_height": state[0] <= 1e-3,
            "velocity_limit": v < self.maximum_v,
            "landing_radius": r < self.target_r,
            "attitude_limit": np.any(
                abs(attitude_euler_angles) < self.landing_attitude_limit
            ),
            "omega_limit": np.any(abs(omega) < self.omega_lim),
        }

        k, w_r_f, w_v_f, max_r_f, max_v_f = map(
            self.reward_coefficients.get,
            ["kappa", "w_r_f", "w_v_f", "max_r_f", "max_v_f"],
        )

        return {
            "goal_conditions": k * all(landing_conditions.values()),
            "final_position": max(max_r_f - r, 0) * w_r_f,
            "final_velocity": (
                max(max_v_f - v, 0) * w_v_f if (r < max_r_f and landing_conditions["zero_height"]) else 0
            ),
        }

    def _normalize_obs(self, obs):
        return (obs / self.state_normalizer).astype(np.float32)

    def _denormalize_action(self, action: ArrayLike):
        gimbal_y = action[0] * self.max_gimbal
        gimbal_z = action[1] * self.max_gimbal
        thrust = (action[2] + 1) / 2.0 * self.max_thrust
        return np.float32([gimbal_y, gimbal_z, thrust])

    def _get_obs(self):
        return self._normalize_obs(self.state)

    def _compute_atarg(self, r, v, mass):
        """
        Compute acceleration target, store in self.atarg_history.
        """
        g = [-9.81, 0, 0]

        def __compute_t_go(r, v) -> float:
            # Solve depressed quartic for t_go
            solutions = np.roots(
                [
                    g[0] ** 2,
                    0,
                    -4 * np.linalg.norm(v) ** 2,
                    -24 * np.dot(r, v),
                    -36 * np.linalg.norm(r) ** 2,
                ]
            )
            real_positive = [val.real for val in solutions if val.imag == 0 and val.real > 0]
            if len(real_positive) == 0:
                return 1.0  # fallback if numerical issues
            return real_positive[0]

        t_go = __compute_t_go(r, v)

        def saturation(q, U) -> np.ndarray:
            q_norm = np.linalg.norm(q)
            if q_norm <= U:
                return q
            else:
                return q * U / q_norm

        a_targ = saturation(
            -6 * r / t_go**2 - 4 * v / t_go - g,
            self.max_thrust / mass,
        )
        self.atarg_history.append(a_targ)
        return a_targ

    def get_atarg(self):
        return self.atarg_history[-1] if len(self.atarg_history) > 0 else np.zeros(3)

    def _check_bounds_violation(self, state: ArrayLike):
        r = np.float32(state[0:3])
        return not self.position_bounds_space.contains(r)

    def _check_landing(self, state):
        """
        (Unused in step, but kept for reference).
        Check if all landing conditions are satisfied.
        """
        r = np.linalg.norm(state[0:3])
        v = np.linalg.norm(state[3:6])
        q = state[6:10]
        omega = state[10:13]
        attitude_euler_angles = self.rotation_obj.as_euler("zyx")

        landing_conditions = {
            "zero_height": state[0] <= 1e-3,
            "velocity_limit": v < self.maximum_v,
            "landing_radius": r < self.target_r,
            "attitude_limit": np.any(
                abs(attitude_euler_angles) < self.landing_attitude_limit
            ),
            "omega_limit": np.any(abs(omega) < self.omega_lim),
        }
        return landing_conditions

    # def _add_meshes_to_plotter(self, resetting: bool = False):
    #     current_loc = self.state[0:3]

    #     self.rocket_body_mesh = pv.Cylinder(
    #         center=current_loc,
    #         direction=self.rotation_obj.apply([1, 0, 0]),
    #         radius=3.66 / 2,
    #         height=50,
    #     )

    #     self.landing_pad_mesh = pv.Circle(radius=self.target_r)
    #     self.landing_pad_mesh.rotate_y(angle=90, inplace=True)

    #     thrust_vector = self.SIM.get_thrust_vector_inertial()

    #     # Add rocket body
    #     self.plotter.add_mesh(
    #         self.rocket_body_mesh,
    #         show_scalar_bar=False,
    #         color="#c8f7c5",
    #         name="rocket_body",
    #     )

    #     # Landing pad coloring for success
    #     if all(self._check_landing(self.state).values()):
    #         self.plotter.add_mesh(
    #             self.landing_pad_mesh, color="#00ff00", name="landing_pad"
    #         )
    #     else:
    #         self.plotter.add_mesh(self.landing_pad_mesh, color="red", name="landing_pad")

    def _scipy_quat_convention(self, leading_scalar_quaternion: ArrayLike):
        """
        Converts from leading-scalar [w, x, y, z] to [x, y, z, w].
        """
        return np.roll(leading_scalar_quaternion, -1)

    def _compute_vtarg(self, r, v):
        """
        Example of velocity-based shaping target.
        """
        tau_1 = 20
        tau_2 = 100
        initial_conditions = self.SIM.states[0]
        v_0 = np.linalg.norm(initial_conditions[3:6])

        rx = r[0]
        if rx > self.waypoint:
            r_hat = r - [self.waypoint, 0, 0]
            v_hat = v - [-2, 0, 0]
            tau = tau_1
        else:
            r_hat = [rx + 1, 0, 0]
            v_hat = v - [-1, 0, 0]
            tau = tau_2

        t_go = max(1e-3, np.linalg.norm(r_hat) / max(1e-3, np.linalg.norm(v_hat)))
        v_targ = -v_0 * (r_hat / max(1e-3, np.linalg.norm(r_hat))) * (1 - np.exp(-t_go / tau))

        self.vtarg_history.append(v_targ)
        return v_targ, t_go

    # -------------------------------------------------------------------------
    # Extra utility methods for trajectory logging / plotting
    # -------------------------------------------------------------------------
    # def get_state(self):
    #     return self.state

    # def get_trajectory_plotly(self):
    #     trajectory_dataframe = self.states_to_dataframe()
    #     return self._trajectory_plot_from_df(trajectory_dataframe)

    # def get_attitude_trajectory(self):
    #     trajectory_dataframe = self.states_to_dataframe()
    #     return self._attitude_traj_from_df(trajectory_dataframe)

    # def _attitude_traj_from_df(self, trajectory_df: DataFrame):
    #     import plotly.express as px
    #     fig = px.line(trajectory_df[["q0", "q1", "q2", "q3"]])
    #     return fig

    # def _trajectory_plot_from_df(self, trajectory_df: DataFrame):
    #     import plotly.express as px
    #     fig = px.line_3d(trajectory_df, x="x", y="y", z="z")
    #     # Additional plotting logic omitted for brevity
    #     return fig

    # def get_atarg_plotly(self):
    #     trajectory_dataframe = self.states_to_dataframe()
    #     return self._atarg_figure(trajectory_dataframe)

    # def _atarg_figure(self, trajectory_df: DataFrame):
    #     import plotly.express as px
    #     atarg_df = self.atarg_to_dataframe()
    #     fig = px.line_3d(trajectory_df, x="x", y="y", z="z")
    #     # Additional plotting logic omitted for brevity
    #     return fig

    # def get_vtarg_trajectory(self):
    #     trajectory_dataframe = self.states_to_dataframe()
    #     return self._vtarg_plot_figure(trajectory_dataframe)

    # def _vtarg_plot_figure(self, trajectory_df: DataFrame):
    #     import plotly.express as px
    #     vtarg_df = self.vtarg_to_dataframe()
    #     fig = px.line_3d(trajectory_df, x="x", y="y", z="z")
    #     # Additional plotting logic omitted for brevity
    #     return fig

    # def states_to_dataframe(self):
    #     import pandas as pd
    #     return pd.DataFrame(self.SIM.states, columns=self.state_names)

    # def actions_to_dataframe(self):
    #     import pandas as pd
    #     return pd.DataFrame(self.SIM.actions, columns=self.action_names)

    # def atarg_to_dataframe(self):
    #     import pandas as pd
    #     return pd.DataFrame(self.atarg_history, columns=["ax", "ay", "az"])

    # def vtarg_to_dataframe(self):
    #     import pandas as pd
    #     return pd.DataFrame(self.vtarg_history, columns=["v_x", "v_y", "v_z"])

    # def used_mass(self):
    #     initial_mass = self.SIM.states[0][-1]
    #     final_mass = self.SIM.states[-1][-1]
    #     return initial_mass - final_mass