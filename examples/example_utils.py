"""Utilities for FRAX operational space control / differential inverse kinematics examples"""

import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
from jax import Array


class SinusoidalTaskTrajectory:
    """An example sinusoidal task-space position trajectory for the robot to follow

    Args:
        init_pos (np.ndarray): Initial position of the end-effector, shape (3,)
        init_rot (np.ndarray): Initial rotation of the end-effector, shape (3, 3)
        amplitude (np.ndarray): X,Y,Z amplitudes of the sinusoid, shape (3,)
        angular_freq (np.ndarray): X,Y,Z angular frequencies of the sinusoid, shape (3,)
        phase (np.ndarray): X,Y,Z phase offsets of the sinusoid, shape (3,)
    """

    def __init__(
        self,
        init_pos: np.ndarray,
        init_rot: np.ndarray,
        amplitude: np.ndarray,
        angular_freq: np.ndarray,
        phase: np.ndarray,
    ):
        self.init_pos = np.asarray(init_pos)
        self.init_rot = np.asarray(init_rot)
        self.amplitude = np.asarray(amplitude)
        self.angular_freq = np.asarray(angular_freq)
        self.phase = np.asarray(phase)

        assert self.init_pos.shape == (3,)
        assert self.init_rot.shape == (3, 3)
        assert self.amplitude.shape == (3,)
        assert self.angular_freq.shape == (3,)
        assert self.phase.shape == (3,)

    # Simple sinusoidal positional trajectory

    def position(self, t: float) -> np.ndarray:
        return self.init_pos + self.amplitude * np.sin(
            self.angular_freq * t + self.phase
        )

    def velocity(self, t: float) -> np.ndarray:
        return (
            self.amplitude
            * self.angular_freq
            * np.cos(self.angular_freq * t + self.phase)
        )

    def acceleration(self, t: float) -> np.ndarray:
        return (
            -self.amplitude
            * self.angular_freq**2
            * np.sin(self.angular_freq * t + self.phase)
        )

    # Maintain a fixed orientation

    def rotation(self, t: float) -> np.ndarray:
        return self.init_rot

    def omega(self, t: float) -> np.ndarray:
        return np.zeros(3)

    def alpha(self, t: float) -> np.ndarray:
        return np.zeros(3)


class PandaEnv:
    """Simulation environment for manipulator end-effector pose tracking using MuJoCo

    Args:
        control_mode (str): Control mode, either "torque" or "velocity"
        traj (Optional[SinusoidalTaskTrajectory]): Task-space trajectory for the target to follow.
        real_time (bool): Whether to run the simulation in "real time". Defaults to False.
    """

    def __init__(
        self,
        control_mode: str,
        traj: Optional[SinusoidalTaskTrajectory] = None,
        real_time: bool = False,
        load_obstacle: bool = False,
    ):
        # Load the Panda scene xml
        if load_obstacle:
            filename = "obstacle_scene.xml"
        else:
            filename = "scene.xml"
        xml_path = Path(__file__).parent / "xml" / filename
        # Set initial Panda joint configuration
        q_init = np.array([0, -np.pi / 3, 0, -5 * np.pi / 6, 0, np.pi / 2, 0])
        # Set initial target position
        target_pos = np.array([0.5, 0, 0.5])

        self.control_mode = control_mode
        self.traj = traj
        self.real_time = real_time

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        ]
        # Set initial target position
        self.data.mocap_pos[self.mocap_id] = target_pos
        # Set initial joint position
        if q_init is not None:
            self.data.qpos[: len(q_init)] = q_init

        mujoco.mj_forward(self.model, self.data)

        self.dt = self.model.opt.timestep
        self.t = 0
        self.last_time = time.time()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def get_joint_state(self) -> np.ndarray:
        # For the Panda, the first 7 joints are the arm
        # (i.e. don't include the gripper state)
        return np.concatenate([self.data.qpos[:7], self.data.qvel[:7]])

    def get_desired_ee_state(self) -> np.ndarray:
        if self.traj is not None:
            pos = self.traj.position(self.t)
            rot = self.traj.rotation(self.t).ravel()
            vel = self.traj.velocity(self.t)
            omega = self.traj.omega(self.t)

            # Update mocap body
            self.data.mocap_pos[self.mocap_id] = pos
            # Maintain a fixed rotation and orientation (facing down)
            # rot_matrix = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            # MuJoCo mocap quat is WXYZ
            self.data.mocap_quat[self.mocap_id] = [0, 1, 0, 0]  # 180 deg around X

            return np.array([*pos, *rot, *vel, *omega])

        # Respond to GUI inputs (moving the mocap body with an applied force)
        pos = self.data.mocap_pos[self.mocap_id]
        # Using a fixed rotation for now
        # TODO: Allow for user modification of the mocap body's rotation?
        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).ravel()
        vel = np.zeros(3)
        omega = np.zeros(3)
        return np.array([*pos, *rot, *vel, *omega])

    def apply_control(self, u: Array) -> None:
        if self.control_mode == "velocity":
            # Slight hack: Modify qvel directly and combine with mj_forward instead of mj_step
            self.data.qvel[:7] = u
        else:  # Torque control
            self.data.ctrl[:7] = u

    def step(self):
        if self.control_mode == "velocity":
            # Slight hack: use mj_forward as a visualizer for the kinematics
            # This avoids any actual physics simulation which would require either
            # 1) modifying the actuator types in the xml, or
            # 2) creating a new controller with gravity/inertia/nonlinear comp
            # This is the easiest solution that still shows how the diff IK is working
            self.data.qpos[:7] += self.data.qvel[:7] * self.dt
            mujoco.mj_forward(self.model, self.data)
        else:
            # Torque control -- use the actual motor actuators with the applied ctrl
            mujoco.mj_step(self.model, self.data)

        self.t += self.dt
        self.viewer.sync()
        if self.real_time:
            elapsed = time.time() - self.last_time
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
            self.last_time = time.time()

    def close(self):
        self.viewer.close()
