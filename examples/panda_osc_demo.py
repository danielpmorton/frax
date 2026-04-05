"""Operational space control demo with the Franka Panda and FRAX kinematics + dynamics"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from frax.robots.franka_panda import load_panda
from frax.utils.rotation_utils import orientation_error_3D
from example_utils import PandaEnv, SinusoidalTaskTrajectory

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")


MANUAL_MODE_MSG = """
====================================
Starting demo in MANUAL mode

To control the end-effector position:
- Double-click on the target
- Right-click-and-drag to perturb it

NOTE: This demo does not include safety constraints
(for example, collision/singularity avoidance).
For a full handling of these, see the OSCBF
demo in this repo, as well as:
https://github.com/StanfordASL/oscbf

To switch to TRAJECTORY mode, run the script
with the --mode trajectory flag
====================================
"""

TRAJ_MODE_MSG = """
====================================
Starting demo in TRAJECTORY mode

This will show the controller tracking
a sinusoidal task-space position trajectory
while maintaining a fixed EE orientation

To switch to MANUAL mode, run the script
with the --mode manual flag
====================================
"""


def main(demo_mode):
    assert demo_mode in {"trajectory", "manual"}
    if demo_mode == "trajectory":
        print(TRAJ_MODE_MSG)
        traj = SinusoidalTaskTrajectory(
            init_pos=(0.4, 0, 0.4),
            init_rot=np.array(
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ),
            amplitude=(0, 0.25, 0),
            angular_freq=(0, 2, 0),
            phase=(0, 0, 0),
        )
    else:
        print(MANUAL_MODE_MSG)
        traj = None

    env = PandaEnv(control_mode="torque", traj=traj, real_time=True)

    # Load FRAX robot model
    robot = load_panda()

    # Define gains
    kp_pos = 50.0 * np.ones(3)
    kp_rot = 20.0 * np.ones(3)
    kd_pos = 20.0 * np.ones(3)
    kd_rot = 10.0 * np.ones(3)
    kp_task = np.concatenate([kp_pos, kp_rot])
    kd_task = np.concatenate([kd_pos, kd_rot])
    kp_joint = 10.0 * np.ones(robot.num_joints)
    kd_joint = 5.0 * np.ones(robot.num_joints)

    # Define nullspace posture task
    is_redundant = True  # 6DOF task, 7DOF robot
    des_q = np.array([0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, 0.0])
    des_qdot = np.zeros(7)

    # Define acceleration terms for EE task
    des_accel = np.zeros(3)
    des_alpha = np.zeros(3)

    @jax.jit
    def operational_space_control(z, z_ee_des):
        # Extract state info
        q = z[:7]
        qdot = z[7:14]
        des_pos = z_ee_des[:3]
        des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
        des_vel = z_ee_des[12:15]
        des_omega = z_ee_des[15:18]

        # FRAX Kinematics + Dynamics
        M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
        pos = ee_tmat[:3, 3]
        rot = ee_tmat[:3, :3]

        # Compute twist
        twist = J @ qdot
        vel = twist[:3]
        omega = twist[3:]

        # Errors
        pos_error = pos - des_pos
        vel_error = vel - des_vel
        rot_error = orientation_error_3D(rot, des_rot)
        omega_error = omega - des_omega
        task_p_error = jnp.concatenate([pos_error, rot_error])
        task_d_error = jnp.concatenate([vel_error, omega_error])

        # Operational space matrices
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia

        # Compute operational space task torques
        task_accel = (
            jnp.concatenate([des_accel, des_alpha])
            - kp_task * task_p_error
            - kd_task * task_d_error
        )
        task_wrench = task_inertia @ task_accel
        tau = J.T @ task_wrench

        # Add compensation for nonlinear effects
        tau += g + c

        if is_redundant:
            # Nullspace projection
            NT = jnp.eye(robot.num_joints) - J.T @ J_bar.T
            # Add nullspace joint task
            q_error = q - des_q
            qdot_error = qdot - des_qdot
            joint_accel = -kp_joint * q_error - kd_joint * qdot_error
            secondary_joint_torques = M @ joint_accel
            tau += NT @ secondary_joint_torques

        # Clamp to torque limits
        return jnp.clip(
            tau,
            -1.0 * jnp.asarray(robot.joint_max_forces),
            jnp.asarray(robot.joint_max_forces),
        )

    try:
        while env.viewer.is_running():
            z = env.get_joint_state()
            z_ee_des = env.get_desired_ee_state()
            u = operational_space_control(z, z_ee_des)
            env.apply_control(u)
            env.step()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["trajectory", "manual"], default="trajectory"
    )
    args = parser.parse_args()
    main(args.mode)
