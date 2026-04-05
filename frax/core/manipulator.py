"""Manipulator kinematics and dynamics"""

from typing import Tuple, Optional

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np

from frax.core.robot import Robot
from frax.utils.general_utils import tuplify


@jax.tree_util.register_static
class Manipulator(Robot):
    """Manipulator kinematics and dynamics

    Args:
        urdf_filename (str): Path to the URDF file to load
        collision_data (Optional[dict]): Collision information. Contains info
            on body/root collision data, as well as self-collision pairs.
            See collision_utils for more detail. Defaults to None.
        joint_ordering (Optional[list[str]]): A specific joint ordering to use.
            Defaults to None (infer ordering from URDF)
        add_floating_base (bool, optional): Whether to add a 6DOF floating base to the model.
            Defaults to False.
        ee_offset (Optional[ArrayLike]): Transformation matrix specifying the end-effector
            offset from the last joint frame. Defaults to None.
    """

    def __init__(
        self,
        urdf_filename: str,
        collision_data: Optional[dict] = None,
        joint_ordering: Optional[list[str]] = None,
        add_floating_base: bool = False,
        ee_offset: Optional[ArrayLike] = None,
    ):
        super().__init__(
            urdf_filename,
            collision_data,
            joint_ordering,
            add_floating_base=add_floating_base,
        )
        # TODO decide if floating base should be an input? Only makes sense for in-space manipulators...
        assert self.is_pure_kinematic_chain

        # TODO REORGANIZE ALL OF THIS BELOW
        self.ee_parent_chain = np.arange(self.num_joints)

        if ee_offset is None:
            ee_offset = tuplify(np.eye(4))
        else:
            ee_offset = np.asarray(ee_offset)
            assert ee_offset.shape == (4, 4)
            ee_offset = tuplify(ee_offset)

        self.ee_offset = ee_offset

    # EE STUFF

    def ee_transform(self, q: Array) -> Array:
        """Transformation matrix of the end effector (EE frame --> world frame)

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Transformation matrix, shape (4, 4)
        """
        transforms = self.joint_to_world_transforms(q)
        return self._ee_transform(transforms)

    def _ee_transform(self, joint_transforms: Array) -> Array:
        """Helper function: Compute EE transform given joint transforms"""
        parent_index = self.ee_parent_chain[-1]
        offset = jnp.asarray(self.ee_offset)
        return self._frame_transform(joint_transforms, offset, parent_index)

    def ee_jacobian(self, q: Array) -> Array:
        """Jacobian [Jv; Jw] of the end effector given the joint configuration

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Jacobian, shape (6, num_joints). The first 3 rows are the linear Jacobian,
                and the last 3 rows are the angular Jacobian
        """
        transforms = self.joint_to_world_transforms(q)
        return self._ee_jacobian(transforms)

    def _ee_jacobian(self, joint_transforms: Array):
        """Helper function: Compute EE jacobian given joint transforms"""
        parent_chain = jnp.asarray(self.ee_parent_chain)
        offset = jnp.asarray(self.ee_offset)
        return self._frame_jacobian(joint_transforms, offset, parent_chain)

    def ee_jacobian_and_derivative(self, q: Array, qd: Array) -> Tuple[Array, Array]:
        """End-effector Jacobian and its time derivative (w.r.t world)

        Args:
            q (Array): Joint positions, shape (num_joints,)
            qd (Array): Joint velocities, shape (num_joints,)

        Returns:
            Tuple[Array, Array]:
                J (Array): EE Jacobian, shape (6, num_joints)
                Jdot (Array): Time derivative of the EE Jacobian, shape (6, num_joints)
        """
        transforms = self.joint_to_world_transforms(q)
        return self._ee_jacobian_and_derivative(qd, transforms)

    def _ee_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        """Helper function: Compute EE Jacobian and time derivative given joint transforms"""
        parent_chain = jnp.asarray(self.ee_parent_chain)
        offset = jnp.asarray(self.ee_offset)
        return self._frame_jacobian_and_derivative(
            qd, joint_transforms, offset, parent_chain
        )

    def ee_manipulability_index(self, q: Array) -> float:
        """Manipulability index of the end-effector Jacobian

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            float: Manipulability index
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._ee_manipulability_index(joint_transforms)

    def _ee_manipulability_index(self, joint_transforms: Array) -> float:
        """Helper function: Computes EE manipulability index given joint transforms"""
        J_full = self._ee_jacobian(joint_transforms)
        return self._manipulability_index_helper(J_full, self.ee_parent_chain)

    def torque_control_matrices(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """Compute the matrices required for operational space torque control
        with just a single evaluation of the kinematics

        Args:
            q (Array): Joint positions, shape (num_joints,)
            qd (Array): Joint velocities, shape (num_joints,)

        Returns:
            Tuple[Array, Array, Array, Array, Array, Array]:
                M: Mass matrix, shape (num_joints, num_joints)
                M_inv: Inverse of the mass matrix, shape (num_joints, num_joints)
                G: Gravity vector, shape (num_joints,)
                C: Centrifugal/coriolis vector, shape (num_joints,)
                J: End effector basic Jacobian, shape (6, num_joints)
                T: End effector transformation matrix, shape (4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        M = self._mass_matrix(joint_transforms)
        M_inv = self.mass_matrix_inverse(M)
        G = self._gravity_vector(joint_transforms)
        C = self._centrifugal_coriolis_vector(qd, joint_transforms)
        J = self._ee_jacobian(joint_transforms)
        T = self._ee_transform(joint_transforms)
        return M, M_inv, G, C, J, T

    def velocity_control_matrices(self, q: Array) -> Tuple[Array, Array]:
        """Compute the matrices required for operational space velocity control
        with just a single evaluation of the kinematics

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Tuple[Array, Array]:
                J: End effector basic Jacobian, shape (6, num_joints)
                T: End effector transformation matrix, shape (4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        J = self._ee_jacobian(joint_transforms)
        T = self._ee_transform(joint_transforms)
        return J, T

    def dynamically_consistent_velocity_control_matrices(
        self, q: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute the matrices required for operational space velocity control
        with just a single evaluation of the kinematics.

        This version also returns the inverse of the mass matrix, which is required
        to construct the dynamically-consistent generalized Jacobian inverse

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Tuple[Array, Array, Array]:
                M_inv: Inverse of the mass matrix, shape (num_joints, num_joints)
                J: End effector basic Jacobian, shape (6, num_joints)
                T: End effector transformation matrix, shape (4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        J = self._ee_jacobian(joint_transforms)
        T = self._ee_transform(joint_transforms)
        M = self._mass_matrix(joint_transforms)
        M_inv = self.mass_matrix_inverse(M)
        return M_inv, J, T
