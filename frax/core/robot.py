"""Robot kinematics and dynamics"""

# Assorted TODOs:
# - Some logic in the jacobian computation (frame, link, joint) is duplicated,
#   this can likely be simplified with some helper functions
# - Also, see if we can use some of the spatial axes computation for the jacobians

from typing import Tuple, Optional

import jax
from jax import Array
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from frax.utils.urdf_parser import parse_urdf
from frax.utils.general_utils import tuplify
from frax.utils.linalg_utils import (
    schur_spd_inverse,
    cholesky_spd_inverse,
)
from frax.utils.transform_utils import (
    create_transform_numpy,
    transform_points,
    joint_transform,
)
from frax.utils.spatial_utils import (
    get_spatial_inertias,
    get_spatial_joint_axes,
    spatial_motion_cross,
    spatial_force_cross,
)


@jax.tree_util.register_static
class Robot:
    """Robot kinematics and dynamics

    Args:
        urdf_filename (str): Path to the URDF file to load
        collision_data (Optional[dict]): Collision information. Contains info
            on body/root collision data, as well as self-collision pairs.
            See collision_utils for more detail. Defaults to None.
        joint_ordering (Optional[list[str]]): A specific joint ordering to use.
            Defaults to None (infer ordering from URDF)
        add_floating_base (bool, optional): Whether to add a 6DOF floating base to the model.
            Defaults to False.
    """

    def __init__(
        self,
        urdf_filename: str,
        collision_data: Optional[dict] = None,
        joint_ordering: Optional[list[str]] = None,
        add_floating_base: bool = False,
    ):
        data = parse_urdf(
            urdf_filename,
            joint_ordering=joint_ordering,
            add_floating_base=add_floating_base,
        )
        data = {k: tuplify(v) for k, v in data.items()}

        assert isinstance(collision_data, dict) or collision_data is None
        if isinstance(collision_data, dict):
            collision_positions = collision_data["positions"]
            collision_radii = collision_data["radii"]
            root_collision_positions = collision_data["root_positions"]
            root_collision_radii = collision_data["root_radii"]
            root_sc_pairs = collision_data["root_sc_pairs"]
            root_sc_tols = collision_data["root_sc_tols"]
            body_sc_pairs = collision_data["body_sc_pairs"]
            body_sc_tols = collision_data["body_sc_tols"]
        else:
            collision_positions = ()
            collision_radii = ()
            root_collision_positions = ()
            root_collision_radii = ()
            root_sc_pairs = ()
            root_sc_tols = ()
            body_sc_pairs = ()
            body_sc_tols = ()

        self.num_joints = data["num_joints"]
        self.joint_types = data["joint_types"]
        self.joint_names = data["joint_names"]
        self.joint_lower_limits = data["joint_lower_limits"]
        self.joint_upper_limits = data["joint_upper_limits"]
        self.joint_max_forces = data["joint_max_forces"]
        self.joint_max_velocities = data["joint_max_velocities"]
        self.joint_axes = data["joint_axes"]
        self.joint_parent_frame_positions = data["joint_parent_frame_positions"]
        self.joint_parent_frame_rotations = data["joint_parent_frame_rotations"]
        self.link_masses = data["link_masses"]
        self.link_local_inertias = data["link_local_inertias"]
        self.link_local_inertia_positions = data["link_local_inertia_positions"]
        self.link_local_inertia_rotations = data["link_local_inertia_rotations"]
        self.parent_idxs = data["parent_idxs"]
        self.includes_floating_dof = add_floating_base  # TODO rename this
        self.collision_positions = collision_positions
        self.collision_radii = collision_radii
        self.root_collision_positions = root_collision_positions
        self.root_collision_radii = root_collision_radii
        self.root_sc_pairs = root_sc_pairs
        self.root_sc_tols = root_sc_tols
        self.body_sc_pairs = body_sc_pairs
        self.body_sc_tols = body_sc_tols

        self.num_actuated_joints = (
            self.num_joints - 6 if self.includes_floating_dof else self.num_joints
        )
        self.has_collision_data = len(collision_positions) > 0
        self.has_root_collision_data = len(root_collision_positions) > 0
        self.has_sc_data = len(body_sc_pairs) > 0
        self.joint_to_prev_joint_tfs = tuplify(
            [
                create_transform_numpy(rot, trans)
                for rot, trans in zip(
                    self.joint_parent_frame_rotations, self.joint_parent_frame_positions
                )
            ]
        )
        self.link_com_to_prev_joint_tfs = tuplify(
            [
                create_transform_numpy(rot, trans)
                for rot, trans in zip(
                    self.link_local_inertia_rotations, self.link_local_inertia_positions
                )
            ]
        )
        # Set up padded collision positions for vmapping with static shape
        (
            self.padded_collision_positions,
            self.collision_slice_indices,
        ) = self._process_collision_data(collision_positions, collision_radii)
        self.flat_collision_radii = tuple(
            jax.tree_util.tree_flatten(self.collision_radii)[0]
        )

        self.total_mass = np.sum(self.link_masses)
        self.inverse_total_mass = 1.0 / self.total_mass

        # Set up joint type masks
        self.prismatic_mask = jnp.asarray(self.joint_types)
        self.revolute_mask = 1 - self.prismatic_mask

        self.ancestor_mask = self._compute_ancestor_mask()
        self.is_pure_kinematic_chain = np.array_equal(
            self.ancestor_mask, np.tril(np.ones((self.num_joints, self.num_joints)))
        )

        self.joint_name_to_index = {name: i for i, name in enumerate(self.joint_names)}

    def _process_collision_data(
        self, positions: tuple, radii: tuple
    ) -> Tuple[tuple, tuple]:
        """Helper function: Sets up a padded representation of the collision sphere positions
        for vmapping with uniform shape

        This should only be called once upon initialization

        Args:
            positions (tuple): Collision sphere locations for each link. Tuple of len (num_links)
                where each entry contains a set of points (length 3) for that link
            radii (tuple): Collision sphere radii for each link. Tuple of len (num_links)
                where each entry contains a set of radii for that link

        Returns:
            Tuple[tuple, tuple]:
                padded_positions: Positions, shape (num_joints, max_spheres_per_link, 3)
                slice_indices: Indices of the *flattened* padded positions to select,
                    corresponding to the non-padded data
        """
        assert isinstance(positions, tuple)
        assert isinstance(radii, tuple)
        if len(positions) == 0 or len(radii) == 0:
            return (), ()
        sphere_counts = tuple(len(rs) for rs in radii)
        max_spheres_per_link = max(sphere_counts)
        padded_positions = np.zeros((self.num_joints, max_spheres_per_link, 3))
        for link_idx in range(self.num_joints):
            for sphere_idx in range(sphere_counts[link_idx]):
                padded_positions[link_idx, sphere_idx] = positions[link_idx][sphere_idx]
        padded_positions = tuplify(padded_positions)
        # mask: (num_joints, max_spheres) - True for non-padded spheres
        sphere_mask = (
            np.arange(max_spheres_per_link) < np.asarray(sphere_counts)[:, None]
        )
        slice_indices = tuplify(np.flatnonzero(sphere_mask.flatten()))
        return padded_positions, slice_indices

    # TODO figure out if the dtype of the mask makes much difference to performance...
    # Right now it's float but switching to bool doesn't seem to change much
    def _compute_ancestor_mask(self) -> Array:
        """Computes the connectivity matrix for the tree structure.

        Returns:
            Array: Shape (num_joints, num_joints). Mask[i, j] = 1 if j is an ancestor of i
        """
        N = self.num_joints
        mask = np.zeros((N, N))

        # Based on how we've parsed the URDF, the base (pelvis) link is assigned idx = 0
        # But it's slightly easier to compute this mask if it is assigned -1
        assert np.min(self.parent_idxs) == -1

        # Assumes topological sort (parents appear before children)
        for i in range(N):
            mask[i, i] = 1.0  # A joint affects its own link
            parent = self.parent_idxs[i]
            while parent != -1:
                mask[i, parent] = 1.0
                parent = self.parent_idxs[parent]

        return mask

    def joint_to_world_transforms(self, q: Array) -> Array:
        """Computes the transformation matrices for all joints (Joint frame --> world frame)

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Transformation matrices, shape (num_joints, 4, 4)
        """
        # Note about different FK methods:
        # Jax's associative scan is O(log(N)) complexity whereas just unrolling
        # the loop is O(N). For a pure kinematic chain like a serial manipulator,
        # associative scan should be best. But for a kinematic tree like a humanoid,
        # it may be simpler to unroll the loop and rely on the parent mapping.
        if self.is_pure_kinematic_chain:
            return self._scanned_fk(q)
        return self._unrolled_fk(q)

    def _local_joint_transforms(self, q: Array) -> Array:
        """Helper function: Computes each joint's transform in parent frame"""
        # Convert static data to jax arrays
        joint_axes = jnp.asarray(self.joint_axes)
        joint_types = jnp.asarray(self.joint_types)
        joint_to_prev_joint_tfs = jnp.asarray(self.joint_to_prev_joint_tfs)
        # Calculate each joint's transformation matrix
        transforms = jax.vmap(joint_transform)(q, joint_axes, joint_types)
        # Multiply the transform by its corresponding link offset
        return joint_to_prev_joint_tfs @ transforms

    def _unrolled_fk(self, q: Array) -> Array:
        """Compute the forward kinematics via unrolling the loop over the joints"""
        # Compute local joint transforms in parent frame
        local_tfs = self._local_joint_transforms(q)
        # Unrolled FK loop. Assumes topological sort of parent-child relationship
        world_tfs = jnp.zeros((self.num_joints, 4, 4))
        for i in range(self.num_joints):
            parent = self.parent_idxs[i]
            parent_tf = world_tfs[parent] if parent != -1 else jnp.eye(4)
            world_tfs = world_tfs.at[i].set(parent_tf @ local_tfs[i])
        return world_tfs

    def _scanned_fk(self, q: Array) -> Array:
        """Compute the forward kinematics via scanning over a pure kinematic chain"""
        assert self.is_pure_kinematic_chain
        # Compute local joint transforms in parent frame
        local_tfs = self._local_joint_transforms(q)
        # Return the cumulative product of the transformations
        return jax.lax.associative_scan(jnp.matmul, local_tfs, reverse=False, axis=0)

    def link_to_world_transforms(self, q: Array) -> Array:
        """Compute the transformation matrices for all link inertial frames (link inertial frame --> world frame)

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Transformation matrices, shape (num_joints, 4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_to_world_transforms(joint_transforms)

    def _link_to_world_transforms(self, joint_transforms: Array) -> Array:
        """Helper function: Computes link inertial transformation matrices, given joint transforms"""
        # Convert static data to jax arrays
        link_com_to_prev_joint_tfs = jnp.asarray(self.link_com_to_prev_joint_tfs)
        # Multiply the transform by its corresponding link offset
        transforms = joint_transforms @ link_com_to_prev_joint_tfs
        return transforms

    # Note: if only the COM positions (not rotations) are needed, using this function
    # as opposed to link_to_world_transforms is a bit faster
    def link_com_positions(self, q: Array) -> Array:
        """Compute the positions of all link COMs in world frame

        Args:
            q (Array): Joint angles, shape (num_joints,)

        Returns:
            Array: Link COM positions in world frame, shape (num_links, 3)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_com_positions(joint_transforms)

    def _link_com_positions(self, joint_transforms: Array) -> Array:
        """Helper function: Compute the positions of all link COMs in world frame, given the joint transforms"""
        # Convert static data to jax arrays
        link_local_inertia_positions = jnp.asarray(self.link_local_inertia_positions)
        # Determine the positions of the link COMs in world frame. Shape (num_joints, 3)
        # Position in world frame = joint-to-world transform x position in joint frame
        homogeneous_pos = jnp.column_stack(
            [link_local_inertia_positions, jnp.ones(self.num_joints)]
        )
        return jnp.einsum("qij,qj->qi", joint_transforms, homogeneous_pos)[:, :3]

    def center_of_mass(self, q: Array) -> Array:
        """Compute the center of mass of the robot, in world frame

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Position of the center of mass, shape (3,)
        """
        transforms = self.joint_to_world_transforms(q)
        return self._center_of_mass(transforms)

    def _center_of_mass(self, joint_transforms: Array) -> Array:
        """Helper function: Compute center of mass given joint transforms"""
        link_masses = jnp.asarray(self.link_masses)
        link_com_positions = self._link_com_positions(joint_transforms)
        # Inertially averaged position
        return (
            jnp.sum(link_masses.reshape(-1, 1) * link_com_positions, axis=0)
            * self.inverse_total_mass
        )

    def center_of_mass_jacobian(self, q: Array) -> Array:
        """Computes the linear Jacobian (Jv) for the motion of the COM

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Jv_COM, shape (3, num_joints)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._center_of_mass_jacobian(joint_transforms)

    def _center_of_mass_jacobian(self, joint_transforms: Array) -> Array:
        """Helper function: Compute center of mass jacobian given joint transforms"""
        link_Jvs = self._link_linear_jacobians(joint_transforms)
        return self._com_jacobian_from_link_jacobians(link_Jvs)

    def _com_jacobian_from_link_jacobians(self, link_Jvs: Array) -> Array:
        """Helper function: Compute center of mass jacobian given link linear jacobians"""
        link_masses = jnp.asarray(self.link_masses)
        # Inertially-weighted average of the link COM jacobians
        return jnp.einsum("l,ldj->dj", link_masses, link_Jvs) * self.inverse_total_mass

    def _frame_transform(
        self, joint_transforms: Array, frame_transform: Array, parent_index: int
    ) -> Array:
        """Computes the transformation matrix (w.r.t world) of a frame
        attached to a link with a specified parent joint

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)
            frame_transform (Array): Transformation matrix of interest in its local frame, shape (4, 4)
            parent_index (int): Index of the frame's parent joint

        Returns:
            Array: Transformation matrix, shape (4, 4)
        """
        return joint_transforms[parent_index] @ frame_transform

    def _frame_jacobian(
        self, joint_transforms: Array, frame_transform: Array, parent_chain: Array
    ) -> Array:
        """Computes the jacobian of a frame attached to a link with a specified parent chain

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)
            frame_transform (Array): Transformation matrix of interest in its local frame, shape (4, 4)
            parent_chain (Array): Ancestor joint indices of the frame's link

        Returns:
            Array: Jacobian, shape (6, num_joints). The first 3 rows are the linear Jacobian,
                and the last 3 rows are the angular Jacobian
        """
        # Get transform of the frame w.r.t the root
        frame_to_root_tf = self._frame_transform(
            joint_transforms, frame_transform, parent_chain[-1]
        )
        frame_pos = frame_to_root_tf[:3, 3]

        # Positions of all parent joints in root frame. Shape (num_joints, 3)
        parent_pos = joint_transforms[parent_chain, :3, 3]

        # Axes of all parent joints in root frame. Shape (num_joints, 3)
        parent_axes = (
            joint_transforms[parent_chain, :3, :3]
            @ jnp.asarray(self.joint_axes)[parent_chain, :, jnp.newaxis]
        ).squeeze(axis=2)

        # Position of frame, with respect to joint j. Shape (num_joints, 3).
        frame_wrt_joints = frame_pos[jnp.newaxis, :] - parent_pos

        # Cross products between joint axis j and frame position with respect to joint j.
        # Shape (num_joints, 3)
        lever_arms = jnp.cross(parent_axes, frame_wrt_joints)

        # Linear jacobian has a prismatic contribution and revolute contribution
        # Prismatic contribution: All prismatic joints' z axes
        # Revolute contribution: Cross product of vector from revolute joints to EE
        Jv = jnp.where(
            self.revolute_mask[parent_chain, None], lever_arms, parent_axes
        ).T
        # Angular jacobian only has a contribution from revolute joints (their axes)
        Jw = jnp.where(
            self.revolute_mask[parent_chain, None],
            parent_axes,
            jnp.zeros_like(parent_axes),
        ).T
        J = jnp.vstack([Jv, Jw])
        # Reconstruct full jacobian from parent computations
        J_full = jnp.zeros((6, self.num_joints)).at[:, parent_chain].set(J)
        return J_full

    # TODO: See if using more spatial algebra would simplify some of the operations here
    def _frame_jacobian_and_derivative(
        self,
        qd: Array,
        joint_transforms: Array,
        frame_transform: Array,  # TODO rename to offset_transform?
        parent_chain: Array,
    ) -> Tuple[Array, Array]:
        """Computes both the jacobian of a frame attached to a link with a specified parent chain,
        and its time derivative

        Frequently, if we need Jdot, we also need J. This function is designed to reduce duplicated
        computations between J and Jdot in that case

        Args:
            qd (Array): Joint velocities, shape (num_joints,)
            joint_transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)
            frame_transform (Array): Transformation matrix of interest in its local frame, shape (4, 4)
            parent_chain (Array): Ancestor joint indices of the frame's link

        Returns:
            Tuple[Array, Array]:
                J (Array): Jacobian, shape (6, num_joints)
                Jdot (Array): Time derivative of the Jacobian, shape (6, num_joints)
        """
        # TODO: Create a version of _joint_jacobians that allows us to just compute it for the parent chain?
        joint_Jvs, joint_Jws = self._joint_jacobians(joint_transforms)
        parent_vels = joint_Jvs[parent_chain] @ qd
        parent_ang_vels = joint_Jws[parent_chain] @ qd

        # NOTE: Many of these operations below are similar to the frame_jacobian function
        # For more documentation, refer to the comments in that function

        parent_axes = (
            joint_transforms[parent_chain, :3, :3]
            @ jnp.asarray(self.joint_axes)[parent_chain, :, jnp.newaxis]
        ).squeeze(axis=2)
        parent_axes_dot = jnp.cross(parent_ang_vels, parent_axes)
        parent_pos = joint_transforms[parent_chain, :3, 3]

        frame_to_root_tf = self._frame_transform(
            joint_transforms, frame_transform, parent_chain[-1]
        )
        frame_pos = frame_to_root_tf[:3, 3]
        frame_pos_wrt_parents = frame_pos[jnp.newaxis, :] - parent_pos
        lever_arms = jnp.cross(parent_axes, frame_pos_wrt_parents)
        Jv = jnp.where(
            self.revolute_mask[parent_chain, None], lever_arms, parent_axes
        ).T
        Jw = jnp.where(
            self.revolute_mask[parent_chain, None],
            parent_axes,
            jnp.zeros_like(parent_axes),
        ).T
        J = jnp.vstack([Jv, Jw])

        frame_vel = Jv @ qd[parent_chain]
        frame_vel_wrt_parents = frame_vel[jnp.newaxis, :] - parent_vels

        lever_arms_dot = jnp.cross(parent_axes_dot, frame_pos_wrt_parents) + jnp.cross(
            parent_axes, frame_vel_wrt_parents
        )
        Jv_dot = jnp.where(
            self.revolute_mask[parent_chain, None], lever_arms_dot, parent_axes_dot
        ).T
        Jw_dot = jnp.where(
            self.revolute_mask[parent_chain, None],
            parent_axes_dot,
            jnp.zeros_like(parent_axes_dot),
        ).T
        J_dot = jnp.vstack([Jv_dot, Jw_dot])

        # Reconstruct full Jacobian and derivative from parent computations
        J_full = jnp.zeros((6, self.num_joints)).at[:, parent_chain].set(J)
        J_dot_full = jnp.zeros((6, self.num_joints)).at[:, parent_chain].set(J_dot)
        return J_full, J_dot_full

    def _manipulability_index_helper(self, J_full: Array, chain_idxs: Array) -> float:
        """Helper function to compute the manipulability indices for all hands and feet"""
        J_reduced = J_full[:, chain_idxs]
        sigmas = jax.lax.linalg.svd(J_reduced, compute_uv=False)
        return jnp.prod(sigmas)

    def link_collision_data(self, q: Array) -> Tuple[Array, Array]:
        """Compute collision data for all links given the joint configuration

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Tuple[Array, Array]:
                positions (Array): Positions of the collision spheres in world frame,
                    shape (num_collision_spheres, 3)
                radii (Array): Radii of the collision spheres, shape (num_collision_spheres,)
        """
        if not self.has_collision_data:
            return jnp.array([]), jnp.array([])
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_collision_data(joint_transforms)

    def _link_collision_data(self, joint_transforms: Array) -> Tuple[Array, Array]:
        """Helper function: Compute the collision data for all links given the joint transforms"""
        positions = self._link_collision_positions(joint_transforms)
        radii = jnp.asarray(self.flat_collision_radii)
        return positions, radii

    def link_collision_positions(self, q: Array) -> Array:
        """Compute the positions of all collision spheres in world frame

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Collision positions, shape (num_collision_spheres, 3)
        """
        if not self.has_collision_data:
            return jnp.array([])
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_collision_positions(joint_transforms)

    def _link_collision_positions(self, joint_transforms: Array) -> Array:
        """Helper function: Compute all collision positions given joint transforms"""
        # Convert static data to jax arrays
        padded_collision_positions = jnp.asarray(self.padded_collision_positions)
        collision_slice_indices = jnp.asarray(self.collision_slice_indices)
        # Compute collision body positions in world frame
        # Shape (num_joints, max_spheres, 3)
        transformed_pts_padded = jax.vmap(transform_points)(
            joint_transforms, padded_collision_positions
        )
        # Flatten and select only the non-padded collision data
        # Flat points shape (num_joints * max_spheres, 3)
        all_pts_flat = transformed_pts_padded.reshape(-1, 3)
        pts_unpadded = all_pts_flat[collision_slice_indices]
        return pts_unpadded

    def self_collision_distances(self, q: Array) -> Array:
        if not self.has_collision_data or not self.has_sc_data:
            return jnp.array([])
        joint_transforms = self.joint_to_world_transforms(q)
        return self._self_collision_distances(joint_transforms)

    def _self_collision_distances(self, joint_transforms: Array) -> Array:
        positions, radii = self._link_collision_data(joint_transforms)
        return self._self_collision_distances_from_link_data(positions, radii)

    def _self_collision_distances_from_link_data(
        self, positions: Array, radii: Array
    ) -> Array:
        # Compute distances between spheres on different links of the body
        # Note: just use the spheres of the full-body collision model associated
        # with the self-collision model (typically a subset)
        pairs = jnp.asarray(self.body_sc_pairs)
        idxs_a = pairs[:, 0]
        idxs_b = pairs[:, 1]
        pos_a = positions[idxs_a]
        pos_b = positions[idxs_b]
        rad_a = radii[idxs_a]
        rad_b = radii[idxs_b]
        center_deltas = pos_b - pos_a
        tols = jnp.asarray(self.body_sc_tols)
        body_to_body_dists = (
            jnp.linalg.norm(center_deltas, axis=-1) - rad_a - rad_b - tols
        )
        if not self.has_root_collision_data:
            return body_to_body_dists
        # Compute distances to the fixed-to-world root
        # (note that these spheres on the root do not require FK)
        root_sc_pairs = jnp.asarray(self.root_sc_pairs)
        root_idxs = root_sc_pairs[:, 0]
        body_idxs = root_sc_pairs[:, 1]
        root_pos = jnp.asarray(self.root_collision_positions)[root_idxs]
        body_pos = positions[body_idxs]
        root_rad = jnp.asarray(self.root_collision_radii)[root_idxs]
        body_rad = radii[body_idxs]
        root_center_deltas = body_pos - root_pos
        root_tols = jnp.asarray(self.root_sc_tols)
        root_to_body_dists = (
            jnp.linalg.norm(root_center_deltas, axis=-1)
            - root_rad
            - body_rad
            - root_tols
        )
        return jnp.concatenate([body_to_body_dists, root_to_body_dists])

    def _joint_jacobians(self, joint_transforms: Array) -> Tuple[Array, Array]:
        """Helper function: Compute an array containing the linear (Jv) and angular (Jw)
        jacobians for every joint origin (w.r.t the world)

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)

        Returns:
            Tuple[Array, Array]:
                Jv_joints (Array): Linear jacobians for every joint, shape (num_joints, 3, num_joints)
                Jw_joints (Array): Angular jacobians for every joint, shape (num_joints, 3, num_joints)
        """
        # Positions of all joints in world frame. Shape (num_joints, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Axes of all joints in world frame. Shape (num_joints, 3)
        joint_axes_world_frame = jnp.einsum(
            "qij,qj->qi", joint_transforms[:, :3, :3], jnp.asarray(self.joint_axes)
        )

        # Positions of joint origin i, with respect to joint j. Shape (num_joints, num_joints, 3).
        joint_origin_wrt_joints = (
            joint_pos[:, jnp.newaxis, :] - joint_pos[jnp.newaxis, :, :]
        )

        # Cross products between joint axis j and joint origin i's position with respect to joint j.
        # Shape (num_joints, num_joints, 3)
        lever_arms = jnp.cross(joint_axes_world_frame, joint_origin_wrt_joints)

        # The ancestor mask zeros out the contributions from any joint that is not an ancestor
        # of the joint of interest
        Jv_joints = self.ancestor_mask[:, None, :] * jnp.where(
            self.revolute_mask[:, None], lever_arms, joint_axes_world_frame[None, :]
        ).transpose(0, 2, 1)

        # Angular jacobian only has a contribution from revolute joints (their axes)
        Jw_joints = self.ancestor_mask[:, None, :] * jnp.where(
            self.revolute_mask[:, None],
            joint_axes_world_frame[None, :],
            jnp.zeros_like(joint_axes_world_frame[None, :]),
        ).transpose(0, 2, 1)
        return Jv_joints, Jw_joints

    def _link_jacobians_from_joint_jacobians(
        self, joint_transforms: Array, joint_Jvs: Array, joint_Jws: Array
    ) -> Tuple[Array, Array]:
        """Helper function: Compute the linear and angular jacobians for every link,
        using the precomputed transforms and jacobians for the joints

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)
            joint_Jvs (Array): Linear jacobians for every joint, shape (num_joints, 3, num_joints)
            joint_Jws (Array): Angular jacobians for every joint, shape (num_joints, 3, num_joints)

        Returns:
            Tuple[Array, Array]:
                link_Jvs (Array): Linear jacobians for every link, shape (num_links, 3, num_joints)
                link_Jws (Array): Angular jacobians for every link, shape (num_links, 3, num_joints)
        """
        # Determine the positions of the link COMs in world frame. Shape (num_joints, 3)
        link_com_pos = self._link_com_positions(joint_transforms)

        # Positions of all joints in world frame. Shape (num_joints, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Shift the linear jacobians from the joint origin to the link COM
        # Jv_com = Jv_joint + Jw x (p_com - p_joint)
        # TODO: this is a bit of an array broadcasting mess right now
        r = link_com_pos - joint_pos
        link_Jvs = joint_Jvs + jnp.cross(
            joint_Jws.transpose(0, 2, 1), r[:, jnp.newaxis, :]
        ).transpose(0, 2, 1)

        # Angular vel is the same for all points on a link, so the link_Jws = joint_Jws
        return link_Jvs, joint_Jws

    def _link_linear_jacobians(self, joint_transforms: Array) -> Array:
        """Helper function: Compute an array containing the linear jacobians Jv for every link

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)

        Returns:
            Array: Linear jacobians for every link, shape (num_links, 3, num_joints)
        """
        # Determine the positions of the link COMs in world frame. Shape (num_joints, 3)
        link_com_pos = self._link_com_positions(joint_transforms)

        # Positions of all joints in world frame. Shape (num_joints, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Axes of all joints in world frame. Shape (num_joints, 3)
        joint_axes_world_frame = jnp.einsum(
            "qij,qj->qi", joint_transforms[:, :3, :3], jnp.asarray(self.joint_axes)
        )

        # Positions of link COM i, with respect to joint j. Shape (num_joints, num_joints, 3).
        link_com_wrt_joints = (
            link_com_pos[:, jnp.newaxis, :] - joint_pos[jnp.newaxis, :, :]
        )

        # Cross products between joint axis j and link COM i's position with respect to joint j.
        # Shape (num_joints, num_joints, 3)
        lever_arms = jnp.cross(joint_axes_world_frame, link_com_wrt_joints)

        # The ancestor mask zeros out the contributions from any joint that is not an ancestor
        # of the link of interest
        return self.ancestor_mask[:, None, :] * jnp.where(
            self.revolute_mask[:, None], lever_arms, joint_axes_world_frame[None, :]
        ).transpose(0, 2, 1)

    def _link_angular_jacobians(self, joint_transforms: Array) -> Array:
        """Helper function: Compute an array containing the angular jacobians Jw for every link

        Args:
            transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)

        Returns:
            Array: Angular jacobians for every link, shape (num_links, 3, num_joints)
        """
        # Axes of all joints in world frame. Shape (num_joints, 3)
        joint_axes_world_frame = jnp.einsum(
            "qij,qj->qi", joint_transforms[:, :3, :3], jnp.asarray(self.joint_axes)
        )
        # Apply the revolute mask to the world-frame joint axes (only revolute joints contribute
        # to the angular jacobian) and then mask out the non-ancestor joints
        return jnp.einsum(
            "lj,j,jd->ldj",
            self.ancestor_mask,
            self.revolute_mask,
            joint_axes_world_frame,
        )

    def mass_matrix(self, q: Array) -> Array:
        """Compute the mass matrix for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (num_joints,)

        Returns:
            Array: The mass matrix, shape (num_joints, num_joints)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._mass_matrix(joint_transforms)

    def _mass_matrix(self, joint_transforms: Array) -> Array:
        """Helper function: Compute mass matrix given joint transforms"""
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._crba_from_spatial_data(spatial_axes, spatial_inertias)

    def mass_matrix_inverse(self, M: Array) -> Array:
        """Compute the inverse of the mass matrix

        Args:
            M (Array): Mass matrix, shape (num_joints, num_joints)

        Returns:
            Array: Inverse of the mass matrix, shape (num_joints, num_joints)
        """
        # NOTE: It seems like for floating-base robots, if we compute the inverse
        # using the schur complement (accounting for the structure induced by the
        # free-floating DOFs), this seems to be much faster and more stable
        if self.includes_floating_dof:
            return schur_spd_inverse(M, split_idx=6)
        # Otherwise, for fixed-base robots, a cholesky factorization seems to be
        # the most stable and fast solution
        else:
            return cholesky_spd_inverse(M)

    def gravity_vector(self, q: Array) -> Array:
        """Compute the gravity vector for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (num_joints,)

        Returns:
            Array: The gravity vector, shape (num_joints,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._gravity_vector(joint_transforms)

    def _gravity_vector(self, joint_transforms: Array) -> Array:
        """Helper function: Compute gravity vector given joint transforms"""
        method = "jacobian"  # Options: "jacobian", "rnea"
        if method == "jacobian":
            # Compute the linear jacobians for every link inertial frame
            # and form the gravity vector from these
            link_Jvs = self._link_linear_jacobians(joint_transforms)
            return self._gravity_vector_from_jacobians(link_Jvs)
        else:
            # Use the recursive newton euler algorithm to compute the gravity vector
            g_accel = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
            spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
                joint_transforms
            )
            return self._rnea_from_spatial_data(
                spatial_axes,
                spatial_inertias,
                qd=None,
                qdd=None,
                gravity_accel=g_accel,
                F_ext=None,
            )

    def _gravity_vector_from_jacobians(self, link_Jvs: Array) -> Array:
        """Helper function: Compute gravity vector given link linear jacobians"""

        # The gravity vector can be computed as follows:
        # G = -1 * sum_{over all links i}(Jvi.T @ (m_i * g_vector))
        # If we know that g_vector only has a z component, we can simplify the computation

        # TODO: make gravity an input? And parse the z-axis assumption automatically

        assume_gravity_acts_only_in_z = True
        if assume_gravity_acts_only_in_z:
            g = -9.81
            mg = g * jnp.asarray(self.link_masses)
            return -mg @ link_Jvs[:, 2, :]
        else:
            g = jnp.array([0.0, 0.0, -9.81])
            return -jnp.einsum("l, ldj, d -> j", self.link_masses, link_Jvs, g)

    def centrifugal_coriolis_vector(self, q: Array, qd: Array) -> Array:
        """Compute the centrifugal and coriolis vector for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (num_joints,)
            qd (Array): Array of joint velocities, shape (num_joints,)

        Returns:
            Array: The centrifugal and coriolis vector, shape (num_joints,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._centrifugal_coriolis_vector(qd, joint_transforms)

    def _centrifugal_coriolis_vector(self, qd: Array, joint_transforms: Array) -> Array:
        """Helper function: Computes the centrifugal/coriolis vector given the joint transforms"""
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._rnea_from_spatial_data(
            spatial_axes, spatial_inertias, qd, qdd=None, gravity_accel=None, F_ext=None
        )

    def nonlinear_bias(self, q: Array, qd: Array) -> Array:
        """Compute the nonlinear bias vector (Centrifugal/Coriolis + Gravity) in a single pass
        ```
        b(q, qd) = c(q, qd) + g(q),
        ```

        Args:
            q (Array): Joint positions, shape (num_joints,)
            qd (Array): Joint velocities, shape (num_joints,)

        Returns:
            Array: The nonlinear bias vector, shape (num_joints,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._nonlinear_bias(qd, joint_transforms)

    def _nonlinear_bias(self, qd: Array, joint_transforms: Array) -> Array:
        """Helper function: Computes the nonlinear bias (c + g) given the joint transforms"""
        g_accel = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._rnea_from_spatial_data(
            spatial_axes,
            spatial_inertias,
            qd=qd,
            qdd=None,
            gravity_accel=g_accel,
            F_ext=None,
        )

    def rnea(
        self,
        q: Array,
        qd: Optional[Array],
        qdd: Optional[Array],
        gravity_accel: Optional[Array],
        F_ext: Optional[Array],
    ) -> Array:
        """Recursive Newton-Euler Algorithm (vectorized form)

        Args:
            q (Array): Joint positions, shape (num_joints,)
            qd (Optional[Array]): Joint velocities, shape (num_joints,). None if not considering
                joint velocities (as is done to compute gravity)
            qdd (Optional[Array]): Joint accelerations, shape (num_joints,). This is currently not used
                for most methods and can be set to None.
            gravity_accel (Optional[Array]): Spatial acceleration from gravity, shape (6,). None if
                not considering gravity (as is done to compute centrifugal/coriolis)
            F_ext (Optional[Array]): External wrenches on each link (expressed in the root/world frame),
                shape (num_joints, 6). This is currently not used for most methods and can be set to None.

        Returns:
            Array: Joint torques, shape (num_joints,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._rnea_from_spatial_data(
            spatial_axes, spatial_inertias, qd, qdd, gravity_accel, F_ext
        )

    def _rnea_from_spatial_data(
        self,
        spatial_axes: Array,
        spatial_inertias: Array,
        qd: Optional[Array],
        qdd: Optional[Array],
        gravity_accel: Optional[Array],
        F_ext: Optional[Array],
    ) -> Array:
        """Helper function: Computes RNEA given precomputed spatial axes and inertias"""

        # FORWARD PASS

        spatial_accel = jnp.zeros((self.num_joints, 6))
        if gravity_accel is not None:
            spatial_accel += gravity_accel[None, :]

        if qd is not None:
            s_qd = spatial_axes * qd[:, None]  # Helper
            # Spatial velocities for every link, summed over contributions from ancestors
            spatial_vel = self.ancestor_mask @ s_qd
            # Spatial accelerations for every link, summed over contributions from ancestors
            spatial_accel += self.ancestor_mask @ spatial_motion_cross(
                spatial_vel, s_qd
            )
        else:
            spatial_vel = jnp.zeros((self.num_joints, 6))

        if qdd is not None:
            spatial_accel += self.ancestor_mask @ (spatial_axes * qdd[:, None])

        # Newton-Euler (part 1): I * a term
        link_forces = jnp.einsum("ijk,ik->ij", spatial_inertias, spatial_accel)

        # Newton-Euler (part 2): v x I * v term
        if qd is not None:
            Iv = jnp.einsum("ijk,ik->ij", spatial_inertias, spatial_vel)
            link_forces += spatial_force_cross(spatial_vel, Iv)

        if F_ext is not None:
            link_forces -= F_ext

        # BACKWARD PASS

        # Sum link forces back towards the root based on ancestor relationship
        net_forces = self.ancestor_mask.T @ link_forces

        # Project forces back onto the joint axes to yield the torques
        return jnp.einsum("ij,ij->i", spatial_axes, net_forces)

    def crba(self, q: Array) -> Array:
        """Composite Rigid Body Algorithm (vectorized form)

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Mass matrix, shape (num_joints, num_joints)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._crba_from_spatial_data(spatial_axes, spatial_inertias)

    def _crba_from_spatial_data(
        self, spatial_axes: Array, spatial_inertias: Array
    ) -> Array:
        """Helper function: Computes CRBA given precomputed spatial axes and inertias"""

        # BACKWARD PASS

        # Sum inertias torwards the root based on ancestor relationship
        composite_inertias = jnp.einsum(
            "ij,jkl->ikl", self.ancestor_mask.T, spatial_inertias
        )

        # Compute all potential inertial coupling between all pairs of joints...
        M_all = jnp.einsum(
            "ij,ijk,lk->il", spatial_axes, composite_inertias, spatial_axes
        )
        # ... then mask out the terms that don't have a parent/child relationship
        M_lower = self.ancestor_mask * M_all

        # Symmetrize the mass matrix from the lower triangular portion
        return M_lower + jnp.tril(M_lower, k=-1).T

    def _spatial_axes_and_inertias(
        self, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        """Helper function for CRBA and RNEA: Computes the spatial joint axes and link inertias from FK

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)

        Returns:
            Tuple[Array, Array]:
                spatial_axes (Array): shape (num_joints, 6)
                spatial_inertias (Array): shape (num_joints, 6, 6)
        """
        # Convert static data to jax arrays
        joint_axes_local = jnp.asarray(self.joint_axes)
        link_masses = jnp.asarray(self.link_masses)
        link_local_inertias = jnp.asarray(self.link_local_inertias)

        spatial_axes = get_spatial_joint_axes(
            joint_transforms, joint_axes_local, self.revolute_mask
        )
        link_transforms = self._link_to_world_transforms(joint_transforms)
        spatial_inertias = get_spatial_inertias(
            link_masses, link_local_inertias, link_transforms
        )
        return spatial_axes, spatial_inertias

    def forward_dynamics(
        self, q: Array, qd: Array, tau: Array, fext: Optional[Array]
    ) -> Array:
        """Compute the joint acceleration resulting from an applied torque (and optionally,
        any external forces acting on the links), given the joint state

        Note: gravity is assumed always applied (for now)

        Args:
            q (Array): Joint positions, shape (num_joints,)
            qd (Array): Joint velocities, shape (num_joints,)
            tau (Array): Joint torques, shape (num_joints,)
            fext (Optional[Array]): External wrenches on each link (expressed in the root/world frame),
                shape (num_joints, 6). Set to None if no external forces are applied

        Returns:
            Array: Joint accelerations, shape (num_joints,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._forward_dynamics(joint_transforms, qd, tau, fext)

    def _forward_dynamics(
        self,
        joint_transforms: Array,
        qd: Array,
        tau: Array,
        fext: Optional[Array],
    ) -> Array:
        """Helper function: Computes the forward dynamics from FK"""
        # Perform a single evaluation of the spatial axes/inertias
        # and use in both CRBA and RNEA
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        M = self._crba_from_spatial_data(spatial_axes, spatial_inertias)
        g_accel = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        bias = self._rnea_from_spatial_data(
            spatial_axes,
            spatial_inertias,
            qd=qd,
            gravity_accel=g_accel,
            qdd=None,
            F_ext=fext,
        )
        # TODO: Decide if it's better to use a cho_factor + cho_solve combo here
        return jsp.linalg.solve(M, tau - bias, assume_a="pos")
