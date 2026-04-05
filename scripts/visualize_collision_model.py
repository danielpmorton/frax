"""Script to visualize the robot collision model in MuJoCo"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from frax.core.robot import Robot
from frax.assets import G1_ASSETS_DIR
from frax.utils.transform_utils import create_transform_numpy
from frax.utils.free_floating_utils import virtual_joints_to_pose_and_twist
from frax.robots.franka_panda import load_panda
from frax.robots.unitree_g1 import (
    load_g1,
)


def visualize_collision_model(
    xml_path: str,
    robot: Robot,
    q: np.ndarray,
):
    xml_path = str(xml_path)

    positions = robot.collision_positions
    radii = robot.collision_radii
    root_positions = robot.root_collision_positions
    root_radii = robot.root_collision_radii
    body_sc_pairs = robot.body_sc_pairs
    root_sc_pairs = robot.root_sc_pairs
    has_sc_data = robot.has_sc_data

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Set joint positions robustly by mapping names
    start_q_idx = 0
    if robot.includes_floating_dof:
        # Map first 6 DOFs to freejoint
        q_ff = q[:6]
        pos, quat_wxyz, _, _ = virtual_joints_to_pose_and_twist(q_ff, np.zeros(6))

        # Find the freejoint (typically first joint in humanoid models)
        ff_joint_id = -1
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                ff_joint_id = j
                break

        if ff_joint_id != -1:
            qaddr = model.jnt_qposadr[ff_joint_id]
            data.qpos[qaddr : qaddr + 3] = pos
            data.qpos[qaddr + 3 : qaddr + 7] = quat_wxyz
        else:
            print(
                "Warning: robot has floating base but no freejoint found in MuJoCo model."
            )

        start_q_idx = 6

    mapped_joints = 0
    for i in range(start_q_idx, robot.num_joints):
        name = robot.joint_names[i]
        val = q[i]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        # Try stripping common prefixes if direct match fails
        if jid == -1:
            for prefix in ["panda_", "g1_"]:
                if name.startswith(prefix):
                    jid = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, name[len(prefix) :]
                    )
                    if jid != -1:
                        break

        if jid != -1:
            qaddr = model.jnt_qposadr[jid]
            # Standard 1-DOF joints
            if model.jnt_type[jid] in [
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ]:
                data.qpos[qaddr] = val
                mapped_joints += 1

    print(f"Mapped {mapped_joints} joints to MuJoCo model.")

    mujoco.mj_forward(model, data)
    joint_transforms = robot.joint_to_world_transforms(q)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Determine the world-frame positions of the collision geometry
        all_body_sphere_positions = []
        all_root_sphere_positions = []

        # MuJoCo's user_scn is a fixed-size rendering buffer.
        # mjv_initGeom initializes a 'slot' in this buffer with default values.
        # mjv_connector then calculates the specific pose and scale for connection types.
        def add_marker(marker_type, size, pos, rgba, mat=None):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                return
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            if mat is None:
                mat = np.eye(3).flatten()
            mujoco.mjv_initGeom(geom, marker_type, size, pos, mat, rgba)
            viewer.user_scn.ngeom += 1
            return geom

        def add_line(pos1, pos2, rgba, width=1.0):
            geom = add_marker(mujoco.mjtGeom.mjGEOM_LINE, [0, 0, 0], [0, 0, 0], rgba)
            if geom is not None:
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_LINE, width, pos1, pos2
                )

        # Clear previous geoms
        viewer.user_scn.ngeom = 0

        print("Visualizing collision model in MuJoCo...")
        print("Close the viewer window to exit.")

        # Handle body spheres
        idx = 0
        idxs_used_for_sc = set()
        if has_sc_data:
            idxs_used_for_sc.update([i for pair in body_sc_pairs for i in pair])
            idxs_used_for_sc.update([pair[1] for pair in root_sc_pairs])

        for i in range(robot.num_joints):
            parent_to_world_tf = joint_transforms[i]
            for j in range(len(positions[i])):
                world_pos = (
                    parent_to_world_tf
                    @ create_transform_numpy(np.eye(3), positions[i][j])
                )[:3, 3]
                rgba = (1, 0, 0, 0.5) if idx in idxs_used_for_sc else (1, 1, 0, 0.5)

                add_marker(
                    mujoco.mjtGeom.mjGEOM_SPHERE, [radii[i][j], 0, 0], world_pos, rgba
                )
                all_body_sphere_positions.append(world_pos)
                idx += 1

        # Handle root spheres
        for pos, rad in zip(root_positions, root_radii):
            add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, [rad, 0, 0], pos, (1, 0, 0, 0.5))
            all_root_sphere_positions.append(pos)

        # Handle self collision lines
        if has_sc_data:
            line_rgba = np.array([1, 0, 0, 1], dtype=np.float32)
            for pair in body_sc_pairs:
                add_line(
                    all_body_sphere_positions[pair[0]],
                    all_body_sphere_positions[pair[1]],
                    line_rgba,
                )

            for pair in root_sc_pairs:
                pos1 = all_root_sphere_positions[pair[0]]
                pos2 = all_body_sphere_positions[pair[1]]
                add_line(pos1, pos2, line_rgba)

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


def panda_main():
    robot = load_panda()
    q = np.array([0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, 0.0])
    xml_path = Path(__file__).parent / "examples" / "xml" / "panda.xml"
    visualize_collision_model(xml_path, robot, q)


def g1_main():
    robot = load_g1()
    q = np.zeros(robot.num_joints)
    # Move the robot up a bit so it's not in the floor
    q[2] = 0.8
    xml_path = G1_ASSETS_DIR / "g1_29dof_rev_1_0.xml"
    visualize_collision_model(xml_path, robot, q)


if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "g1"], default="panda")
    args = parser.parse_args()

    if args.robot == "panda":
        panda_main()
    else:
        g1_main()
