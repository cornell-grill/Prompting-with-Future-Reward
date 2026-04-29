from reward.reward_helpers import *
import numpy as np


KEEP_GRIPPER_CLOSED = False

subgoals = [
    "Grasp the cup",
    "Move the cup directly above the plant",
    "Tilt the cup to pour water onto the plant",
]


def _x_rotation(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _y_rotation(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _z_rotation(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _cup_up_vector_from_euler_xyz_deg(cup, env_id=0):
    # rotation_euler stores relative xyz Euler angles (deg) from initial orientation.
    euler = np.array(cup.get("rotation_euler"), dtype=float)
    if euler.ndim > 1:
        euler = euler[env_id]

    rx, ry, rz = np.radians(euler)
    rot = _x_rotation(rx) @ _y_rotation(ry) @ _z_rotation(rz)
    return rot @ np.array([0.0, 0.0, 1.0])


def get_sideways_tilt_deg(cup, env_id=0):
    """Return tilt from upright (world +Z), ignoring yaw-only rotation."""
    cup_up = _cup_up_vector_from_euler_xyz_deg(cup, env_id=env_id)
    cup_up = np.array(cup_up, dtype=float)
    norm = np.linalg.norm(cup_up)
    if norm < 1e-8:
        return 0.0
    unit_up = cup_up / norm
    cos_theta = np.clip(unit_up[2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def determine_stage(context):
    gripper = context.get("gripper", {})
    is_grasping = bool(gripper.get("is_grasping")[0])

    objects = context.get("objects", {})
    cup = objects.get("cup")
    plant = objects.get("plant")

    # Stage 1: Not grasping the cup.
    if not is_grasping:
        return 1

    cup_pos = np.array(cup.get("position"), dtype=float)[0]
    plant_pos = np.array(plant.get("position"), dtype=float)[0]
    plant_bbox = np.array(plant.get("bbox"), dtype=float)
    plant_radius = min(plant_bbox[0], plant_bbox[1]) / 2.0

    horizontal_distance = euclid_distance(cup_pos[:2], plant_pos[:2])
    return 3 if horizontal_distance <= plant_radius and cup_pos[2] >= plant_pos[2] else 2


def determine_success(context):
    objects = context.get("objects", {})
    cup = objects.get("cup")
    plant = objects.get("plant")
    if cup is None or plant is None:
        return False

    cup_pos = np.array(cup.get("position"), dtype=float)[0]
    plant_pos = np.array(plant.get("position"), dtype=float)[0]
    plant_bbox = np.array(plant.get("bbox"), dtype=float)
    plant_radius = min(plant_bbox[0], plant_bbox[1]) / 2.0
    plant_height = float(plant.get("bbox")[2])
    plant_top_z = plant_pos[2] + 0.5 * plant_height
    horizontal_distance = euclid_distance(cup_pos[:2], plant_pos[:2])

    tilt_angle = get_sideways_tilt_deg(cup, env_id=0)
    return bool(
        horizontal_distance <= plant_radius
        and cup_pos[2] > plant_top_z
        and 90.0 <= tilt_angle <= 180.0
    )
