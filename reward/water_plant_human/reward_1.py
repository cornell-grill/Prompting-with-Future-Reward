from reward.reward_helpers import *
import numpy as np
from reward.water_plant_human.context import get_sideways_tilt_deg


def should_release(context, stage, env_id=0):
    return False


def should_grasp(context, prev_context, stage, env_id=0):
    cup = context.get("objects", {}).get("cup")
    prev_cup = prev_context.get("objects", {}).get("cup") if prev_context is not None else None
    gripper = context.get("gripper", {})

    if cup is None or prev_cup is None:
        return False

    is_grasping = bool(gripper.get("is_grasping")[env_id])
    cup_pos = np.array(cup.get("position"), dtype=float)[env_id]
    prev_cup_pos = np.array(prev_cup.get("position"), dtype=float)[env_id]

    return bool(is_grasping and cup_pos[2] > prev_cup_pos[2] + 0.015)


def compute_reward(context, prev_context, stage, env_id=0):
    reward = 0.0

    objects = context.get("objects", {})
    cup = objects.get("cup")
    plant = objects.get("plant")
    gripper = context.get("gripper", {})

    if cup is None or plant is None:
        return reward

    if prev_context is not None:
        plant_disp = compute_displacement(plant, prev_context, env_id)
        if plant_disp is not None:
            reward -= 10.0 * euclid_distance(plant_disp, np.zeros(3))

        cup_disp = compute_displacement(cup, prev_context, env_id)
        if cup_disp is not None and cup_disp[2] > 0:
            reward += 10.0 * cup_disp[2]

    cup_pos = np.array(cup.get("position"), dtype=float)[env_id]
    gripper_pos = np.array(gripper.get("position"), dtype=float)[env_id]
    distance = euclid_distance(gripper_pos, cup_pos)
    reward += 10.0 * max(0.0, (0.5 - distance) / 0.5)

    if bool(gripper.get("is_grasping")[env_id]):
        reward += 50.0

    cup_tilt_angle = get_sideways_tilt_deg(cup, env_id=env_id)
    reward += 10.0 * min(cup_tilt_angle, 30.0) / 30.0
    if cup_tilt_angle > 30.0:
        reward -= 50.0

    return reward
