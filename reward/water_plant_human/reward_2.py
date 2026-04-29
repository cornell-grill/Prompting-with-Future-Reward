from reward.reward_helpers import *
import numpy as np
from reward.water_plant_human.context import get_sideways_tilt_deg


def should_release(context, stage, env_id=0):
    return False


def should_grasp(context, prev_context, stage, env_id=0):
    return False


def compute_reward(context, prev_context, stage, env_id=0):
    reward = 0.0

    objects = context.get("objects", {})
    cup = objects.get("cup")
    plant = objects.get("plant")

    if cup is None or plant is None:
        return reward

    if prev_context is not None:
        plant_disp = compute_displacement(plant, prev_context, env_id)
        if plant_disp is not None:
            reward -= 10.0 * euclid_distance(plant_disp, np.zeros(3))

    cup_pos = np.array(cup.get("position"), dtype=float)[env_id]
    plant_pos = np.array(plant.get("position"), dtype=float)[env_id]
    plant_bbox = np.array(plant.get("bbox"), dtype=float)

    plant_radius = min(plant_bbox[0], plant_bbox[1]) / 2.0
    plant_height = plant_bbox[2]

    max_horizontal_distance = 2.0 * plant_radius
    horizontal_distance = euclid_distance(cup_pos[:2], plant_pos[:2])
    if max_horizontal_distance > 1e-8:
        reward += 10.0 * max(0.0, (max_horizontal_distance - horizontal_distance) / max_horizontal_distance)

    plant_top_z = plant_pos[2] + 0.5 * plant_height
    reward += 10.0 * min(1.0, (cup_pos[2] - plant_top_z) / max(plant_height, 1e-8))

    cup_tilt_angle = get_sideways_tilt_deg(cup, env_id=env_id)
    reward += 10.0 * min(cup_tilt_angle, 30.0) / 30.0
    if cup_tilt_angle > 30.0:
        reward -= 50.0

    return reward
