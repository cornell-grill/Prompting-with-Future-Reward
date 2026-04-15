from reward.reward_helpers import *


def should_release(context, stage, env_id=0):
    return False


def should_grasp(context, prev_context, stage, env_id=0):
    return False


def compute_reward(context, prev_context, stage, env_id=0):
    reward = 0.0

    objects = context.get("objects", {})
    charger = objects.get("charger")

    if charger is None or prev_context is None:
        return reward

    # STEP-BASED UNPLUG PROGRESS (KEY CHANGE)
    prev_pos = prev_context.get("objects").get("charger").get("position")[env_id]
    curr_pos = charger.get("position")[env_id]

    z_delta = curr_pos[2] - prev_pos[2]

    # only reward upward motion
    reward += max(0.0, z_delta) * 10.0

    # success bonus (absolute condition is OK here)
    if z_delta >= 0.015:
        reward += 100.0

    return reward