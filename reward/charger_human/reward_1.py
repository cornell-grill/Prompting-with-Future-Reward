from reward.reward_helpers import *


def should_release(context, stage, env_id=0):
    return False


def should_grasp(context, prev_context, stage, env_id=0):
    charger = context.get("objects").get("charger")
    prev_charger = prev_context.get("objects").get("charger")

    charger_pos = charger.get("position")[env_id]
    prev_pos = prev_charger.get("position")[env_id]

    # lift signal (same pattern as cucumber)
    z_lift = charger_pos[2] - prev_pos[2]

    is_grasping = context.get("gripper").get("is_grasping")[env_id]

    return is_grasping and z_lift > 0.01


def compute_reward(context, prev_context, stage, env_id=0):
    reward = 0.0

    objects = context.get("objects", {})
    charger = objects.get("charger")
    gripper = context.get("gripper", {})

    # STEP PROGRESS (cucumber style)
    if charger is not None and prev_context is not None:
        prev_pos = prev_context.get("objects").get("charger").get("position")[env_id]
        curr_pos = charger.get("position")[env_id]

        z_delta = curr_pos[2] - prev_pos[2]
        reward += max(0.0, z_delta) * 10.0

    # proximity shaping
    charger_pos = np.array(charger.get("position")[env_id])
    gripper_pos = np.array(gripper.get("position")[env_id])

    dist = euclid_distance(gripper_pos, charger_pos)
    reward += max(0.0, 0.5 - dist) * 5.0

    # grasp bonus
    is_grasping = gripper.get("is_grasping", False)
    if isinstance(is_grasping, (list, np.ndarray)):
        is_grasping = is_grasping[env_id]

    if is_grasping and gripper.get("grasped_object") == "charger":
        reward += 50.0

    return reward