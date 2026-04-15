from reward.reward_helpers import *

KEEP_GRIPPER_CLOSED = False

subgoals = ['Grasp the charger plug', 'Pull the charger out of the power strip']


def determine_stage(context):
    gripper = context.get("gripper", {})
    is_grasping = gripper.get("is_grasping")[0]

    if not is_grasping:
        return 1
    return 2


def determine_success(context):
    charger = context.get("objects").get("charger")

    # SAFE initial state access (prevents crash)
    if context.get("initial_state") is None:
        return False

    charger_pos = charger.get("position")[0]
    initial_pos = context.get("initial_state").get("charger").get("position")[0]

    z_displacement = charger_pos[2] - initial_pos[2]

    return z_displacement > 0.015