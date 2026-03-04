from reward.reward_helpers import *


KEEP_GRIPPER_CLOSED = False

subgoals = ['Grasp the green cucumber', 'Move the green cucumber directly above the basket', 'Release the green cucumber into the basket']


def determine_stage(context):
    """Determine current subgoal stage based on state.

    Stage 1: Not grasping cucumber (need to pick it up)
    Stage 2: Grasping cucumber but not over basket (need to move over basket)
    Stage 3: Cucumber over basket, ready to release

    Args:
        context: Current state context dict
        prev_context: Previous state context (optional, for transition detection)

    Returns:
        int: Subgoal stage (1, 2, or 3)
    """
    gripper = context.get("gripper", {})
    is_grasping = gripper.get("is_grasping")[0]

    objects = context.get("objects", {})
    cucumber = objects.get("cucumber")
    basket = objects.get("basket")

    print("is_grasping: ", is_grasping)

    # Stage 1: Not grasping cucumber
    if not is_grasping:
        return 1

    # If we're grasping cucumber, check if it's over basket
    cucumber_pos = cucumber.get("position")[0]

    basket_bbox = basket.get("bbox")
    basket_center = basket.get("position")[0]
    horizontal_distance = euclid_distance(
        cucumber_pos[:2], basket_center[:2]  # x, y only
    )
    basket_radius = min(basket_bbox[0], basket_bbox[1]) / 2.0

    return 3 if horizontal_distance <= basket_radius and cucumber_pos[2] >= basket_center[2] else 2


def determine_success(context):
    """Determine if the task is successful based on the state."""
    cucumber = context.get("objects").get("cucumber")
    basket = context.get("objects").get("basket")
    grasping = context.get("gripper").get("is_grasping")[0]

    return (
        within_object(cucumber, basket, first_env=True)
        and not grasping
    )
