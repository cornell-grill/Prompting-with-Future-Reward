from reward.reward_helpers import *


def should_release(context, stage, env_id=0):
    """Determine if the gripper should release based on context
     at environment 0 and stage.

    Args:
        context: Current state context dict
        stage: Current subgoal stage
        env_id: Environment ID
    Returns:
        bool: True if the gripper should release
    """
    cucumber_pos = context.get("objects").get("cucumber").get("position")[env_id]
    basket = context.get("objects").get("basket")
    basket_center = basket.get("position")[env_id]
    basket_bbox = basket.get("bbox")
    horizontal_distance = euclid_distance(
        cucumber_pos[:2], basket_center[:2]  # x, y only
    )
    basket_radius = min(basket_bbox[0], basket_bbox[1]) / 2

    return horizontal_distance <= basket_radius and cucumber_pos[2] >= basket_center[2] 

def should_grasp(context, prev_context, stage, env_id=0):
    """Determine if the gripper should grasp based on context
     at environment 0 and stage.

    Args:
        context: Current state context dict
        prev_context: Previous state context dict
        stage: Current subgoal stage
        env_id: Environment ID
    Returns:
        bool: True if the gripper should grasp
    """
    return False


def compute_reward(context, prev_context, stage, env_id=0):
    """Compute reward for a given state based on current subgoal.

    Args:
        context: Current state context dict
        prev_context: Previous state context dict (for computing displacements)
        stage: Current subgoal stage
        env_id: Environment ID
    Returns:
        float: Reward value (higher is better)
    """
    reward = 0.0

    # Get object information
    objects = context.get("objects", {})
    cucumber = objects.get("cucumber")
    basket = objects.get("basket")
    gripper = context.get("gripper", {})

    # Penalty for basket movement (applies to all stages)
    if basket is not None and prev_context is not None:
        basket_penalty = penalize_movement(basket, prev_context, env_id)
        reward -= basket_penalty

    # Subgoal 3: Release cucumber in basket
    # Large bonus if cucumber is in basket
    if within_object(cucumber, basket, first_env=True):
        reward += 100.0

    # Bonus for release (not grasping)
    is_grasping = gripper.get("is_grasping", False)
    if isinstance(is_grasping, (list, np.ndarray)):
        is_grasping = is_grasping[env_id]
    
    if not is_grasping:
        reward += 20.0

    return reward
