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
    return False


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

    # Subgoal 2: Move cucumber over basket
    # Reward horizontal proximity to basket AND being above it
    if cucumber is not None and basket is not None:
        cucumber_pos = cucumber.get("position")
        basket_extents = basket.get("bbox")
        basket_center = basket.get("position")
        
        # Extract environment-specific positions
        cucumber_pos = np.array(cucumber_pos)
        basket_center = np.array(basket_center)
        if cucumber_pos.ndim > 1:
            cucumber_pos = cucumber_pos[env_id]
        if basket_center.ndim > 1:
            basket_center = basket_center[env_id]
        
        basket_radius = min(basket_extents[0], basket_extents[1]) / 2.0
        # Horizontal distance (x, y only)
        horizontal_distance = euclid_distance(
            cucumber_pos[:2], basket_center[:2]
        )

        # Reward horizontal proximity (within 2x basket radius)
        max_horizontal_distance = basket_radius * 2.0
        if horizontal_distance <= max_horizontal_distance:
            # Reward decreases with horizontal distance
            horizontal_reward = (
                max(
                    0.0,
                    (max_horizontal_distance - horizontal_distance)
                    / max_horizontal_distance,
                )
                * 10.0
            )
            reward += horizontal_reward

            # Reward being above the basket (positive Z difference)
            z_diff = cucumber_pos[2] - basket_center[2]
            if z_diff > 0:
                height_reward = min(
                    z_diff * 20.0, 30.0
                )  # Cap at 30 for being 1.5 units above
                reward += height_reward

    return reward
