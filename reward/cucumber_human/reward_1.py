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
    cucumber_pos = context.get("objects").get("cucumber").get("position")[env_id]
    prev_cucumber_pos = prev_context.get("objects").get("cucumber").get("position")[env_id]
    gripper_pos = context.get("gripper").get("position")[env_id]

    # TODO: What should these thesholds be?
    is_cucumber_lifted = cucumber_pos[2] > prev_cucumber_pos[2] + 0.01
    is_grasping = context.get("gripper").get("is_grasping")[env_id]

    return is_grasping and is_cucumber_lifted 
    # and _euclid_distance(gripper_pos, cucumber_pos) < 0.1

    
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

    # Subgoal 1: Pick up cucumber
    # Reward positive Z-displacement of cucumber (lifting it up)
    if cucumber is not None and prev_context is not None:
        displacement = compute_displacement(cucumber, prev_context, env_id)
        if displacement is not None:
            # Reward upward movement (positive Z)
            z_displacement = displacement[2]
            if z_displacement > 0:
                reward += z_displacement * 10.0  # Scale up for significant reward

    # Encourage the gripper to move closer to the cucumber
    cucumber_pos = cucumber.get("position")
    gripper_pos = gripper.get("position")
    # Extract environment-specific positions
    cucumber_pos = np.array(cucumber_pos)
    gripper_pos = np.array(gripper_pos)
    if cucumber_pos.ndim > 1:
        cucumber_pos = cucumber_pos[env_id]
    if gripper_pos.ndim > 1:
        gripper_pos = gripper_pos[env_id]
    
    distance = euclid_distance(gripper_pos, cucumber_pos)
    # Provide a small bonus that increases as the distance shrinks.
    # Cap the effect at proximity_cap so that being within ~0.5m gives positive reward
    proximity_cap = 0.5
    proximity_weight = 5.0
    proximity_bonus = max(0.0, proximity_cap - distance) * proximity_weight
    reward += proximity_bonus

    # Large bonus for successful grasp
    is_grasping = gripper.get("is_grasping", False)
    if isinstance(is_grasping, (list, np.ndarray)):
        is_grasping = is_grasping[env_id]
    
    if (
        is_grasping
        and gripper.get("grasped_object") == "cucumber"
    ):
        reward += 50.0

    return reward
