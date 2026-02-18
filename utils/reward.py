import numpy as np


def _euclid_distance(a, b):
    """
    Compute the Euclidean distance between two points input as lists or tuples.
    """
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))


def _compute_displacement(current_obj, prev_context, env_id=0):
    """Compute displacement of an object from previous state.

    Args:
        current_obj: Current object dict with 'position' key
        prev_context: Previous state context dict
        env_id: Environment ID for indexing

    Returns:
        np.array: [dx, dy, dz] displacement vector, or None if cannot compute
    """
    if current_obj is None or prev_context is None:
        return None

    current_pos = current_obj.get("position")
    obj_name = current_obj.get("name")
    if obj_name is None or current_pos is None:
        return None

    prev_objects = prev_context.get("objects", {})
    prev_obj = prev_objects.get(obj_name)

    if prev_obj is None:
        obj_id = current_obj.get("id")
        if obj_id is not None:
            for k, v in prev_objects.items():
                if v.get("id") == obj_id:
                    prev_obj = v
                    break

    if prev_obj is None:
        return None

    prev_pos = prev_obj.get("position")

    if prev_pos is None:
        return None

    current_pos = np.array(current_pos)
    prev_pos = np.array(prev_pos)
    
    # Handle multi-environment arrays
    if current_pos.ndim > 1:
        current_pos = current_pos[env_id]
    if prev_pos.ndim > 1:
        prev_pos = prev_pos[env_id]
    
    return current_pos - prev_pos


def _penalize_movement(object, prev_context, env_id=0):
    """Compute penalty for object movement from initial position.

    Args:
        object: Current object dict
        prev_context: Previous state context dict
        env_id: Environment ID for indexing

    Returns:
        float: Penalty value (positive, higher = more movement)
    """
    if object is None or prev_context is None:
        return 0.0

    current_pos = object.get("position")

    if current_pos is None:
        return 0.0

    # Find object in initial state
    initial_objects = prev_context.get("objects", {})
    initial_object = initial_objects.get(object.get("name"))

    if initial_object is None:
        obj_id = object.get("id")
        if obj_id is not None:
            for k, v in initial_objects.items():
                if v.get("id") == obj_id:
                    initial_object = v
                    break

    if initial_object is None:
        return 0.0

    initial_pos = initial_object.get("position")

    if initial_pos is None:
        return 0.0

    current_pos = np.array(current_pos)
    initial_pos = np.array(initial_pos)
    
    # Handle multi-environment arrays
    if current_pos.ndim > 1:
        current_pos = current_pos[env_id]
    if initial_pos.ndim > 1:
        initial_pos = initial_pos[env_id]

    displacement = _euclid_distance(current_pos, initial_pos)
    # Penalty scales with displacement (multiply by 10 to make it significant)
    return displacement * 10.0


def _within_object(obj, other, first_env=False):
    """Check if obj is spatially contained within other bbox.

    Args:
        obj: object dict with 'position' (numpy array of positions per env) or 'bbox'
        other: other object dict with 'bbox'
        first_env: if True, only check first environment; otherwise check all in parallel

    Returns:
        bool or np.ndarray: True/False if first_env=True, otherwise array of bools per env
    """
    obj_pos = obj.get("position")
    other_pos = other.get("position")
    other_bbox = other.get("bbox")
    
    obj_pos = np.array(obj_pos)
    other_pos = np.array(other_pos)
    other_extents = np.array(other_bbox)

    other_min = other_pos - other_extents / 2.0
    other_max = other_pos + other_extents / 2.0

    if first_env:
        # Only check first environment
        obj_pos = obj_pos[0]
        other_min = other_min[0] if other_min.ndim > 1 else other_min
        other_max = other_max[0] if other_max.ndim > 1 else other_max
        
        in_x = other_min[0] <= obj_pos[0] <= other_max[0]
        in_y = other_min[1] <= obj_pos[1] <= other_max[1]
        in_z = other_min[2] <= obj_pos[2] <= other_max[2]
        return in_x and in_y and in_z
    else:
        # Parallel check for all environments
        in_x = (other_min[:, 0] <= obj_pos[:, 0]) & (obj_pos[:, 0] <= other_max[:, 0])
        in_y = (other_min[:, 1] <= obj_pos[:, 1]) & (obj_pos[:, 1] <= other_max[:, 1])
        in_z = (other_min[:, 2] <= obj_pos[:, 2]) & (obj_pos[:, 2] <= other_max[:, 2])
        return in_x & in_y & in_z


KEEP_GRIPPER_CLOSED = False

NUM_SUBGOALS = 3

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
    horizontal_distance = _euclid_distance(
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
        _within_object(cucumber, basket, first_env=True)
        and not grasping
    )


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
    if stage < 3:
        return False

    cucumber_pos = context.get("objects").get("cucumber").get("position")[env_id]
    basket = context.get("objects").get("basket")
    basket_center = basket.get("position")[env_id]
    basket_bbox = basket.get("bbox")
    horizontal_distance = _euclid_distance(
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
    if stage != 1:
        return False
    
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
        basket_penalty = _penalize_movement(basket, prev_context, env_id)
        reward -= basket_penalty

    # Subgoal-specific rewards
    if stage == 1:
        # Subgoal 1: Pick up cucumber
        # Reward positive Z-displacement of cucumber (lifting it up)
        if cucumber is not None and prev_context is not None:
            displacement = _compute_displacement(cucumber, prev_context, env_id)
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
        
        distance = _euclid_distance(gripper_pos, cucumber_pos)
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

    elif stage == 2:
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
            horizontal_distance = _euclid_distance(
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

    elif stage == 3:
        # Subgoal 3: Release cucumber in basket
        # Large bonus if cucumber is in basket
        if _within_object(cucumber, basket, first_env=True):
            reward += 100.0

        # Bonus for release (not grasping)
        is_grasping = gripper.get("is_grasping", False)
        if isinstance(is_grasping, (list, np.ndarray)):
            is_grasping = is_grasping[env_id]
        
        if not is_grasping:
            reward += 20.0

    return reward
