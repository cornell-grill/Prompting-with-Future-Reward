import numpy as np


def euclid_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))


def compute_displacement(current_obj, prev_state_context):
    """Compute displacement of an object from previous state.
    
    Args:
        current_obj: Current object dict with 'position' key
        prev_state_context: Previous state context dict
    
    Returns:
        np.array: [dx, dy, dz] displacement vector, or None if cannot compute
    """
    if current_obj is None or prev_state_context is None:
        return None
    
    current_pos = current_obj.get('position')    
    obj_name = current_obj.get('name')
    if obj_name is None or current_pos is None:
        return None
    
    prev_objects = prev_state_context.get('objects', {})
    prev_obj = prev_objects.get(obj_name)
    
    if prev_obj is None:
        obj_id = current_obj.get('id')
        if obj_id is not None:
            for k, v in prev_objects.items():
                if v.get('id') == obj_id:
                    prev_obj = v
                    break
    
    if prev_obj is None:
        return None
    
    prev_pos = prev_obj.get('position')
    if prev_pos is None:
        # Try bbox center
        prev_bbox = prev_obj.get('bbox')
        if prev_bbox:
            prev_pos = prev_bbox.get('center')
    
    if prev_pos is None:
        return None
    
    current_pos = np.array(current_pos)
    prev_pos = np.array(prev_pos)
    return current_pos - prev_pos


def penalize_movement(object, prev_state_context):
    """Compute penalty for object movement from initial position.
    
    Args:
        object: Current object dict
        prev_state_context: Previous state context dict
    
    Returns:
        float: Penalty value (positive, higher = more movement)
    """
    if object is None or prev_state_context is None:
        return 0.0
    
    current_pos = object.get('position')
    if current_pos is None:
        bbox = object.get('bbox')
        if bbox:
            current_pos = bbox.get('center')
    
    if current_pos is None:
        return 0.0
    
    # Find object in initial state
    initial_objects = prev_state_context.get('objects', {})
    initial_object = initial_objects.get(object.get('name'))
    
    if initial_object is None:
        obj_id = object.get('id')
        if obj_id is not None:
            for k, v in initial_objects.items():
                if v.get('id') == obj_id:
                    initial_object = v
                    break
    
    if initial_object is None:
        return 0.0
    
    initial_pos = initial_object.get('position')
    if initial_pos is None:
        initial_bbox = initial_object.get('bbox')
        if initial_bbox:
            initial_pos = initial_bbox.get('center')
    
    if initial_pos is None:
        return 0.0
    
    displacement = euclid_distance(current_pos, initial_pos)
    # Penalty scales with displacement (multiply by 10 to make it significant)
    return displacement * 10.0


def within_object(obj, other):
    """Check if obj is spatially contained within other bbox.
    
    Args:
        obj: object dict with 'position' or 'bbox'
        other: other object dict with 'bbox'
    
    Returns:
        bool: True if obj is in other, False otherwise
    """
    if obj is None or other is None:
        return False
    
    obj_pos = obj.get('position')
    if obj_pos is None:
        obj_bbox = obj.get('bbox')
        if obj_bbox:
            obj_pos = obj_bbox.get('center')
    
    other_bbox = other.get('bbox')
    if other_bbox is None or obj_pos is None:
        return False
    
    other_center = other_bbox.get('center')
    other_extents = other_bbox.get('extents')
    
    if other_center is None or other_extents is None:
        return False
    
    # Check if obj position is within other bbox
    obj_pos = np.array(obj_pos)
    other_center = np.array(other_center)
    other_extents = np.array(other_extents)
    
    other_min = other_center - other_extents / 2.0
    other_max = other_center + other_extents / 2.0
    
    in_x = other_min[0] <= obj_pos[0] <= other_max[0]
    in_y = other_min[1] <= obj_pos[1] <= other_max[1]
    in_z = other_min[2] <= obj_pos[2] <= other_max[2]
    
    return in_x and in_y and in_z


def determine_subgoal_stage(state_context):
    """Determine current subgoal stage based on state.
    
    Stage 1: Not grasping cucumber (need to pick it up)
    Stage 2: Grasping cucumber but not over basket (need to move over basket)
    Stage 3: Cucumber over basket, ready to release
    
    Args:
        state_context: Current state context dict
        prev_state_context: Previous state context (optional, for transition detection)
    
    Returns:
        int: Subgoal stage (1, 2, or 3)
    """
    if state_context is None:
        return 1
    
    gripper = state_context.get('gripper', {})
    is_grasping = gripper.get('is_grasping', False)
    gripped_object = gripper.get('gripped_object')
    
    objects = state_context.get('objects', {})
    cucumber = objects.get('cucumber')
    basket = objects.get('basket')
    
    # Stage 1: Not grasping cucumber
    if not is_grasping or gripped_object != 'cucumber':
        return 1
    
    # If we're grasping cucumber, check if it's over basket
    if cucumber is not None and basket is not None:
        cucumber_pos = cucumber.get('position')
        if cucumber_pos is None:
            cucumber_bbox = cucumber.get('bbox')
            if cucumber_bbox:
                cucumber_pos = cucumber_bbox.get('center')
        
        basket_bbox = basket.get('bbox')
        if basket_bbox:
            basket_center = basket_bbox.get('center')
            
            if cucumber_pos is not None and basket_center is not None:
                horizontal_distance = euclid_distance(
                    cucumber_pos[:2],  # x, y only
                    basket_center[:2]
                )
                basket_extents = basket_bbox.get('extents', [0, 0, 0])
                basket_radius = min(basket_extents[0], basket_extents[1]) / 2.0
                
                if horizontal_distance <= basket_radius:
                    if cucumber_pos[2] >= basket_center[2] - 0.05:  # Allow slight tolerance
                        return 3
    
    # Stage 2: Grasping cucumber but not yet over basket
    return 2


def compute_reward(state_context, prev_state_context, subgoal_stage):
    """Compute reward for a given state based on current subgoal.
    
    Args:
        state_context: Current state context dict
        prev_state_context: Previous state context dict (for computing displacements)
        subgoal_stage: Current subgoal stage (1, 2, or 3)
        prev_state_context: Initial state context (for basket movement penalty)
    
    Returns:
        float: Reward value (higher is better)
    """
    if state_context is None:
        return 0.0
    
    reward = 0.0
    
    # Get object information
    objects = state_context.get('objects', {})
    cucumber = objects.get('cucumber')
    basket = objects.get('basket')
    gripper = state_context.get('gripper', {})
    
    # Penalty for basket movement (applies to all stages)
    if basket is not None and prev_state_context is not None:
        basket_penalty = penalize_movement(basket, prev_state_context)
        reward -= basket_penalty
    
    # Subgoal-specific rewards
    if subgoal_stage == 1:
        # Subgoal 1: Pick up cucumber
        # Reward positive Z-displacement of cucumber (lifting it up)
        if cucumber is not None and prev_state_context is not None:
            displacement = compute_displacement(cucumber, prev_state_context)
            if displacement is not None:
                # Reward upward movement (positive Z)
                z_displacement = displacement[2]
                if z_displacement > 0:
                    reward += z_displacement * 10.0  # Scale up for significant reward

        # Encourage the gripper to move closer to the cucumber
        cucumber_pos = None
        if cucumber is not None:
            cucumber_pos = cucumber.get('position')
            if cucumber_pos is None:
                cucumber_bbox = cucumber.get('bbox')
                if cucumber_bbox:
                    cucumber_pos = cucumber_bbox.get('center')

        gripper_pos = gripper.get('position')
        if gripper_pos is not None and cucumber_pos is not None:
            distance = euclid_distance(gripper_pos, cucumber_pos)
            # Provide a small bonus that increases as the distance shrinks.
            # Cap the effect at proximity_cap so that being within ~0.5m gives positive reward
            proximity_cap = 0.5
            proximity_weight = 5.0
            proximity_bonus = max(0.0, proximity_cap - distance) * proximity_weight
            reward += proximity_bonus
        
        # Large bonus for successful grasp
        if gripper.get('is_grasping', False) and gripper.get('gripped_object') == 'cucumber':
            reward += 50.0
    
    elif subgoal_stage == 2:
        # Subgoal 2: Move cucumber over basket
        # Reward horizontal proximity to basket AND being above it
        if cucumber is not None and basket is not None:
            cucumber_pos = cucumber.get('position')
            if cucumber_pos is None:
                cucumber_bbox = cucumber.get('bbox')
                if cucumber_bbox:
                    cucumber_pos = cucumber_bbox.get('center')
            
            basket_bbox = basket.get('bbox')
            if basket_bbox:
                basket_center = basket_bbox.get('center')
                basket_extents = basket_bbox.get('extents', [0, 0, 0])
                basket_radius = min(basket_extents[0], basket_extents[1]) / 2.0
                
                if cucumber_pos is not None and basket_center is not None:
                    # Horizontal distance (x, y only)
                    horizontal_distance = euclid_distance(
                        cucumber_pos[:2],
                        basket_center[:2]
                    )
                    
                    # Reward horizontal proximity (within 2x basket radius)
                    max_horizontal_distance = basket_radius * 2.0
                    if horizontal_distance <= max_horizontal_distance:
                        # Reward decreases with horizontal distance
                        horizontal_reward = max(0.0, (max_horizontal_distance - horizontal_distance) / max_horizontal_distance) * 10.0
                        reward += horizontal_reward
                    
                    # Reward being above the basket (positive Z difference)
                    z_diff = cucumber_pos[2] - basket_center[2]
                    if z_diff > 0:
                        height_reward = min(z_diff * 20.0, 30.0)  # Cap at 30 for being 1.5 units above
                        reward += height_reward
    
    elif subgoal_stage == 3:
        # Subgoal 3: Release cucumber in basket
        # Large bonus if cucumber is in basket
        if within_object(cucumber, basket):
            reward += 100.0
        
        # Bonus for release (not grasping)
        if not gripper.get('is_grasping', False):
            reward += 20.0
    
    return reward


