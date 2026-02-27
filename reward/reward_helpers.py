import numpy as np
import os
import json

def save_context(context, filename, folderpath="."):
        """Save context dictionary to a JSON file."""
        os.makedirs(folderpath, exist_ok=True)
        
        # Ensure filename ends with .json
        if not filename.endswith(".json"):
            filename += ".json"
        
        filepath = os.path.join(folderpath, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        serializable_context = convert_to_serializable(context)
        
        with open(filepath, "w") as f:
            json.dump(serializable_context, f, indent=2)

def reduce_context(context, upto_env):
    """Reduce context dictionary to only include envs up to upto_envs"""
    context["gripper"]["position"] = context["gripper"]["position"][:upto_env]
    context["gripper"]["is_grasping"] = context["gripper"]["is_grasping"][:upto_env]
    for obj_name in context["objects"]:
        context["objects"][obj_name]["position"] = context["objects"][obj_name]["position"][:upto_env]
    return context

import numpy as np


def euclid_distance(a, b):
    """
    Compute the Euclidean distance between two points input as lists or tuples.
    """
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))


def compute_displacement(current_obj, prev_context, env_id=0):
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


def penalize_movement(object, prev_context, env_id=0):
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

    displacement = euclid_distance(current_pos, initial_pos)
    # Penalty scales with displacement (multiply by 10 to make it significant)
    return displacement * 10.0


def within_object(obj, other, first_env=False):
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
