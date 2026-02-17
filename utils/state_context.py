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