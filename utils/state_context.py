from meshes.mesh_world import MeshWorld
import numpy as np
import os
import json

def save_env_states(mesh_world, filename, env_state_log_dir, context=None, max_envs=None):
    """Write per-environment states to env_state_logs_dir/<filename>.jsonl."""
    if mesh_world is None or env_state_log_dir is None:
        return
    os.makedirs(env_state_log_dir, exist_ok=True)
    total_envs = getattr(mesh_world, 'num_envs', 1)
    max_envs = total_envs if max_envs is None else min(max_envs, total_envs)
    log_path = os.path.join(env_state_log_dir, f"{filename}.jsonl")
    with open(log_path, "w") as fp:
        for env_idx in range(max_envs):
            record = {
                "env_idx": env_idx,
                "context": context or filename,
            }
            try:
                record["state"] = get_state_context(mesh_world, env_idx=env_idx)
            except Exception as exc:
                record["error"] = str(exc)
            fp.write(json.dumps(record))
            fp.write("\n")

def get_state_context(mesh_world : MeshWorld, env_idx=0):
	"""Return a JSON-serializable dict describing the current scene state.

	Args:
		mesh_world (MeshWorld): A `MeshWorld` instance.
		env_idx (int): Which environment index to report (default 0, the main environment).
	"""

	infos = mesh_world.get_info()

	state = {}

	gripper_pos = None
	try:
		gp = infos.get('gripper_position', None)
		if gp is not None and len(gp) > env_idx:
			gripper_pos = [float(x) for x in gp[env_idx]]
	except Exception:
		gripper_pos = None

	is_gripping = False
	gripped_object = None
	try:
		for obj in mesh_world.env.unwrapped.objects:
			try:
				grasping = mesh_world.agent.is_grasping(obj, min_force=getattr(mesh_world, 'min_force', 0.1), max_angle=getattr(mesh_world, 'max_angle', 80))
				if grasping is None:
					continue
				if grasping[env_idx].cpu().numpy():
					is_gripping = True
					gripped_object = obj.name
					break
			except Exception:
				continue
	except Exception:
		is_gripping = False

	state['gripper'] = {
		'position': gripper_pos,
		'is_gripping': bool(is_gripping),
		'gripped_object': gripped_object,
	}

	# Objects with position and bbox
	objects = {}
	obj_poses = infos.get('object_poses', [])


	for idx, obj in enumerate(mesh_world.env.unwrapped.objects):
		entry = {'id': idx}

		name = getattr(obj, 'name', None)
		entry['name'] = name

		center = None
		if idx < len(obj_poses):
			pose_arr = obj_poses[idx]
			if hasattr(pose_arr, 'shape') and pose_arr.shape[0] > env_idx:
				p0 = pose_arr[env_idx]
				center = [float(x) for x in p0[:3]]

		# bbox: compute center and extents from collision mesh bounds if available
		bbox = None
		try:
			col_mesh = obj.get_first_collision_mesh(to_world_frame=True)

			if col_mesh is not None and hasattr(col_mesh, 'bounds'):
				mins, maxs = col_mesh.bounds
				mins_arr = np.array([float(x) for x in mins])
				maxs_arr = np.array([float(x) for x in maxs])
				extents = (maxs_arr - mins_arr).tolist()
				bbox_center = ((mins_arr + maxs_arr) / 2.0).tolist()
				bbox = {'center': bbox_center, 'extents': extents}
				if center is None:
					center = bbox_center
		except Exception:
			bbox = None

		entry['position'] = center
		entry['bbox'] = bbox
		objects[name] = entry

	state['objects'] = objects

	return state
