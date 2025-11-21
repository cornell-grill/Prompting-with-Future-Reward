from meshes.mesh_world import MeshWorld
import numpy as np

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
			# This is the first position out of 81 in gp. Unusure if it is the ideal selection.
			gripper_pos = [float(x) for x in gp[env_idx]]
	except Exception:
		gripper_pos = None

	is_gripping = False
	gripped_object = None
	try:
		for obj in mesh_world.env.unwrapped.objects:
			try:
				grasping = mesh_world.agent.is_grasping(obj, min_force=getattr(mesh_world, 'min_force', 0.1), max_angle=getattr(mesh_world, 'max_angle', 80))
				# grasping may be a tensor/array of length num_envs
				if grasping is None:
					continue
				val = bool(int(grasping[env_idx].cpu().numpy()))
				if val:
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
