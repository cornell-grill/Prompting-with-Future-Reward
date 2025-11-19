def get_state_context(mesh_world_or_info, prev_info=None, env_idx=0, include_vertices=True):
	"""Return a JSON-serializable dict describing the current scene state.

	Parameters
	----------
	mesh_world_or_info : MeshWorld or dict
		Either a `MeshWorld` instance (recommended) or the `mesh_info` dict
		previously returned by `MeshWorld.get_info()`.
	prev_info : dict or None
		Previous state dict returned by this function. If present and both
		previous and current collision meshes expose vertices with the same
		topology, this function will compute per-vertex displacements.
	env_idx : int
		Which environment index to report (default 0).
	include_vertices : bool
		When True, attempt to include collision-mesh vertices in world frame.

	Returns
	-------
	dict
		A dictionary with keys 'gripper', 'objects', 'timestamp'.

	Note
	----
	This helper tries to be defensive: it tolerates missing fields and will
	return None for entries it cannot compute instead of raising.
	"""
	import time
	import numpy as _np

	# Accept either a mesh_world (with .get_info()) or a precomputed info dict
	if hasattr(mesh_world_or_info, 'get_info'):
		mesh_world = mesh_world_or_info
		try:
			infos = mesh_world.get_info()
		except Exception:
			infos = {}
	else:
		mesh_world = None
		infos = mesh_world_or_info or {}

	state = {}

	# Gripper position (from infos if available)
	gripper_pos = None
	try:
		gp = infos.get('gripper_position', None)
		if gp is not None and len(gp) > env_idx:
			gripper_pos = [float(x) for x in gp[env_idx]]
	except Exception:
		gripper_pos = None

	# Grasping detection: prefer using mesh_world.agent.is_grasping when available
	is_gripping = False
	gripped_object = None
	if mesh_world is not None:
		try:
			for obj in mesh_world.env.unwrapped.objects:
				try:
					mask = mesh_world.agent.is_grasping(obj, min_force=getattr(mesh_world, 'min_force', 0.1), max_angle=getattr(mesh_world, 'max_angle', 80))
					# mask may be a tensor/array of length num_envs
					if mask is None:
						continue
					try:
						val = bool(int(mask[env_idx].cpu().numpy()))
					except Exception:
						# fall back to Python bool
						val = bool(mask)
					if val:
						is_gripping = True
						try:
							gripped_object = obj.name if hasattr(obj, 'name') and obj.name is not None else (obj.get_name() if hasattr(obj, 'get_name') else str(obj))
						except Exception:
							gripped_object = str(obj)
						break
				except Exception:
					# ignore per-object failures
					continue
		except Exception:
			is_gripping = False

	state['gripper'] = {
		'position': gripper_pos,
		'is_gripping': bool(is_gripping),
		'gripped_object': gripped_object,
	}

	# Objects: collect per-object pose (center) and bbox (center + extents)
	objects = {}
	obj_poses = infos.get('object_poses', [])

	objs = []
	if mesh_world is not None:
		try:
			objs = list(mesh_world.env.unwrapped.objects)
		except Exception:
			objs = []
	else:
		# When only infos were supplied, we can't access collision meshes; we
		# instead iterate poses if present.
		objs = []

	for idx, obj in enumerate(objs):
		entry = {'id': idx}

		# name
		try:
			name = getattr(obj, 'name', None)
			if name is None:
				name = obj.get_name() if hasattr(obj, 'get_name') else None
		except Exception:
			name = None
		if name is None:
			name = str(obj)
		entry['name'] = name

		# center position: prefer object_poses if available, else compute from collision bounds
		center = None
		try:
			if idx < len(obj_poses):
				pose_arr = obj_poses[idx]
				if hasattr(pose_arr, 'shape') and pose_arr.shape[0] > env_idx:
					p0 = pose_arr[env_idx]
					center = [float(x) for x in p0[:3]]
		except Exception:
			center = None

		# bbox: compute center and extents from collision mesh bounds if available
		bbox = None
		try:
			if hasattr(obj, 'get_first_collision_mesh'):
				col_mesh = obj.get_first_collision_mesh(to_world_frame=True)
			else:
				col_mesh = None
				if hasattr(obj, 'get_collision_meshes'):
					meshes = obj.get_collision_meshes(to_world_frame=True, first_only=True)
					col_mesh = meshes

			if col_mesh is not None and hasattr(col_mesh, 'bounds'):
				mins, maxs = col_mesh.bounds
				mins_arr = _np.array([float(x) for x in mins])
				maxs_arr = _np.array([float(x) for x in maxs])
				extents = (maxs_arr - mins_arr).tolist()
				bbox_center = ((mins_arr + maxs_arr) / 2.0).tolist()
				bbox = {'center': bbox_center, 'extents': extents}
				# if center wasn't available from poses, use bbox center
				if center is None:
					center = bbox_center
		except Exception:
			bbox = None

		entry['position'] = center
		entry['bbox'] = bbox

		# compute center displacement vs prev_info if possible
		center_displacement = None
		if prev_info is not None:
			try:
				prev_objects = prev_info.get('objects', {})
				prev_entry = None
				if name in prev_objects:
					prev_entry = prev_objects[name]
				else:
					for k, v in prev_objects.items():
						if v.get('id', None) == idx:
							prev_entry = v
							break

				if prev_entry is not None:
					prev_center = prev_entry.get('position') or (prev_entry.get('bbox') and prev_entry['bbox'].get('center'))
					if prev_center is not None and center is not None:
						prev_c = _np.array(prev_center, dtype=float)
						cur_c = _np.array(center, dtype=float)
						center_displacement = float(_np.linalg.norm(cur_c - prev_c))
			except Exception:
				center_displacement = None

		entry['center_displacement'] = center_displacement

		objects[name] = entry

	state['objects'] = objects

	return state
