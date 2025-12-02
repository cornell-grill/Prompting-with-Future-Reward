""" This file currently moves the gripper to grasp a cumber using state-based reward """
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import base64
import time
import argparse
import multiprocessing

from meshes.mesh_world import MeshWorld
from utils.prompt import *
from utils.camera import get_up_direction

from utils.prompt_gpt import *
from gaussians.gaussian_world import GaussianWorld

from pytorch3d.renderer import look_at_view_transform

from utils.state_context import get_state_context, save_env_states
from utils.reward import compute_reward, determine_subgoal_stage, determine_success


def robo4d_parse():
    parser = argparse.ArgumentParser(description="Robo4D")
    parser.add_argument("--scene_name", type=str, default="basket_world")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--image_size", type=int, default=500)
    parser.add_argument("--total_steps", type=int, default=10)
    parser.add_argument("--camera_view_id", type=int, default=1)
    parser.add_argument("--plane_action", action="store_true")
    parser.add_argument("--cem_iteration", type=int, default=3)
    parser.add_argument("--num_sample_each_group", type=int, default=6)
    parser.add_argument("--num_sample_actions", type=int, default=81)
    parser.add_argument("--num_sample_vlm", type=int, default=36)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--try_release", action="store_true")
    parser.add_argument("--replan", action="store_true")
    parser.add_argument("--testing", action="store_true")
    return parser

parser = robo4d_parse()
args = parser.parse_args()

# render settings
image_size = args.image_size
znear = 0.01
zfar = 100
FoV = 60

output_path = os.path.join('results', f'{"naive_reward"}/{args.scene_name}/{args.name}')
if not os.path.exists(output_path):
    os.makedirs(output_path)

state_out_path = os.path.join(output_path, "env_state_logs")
os.makedirs(state_out_path, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

distance = 1.5
scene_name = args.scene_name
robot_translation = [-0.45, 0.0, 0.0]
action_ids = [2]

max_replan = 5

gaussian_world = GaussianWorld(scene_name, parser, post_process=False)
center = np.array([0, 0, 0])
radius = gaussian_world.radius * distance

close_gripper = False
print('!!! Start Gripper Open !!!')

robot_uids = 'PandaRobotiqHand'

# camera config
elev = torch.tensor([-70, 0, 70, 0], device=device)
azim = torch.tensor([0, 70, 0, 0], device=device)
up = get_up_direction(elev, azim)
at = torch.tensor(center[None], device=device).float()

cameras_config = []
for i in range(4):
    cameras_config.append({
        'elev': elev[i].item(),
        'azim': azim[i].item(),
    })

# gaussian camera config
R_fixed, T_fixed = look_at_view_transform(dist=radius, elev=elev, azim=azim, up=up, at=at, device=device)
up[:, 0] = -up[:, 0]
at[:, 0] = -at[:, 0]
elev = 180 - elev
R_gaussian_fixed, T_gaussian_fixed = look_at_view_transform(dist=radius, elev=elev, azim=azim, up=up, at=at, device="cpu")
R_gaussian_fixed = R_gaussian_fixed.numpy()
T_gaussian_fixed = T_gaussian_fixed.numpy()

plane_action_dimensions = [[0, 2], [1, 2], [0, 2], [0, 1]]

# build mesh world
mesh_world = MeshWorld(scene_name, num_envs=args.num_sample_actions, scene_traslation=-np.array(robot_translation), radius=radius, \
                       image_size=image_size, record_video=args.record_video, robot_uids=robot_uids, need_render=True, dir=output_path, \
                        close_gripper=close_gripper, cameras_config=cameras_config)

success = False
trajectory = []
history = []
excute_frames = []
history_object_states = []
output_actions = []
replan_time = 0

initial_joint_angles = mesh_world.agent.robot.get_qpos()[0].cpu().numpy().tolist()
print('initial_joint_angles: ', initial_joint_angles)
trajectory.append(torch.tensor(initial_joint_angles))

encoded_image = None

robot_images, robot_depth_images = mesh_world.get_image_depth()
current_robot_images, current_robot_depths = robot_images, robot_depth_images
rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, rotate_num=4)
depthmaps[np.where(depthmaps == 0)] = zfar
current_robot_depths[np.where(current_robot_depths == 0)] = zfar

current_robot_images = current_robot_images[:, 0, ...]
current_robot_depths = current_robot_depths[:, 0, ..., 0]

robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

excute_frames.append(current_images[args.camera_view_id])

encoded_images = []
for i in range(len(current_images)):
    plt.imsave(f'{output_path}/subgoal_view_{i + 1}.png', current_images[i])
    with open(f'{output_path}/subgoal_view_{i + 1}.png', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    encoded_images.append([encoded_image])

view_id = args.camera_view_id
object_states = []
object_transformations = []

# Initialize state tracking for reward-based evaluation
initial_state_context = get_state_context(mesh_world, env_idx=0)
prev_state_context = initial_state_context.copy()
save_env_states(mesh_world, "initial_state", state_out_path, context={"phase": "initial"})

current_subgoal = determine_subgoal_stage(initial_state_context)

while len(trajectory) <= args.total_steps:
    grasp = release = False
    step_idx = max(len(trajectory) - 1, 0)
    save_env_states(mesh_world, f"step_{step_idx}_state_start", state_out_path, context={"phase": "loop_start", "step": step_idx})

    # render the current state
    robot_images, robot_depth_images = mesh_world.get_image_depth()
    current_robot_images, current_robot_depths = robot_images, robot_depth_images
    rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=object_transformations, rotate_num=4)
    depthmaps[np.where(depthmaps == 0)] = zfar
    current_robot_depths[np.where(current_robot_depths == 0)] = zfar

    current_robot_images = current_robot_images[:, 0, ...]
    current_robot_depths = current_robot_depths[:, 0, ..., 0]

    robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
    current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

    for i in range(len(current_images)):
        plt.imsave(f'{output_path}/{len(trajectory)}_view_{i + 1}.png', current_images[i])

    # Update current subgoal based on state
    current_state_context = get_state_context(mesh_world, env_idx=0)
    save_env_states(mesh_world, f"step_{step_idx}_current_state", state_out_path, context={"phase": "current_state", "step": step_idx})
    current_subgoal = determine_subgoal_stage(current_state_context)
    print(f'Current subgoal: {current_subgoal}')
    
    # Check for task success (all subgoals completed)
    if determine_success(current_state_context):
        print('!!! Success !!!')
        success = True
        break

    # Check if we should release (using reward-based decision)
    # This happens before CEM if we're already grasping and in subgoal 3
    if mesh_world.grasping_now and current_subgoal == 3:
        # Simulate release to check reward
        joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.release()
        save_env_states(mesh_world, f"step_{step_idx}_release_preview", state_out_path, context={"phase": "release_preview", "step": step_idx})
        
        # Get state after release simulation
        release_state = get_state_context(mesh_world, env_idx=0)
        release_reward = compute_reward(release_state, prev_state_context, current_subgoal)
        
        # Reset to before release for CEM
        mesh_world.env.unwrapped.set_state(mesh_world.history_states[-1])
        
        # Execute release if reward is high enough
        if release_reward > 50.0:
            print(f'Executing release with reward: {release_reward:.2f}')
            release = True
            output_actions.append('release')
            joint_angles_list, action_object_transformations, root_images, robot_depth_images = mesh_world.release(non_stop=True)
            save_env_states(mesh_world, f"step_{step_idx}_release_execute", state_out_path, context={"phase": "release_execute", "step": step_idx})
            
            # Update state and continue
            prev_state_context = get_state_context(mesh_world, env_idx=0)
            continue

    action_dimenstions = 6
    means = np.zeros(action_dimenstions)
    variances = np.ones(action_dimenstions) * 0.5
    translation_variance = 1.5
    # careful! rotation variance
    # rotation_variance = 1.5
    rotation_variance = 1.0
    # only move parallel to the image plane
    # if args.plane_action:
    #     variances[plane_action_dimensions[view_id]] = translation_variance
    # else:
    #     variances[: 3] = translation_variance

    variances[3:] = rotation_variance

    covariance = np.zeros((action_dimenstions, action_dimenstions))
    np.fill_diagonal(covariance, variances)

    # Track best grasp action across iterations
    best_grasp_filtered_idx = None  # Index into filtered grasp array for set_grasp_state
    
    for iteration in range(args.cem_iteration):
        prev_time = time.time()
        prompt = []

        samples = np.random.multivariate_normal(means, covariance, size=args.num_sample_actions)
        grasp = False
        release = False

        if not mesh_world.grasping_now:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, grasp_object_transformations, grasp_robot_images, grasp_robot_depth_images, is_grasping = mesh_world.sample_action_distribution_batch(samples, try_grasp=True)
            save_env_states(mesh_world, f"step_{step_idx}_iter_{iteration}_try_grasp", state_out_path, context={"phase": "try_grasp_sample", "step": step_idx, "iteration": iteration})
            
            # Compute rewards for regular actions
            rewards = []
            current_state_before_actions = get_state_context(mesh_world, env_idx=0)
            
            # Process regular (non-grasp) actions
            # Note: We reconstruct from transformations because try_grasp() modifies all environments
            # after regular actions, so querying would give us states after try_grasp, not after the action
            for act_id in range(len(action_object_transformations)):
                # Reconstruct state from transformations (transformations are relative to initial state)
                mock_state = initial_state_context.copy()
                
                if len(action_object_transformations.shape) == 3:
                    obj_transforms = action_object_transformations[act_id, :, :3].cpu().numpy()
                    objects = mock_state.get('objects', {})
                    obj_names = list(objects.keys())
                    
                    for obj_idx, obj_name in enumerate(obj_names):
                        if obj_idx < obj_transforms.shape[0]:
                            obj = objects.get(obj_name)
                            if obj:
                                initial_pos = obj.get('position') or (obj.get('bbox') and obj['bbox'].get('center'))
                                if initial_pos:
                                    new_pos = np.array(initial_pos) + obj_transforms[obj_idx]
                                    obj['position'] = new_pos.tolist()
                                    if obj.get('bbox'):
                                        obj['bbox']['center'] = new_pos.tolist()
                
                # Update gripper state from current state (for grasping detection)
                mock_state['gripper'] = current_state_before_actions.get('gripper', mock_state.get('gripper', {}))
                
                reward = compute_reward(mock_state, prev_state_context, current_subgoal)
                rewards.append(reward)
            
            # Process grasp actions if any
            if is_grasping.sum() > 0:
                print(f'{is_grasping.sum()} could grasp!')
                grasp_rewards = []
                grasp_indices = np.where(is_grasping)[0]
                
                for grasp_idx, orig_idx in enumerate(grasp_indices):
                    # Query actual state from the environment that executed this grasp action
                    grasp_state = get_state_context(mesh_world, env_idx=orig_idx)
                    
                    reward = compute_reward(grasp_state, prev_state_context, current_subgoal)
                    grasp_rewards.append(reward)
                
                # Replace rewards for grasp actions with grasp rewards
                for grasp_idx, orig_idx in enumerate(grasp_indices):
                    if orig_idx < len(rewards):
                        rewards[orig_idx] = grasp_rewards[grasp_idx]
                
                # Check if we should execute grasp (highest reward grasp action)
                best_grasp_idx = np.argmax(grasp_rewards)  # Index into filtered grasp array
                best_grasp_orig_idx = grasp_indices[best_grasp_idx]  # Original action index
                best_grasp_reward = grasp_rewards[best_grasp_idx]
                
                # Execute grasp if reward is high enough (threshold)
                if best_grasp_reward > 30.0:  # Threshold for successful grasp
                    grasp = True
                    # Store the filtered index (for set_grasp_state) and original index (for CEM update)
                    best_grasp_filtered_idx = best_grasp_idx  # Index into filtered array for set_grasp_state
                    means = post_samples[best_grasp_orig_idx]  # Use original index for CEM
                    print(f'Executing grasp with reward: {best_grasp_reward:.2f}')
                    break
            
            rewards = np.array(rewards)
            
            # Select elite samples based on reward (top 20%)
            num_elite = max(1, int(len(rewards) * 0.2))
            elite_indices = np.argsort(rewards)[-num_elite:]
            elite_samples = post_samples[elite_indices]
            
            # Update CEM distribution
            means = np.mean(elite_samples, axis=0)
            covariance = np.cov(elite_samples, rowvar=False)
            covariance += np.eye(covariance.shape[0]) * 0.01
            
            print(f'iteration: {iteration}, max reward: {np.max(rewards):.2f}, mean reward: {np.mean(rewards):.2f}, elite mean: {np.mean(rewards[elite_indices]):.2f}')
            
            if grasp:
                break

        elif args.try_release and mesh_world.grasping_now:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, release_object_transformations, release_robot_images, release_robot_depth_images = mesh_world.sample_action_distribution_batch(samples, try_release=True)
            save_env_states(mesh_world, f"step_{step_idx}_iter_{iteration}_try_release", state_out_path, context={"phase": "try_release_sample", "step": step_idx, "iteration": iteration})

            # Compute rewards for release actions
            rewards = []
            
            for release_id in range(len(release_object_transformations)):
                # Query actual state from the environment that executed this release action
                release_state = get_state_context(mesh_world, env_idx=release_id)
                
                reward = compute_reward(release_state, prev_state_context, current_subgoal)
                rewards.append(reward)
            
            rewards = np.array(rewards)
            
            # Check if we should execute release (highest reward release action)
            best_release_idx = np.argmax(rewards)
            best_release_reward = rewards[best_release_idx]
            
            # Execute release if reward is high enough (threshold for subgoal 3)
            if best_release_reward > 50.0 and current_subgoal == 3:
                release = True
                means = post_samples[best_release_idx]
                print(f'Executing release with reward: {best_release_reward:.2f}')
                break
            
            # Still update CEM distribution even if not releasing
            num_elite = max(1, int(len(rewards) * 0.2))
            elite_indices = np.argsort(rewards)[-num_elite:]
            elite_samples = post_samples[elite_indices]
            
            means = np.mean(elite_samples, axis=0)
            covariance = np.cov(elite_samples, rowvar=False)
            covariance += np.eye(covariance.shape[0]) * 0.01
            
            print(f'iteration: {iteration}, max reward: {np.max(rewards):.2f}, mean reward: {np.mean(rewards):.2f}')
            
            if release:
                break

        else:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images = mesh_world.sample_action_distribution_batch(samples)
            save_env_states(mesh_world, f"step_{step_idx}_iter_{iteration}_regular_sample", state_out_path, context={"phase": "regular_sample", "step": step_idx, "iteration": iteration})
            
            # Compute rewards for all sampled actions
            rewards = []
            current_state_before_actions = get_state_context(mesh_world, env_idx=0)
            
            # Query actual states from each environment (try_grasp is not called here, so states are accurate)
            for act_id in range(len(action_object_transformations)):
                action_state = get_state_context(mesh_world, env_idx=act_id)
                
                reward = compute_reward(action_state, prev_state_context, current_subgoal)
                rewards.append(reward)
            
            rewards = np.array(rewards)
            
            # Select elite samples based on reward (top 20%)
            num_elite = max(1, int(len(rewards) * 0.2))
            elite_indices = np.argsort(rewards)[-num_elite:]
            elite_samples = post_samples[elite_indices]
            
            # Update CEM distribution
            means = np.mean(elite_samples, axis=0)
            covariance = np.cov(elite_samples, rowvar=False)
            covariance += np.eye(covariance.shape[0]) * 0.01
            
            print(f'iteration: {iteration}, max reward: {np.max(rewards):.2f}, mean reward: {np.mean(rewards):.2f}, elite mean: {np.mean(rewards[elite_indices]):.2f}')

    # Execute the best action after CEM iterations
    joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.sample_action_distribution_batch(means[None], non_stop=True)
    save_env_states(mesh_world, f"step_{step_idx}_execute", state_out_path, context={"phase": "execute_action", "step": step_idx})
    
    # Save actions
    output_actions.append(joint_angles_list[0].cpu().numpy().tolist())
    
    # Handle grasp execution
    if grasp and best_grasp_filtered_idx is not None:
        print('grasp!')
        grasp_id = best_grasp_filtered_idx  # Use filtered index for set_grasp_state
        grasp_joint_angles_list, grasp_action_object_transformations, grasp_robot_images, grasp_robot_depth_images = mesh_world.set_grasp_state(grasp_id)
        save_env_states(mesh_world, f"step_{step_idx}_set_grasp", state_out_path, context={"phase": "set_grasp_state", "step": step_idx})
        joint_angles_list, action_object_transformations, robot_images, robot_depth_images = grasp_joint_angles_list, grasp_action_object_transformations, grasp_robot_images, grasp_robot_depth_images,
        output_actions.append('grasp')
        output_actions.append(joint_angles_list[0].cpu().numpy().tolist())
    
    # Handle release execution
    if args.try_release and release:
        print('release!')
        release_joint_angles_list, release_action_object_transformations, release_robot_images, release_robot_depth_images = mesh_world.release(non_stop=True)
        save_env_states(mesh_world, f"step_{step_idx}_final_release_execute", state_out_path, context={"phase": "final_release_execute", "step": step_idx})
        joint_angles_list, action_object_transformations, robot_images, robot_depth_images = release_joint_angles_list, release_action_object_transformations, release_robot_images, release_robot_depth_images,
        output_actions.append('release')
        output_actions.append(joint_angles_list[0].cpu().numpy().tolist())
    
    if args.record_video:
        mesh_world.env.flush_video()
    
    object_transformations = action_object_transformations[0]
    
    # Update prev_state_context for next iteration
    prev_state_context = get_state_context(mesh_world, env_idx=0)
    
    # Render and save trajectory frame
    rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=object_transformations, rotate_num=4)
    depthmaps[np.where(depthmaps == 0)] = zfar
    robot_depth_images[np.where(robot_depth_images == 0)] = zfar
    
    robot_images = robot_images[view_id, 0, ...] if len(robot_images.shape) > 3 else robot_images[0, ...]
    robot_depth_images = robot_depth_images[view_id, 0, ..., 0] if len(robot_depth_images.shape) > 3 else robot_depth_images[0, ..., 0]
    
    robot_mask = np.where((np.any(robot_images != 0, axis=-1)) * (robot_depth_images < depthmaps[view_id]), 1, 0)
    images = np.where(robot_mask[:, :, None], robot_images, rgbmaps[view_id])
    
    images = images[:, :, :3] if len(images.shape) == 3 else images
    
    # Save trajectory image
    plt.imsave(f'{output_path}/{len(trajectory)}.png', images)
    excute_frames.append(images)
    trajectory.append(joint_angles_list[0])
    
    # Update subgoal based on new state
    current_state_context = get_state_context(mesh_world, env_idx=0)
    new_subgoal = determine_subgoal_stage(current_state_context)
    if new_subgoal != current_subgoal:
        print(f'Subgoal transition: {current_subgoal} -> {new_subgoal}')
        current_subgoal = new_subgoal



# concatenate excute_frames to one image and save it
blank_line = np.ones((image_size, image_size // 20, 3))
for i in range(len(excute_frames)):
    if i == 0:
        image = excute_frames[i]
    else:
        image = np.concatenate([image, blank_line, excute_frames[i]], axis=1)
plt.imsave(f'{output_path}/trajectory.png', image)

# save actions
with open(f'{output_path}/actions.txt', 'w') as f:
    for action in output_actions:
        f.write(f'{action}\n')

mesh_world.reset()
