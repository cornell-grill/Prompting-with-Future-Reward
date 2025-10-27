""" This file currently moves the gripper to grasp a cumber using state-based reward """
import os
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
    parser.add_argument("--debug", action="store_false")
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
if(args.debug):
    mesh_info = mesh_world.get_info()
    print('--- scene loaded ---')
    print(mesh_info)

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

while len(trajectory) <= args.total_steps:
    grasp = release = False

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

    # TODO: get success criteria (distance and grasp?)
    if mesh_world.grasping_now:
        print('!!! Success !!!')
        break

    # TODO: Have stages stored as a variable
    stage = 1
    subgoal_id = stage - 1

    # sample release action
    if mesh_world.grasping_now:
        joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.release()

        current_robot_images, current_robot_depths = robot_images, robot_depth_images

        rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=action_object_transformations[0], rotate_num=4)
        depthmaps[np.where(depthmaps == 0)] = zfar
        current_robot_depths[np.where(current_robot_depths == 0)] = zfar

        current_robot_images = current_robot_images[:, 0, ...]
        current_robot_depths = current_robot_depths[:, 0, ..., 0]

        robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
        current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

        encoded_images = []
        for i in range(len(current_images)):
            plt.imsave(f'{output_path}/{len(trajectory)}_release_{i + 1}.png', current_images[i])

        # TODO: implement release detections
        release = False

        if release:
            print('release!')
            output_actions.append('release')
            joint_angles_list, action_object_transformations, root_images, robot_depth_images = mesh_world.release(non_stop=True)
    
    if release:
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

    for iteration in range(args.cem_iteration):
        prev_time = time.time()
        prompt = []

        samples = np.random.multivariate_normal(means, covariance, size=args.num_sample_actions)
        grasp = False
        release = False

        if not mesh_world.grasping_now:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, grasp_object_transformations, grasp_robot_images, grasp_robot_depth_images, is_grasping = mesh_world.sample_action_distribution_batch(samples, try_grasp=True)
            if is_grasping.sum():
                print(f'{is_grasping.sum()} could grasp!')
                grasp_object_transformations = grasp_object_transformations[is_grasping]
                grasp_robot_images = grasp_robot_images[:, is_grasping]
                grasp_robot_depth_images = grasp_robot_depth_images[:, is_grasping]
                rgbmaps, depthmaps, alphamaps = [], [], []
                for grasp_id in range(len(grasp_object_transformations)):
                    rgbmap, depthmap, alphamap = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=grasp_object_transformations[grasp_id], rotate_num=4)
                    depthmap[np.where(depthmap == 0)] = zfar

                    rgbmaps.append(rgbmap[:])
                    depthmaps.append(depthmap[:])
                    alphamaps.append(alphamap[:])

                rgbmaps = np.stack(rgbmaps)
                depthmaps = np.stack(depthmaps)
                alphamaps = np.stack(alphamaps)

                rgbmaps = rgbmaps.transpose(1, 0, 2, 3, 4)
                depthmaps = depthmaps.transpose(1, 0, 2, 3)
                alphamaps = alphamaps.transpose(1, 0, 2, 3)

                robot_mask = np.where((np.any(grasp_robot_images != 0, axis=-1)) * (grasp_robot_depth_images[..., 0] < depthmaps), 1, 0)
                images = np.where(robot_mask[:, :, :, :, None], grasp_robot_images, rgbmaps)

                for grasp_id in range(len(grasp_object_transformations)):
                    img_views = images[:, grasp_id]
                    # print('img_views: ', img_views.shape)
                    encoded_images = []
                    for idx, img in enumerate(img_views):
                        plt.imsave(f'{output_path}/{len(trajectory)}_{iteration}_grasp_{grasp_id + 1}_view_{idx + 1}.png', img)

                    # change = None
                    # try_time = 0
                    # while change is None and try_time < 5:
                    #     try_time += 1
                    #     try:
                    #         content = generate_grasp(encoded_images, grasp_prompt)
                    #         grasp = get_grasp(content)
                    #         change = True
                    #     except Exception as e:
                    #         print('catched', e)
                    #         pass
                    # TODO: Implement logic to decide if we should grasp

                    if grasp:
                        means = post_samples[is_grasping][grasp_id]
                        break
                if grasp:
                    break

        elif args.try_release and mesh_world.grasping_now:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, release_object_transformations, release_robot_images, release_robot_depth_images = mesh_world.sample_action_distribution_batch(samples, try_release=True)

            rgbmaps, depthmaps, alphamaps = [], [], []
            for release_id in range(len(release_object_transformations)):
                rgbmap, depthmap, alphamap = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=release_object_transformations[release_id], rotate_num=4)
                depthmap[np.where(depthmap == 0)] = zfar

                rgbmaps.append(rgbmap[:])
                depthmaps.append(depthmap[:])
                alphamaps.append(alphamap[:])

            rgbmaps = np.stack(rgbmaps)
            depthmaps = np.stack(depthmaps)
            alphamaps = np.stack(alphamaps)

            rgbmaps = rgbmaps.transpose(1, 0, 2, 3, 4)
            depthmaps = depthmaps.transpose(1, 0, 2, 3)
            alphamaps = alphamaps.transpose(1, 0, 2, 3)

            robot_mask = np.where((np.any(release_robot_images != 0, axis=-1)) * (release_robot_depth_images[..., 0] < depthmaps), 1, 0)
            images = np.where(robot_mask[:, :, :, :, None], release_robot_images, rgbmaps)

            processes = []
            queue = multiprocessing.Queue()
            best_of_each_group = []
            for release_id in range(len(release_object_transformations)):
                img_views = images[:, release_id]
                encoded_images = []
                for idx, img in enumerate(img_views):
                    plt.imsave(f'{output_path}/{len(trajectory)}_{iteration}_release_{release_id + 1}_view_{idx + 1}.png', img)

            # TODO: release logic
            #     p = multiprocessing.Process(target=prompt_release_helper, args=(release_id, queue, encoded_images, release_prompt, args))
            #     processes.append(p)
            #     p.start()

            # for p in processes:
            #     p.join()

            # for p in processes:
            #     release_id, release, content = queue.get()

            #     with open(f'{output_path}/{len(trajectory)}_{iteration}_{release_id + 1}_release.txt', 'w') as f:
            #         f.write(content)
            #     if release:
            #         means = post_samples[release_id]
            #         print('release!')
            #         break
            if release:
                break

        else:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images = mesh_world.sample_action_distribution_batch(samples)







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
