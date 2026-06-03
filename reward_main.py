"""CEM planner with 7D actions (6 pose + 1 gripper) and code reward scoring.

Usage:
    python reward_main.py --scene_name basket_separate \
        --instruction "put the green cucumber into the basket" \
        --reward_name cucumber_human_reward_main --num_samples 100 --delta_t 1.5
"""

import os
import time
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from meshes.mesh_world import MeshWorld
from gaussians.gaussian_world import GaussianWorld
from pytorch3d.renderer import look_at_view_transform
from utils.camera import get_up_direction
from reward.reward_manager import RewardManager
from reward.reward_helpers import save_context


def build_parser():
    parser = argparse.ArgumentParser(description="CEM Planner (7D + Code Reward)")
    parser.add_argument("--scene_name", type=str, default="basket_separate")
    parser.add_argument("--instruction", type=str, default="put the green cucumber into the basket")
    parser.add_argument("--name", type=str, default="demo")
    parser.add_argument("--image_size", type=int, default=500)
    parser.add_argument("--total_steps", type=int, default=10)
    parser.add_argument("--camera_view_id", type=int, default=1)

    parser.add_argument("--cem_iteration", type=int, default=3)
    # Reward-only sampler controls:
    # - num_samples: number of candidate 7D actions per CEM iteration (parallel envs)
    # - delta_t: action duration per candidate in seconds (smaller = replan more frequently,
    #   finer-grained control). Converted to physics ticks inside cem_step().
    # - elite_fraction: top-K fraction of candidates kept as elites for Gaussian refit
    parser.add_argument("--num_samples", type=int, default=36)
    parser.add_argument("--delta_t", type=float, default=1.5)
    parser.add_argument("--elite_fraction", type=float, default=0.1)

    parser.add_argument("--reward_name", type=str, default="cucumber_human_reward_main")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--replan", action="store_true")

    # Legacy flags from main.py (intentionally disabled here):
    # This script is reward-only + 7D (pose + binary gripper). These flags exist in main.py
    # to support the VLM pipeline and/or special-case grasp/release logic.
    #
    # parser.add_argument("--use_reward", action="store_true")
    #   Disabled: reward_main.py always uses code reward (no VLM branch).
    #
    # parser.add_argument("--num_sample_actions", type=int, default=81)
    #   Disabled: renamed to --num_samples for clarity (same concept: candidates per CEM iteration).
    #
    # parser.add_argument("--substeps", type=int, default=10)
    #   Disabled: replaced by --delta_t (seconds). Internally, substeps = round(delta_t / 0.05)
    #   where 0.05s = 1 / ManiSkill control_freq (default 20 Hz). delta_t is friendlier
    #   (real physical units, matches how we talk about action duration in papers / meetings).
    #
    # parser.add_argument("--plane_action", action="store_true")
    #   Disabled: 6D view-dependent variance heuristic from the VLM pipeline; not needed for reward-only.
    #
    # parser.add_argument("--release", action="store_true")
    # parser.add_argument("--try_release", action="store_true")
    #   Disabled: old explicit release branches; reward_main uses the 7th gripper dimension instead.
    return parser


def parse_args():
    return build_parser().parse_args()


def save_candidate_images(
    output_path,
    step_id,
    iteration,
    gaussian_world,
    action_object_transformations,
    robot_images,
    robot_depth_images,
    R_gaussian,
    T_gaussian,
    image_size,
    FoV,
    device,
    zfar,
    view_id,
):
    """Save candidate rollout images from sampler-returned robot renders."""
    num_candidates = len(action_object_transformations)
    if num_candidates <= 0:
        return

    robot_images = robot_images[view_id, :num_candidates]
    robot_depth_images = robot_depth_images[view_id, :num_candidates]

    rgbmaps, depthmaps = [], []
    for act_id in range(num_candidates):
        rgbmap, depthmap, _ = gaussian_world.render(
            R_gaussian,
            T_gaussian,
            image_size,
            -FoV / 180.0 * np.pi,
            device,
            object_states=action_object_transformations[act_id],
            rotate_num=1,
        )
        depthmap[np.where(depthmap == 0)] = zfar
        rgbmaps.append(rgbmap[0])
        depthmaps.append(depthmap[0])

    rgbmaps = np.stack(rgbmaps)
    depthmaps = np.stack(depthmaps)
    robot_mask = np.where(
        (np.any(robot_images != 0, axis=-1)) * (robot_depth_images[..., 0] < depthmaps),
        1,
        0,
    )
    images = np.where(robot_mask[:, :, :, None], robot_images, rgbmaps)

    vis_images = []
    for act_id, img in enumerate(images):
        img = img[:, :, :3]
        vis_images.append(img.copy())
        plt.imsave(f"{output_path}/{step_id}_{act_id + 1}.png", img)

    grid_cols = min(10, len(vis_images))
    vis_all_images = []
    for start in range(0, len(vis_images), grid_cols):
        end = min(start + grid_cols, len(vis_images))
        vis_group_images = np.concatenate(vis_images[start:end], axis=1)
        if end - start < grid_cols:
            pad = np.ones(
                (
                    vis_group_images.shape[0],
                    vis_images[0].shape[1] * (grid_cols - (end - start)),
                    vis_group_images.shape[2],
                )
            )
            vis_group_images = np.concatenate([vis_group_images, pad], axis=1)
        vis_all_images.append(vis_group_images)
    vis_all_images = np.concatenate(vis_all_images, axis=0)
    plt.imsave(f"{output_path}/{step_id}_{iteration}_all.png", vis_all_images)


def build_worlds(args, parser):
    """Initialize GaussianWorld, MeshWorld, cameras, and reward manager."""
    image_size = args.image_size
    scene_name = args.scene_name
    robot_translation = [-0.45, 0.0, 0.0]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    reward_manager = RewardManager(args.reward_name)
    reward_manager.load_context()
    if reward_manager.context is None:
        raise RuntimeError("context.py not available yet.")

    close_gripper = reward_manager.context.KEEP_GRIPPER_CLOSED
    if close_gripper:
        print("!!! Keep Gripper Closed !!!")

    gaussian_world = GaussianWorld(scene_name, parser, post_process=False)
    distance = 1.5
    radius = gaussian_world.radius * distance
    center = np.array([0, 0, 0])

    elev = torch.tensor([-70, 0, 70, 0], device=device)
    azim = torch.tensor([0, 70, 0, 0], device=device)
    up = get_up_direction(elev, azim)
    at = torch.tensor(center[None], device=device).float()

    cameras_config = [
        {"elev": elev[i].item(), "azim": azim[i].item()} for i in range(4)
    ]

    R_fixed, T_fixed = look_at_view_transform(
        dist=radius, elev=elev, azim=azim, up=up, at=at, device=device,
    )
    up[:, 0] = -up[:, 0]
    at[:, 0] = -at[:, 0]
    elev_g = 180 - elev
    R_gaussian_fixed, T_gaussian_fixed = look_at_view_transform(
        dist=radius, elev=elev_g, azim=azim, up=up, at=at, device="cpu",
    )
    R_gaussian_fixed = R_gaussian_fixed.numpy()
    T_gaussian_fixed = T_gaussian_fixed.numpy()

    robot_uids = "PandaRobotiqHand"
    mesh_world = MeshWorld(
        scene_name,
        num_envs=args.num_samples,
        scene_traslation=-np.array(robot_translation),
        radius=radius,
        image_size=image_size,
        record_video=args.record_video,
        robot_uids=robot_uids,
        need_render=True,
        dir=None,
        close_gripper=close_gripper,
        cameras_config=cameras_config,
    )

    return (
        mesh_world,
        gaussian_world,
        reward_manager,
        device,
        R_gaussian_fixed,
        T_gaussian_fixed,
        radius,
        close_gripper,
    )


def cem_step(
    mesh_world,
    gaussian_world,
    reward_manager,
    context,
    prev_context,
    stage,
    args,
    output_path,
    step_id,
    R_gaussian,
    T_gaussian,
    image_size,
    FoV,
    device,
    zfar,
    view_id,
):
    """Run one CEM planning step: sample, simulate, score, refit, execute.

    Returns:
        context: Updated state context after executing the best action.
        prev_context: Context before execution (for next step's prev_context).
        joint_angles: Joint angles after execution.
        action_object_transformations: Object transforms after execution.
    """
    action_dims = 7
    means = np.zeros(action_dims)
    variances = np.array([1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 0.5])
    covariance = np.diag(variances)
    best_sample = None
    best_reward = -np.inf

    # Convert action duration from seconds (user-facing) to physics ticks (engine-facing).
    #
    # ManiSkill's env.step() advances physics by exactly one control tick. At its default
    # control frequency of 20 Hz, one tick = 1/20 = 0.05 seconds. We can only run physics
    # in integer numbers of ticks, so we round delta_t to the nearest whole tick count.
    #
    # Examples (at 20 Hz):
    #   delta_t = 1.50s -> round(1.50 / 0.05) = 30 ticks, matching main.py
    #   delta_t = 0.15s -> round(0.15 / 0.05) = 3  ticks
    #   delta_t = 0.10s -> round(0.10 / 0.05) = 2  ticks
    #   delta_t = 0.05s -> round(0.05 / 0.05) = 1  tick
    #
    # If ManiSkill's control_freq is changed later, update the 0.05 below to 1 / new_freq.
    if args.delta_t <= 0:
        raise ValueError("--delta_t must be positive (seconds)")
    substeps = max(1, round(args.delta_t / 0.05))
    print(f"  action duration: delta_t={args.delta_t:.2f}s -> substeps={substeps}")

    for iteration in range(args.cem_iteration):
        t0 = time.time()

        samples = np.random.multivariate_normal(means, covariance, size=args.num_samples)

        (
            joint_angles_list,
            action_obj_transforms,
            post_samples,
            robot_images,
            robot_depth_images,
            cur_context,
        ) = (
            mesh_world.sample_action_batch(
                samples,
                substeps=substeps,
                need_context=True,
                return_images=True,
            )
        )
        save_context(cur_context, f"step_{step_id}_iteration_{iteration}_sample", os.path.join(output_path, "states"))
        save_candidate_images(
            output_path=output_path,
            step_id=step_id,
            iteration=iteration,
            gaussian_world=gaussian_world,
            action_object_transformations=action_obj_transforms,
            robot_images=robot_images,
            robot_depth_images=robot_depth_images,
            R_gaussian=R_gaussian,
            T_gaussian=T_gaussian,
            image_size=image_size,
            FoV=FoV,
            device=device,
            zfar=zfar,
            view_id=view_id,
        )
        reward_manager.attach_initial_state(cur_context)
        rewards = np.array([
            reward_manager.rw.compute_reward(cur_context, prev_context, stage, i)
            for i in range(len(post_samples))
        ])

        # Elite selection: standard CEM top-K fraction by code reward
        # Keep the globally highest-scoring candidates (no grouping) to refit the Gaussian.
        if not 0.0 < args.elite_fraction <= 1.0:
            raise ValueError("--elite_fraction must be in (0, 1]")
        num_elites = max(1, int(len(post_samples) * args.elite_fraction))
        elite_ids = np.argsort(rewards)[-num_elites:]
        elite_samples = post_samples[elite_ids]
        best_id = elite_ids[-1]
        if rewards[best_id] > best_reward:
            best_reward = rewards[best_id]
            best_sample = post_samples[best_id].copy()

        means = np.mean(elite_samples, axis=0)
        covariance = np.cov(elite_samples, rowvar=False) + 1e-6 * np.eye(action_dims)

        print(
            f"  iter {iteration}: simulate+score {time.time() - t0:.2f}s | "
            f"best={rewards[elite_ids[-1]]:.4f} median={np.median(rewards):.4f}"
        )
        print("  best action sample:", post_samples[elite_ids[-1]])

    if best_sample is None:
        best_sample = means
    prev_context = context
    joint_angles, action_obj_transforms, robot_images, robot_depth_images, new_context = mesh_world.sample_action_batch(
        best_sample[None],
        substeps=substeps,
        non_stop=True,
        need_context=True,
        return_images=True,
    )
    reward_manager.attach_initial_state(new_context)
    save_context(new_context, f"step_{step_id}_final_sample", os.path.join(output_path, "states"))
    return new_context, prev_context, joint_angles, action_obj_transforms, robot_images, robot_depth_images


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_path = os.path.join(
        "results", f"{args.instruction}/{args.scene_name}/reward_{args.name}",
    )
    os.makedirs(output_path, exist_ok=True)
    state_output_path = os.path.join(output_path, "states")
    os.makedirs(state_output_path, exist_ok=True)

    (
        mesh_world,
        gaussian_world,
        reward_manager,
        device,
        R_gaussian_fixed,
        T_gaussian_fixed,
        radius,
        close_gripper,
    ) = build_worlds(args, parser)

    image_size = args.image_size
    znear, zfar = 0.01, 100
    FoV = 60
    view_id = args.camera_view_id

    R_gaussian = R_gaussian_fixed[view_id : view_id + 1]
    T_gaussian = T_gaussian_fixed[view_id : view_id + 1]

    trajectory = []
    output_actions = []
    excute_frames = []
    object_transformations = []
    max_replan = 5
    replan_time = 0

    if close_gripper:
        output_actions.append("grasp")

    initial_qpos = mesh_world.agent.robot.get_qpos()[0].cpu().numpy().tolist()
    print("initial_joint_angles:", initial_qpos)
    trajectory.append(torch.tensor(initial_qpos))

    context = mesh_world.get_context()
    reward_manager.set_initial_context(context)
    reward_manager.attach_initial_state(context)
    prev_context = context
    save_context(context, "initial_context", state_output_path)

    # save subgoal views and subgoal text like main.py
    robot_images, robot_depth_images = mesh_world.get_image_depth()
    current_robot_images, current_robot_depths = robot_images, robot_depth_images
    rgbmaps, depthmaps, alphamaps = gaussian_world.render(
        R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi,
        device, object_states=object_transformations, rotate_num=4,
    )
    depthmaps[np.where(depthmaps == 0)] = zfar
    current_robot_depths[np.where(current_robot_depths == 0)] = zfar

    current_robot_images = current_robot_images[:, 0, ...]
    current_robot_depths = current_robot_depths[:, 0, ..., 0]

    robot_mask = np.where(
        (np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0,
    )
    current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

    for i in range(len(current_images)):
        plt.imsave(f'{output_path}/subgoal_view_{i + 1}.png', current_images[i])

    excute_frames.append(current_images[view_id])

    subgoals = reward_manager.context.subgoals
    print("subgoals:", subgoals)

    while len(trajectory) <= args.total_steps:
        prev_context = context
        context = mesh_world.get_context()
        reward_manager.attach_initial_state(context)
        save_context(context, f"step_{len(trajectory)}_start", state_output_path)

        if reward_manager.context.determine_success(context):
            print("!!! Reward Function says Success !!!")
            break

        stage = reward_manager.context.determine_stage(context)
        reward_manager.update_stage(stage)
        if reward_manager.rw is None:
            raise RuntimeError(f"Reward file reward_{stage}.py not available.")

        print(f"step {len(trajectory)} | stage {stage}: {subgoals[stage - 1]}")

        with open(f'{output_path}/{len(trajectory)}_view_content.txt', 'w') as f:
            f.write(f'selected_view_id: {view_id}\n')

        # Save view images like main.py
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

        (
            context,
            prev_context,
            joint_angles,
            action_obj_transforms,
            robot_images,
            robot_depth_images,
        ) = cem_step(
            mesh_world,
            gaussian_world,
            reward_manager,
            context,
            prev_context,
            stage,
            args,
            output_path,
            len(trajectory),
            R_gaussian,
            T_gaussian,
            image_size,
            FoV,
            device,
            zfar,
            view_id,
        )

        output_actions.append(joint_angles[0].cpu().numpy().tolist())

        if args.record_video:
            mesh_world.env.flush_video()

        object_transformations = action_obj_transforms[0]

        rgbmaps, depthmaps, alphamaps = gaussian_world.render(
            R_gaussian, T_gaussian, image_size, -FoV / 180.0 * np.pi,
            device, object_states=object_transformations, rotate_num=1,
        )
        depthmaps[np.where(depthmaps == 0)] = zfar

        robot_img = robot_images[view_id, 0, ...]
        robot_depth = robot_depth_images[view_id, 0, ..., 0]
        robot_depth[np.where(robot_depth == 0)] = zfar

        robot_mask = np.where(
            (np.any(robot_img != 0, axis=-1)) * (robot_depth < depthmaps), 1, 0,
        )
        composited = np.where(robot_mask[:, :, :, None], robot_img, rgbmaps)
        composited = composited[:, :, :, :3]

        plt.imsave(f"{output_path}/{len(trajectory)}.png", composited[0])
        excute_frames.append(composited[0])
        trajectory.append(joint_angles[0])

        if args.replan and replan_time < max_replan and mesh_world.object_drop:
            print("object drop replan!")
            context = prev_context
            trajectory.pop()
            excute_frames.pop()
            output_actions.pop()
            mesh_world.history_states = mesh_world.history_states[:-1]
            mesh_world.object_drop = False
            mesh_world.grasping_now = True
            mesh_world.grasping_pos = mesh_world.prev_grasping_pos
            replan_time += 1

    blank = np.ones((image_size, image_size // 20, 3))
    strip = excute_frames[0]
    for frame in excute_frames[1:]:
        strip = np.concatenate([strip, blank, frame], axis=1)
    plt.imsave(f"{output_path}/trajectory.png", strip)

    with open(f"{output_path}/actions.txt", "w") as f:
        for action in output_actions:
            f.write(f"{action}\n")

    mesh_world.reset()
    print("Done.")


if __name__ == "__main__":
    main()
