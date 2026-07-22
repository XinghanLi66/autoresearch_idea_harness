"""Evaluation harness for the dex-retarget task.

Evaluates CustomOptimizer on synthetic retargeting data using a configurable robot hand.
Generates random robot joint configurations, computes FK to get target link positions,
then runs retargeting from a perturbed starting point and measures recovery quality.

Metrics:
  - mpjpe_mm: Mean Per-Joint Position Error in millimeters (lower is better)
  - smoothness: Mean absolute jerk of joint trajectory (lower is better)
  - fps: Retargeting throughput in frames per second (higher is better)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from dex_retargeting.constants import RetargetingType, HandType, get_default_config_path, RobotName
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.robot_wrapper import RobotWrapper

ROBOT_MAP = {
    "allegro": RobotName.allegro,
    "svh": RobotName.svh,
    "leap": RobotName.leap,
}


def sample_qpos(optimizer, rng):
    """Sample a random valid joint configuration and a perturbed init."""
    robot = optimizer.robot
    adaptor = optimizer.adaptor
    joint_limit = robot.joint_limits
    eps = 1e-5

    random_qpos = rng.uniform(joint_limit[:, 0] + eps, joint_limit[:, 1] - eps)
    if adaptor is not None:
        random_qpos = adaptor.forward_qpos(random_qpos)

    # Perturbed starting point
    init_qpos = np.clip(
        random_qpos + rng.randn(robot.dof) * 0.3,
        joint_limit[:, 0] + eps,
        joint_limit[:, 1] - eps,
    )
    return random_qpos, init_qpos


def generate_smooth_trajectory(optimizer, num_frames, rng):
    """Generate a smooth trajectory of target positions by interpolating random configs."""
    robot = optimizer.robot
    num_keyframes = max(4, num_frames // 20)
    keyframe_qpos = []

    for _ in range(num_keyframes):
        qpos, _ = sample_qpos(optimizer, rng)
        keyframe_qpos.append(qpos)
    keyframe_qpos.append(keyframe_qpos[0])  # loop back

    # Interpolate between keyframes
    frames_per_segment = num_frames // (len(keyframe_qpos) - 1)
    all_target_pos = []
    all_gt_qpos = []

    for i in range(len(keyframe_qpos) - 1):
        for j in range(frames_per_segment):
            t = j / frames_per_segment
            # Smooth interpolation (cosine)
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            qpos = (1 - t_smooth) * keyframe_qpos[i] + t_smooth * keyframe_qpos[i + 1]

            if optimizer.adaptor is not None:
                qpos = optimizer.adaptor.forward_qpos(qpos)

            robot.compute_forward_kinematics(qpos)
            target_pos = np.array(
                [robot.get_link_pose(idx)[:3, 3] for idx in optimizer.target_link_indices]
            )
            all_target_pos.append(target_pos)
            all_gt_qpos.append(qpos.copy())

    return all_target_pos[:num_frames], all_gt_qpos[:num_frames]


def compute_smoothness(joint_trajectory):
    """Compute mean absolute jerk (3rd derivative) of joint trajectory."""
    if len(joint_trajectory) < 4:
        return 0.0
    traj = np.array(joint_trajectory)
    # 1st derivative (velocity)
    vel = np.diff(traj, axis=0)
    # 2nd derivative (acceleration)
    acc = np.diff(vel, axis=0)
    # 3rd derivative (jerk)
    jerk = np.diff(acc, axis=0)
    return float(np.mean(np.abs(jerk)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate custom retargeting optimizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--robot", type=str, default="allegro",
                        choices=list(ROBOT_MAP.keys()),
                        help="Robot hand to evaluate on")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Set up paths
    robot_dir = Path(__file__).parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir.absolute()))

    robot_name = ROBOT_MAP[args.robot]
    print(f"Robot: {args.robot}")

    # Load robot hand with custom optimizer type
    config_path = get_default_config_path(
        robot_name, RetargetingType.position, HandType.right
    )

    # Override to use custom optimizer type, disable smoothing penalty for fair eval
    override = {
        "type": "custom",
        "normal_delta": 4e-3,
        "low_pass_alpha": 1.0,
    }
    config = RetargetingConfig.load_from_file(config_path, override)
    retargeting = config.build()
    optimizer = retargeting.optimizer

    print(f"Robot DOF: {optimizer.robot.dof}")
    print(f"Optimized joints: {optimizer.opt_dof}")
    print(f"Target links: {optimizer.body_names}")
    print(f"Num frames: {args.num_frames}")
    print(f"Seed: {args.seed}")
    print()

    # Generate smooth synthetic trajectory
    target_positions, gt_qpos_list = generate_smooth_trajectory(
        optimizer, args.num_frames, rng
    )

    # Run retargeting
    errors = []
    joint_trajectory = []
    retargeting.reset()

    tic = time.perf_counter()
    for frame_idx, target_pos in enumerate(target_positions):
        fixed_qpos = gt_qpos_list[frame_idx][optimizer.idx_pin2fixed]
        computed_qpos = retargeting.retarget(target_pos, fixed_qpos=fixed_qpos)

        # Evaluate: compute FK with the result and measure position error
        target_joint_qpos = computed_qpos[optimizer.idx_pin2target]
        joint_trajectory.append(target_joint_qpos.copy())

        full_qpos = np.zeros(optimizer.robot.dof)
        full_qpos[optimizer.idx_pin2target] = target_joint_qpos
        full_qpos[optimizer.idx_pin2fixed] = fixed_qpos
        if optimizer.adaptor is not None:
            full_qpos = optimizer.adaptor.forward_qpos(full_qpos)

        optimizer.robot.compute_forward_kinematics(full_qpos)
        computed_pos = np.array(
            [optimizer.robot.get_link_pose(idx)[:3, 3] for idx in optimizer.target_link_indices]
        )

        # Per-joint position error
        per_joint_error = np.linalg.norm(computed_pos - target_pos, axis=-1)
        errors.append(per_joint_error)

    elapsed = time.perf_counter() - tic

    # Compute metrics
    errors = np.array(errors)  # (num_frames, num_links)
    mpjpe_m = float(np.mean(errors))
    mpjpe_mm = mpjpe_m * 1000.0  # convert to mm
    smoothness = compute_smoothness(joint_trajectory)
    fps = len(target_positions) / elapsed

    # Print per-frame stats
    print("Per-link mean errors (mm):")
    for i, name in enumerate(optimizer.body_names):
        link_err = float(np.mean(errors[:, i])) * 1000.0
        print(f"  {name}: {link_err:.2f}")
    print()

    print(f"Overall MPJPE: {mpjpe_mm:.4f} mm")
    print(f"Smoothness (mean |jerk|): {smoothness:.6f}")
    print(f"FPS: {fps:.1f}")
    print()

    # Output for parser
    print(f"RETARGET_METRICS mpjpe_mm={mpjpe_mm:.4f} smoothness={smoothness:.6f} fps={fps:.1f}")


if __name__ == "__main__":
    main()
