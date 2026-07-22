# Hand Retargeting Optimizer Design

> Deprecated 2026-04-29: this task evaluates on a task-local synthetic smooth
> hand trajectory rather than a captured or community-standard retargeting
> benchmark. Keep it only for historical results until a replacement task uses
> real motion data.

## Objective
Design a better retargeting optimizer that maps human hand joint positions to robot hand joint angles. You should implement `get_objective_function()` in `custom_optimizer.py`, which returns a closure used by nlopt (LD_SLSQP) to solve the inverse kinematics optimization.

## Background
Hand retargeting maps human hand poses to dexterous robot hand configurations. Given target 3D positions for key robot links (fingertips, finger mids), the optimizer finds joint angles that minimize position error while maintaining smoothness across frames.

The evaluation tests your optimizer on **three different robot hands** with varying kinematics and DOF counts:
- **Allegro Hand Right** (16 DOF, 4 fingers, 8 target links)
- **Schunk SVH Hand Right** (9 DOF, 5 fingers, 10 target links)
- **LEAP Hand Right** (16 DOF, 4 fingers, 8 target links)

Your implementation must generalize across all three hands without hand-specific logic.

## What to Implement
Edit `custom_optimizer.py` to implement a high-quality objective function. You can modify:
- The `get_objective_function()` method
- The EDITABLE AREA above the class (for helper functions, custom losses, etc.)

The objective function closure `objective(x, grad) -> float` must:
1. Construct the full robot joint configuration from `x` (optimized joints) and `fixed_qpos`
2. Compute forward kinematics to get current link positions
3. Compute a scalar loss measuring retargeting quality
4. Fill `grad[:]` with the analytical gradient if `grad.size > 0` (required for LD_SLSQP)
5. Return the scalar loss value

## Available Utilities
- `self.robot.compute_forward_kinematics(qpos)` — compute FK for full joint vector
- `self.robot.get_link_pose(link_index)` — returns 4x4 homogeneous transform
- `self.robot.compute_single_link_local_jacobian(qpos, link_index)` — returns 6xN Jacobian (body frame)
- `self.target_link_indices` — pinocchio frame IDs for the target links
- `self.idx_pin2target` / `self.idx_pin2fixed` — index mappings for optimized vs fixed joints
- `self.adaptor` — kinematic adaptor for mimic joints (handle if not None)
- `torch.nn.SmoothL1Loss` (Huber loss) — commonly used for robust position matching

## Evaluation Metrics
- **MPJPE (mm)**: Mean per-joint position error — primary metric, lower is better
- **Smoothness**: Mean absolute jerk of joint trajectory — lower is better
- **FPS**: Optimization throughput — higher is better

The evaluation runs retargeting over a 200-frame synthetic trajectory of smooth hand motions on each of the three robot hands.
