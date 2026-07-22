"""Custom retargeting optimizer for dex-retargeting.

This file defines CustomOptimizer — a retargeting optimizer that maps human hand
joint positions to robot dexterous hand joint angles. The key method to implement
is get_objective_function(), which returns an objective closure for nlopt.

The optimizer inherits from the Optimizer base class and follows the same pattern
as PositionOptimizer / VectorOptimizer / DexPilotOptimizer.
"""

from typing import List

import nlopt
import numpy as np
import torch

from dex_retargeting.optimizer import Optimizer
from dex_retargeting.robot_wrapper import RobotWrapper

# ── EDITABLE AREA START ──────────────────────────────────────────────────────


# ── EDITABLE AREA END ────────────────────────────────────────────────────────


class CustomOptimizer(Optimizer):
    """Custom optimizer for hand retargeting.

    This optimizer receives target 3D link positions (same as PositionOptimizer)
    and must minimize the error between robot FK link positions and the targets.

    You may add helper functions, custom loss functions, or additional state
    in the EDITABLE AREA above and in get_objective_function() below.
    """

    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta: float = 0.02,
        norm_delta: float = 4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_delta = huber_delta
        self.norm_delta = norm_delta

        # Cache link indices for FK queries
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    # ── EDITABLE: implement your objective function ──────────────────────────

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """Return an objective function closure for nlopt.

        Args:
            target_pos: (N, 3) target 3D positions for each link in target_link_names
            fixed_qpos: joint values for non-optimized joints
            last_qpos: previous optimization result (for regularization)

        Returns:
            objective: callable(x, grad) -> float
                x: current joint angles being optimized (shape: opt_dof)
                grad: gradient array to fill in-place (shape: opt_dof), empty if not needed
                returns: scalar loss value

        Available robot utilities (via self.robot):
            - self.robot.compute_forward_kinematics(qpos)  # full joint vector
            - self.robot.get_link_pose(link_index) -> 4x4 homogeneous matrix
            - self.robot.compute_single_link_local_jacobian(qpos, link_index) -> 6xN Jacobian

        Available index mappings:
            - self.idx_pin2target: indices of optimized joints in full qpos
            - self.idx_pin2fixed: indices of fixed joints in full qpos
            - self.target_link_indices: pinocchio frame IDs for target links
            - self.num_joints: total DOF of the robot

        The objective function should:
            1. Construct full qpos from x (optimized) and fixed_qpos
            2. Compute FK to get current link positions
            3. Compute loss between current and target positions
            4. Compute analytical gradient if grad.size > 0
            5. Return scalar loss
        """
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            body_pos = np.stack(
                [self.robot.get_link_pose(idx)[:3, 3] for idx in self.target_link_indices],
                axis=0,
            )

            # Naive L2 loss without gradient — replace with a better objective!
            diff = body_pos - target_pos
            loss = float(np.mean(diff ** 2))

            if grad.size > 0:
                grad[:] = 0.0

            return loss

        return objective

    # ── END EDITABLE ─────────────────────────────────────────────────────────
