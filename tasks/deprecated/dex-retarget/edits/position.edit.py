"""Position baseline — rigorous codebase edit ops.

Replaces the stub objective with PositionOptimizer's approach:
Huber loss on 3D link positions with analytical Jacobian gradient.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "dex-retargeting/src/dex_retargeting/custom_optimizer.py"

# ── 1. Replace get_objective_function method (lines 57-119) ──────────────────

_POSITION_OBJECTIVE = """\
    # ── PositionOptimizer objective ──────────────────────────────────────

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)
        huber_loss = torch.nn.SmoothL1Loss(beta=self.huber_delta)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )

            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            huber_distance = huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    # ── END EDITABLE ─────────────────────────────────────────────────────────
"""

# ── 2. Replace EDITABLE AREA (lines 20-23) — no extra helpers needed ─────────

_POSITION_HELPERS = """\
# ── EDITABLE AREA START ──────────────────────────────────────────────────────


# ── EDITABLE AREA END ────────────────────────────────────────────────────────
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 57,
        "end_line": 119,
        "content": _POSITION_OBJECTIVE,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 23,
        "content": _POSITION_HELPERS,
    },
]
