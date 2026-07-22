"""Vector baseline — rigorous codebase edit ops.

Adapts VectorOptimizer's approach to work with position targets:
Computes direction vectors from wrist to each link, then uses
Huber loss on vector differences with analytical Jacobian gradient.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "dex-retargeting/src/dex_retargeting/custom_optimizer.py"

# ── 1. Replace get_objective_function method (lines 57-119) ──────────────────

_VECTOR_OBJECTIVE = """\
    # ── VectorOptimizer-style objective ──────────────────────────────────

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        # Compute target vectors: from first link (wrist proxy) to all others
        target_vectors = target_pos[1:] - target_pos[0:1]  # (N-1, 3)
        torch_target_vec = torch.as_tensor(target_vectors)
        torch_target_vec.requires_grad_(False)
        huber_loss = torch.nn.SmoothL1Loss(beta=self.huber_delta, reduction="mean")

        # Anchor: absolute position of the wrist (first target link)
        # Without this, the free joint (6-DOF global position/rotation) is
        # unconstrained by vector-only objectives and drifts arbitrarily.
        torch_wrist_target = torch.as_tensor(target_pos[0:1])
        torch_wrist_target.requires_grad_(False)
        wrist_huber = torch.nn.SmoothL1Loss(beta=self.huber_delta, reduction="mean")

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Compute robot vectors from first link to all others
            origin_pos = torch_body_pos[0:1, :]
            task_pos = torch_body_pos[1:, :]
            robot_vec = task_pos - origin_pos

            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            vec_loss = huber_loss(vec_dist, torch.zeros_like(vec_dist))

            # Wrist position anchor to prevent free joint drift
            wrist_loss = wrist_huber(origin_pos, torch_wrist_target)

            huber_distance = vec_loss + wrist_loss
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

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    # ── END EDITABLE ─────────────────────────────────────────────────────────
"""

# ── 2. Replace EDITABLE AREA (lines 20-23) ──────────────────────────────────

_VECTOR_HELPERS = """\
# ── EDITABLE AREA START ──────────────────────────────────────────────────────


# ── EDITABLE AREA END ────────────────────────────────────────────────────────
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 57,
        "end_line": 119,
        "content": _VECTOR_OBJECTIVE,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 23,
        "content": _VECTOR_HELPERS,
    },
]
