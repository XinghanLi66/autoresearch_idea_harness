"""DexPilot baseline — rigorous codebase edit ops.

Adapts DexPilotOptimizer's approach to work with position targets:
Uses finger-to-finger distance projection for stable grasping, with
weighted Huber loss and analytical Jacobian gradient.

The key DexPilot innovation is projecting close finger pairs onto fixed
distance targets to encourage stable grasps.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "dex-retargeting/src/dex_retargeting/custom_optimizer.py"

# ── 1. Replace get_objective_function method (lines 57-119) ──────────────────

_DEXPILOT_OBJECTIVE = """\
    # ── DexPilot-style objective ─────────────────────────────────────────

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        # Build pairwise vectors between all links (DexPilot-style)
        n_links = len(target_pos)
        origin_indices = []
        task_indices = []
        # Finger-to-finger pairs (exclude first link as "base")
        for i in range(1, n_links):
            for j in range(i + 1, n_links):
                origin_indices.append(j)
                task_indices.append(i)
        # Base-to-finger pairs
        for i in range(1, n_links):
            origin_indices.append(0)
            task_indices.append(i)

        n_finger_pairs = (n_links - 1) * (n_links - 2) // 2
        n_base_pairs = n_links - 1
        n_pairs = len(origin_indices)

        # Compute target vectors
        target_origin_pos = target_pos[origin_indices]
        target_task_pos = target_pos[task_indices]
        target_vectors = target_task_pos - target_origin_pos

        # DexPilot projection: if finger pair distance < threshold, project to fixed dist
        project_dist = 0.03
        escape_dist = 0.05
        eta1 = 1e-4
        eta2 = 3e-2

        target_dists = np.linalg.norm(target_vectors[:n_finger_pairs], axis=1)
        projected = np.zeros(n_finger_pairs, dtype=bool)
        projected[target_dists < project_dist] = True
        projected[target_dists > escape_dist] = False

        # Weight: high weight for projected (close) pairs
        weight = np.ones(n_pairs, dtype=np.float32)
        weight[:n_finger_pairs] = np.where(projected, 200.0, 1.0)
        weight[n_finger_pairs:] = float(n_pairs)
        torch_weight = torch.from_numpy(weight)

        # Apply projection
        final_vectors = target_vectors.copy()
        for i in range(n_finger_pairs):
            if projected[i]:
                direction = target_vectors[i] / (target_dists[i] + 1e-6)
                final_vectors[i] = direction * eta1

        torch_target_vec = torch.as_tensor(final_vectors)
        torch_target_vec.requires_grad_(False)
        huber_loss = torch.nn.SmoothL1Loss(beta=self.huber_delta, reduction="none")

        torch_origin_indices = torch.tensor(origin_indices)
        torch_task_indices = torch.tensor(task_indices)

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

            origin_link_pos = torch_body_pos[torch_origin_indices, :]
            task_link_pos = torch_body_pos[torch_task_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            vec_loss = (
                huber_loss(vec_dist, torch.zeros_like(vec_dist)) * torch_weight / n_pairs
            ).sum()

            # Wrist position anchor to prevent free joint drift
            wrist_pos = torch_body_pos[0:1, :]
            wrist_loss = wrist_huber(wrist_pos, torch_wrist_target)

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

_DEXPILOT_HELPERS = """\
# ── EDITABLE AREA START ──────────────────────────────────────────────────────


# ── EDITABLE AREA END ────────────────────────────────────────────────────────
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 57,
        "end_line": 119,
        "content": _DEXPILOT_OBJECTIVE,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 23,
        "content": _DEXPILOT_HELPERS,
    },
]
