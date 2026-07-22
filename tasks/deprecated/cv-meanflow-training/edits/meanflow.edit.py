"""MeanFlow baseline.

MeanFlow trains the network to predict the mean velocity u(x_t, t, t_next)
using the continuous JVP formulation: u = v - (t - t_next) * dv/dt.
No discrete Euler-step approximation is used (alpha=0).

Reference: Li et al., "MeanFlow" (2025)
"""

_FILE = "alphaflow-main/custom_train.py"

_MEANFLOW = '''\
def sample_traj_params(batch_size, cur_step, max_steps, device):
    """MeanFlow: alpha=0, ratio_fm=0.75. Only JVP-based continuous training."""
    ratio_fm = 0.75
    alpha = 0.0

    batch_size_fm = int(batch_size * ratio_fm)
    batch_size_mf = batch_size - batch_size_fm

    t_fm = sample_logit_norm(batch_size_fm, device, loc=-0.4)
    t_next_fm = t_fm.clone()
    dt_fm = torch.zeros_like(t_fm)

    t_1 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
    t_2 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
    t_mf = torch.maximum(t_1, t_2)
    t_next_mf = torch.minimum(t_1, t_2)
    dt_mf = torch.zeros_like(t_mf)  # alpha=0 -> dt=0

    t = torch.cat([t_fm, t_mf]).view(batch_size, 1, 1, 1)
    t_next = torch.cat([t_next_fm, t_next_mf]).view(batch_size, 1, 1, 1)
    dt = torch.cat([dt_fm, dt_mf]).view(batch_size, 1, 1, 1)

    return t, t_next, dt, alpha


def compute_mean_velocity_target(net, x_t, t, t_next, dt, velocity, device):
    """MeanFlow: use JVP for all MeanFlow samples, no discrete path."""
    B = x_t.shape[0]
    t_flat = t.view(B)
    t_next_flat = t_next.view(B)

    mask_fm = torch.isclose(t_flat, t_next_flat)
    mask_c = ~mask_fm

    mean_velocity = velocity.clone()

    if mask_c.any():
        idx = mask_c.nonzero(as_tuple=True)[0]
        x_c = x_t[idx]
        t_c = t_flat[idx]
        t_next_c = t_next_flat[idx]

        def wrap_net(x, t_in):
            t_in_5d = t_in.view(-1, 1, 1, 1)
            t_next_5d = t_next_c.view(-1, 1, 1, 1)
            return net(x, sigma=t_in_5d, sigma_next=t_next_5d)

        _, dudt = jvp(wrap_net, (x_c, t_c), (velocity[idx], torch.ones_like(t_c)))
        u_c = velocity[idx] - (t_c - t_next_c).view(-1, 1, 1, 1) * dudt
        mean_velocity[idx] = u_c

    return mean_velocity
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 141,
        "end_line": 179,
        "content": _MEANFLOW,
    },
]
