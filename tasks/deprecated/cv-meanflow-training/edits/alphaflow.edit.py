"""AlphaFlow baseline.

AlphaFlow improves MeanFlow by introducing a discrete training phase early in
training. alpha decays from 1.0 to 0.0 via sigmoid schedule, transitioning
from cheap Euler-step approximation to precise JVP-based mean velocity targets.

Reference: Zhang et al., "AlphaFlow: Understanding and Improving MeanFlow Models" (2025)
"""

_FILE = "alphaflow-main/custom_train.py"

_ALPHAFLOW = '''\
def sample_traj_params(batch_size, cur_step, max_steps, device):
    """AlphaFlow: alpha decays sigmoid 1->0, ratio_fm=0.5.

    Early training uses discrete Euler-step approximation (alpha>0),
    later transitions to JVP-based continuous training (alpha->0).
    """
    ratio_fm = 0.5

    # Sigmoid decay: alpha goes from 1.0 to 0.0 over training
    gamma = 25.0
    progress = (cur_step - 0) / max(1, max_steps)
    middle = 0.5
    raw = 1.0 + (0.0 - 1.0) * (1 / (1 + math.exp(-gamma * (progress - middle))))
    clamp = 0.005
    if raw < clamp:
        alpha = 0.0
    elif raw > 1 - clamp:
        alpha = 1.0
    else:
        alpha = raw

    batch_size_fm = int(batch_size * ratio_fm)
    batch_size_mf = batch_size - batch_size_fm

    t_fm = sample_logit_norm(batch_size_fm, device, loc=-0.4)
    t_next_fm = t_fm.clone()
    dt_fm = torch.zeros_like(t_fm)

    t_1 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
    t_2 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
    t_mf = torch.maximum(t_1, t_2)
    t_next_mf = torch.minimum(t_1, t_2)
    # dt = alpha * (t - t_next): when alpha>0, use discrete path
    dt_mf = alpha * (t_mf - t_next_mf)

    t = torch.cat([t_fm, t_mf]).view(batch_size, 1, 1, 1)
    t_next = torch.cat([t_next_fm, t_next_mf]).view(batch_size, 1, 1, 1)
    dt = torch.cat([dt_fm, dt_mf]).view(batch_size, 1, 1, 1)

    return t, t_next, dt, alpha


def compute_mean_velocity_target(net, x_t, t, t_next, dt, velocity, device):
    """AlphaFlow: JVP for continuous samples, Euler step for discrete samples."""
    B = x_t.shape[0]
    t_flat = t.view(B)
    t_next_flat = t_next.view(B)
    dt_flat = dt.view(B)

    mask_fm = torch.isclose(t_flat, t_next_flat)
    mask_d = dt_flat > 0
    mask_c = ~mask_fm & ~mask_d

    mean_velocity = velocity.clone()

    # Continuous path: JVP
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

    # Discrete path: Euler step approximation
    if mask_d.any():
        idx = mask_d.nonzero(as_tuple=True)[0]
        x_d = x_t[idx]
        v_d = velocity[idx]
        t_d = t_flat[idx].view(-1, 1, 1, 1)
        t_next_d = t_next_flat[idx].view(-1, 1, 1, 1)
        dt_d = dt_flat[idx].view(-1, 1, 1, 1)

        with torch.no_grad():
            x_next = x_d - dt_d * v_d
            t_minus_dt = t_d - dt_d
            safe_denom = (t_d - t_next_d).clamp(min=1e-6)
            v_next = net(x_next, sigma=t_minus_dt, sigma_next=t_next_d)
            u_d = (dt_d * v_d + (t_minus_dt - t_next_d) * v_next) / safe_denom
            u_d = u_d.clamp(-4.0, 4.0)
        mean_velocity[idx] = u_d

    return mean_velocity
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 141,
        "end_line": 179,
        "content": _ALPHAFLOW,
    },
]
