"""PureFlow baseline.

Pure flow matching without mean velocity training. Uses only instantaneous
velocity targets v = x_1 - x_0, which is the simplest and most stable
training objective but may require more sampling steps at inference.

This serves as a simple baseline to compare against MeanFlow and AlphaFlow.
"""

_FILE = "alphaflow-main/custom_train.py"

_PUREFLOW = '''\
def sample_traj_params(batch_size, cur_step, max_steps, device):
    """PureFlow: ratio_fm=1.0, alpha=0. Pure flow matching only.

    All samples use t == t_next (instantaneous velocity), no mean velocity.
    This is the simplest training objective.
    """
    ratio_fm = 1.0
    alpha = 0.0

    # All samples are FM samples: t == t_next
    t = sample_logit_norm(batch_size, device, loc=-0.4)
    t_next = t.clone()
    dt = torch.zeros_like(t)

    t = t.view(batch_size, 1, 1, 1)
    t_next = t_next.view(batch_size, 1, 1, 1)
    dt = dt.view(batch_size, 1, 1, 1)

    return t, t_next, dt, alpha


def compute_mean_velocity_target(net, x_t, t, t_next, dt, velocity, device):
    """PureFlow: all samples use instantaneous velocity, no JVP needed."""
    # Since all samples have t == t_next, mean velocity equals instantaneous velocity
    return velocity
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 141,
        "end_line": 179,
        "content": _PUREFLOW,
    },
]
