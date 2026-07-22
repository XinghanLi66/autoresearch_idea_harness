def diagonal_term(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Compute the diagonal (interpolant) term of the loss."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def psd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    u: float,
    h: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    psd_type: str,
    stopgrad_type: str,
) -> float:
    """Compute the PSD (Progressive Self-Distillation) term of the loss."""
    Is = interp.calc_It(s, x0, x1)
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )
    if stopgrad_type == "convex":
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                u,
                Is,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )
        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                u,
                t,
                X_su,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(
            params,
            s,
            u,
            Is,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )
        X_ut, phi_ut = X.apply(
            params,
            u,
            t,
            X_su,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")
    psd_loss = jnp.sum((student - teacher) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * psd_loss + weight_st


def lsd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the LSD term of the loss."""
    Is = interp.calc_It(s, x0, x1)
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )
    if stopgrad_type == "convex":
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                t,
                Xst_Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            t,
            Xst_Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    weight_st = X.apply(params, s, t, method="calc_weight")
    error = b_eval - dt_Xst
    lsd_loss = jnp.sum(error**2)
    return jnp.exp(-weight_st) * lsd_loss + weight_st


def esd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the ESD term of the loss."""
    Is = interp.calc_It(s, x0, x1)
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_s", rngs=rng
    )
    if stopgrad_type == "full":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
        _, grad_Xst_b = jax.lax.stop_gradient(
            jax.jvp(
                lambda x: X.apply(
                    teacher_params, s, t, x, label, train=False, rngs=rng
                ),
                primals=(Is,),
                tangents=(b_eval,),
            )
        )
    elif stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            s,
            Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    esd_loss = jnp.sum((ds_Xst + grad_Xst_b) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * esd_loss + weight_st
