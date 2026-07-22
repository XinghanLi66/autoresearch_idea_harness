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
    """Starter diagonal term for agent edits."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    return jnp.sum((bt - It_dot) ** 2)


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
    """Starter PSD term for agent edits."""
    del stopgrad_type
    Is = interp.calc_It(s, x0, x1)
    _, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )
    X_su, phi_su = X.apply(
        teacher_params, s, u, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )
    _, phi_ut = X.apply(
        teacher_params,
        u,
        t,
        X_su,
        label,
        train=False,
        rngs=rng,
        return_X_and_phi=True,
    )
    phi_su = jax.lax.stop_gradient(phi_su)
    phi_ut = jax.lax.stop_gradient(phi_ut)
    if psd_type == "uniform":
        teacher = (1 - h) * phi_su + h * phi_ut
    elif psd_type == "midpoint":
        teacher = 0.5 * (phi_su + phi_ut)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")
    return jnp.sum((phi_st - teacher) ** 2)


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
    """Starter LSD term for agent edits."""
    Is = interp.calc_It(s, x0, x1)
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )
    if stopgrad_type == "none":
        b_eval = X.apply(
            params, t, Xst_Is, label, train=False, method="calc_b", rngs=rng
        )
    elif stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                t,
                jax.lax.stop_gradient(Xst_Is),
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    return jnp.sum((b_eval - dt_Xst) ** 2)


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
    """Starter ESD term for agent edits."""
    Is = interp.calc_It(s, x0, x1)
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_s", rngs=rng
    )
    if stopgrad_type == "none":
        b_eval = X.apply(
            params, s, Is, label, train=False, method="calc_b", rngs=rng
        )
    elif stopgrad_type in ("convex", "full"):
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params, s, Is, label, train=False, method="calc_b", rngs=rng
            )
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    _, grad_Xst_b = jax.jvp(
        lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
        primals=(Is,),
        tangents=(b_eval,),
    )
    return jnp.sum((ds_Xst + grad_Xst_b) ** 2)
