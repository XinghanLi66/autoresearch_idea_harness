"""AdaptDiffuser baseline — self-evolving diffusion planner.

Reference: AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners (Liang et al., 2023)
Paper: https://arxiv.org/abs/2302.01877

Key differences from Diffuser:
  - Same architecture (trajectory diffusion + classifier guidance)
  - Adds finetune mode: generate synthetic high-quality trajectories and finetune
  - Three-stage pipeline: train → finetune → inference
  - Inference loads finetuned model instead of original checkpoint

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CleanDiffuser/pipelines/custom_planner.py"

_FINETUNE_MODE = """\
    # ---------------------- Finetune ------------------------
    elif args.mode == "finetune":

        agent.load(save_path + f"diffusion_ckpt_{args.ft_ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ft_ckpt}.pt")

        agent.eval()

        traj_buffer = torch.empty((50000, args.task.horizon, obs_dim + act_dim), device=args.device)
        sample_bs, preserve_bs, ptr = 20000, 1000, 0

        gen_dl = DataLoader(
            dataset, batch_size=sample_bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        for batch in loop_dataloader(gen_dl):

            # generate high-quality synthetic trajectories
            prior = torch.zeros((sample_bs, args.task.horizon, obs_dim + act_dim), device=args.device)
            prior[:, 0, :obs_dim] = batch["obs"]["state"][:, 0].to(args.device)
            traj, log = agent.sample(
                prior, n_samples=sample_bs, sample_steps=args.sampling_steps, solver=args.solver,
                use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature)
            logp = log["log_p"]

            # filter out low-valued trajectories
            selected_traj = traj[logp[:, 0] > args.task.metric_value]
            num_selected = selected_traj.shape[0]
            if ptr + num_selected > 50000:
                num_selected = 50000 - ptr
                selected_traj = selected_traj[:num_selected]
            traj_buffer[ptr:ptr + num_selected] = selected_traj
            ptr += num_selected

            print(f'{num_selected} of 10000 trajs have been selected. Progress: {ptr} / {50000}')
            if ptr == 50000:
                break

        # self-evolving finetuning

        agent.train()
        # Properly update PyTorch optimizer LR (the package's reference also has the
        # `learning_rate` no-op bug — fix it here so finetune actually runs at 1e-5).
        for _g in agent.optimizer.param_groups:
            _g["lr"] = 1e-5

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "gradient_steps": 0}
        while n_gradient_step < 200_000:
            x = traj_buffer[torch.randint(0, 50000, (32,))]
            log["avg_loss_diffusion"] += agent.update(x)['loss']
            if (n_gradient_step + 1) % 1000 == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= 1000
                print(log)
                log = {"avg_loss_diffusion": 0., "gradient_steps": 0}
            if (n_gradient_step + 1) % 5_000 == 0:
                agent.save(save_path + f"finetuned_diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"finetuned_diffusion_ckpt_latest.pt")
            n_gradient_step += 1
"""

_INFERENCE_FINETUNED = """\
        # ============================================================================
        # EDITABLE REGION 5: Inference Setup (lines 186-197)
        # ============================================================================

        # AdaptDiffuser's finetune phase saves at 5k-step intervals up to 200k, NOT
        # at the 1M-step training horizon — so we can't reuse args.ckpt (which the
        # train scripts pass as 1000000 to load the train-phase classifier). Load
        # the latest finetuned diffusion checkpoint by name; classifier still loads
        # via args.ckpt (saved during train phase at 1M).
        agent.load(save_path + "finetuned_diffusion_ckpt_latest.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")

        agent.eval()
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    # 3. Replace inference setup to load finetuned model (lines 128-135)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 128,
        "end_line": 135,
        "content": _INFERENCE_FINETUNED,
    },
    # 2. Replace finetune placeholder (lines 121-123) with actual finetune mode
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 121,
        "end_line": 123,
        "content": _FINETUNE_MODE,
    },
    # 1. Replace config path (line 21) to use adaptdiffuser config
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 21,
        "end_line": 21,
        "content": '@hydra.main(config_path="../configs/adaptdiffuser/mujoco", config_name="mujoco", version_base=None)\n',
    },
]
